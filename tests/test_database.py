"""
Unit and Integration Tests for Mudra Database, Reconciliation, and Paper Portfolio Engines.
Runs exclusively in-memory (sqlite:///:memory:) to prevent polluting production cloud records.
"""
import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.database.models import (
    Base,
    MarketTick,
    ApiTelemetry,
    CouncilAudit,
    Reconciliation,
    PaperPortfolio,
)
from src.database.connection import init_db
from src.engine.reconciliation import reconcile_pending_ticks, get_reconciliation_summary
from src.engine.paper_portfolio import execute_paper_order, get_portfolio_summary, INITIAL_CAPITAL_USD


@pytest.fixture
def in_memory_db():
    """Provides an isolated, in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session, engine
    session.close()


def test_table_creation_and_inspection(in_memory_db):
    """Verifies that all 5 tables are registered and created in the database schema."""
    session, engine = in_memory_db
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "market_ticks" in tables
    assert "api_telemetry" in tables
    assert "council_audits" in tables
    assert "reconciliations" in tables
    assert "paper_portfolio" in tables


def test_fresh_market_tick_creation_and_insertion(in_memory_db):
    """
    Regression test for Issue 1:
    Verifies that fresh MarketTick creation and insertion succeed using only
    supported model fields, and confirms that passing unsupported attributes
    (such as taker_buy_ratio or number_of_trades) raises TypeError.
    """
    session, _ = in_memory_db
    now = datetime.now(timezone.utc)

    # 1. Verify that attempting to pass unsupported fields raises TypeError
    with pytest.raises(TypeError, match="taker_buy_ratio"):
        MarketTick(
            timestamp=now,
            open=83000.0,
            high=83500.0,
            low=82800.0,
            close=83200.0,
            volume=500.0,
            taker_buy_ratio=0.55,
        )

    with pytest.raises(TypeError, match="number_of_trades"):
        MarketTick(
            timestamp=now,
            open=83000.0,
            high=83500.0,
            low=82800.0,
            close=83200.0,
            volume=500.0,
            number_of_trades=1200,
        )

    # 2. Verify that creating and inserting with only supported model fields succeeds
    fresh_tick = MarketTick(
        timestamp=now,
        open=83000.0,
        high=83500.0,
        low=82800.0,
        close=83200.0,
        volume=500.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
        market_regime="BULL_MOMENTUM",
        predicted_return_1h=0.45,
        predicted_price_1h=83574.40,
        model_confidence=0.85,
        features={"taker_buy_ratio": 0.55, "rsi": 62.1},
    )
    session.add(fresh_tick)
    session.commit()

    # 3. Verify retrieved record from database
    persisted = session.get(MarketTick, fresh_tick.id)
    assert persisted is not None
    assert persisted.id == fresh_tick.id
    assert persisted.close == 83200.0
    assert persisted.features["taker_buy_ratio"] == 0.55


def test_models_to_dict_serialization():
    """Verifies to_dict serialization for each ORM model."""
    now = datetime.now(timezone.utc)
    
    tick = MarketTick(
        id=1,
        timestamp=now,
        open=60000.0,
        high=61000.0,
        low=59000.0,
        close=60500.0,
        volume=120.5,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
        market_regime="BULL_MOMENTUM",
        predicted_return_1h=0.75,
        predicted_price_1h=60953.75,
        model_confidence=0.82,
        features={"rsi": 58.2},
    )
    d = tick.to_dict()
    assert d["id"] == 1
    assert d["close"] == 60500.0
    assert d["model_version"] == "xgb_onnx_v1.0.0"
    assert d["features"]["rsi"] == 58.2

    telemetry = ApiTelemetry(
        id=1,
        timestamp=now,
        service_name="binance_vision",
        endpoint="/api/v3/klines",
        latency_ms=142.5,
        status_code=200,
        retry_count=0,
    )
    td = telemetry.to_dict()
    assert td["service_name"] == "binance_vision"
    assert td["latency_ms"] == 142.5


def test_reconciliation_math_and_accurate_tagging(in_memory_db):
    """Tests delayed reconciliation evaluating T-1 against realized T candle."""
    session, _ = in_memory_db
    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    t1 = datetime.now(timezone.utc)

    # Prior tick: price $60,000, model predicted $60,180 (+0.30% UP)
    prior_tick = MarketTick(
        timestamp=t0,
        open=59800.0,
        high=60200.0,
        low=59700.0,
        close=60000.0,
        volume=100.0,
        predicted_return_1h=0.30,
        predicted_price_1h=60180.0,
    )
    session.add(prior_tick)
    session.flush()

    # Current tick closes at $60,150 (+0.25% UP)
    curr_tick = MarketTick(
        timestamp=t1,
        open=60000.0,
        high=60200.0,
        low=59900.0,
        close=60150.0,
        volume=110.0,
    )
    session.add(curr_tick)
    session.flush()

    reconciled = reconcile_pending_ticks(session, curr_tick)
    session.commit()

    assert len(reconciled) == 1
    rec = reconciled[0]
    assert rec.tick_id == prior_tick.id
    assert rec.direction_correct is True
    assert rec.absolute_error_usd == pytest.approx(30.0, abs=1e-2)  # |60150 - 60180| = 30
    assert rec.percentage_error < 0.10
    assert rec.is_black_swan is False
    assert rec.error_attribution_tag == "ACCURATE"


def test_reconciliation_micro_move_and_sideways_drift(in_memory_db):
    """Tests that micro-moves are binary UP/DOWN (no FLAT) and mismatches in chop tag SIDEWAYS_DRIFT."""
    session, _ = in_memory_db
    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    t1 = datetime.now(timezone.utc)

    # Prior tick: $84,000, model micro-move +0.01% -> $84,008.40
    prior_tick = MarketTick(
        timestamp=t0,
        open=83900.0,
        high=84100.0,
        low=83800.0,
        close=84000.0,
        volume=100.0,
        predicted_return_1h=0.01,
        predicted_price_1h=84008.40,
    )
    session.add(prior_tick)
    session.flush()

    # Realized tick closes at $83,960.00 (-0.048% DOWN, $48.40 diff)
    curr_tick = MarketTick(
        timestamp=t1,
        open=84000.0,
        high=84050.0,
        low=83920.0,
        close=83960.0,
        volume=90.0,
    )
    session.add(curr_tick)
    session.flush()

    reconciled = reconcile_pending_ticks(session, curr_tick)
    session.commit()

    assert len(reconciled) == 1
    rec = reconciled[0]
    assert rec.direction_predicted == "UP"
    assert rec.direction_realized == "DOWN"
    assert rec.direction_correct is False
    assert rec.absolute_error_usd == pytest.approx(48.40, abs=1e-2)
    assert rec.error_attribution_tag == "SIDEWAYS_DRIFT"


def test_reconciliation_black_swan_outlier_isolation(in_memory_db):
    """Tests that extreme flash crashes (>3.8% return) are flagged as black swans."""
    session, _ = in_memory_db
    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    t1 = datetime.now(timezone.utc)

    prior = MarketTick(
        timestamp=t0,
        open=60000.0,
        high=60100.0,
        low=59900.0,
        close=60000.0,
        volume=100.0,
        predicted_return_1h=0.50,
        predicted_price_1h=60300.0,
    )
    session.add(prior)
    session.flush()

    # Flash crash: Drops 5.0% to $57,000
    crash_tick = MarketTick(
        timestamp=t1,
        open=60000.0,
        high=60000.0,
        low=56800.0,
        close=57000.0,
        volume=800.0,
    )
    session.add(crash_tick)
    session.flush()

    reconciled = reconcile_pending_ticks(session, crash_tick)
    session.commit()

    rec = reconciled[0]
    assert rec.direction_correct is False
    assert rec.is_black_swan is True
    assert rec.error_attribution_tag == "BLACK_SWAN_OUTLIER"


def test_veto_attribution_saved_capital_vs_opportunity_cost(in_memory_db):
    """Tests that Risk Critic vetoes are correctly attributed at T+1."""
    session, _ = in_memory_db
    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    t1 = datetime.now(timezone.utc)

    # 1. Critic Vetoed BUY, and Market Fell (SAVED_CAPITAL)
    tick1 = MarketTick(
        timestamp=t0,
        open=60000.0,
        high=60200.0,
        low=59800.0,
        close=60000.0,
        volume=100.0,
        predicted_price_1h=60600.0,
    )
    session.add(tick1)
    session.flush()

    audit1 = CouncilAudit(
        tick_id=tick1.id,
        strategist_action="BUY",
        strategist_conviction=7,
        risk_verdict="VETOED",
        kelly_fraction=0.0,
        final_decision="HOLD",
    )
    session.add(audit1)
    session.flush()

    # Market drops to $59,200
    down_tick = MarketTick(timestamp=t1, open=60000.0, high=60000.0, low=59100.0, close=59200.0, volume=100.0)
    session.add(down_tick)
    session.flush()

    reconcile_pending_ticks(session, down_tick)
    session.commit()
    assert audit1.veto_attribution == "SAVED_CAPITAL"


def test_paper_portfolio_full_lifecycle(in_memory_db):
    """
    Tests complete $10,000 forward paper portfolio lifecycle:
    1. Initial deposit
    2. BUY order execution with fee & slippage deduction
    3. Mark-to-market rally with Alpha over Buy & Hold benchmark
    4. SELL liquidation with realized PnL
    """
    session, _ = in_memory_db

    # Step 1: Initial Deposit & HOLD
    r1 = execute_paper_order(session, "HOLD", 60000.0)
    assert r1.cash_balance == INITIAL_CAPITAL_USD
    assert r1.btc_balance == 0.0
    assert r1.total_equity_usd == INITIAL_CAPITAL_USD
    assert r1.benchmark_btc_equity == INITIAL_CAPITAL_USD
    assert r1.alpha_usd == 0.0

    # Step 2: BUY with 10% Kelly fraction ($1,000 deployed)
    r2 = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10)
    assert r2.cash_balance == pytest.approx(9000.0, abs=1e-2)
    assert r2.btc_balance > 0.016
    assert r2.fee_deducted_usd == pytest.approx(0.50, abs=1e-2)      # 0.05% of $1,000 = $0.50
    assert r2.slippage_usd == pytest.approx(0.20, abs=1e-2)          # 0.02% of $1,000 = $0.20

    # Step 3: Market Rallies +5% to $63,000 (HOLD to mark to market)
    r3 = execute_paper_order(session, "HOLD", 63000.0)
    # Benchmark rose to $10,500.00 (since 100% held in BTC would be $10,500)
    assert r3.benchmark_btc_equity == pytest.approx(10500.0, abs=1.0)
    # Portfolio equity gained on the 10% BTC position (~$10,049)
    assert r3.total_equity_usd > 10040.0
    # Because only 10% was deployed while BTC surged 5%, active alpha vs 100% BTC benchmark is negative
    assert r3.alpha_usd < 0.0

    # Step 4: SELL all BTC at $63,000
    r4 = execute_paper_order(session, "SELL", 63000.0)
    assert r4.btc_balance == 0.0
    assert r4.cash_balance > 10045.0
    assert r4.realized_pnl_usd > 40.0

    summary = get_portfolio_summary(session, 63000.0)
    assert summary["total_trades"] == 2
    assert summary["cash_balance"] > 10045.0
