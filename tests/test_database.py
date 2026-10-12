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
from src.engine.paper_portfolio import (
    execute_paper_order,
    get_portfolio_summary,
    get_average_entry_price,
    INITIAL_CAPITAL_USD,
)
from src.api.schemas import ReconciliationSummaryResponse, PortfolioSummaryResponse


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
    assert summary["current_btc_price"] == 63000.0
    assert summary["btc_position_value_usd"] == 0.0
    assert summary["total_equity_usd"] == round(summary["cash_balance"] + summary["btc_position_value_usd"], 2)
    assert summary["realized_pnl_usd"] > 40.0
    assert summary["total_fees_usd"] > 0.80  # Both buy and sell fees

    # Step 5: Subsequent HOLD must NOT erase cumulative realized PnL or total fees
    execute_paper_order(session, "HOLD", 64000.0)
    summary_hold = get_portfolio_summary(session, 64000.0)
    assert summary_hold["realized_pnl_usd"] == summary["realized_pnl_usd"]
    assert summary_hold["total_fees_usd"] == summary["total_fees_usd"]
    assert summary_hold["total_trades"] == 2

    # Step 6: Subsequent BUY increases total fees and trades, preserves earlier realized PnL
    execute_paper_order(session, "BUY", 64000.0, kelly_fraction=0.10)
    summary_buy = get_portfolio_summary(session, 64000.0)
    assert summary_buy["realized_pnl_usd"] == summary["realized_pnl_usd"]
    assert summary_buy["total_fees_usd"] > summary["total_fees_usd"]
    assert summary_buy["total_trades"] == 3
    assert summary_buy["btc_position_value_usd"] > 0.0
    assert summary_buy["total_equity_usd"] == round(summary_buy["cash_balance"] + summary_buy["btc_position_value_usd"], 2)

    # Validate against PortfolioSummaryResponse schema
    schema_model = PortfolioSummaryResponse(**summary_buy)
    assert schema_model.realized_pnl_usd == summary_buy["realized_pnl_usd"]
    assert schema_model.total_fees_usd == summary_buy["total_fees_usd"]
    assert schema_model.cumulative_return_pct == summary_buy["cumulative_return_pct"]


def test_empty_portfolio_summary(in_memory_db):
    """Verifies that an uninitialized / empty portfolio ledger returns valid baseline schema."""
    session, _ = in_memory_db
    summary = get_portfolio_summary(session, current_price=65000.0)

    assert summary["cash_balance"] == INITIAL_CAPITAL_USD
    assert summary["btc_balance"] == 0.0
    assert summary["current_btc_price"] == 65000.0
    assert summary["btc_position_value_usd"] == 0.0
    assert summary["total_equity_usd"] == INITIAL_CAPITAL_USD
    assert summary["cumulative_return_pct"] == 0.0
    assert summary["benchmark_btc_equity"] == INITIAL_CAPITAL_USD
    assert summary["alpha_usd"] == 0.0
    assert summary["alpha_pct"] == 0.0
    assert summary["realized_pnl_usd"] == 0.0
    assert summary["total_fees_usd"] == 0.0
    assert summary["drawdown_pct"] == 0.0
    assert summary["total_trades"] == 0

    schema_model = PortfolioSummaryResponse(**summary)
    assert schema_model.total_equity_usd == INITIAL_CAPITAL_USD
    assert schema_model.btc_position_value_usd == 0.0


def test_reconciliation_summary_metrics_and_tag_breakdown(in_memory_db):
    """
    Verifies that get_reconciliation_summary produces accurate aggregation,
    including IC calculation, black swan counts, and complete tag_breakdown frequency mapping.
    """
    session, _ = in_memory_db

    # 1. Empty case
    empty_summary = get_reconciliation_summary(session, limit=24)
    assert empty_summary["total_reconciled"] == 0
    assert empty_summary["directional_accuracy_pct"] == 0.0
    assert empty_summary["information_coefficient_ic"] == 0.0
    assert empty_summary["black_swan_count"] == 0
    assert empty_summary["tag_breakdown"] == {}

    # 2. Populated records
    now = datetime.now(timezone.utc)
    t1 = MarketTick(timestamp=now - timedelta(hours=4), open=60000.0, high=60200.0, low=59900.0, close=60100.0, volume=100.0)
    t2 = MarketTick(timestamp=now - timedelta(hours=3), open=60100.0, high=60300.0, low=60000.0, close=60200.0, volume=100.0)
    t3 = MarketTick(timestamp=now - timedelta(hours=2), open=60200.0, high=60400.0, low=60100.0, close=60300.0, volume=100.0)
    t4 = MarketTick(timestamp=now - timedelta(hours=1), open=60300.0, high=60500.0, low=57000.0, close=57200.0, volume=100.0)
    session.add_all([t1, t2, t3, t4])
    session.flush()

    r1 = Reconciliation(
        timestamp=now - timedelta(hours=3),
        tick_id=t1.id,
        predicted_price=60200.0,
        actual_price_at_t_plus_1=60200.0,
        predicted_return_pct=0.0017,
        actual_return_pct=0.0017,
        residual_error=0.0,
        absolute_error_usd=0.0,
        percentage_error=0.0,
        direction_predicted="UP",
        direction_realized="UP",
        direction_correct=True,
        is_black_swan=False,
        error_attribution_tag="ACCURATE",
    )
    r2 = Reconciliation(
        timestamp=now - timedelta(hours=2),
        tick_id=t2.id,
        predicted_price=60100.0,
        actual_price_at_t_plus_1=60150.0,
        predicted_return_pct=-0.0010,
        actual_return_pct=-0.0008,
        residual_error=50.0,
        absolute_error_usd=50.0,
        percentage_error=0.08,
        direction_predicted="DOWN",
        direction_realized="DOWN",
        direction_correct=True,
        is_black_swan=False,
        error_attribution_tag="ACCURATE",
    )
    r3 = Reconciliation(
        timestamp=now - timedelta(hours=1),
        tick_id=t3.id,
        predicted_price=60500.0,
        actual_price_at_t_plus_1=60250.0,
        predicted_return_pct=0.005,
        actual_return_pct=-0.0005,
        residual_error=-250.0,
        absolute_error_usd=250.0,
        percentage_error=0.41,
        direction_predicted="UP",
        direction_realized="DOWN",
        direction_correct=False,
        is_black_swan=False,
        error_attribution_tag="SIDEWAYS_DRIFT",
    )
    r4 = Reconciliation(
        timestamp=now,
        tick_id=t4.id,
        predicted_price=60400.0,
        actual_price_at_t_plus_1=57200.0,
        predicted_return_pct=0.002,
        actual_return_pct=-0.051,
        residual_error=-3200.0,
        absolute_error_usd=3200.0,
        percentage_error=5.3,
        direction_predicted="UP",
        direction_realized="DOWN",
        direction_correct=False,
        is_black_swan=True,
        error_attribution_tag="BLACK_SWAN_OUTLIER",
    )
    session.add_all([r1, r2, r3, r4])
    session.commit()

    summary = get_reconciliation_summary(session, limit=24)
    assert summary["total_reconciled"] == 4
    assert summary["directional_accuracy_pct"] == 50.0
    assert summary["black_swan_count"] == 1
    assert summary["black_swans_filtered"] == 1
    assert summary["tag_breakdown"] == {
        "ACCURATE": 2,
        "SIDEWAYS_DRIFT": 1,
        "BLACK_SWAN_OUTLIER": 1,
    }
    assert summary["information_coefficient_ic"] == summary["information_coefficient"]
    assert isinstance(summary["information_coefficient_ic"], float)
    assert -1.0 <= summary["information_coefficient_ic"] <= 1.0
    assert summary["information_coefficient_ic"] != 0.0

    # Ensure schema validation preserves all fields
    schema_model = ReconciliationSummaryResponse(**summary)
    assert schema_model.total_reconciled == 4
    assert schema_model.directional_accuracy_pct == 50.0
    assert schema_model.black_swan_count == 1
    assert schema_model.information_coefficient_ic == summary["information_coefficient_ic"]
    assert schema_model.tag_breakdown["ACCURATE"] == 2


def test_net_realized_pnl_accounting_two_completed_trades(in_memory_db):
    """
    Deterministic regression test verifying net realized P&L accounting:
    1. Entry fee is incorporated into cost basis so that net realized PnL deducts
       both entry and exit transaction fees (and slippage) without double-counting.
    2. Net realized PnL strictly equals cash delta for closed positions.
    3. Subsequent HOLD preserves cumulative realized PnL and total fees.
    4. Second BUY and SELL trade properly scopes cost basis to the new trade and
       accumulates realized PnL and fees.
    5. Total equity strictly matches cash + marked-to-market BTC position value.
    """
    session, _ = in_memory_db

    # --- TRADE 1 ---
    # Step 1: BUY at $60,000 ($1,000 deployed from $10,000 initial bankroll)
    r_buy1 = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10)
    assert r_buy1.cash_balance == pytest.approx(9000.0, abs=1e-2)
    assert r_buy1.fee_deducted_usd == pytest.approx(0.50, abs=1e-2)   # 0.05% of $1,000
    assert r_buy1.slippage_usd == pytest.approx(0.20, abs=1e-2)       # 0.02% of $1,000
    assert r_buy1.realized_pnl_usd == 0.0                             # Zero realized on buy
    btc_amount_1 = r_buy1.btc_amount

    # Step 2: SELL at $63,000 (position closed)
    r_sell1 = execute_paper_order(session, "SELL", 63000.0)
    assert r_sell1.btc_balance == 0.0
    # Expected net realized PnL:
    # Gross proceeds: btc_amount_1 * 62987.40 = $1,049.055
    # Sell fee: 1049.055 * 0.05% = $0.525
    # Net proceeds: $1,048.531
    # Cost basis (including entry fee): $1,000.000
    # Net realized PnL: $1,048.531 - $1,000.000 = $48.531
    expected_pnl_1 = 48.53
    assert r_sell1.realized_pnl_usd == pytest.approx(expected_pnl_1, abs=1e-2)
    # Crucial identity: cash delta from initial capital must match realized PnL exactly
    assert r_sell1.cash_balance - INITIAL_CAPITAL_USD == pytest.approx(r_sell1.realized_pnl_usd, abs=1e-2)

    summary1 = get_portfolio_summary(session, 63000.0)
    assert summary1["realized_pnl_usd"] == pytest.approx(expected_pnl_1, abs=1e-2)
    assert summary1["total_fees_usd"] == pytest.approx(1.02, abs=1e-2)  # Buy fee (0.50) + Sell fee (0.52)
    assert summary1["total_equity_usd"] == round(summary1["cash_balance"] + summary1["btc_position_value_usd"], 2)

    # --- INTERMEDIATE HOLD ---
    # Step 3: HOLD at $64,000
    r_hold = execute_paper_order(session, "HOLD", 64000.0)
    assert r_hold.realized_pnl_usd == 0.0
    summary_hold = get_portfolio_summary(session, 64000.0)
    assert summary_hold["realized_pnl_usd"] == summary1["realized_pnl_usd"]
    assert summary_hold["total_fees_usd"] == summary1["total_fees_usd"]
    assert summary_hold["total_equity_usd"] == summary1["total_equity_usd"]

    # --- TRADE 2 ---
    # Step 4: BUY at $64,000 with 10% Kelly (deploy 10% of ~$10,048.53 = ~$1,004.85)
    r_buy2 = execute_paper_order(session, "BUY", 64000.0, kelly_fraction=0.10)
    assert r_buy2.fee_deducted_usd == pytest.approx(0.50, abs=1e-2)
    assert r_buy2.realized_pnl_usd == 0.0
    capital_spent_2 = 10048.53 - r_buy2.cash_balance

    # Step 5: SELL at $65,000
    r_sell2 = execute_paper_order(session, "SELL", 65000.0)
    assert r_sell2.btc_balance == 0.0
    # Expected Trade 2 net realized PnL: ~$14.27
    expected_pnl_2 = 14.27
    assert r_sell2.realized_pnl_usd == pytest.approx(expected_pnl_2, abs=1e-2)

    # Cumulative summary across all trades
    summary2 = get_portfolio_summary(session, 65000.0)
    expected_cum_pnl = expected_pnl_1 + expected_pnl_2  # 48.53 + 14.27 = 62.80
    assert summary2["realized_pnl_usd"] == pytest.approx(expected_cum_pnl, abs=1e-2)
    assert summary2["total_fees_usd"] == pytest.approx(2.04, abs=1e-2)  # 4 trade legs ~ $2.04
    assert summary2["total_trades"] == 4
    assert summary2["total_equity_usd"] == round(summary2["cash_balance"] + summary2["btc_position_value_usd"], 2)
    # Total portfolio profit equals cumulative realized PnL when inventory is zero
    assert summary2["total_equity_usd"] - INITIAL_CAPITAL_USD == pytest.approx(summary2["realized_pnl_usd"], abs=1e-2)


def test_get_average_entry_price_missing_eligible_buy_isolation(in_memory_db):
    """
    Verifies that get_average_entry_price strictly isolates cost basis between trades:
    1. Returns 0.0 on an uninitialized / empty ledger.
    2. After a completed trade (BUY then SELL), querying cost basis for unacquired inventory
       returns 0.0 instead of falling back to the previous liquidated trade's BUY records.
    3. An orphaned open position with no eligible BUY records safely evaluates to 0.0 cost basis
       without crashing or borrowing earlier trade data.
    """
    session, _ = in_memory_db

    # 1. Empty ledger returns 0.0
    assert get_average_entry_price(session, target_btc=0.016) == 0.0

    # 2. Complete Trade 1: BUY and SELL
    execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10)
    execute_paper_order(session, "SELL", 63000.0)

    # All BTC is now liquidated; last_sell_id exists.
    # Querying open inventory now MUST NOT fall back to Trade 1's BUY row.
    assert get_average_entry_price(session, target_btc=0.016) == 0.0

    # 3. Orphaned open inventory without a BUY record evaluates safely to 0.0 cost basis
    # Simulate an orphaned snapshot row with positive BTC balance but no BUY record
    orphaned_snapshot = PaperPortfolio(
        timestamp=datetime.now(timezone.utc),
        action="HOLD",
        btc_amount=0.0,
        execution_price=64000.0,
        fee_deducted_usd=0.0,
        slippage_usd=0.0,
        cash_balance=10048.53,
        btc_balance=0.01,
        total_equity_usd=10048.53 + (0.01 * 64000.0),
        realized_pnl_usd=0.0,
        unrealized_pnl_usd=0.0,
        drawdown_pct=0.0,
        cumulative_return_pct=0.0,
        benchmark_btc_equity=10000.0,
        alpha_usd=0.0,
        alpha_pct=0.0,
        notes="Orphaned test record without corresponding BUY row",
    )
    session.add(orphaned_snapshot)
    session.commit()

    # Must still return 0.0 cost basis and not borrow Trade 1's records
    assert get_average_entry_price(session, target_btc=0.01) == 0.0


