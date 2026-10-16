# tests/test_hourly_worker.py
"""
Deterministic Unit & Regression Tests for Mudra Hourly Worker Lifecycle,
Exit Code Contracts, and Durable Outbox Processing.
TASK-P1-05.3: Verify External Scheduler Integration, Then Fix Worker Reliability.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

import scripts.run_hourly_worker as worker
import src.engine.sync as sync
from src.database.models import Base, MarketTick, CouncilAudit, PaperPortfolio
from src.database.connection import init_db as real_init_db
from src.engine.paper_portfolio import execute_paper_order


@pytest.fixture
def isolated_db(monkeypatch):
    """Provides an isolated in-memory SQLite database session for worker testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    real_init_db(engine)
    Session = sessionmaker(bind=engine)
    
    # Patch connection engine and SessionLocal in database.connection and worker
    monkeypatch.setattr("src.database.connection.engine", engine)
    monkeypatch.setattr("src.database.connection.SessionLocal", Session)
    monkeypatch.setattr("src.database.connection.test_connection", lambda: (True, "SQLite 3.x", 1.0))
    monkeypatch.setattr("src.database.connection.init_db", lambda target_engine=None: real_init_db(engine))
    
    session = Session()
    yield session, engine
    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def temp_outbox(tmp_path, monkeypatch):
    """Provides an isolated temporary outbox file path for testing."""
    outbox_path = tmp_path / "sync_outbox.json"
    monkeypatch.setattr(worker, "OUTBOX_FILE", outbox_path)
    return outbox_path


# -----------------------------------------------------------------------------
# 1. Exit Code & Success/Failure Propagation Contracts
# -----------------------------------------------------------------------------

def test_worker_success_on_new_sync(isolated_db, temp_outbox, monkeypatch):
    """Verifies that a successful quant cycle returns True (exit 0)."""
    monkeypatch.setattr(
        worker,
        "sync_hourly_quant_cycle",
        lambda force=False: (True, "Recorded tick #1 at $60,000.00", {"total_equity_usd": 10000.0})
    )
    assert worker.run_worker(force=False) is True


def test_worker_success_on_already_recorded_candle(isolated_db, temp_outbox, monkeypatch):
    """Verifies that an already-recorded candle is treated as an idempotent success (exit 0)."""
    monkeypatch.setattr(
        worker,
        "sync_hourly_quant_cycle",
        lambda force=False: (False, "Candle for 14:00 UTC already recorded in Neon.", {"tick_id": 42})
    )
    assert worker.run_worker(force=False) is True


def test_worker_failure_on_feed_error(isolated_db, temp_outbox, monkeypatch):
    """Verifies that a genuine operational failure (market feed error) returns False (exit 1)."""
    monkeypatch.setattr(
        worker,
        "sync_hourly_quant_cycle",
        lambda force=False: (False, "Market feed error: Failed to fetch confirmed 1-hour candle from all market feeds.", {})
    )
    assert worker.run_worker(force=False) is False


def test_worker_failure_on_database_unreachable(isolated_db, temp_outbox, monkeypatch):
    """Verifies that database connection test failure immediately returns False (exit 1)."""
    monkeypatch.setattr(
        "src.database.connection.test_connection",
        lambda: (False, "Connection timeout to Neon serverless", 10000.0)
    )
    assert worker.run_worker(force=False) is False


def test_worker_failure_on_unhandled_exception(isolated_db, temp_outbox, monkeypatch):
    """Verifies that an unexpected exception during quant cycle returns False instead of concealing failure."""
    def _crashing_sync(force=False):
        raise RuntimeError("Unexpected ONNX runtime failure")

    monkeypatch.setattr(worker, "sync_hourly_quant_cycle", _crashing_sync)
    assert worker.run_worker(force=False) is False


def test_worker_failure_on_init_db_error(isolated_db, temp_outbox, monkeypatch):
    """Verifies that database schema verification/initialization failure at startup returns False (exit 1)."""
    def _crashing_init(target_engine=None, *args, **kwargs):
        raise RuntimeError("PostgreSQL relation creation failure")

    monkeypatch.setattr("src.database.connection.verify_db_schema", _crashing_init)
    monkeypatch.setattr("src.database.connection.init_db", _crashing_init)
    assert worker.run_worker(force=False) is False


# -----------------------------------------------------------------------------
# 2. Durable Outbox Processing & Crash Resilience
# -----------------------------------------------------------------------------

def test_missing_outbox_is_safe_noop(isolated_db, temp_outbox):
    """Verifies that a missing outbox file is treated as a safe no-op returning 0."""
    session, _ = isolated_db
    assert not temp_outbox.exists()
    assert worker.flush_outbox(session) == 0


def test_empty_outbox_file_is_unlinked(isolated_db, temp_outbox):
    """Verifies that an empty or whitespace-only outbox file is safely removed."""
    session, _ = isolated_db
    temp_outbox.write_text("   \n  ", encoding="utf-8")
    assert worker.flush_outbox(session) == 0
    assert not temp_outbox.exists()


def test_malformed_json_outbox_is_preserved(isolated_db, temp_outbox):
    """Verifies that corrupted JSON is preserved on disk rather than silently discarded."""
    session, _ = isolated_db
    corrupted_data = '{"type": "tick", "payload": {unfinished_json...'
    temp_outbox.write_text(corrupted_data, encoding="utf-8")

    assert worker.flush_outbox(session) == 0
    assert temp_outbox.exists()
    assert temp_outbox.read_text(encoding="utf-8") == corrupted_data


def test_non_list_outbox_is_preserved(isolated_db, temp_outbox):
    """Verifies that non-list JSON structures are preserved on disk."""
    session, _ = isolated_db
    temp_outbox.write_text('{"error": "bad structure"}', encoding="utf-8")

    assert worker.flush_outbox(session) == 0
    assert temp_outbox.exists()


def test_successful_outbox_flush_persists_ticks_and_unlinks(isolated_db, temp_outbox):
    """Verifies that valid cached ticks are inserted and the file is deleted upon commit."""
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    outbox_payload = [
        {
            "type": "tick",
            "payload": {
                "timestamp": t0.isoformat(),
                "open": 60000.0,
                "high": 60500.0,
                "low": 59500.0,
                "close": 60200.0,
                "volume": 120.0,
                "model_version": "xgb_onnx_v1.0.0",
                "feature_schema_version": "v1_39features",
                "market_regime": "BULL_MOMENTUM",
                "predicted_return_1h": 0.35,
                "predicted_price_1h": 60410.70,
                "model_confidence": 0.85,
                "features": {"rsi": 55.4},
            }
        }
    ]
    temp_outbox.write_text(json.dumps(outbox_payload), encoding="utf-8")

    flushed = worker.flush_outbox(session)
    assert flushed == 1
    assert not temp_outbox.exists()

    # Verify tick in database
    persisted = session.execute(
        select(MarketTick).where(MarketTick.timestamp == t0)
    ).scalar_one_or_none()
    assert persisted is not None
    assert persisted.close == 60200.0
    assert persisted.features["rsi"] == 55.4


def test_outbox_preserved_when_commit_fails(isolated_db, temp_outbox, monkeypatch):
    """
    Critical durability test:
    If a database error or rollback occurs during session commit, the outbox file
    MUST NOT be deleted from disk.
    """
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    outbox_payload = [
        {
            "type": "tick",
            "payload": {
                "timestamp": t0.isoformat(),
                "open": 61000.0,
                "high": 61500.0,
                "low": 60800.0,
                "close": 61200.0,
                "volume": 85.0,
            }
        }
    ]
    temp_outbox.write_text(json.dumps(outbox_payload), encoding="utf-8")

    # Force commit failure on the session
    def _failing_commit():
        raise RuntimeError("Database disk full or constraint violation")

    monkeypatch.setattr(session, "commit", _failing_commit)

    flushed = worker.flush_outbox(session)
    assert flushed == 0
    # File MUST remain intact
    assert temp_outbox.exists()


def test_outbox_replay_idempotency_prevents_duplicates(isolated_db, temp_outbox):
    """
    Crash recovery test:
    Simulates a crash after database commit but before file deletion.
    On the subsequent replay, the existing ticks must be skipped idempotently,
    and the file must be removed without duplicating rows in market_ticks.
    """
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)

    # 1. Manually insert the tick into the database (as if earlier run committed it)
    existing_tick = MarketTick(
        timestamp=t0,
        open=62000.0,
        high=62500.0,
        low=61800.0,
        close=62200.0,
        volume=90.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(existing_tick)
    session.commit()

    # 2. Outbox file is still present on disk (due to prior crash before unlink)
    outbox_payload = [
        {
            "type": "tick",
            "payload": {
                "timestamp": t0.isoformat(),
                "open": 62000.0,
                "high": 62500.0,
                "low": 61800.0,
                "close": 62200.0,
                "volume": 90.0,
            }
        }
    ]
    temp_outbox.write_text(json.dumps(outbox_payload), encoding="utf-8")

    # 3. Flashing outbox on restart
    flushed = worker.flush_outbox(session)
    # 0 new rows added because it was already present
    assert flushed == 0
    # File is now cleanly unlinked
    assert not temp_outbox.exists()

    # Verify strictly 1 row in market_ticks
    all_ticks = session.execute(
        select(MarketTick).where(MarketTick.timestamp == t0)
    ).scalars().all()
    assert len(all_ticks) == 1


def test_successful_outbox_flush_default_session_and_timezone_normalization(isolated_db, temp_outbox):
    """
    Verifies that:
    1. Calling flush_outbox(session=None) manages its own transaction via get_db(),
       persists cached ticks, and unlinks the outbox file upon successful commit.
    2. Aware timestamps with non-UTC timezone offsets (e.g., +05:30) are correctly
       normalized to UTC via astimezone(timezone.utc) prior to database insertion.
    3. Replaying the same candle with an aware offset idempotently detects the
       existing record and avoids duplicate row insertion.
    """
    session, _ = isolated_db

    # 17:30:00+05:30 is equivalent to 12:00:00 UTC
    offset_ts_str = "2026-01-01T17:30:00+05:30"
    expected_utc = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    outbox_payload = [
        {
            "type": "tick",
            "payload": {
                "timestamp": offset_ts_str,
                "open": 63000.0,
                "high": 63500.0,
                "low": 62800.0,
                "close": 63200.0,
                "volume": 110.0,
                "model_version": "xgb_onnx_v1.0.0",
                "feature_schema_version": "v1_39features",
                "features": {"rsi": 48.2},
            }
        }
    ]
    temp_outbox.write_text(json.dumps(outbox_payload), encoding="utf-8")

    # Call flush_outbox with default session=None (exercises get_db() context manager)
    flushed = worker.flush_outbox(session=None)
    assert flushed == 1
    assert not temp_outbox.exists()

    # Verify tick was inserted with normalized UTC timestamp
    persisted = session.execute(
        select(MarketTick).where(MarketTick.timestamp == expected_utc)
    ).scalar_one_or_none()
    assert persisted is not None
    assert persisted.close == 63200.0
    assert persisted.features["rsi"] == 48.2
    # In SQLite DateTime strips tzinfo, but numerical values correspond to 12:00:00 UTC
    assert persisted.timestamp.hour == 12
    assert persisted.timestamp.minute == 0

    # Test replay idempotency with session=None
    temp_outbox.write_text(json.dumps(outbox_payload), encoding="utf-8")
    replay_flushed = worker.flush_outbox()  # default parameter is session=None
    assert replay_flushed == 0
    assert not temp_outbox.exists()

    all_ticks = session.execute(
        select(MarketTick).where(MarketTick.timestamp == expected_utc)
    ).scalars().all()
    assert len(all_ticks) == 1


# -----------------------------------------------------------------------------
# 3. Paper-Trade Deduplication & --force Idempotency Contracts
# -----------------------------------------------------------------------------

def test_force_sync_deduplicates_paper_trade_and_fees(isolated_db, monkeypatch):
    """
    Verifies that running with force=True on an existing candle:
    1. Recomputes and updates the permitted forecast fields on MarketTick.
    2. Records an auditable trail of forced reanalysis in CouncilAudit.
    3. Strictly deduplicates paper trade execution: creates ZERO new PaperPortfolio rows.
    4. Leaves cash balance, BTC balance, and total fees unchanged (zero duplicate friction).
    """
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    candle_data = {
        "timestamp": t0,
        "open": 60000.0,
        "high": 60200.0,
        "low": 59800.0,
        "close": 60100.0,
        "volume": 75.0,
    }

    # Setup mocks for quant cycle
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: candle_data)
    mock_signals = MagicMock()
    mock_signals.invoke.return_value = {"volatility_range_pct": 1.2, "rsi": 55.0}
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)

    mock_ml = MagicMock()
    mock_ml.invoke.return_value = {
        "expected_return_pct": 0.45,
        "predicted_target_price": 60370.0,
        "confidence": 0.80,
        "clears_hurdle": True,
    }
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)

    monkeypatch.setattr(
        sync,
        "run_trading_graph",
        lambda **k: {
            "final_decision": "BUY",
            "risk_verdict": "APPROVED",
            "strategist_action": "BUY",
            "strategist_conviction": 8,
            "kelly_fraction": 0.10,
            "thesis": "Initial bullish momentum",
        }
    )

    # 1. First Run (force=False): fresh candle execution
    ok1, msg1, summ1 = sync.sync_hourly_quant_cycle(force=False)
    assert ok1 is True
    assert "Recorded tick #1" in msg1

    tick1 = session.execute(select(MarketTick).where(MarketTick.timestamp == t0)).scalar_one()
    assert tick1.predicted_return_1h == 0.45

    # Exactly 1 paper trade executed
    buy_orders_1 = session.execute(
        select(PaperPortfolio).where(PaperPortfolio.action == "BUY")
    ).scalars().all()
    assert len(buy_orders_1) == 1
    assert buy_orders_1[0].tick_id == tick1.id

    cash_1 = summ1["cash_balance"]
    btc_1 = summ1["btc_balance"]
    fees_1 = summ1["total_fees_usd"]
    assert cash_1 < 10000.0
    assert btc_1 > 0.0
    assert fees_1 > 0.0

    # 2. Second Run with force=True and updated forecast values
    mock_ml.invoke.return_value = {
        "expected_return_pct": 0.95,
        "predicted_target_price": 60670.0,
        "confidence": 0.92,
        "clears_hurdle": True,
    }
    monkeypatch.setattr(
        sync,
        "run_trading_graph",
        lambda **k: {
            "final_decision": "BUY",
            "risk_verdict": "APPROVED",
            "strategist_action": "BUY",
            "strategist_conviction": 9,
            "kelly_fraction": 0.10,
            "thesis": "Reanalyzed higher conviction breakout",
        }
    )

    ok2, msg2, summ2 = sync.sync_hourly_quant_cycle(force=True)
    assert ok2 is True
    assert "Trade deduplicated | No fees" in msg2

    # Verification 1: Forecast fields updated on existing tick
    session.expire_all()
    tick1_updated = session.execute(select(MarketTick).where(MarketTick.timestamp == t0)).scalar_one()
    assert tick1_updated.id == tick1.id
    assert tick1_updated.predicted_return_1h == 0.95

    # Verification 2: Auditability preserved (2 audits: initial and reanalysis)
    audits = session.execute(
        select(CouncilAudit).where(CouncilAudit.tick_id == tick1.id).order_by(CouncilAudit.id.asc())
    ).scalars().all()
    assert len(audits) == 2
    assert "[FORCED_REANALYSIS]" in audits[1].execution_notes

    # Verification 3: FINANCIAL IDEMPOTENCY INVARIANTS
    # Strictly 1 BUY order in the ledger
    buy_orders_2 = session.execute(
        select(PaperPortfolio).where(PaperPortfolio.action == "BUY")
    ).scalars().all()
    assert len(buy_orders_2) == 1

    # Total rows in paper_portfolio (including INITIAL_DEPOSIT if any): exactly 2
    all_portfolio = session.execute(select(PaperPortfolio)).scalars().all()
    assert len(all_portfolio) == 2

    # Balances and fees must be 100% IDENTICAL to run 1 (zero duplicate deduction)
    assert summ2["cash_balance"] == cash_1
    assert summ2["btc_balance"] == btc_1
    assert summ2["total_fees_usd"] == fees_1


def test_force_false_on_existing_candle_remains_idempotent_noop(isolated_db, monkeypatch):
    """Verifies that force=False on an already recorded candle returns early without modifying data."""
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    candle_data = {"timestamp": t0, "open": 60000.0, "high": 60200.0, "low": 59800.0, "close": 60100.0, "volume": 75.0}

    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: candle_data)
    mock_signals = MagicMock()
    mock_signals.invoke.return_value = {"volatility_range_pct": 1.0}
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)

    mock_ml = MagicMock()
    mock_ml.invoke.return_value = {"expected_return_pct": 0.3, "predicted_target_price": 60200.0, "confidence": 0.7, "clears_hurdle": True}
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)
    monkeypatch.setattr(sync, "run_trading_graph", lambda **k: {"final_decision": "HOLD", "risk_verdict": "APPROVED", "strategist_action": "HOLD", "strategist_conviction": 5, "kelly_fraction": 0.10})

    # Run 1
    ok1, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok1 is True

    # Run 2 with force=False
    ok2, msg2, summ2 = sync.sync_hourly_quant_cycle(force=False)
    assert ok2 is False
    assert "already recorded" in msg2
    assert "tick_id" in summ2

    # Verify no new rows created
    assert session.execute(select(MarketTick)).scalars().all().__len__() == 1
    assert session.execute(select(CouncilAudit)).scalars().all().__len__() == 1


def test_force_sync_on_tick_without_paper_trade_executes_once(isolated_db, monkeypatch):
    """
    Verifies that if a MarketTick exists in the database without any linked paper order
    (e.g., ingested via offline outbox sync), calling force=True executes the trade exactly once.
    """
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    candle_data = {"timestamp": t0, "open": 61000.0, "high": 61200.0, "low": 60900.0, "close": 61100.0, "volume": 80.0}

    # 1. Pre-insert MarketTick without paper trade or audit
    offline_tick = MarketTick(
        timestamp=t0,
        open=61000.0,
        high=61200.0,
        low=60900.0,
        close=61100.0,
        volume=80.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(offline_tick)
    session.commit()
    tick_id = offline_tick.id

    # Verify no paper portfolio rows exist
    assert session.execute(select(PaperPortfolio).where(PaperPortfolio.tick_id == tick_id)).scalar_one_or_none() is None

    # Setup mocks
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: candle_data)
    mock_signals = MagicMock()
    mock_signals.invoke.return_value = {"volatility_range_pct": 1.0}
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)
    mock_ml = MagicMock()
    mock_ml.invoke.return_value = {"expected_return_pct": 0.5, "predicted_target_price": 61400.0, "confidence": 0.8, "clears_hurdle": True}
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)
    monkeypatch.setattr(sync, "run_trading_graph", lambda **k: {"final_decision": "BUY", "risk_verdict": "APPROVED", "strategist_action": "BUY", "strategist_conviction": 7, "kelly_fraction": 0.10, "thesis": "Catch up execution"})

    # 2. First force run executes the missing paper trade
    ok1, msg1, summ1 = sync.sync_hourly_quant_cycle(force=True)
    assert ok1 is True
    assert "Recorded tick" in msg1

    linked_order = session.execute(
        select(PaperPortfolio).where(PaperPortfolio.tick_id == tick_id)
    ).scalar_one_or_none()
    assert linked_order is not None
    assert linked_order.action == "BUY"

    # 3. Second force run must now deduplicate
    ok2, msg2, summ2 = sync.sync_hourly_quant_cycle(force=True)
    assert ok2 is True
    assert "Trade deduplicated | No fees" in msg2

    all_orders = session.execute(
        select(PaperPortfolio).where(PaperPortfolio.tick_id == tick_id)
    ).scalars().all()
    assert len(all_orders) == 1


def test_new_candle_executes_new_trade_normally(isolated_db, monkeypatch):
    """Verifies that subsequent hourly candles at T+1 execute new trades normally."""
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)

    mock_signals = MagicMock()
    mock_signals.invoke.return_value = {"volatility_range_pct": 1.0}
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)
    mock_ml = MagicMock()
    mock_ml.invoke.return_value = {"expected_return_pct": 0.5, "predicted_target_price": 62300.0, "confidence": 0.8, "clears_hurdle": True}
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)
    monkeypatch.setattr(sync, "run_trading_graph", lambda **k: {"final_decision": "BUY", "risk_verdict": "APPROVED", "strategist_action": "BUY", "strategist_conviction": 8, "kelly_fraction": 0.10})

    # Candle 0
    c0 = {"timestamp": t0, "open": 62000.0, "high": 62200.0, "low": 61900.0, "close": 62100.0, "volume": 70.0}
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c0)
    ok0, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok0 is True

    # Candle 1 (next hour)
    c1 = {"timestamp": t1, "open": 62100.0, "high": 62400.0, "low": 62000.0, "close": 62300.0, "volume": 75.0}
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c1)
    ok1, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok1 is True

    # Two distinct ticks and two distinct trade executions
    ticks = session.execute(select(MarketTick).order_by(MarketTick.timestamp.asc())).scalars().all()
    assert len(ticks) == 2

    trades = session.execute(select(PaperPortfolio).where(PaperPortfolio.action == "BUY")).scalars().all()
    assert len(trades) == 2
    assert trades[0].tick_id == ticks[0].id
    assert trades[1].tick_id == ticks[1].id


def test_atomic_rollback_on_failure_leaves_zero_partial_records(isolated_db, monkeypatch):
    """Verifies that an unhandled exception inside sync rolls back all tables atomically."""
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc)
    c0 = {"timestamp": t0, "open": 63000.0, "high": 63200.0, "low": 62900.0, "close": 63100.0, "volume": 60.0}

    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c0)
    mock_signals = MagicMock()
    mock_signals.invoke.return_value = {"volatility_range_pct": 1.0}
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)
    mock_ml = MagicMock()
    mock_ml.invoke.return_value = {"expected_return_pct": 0.5, "predicted_target_price": 63400.0, "confidence": 0.8, "clears_hurdle": True}
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)

    # Inject failure during Council Deliberation
    def _crashing_deliberation(**k):
        raise RuntimeError("LLM rate limit or network partition")

    monkeypatch.setattr(sync, "run_trading_graph", _crashing_deliberation)

    with pytest.raises(RuntimeError, match="LLM rate limit"):
        sync.sync_hourly_quant_cycle(force=False)

    # Verify atomic rollback: ZERO ticks, ZERO audits, ZERO portfolio changes
    session.expire_all()
    assert session.execute(select(MarketTick)).scalars().all().__len__() == 0
    assert session.execute(select(CouncilAudit)).scalars().all().__len__() == 0
    assert session.execute(select(PaperPortfolio)).scalars().all().__len__() == 0


def test_execute_paper_order_concurrent_idempotency_guard(isolated_db):
    """
    Verifies that calling execute_paper_order directly with the same tick_id:
    1. Returns the existing committed record on duplicate invocation.
    2. Does not apply double deduction of fees or cash.
    3. Prevents duplicate database records via the unique tick_id index.
    """
    session, _ = isolated_db
    order1 = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10, tick_id=501)
    session.commit()

    cash1 = order1.cash_balance
    btc1 = order1.btc_balance
    id1 = order1.id

    # Call 2 with identical tick_id
    order2 = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10, tick_id=501)
    assert order2.id == id1
    assert order2.cash_balance == cash1
    assert order2.btc_balance == btc1

    # Verify only 1 order with tick_id=501 in database
    records = session.execute(select(PaperPortfolio).where(PaperPortfolio.tick_id == 501)).scalars().all()
    assert len(records) == 1


def test_execute_paper_order_savepoint_conflict_recovery(isolated_db, monkeypatch):
    """
    Verifies that when a concurrent race causes Layer 0's pre-check to miss a committed order,
    the nested transaction savepoint catches IntegrityError on unique tick_id collision,
    rolls back the savepoint cleanly, and recovers the winning committed order (Finding D-05).
    """
    session, _ = isolated_db
    order1 = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10, tick_id=701)
    session.commit()

    # Simulate race window where Layer 0 pre-check misses the order (e.g. concurrent execution before commit)
    real_execute = session.execute
    call_count = 0

    def _simulated_race_execute(stmt, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First execute in execute_paper_order is the Layer 0 query: return None
            mock_res = MagicMock()
            mock_res.scalar_one_or_none.return_value = None
            return mock_res
        return real_execute(stmt, *args, **kwargs)

    monkeypatch.setattr(session, "execute", _simulated_race_execute)

    # Calling execute_paper_order will bypass Layer 0, hit IntegrityError in session.flush(),
    # roll back the savepoint cleanly, and recover order1 in the except IntegrityError block.
    order2 = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10, tick_id=701)

    assert order2.id == order1.id
    assert order2.cash_balance == order1.cash_balance
    assert order2.btc_balance == order1.btc_balance

    # Ensure transaction is clean and exactly 1 order exists for tick_id=701
    all_orders = session.execute(select(PaperPortfolio).where(PaperPortfolio.tick_id == 701)).scalars().all()
    assert len(all_orders) == 1


def test_init_db_detects_duplicate_timestamps_and_skips_unique_index():
    """
    Verifies that init_db() detects existing duplicate timestamps in market_ticks,
    skips creating the unique index to prevent crash/corruption, and preserves all existing
    historical records without deletion or alteration (Finding D-02).
    """
    from sqlalchemy import create_engine, inspect, Table, Column, Integer, DateTime, Float, String, MetaData, func
    from src.database.connection import init_db as run_init_db

    test_engine = create_engine("sqlite:///:memory:")
    meta = MetaData()
    ticks_table = Table(
        "market_ticks",
        meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", DateTime(timezone=True), nullable=False),
        Column("open", Float, nullable=False),
        Column("high", Float, nullable=False),
        Column("low", Float, nullable=False),
        Column("close", Float, nullable=False),
        Column("volume", Float, nullable=False),
        Column("model_version", String(32), nullable=False, default="v1"),
        Column("feature_schema_version", String(16), nullable=False, default="v1"),
    )
    meta.create_all(test_engine)

    t_dup = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    with test_engine.connect() as conn:
        conn.execute(ticks_table.insert(), [
            {"timestamp": t_dup, "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 10.0},
            {"timestamp": t_dup, "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 12.0},
        ])
        conn.commit()

    # Run init_db on this engine with duplicate rows
    run_init_db(test_engine)

    # 1. Verify zero historical records deleted (both duplicate rows preserved)
    with test_engine.connect() as conn:
        row_count = conn.execute(select(func.count()).select_from(ticks_table)).scalar()
        assert row_count == 2

    # 2. Verify uq_market_ticks_timestamp was NOT created
    inspector = inspect(test_engine)
    indexes = [idx["name"] for idx in inspector.get_indexes("market_ticks")]
    assert "uq_market_ticks_timestamp" not in indexes


def test_init_db_fails_closed_when_column_addition_fails(monkeypatch):
    """
    Verifies that init_db() raises RuntimeError with actionable diagnostics
    if adding the required tick_id column fails, rather than silently swallowing the error (Finding D-01).
    """
    from sqlalchemy import create_engine
    from src.database.connection import init_db as run_init_db

    test_engine = create_engine("sqlite:///:memory:")
    real_connect = test_engine.connect

    def _crashing_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        real_exec = conn.execute

        def _exec_with_failure(stmt, *a, **kw):
            stmt_str = str(stmt)
            if "ALTER TABLE paper_portfolio ADD COLUMN" in stmt_str:
                raise RuntimeError("Simulated disk I/O or permissions failure during ALTER TABLE")
            return real_exec(stmt, *a, **kw)

        conn.execute = _exec_with_failure
        return conn

    # First create legacy table without tick_id
    with test_engine.connect() as conn:
        conn.execute(text("CREATE TABLE paper_portfolio (id INTEGER PRIMARY KEY AUTOINCREMENT, action VARCHAR(16));"))
        conn.commit()

    monkeypatch.setattr(test_engine, "connect", _crashing_connect)

    with pytest.raises(RuntimeError, match="Database migration failed: unable to add required column 'tick_id'"):
        run_init_db(test_engine)


def test_has_tick_id_column_and_fallback_on_legacy_schema():
    """
    Verifies that has_tick_id_column accurately detects when tick_id is absent from disk,
    and is_paper_trade_executed_for_tick safely uses the CouncilAudit fallback without crashing (Finding D-04).
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.database.connection import has_tick_id_column
    from src.database.models import CouncilAudit
    from src.engine.sync import is_paper_trade_executed_for_tick

    test_engine = create_engine("sqlite:///:memory:")
    with test_engine.connect() as conn:
        conn.execute(text("CREATE TABLE paper_portfolio (id INTEGER PRIMARY KEY, action VARCHAR(16));"))
        conn.commit()
    CouncilAudit.__table__.create(test_engine)
    with test_engine.connect() as conn:
        conn.execute(
            CouncilAudit.__table__.insert(),
            [{
                "id": 1,
                "tick_id": 999,
                "timestamp": datetime.now(timezone.utc),
                "strategist_action": "HOLD",
                "strategist_conviction": 5,
                "risk_verdict": "APPROVED",
                "kelly_fraction": 0.1,
                "final_decision": "HOLD",
                "execution_notes": "Initial thesis",
            }],
        )
        conn.commit()

    Session = sessionmaker(bind=test_engine)
    session = Session()

    # 1. has_tick_id_column detects tick_id is absent on disk
    assert has_tick_id_column(session) is False

    # 2. is_paper_trade_executed_for_tick uses CouncilAudit fallback cleanly without SQL crash
    executed = is_paper_trade_executed_for_tick(session, 999)
    assert executed is True

    # 3. For a tick without council audit, returns False
    not_executed = is_paper_trade_executed_for_tick(session, 888)
    assert not_executed is False

    session.close()


# -----------------------------------------------------------------------------
# 4. Candle Gap Detection & Historical Backfill Contracts (TASK-P1-05.7)
# -----------------------------------------------------------------------------

def test_backfill_empty_database_bootstrap_inserts_zero_historical_candles(isolated_db, monkeypatch):
    """
    Verifies that on an empty database (initial bootstrap with zero prior MarketTick rows),
    sync_hourly_quant_cycle records ONLY the latest confirmed candle and does NOT dump
    the 99 preceding historical candles from candle['candles'] into market_ticks.
    """
    import pandas as pd
    session, _ = isolated_db
    t_latest = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)

    history = []
    for i in range(50, -1, -1):
        ts_i = t_latest - timedelta(hours=i)
        history.append({
            "timestamp": pd.Timestamp(ts_i),
            "open": 60000.0 + i,
            "high": 60200.0 + i,
            "low": 59900.0 + i,
            "close": 60100.0 + i,
            "volume": 50.0 + i,
            "taker_buy_ratio": 0.52,
            "number_of_trades": 1000 + i,
        })

    candle_data = {
        "timestamp": t_latest,
        "open": 60000.0,
        "high": 60200.0,
        "low": 59900.0,
        "close": 60100.0,
        "volume": 50.0,
        "taker_buy_ratio": 0.52,
        "number_of_trades": 1000,
        "candles": history,
    }

    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: candle_data)
    mock_signals = MagicMock()
    mock_signals.invoke.return_value = {"volatility_range_pct": 1.1}
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)
    mock_ml = MagicMock()
    mock_ml.invoke.return_value = {
        "expected_return_pct": 0.35,
        "predicted_target_price": 60310.35,
        "confidence": 0.75,
        "clears_hurdle": True,
    }
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)
    monkeypatch.setattr(
        sync,
        "run_trading_graph",
        lambda **k: {
            "final_decision": "HOLD",
            "risk_verdict": "APPROVED",
            "strategist_action": "HOLD",
            "strategist_conviction": 5,
            "kelly_fraction": 0.10,
            "thesis": "Bootstrap hold",
        },
    )

    ok, msg, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok is True
    assert "Recorded tick" in msg

    all_ticks = session.execute(select(MarketTick).order_by(MarketTick.timestamp.asc())).scalars().all()
    assert len(all_ticks) == 1
    assert sync._normalize_utc_timestamp(all_ticks[0].timestamp) == t_latest
    assert all_ticks[0].predicted_price_1h == pytest.approx(60310.35)


def test_single_missing_hour_gap_backfills_and_reconciles_prior_prediction(isolated_db, monkeypatch):
    """
    Verifies that when a single hourly worker cycle is missed (e.g. t0=10:00 recorded,
    t1=11:00 missed, worker runs at t2=12:00):
    1. t1 (11:00 UTC) is backfilled into market_ticks with OHLCV and NULL prediction/feature fields.
    2. t0 (10:00 UTC) is reconciled against t1's realized close ($60,240.00), NOT t2's ($60,900.00).
    3. Backfilled tick t1 receives zero Reconciliation rows, zero CouncilAudit rows, and zero PaperPortfolio trades.
    """
    import pandas as pd
    from src.database.models import Reconciliation

    session, _ = isolated_db
    t0 = datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 2, 2, 11, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 2, 2, 12, 0, tzinfo=timezone.utc)

    mock_signals = MagicMock()
    mock_signals.invoke.return_value = {"volatility_range_pct": 1.0}
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)

    mock_ml = MagicMock()
    mock_ml.invoke.return_value = {
        "expected_return_pct": 0.50,
        "predicted_target_price": 60300.0,
        "confidence": 0.80,
        "clears_hurdle": True,
    }
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)
    monkeypatch.setattr(
        sync,
        "run_trading_graph",
        lambda **k: {
            "final_decision": "BUY",
            "risk_verdict": "APPROVED",
            "strategist_action": "BUY",
            "strategist_conviction": 8,
            "kelly_fraction": 0.10,
            "thesis": "Bullish setup",
        },
    )

    # 1. Run at t0 (10:00 UTC)
    c0 = {
        "timestamp": t0,
        "open": 59900.0,
        "high": 60100.0,
        "low": 59800.0,
        "close": 60000.0,
        "volume": 80.0,
        "candles": [
            {"timestamp": pd.Timestamp(t0), "open": 59900.0, "high": 60100.0, "low": 59800.0, "close": 60000.0, "volume": 80.0}
        ],
    }
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c0)
    ok0, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok0 is True

    # 2. Worker misses t1 (11:00 UTC) and runs at t2 (12:00 UTC)
    mock_ml.invoke.return_value = {
        "expected_return_pct": 0.40,
        "predicted_target_price": 61143.60,
        "confidence": 0.78,
        "clears_hurdle": True,
    }
    c2 = {
        "timestamp": t2,
        "open": 60240.0,
        "high": 61000.0,
        "low": 60200.0,
        "close": 60900.0,
        "volume": 110.0,
        "candles": [
            {"timestamp": pd.Timestamp(t0), "open": 59900.0, "high": 60100.0, "low": 59800.0, "close": 60000.0, "volume": 80.0},
            {"timestamp": pd.Timestamp(t1), "open": 60000.0, "high": 60350.0, "low": 59950.0, "close": 60240.0, "volume": 95.0},
            {"timestamp": pd.Timestamp(t2), "open": 60240.0, "high": 61000.0, "low": 60200.0, "close": 60900.0, "volume": 110.0},
        ],
    }
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c2)
    ok2, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok2 is True

    session.expire_all()
    ticks = session.execute(select(MarketTick).order_by(MarketTick.timestamp.asc())).scalars().all()
    assert len(ticks) == 3

    tick_t0, tick_t1, tick_t2 = ticks[0], ticks[1], ticks[2]
    assert sync._normalize_utc_timestamp(tick_t0.timestamp) == t0
    assert sync._normalize_utc_timestamp(tick_t1.timestamp) == t1
    assert sync._normalize_utc_timestamp(tick_t2.timestamp) == t2

    # Backfilled tick t1 has exact OHLCV and NULL forecast/feature fields
    assert tick_t1.open == 60000.0
    assert tick_t1.high == 60350.0
    assert tick_t1.low == 59950.0
    assert tick_t1.close == 60240.0
    assert tick_t1.volume == 95.0
    assert tick_t1.market_regime is None
    assert tick_t1.predicted_return_1h is None
    assert tick_t1.predicted_price_1h is None
    assert tick_t1.model_confidence is None
    assert tick_t1.features is None

    # t0 is reconciled strictly against backfilled t1 ($60,240.00), NOT t2 ($60,900.00)
    recs = session.execute(select(Reconciliation)).scalars().all()
    assert len(recs) == 1
    assert recs[0].tick_id == tick_t0.id
    assert recs[0].actual_price_at_t_plus_1 == pytest.approx(60240.0)
    assert recs[0].direction_correct is True

    # Zero CouncilAudit or PaperPortfolio records for backfilled tick_t1
    audits = session.execute(select(CouncilAudit).order_by(CouncilAudit.id.asc())).scalars().all()
    assert [a.tick_id for a in audits] == [tick_t0.id, tick_t2.id]

    orders = session.execute(select(PaperPortfolio).where(PaperPortfolio.tick_id.isnot(None))).scalars().all()
    assert [o.tick_id for o in orders] == [tick_t0.id, tick_t2.id]


def test_multi_hour_outage_backfills_chronologically_and_reconciles_without_retrospective_trades(isolated_db, monkeypatch):
    """
    Verifies a multi-hour outage (t0=10:00 recorded, t1=11:00, t2=12:00, t3=13:00 missed,
    worker runs at t4=14:00):
    1. t1, t2, t3 are inserted in ascending chronological order even if candle['candles'] is unsorted.
    2. t0's prediction and VETOED CouncilAudit are reconciled against t1 ($59,400.00 -> SAVED_CAPITAL).
    3. Backfilled ticks t1, t2, t3 never generate Reconciliation, CouncilAudit, or PaperPortfolio records.
    """
    import pandas as pd
    from src.database.models import Reconciliation

    session, _ = isolated_db
    t0 = datetime(2026, 2, 3, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)
    t3 = t0 + timedelta(hours=3)
    t4 = t0 + timedelta(hours=4)

    # Seed t0 with a bullish prediction and a VETOED BUY CouncilAudit
    tick0 = MarketTick(
        timestamp=t0,
        open=59900.0,
        high=60100.0,
        low=59800.0,
        close=60000.0,
        volume=100.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
        market_regime="HIGH_VOL_CHOP",
        predicted_return_1h=1.0,
        predicted_price_1h=60600.0,
        model_confidence=0.65,
        features={"volatility_range_pct": 4.2},
    )
    session.add(tick0)
    session.flush()

    audit0 = CouncilAudit(
        timestamp=t0,
        tick_id=tick0.id,
        deliberation_latency_ms=12.0,
        strategist_action="BUY",
        strategist_conviction=6,
        risk_verdict="VETOED",
        veto_reasons=["High volatility chop"],
        kelly_fraction=0.0,
        final_decision="HOLD",
        execution_notes="Vetoed entry",
    )
    session.add(audit0)
    session.commit()
    tick0_id = tick0.id

    # Run quant cycle at t4 with intentionally shuffled history order [t3, t1, t4, t0, t2]
    c4 = {
        "timestamp": t4,
        "open": 59700.0,
        "high": 60200.0,
        "low": 59600.0,
        "close": 60100.0,
        "volume": 90.0,
        "candles": [
            {"timestamp": pd.Timestamp(t3), "open": 59500.0, "high": 59800.0, "low": 59400.0, "close": 59700.0, "volume": 85.0},
            {"timestamp": pd.Timestamp(t1), "open": 60000.0, "high": 60050.0, "low": 59300.0, "close": 59400.0, "volume": 120.0},
            {"timestamp": pd.Timestamp(t4), "open": 59700.0, "high": 60200.0, "low": 59600.0, "close": 60100.0, "volume": 90.0},
            {"timestamp": pd.Timestamp(t0), "open": 59900.0, "high": 60100.0, "low": 59800.0, "close": 60000.0, "volume": 100.0},
            {"timestamp": pd.Timestamp(t2), "open": 59400.0, "high": 59600.0, "low": 59350.0, "close": 59500.0, "volume": 80.0},
        ],
    }
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c4)
    mock_signals = MagicMock()
    mock_signals.invoke.return_value = {"volatility_range_pct": 1.2}
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)
    mock_ml = MagicMock()
    mock_ml.invoke.return_value = {
        "expected_return_pct": 0.25,
        "predicted_target_price": 60250.25,
        "confidence": 0.70,
        "clears_hurdle": True,
    }
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)
    monkeypatch.setattr(
        sync,
        "run_trading_graph",
        lambda **k: {
            "final_decision": "HOLD",
            "risk_verdict": "APPROVED",
            "strategist_action": "HOLD",
            "strategist_conviction": 5,
            "kelly_fraction": 0.10,
            "thesis": "Post-outage stabilization",
        },
    )

    ok4, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok4 is True

    session.expire_all()
    ticks = session.execute(select(MarketTick).order_by(MarketTick.id.asc())).scalars().all()
    assert len(ticks) == 5
    assert [sync._normalize_utc_timestamp(t.timestamp) for t in ticks] == [t0, t1, t2, t3, t4]

    for bf_tick in ticks[1:4]:
        assert bf_tick.predicted_price_1h is None
        assert bf_tick.predicted_return_1h is None
        assert bf_tick.model_confidence is None
        assert bf_tick.market_regime is None
        assert bf_tick.features is None

    # t0 reconciled against t1 ($59,400.00 -> -1.0% drop), attributing VETO as SAVED_CAPITAL
    recs = session.execute(select(Reconciliation)).scalars().all()
    assert len(recs) == 1
    assert recs[0].tick_id == tick0_id
    assert recs[0].actual_price_at_t_plus_1 == pytest.approx(59400.0)
    assert recs[0].actual_return_pct == pytest.approx(-1.0)

    updated_audit0 = session.execute(select(CouncilAudit).where(CouncilAudit.tick_id == tick0_id)).scalar_one()
    assert updated_audit0.veto_attribution == "SAVED_CAPITAL"


def test_non_contiguous_intermediate_gaps_healed_and_reconciled(isolated_db, monkeypatch):
    """
    Verifies that when market_ticks has non-contiguous gaps within the fetched window
    (e.g. t0=10:00 and t2=12:00 exist with predictions, while t1=11:00 and t3=13:00 are missing),
    running at t4=14:00 backfills BOTH t1 and t3 without duplicating t0 or t2, and reconciles
    both t0 (against t1) and t2 (against t3).
    """
    import pandas as pd
    from src.database.models import Reconciliation

    session, _ = isolated_db
    t0 = datetime(2026, 2, 4, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)
    t3 = t0 + timedelta(hours=3)
    t4 = t0 + timedelta(hours=4)

    tick0 = MarketTick(
        timestamp=t0,
        open=60000.0,
        high=60100.0,
        low=59900.0,
        close=60000.0,
        volume=50.0,
        predicted_return_1h=0.50,
        predicted_price_1h=60300.0,
    )
    tick2 = MarketTick(
        timestamp=t2,
        open=60500.0,
        high=60600.0,
        low=60400.0,
        close=60500.0,
        volume=60.0,
        predicted_return_1h=0.50,
        predicted_price_1h=60802.5,
    )
    session.add_all([tick0, tick2])
    session.commit()
    id0, id2 = tick0.id, tick2.id

    c4 = {
        "timestamp": t4,
        "open": 61000.0,
        "high": 61200.0,
        "low": 60900.0,
        "close": 61100.0,
        "volume": 75.0,
        "candles": [
            {"timestamp": pd.Timestamp(t0), "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60000.0, "volume": 50.0},
            {"timestamp": pd.Timestamp(t1), "open": 60000.0, "high": 60350.0, "low": 59950.0, "close": 60300.0, "volume": 55.0},
            {"timestamp": pd.Timestamp(t2), "open": 60500.0, "high": 60600.0, "low": 60400.0, "close": 60500.0, "volume": 60.0},
            {"timestamp": pd.Timestamp(t3), "open": 60500.0, "high": 60950.0, "low": 60450.0, "close": 60900.0, "volume": 65.0},
            {"timestamp": pd.Timestamp(t4), "open": 61000.0, "high": 61200.0, "low": 60900.0, "close": 61100.0, "volume": 75.0},
        ],
    }
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c4)
    mock_signals = MagicMock()
    mock_signals.invoke.return_value = {"volatility_range_pct": 1.0}
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)
    mock_ml = MagicMock()
    mock_ml.invoke.return_value = {
        "expected_return_pct": 0.20,
        "predicted_target_price": 61222.2,
        "confidence": 0.68,
        "clears_hurdle": True,
    }
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)
    monkeypatch.setattr(
        sync,
        "run_trading_graph",
        lambda **k: {
            "final_decision": "HOLD",
            "risk_verdict": "APPROVED",
            "strategist_action": "HOLD",
            "strategist_conviction": 5,
            "kelly_fraction": 0.10,
        },
    )

    ok4, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok4 is True

    session.expire_all()
    all_ticks = session.execute(select(MarketTick).order_by(MarketTick.timestamp.asc())).scalars().all()
    assert len(all_ticks) == 5
    assert [sync._normalize_utc_timestamp(t.timestamp) for t in all_ticks] == [t0, t1, t2, t3, t4]

    recs = session.execute(select(Reconciliation).order_by(Reconciliation.tick_id.asc())).scalars().all()
    assert len(recs) == 2
    assert recs[0].tick_id == id0
    assert recs[0].actual_price_at_t_plus_1 == pytest.approx(60300.0)
    assert recs[1].tick_id == id2
    assert recs[1].actual_price_at_t_plus_1 == pytest.approx(60900.0)


def test_existing_candle_with_force_false_heals_prior_gap_and_reconciles_idempotently(isolated_db, monkeypatch):
    """
    Verifies that when the latest candle t2 is ALREADY recorded in market_ticks and force=False,
    sync_hourly_quant_cycle still backfills any missing prior candle t1 and reconciles t0
    before returning the idempotent (False, 'Candle ... already recorded in Neon.', {'tick_id': ...}) result.
    """
    import pandas as pd
    from src.database.models import Reconciliation

    session, _ = isolated_db
    t0 = datetime(2026, 2, 5, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)

    tick0 = MarketTick(
        timestamp=t0,
        open=60000.0,
        high=60100.0,
        low=59900.0,
        close=60000.0,
        volume=50.0,
        predicted_return_1h=0.50,
        predicted_price_1h=60300.0,
    )
    tick2 = MarketTick(
        timestamp=t2,
        open=60400.0,
        high=60600.0,
        low=60300.0,
        close=60500.0,
        volume=60.0,
        predicted_return_1h=0.25,
        predicted_price_1h=60651.25,
    )
    session.add_all([tick0, tick2])
    session.commit()
    id0, id2 = tick0.id, tick2.id

    c2 = {
        "timestamp": t2,
        "open": 60400.0,
        "high": 60600.0,
        "low": 60300.0,
        "close": 60500.0,
        "volume": 60.0,
        "candles": [
            {"timestamp": pd.Timestamp(t0), "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60000.0, "volume": 50.0},
            {"timestamp": pd.Timestamp(t1), "open": 60000.0, "high": 60450.0, "low": 59950.0, "close": 60400.0, "volume": 55.0},
            {"timestamp": pd.Timestamp(t2), "open": 60400.0, "high": 60600.0, "low": 60300.0, "close": 60500.0, "volume": 60.0},
        ],
    }
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c2)

    ok, msg, summ = sync.sync_hourly_quant_cycle(force=False)
    assert ok is False
    assert "already recorded in Neon" in msg
    assert summ == {"tick_id": id2}

    session.expire_all()
    all_ticks = session.execute(select(MarketTick).order_by(MarketTick.timestamp.asc())).scalars().all()
    assert len(all_ticks) == 3
    assert sync._normalize_utc_timestamp(all_ticks[1].timestamp) == t1
    assert all_ticks[1].close == pytest.approx(60400.0)

    recs = session.execute(select(Reconciliation)).scalars().all()
    assert len(recs) == 1
    assert recs[0].tick_id == id0
    assert recs[0].actual_price_at_t_plus_1 == pytest.approx(60400.0)


def test_large_gap_over_24h_and_window_overflow_logs_warning_and_backfills_available(isolated_db, caplog):
    """
    Verifies gap warning thresholds and window boundaries:
    1. A 30-hour gap (> 24h threshold, within 100-candle window) logs a large-gap warning
       and backfills all 29 missing hourly candles.
    2. An outage exceeding the exchange window (e.g. prior tick 120 hours before latest_ts,
       while exchange window only covers the most recent 10 hours) logs a window-exceeded
       warning and backfills all available historical candles in the window.
    """
    import logging
    import pandas as pd

    session, _ = isolated_db
    t0 = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    session.add(
        MarketTick(
            timestamp=t0,
            open=60000.0,
            high=60100.0,
            low=59900.0,
            close=60000.0,
            volume=50.0,
        )
    )
    session.commit()

    # Case 1: 30-hour gap (t0 to t0 + 30h)
    t_30 = t0 + timedelta(hours=30)
    candles_30 = [
        {
            "timestamp": pd.Timestamp(t0 + timedelta(hours=h)),
            "open": 60000.0 + h,
            "high": 60100.0 + h,
            "low": 59900.0 + h,
            "close": 60050.0 + h,
            "volume": 40.0 + h,
        }
        for h in range(0, 31)
    ]
    with caplog.at_level(logging.WARNING, logger="mudra.sync"):
        inserted_30 = sync.backfill_missing_candles(session, {"timestamp": t_30, "candles": candles_30})
        session.commit()

    assert inserted_30 == 29
    assert "Detected large candle gap of 30.0 hours" in caplog.text

    # Case 2: Outage exceeding available exchange window (last tick is t0+29h, latest is t0+150h, window starts at t0+145h)
    caplog.clear()
    t_150 = t0 + timedelta(hours=150)
    candles_150 = [
        {
            "timestamp": pd.Timestamp(t0 + timedelta(hours=h)),
            "open": 61000.0 + h,
            "high": 61100.0 + h,
            "low": 60900.0 + h,
            "close": 61050.0 + h,
            "volume": 50.0,
        }
        for h in range(145, 151)
    ]
    with caplog.at_level(logging.WARNING, logger="mudra.sync"):
        inserted_150 = sync.backfill_missing_candles(session, {"timestamp": t_150, "candles": candles_150})
        session.commit()

    assert inserted_150 == 5  # hours 145, 146, 147, 148, 149
    assert "exceeds exchange history window" in caplog.text


def test_backfill_skips_malformed_historical_candles_and_handles_savepoint_integrity_error(isolated_db, monkeypatch, caplog):
    """
    Verifies that:
    1. Malformed historical candle entries (missing keys, NaN/inf, negative price, high < low,
       boolean values, invalid timestamp) are logged and skipped without aborting valid backfills.
    2. A concurrent duplicate insert triggering IntegrityError inside begin_nested() is caught
       and rolled back via savepoint without aborting the outer session transaction.
    """
    import logging
    import pandas as pd

    session, _ = isolated_db
    t0 = datetime(2026, 2, 6, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)
    t3 = t0 + timedelta(hours=3)
    t4 = t0 + timedelta(hours=4)

    session.add(
        MarketTick(
            timestamp=t0,
            open=60000.0,
            high=60100.0,
            low=59900.0,
            close=60000.0,
            volume=50.0,
        )
    )
    session.commit()

    mixed_candles = [
        "not-a-dict",
        {"timestamp": pd.Timestamp(t1), "open": 60000.0},  # missing high/low/close/volume
        {"timestamp": "bad-ts", "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 10.0},
        {"timestamp": pd.Timestamp(t1), "open": float("nan"), "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 10.0},
        {"timestamp": pd.Timestamp(t1), "open": True, "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 10.0},
        {"timestamp": pd.Timestamp(t1), "open": -100.0, "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 10.0},
        {"timestamp": pd.Timestamp(t1), "open": 60000.0, "high": 59000.0, "low": 59900.0, "close": 60050.0, "volume": 10.0},  # high < low
        # Valid candle at t2:
        {"timestamp": pd.Timestamp(t2), "open": 60100.0, "high": 60300.0, "low": 60000.0, "close": 60200.0, "volume": 55.0},
        # Valid candle at t3:
        {"timestamp": pd.Timestamp(t3), "open": 60200.0, "high": 60400.0, "low": 60100.0, "close": 60300.0, "volume": 65.0},
    ]

    with caplog.at_level(logging.WARNING, logger="mudra.sync"):
        inserted = sync.backfill_missing_candles(session, {"timestamp": t4, "candles": mixed_candles})
        session.commit()

    assert inserted == 2
    assert "Skipping malformed historical candle entry" in caplog.text

    # Simulate a concurrent worker committing t1 right before our savepoint flush:
    # 1. Commit t1 into market_ticks so uq_market_ticks_timestamp holds t1.
    session.add(
        MarketTick(
            timestamp=t1,
            open=60000.0,
            high=60200.0,
            low=59900.0,
            close=60100.0,
            volume=50.0,
        )
    )
    session.commit()

    # 2. Monkeypatch the existing_rows SELECT inside backfill_missing_candles so it returns
    #    only [t0, t2, t3] (simulating t1 committed concurrently after the SELECT).
    t5 = t0 + timedelta(hours=5)
    real_execute = session.execute
    exec_count = [0]

    def _simulated_race_execute(stmt, *args, **kwargs):
        exec_count[0] += 1
        if exec_count[0] == 2:
            # Second execute in backfill_missing_candles is existing_rows query: omit t1
            mock_res = MagicMock()
            mock_res.scalars.return_value.all.return_value = [t0, t2, t3]
            return mock_res
        return real_execute(stmt, *args, **kwargs)

    monkeypatch.setattr(session, "execute", _simulated_race_execute)

    conflict_candles = [
        {"timestamp": pd.Timestamp(t0), "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60000.0, "volume": 50.0},
        {"timestamp": pd.Timestamp(t1), "open": 60000.0, "high": 60200.0, "low": 59900.0, "close": 60100.0, "volume": 50.0},
        {"timestamp": pd.Timestamp(t4), "open": 60300.0, "high": 60500.0, "low": 60200.0, "close": 60400.0, "volume": 70.0},
    ]
    inserted_after_conflict = sync.backfill_missing_candles(session, {"timestamp": t5, "candles": conflict_candles})
    monkeypatch.setattr(session, "execute", real_execute)
    session.commit()

    # t1 hit IntegrityError on uq_market_ticks_timestamp and rolled back its savepoint cleanly;
    # t4 succeeded in the same outer transaction!
    assert inserted_after_conflict == 1
    persisted_ts = [
        sync._normalize_utc_timestamp(t.timestamp)
        for t in session.execute(select(MarketTick).order_by(MarketTick.timestamp.asc())).scalars().all()
    ]
    assert persisted_ts == [t0, t1, t2, t3, t4]


def test_normalize_utc_timestamp_handles_pandas_naive_and_aware_timestamps():
    """
    Verifies that _normalize_utc_timestamp normalizes pandas Timestamps, naive datetimes,
    non-UTC timezone-aware datetimes, and ISO-8601 strings to exact Python UTC datetimes,
    and rejects invalid inputs.
    """
    import pandas as pd

    expected = datetime(2026, 2, 7, 12, 0, tzinfo=timezone.utc)

    # 1. pandas Timestamp (UTC and naive)
    res_pd_utc = sync._normalize_utc_timestamp(pd.Timestamp("2026-02-07 12:00:00+00:00"))
    assert res_pd_utc == expected
    assert type(res_pd_utc) is datetime

    res_pd_naive = sync._normalize_utc_timestamp(pd.Timestamp("2026-02-07 12:00:00"))
    assert res_pd_naive == expected
    assert type(res_pd_naive) is datetime

    # 2. Naive Python datetime and +05:30 offset aware datetime
    res_naive = sync._normalize_utc_timestamp(datetime(2026, 2, 7, 12, 0))
    assert res_naive == expected
    assert res_naive.tzinfo == timezone.utc

    ist_tz = timezone(timedelta(hours=5, minutes=30))
    res_ist = sync._normalize_utc_timestamp(datetime(2026, 2, 7, 17, 30, tzinfo=ist_tz))
    assert res_ist == expected

    # 3. ISO-8601 strings
    assert sync._normalize_utc_timestamp("2026-02-07T12:00:00Z") == expected
    assert sync._normalize_utc_timestamp("2026-02-07T17:30:00+05:30") == expected

    # 4. Invalid inputs raise ValueError or TypeError
    for bad_val in (None, True, False, float("nan"), pd.NaT, "", "   ", 12345):
        with pytest.raises((ValueError, TypeError)):
            sync._normalize_utc_timestamp(bad_val)


def test_existing_candle_with_force_false_reconciles_when_outcome_candle_already_exists(isolated_db, monkeypatch):
    """
    Verifies that when the latest candle (t2) AND the relevant outcome candle (t1) ALREADY
    exist in market_ticks (so backfilled_count == 0), while an earlier prediction (t0)
    remains unresolved, running sync_hourly_quant_cycle(force=False):
    1. Reconciles t0 against t1's realized close ($60,420.00) even though backfilled_count == 0.
    2. Does not rerun the trading graph, create CouncilAudit rows, or execute new paper trades.
    3. Preserves strict idempotency on repeated force=False runs (reconciled exactly once).
    """
    import pandas as pd
    from src.database.models import Reconciliation

    session, _ = isolated_db
    t0 = datetime(2026, 2, 8, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)

    tick0 = MarketTick(
        timestamp=t0,
        open=59900.0,
        high=60100.0,
        low=59800.0,
        close=60000.0,
        volume=50.0,
        predicted_return_1h=0.50,
        predicted_price_1h=60300.0,
    )
    tick1 = MarketTick(
        timestamp=t1,
        open=60000.0,
        high=60500.0,
        low=59950.0,
        close=60420.0,
        volume=55.0,
    )
    tick2 = MarketTick(
        timestamp=t2,
        open=60420.0,
        high=60600.0,
        low=60350.0,
        close=60500.0,
        volume=60.0,
        predicted_return_1h=0.25,
        predicted_price_1h=60651.25,
    )
    session.add_all([tick0, tick1, tick2])
    session.commit()
    id0, id1, id2 = tick0.id, tick1.id, tick2.id

    # Seed one existing paper order for tick0 to verify no additional paper trades are created
    execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10, tick_id=id0)
    session.commit()
    initial_portfolio_count = len(session.execute(select(PaperPortfolio)).scalars().all())
    initial_audit_count = len(session.execute(select(CouncilAudit)).scalars().all())

    # Confirm t0 is initially unresolved and all 3 candles already exist in market_ticks
    assert len(session.execute(select(Reconciliation)).scalars().all()) == 0
    assert len(session.execute(select(MarketTick)).scalars().all()) == 3

    c2 = {
        "timestamp": t2,
        "open": 60420.0,
        "high": 60600.0,
        "low": 60350.0,
        "close": 60500.0,
        "volume": 60.0,
        "candles": [
            {"timestamp": pd.Timestamp(t0), "open": 59900.0, "high": 60100.0, "low": 59800.0, "close": 60000.0, "volume": 50.0},
            {"timestamp": pd.Timestamp(t1), "open": 60000.0, "high": 60500.0, "low": 59950.0, "close": 60420.0, "volume": 55.0},
            {"timestamp": pd.Timestamp(t2), "open": 60420.0, "high": 60600.0, "low": 60350.0, "close": 60500.0, "volume": 60.0},
        ],
    }
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c2)

    graph_calls = [0]
    def _forbidden_graph(**kwargs):
        graph_calls[0] += 1
        raise AssertionError("run_trading_graph must not be called when existing and not force")

    order_calls = [0]
    def _forbidden_order(*args, **kwargs):
        order_calls[0] += 1
        raise AssertionError("execute_paper_order must not be called when existing and not force")

    monkeypatch.setattr(sync, "run_trading_graph", _forbidden_graph)
    monkeypatch.setattr(sync, "execute_paper_order", _forbidden_order)

    # 1. First idempotent run (force=False): backfilled_count is 0, but t0 must be reconciled against t1
    ok1, msg1, summ1 = sync.sync_hourly_quant_cycle(force=False)
    assert ok1 is False
    assert "already recorded in Neon" in msg1
    assert summ1 == {"tick_id": id2}
    assert graph_calls[0] == 0
    assert order_calls[0] == 0

    session.expire_all()
    assert len(session.execute(select(MarketTick)).scalars().all()) == 3
    recs_1 = session.execute(select(Reconciliation)).scalars().all()
    assert len(recs_1) == 1
    assert recs_1[0].tick_id == id0
    assert recs_1[0].actual_price_at_t_plus_1 == pytest.approx(60420.0)
    assert len(session.execute(select(PaperPortfolio)).scalars().all()) == initial_portfolio_count
    assert len(session.execute(select(CouncilAudit)).scalars().all()) == initial_audit_count

    # 2. Second idempotent run (force=False): reconciliation must NOT duplicate
    ok2, msg2, summ2 = sync.sync_hourly_quant_cycle(force=False)
    assert ok2 is False
    assert "already recorded in Neon" in msg2
    assert summ2 == {"tick_id": id2}

    session.expire_all()
    recs_2 = session.execute(select(Reconciliation)).scalars().all()
    assert len(recs_2) == 1
    assert recs_2[0].id == recs_1[0].id
    assert len(session.execute(select(PaperPortfolio)).scalars().all()) == initial_portfolio_count
    assert len(session.execute(select(CouncilAudit)).scalars().all()) == initial_audit_count


def test_backfill_inserts_window_start_when_prior_tick_predates_window_and_later_tick_exists(isolated_db, monkeypatch):
    """
    Verifies that when:
    - The database has a prior tick earlier than the fetched history window (e.g. 09:00 UTC),
    - The fetched window starts at 10:00 UTC,
    - 10:00 UTC is missing from market_ticks,
    - 11:00 UTC already exists in market_ticks,
    - 12:00 UTC is missing from market_ticks, and
    - The current sync cycle runs for latest candle 13:00 UTC:
    1. 10:00 UTC is backfilled into market_ticks (not skipped due to 11:00 UTC existing).
    2. 11:00 UTC is not duplicated.
    3. Later eligible missing candle 12:00 UTC is also backfilled.
    4. Delayed reconciliation settles both 09:00 UTC (against 10:00 UTC) and 11:00 UTC (against 12:00 UTC).
    5. No retrospective predictions, CouncilAudit rows, or PaperPortfolio trades are created for backfilled ticks (10:00 and 12:00).
    """
    import pandas as pd
    from src.database.models import Reconciliation

    session, _ = isolated_db
    t_pre = datetime(2026, 2, 9, 9, 0, tzinfo=timezone.utc)   # Earlier than fetched window
    t_10 = datetime(2026, 2, 9, 10, 0, tzinfo=timezone.utc)   # Fetched window start (missing)
    t_11 = datetime(2026, 2, 9, 11, 0, tzinfo=timezone.utc)   # Already exists in DB
    t_12 = datetime(2026, 2, 9, 12, 0, tzinfo=timezone.utc)   # Later eligible missing candle
    t_13 = datetime(2026, 2, 9, 13, 0, tzinfo=timezone.utc)   # Latest confirmed candle

    tick_pre = MarketTick(
        timestamp=t_pre,
        open=59800.0,
        high=60050.0,
        low=59750.0,
        close=60000.0,
        volume=45.0,
        predicted_return_1h=0.50,
        predicted_price_1h=60300.0,
    )
    tick_11 = MarketTick(
        timestamp=t_11,
        open=60250.0,
        high=60500.0,
        low=60200.0,
        close=60400.0,
        volume=55.0,
        predicted_return_1h=0.50,
        predicted_price_1h=60702.0,
    )
    session.add_all([tick_pre, tick_11])
    session.commit()
    id_pre, id_11 = tick_pre.id, tick_11.id

    # Fetched window starts at 10:00 UTC (t_pre at 09:00 UTC is outside/earlier than the fetched window)
    c13 = {
        "timestamp": t_13,
        "open": 60800.0,
        "high": 61000.0,
        "low": 60750.0,
        "close": 60900.0,
        "volume": 70.0,
        "candles": [
            {"timestamp": pd.Timestamp(t_10), "open": 60000.0, "high": 60300.0, "low": 59950.0, "close": 60250.0, "volume": 50.0},
            {"timestamp": pd.Timestamp(t_11), "open": 60250.0, "high": 60500.0, "low": 60200.0, "close": 60400.0, "volume": 55.0},
            {"timestamp": pd.Timestamp(t_12), "open": 60400.0, "high": 60850.0, "low": 60350.0, "close": 60800.0, "volume": 60.0},
            {"timestamp": pd.Timestamp(t_13), "open": 60800.0, "high": 61000.0, "low": 60750.0, "close": 60900.0, "volume": 70.0},
        ],
    }
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c13)

    mock_signals = MagicMock()
    mock_signals.invoke.return_value = {"volatility_range_pct": 1.1}
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)

    mock_ml = MagicMock()
    mock_ml.invoke.return_value = {
        "expected_return_pct": 0.30,
        "predicted_target_price": 61082.70,
        "confidence": 0.76,
        "clears_hurdle": True,
    }
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)
    monkeypatch.setattr(
        sync,
        "run_trading_graph",
        lambda **k: {
            "final_decision": "BUY",
            "risk_verdict": "APPROVED",
            "strategist_action": "BUY",
            "strategist_conviction": 7,
            "kelly_fraction": 0.10,
            "thesis": "Window-start gap recovery test",
        },
    )

    ok, msg, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok is True
    assert "Recorded tick" in msg

    session.expire_all()
    all_ticks = session.execute(select(MarketTick).order_by(MarketTick.timestamp.asc())).scalars().all()
    assert len(all_ticks) == 5
    assert [sync._normalize_utc_timestamp(t.timestamp) for t in all_ticks] == [t_pre, t_10, t_11, t_12, t_13]

    tick_10_db = all_ticks[1]
    tick_11_db = all_ticks[2]
    tick_12_db = all_ticks[3]
    tick_13_db = all_ticks[4]

    # 1. 10:00 and 12:00 are backfilled with OHLCV and NULL prediction/feature fields
    assert tick_10_db.close == pytest.approx(60250.0)
    assert tick_10_db.predicted_price_1h is None
    assert tick_10_db.predicted_return_1h is None
    assert tick_10_db.model_confidence is None
    assert tick_10_db.market_regime is None
    assert tick_10_db.features is None

    assert tick_12_db.close == pytest.approx(60800.0)
    assert tick_12_db.predicted_price_1h is None
    assert tick_12_db.predicted_return_1h is None
    assert tick_12_db.model_confidence is None
    assert tick_12_db.market_regime is None
    assert tick_12_db.features is None

    # 2. 11:00 is not duplicated and retains its original record & prediction
    assert tick_11_db.id == id_11
    assert tick_11_db.predicted_price_1h == pytest.approx(60702.0)

    # 3. Both 09:00 (against backfilled 10:00) and 11:00 (against backfilled 12:00) are reconciled
    recs = session.execute(select(Reconciliation).order_by(Reconciliation.tick_id.asc())).scalars().all()
    assert len(recs) == 2
    assert recs[0].tick_id == id_pre
    assert recs[0].actual_price_at_t_plus_1 == pytest.approx(60250.0)
    assert recs[1].tick_id == id_11
    assert recs[1].actual_price_at_t_plus_1 == pytest.approx(60800.0)

    # 4. Zero CouncilAudit rows and zero PaperPortfolio trades for backfilled ticks (10:00 and 12:00)
    audits = session.execute(select(CouncilAudit)).scalars().all()
    assert [a.tick_id for a in audits] == [tick_13_db.id]

    orders = session.execute(select(PaperPortfolio).where(PaperPortfolio.tick_id.isnot(None))).scalars().all()
    assert [o.tick_id for o in orders] == [tick_13_db.id]