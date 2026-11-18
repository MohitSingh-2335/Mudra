# tests/test_hourly_worker.py
"""Unit tests for Mudra hourly worker lifecycle, outbox processing, and backfill."""

import json
import logging
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import scripts.run_hourly_worker as worker
import src.engine.sync as sync
from src.database.models import Base, MarketTick, CouncilAudit, PaperPortfolio, Reconciliation
from src.database.connection import init_db as real_init_db
from src.engine.paper_portfolio import execute_paper_order


@pytest.fixture
def isolated_db(monkeypatch):
    """Isolated in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    real_init_db(engine)
    Session = sessionmaker(bind=engine)
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
    """Temporary outbox file path."""
    outbox_path = tmp_path / "sync_outbox.json"
    monkeypatch.setattr(worker, "OUTBOX_FILE", outbox_path)
    return outbox_path


def test_worker_success_on_new_sync(isolated_db, temp_outbox, monkeypatch):
    """Worker returns True on new tick sync."""
    monkeypatch.setattr(worker, "sync_hourly_quant_cycle", lambda force=False: (True, "Recorded tick #1", {}))
    assert worker.run_worker(force=False) is True


def test_worker_success_on_already_recorded(isolated_db, temp_outbox, monkeypatch):
    """Worker returns True on idempotent already-recorded candle."""
    monkeypatch.setattr(worker, "sync_hourly_quant_cycle", lambda force=False: (False, "Candle already recorded in Neon.", {}))
    assert worker.run_worker(force=False) is True


def test_worker_failure_on_feed_error(isolated_db, temp_outbox, monkeypatch):
    """Worker returns False on feed failure."""
    monkeypatch.setattr(worker, "sync_hourly_quant_cycle", lambda force=False: (False, "Market feed error", {}))
    assert worker.run_worker(force=False) is False


def test_worker_failure_on_db_timeout(isolated_db, temp_outbox, monkeypatch):
    """Worker returns False when db is unreachable."""
    monkeypatch.setattr("src.database.connection.test_connection", lambda: (False, "Timeout", 5000.0))
    assert worker.run_worker(force=False) is False


def test_worker_failure_on_exception(isolated_db, temp_outbox, monkeypatch):
    """Worker returns False on unhandled exceptions."""
    monkeypatch.setattr(worker, "sync_hourly_quant_cycle", MagicMock(side_effect=RuntimeError("crash")))
    assert worker.run_worker(force=False) is False


def test_outbox_missing_noop(isolated_db, temp_outbox):
    """Missing outbox file is safe no-op."""
    session, _ = isolated_db
    assert worker.flush_outbox(session) == 0


def test_outbox_empty_unlinked(isolated_db, temp_outbox):
    """Empty outbox file is unlinked."""
    session, _ = isolated_db
    temp_outbox.write_text("   \n ", encoding="utf-8")
    assert worker.flush_outbox(session) == 0
    assert not temp_outbox.exists()


def test_outbox_malformed_preserved(isolated_db, temp_outbox):
    """Malformed outbox file is preserved for investigation."""
    session, _ = isolated_db
    temp_outbox.write_text("{corrupt json", encoding="utf-8")
    assert worker.flush_outbox(session) == 0
    assert temp_outbox.exists()


def test_outbox_flush_persists_and_unlinks(isolated_db, temp_outbox):
    """Valid outbox rows are inserted into market_ticks and file is deleted."""
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    temp_outbox.write_text(json.dumps([{
        "type": "tick",
        "payload": {
            "timestamp": t0.isoformat(),
            "open": 60000.0, "high": 60500.0, "low": 59500.0, "close": 60200.0, "volume": 120.0,
            "model_version": "xgb_v1", "feature_schema_version": "v1_39features",
            "features": {"rsi": 55.4},
        }
    }]), encoding="utf-8")

    assert worker.flush_outbox(session) == 1
    assert not temp_outbox.exists()
    tick = session.execute(select(MarketTick).where(MarketTick.timestamp == t0)).scalar_one_or_none()
    assert tick is not None
    assert tick.close == 60200.0


def test_outbox_replay_idempotent(isolated_db, temp_outbox):
    """Replaying an outbox with already-persisted tick skips duplicate cleanly."""
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)
    session.add(MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60050.0, volume=10.0))
    session.commit()

    temp_outbox.write_text(json.dumps([{
        "type": "tick",
        "payload": {"timestamp": t0.isoformat(), "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 10.0}
    }]), encoding="utf-8")

    assert worker.flush_outbox(session) == 0
    assert not temp_outbox.exists()


def test_force_sync_deduplicates_paper_trade(isolated_db, monkeypatch):
    """force=True re-evaluates forecasts but never duplicates paper trades or fees."""
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    candle = {"timestamp": t0, "open": 60000.0, "high": 60200.0, "low": 59800.0, "close": 60100.0, "volume": 75.0}

    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: candle)
    monkeypatch.setattr(sync, "calculate_technical_signals", MagicMock(invoke=lambda *a: {"rsi": 55.0}))
    mock_ml = MagicMock(invoke=lambda *a: {"expected_return_pct": 0.45, "predicted_target_price": 60370.0, "confidence": 0.8, "clears_hurdle": True})
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)
    monkeypatch.setattr(sync, "run_trading_graph", lambda **k: {
        "final_decision": "BUY", "risk_verdict": "APPROVED", "strategist_action": "BUY",
        "strategist_conviction": 8, "kelly_fraction": 0.10, "thesis": "Bullish",
    })

    ok1, msg1, summ1 = sync.sync_hourly_quant_cycle(force=False)
    assert ok1 is True
    trades1 = session.execute(select(PaperPortfolio).where(PaperPortfolio.action == "BUY")).scalars().all()
    assert len(trades1) == 1

    # Second run with force=True
    mock_ml.invoke = lambda *a: {"expected_return_pct": 0.95, "predicted_target_price": 60670.0, "confidence": 0.9, "clears_hurdle": True}
    ok2, msg2, summ2 = sync.sync_hourly_quant_cycle(force=True)
    assert ok2 is True
    assert "Trade deduplicated | No fees" in msg2

    trades2 = session.execute(select(PaperPortfolio).where(PaperPortfolio.action == "BUY")).scalars().all()
    assert len(trades2) == 1  # ZERO duplicate trade!
    audits = session.execute(select(CouncilAudit)).scalars().all()
    assert len(audits) == 2
    assert "[FORCED_REANALYSIS]" in audits[1].execution_notes


def test_force_false_idempotent_noop(isolated_db, monkeypatch):
    """force=False on existing candle is an idempotent no-op."""
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    session.add(MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60050.0, volume=10.0))
    session.commit()

    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: {"timestamp": t0, "close": 60050.0, "candles": []})
    ok, msg, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok is False
    assert "already recorded" in msg


def test_atomic_rollback_on_failure(isolated_db, monkeypatch):
    """Unexpected exception in execution rolls back session with zero partial records."""
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc)
    candle = {"timestamp": t0, "open": 60000.0, "high": 60200.0, "low": 59800.0, "close": 60100.0, "volume": 75.0}

    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: candle)
    mock_signals = MagicMock()
    mock_signals.invoke.side_effect = RuntimeError("signal crash")
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)

    with pytest.raises(RuntimeError):
        sync.sync_hourly_quant_cycle(force=False)

    ticks = session.execute(select(MarketTick)).scalars().all()
    assert len(ticks) == 0


def test_execute_paper_order_concurrency(isolated_db):
    """Paper order deduplicates when called with same tick_id."""
    session, _ = isolated_db
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tick = MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60000.0, volume=10.0)
    session.add(tick)
    session.commit()

    o1 = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10, tick_id=tick.id)
    o2 = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10, tick_id=tick.id)
    assert o1.id == o2.id


def test_backfill_empty_db_bootstrap(isolated_db, monkeypatch):
    """Empty db only records latest confirmed candle, not dumping all 100 historical candles."""
    session, _ = isolated_db
    t_latest = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
    history = [{"timestamp": pd.Timestamp(t_latest - timedelta(hours=i)), "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60000.0, "volume": 10.0} for i in range(50, -1, -1)]

    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: {
        "timestamp": t_latest, "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60000.0, "volume": 10.0, "candles": history,
    })
    monkeypatch.setattr(sync, "calculate_technical_signals", MagicMock(invoke=lambda *a: {}))
    monkeypatch.setattr(sync, "get_model_return_forecast", MagicMock(invoke=lambda *a: {"expected_return_pct": 0.1, "predicted_target_price": 60060.0, "confidence": 0.7, "clears_hurdle": False}))
    monkeypatch.setattr(sync, "run_trading_graph", lambda **k: {"final_decision": "HOLD", "risk_verdict": "APPROVED", "strategist_action": "HOLD", "strategist_conviction": 5, "kelly_fraction": 0.0, "thesis": "hold"})

    ok, msg, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok is True
    ticks = session.execute(select(MarketTick)).scalars().all()
    assert len(ticks) == 1


def test_single_missing_hour_backfill_and_reconcile(isolated_db, monkeypatch):
    """Missed hour is backfilled and prior tick prediction is reconciled against it."""
    session, _ = isolated_db
    t0 = datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 2, 2, 11, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 2, 2, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(sync, "calculate_technical_signals", MagicMock(invoke=lambda *a: {}))
    monkeypatch.setattr(sync, "get_model_return_forecast", MagicMock(invoke=lambda *a: {"expected_return_pct": 0.5, "predicted_target_price": 60300.0, "confidence": 0.8, "clears_hurdle": True}))
    monkeypatch.setattr(sync, "run_trading_graph", lambda **k: {"final_decision": "BUY", "risk_verdict": "APPROVED", "strategist_action": "BUY", "strategist_conviction": 8, "kelly_fraction": 0.10, "thesis": "Bullish"})

    # 1. Run at t0 to establish baseline tick with prediction
    c0 = {
        "timestamp": t0, "open": 59900.0, "high": 60100.0, "low": 59800.0, "close": 60000.0, "volume": 80.0,
        "candles": [{"timestamp": pd.Timestamp(t0), "open": 59900.0, "high": 60100.0, "low": 59800.0, "close": 60000.0, "volume": 80.0}],
    }
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c0)
    ok0, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok0 is True

    # 2. Run at t2 with t1 missed in database but present in candle history
    c2 = {
        "timestamp": t2, "open": 60240.0, "high": 61000.0, "low": 60200.0, "close": 60900.0, "volume": 110.0,
        "candles": [
            {"timestamp": pd.Timestamp(t0), "open": 59900.0, "high": 60100.0, "low": 59800.0, "close": 60000.0, "volume": 80.0},
            {"timestamp": pd.Timestamp(t1), "open": 60000.0, "high": 60350.0, "low": 59950.0, "close": 60240.0, "volume": 95.0},
            {"timestamp": pd.Timestamp(t2), "open": 60240.0, "high": 61000.0, "low": 60200.0, "close": 60900.0, "volume": 110.0},
        ],
    }
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c2)
    ok2, msg2, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok2 is True

    ticks = session.execute(select(MarketTick).order_by(MarketTick.timestamp.asc())).scalars().all()
    assert len(ticks) == 3
    assert sync._normalize_utc_timestamp(ticks[1].timestamp) == t1

    # Reconciled t0 against t1 realized close 60240.0
    rec = session.execute(select(Reconciliation)).scalar_one_or_none()
    assert rec is not None
    assert rec.actual_price_at_t_plus_1 == pytest.approx(60240.0)


def test_large_gap_warning(isolated_db, caplog):
    """Large gaps > 24h log warnings."""
    session, _ = isolated_db
    t0 = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    session.add(MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60000.0, volume=50.0))
    session.commit()

    t_30 = t0 + timedelta(hours=30)
    candles_30 = [{"timestamp": pd.Timestamp(t0 + timedelta(hours=h)), "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 40.0} for h in range(31)]

    with caplog.at_level(logging.WARNING, logger="mudra.sync"):
        inserted = sync.backfill_missing_candles(session, {"timestamp": t_30, "candles": candles_30})
        session.commit()

    assert inserted == 29
    assert "Detected large candle gap of 30.0 hours" in caplog.text


def test_normalize_utc_timestamp():
    """_normalize_utc_timestamp handles naive and aware timestamps."""
    expected = datetime(2026, 2, 7, 12, 0, tzinfo=timezone.utc)
    assert sync._normalize_utc_timestamp(pd.Timestamp("2026-02-07 12:00:00+00:00")) == expected
    assert sync._normalize_utc_timestamp(datetime(2026, 2, 7, 12, 0)) == expected
    assert sync._normalize_utc_timestamp("2026-02-07T12:00:00Z") == expected
    with pytest.raises(ValueError):
        sync._normalize_utc_timestamp(None)
