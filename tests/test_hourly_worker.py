# tests/test_hourly_worker.py
"""
Deterministic Unit & Regression Tests for Mudra Hourly Worker Lifecycle,
Exit Code Contracts, and Durable Outbox Processing.
TASK-P1-05.3: Verify External Scheduler Integration, Then Fix Worker Reliability.
"""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import scripts.run_hourly_worker as worker
from src.database.models import Base, MarketTick


@pytest.fixture
def isolated_db(monkeypatch):
    """Provides an isolated in-memory SQLite database session for worker testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    
    # Patch connection engine and SessionLocal in database.connection and worker
    monkeypatch.setattr("src.database.connection.engine", engine)
    monkeypatch.setattr("src.database.connection.SessionLocal", Session)
    monkeypatch.setattr("src.database.connection.test_connection", lambda: (True, "SQLite 3.x", 1.0))
    monkeypatch.setattr("src.database.connection.init_db", lambda target_engine=None: Base.metadata.create_all(bind=engine))
    
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
    """Verifies that database schema initialization failure returns False (exit 1)."""
    def _crashing_init(target_engine=None):
        raise RuntimeError("PostgreSQL relation creation failure")

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

