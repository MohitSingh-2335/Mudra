"""
tests/test_portfolio_concurrency.py

Deterministic integration and concurrency tests for Mudra paper portfolio ledger.
Validates PostgreSQL transaction-scoped advisory locking (pg_advisory_xact_lock)
and verifies that concurrent multi-worker paper orders for different tick IDs
are strictly serialized without lost updates, stale balance reads, or duplicate initial deposits.
"""
import os
import threading
import time
from typing import Optional, Dict, Any
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, PaperPortfolio, MarketTick
from src.database.connection import init_db
from src.engine.paper_portfolio import (
    execute_paper_order,
    PORTFOLIO_ADVISORY_LOCK_ID,
    INITIAL_CAPITAL_USD,
    TAKER_FEE_RATE,
    SLIPPAGE_RATE,
)


def get_safe_test_postgres_url() -> Optional[str]:
    """
    Retrieves the isolated test PostgreSQL URL from environment.
    Strictly prevents execution against production Neon instances.
    """
    url = os.getenv("TEST_DATABASE_URL")
    if not url or url.strip() == "":
        return None

    # Safety Guard: Refuse to run against production Neon
    lower_url = url.lower()
    if "neon.tech" in lower_url or "neondb" in lower_url:
        raise RuntimeError(
            "SAFETY VIOLATION: TEST_DATABASE_URL must never point to live Neon database!"
        )

    if not lower_url.startswith("postgres"):
        return None

    return url.strip()


def is_postgres_available() -> bool:
    """Verifies that an isolated PostgreSQL test service is reachable."""
    url = get_safe_test_postgres_url()
    if not url:
        return False
    try:
        eng = create_engine(url, connect_args={"connect_timeout": 3})
        with eng.connect() as conn:
            conn.execute(text("SELECT 1;"))
        eng.dispose()
        return True
    except Exception:
        return False


@pytest.fixture
def isolated_pg_engine():
    """
    Provides a freshly initialized, isolated PostgreSQL test database engine.
    Drops and recreates schema before and after each test.
    """
    url = get_safe_test_postgres_url()
    if not url:
        pytest.skip("Isolated PostgreSQL test instance not available (set TEST_DATABASE_URL)")

    engine = create_engine(url, pool_size=5, max_overflow=5)
    
    # Setup: clean schema
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS paper_portfolio CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS market_ticks CASCADE;"))
        conn.commit()

    init_db(engine)

    try:
        yield engine
    finally:
        # Teardown: cleanup tables even if assertions fail
        try:
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS paper_portfolio CASCADE;"))
                conn.execute(text("DROP TABLE IF EXISTS market_ticks CASCADE;"))
                conn.commit()
        except Exception:
            pass
        engine.dispose()


@pytest.fixture
def isolated_sqlite_db():
    """Provides an isolated in-memory SQLite database session for unit tests."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session, engine
    finally:
        session.close()
        engine.dispose()


def test_sqlite_skips_advisory_lock_without_error(isolated_sqlite_db):
    """
    Verifies that execute_paper_order running on SQLite skips PostgreSQL advisory lock
    acquisition cleanly without raising syntax or operational errors.
    """
    session, _ = isolated_sqlite_db
    order = execute_paper_order(
        session=session,
        action="BUY",
        current_price=60000.0,
        kelly_fraction=0.10,
        tick_id=101,
    )
    session.commit()
    assert order.id is not None
    assert order.action == "BUY"
    assert order.cash_balance < INITIAL_CAPITAL_USD


@pytest.mark.skipif(
    not is_postgres_available(),
    reason="Isolated PostgreSQL instance not available (set TEST_DATABASE_URL)",
)
def test_postgresql_concurrent_multi_tick_paper_orders_serializes(isolated_pg_engine):
    """
    Verifies that two concurrent worker transactions executing paper orders for
    different tick IDs (tick_101 and tick_102) from an initial $10,000 baseline
    are strictly serialized via pg_advisory_xact_lock without stale-snapshot lost updates.
    """
    PgSessionLocal = sessionmaker(bind=isolated_pg_engine, autocommit=False, autoflush=False)

    # 1. Seed two prerequisite MarketTick records
    with PgSessionLocal() as session:
        t1 = MarketTick(
            id=101,
            timestamp="2026-01-01 12:00:00+00",
            open=60000.0, high=60100.0, low=59900.0, close=60000.0, volume=10.0,
            model_version="v1", feature_schema_version="v1"
        )
        t2 = MarketTick(
            id=102,
            timestamp="2026-01-01 13:00:00+00",
            open=60000.0, high=60100.0, low=59900.0, close=60000.0, volume=10.0,
            model_version="v1", feature_schema_version="v1"
        )
        session.add_all([t1, t2])
        session.commit()

    # Synchronization barrier to ensure both threads launch execute_paper_order concurrently
    start_barrier = threading.Barrier(2)
    thread_results: Dict[str, Any] = {}
    thread_errors: Dict[str, Exception] = {}

    def _worker_task(worker_name: str, tick_id: int):
        session = PgSessionLocal()
        try:
            # Wait at barrier so both threads invoke execute_paper_order simultaneously
            start_barrier.wait(timeout=5)

            # Both threads invoke execute_paper_order for different tick IDs
            order = execute_paper_order(
                session=session,
                action="BUY",
                current_price=60000.0,
                kelly_fraction=0.10,
                notes=f"Concurrent order from {worker_name}",
                tick_id=tick_id,
            )
            # Hold transaction briefly to guarantee lock contention
            time.sleep(0.05)
            session.commit()
            thread_results[worker_name] = {
                "id": order.id,
                "cash": order.cash_balance,
                "btc": order.btc_balance,
                "tick_id": order.tick_id,
            }
        except Exception as exc:
            session.rollback()
            thread_errors[worker_name] = exc
        finally:
            session.close()

    t_worker1 = threading.Thread(target=_worker_task, args=("Worker-1", 101))
    t_worker2 = threading.Thread(target=_worker_task, args=("Worker-2", 102))

    t_worker1.start()
    t_worker2.start()

    t_worker1.join(timeout=10)
    t_worker2.join(timeout=10)

    assert not thread_errors, f"Worker execution encountered errors: {thread_errors}"
    assert len(thread_results) == 2, "Both workers must complete successfully"

    # Query all records in paper_portfolio in insertion order
    with PgSessionLocal() as session:
        records = session.execute(
            select(PaperPortfolio).order_by(PaperPortfolio.id.asc())
        ).scalars().all()

        # Invariant 1: Exactly 3 records exist (INITIAL_DEPOSIT, First BUY, Second BUY)
        assert len(records) == 3, f"Expected 3 records, got {len(records)}: {[r.action for r in records]}"
        assert records[0].action == "INITIAL_DEPOSIT"
        assert records[0].cash_balance == INITIAL_CAPITAL_USD

        # Invariant 2: Both distinct tick-linked orders are recorded once
        trade_1 = records[1]
        trade_2 = records[2]
        assert {trade_1.tick_id, trade_2.tick_id} == {101, 102}

        # Invariant 3: Sequential Reference Calculation & Absence of Lost Update
        # Trade 1:
        # capital_to_spend = 10,000 * 0.10 = 1,000.00
        # fee = 1,000 * 0.0005 = 0.50
        # slippage = 1,000 * 0.0002 = 0.20
        # exec_price = 60000 * 1.0002 = 60012.0
        # net_capital = 1,000 - 0.50 = 999.50
        # btc_bought = 999.50 / 60012.0
        # cash_after_t1 = 10,000 - 1,000 = 9,000.00
        expected_cash_1 = INITIAL_CAPITAL_USD - (INITIAL_CAPITAL_USD * 0.10)
        assert abs(trade_1.cash_balance - expected_cash_1) < 1e-4

        # Trade 2: Must be calculated from trade 1's committed balance ($9,000), NOT $10,000!
        # If lost update occurred, trade_2.cash_balance would be 9,000.00 as well!
        assert trade_2.cash_balance < trade_1.cash_balance, (
            f"Lost update detected! Trade 2 cash ({trade_2.cash_balance}) was not decremented "
            f"from Trade 1 committed cash ({trade_1.cash_balance})."
        )

        expected_capital_2 = expected_cash_1 * 0.10  # 900.00
        expected_cash_2 = expected_cash_1 - expected_capital_2  # 8,100.00
        assert abs(trade_2.cash_balance - expected_cash_2) < 1e-4

        # Total BTC balance must equal sum of btc_amounts
        expected_total_btc = trade_1.btc_amount + trade_2.btc_amount
        assert abs(trade_2.btc_balance - expected_total_btc) < 1e-6


@pytest.mark.skipif(
    not is_postgres_available(),
    reason="Isolated PostgreSQL instance not available (set TEST_DATABASE_URL)",
)
def test_postgresql_concurrent_initial_deposit_race(isolated_pg_engine):
    """
    Verifies that when two workers execute on a completely empty portfolio table concurrently,
    pg_advisory_xact_lock prevents duplicate INITIAL_DEPOSIT creation. Exactly one INITIAL_DEPOSIT
    is created, and both trades execute sequentially.
    """
    PgSessionLocal = sessionmaker(bind=isolated_pg_engine, autocommit=False, autoflush=False)

    start_barrier = threading.Barrier(2)
    thread_errors: Dict[str, Exception] = {}

    def _worker_task(worker_name: str, tick_id: int):
        session = PgSessionLocal()
        try:
            start_barrier.wait(timeout=5)
            execute_paper_order(
                session=session,
                action="BUY",
                current_price=60000.0,
                kelly_fraction=0.10,
                tick_id=tick_id,
            )
            time.sleep(0.05)
            session.commit()
        except Exception as exc:
            session.rollback()
            thread_errors[worker_name] = exc
        finally:
            session.close()

    t1 = threading.Thread(target=_worker_task, args=("Worker-A", 201))
    t2 = threading.Thread(target=_worker_task, args=("Worker-B", 202))

    t1.start()
    t2.start()

    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not thread_errors, f"Errors during initial deposit race: {thread_errors}"

    with PgSessionLocal() as session:
        deposits = session.execute(
            select(PaperPortfolio).where(PaperPortfolio.action == "INITIAL_DEPOSIT")
        ).scalars().all()
        assert len(deposits) == 1, f"Expected exactly 1 INITIAL_DEPOSIT, found {len(deposits)}"

        total_orders = session.execute(select(PaperPortfolio)).scalars().all()
        assert len(total_orders) == 3
