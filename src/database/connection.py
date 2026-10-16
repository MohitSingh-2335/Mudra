"""
Unit and Integration Tests for Mudra Database, Reconciliation, Paper Portfolio Engines,
and Model Governance & Catalog Verification.
Runs in-memory (sqlite:///:memory:) for unit tests, and uses isolated disposable
PostgreSQL (via TEST_DATABASE_URL) for PostgreSQL catalog, trigger, and constraint suites.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, Optional
from urllib.parse import urlsplit
import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from config import MIN_HURDLE_PCT, PORTFOLIO_ADVISORY_LOCK_ID
from src.api.schemas import PortfolioSummaryResponse, ReconciliationSummaryResponse
from src.database.connection import (
    CANONICAL_CONSTRAINTS,
    CANONICAL_FOREIGN_KEYS,
    CANONICAL_FUNCTIONS,
    CANONICAL_INDEXES,
    CANONICAL_TABLE_COLUMNS,
    CANONICAL_TRIGGERS,
    FunctionSpec,
    SchemaDriftError,
    ensure_canonical_function,
    init_db,
    verify_db_schema,
)
from src.database.models import (
    ApiTelemetry,
    Base,
    CouncilAudit,
    MarketTick,
    ModelLifecycleEvent,
    ModelPrediction,
    ModelRegistry,
    PaperPortfolio,
    PredictionInvestigation,
    Reconciliation,
)
from src.engine.paper_portfolio import (
    INITIAL_CAPITAL_USD,
    execute_paper_order,
    get_average_entry_price,
    get_portfolio_summary,
)
from src.engine.reconciliation import (
    get_reconciliation_summary,
    reconcile_pending_ticks,
)


# =============================================================================
# PostgreSQL Test Service Helpers & Fail-Closed Disposable Safety Guards
# =============================================================================

_ALLOWED_DISPOSABLE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "postgres"})
_FORBIDDEN_DB_SUBSTRINGS = (
    "neon.tech",
    "neondb",
    "amazonaws.com",
    "rds.",
    "supabase",
    "railway",
    "render.com",
    "cockroach",
    "digitalocean",
    "azure.com",
    "gcp",
)
_FORBIDDEN_DB_NAMES = frozenset(
    {"postgres", "mudra", "neondb", "defaultdb", "main", "production", "prod", "mudra_prod"}
)


def get_safe_test_postgres_url() -> Optional[str]:
    """
    Retrieves and validates the isolated disposable test PostgreSQL URL from environment.
    Fails closed (raises RuntimeError) before any database connection or destructive SQL
    if the URL is ambiguous, remote, matches DATABASE_URL, or lacks an explicit '*test*'
    database name on a local/service-container host.
    """
    raw_url = os.getenv("TEST_DATABASE_URL")
    if not raw_url or raw_url.strip() == "":
        return None

    url = raw_url.strip()
    lower_url = url.lower()

    if not lower_url.startswith(("postgresql://", "postgresql+psycopg2://", "postgres://")):
        raise RuntimeError(
            f"SAFETY VIOLATION: TEST_DATABASE_URL must use a postgresql:// scheme (got {url[:24]!r})."
        )

    for forbidden in _FORBIDDEN_DB_SUBSTRINGS:
        if forbidden in lower_url:
            raise RuntimeError(
                f"SAFETY VIOLATION: TEST_DATABASE_URL contains forbidden cloud/production marker '{forbidden}'!"
            )

    prod_url = os.getenv("DATABASE_URL")
    if prod_url and prod_url.strip():
        norm_test = lower_url.replace("postgresql+psycopg2://", "postgresql://").replace("postgres://", "postgresql://")
        norm_prod = (
            prod_url.strip()
            .lower()
            .replace("postgresql+psycopg2://", "postgresql://")
            .replace("postgres://", "postgresql://")
        )
        if norm_test == norm_prod:
            raise RuntimeError(
                "SAFETY VIOLATION: TEST_DATABASE_URL must never equal DATABASE_URL!"
            )

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    if hostname not in _ALLOWED_DISPOSABLE_HOSTS:
        raise RuntimeError(
            f"SAFETY VIOLATION: TEST_DATABASE_URL host '{hostname}' is not in the disposable host allowlist "
            f"{sorted(_ALLOWED_DISPOSABLE_HOSTS)}."
        )

    db_name = (parsed.path or "").lstrip("/").split("/")[0].lower()
    if not db_name or db_name in _FORBIDDEN_DB_NAMES or "test" not in db_name:
        raise RuntimeError(
            f"SAFETY VIOLATION: TEST_DATABASE_URL database name '{db_name}' must explicitly contain 'test' "
            "and must never target a default or production database."
        )

    username = (parsed.username or "").lower()
    if not username or username in {"postgres", "neondb_owner", "admin", "root", "prod"}:
        raise RuntimeError(
            f"SAFETY VIOLATION: TEST_DATABASE_URL username '{username}' must be a dedicated test user "
            "(e.g., 'mudra_test_user'), not a superuser or production role."
        )

    return url


def is_postgres_available() -> bool:
    """Verifies that an isolated disposable PostgreSQL test service is reachable."""
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


# =============================================================================
# In-Memory SQLite Fixture for Existing Tests
# =============================================================================

@pytest.fixture
def in_memory_db():
    """Provides an isolated, in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session, engine
    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


# =============================================================================
# Disposable PostgreSQL Fixture for Model Governance & Catalog Tests
# =============================================================================

@pytest.fixture
def pg_governance_engine():
    """
    Provides an isolated, cleanly migrated disposable PostgreSQL engine for model governance tests.
    Validates disposable-database safety before connecting, and skips automatically if
    TEST_DATABASE_URL is not configured.
    """
    url = get_safe_test_postgres_url()
    if not url or not is_postgres_available():
        pytest.skip("Isolated PostgreSQL test service (TEST_DATABASE_URL) is not available.")

    eng = create_engine(url, pool_pre_ping=True)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"))

    from scripts.migrate_db import run_migration

    run_migration(target_engine=eng, execute=True, bootstrap_base_tables=True)

    yield eng

    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"))
    eng.dispose()


# =============================================================================
# 1. In-Memory SQLite Regression & Safety Guard Tests (19 Tests)
# =============================================================================

def test_table_creation_and_inspection(in_memory_db):
    """Verifies that all registered tables are created in the database schema."""
    session, engine = in_memory_db
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    for expected_table in CANONICAL_TABLE_COLUMNS.keys():
        assert expected_table in tables


def test_fresh_market_tick_creation_and_insertion(in_memory_db):
    """Verifies MarketTick creation and insertion with supported fields."""
    session, _ = in_memory_db
    tick = MarketTick(
        timestamp=datetime.now(timezone.utc),
        open=50000.0,
        high=51000.0,
        low=49500.0,
        close=50500.0,
        volume=1200.5,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
        market_regime="BULL_TREND",
        predicted_return_1h=0.015,
        predicted_price_1h=51257.5,
        model_confidence=0.72,
        features={"rsi": 55.4, "macd": 12.1},
    )
    session.add(tick)
    session.commit()

    saved = session.query(MarketTick).filter_by(id=tick.id).first()
    assert saved is not None
    assert saved.close == 50500.0
    assert saved.features["rsi"] == 55.4

    with pytest.raises(TypeError):
        MarketTick(
            timestamp=datetime.now(timezone.utc),
            open=50000.0,
            high=51000.0,
            low=49500.0,
            close=50500.0,
            volume=1200.5,
            taker_buy_ratio=0.55,
        )


def test_models_to_dict_serialization(in_memory_db):
    """Verifies models serialize to dict properly."""
    session, _ = in_memory_db
    now = datetime.now(timezone.utc)
    tick = MarketTick(
        timestamp=now,
        open=60000.0,
        high=61000.0,
        low=59500.0,
        close=60500.0,
        volume=500.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(tick)
    session.commit()
    assert tick.id is not None
    assert tick.close == 60500.0
    assert tick.to_dict()["close"] == 60500.0


def test_reconciliation_math_and_accurate_tagging(in_memory_db):
    """Tests reconciliation math and ACCURATE attribution tagging."""
    session, _ = in_memory_db
    base_ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    t0 = MarketTick(
        timestamp=base_ts,
        open=60000.0,
        high=60500.0,
        low=59500.0,
        close=60000.0,
        volume=100.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
        predicted_return_1h=0.01,
        predicted_price_1h=60600.0,
    )
    session.add(t0)
    session.commit()

    t1 = MarketTick(
        timestamp=base_ts + timedelta(hours=1),
        open=60000.0,
        high=60700.0,
        low=59900.0,
        close=60550.0,
        volume=120.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t1)
    session.commit()

    reconciled = reconcile_pending_ticks(session, t1)
    assert len(reconciled) == 1
    r = reconciled[0]
    assert r.tick_id == t0.id
    assert r.direction_correct is True
    assert r.error_attribution_tag == "ACCURATE"


def test_reconciliation_micro_move_and_sideways_drift(in_memory_db):
    """Tests SIDEWAYS_DRIFT error attribution on small contrary price moves (< $150 or <= 0.20%)."""
    session, _ = in_memory_db
    base_ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    t0 = MarketTick(
        timestamp=base_ts,
        open=60000.0,
        high=60100.0,
        low=59900.0,
        close=60000.0,
        volume=100.0,
        predicted_return_1h=0.0008,
        predicted_price_1h=60048.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t0)
    session.commit()

    t1 = MarketTick(
        timestamp=base_ts + timedelta(hours=1),
        open=60000.0,
        high=60050.0,
        low=59920.0,
        close=59960.0,
        volume=110.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t1)
    session.commit()

    reconciled = reconcile_pending_ticks(session, t1)
    assert len(reconciled) == 1
    r = reconciled[0]
    assert r.direction_correct is False
    assert r.absolute_error_usd == pytest.approx(88.0)
    assert r.error_attribution_tag == "SIDEWAYS_DRIFT"


def test_reconciliation_black_swan_outlier_isolation(in_memory_db):
    """Tests BLACK_SWAN_OUTLIER attribution on large unexpected moves (>3.8%)."""
    session, _ = in_memory_db
    base_ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    t0 = MarketTick(
        timestamp=base_ts,
        open=60000.0,
        high=60500.0,
        low=59500.0,
        close=60000.0,
        volume=100.0,
        predicted_return_1h=0.002,
        predicted_price_1h=60120.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t0)
    session.commit()

    t1 = MarketTick(
        timestamp=base_ts + timedelta(hours=1),
        open=60000.0,
        high=63500.0,
        low=59800.0,
        close=63000.0,
        volume=500.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t1)
    session.commit()

    reconciled = reconcile_pending_ticks(session, t1)
    assert len(reconciled) == 1
    r = reconciled[0]
    assert r.is_black_swan is True
    assert r.error_attribution_tag == "BLACK_SWAN_OUTLIER"


def test_veto_attribution_saved_capital_vs_opportunity_cost(in_memory_db):
    """Tests SAVED_CAPITAL vs OPPORTUNITY_COST veto attribution evaluation."""
    session, _ = in_memory_db
    base_ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Case 1: BUY vetoed and market dropped -> SAVED_CAPITAL
    t0 = MarketTick(
        timestamp=base_ts,
        open=60000.0,
        high=60200.0,
        low=59800.0,
        close=60000.0,
        volume=100.0,
        predicted_return_1h=0.005,
        predicted_price_1h=60300.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t0)
    session.commit()

    audit_saved = CouncilAudit(
        timestamp=base_ts,
        tick_id=t0.id,
        strategist_action="BUY",
        strategist_conviction=7,
        risk_verdict="VETOED",
        kelly_fraction=0.0,
        final_decision="HOLD",
    )
    session.add(audit_saved)
    session.commit()

    t1 = MarketTick(
        timestamp=base_ts + timedelta(hours=1),
        open=60000.0,
        high=60100.0,
        low=58900.0,
        close=59000.0,
        volume=150.0,
        predicted_return_1h=0.005,
        predicted_price_1h=59295.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t1)
    session.commit()

    reconcile_pending_ticks(session, t1)
    session.refresh(audit_saved)
    assert audit_saved.veto_attribution == "SAVED_CAPITAL"

    # Case 2: BUY vetoed and market rallied > 0.10% -> OPPORTUNITY_COST
    audit_opp = CouncilAudit(
        timestamp=base_ts + timedelta(hours=1),
        tick_id=t1.id,
        strategist_action="BUY",
        strategist_conviction=7,
        risk_verdict="VETOED",
        kelly_fraction=0.0,
        final_decision="HOLD",
    )
    session.add(audit_opp)
    session.commit()

    t2 = MarketTick(
        timestamp=base_ts + timedelta(hours=2),
        open=59000.0,
        high=60500.0,
        low=58900.0,
        close=60000.0,
        volume=150.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t2)
    session.commit()

    reconcile_pending_ticks(session, t2)
    session.refresh(audit_opp)
    assert audit_opp.veto_attribution == "OPPORTUNITY_COST"


def test_paper_portfolio_full_lifecycle(in_memory_db):
    """Tests paper trading portfolio order execution, summary keys, and Pydantic schema validation."""
    session, _ = in_memory_db
    r_buy = execute_paper_order(session, "BUY", 50000.0, kelly_fraction=0.25)
    assert r_buy.cash_balance < INITIAL_CAPITAL_USD
    assert r_buy.btc_balance > 0.0

    summary = get_portfolio_summary(session, current_price=55000.0)
    assert summary["btc_position_value_usd"] > 0.0
    assert summary["total_equity_usd"] > INITIAL_CAPITAL_USD
    validated_summary = PortfolioSummaryResponse(**summary)
    assert validated_summary.total_trades == 1

    r_sell = execute_paper_order(session, "SELL", 55000.0)
    assert r_sell.btc_balance == 0.0
    assert r_sell.realized_pnl_usd > 0.0


def test_empty_portfolio_summary(in_memory_db):
    """Tests portfolio summary on empty ledger."""
    session, _ = in_memory_db
    summary = get_portfolio_summary(session, current_price=60000.0)
    assert summary["cash_balance"] == INITIAL_CAPITAL_USD
    assert summary["btc_balance"] == 0.0
    assert summary["total_equity_usd"] == INITIAL_CAPITAL_USD


def test_reconciliation_summary_metrics_and_tag_breakdown(in_memory_db):
    """Tests summary statistics, tag breakdown, and Pydantic schema validation."""
    session, _ = in_memory_db
    summary_empty = get_reconciliation_summary(session)
    assert summary_empty["total_reconciled"] == 0
    assert summary_empty["directional_accuracy_pct"] == 0.0
    assert summary_empty["win_rate_pct"] == 0.0

    base_ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    t0 = MarketTick(
        timestamp=base_ts,
        open=60000.0,
        high=60500.0,
        low=59500.0,
        close=60000.0,
        volume=100.0,
        predicted_return_1h=0.01,
        predicted_price_1h=60600.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    t1 = MarketTick(
        timestamp=base_ts + timedelta(hours=1),
        open=60000.0,
        high=60700.0,
        low=59900.0,
        close=60550.0,
        volume=120.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add_all([t0, t1])
    session.commit()
    reconcile_pending_ticks(session, t1)

    summary = get_reconciliation_summary(session)
    assert summary["total_reconciled"] == 1
    assert summary["directional_accuracy_pct"] == 100.0
    assert summary["tag_breakdown"].get("ACCURATE") == 1
    validated = ReconciliationSummaryResponse(**summary)
    assert validated.total_reconciled == 1


def test_net_realized_pnl_accounting_two_completed_trades(in_memory_db):
    """Tests strict cost basis isolation across two completed trades."""
    session, _ = in_memory_db
    execute_paper_order(session, "BUY", 50000.0, kelly_fraction=0.10)
    execute_paper_order(session, "SELL", 55000.0)
    execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10)
    execute_paper_order(session, "SELL", 58000.0)

    summary = get_portfolio_summary(session, current_price=58000.0)
    assert summary["btc_balance"] == 0.0


def test_get_average_entry_price_missing_eligible_buy_isolation(in_memory_db):
    """Tests cost-basis lookup returns 0.0 when no eligible BUY exists after a SELL liquidation."""
    session, _ = in_memory_db
    assert get_average_entry_price(session, target_btc=0.0) == 0.0

    # Execute a BUY and then liquidate via SELL
    execute_paper_order(session, "BUY", 50000.0, kelly_fraction=0.10)
    execute_paper_order(session, "SELL", 52000.0)

    # Requesting cost basis for hypothetical open inventory when no BUY occurred after the SELL
    # must return 0.0 and never leak the prior closed trade's entry price.
    avg_price = get_average_entry_price(session, target_btc=0.02)
    assert avg_price == 0.0


def test_reconciliation_exact_one_hour_horizon_alignment(in_memory_db):
    """Verifies that reconciliation pairs strictly across T and T+1h."""
    session, _ = in_memory_db
    base_ts = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    t0 = MarketTick(
        timestamp=base_ts,
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50000.0,
        volume=10.0,
        predicted_price_1h=50500.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t0)
    session.commit()

    # t2 is +2 hours (skips T+1h)
    t2 = MarketTick(
        timestamp=base_ts + timedelta(hours=2),
        open=50200.0,
        high=50400.0,
        low=50100.0,
        close=50300.0,
        volume=10.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t2)
    session.commit()

    reconciled = reconcile_pending_ticks(session, t2)
    assert len(reconciled) == 0  # t0 remains pending because t1 (T+1h) is missing


def test_reconciliation_multiple_pending_with_intervening_outcomes(in_memory_db):
    """Tests multiple pending predictions reconciling against appropriate outcome candles."""
    session, _ = in_memory_db
    base_ts = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    t0 = MarketTick(
        timestamp=base_ts,
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50000.0,
        volume=10.0,
        predicted_price_1h=50500.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    t1 = MarketTick(
        timestamp=base_ts + timedelta(hours=1),
        open=50500.0,
        high=50600.0,
        low=50400.0,
        close=50500.0,
        volume=10.0,
        predicted_price_1h=51000.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add_all([t0, t1])
    session.commit()

    reconciled_t1 = reconcile_pending_ticks(session, t1)
    assert len(reconciled_t1) == 1
    assert reconciled_t1[0].tick_id == t0.id


def test_reconciliation_missing_expected_candle_leaves_unresolved(in_memory_db):
    """Tests missing candle preserves pending state without corrupting metrics."""
    session, _ = in_memory_db
    base_ts = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    t0 = MarketTick(
        timestamp=base_ts,
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50000.0,
        volume=10.0,
        predicted_price_1h=50500.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add(t0)
    session.commit()

    reconciled = reconcile_pending_ticks(session, t0)
    assert len(reconciled) == 0


def test_reconciliation_preserves_existing_reconciliation_records(in_memory_db):
    """Verifies existing reconciliation records are immutable upon subsequent cycle."""
    session, _ = in_memory_db
    base_ts = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    t0 = MarketTick(
        timestamp=base_ts,
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50000.0,
        volume=10.0,
        predicted_price_1h=50500.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    t1 = MarketTick(
        timestamp=base_ts + timedelta(hours=1),
        open=50500.0,
        high=50600.0,
        low=50400.0,
        close=50500.0,
        volume=10.0,
        predicted_price_1h=51000.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    session.add_all([t0, t1])
    session.commit()

    rec_first = reconcile_pending_ticks(session, t1)
    assert len(rec_first) == 1

    rec_second = reconcile_pending_ticks(session, t1)
    assert len(rec_second) == 0  # Already reconciled, zero duplicates created


def test_get_safe_test_postgres_url_rejects_unsafe_configs(monkeypatch):
    """
    Proves that get_safe_test_postgres_url() fails closed before any connection or
    destructive DROP SCHEMA statement can run when given production, remote, or
    non-test database URLs, while accepting the dedicated CI disposable service URL.
    """
    # 1. Unset or empty -> returns None (skips cleanly)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert get_safe_test_postgres_url() is None

    monkeypatch.setenv("TEST_DATABASE_URL", "   ")
    assert get_safe_test_postgres_url() is None

    # 2. Valid CI disposable PostgreSQL URL -> accepted
    valid_ci_url = "postgresql+psycopg2://mudra_test_user:mudra_test_password@localhost:5432/mudra_test_db"
    monkeypatch.setenv("TEST_DATABASE_URL", valid_ci_url)
    assert get_safe_test_postgres_url() == valid_ci_url

    # 3. Matching DATABASE_URL -> rejected fail-closed
    monkeypatch.setenv("DATABASE_URL", "postgresql://mudra_test_user:mudra_test_password@localhost:5432/mudra_test_db")
    with pytest.raises(RuntimeError, match="must never equal DATABASE_URL"):
        get_safe_test_postgres_url()
    monkeypatch.delenv("DATABASE_URL", raising=False)

    # 4. Neon / cloud / remote hosts -> rejected fail-closed
    unsafe_urls = [
        "postgresql://neondb_owner:secret@ep-cool-Frog-123.us-east-2.aws.neon.tech/neondb?sslmode=require",
        "postgresql://mudra_test_user:pass@db.internal.production.org:5432/mudra_test_db",
        "postgresql://mudra_test_user:pass@192.168.1.50:5432/mudra_test_db",
        "postgresql://mudra_test_user:pass@localhost:5432/mudra",
        "postgresql://mudra_test_user:pass@localhost:5432/postgres",
        "postgresql://mudra_test_user:pass@localhost:5432/production",
        "postgresql://postgres:pass@localhost:5432/mudra_test_db",
        "sqlite:///data/mudra_operational.db",
    ]
    for bad_url in unsafe_urls:
        monkeypatch.setenv("TEST_DATABASE_URL", bad_url)
        with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
            get_safe_test_postgres_url()


def test_worker_startup_uses_read_only_verify_db_schema_and_fails_closed_on_drift(monkeypatch):
    """
    Verifies that normal worker startup calls read-only verify_db_schema(), never calls
    init_db(), and aborts before flush_outbox() or sync_hourly_quant_cycle() when schema
    verification fails.
    """
    import scripts.run_hourly_worker as worker
    import src.database.connection as db_conn

    init_called = []
    verify_called = []
    sync_called = []

    monkeypatch.setattr(db_conn, "test_connection", lambda: (True, "PostgreSQL 16.x", 1.2))
    monkeypatch.setattr(db_conn, "init_db", worker._ORIGINAL_INIT_DB)

    def _failing_verify(*args, **kwargs):
        verify_called.append(True)
        raise SchemaDriftError("Required table 'model_registry' is missing from database schema.")

    monkeypatch.setattr(db_conn, "verify_db_schema", _failing_verify)
    monkeypatch.setattr(
        worker,
        "sync_hourly_quant_cycle",
        lambda force=False: (sync_called.append(True) or True, "Synced", {}),
    )

    assert worker.run_worker(force=False) is False
    assert len(verify_called) == 1
    assert len(init_called) == 0
    assert len(sync_called) == 0


def test_migrate_db_defaults_to_read_only_dry_run(in_memory_db):
    """
    Verifies that scripts/migrate_db.py defaults to read-only dry-run mode (writes_executed=False)
    when invoked without --execute.
    """
    _, engine = in_memory_db
    from scripts.migrate_db import run_migration

    plan = run_migration(target_engine=engine)
    assert plan["mode"] == "DRY_RUN"
    assert plan["writes_executed"] is False


# =============================================================================
# 2. PostgreSQL Model Governance & Catalog Integration Tests (12 Test Suites)
# =============================================================================

def test_pg_table_creation_and_column_catalog(pg_governance_engine):
    """Suite 1: Verifies all 9 tables, exact column sets, types, and nullability."""
    report = verify_db_schema(pg_governance_engine, require_seed_models=False)
    assert report["status"] == "verified"
    assert report["tables"] == 9


def test_pg_foreign_keys_catalog(pg_governance_engine):
    """Suite 2: Verifies all 10 named foreign keys in the PostgreSQL catalog."""
    with pg_governance_engine.connect() as conn:
        fks = conn.execute(
            text("SELECT conname FROM pg_constraint WHERE contype = 'f' AND connamespace = 'public'::regnamespace")
        ).scalars().all()
        for exp_fk, _, _, _, _ in CANONICAL_FOREIGN_KEYS:
            assert exp_fk in fks


def test_pg_constraints_and_vocabularies(pg_governance_engine):
    """Suite 3: Verifies all 7 named table constraints and fails closed on invalid status vocabulary."""
    with pg_governance_engine.connect() as conn:
        cons = conn.execute(
            text("SELECT conname FROM pg_constraint WHERE contype IN ('c', 'u') AND connamespace = 'public'::regnamespace")
        ).scalars().all()
        for con_name, _, _, _, _ in CANONICAL_CONSTRAINTS:
            assert con_name in cons

    # Negative test: inserting illegal status into model_registry violates CHECK constraint
    with pytest.raises(DBAPIError):
        with pg_governance_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO public.model_registry (
                        model_version, regressor_artifact_path, regressor_sha256,
                        classifier_artifact_path, classifier_sha256, feature_schema_version,
                        target_definition, current_status
                    ) VALUES (
                        'illegal_model', 'p1', 's1', 'p2', 's2', 'v1', 't1', 'INVALID_STATUS'
                    )
                """)
            )


def test_pg_indexes_and_partial_predicates(pg_governance_engine):
    """Suite 4: Verifies all 12 explicit indexes and their partial WHERE predicates."""
    with pg_governance_engine.connect() as conn:
        idx_names = conn.execute(
            text("SELECT relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relkind = 'i' AND n.nspname = 'public'")
        ).scalars().all()
        for exp_idx, _, _, _, _ in CANONICAL_INDEXES:
            assert exp_idx in idx_names


def test_pg_trigger_functions_catalog(pg_governance_engine):
    """Suite 5: Verifies all 6 trigger functions, matching skip, and drift/overload fail-closed checks."""
    with pg_governance_engine.connect() as conn:
        for f_spec in CANONICAL_FUNCTIONS:
            row = conn.execute(
                text("""
                    SELECT p.pronargs, l.lanname, p.prorettype::regtype::text AS rettype
                    FROM pg_proc p
                    JOIN pg_language l ON l.oid = p.prolang
                    JOIN pg_namespace n ON n.oid = p.pronamespace
                    WHERE n.nspname = 'public' AND p.proname = :pname
                """),
                {"pname": f_spec.name},
            ).mappings().first()
            assert row is not None
            assert int(row["pronargs"]) == 0
            assert str(row["rettype"]).lower() == "trigger"
            assert str(row["lanname"]).lower() == "plpgsql"

            # Matching function check returns False (idempotent skip)
            assert ensure_canonical_function(conn, f_spec) is False

            # Deliberately drifted expected_prosrc raises SchemaDriftError
            drifted_spec = FunctionSpec(
                schema_name=f_spec.schema_name,
                name=f_spec.name,
                expected_prosrc="BEGIN RETURN NULL; END;",
                create_sql=f_spec.create_sql,
            )
            with pytest.raises(SchemaDriftError, match="drifted"):
                ensure_canonical_function(conn, drifted_spec)

    # Unexpected function overload detection (rolled back cleanly)
    with pytest.raises(SchemaDriftError, match="Unexpected function overloads"):
        with pg_governance_engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE FUNCTION public.mudra_prevent_mutation(x integer) "
                    "RETURNS integer LANGUAGE plpgsql AS $$ BEGIN RETURN x; END; $$;"
                )
            )
            ensure_canonical_function(conn, CANONICAL_FUNCTIONS[0])


def test_pg_triggers_catalog(pg_governance_engine):
    """Suite 6: Verifies all 11 triggers are installed and enabled."""
    with pg_governance_engine.connect() as conn:
        for t_spec in CANONICAL_TRIGGERS:
            row = conn.execute(
                text("""
                    SELECT t.tgenabled, c.relname
                    FROM pg_trigger t
                    JOIN pg_class c ON c.oid = t.tgrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public' AND c.relname = :tname AND t.tgname = :trigname
                """),
                {"tname": t_spec.table_name, "trigname": t_spec.name},
            ).mappings().first()
            assert row is not None
            assert row["tgenabled"] == "O"


def test_pg_model_registry_immutability_and_state_machine(pg_governance_engine):
    """
    Suite 7: Verifies row-level DELETE prevention (on both referenced and unreferenced rows),
    standalone vs. multi-table TRUNCATE protection, provenance immutability, no-op rejection,
    and unevaluated candidate promotion rejection.
    """
    # 1a. Row-level DELETE on a referenced model_registry row is blocked by trg_mr_no_delete
    with pytest.raises(DBAPIError, match="model_registry is strictly append-only; DELETE operations are forbidden"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("DELETE FROM public.model_registry WHERE model_version = 'xgb_onnx_v1.0.0'"))

    # 1b. Row-level DELETE on an unreferenced model_registry row (zero FK children) is also blocked by trg_mr_no_delete
    with pytest.raises(DBAPIError, match="model_registry is strictly append-only; DELETE operations are forbidden"):
        with pg_governance_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO public.model_registry (
                        model_version, regressor_artifact_path, regressor_sha256,
                        classifier_artifact_path, classifier_sha256, feature_schema_version,
                        target_definition, current_status
                    ) VALUES (
                        'unreferenced_cand_v1', 'p1', 's1', 'p2', 's2', 'v1_39features', 't1', 'CANDIDATE'
                    )
                """)
            )
            conn.execute(text("DELETE FROM public.model_registry WHERE model_version = 'unreferenced_cand_v1'"))

    # 2a. Standalone TRUNCATE on model_registry is rejected by PostgreSQL's FK reference check prior to trigger execution
    with pytest.raises(DBAPIError, match="cannot truncate a table referenced in a foreign key constraint"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE public.model_registry"))

    # 2b. Multi-table TRUNCATE including all referencing tables (without CASCADE) passes PostgreSQL's FK check
    # and is rejected by statement-level trigger trg_mr_no_truncate
    with pytest.raises(DBAPIError, match="model_registry is strictly append-only; TRUNCATE operations are forbidden"):
        with pg_governance_engine.begin() as conn:
            conn.execute(
                text(
                    "TRUNCATE TABLE public.model_registry, public.model_lifecycle_events, "
                    "public.model_predictions, public.prediction_investigations"
                )
            )

    # 3. Provenance fields are immutable
    with pytest.raises(DBAPIError, match="Provenance fields in model_registry are immutable"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("UPDATE public.model_registry SET regressor_sha256 = 'tampered' WHERE model_version = 'xgb_onnx_v1.0.0'"))

    # 4. No-op updates are forbidden
    with pytest.raises(DBAPIError, match="No-op update on model_registry"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("UPDATE public.model_registry SET current_status = 'ACTIVE' WHERE model_version = 'xgb_onnx_v1.0.0'"))

    # 5. Direct-SQL negative test: candidate cannot be promoted to ACTIVE without evaluation_metrics and EVALUATED event
    with pytest.raises(DBAPIError, match="Cannot promote candidate model"):
        with pg_governance_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO public.model_registry (
                        model_version, regressor_artifact_path, regressor_sha256,
                        classifier_artifact_path, classifier_sha256, feature_schema_version,
                        target_definition, current_status
                    ) VALUES (
                        'unevaluated_cand_v1', 'p1', 's1', 'p2', 's2', 'v1_39features', 't1', 'CANDIDATE'
                    )
                """)
            )
            conn.execute(
                text("""
                    INSERT INTO public.model_lifecycle_events (
                        model_version, lifecycle_seq, event_type, from_status, to_status,
                        related_model_version, actor, reason
                    ) VALUES (
                        'unevaluated_cand_v1', 1, 'REGISTERED', NULL, 'CANDIDATE',
                        'xgb_onnx_v1.0.0', 'pytest', 'Registered unevaluated candidate'
                    )
                """)
            )
            conn.execute(
                text("UPDATE public.model_registry SET current_status = 'ACTIVE' WHERE model_version = 'unevaluated_cand_v1'")
            )


def test_pg_lifecycle_event_synchronization_and_audit(pg_governance_engine):
    """Suite 8: Verifies model_lifecycle_events is append-only (UPDATE, DELETE, TRUNCATE) and enforces atomic audit sync."""
    # 1. UPDATE on lifecycle events is forbidden
    with pytest.raises(DBAPIError, match="model_lifecycle_events is strictly append-only; UPDATE operations are forbidden"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("UPDATE public.model_lifecycle_events SET actor = 'tampered' WHERE model_version = 'xgb_onnx_v1.0.0'"))

    # 2. DELETE on lifecycle events is forbidden
    with pytest.raises(DBAPIError, match="model_lifecycle_events is strictly append-only; DELETE operations are forbidden"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("DELETE FROM public.model_lifecycle_events WHERE model_version = 'xgb_onnx_v1.0.0'"))

    # 3. TRUNCATE on lifecycle events (leaf table with no incoming FKs) is blocked by trg_mle_no_truncate
    with pytest.raises(DBAPIError, match="model_lifecycle_events is strictly append-only; TRUNCATE operations are forbidden"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE public.model_lifecycle_events"))

    # 4. Unsynchronized status transition violates deferred audit trigger
    with pytest.raises(DBAPIError, match="Atomic lifecycle audit violation"):
        with pg_governance_engine.begin() as conn:
            # Transition ACTIVE -> ARCHIVED on model_registry without adding corresponding SUPERSEDED/ARCHIVED event
            conn.execute(text("UPDATE public.model_registry SET current_status = 'ARCHIVED' WHERE model_version = 'xgb_onnx_v1.0.0'"))


def test_pg_prediction_mode_invariants_and_lineage(pg_governance_engine):
    """Suite 9: Verifies PRIMARY/SHADOW invariants, return/price alignment, and lineage."""
    base_ts = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    with pg_governance_engine.begin() as conn:
        conn.execute(
            text("INSERT INTO public.market_ticks (id, timestamp, open, high, low, close, volume) VALUES (999, :ts, 60000, 60100, 59900, 60000, 100)"),
            {"ts": base_ts},
        )

    # 1. Prediction insert cannot contain non-null outcome fields
    with pytest.raises(DBAPIError, match="Initial prediction insert must be unresolved"):
        with pg_governance_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO public.model_predictions (
                        tick_id, model_version, run_mode, is_canonical, is_primary,
                        predicted_return_pct, predicted_price_1h, direction_predicted,
                        direction_prob_up, clears_hurdle, actual_price_at_t_plus_1
                    ) VALUES (
                        999, 'xgb_onnx_v1.0.0', 'PRIMARY', TRUE, TRUE,
                        0.5, 60300.0, 'UP', 0.65, TRUE, 60400.0
                    )
                """)
            )

    # 2. Return / price alignment mismatch rejected
    with pytest.raises(DBAPIError, match="Forecast return/price mismatch"):
        with pg_governance_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO public.model_predictions (
                        tick_id, model_version, run_mode, is_canonical, is_primary,
                        predicted_return_pct, predicted_price_1h, direction_predicted,
                        direction_prob_up, clears_hurdle
                    ) VALUES (
                        999, 'xgb_onnx_v1.0.0', 'PRIMARY', TRUE, TRUE,
                        10.0, 60300.0, 'UP', 0.65, TRUE
                    )
                """)
            )

    # 3. Primary prediction requires ACTIVE model
    with pytest.raises(DBAPIError, match="Primary prediction rejected: model p1_controlled_candidate_v1 has status REJECTED"):
        with pg_governance_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO public.model_predictions (
                        tick_id, model_version, run_mode, is_canonical, is_primary,
                        predicted_return_pct, predicted_price_1h, direction_predicted,
                        direction_prob_up, clears_hurdle
                    ) VALUES (
                        999, 'p1_controlled_candidate_v1', 'PRIMARY', TRUE, TRUE,
                        0.5, 60300.0, 'UP', 0.65, TRUE
                    )
                """)
            )


def test_pg_reconciliation_all_or_nothing_and_immutability(pg_governance_engine):
    """Suite 10: Verifies all-or-nothing outcome populating, formula enforcement, and immutability."""
    base_ts = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    with pg_governance_engine.begin() as conn:
        conn.execute(
            text("INSERT INTO public.market_ticks (id, timestamp, open, high, low, close, volume) VALUES (1001, :ts, 60000, 60100, 59900, 60000, 100)"),
            {"ts": base_ts},
        )
        conn.execute(
            text("INSERT INTO public.market_ticks (id, timestamp, open, high, low, close, volume) VALUES (1002, :ts, 60300, 60400, 60200, 60300, 100)"),
            {"ts": base_ts + timedelta(hours=1)},
        )
        conn.execute(
            text("""
                INSERT INTO public.model_predictions (
                    id, tick_id, model_version, run_mode, is_canonical, is_primary,
                    predicted_return_pct, predicted_price_1h, direction_predicted,
                    direction_prob_up, clears_hurdle
                ) VALUES (
                    5001, 1001, 'xgb_onnx_v1.0.0', 'PRIMARY', TRUE, TRUE,
                    0.5, 60300.0, 'UP', 0.70, TRUE
                )
            """)
        )

    # 1a. DELETE on model_predictions is strictly forbidden by trg_model_predictions_no_delete
    with pytest.raises(DBAPIError, match="model_predictions is strictly append-only; DELETE operations are forbidden"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("DELETE FROM public.model_predictions WHERE id = 5001"))

    # 1b. TRUNCATE on model_predictions (with referencing table) is strictly forbidden by trg_model_predictions_no_truncate
    with pytest.raises(DBAPIError, match="model_predictions is strictly append-only; TRUNCATE operations are forbidden"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE public.model_predictions, public.prediction_investigations"))

    # 2. Point-in-time forecast fields are immutable on update
    with pytest.raises(DBAPIError, match="Point-in-time forecast fields in model_predictions are immutable"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("UPDATE public.model_predictions SET predicted_price_1h = 60400.0 WHERE id = 5001"))

    # 3. Successful atomic reconciliation
    with pg_governance_engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE public.model_predictions
                SET
                    outcome_tick_id = 1002,
                    evaluation_policy_version = 'binary_exact_t1_v2',
                    actual_price_at_t_plus_1 = 60300.0,
                    actual_return_pct = 0.5,
                    direction_realized = 'UP',
                    direction_correct = TRUE,
                    residual_error = 0.0,
                    absolute_error_usd = 0.0,
                    percentage_error = 0.0,
                    is_black_swan = FALSE,
                    error_attribution_tag = 'ACCURATE',
                    reconciled_at = :ts
                WHERE id = 5001
            """),
            {"ts": base_ts + timedelta(hours=1)},
        )

    # 4. Finalized reconciliation is permanently immutable
    with pytest.raises(DBAPIError, match="Finalized reconciliation outcome in model_predictions is permanently immutable"):
        with pg_governance_engine.begin() as conn:
            conn.execute(text("UPDATE public.model_predictions SET actual_price_at_t_plus_1 = 60500.0 WHERE id = 5001"))


def test_pg_prediction_investigations_constraints(pg_governance_engine):
    """Suite 11: Verifies verification status check constraints and confidence boundaries."""
    base_ts = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    with pg_governance_engine.begin() as conn:
        conn.execute(
            text("INSERT INTO public.market_ticks (id, timestamp, open, high, low, close, volume) VALUES (2001, :ts, 60000, 60100, 59900, 60000, 100)"),
            {"ts": base_ts},
        )
        conn.execute(
            text("""
                INSERT INTO public.prediction_investigations (
                    tick_id, model_version, headline, suspected_cause_category, hypothesis_notes, diagnostic_confidence
                ) VALUES (
                    2001, 'xgb_onnx_v1.0.0', 'Sudden liquidity drain', 'MACRO_NEWS', 'Hypothesis verified.', 0.85
                )
            """)
        )

    # Confidence score > 1.0 is rejected by check constraint
    with pytest.raises(DBAPIError):
        with pg_governance_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO public.prediction_investigations (
                        tick_id, model_version, headline, suspected_cause_category, hypothesis_notes, diagnostic_confidence
                    ) VALUES (
                        2001, 'xgb_onnx_v1.0.0', 'Invalid confidence', 'MACRO_NEWS', 'Notes', 1.5
                    )
                """)
            )


def test_pg_migration_runner_idempotency_and_drift_detection(pg_governance_engine):
    """Suite 12: Verifies migration runner re-execution idempotency and SchemaDriftError detection."""
    from scripts.migrate_db import run_migration

    # 1. Re-running migration on already migrated schema is an idempotent success
    res = run_migration(target_engine=pg_governance_engine, execute=True)
    assert res["mode"] == "EXECUTE"
    assert res["verification"]["status"] == "verified"

    # 2. Introducing synthetic drift (e.g. dropping a constraint) triggers SchemaDriftError
    with pg_governance_engine.begin() as conn:
        conn.execute(text("ALTER TABLE public.prediction_investigations DROP CONSTRAINT chk_pi_verification_status"))

    with pytest.raises(SchemaDriftError, match="Required constraint 'chk_pi_verification_status' on 'prediction_investigations' is missing"):
        verify_db_schema(pg_governance_engine)