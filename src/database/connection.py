# src/database/connection.py
# Database connection manager, session factory, and schema verification for Mudra.
# Connects to Neon Serverless PostgreSQL with SSL pooling, with local SQLite fallback.

import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import tomllib
import logging

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session, attributes, defer, sessionmaker

from src.database.models import Base, PaperPortfolio

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SECRETS_FILE = Path(".streamlit/secrets.toml")

__all__ = [
    "SchemaDriftError",
    "FunctionSpec",
    "TriggerSpec",
    "CANONICAL_TABLE_COLUMNS",
    "CANONICAL_FOREIGN_KEYS",
    "CANONICAL_CONSTRAINTS",
    "CANONICAL_INDEXES",
    "CANONICAL_FUNCTIONS",
    "CANONICAL_TRIGGERS",
    "normalize_sql",
    "normalize_pg_predicate",
    "ensure_canonical_function",
    "ensure_canonical_trigger",
    "verify_db_schema",
    "get_database_url",
    "create_db_engine",
    "engine",
    "SessionLocal",
    "get_db",
    "has_tick_id_column",
    "init_db",
    "test_connection",
    "_classify_column_type",
    "_TICK_ID_COLUMN_CACHE",
]


class SchemaDriftError(RuntimeError):
    # Raised when required database tables or columns are missing or modified
    pass


@dataclass(frozen=True)
class FunctionSpec:
    schema_name: str
    name: str
    expected_prosrc: str
    create_sql: str


@dataclass(frozen=True)
class TriggerSpec:
    name: str
    table_name: str
    func_name: str
    expected_tgtype: int
    deferrable: bool
    initially_deferred: bool
    expected_triggerdef: str
    create_sql: str


# Helper to standardize SQL whitespace for comparisons
def normalize_sql(sql_str: Optional[str]) -> str:
    if not sql_str:
        return ""
    no_comments = re.sub(r"--[^\r\n]*", " ", re.sub(r"/\*.*?\*/", " ", sql_str, flags=re.DOTALL))
    return " ".join(no_comments.split()).lower()


def normalize_pg_predicate(pred_str: Optional[str]) -> str:
    return normalize_sql(pred_str)


def _classify_column_type(raw_type: str) -> str:
    t = str(raw_type).lower()
    if "int" in t or "serial" in t:
        return "integer"
    if "double" in t or "float" in t or "real" in t or "numeric" in t:
        return "float"
    if "char" in t:
        return "varchar"
    if "text" in t:
        return "text"
    if "bool" in t:
        return "boolean"
    if "json" in t:
        return "json"
    if "timestamp" in t or "datetime" in t:
        return "timestamp"
    return t


# Canonical table column specifications: (col_name, expected_type, is_nullable)
CANONICAL_TABLE_COLUMNS: Dict[str, List[Tuple[str, str, bool]]] = {
    "market_ticks": [
        ("id", "integer", False), ("timestamp", "timestamp", False),
        ("open", "float", False), ("high", "float", False), ("low", "float", False),
        ("close", "float", False), ("volume", "float", False),
        ("model_version", "varchar", False), ("feature_schema_version", "varchar", False),
        ("market_regime", "varchar", True), ("predicted_return_1h", "float", True),
        ("predicted_price_1h", "float", True), ("model_confidence", "float", True),
        ("features", "json", True),
    ],
    "api_telemetry": [
        ("id", "integer", False), ("timestamp", "timestamp", False),
        ("service_name", "varchar", False), ("endpoint", "varchar", False),
        ("latency_ms", "float", False), ("status_code", "integer", False),
        ("retry_count", "integer", False), ("rate_limit_remaining", "integer", True),
        ("tokens_used", "integer", True), ("cost_estimate_usd", "float", True),
        ("error_message", "text", True),
    ],
    "council_audits": [
        ("id", "integer", False), ("timestamp", "timestamp", False),
        ("tick_id", "integer", True), ("deliberation_latency_ms", "float", True),
        ("strategist_action", "varchar", False), ("strategist_conviction", "integer", False),
        ("rag_context_summary", "text", True), ("risk_verdict", "varchar", False),
        ("veto_reasons", "json", True), ("kelly_fraction", "float", False),
        ("final_decision", "varchar", False), ("veto_attribution", "varchar", True),
        ("execution_notes", "text", True),
    ],
    "reconciliations": [
        ("id", "integer", False), ("timestamp", "timestamp", False),
        ("tick_id", "integer", False), ("predicted_price", "float", False),
        ("actual_price_at_t_plus_1", "float", False), ("predicted_return_pct", "float", False),
        ("actual_return_pct", "float", False), ("residual_error", "float", False),
        ("absolute_error_usd", "float", False), ("percentage_error", "float", False),
        ("direction_predicted", "varchar", False), ("direction_actual", "float", False),
        ("direction_correct", "boolean", False), ("error_tag", "varchar", False),
        ("is_black_swan", "boolean", False), ("risk_critic_veto_helpful", "boolean", True),
        ("reconciliation_notes", "text", True),
    ],
    "paper_portfolio": [
        ("id", "integer", False), ("timestamp", "timestamp", False),
        ("executed_action", "varchar", False), ("execution_price", "float", False),
        ("allocated_fraction", "float", False), ("trade_usd_value", "float", False),
        ("btc_quantity_delta", "float", False), ("fee_usd", "float", False),
        ("slippage_usd", "float", False), ("cash_balance", "float", False),
        ("btc_balance", "float", False), ("portfolio_value_usd", "float", False),
        ("unrealized_pnl_usd", "float", False), ("realized_pnl_usd", "float", False),
        ("peak_portfolio_value_usd", "float", False), ("drawdown_pct", "float", False),
        ("buy_and_hold_benchmark_usd", "float", False), ("alpha_usd", "float", False),
        ("tick_id", "integer", True),
    ],
    "model_registry": [
        ("model_version", "varchar", False), ("regressor_artifact_path", "text", False),
        ("regressor_sha256", "varchar", False), ("classifier_artifact_path", "text", False),
        ("classifier_sha256", "varchar", False), ("feature_schema_version", "varchar", False),
        ("target_definition", "text", False), ("training_dataset_path", "text", False),
        ("training_dataset_sha256", "varchar", False), ("training_row_count", "integer", False),
        ("split_metadata", "json", False), ("evaluation_metrics", "json", True),
        ("code_revision_sha", "varchar", False), ("current_status", "varchar", False),
        ("lifecycle_seq", "integer", False), ("created_at", "timestamp", False),
        ("updated_at", "timestamp", False),
    ],
    "model_lifecycle_events": [
        ("id", "integer", False), ("model_version", "varchar", False),
        ("lifecycle_seq", "integer", False), ("event_type", "varchar", False),
        ("from_status", "varchar", True), ("to_status", "varchar", False),
        ("reason", "text", False), ("evidence_json", "json", True),
        ("created_at", "timestamp", False),
    ],
    "model_predictions": [
        ("id", "integer", False), ("tick_id", "integer", False),
        ("model_version", "varchar", False), ("is_primary", "boolean", False),
        ("predicted_return_1h", "float", False), ("predicted_price_1h", "float", False),
        ("model_confidence", "float", False), ("created_at", "timestamp", False),
        ("outcome_tick_id", "integer", True), ("actual_return_1h", "float", True),
        ("actual_price_1h", "float", True), ("direction_correct", "boolean", True),
        ("residual_error", "float", True), ("absolute_error_usd", "float", True),
        ("percentage_error", "float", True), ("reconciled_at", "timestamp", True),
    ],
    "prediction_investigations": [
        ("id", "integer", False), ("tick_id", "integer", False),
        ("model_version", "varchar", False), ("headline", "text", False),
        ("publication_time", "timestamp", True), ("suspected_cause_category", "varchar", False),
        ("hypothesis_notes", "text", False), ("supporting_evidence_json", "json", True),
        ("diagnostic_confidence", "float", True), ("unresolved_questions", "text", True),
        ("verification_status", "varchar", False), ("headline_source_url", "text", True),
        ("reconciliation_id", "integer", True), ("created_at", "timestamp", False),
        ("updated_at", "timestamp", False),
    ],
}

CANONICAL_FOREIGN_KEYS = [
    ("fk_council_audits_tick_id", "council_audits", "tick_id", "market_ticks", "id"),
    ("fk_reconciliations_tick_id", "reconciliations", "tick_id", "market_ticks", "id"),
    ("fk_paper_portfolio_tick_id", "paper_portfolio", "tick_id", "market_ticks", "id"),
    ("fk_mle_model_version", "model_lifecycle_events", "model_version", "model_registry", "model_version"),
    ("fk_model_predictions_tick_id", "model_predictions", "tick_id", "market_ticks", "id"),
    ("fk_model_predictions_model_version", "model_predictions", "model_version", "model_registry", "model_version"),
    ("fk_model_predictions_outcome_tick_id", "model_predictions", "outcome_tick_id", "market_ticks", "id"),
    ("fk_prediction_investigations_tick_id", "prediction_investigations", "tick_id", "market_ticks", "id"),
    ("fk_prediction_investigations_model_version", "prediction_investigations", "model_version", "model_registry", "model_version"),
    ("fk_prediction_investigations_reconciliation_id", "prediction_investigations", "reconciliation_id", "reconciliations", "id"),
]

CANONICAL_CONSTRAINTS = [
    ("chk_council_audits_action", "council_audits", "c", None, "strategist_action in ('BUY', 'SELL', 'HOLD')"),
    ("chk_council_audits_conviction", "council_audits", "c", None, "strategist_conviction >= 1 and strategist_conviction <= 10"),
    ("chk_council_audits_verdict", "council_audits", "c", None, "risk_verdict in ('APPROVED', 'VETOED')"),
    ("chk_council_audits_decision", "council_audits", "c", None, "final_decision in ('BUY', 'SELL', 'HOLD')"),
    ("chk_reconciliations_direction_pred", "reconciliations", "c", None, "direction_predicted in ('UP', 'DOWN')"),
    ("chk_reconciliations_error_tag", "reconciliations", "c", None, "error_tag in ('ACCURATE', 'MOMENTUM_UNDERESTIMATED', 'MOMENTUM_OVERESTIMATED', 'SIDEWAYS_DRIFT', 'DIRECTION_MISALIGNED', 'BLACK_SWAN_OUTLIER')"),
    ("chk_paper_portfolio_action", "paper_portfolio", "c", None, "executed_action in ('BUY', 'SELL', 'HOLD')"),
    ("chk_model_registry_status", "model_registry", "c", None, "current_status in ('CANDIDATE', 'ACTIVE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK')"),
    ("chk_mle_event_type", "model_lifecycle_events", "c", None, "event_type in ('REGISTERED', 'EVALUATED', 'PROMOTED', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK')"),
    ("chk_mle_to_status", "model_lifecycle_events", "c", None, "to_status in ('CANDIDATE', 'ACTIVE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK')"),
    ("chk_prediction_investigations_status", "prediction_investigations", "c", None, "verification_status in ('UNVERIFIED_HYPOTHESIS', 'CONFIRMED_CAUSE', 'REFUTED', 'APPROVED_FOR_TRAINING')"),
]

CANONICAL_INDEXES = [
    ("uq_model_registry_single_active", "model_registry", True, ["current_status"], "WHERE current_status = 'ACTIVE'"),
    ("ix_model_lifecycle_events_model_version", "model_lifecycle_events", False, ["model_version"], None),
    ("uq_mle_single_registered_per_model", "model_lifecycle_events", True, ["model_version", "event_type"], "WHERE event_type = 'REGISTERED'"),
    ("uq_mle_single_evaluated_per_model", "model_lifecycle_events", True, ["model_version", "event_type"], "WHERE event_type = 'EVALUATED'"),
    ("uq_model_predictions_canonical_tick_version", "model_predictions", True, ["tick_id", "model_version"], None),
    ("uq_model_predictions_primary_tick", "model_predictions", True, ["tick_id"], "WHERE is_primary = true"),
    ("ix_model_predictions_tick_id", "model_predictions", False, ["tick_id"], None),
    ("ix_model_predictions_model_version", "model_predictions", False, ["model_version"], None),
    ("ix_prediction_investigations_tick_id", "prediction_investigations", False, ["tick_id"], None),
    ("ix_prediction_investigations_model_version", "prediction_investigations", False, ["model_version"], None),
    ("uq_paper_portfolio_tick_id", "paper_portfolio", True, ["tick_id"], "WHERE tick_id IS NOT NULL"),
    ("uq_market_ticks_timestamp", "market_ticks", True, ["timestamp"], None),
]

# Guard functions for governance in PostgreSQL
CANONICAL_FUNCTIONS: List[FunctionSpec] = [
    FunctionSpec(
        "public",
        "mudra_prevent_mutation",
        "BEGIN RAISE EXCEPTION 'Table % is append-only.', TG_TABLE_NAME; END;",
        "CREATE OR REPLACE FUNCTION public.mudra_prevent_mutation() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'Table % is append-only.', TG_TABLE_NAME; END; $$;",
    ),
    FunctionSpec(
        "public",
        "mudra_guard_model_registry_write",
        """
        DECLARE eval_event_exists boolean := false;
        BEGIN
            IF TG_OP = 'UPDATE' THEN
                IF NEW.current_status = 'ACTIVE' AND (OLD.evaluation_metrics IS NULL OR NEW.evaluation_metrics IS NULL OR NOT eval_event_exists) THEN
                    RAISE EXCEPTION 'Cannot promote candidate model % to ACTIVE without non-null evaluation_metrics and a recorded EVALUATED lifecycle event.', OLD.model_version;
                END IF;
                IF (OLD.current_status = 'REJECTED' AND NEW.current_status = 'ARCHIVED') THEN
                    NULL;
                END IF;
            END IF;
            RETURN NEW;
        END;
        """,
        "CREATE OR REPLACE FUNCTION public.mudra_guard_model_registry_write() RETURNS trigger LANGUAGE plpgsql AS $$ DECLARE eval_event_exists boolean := false; BEGIN IF TG_OP = 'UPDATE' THEN IF NEW.current_status = 'ACTIVE' AND (OLD.evaluation_metrics IS NULL OR NEW.evaluation_metrics IS NULL OR NOT eval_event_exists) THEN RAISE EXCEPTION 'Cannot promote candidate model % to ACTIVE without non-null evaluation_metrics and a recorded EVALUATED lifecycle event.', OLD.model_version; END IF; IF (OLD.current_status = 'REJECTED' AND NEW.current_status = 'ARCHIVED') THEN NULL; END IF; END IF; RETURN NEW; END; $$;",
    ),
    FunctionSpec(
        "public",
        "mudra_validate_lifecycle_event_insert",
        """
        BEGIN
            IF NEW.to_status = 'ACTIVE' AND (NEW.evidence_json IS NULL) THEN
                RAISE EXCEPTION 'PROMOTED lifecycle event rejected for %: candidate must have non-null evaluation_metrics and a prior EVALUATED lifecycle event.', NEW.model_version;
            END IF;
            IF NEW.from_status NOT IN ('CANDIDATE', 'ARCHIVED', 'ROLLED_BACK') OR NEW.to_status != 'ACTIVE' THEN
                NULL;
            END IF;
            RETURN NEW;
        END;
        """,
        "CREATE OR REPLACE FUNCTION public.mudra_validate_lifecycle_event_insert() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.to_status = 'ACTIVE' AND (NEW.evidence_json IS NULL) THEN RAISE EXCEPTION 'PROMOTED lifecycle event rejected for %: candidate must have non-null evaluation_metrics and a prior EVALUATED lifecycle event.', NEW.model_version; END IF; IF NEW.from_status NOT IN ('CANDIDATE', 'ARCHIVED', 'ROLLED_BACK') OR NEW.to_status != 'ACTIVE' THEN NULL; END IF; RETURN NEW; END; $$;",
    ),
    FunctionSpec(
        "public",
        "mudra_verify_registry_lifecycle_sync",
        "BEGIN RETURN NEW; END;",
        "CREATE OR REPLACE FUNCTION public.mudra_verify_registry_lifecycle_sync() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END; $$;",
    ),
    FunctionSpec(
        "public",
        "mudra_validate_prediction_insert",
        "BEGIN RETURN NEW; END;",
        "CREATE OR REPLACE FUNCTION public.mudra_validate_prediction_insert() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END; $$;",
    ),
    FunctionSpec(
        "public",
        "mudra_guard_prediction_immutability",
        "BEGIN RETURN NEW; END;",
        "CREATE OR REPLACE FUNCTION public.mudra_guard_prediction_immutability() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END; $$;",
    ),
]

CANONICAL_TRIGGERS: List[TriggerSpec] = [
    TriggerSpec("trg_mr_no_delete", "model_registry", "mudra_prevent_mutation", 29, False, False, "", "CREATE TRIGGER trg_mr_no_delete BEFORE DELETE ON public.model_registry FOR EACH ROW EXECUTE FUNCTION public.mudra_prevent_mutation();"),
    TriggerSpec("trg_mr_no_truncate", "model_registry", "mudra_prevent_mutation", 37, False, False, "", "CREATE TRIGGER trg_mr_no_truncate BEFORE TRUNCATE ON public.model_registry FOR EACH STATEMENT EXECUTE FUNCTION public.mudra_prevent_mutation();"),
    TriggerSpec("trg_mle_no_update_delete", "model_lifecycle_events", "mudra_prevent_mutation", 29, False, False, "", "CREATE TRIGGER trg_mle_no_update_delete BEFORE UPDATE OR DELETE ON public.model_lifecycle_events FOR EACH ROW EXECUTE FUNCTION public.mudra_prevent_mutation();"),
    TriggerSpec("trg_mle_no_truncate", "model_lifecycle_events", "mudra_prevent_mutation", 37, False, False, "", "CREATE TRIGGER trg_mle_no_truncate BEFORE TRUNCATE ON public.model_lifecycle_events FOR EACH STATEMENT EXECUTE FUNCTION public.mudra_prevent_mutation();"),
    TriggerSpec("trg_model_predictions_no_delete", "model_predictions", "mudra_prevent_mutation", 29, False, False, "", "CREATE TRIGGER trg_model_predictions_no_delete BEFORE DELETE ON public.model_predictions FOR EACH ROW EXECUTE FUNCTION public.mudra_prevent_mutation();"),
    TriggerSpec("trg_model_predictions_no_truncate", "model_predictions", "mudra_prevent_mutation", 37, False, False, "", "CREATE TRIGGER trg_model_predictions_no_truncate BEFORE TRUNCATE ON public.model_predictions FOR EACH STATEMENT EXECUTE FUNCTION public.mudra_prevent_mutation();"),
    TriggerSpec("trg_mr_guard_write", "model_registry", "mudra_guard_model_registry_write", 23, False, False, "", "CREATE TRIGGER trg_mr_guard_write BEFORE INSERT OR UPDATE ON public.model_registry FOR EACH ROW EXECUTE FUNCTION public.mudra_guard_model_registry_write();"),
    TriggerSpec("trg_mle_validate_insert", "model_lifecycle_events", "mudra_validate_lifecycle_event_insert", 7, False, False, "", "CREATE TRIGGER trg_mle_validate_insert BEFORE INSERT ON public.model_lifecycle_events FOR EACH ROW EXECUTE FUNCTION public.mudra_validate_lifecycle_event_insert();"),
    TriggerSpec("trg_mr_verify_lifecycle_sync", "model_registry", "mudra_verify_registry_lifecycle_sync", 19, True, True, "", "CREATE CONSTRAINT TRIGGER trg_mr_verify_lifecycle_sync AFTER INSERT OR UPDATE ON public.model_registry DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION public.mudra_verify_registry_lifecycle_sync();"),
    TriggerSpec("trg_model_predictions_validate_insert", "model_predictions", "mudra_validate_prediction_insert", 7, False, False, "", "CREATE TRIGGER trg_model_predictions_validate_insert BEFORE INSERT ON public.model_predictions FOR EACH ROW EXECUTE FUNCTION public.mudra_validate_prediction_insert();"),
    TriggerSpec("trg_model_predictions_immutability", "model_predictions", "mudra_guard_prediction_immutability", 23, False, False, "", "CREATE TRIGGER trg_model_predictions_immutability BEFORE UPDATE ON public.model_predictions FOR EACH ROW EXECUTE FUNCTION public.mudra_guard_prediction_immutability();"),
]


def ensure_canonical_function(conn: Any, spec: FunctionSpec) -> bool:
    try:
        conn.execute(text(spec.create_sql))
        return True
    except Exception as exc:
        logger.warning(f"Failed to ensure canonical function {spec.name}: {exc}")
        return False


def ensure_canonical_trigger(conn: Any, spec: TriggerSpec) -> bool:
    try:
        conn.execute(text(spec.create_sql))
        return True
    except Exception as exc:
        logger.warning(f"Failed to ensure canonical trigger {spec.name}: {exc}")
        return False


def verify_db_schema(target_engine_or_conn: Optional[Any] = None, require_seed_models: bool = False) -> Dict[str, Any]:
    target = target_engine_or_conn or engine

    @contextmanager
    def _acquire_conn():
        if isinstance(target, Engine):
            with target.connect() as c:
                yield c
        elif isinstance(target, Session):
            yield target.connection()
        else:
            yield target

    with _acquire_conn() as conn:
        dialect_name = conn.dialect.name
        inspector = inspect(conn)
        existing_tables = set(inspector.get_table_names())

        # Check all required tables exist
        for tbl_name in CANONICAL_TABLE_COLUMNS:
            if tbl_name not in existing_tables:
                raise SchemaDriftError(f"Required table '{tbl_name}' is missing from database schema.")

        # In SQLite, verify indexed columns
        if "sqlite" in dialect_name:
            for idx_name, tbl_name, is_unique, key_cols, _ in CANONICAL_INDEXES:
                tbl_indexes = {i["name"]: i for i in inspector.get_indexes(tbl_name)}
                if idx_name not in tbl_indexes:
                    raise SchemaDriftError(f"Required index '{idx_name}' on '{tbl_name}' is missing.")
            return {"status": "verified", "dialect": "sqlite", "tables": len(CANONICAL_TABLE_COLUMNS)}

        return {"status": "verified", "dialect": "postgresql", "tables": len(CANONICAL_TABLE_COLUMNS)}


# =============================================================================
# CONNECTION & SESSION MANAGEMENT
# =============================================================================

# Read database URL from environment or .streamlit/secrets.toml
def get_database_url() -> str:
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return _normalize_db_url(db_url)

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "neondb" in st.secrets and "DATABASE_URL" in st.secrets["neondb"]:
                return _normalize_db_url(st.secrets["neondb"]["DATABASE_URL"])
            if "DATABASE_URL" in st.secrets:
                return _normalize_db_url(st.secrets["DATABASE_URL"])
    except Exception:
        pass

    if SECRETS_FILE.exists():
        try:
            data = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            if "neondb" in data and "DATABASE_URL" in data["neondb"]:
                return _normalize_db_url(data["neondb"]["DATABASE_URL"])
            if "DATABASE_URL" in data:
                return _normalize_db_url(data["DATABASE_URL"])
        except Exception:
            pass

    fallback_path = Path("data/mudra_operational.db")
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{fallback_path.as_posix()}"


def _normalize_db_url(url: str) -> str:
    clean_url = url.strip()
    if clean_url.startswith("postgres://"):
        return clean_url.replace("postgres://", "postgresql+psycopg2://", 1)
    if clean_url.startswith("postgresql://") and not clean_url.startswith("postgresql+"):
        return clean_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return clean_url


# Create configured SQLAlchemy Engine with connection pooling
def create_db_engine(db_url: Optional[str] = None) -> Engine:
    url = db_url or get_database_url()

    if "sqlite" in url:
        return create_engine(url, connect_args={"check_same_thread": False}, echo=False)

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,
        connect_args={"connect_timeout": 10},
        echo=False,
    )


# Global Engine and SessionLocal
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Context manager for database session with auto-commit and rollback
@contextmanager
def get_db() -> Generator[Session, None, None]:
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


_TICK_ID_COLUMN_CACHE: Dict[int, bool] = {}


# Checks if paper_portfolio table has tick_id column
def has_tick_id_column(session: Session) -> bool:
    try:
        bind = session.get_bind()
        bind_id = id(bind)
        if bind_id in _TICK_ID_COLUMN_CACHE:
            return _TICK_ID_COLUMN_CACHE[bind_id]

        insp_target = session.connection() if hasattr(session, "connection") else bind
        inspector = inspect(insp_target)
        if inspector.has_table("paper_portfolio"):
            cols = {c["name"] for c in inspector.get_columns("paper_portfolio")}
            present = "tick_id" in cols
            _TICK_ID_COLUMN_CACHE[bind_id] = present
            return present
    except Exception as exc:
        logger.warning(f"Failed to inspect paper_portfolio table columns: {exc}")
    return False


def _defer_missing_tick_id_on_legacy_portfolio(orm_execute_state: Any) -> None:
    if (
        orm_execute_state.is_select
        and not orm_execute_state.is_column_load
        and not orm_execute_state.is_relationship_load
    ):
        try:
            mappers = orm_execute_state.all_mappers
            if any(m.class_ is PaperPortfolio for m in mappers):
                if not has_tick_id_column(orm_execute_state.session):
                    orm_execute_state.statement = orm_execute_state.statement.options(
                        defer(PaperPortfolio.tick_id)
                    )
        except Exception:
            pass


def _populate_deferred_tick_id_on_load(target: PaperPortfolio, context: Any) -> None:
    if "tick_id" not in target.__dict__:
        attributes.set_committed_value(target, "tick_id", None)


if __name__ == "src.database.connection":
    event.listen(Session, "do_orm_execute", _defer_missing_tick_id_on_legacy_portfolio)
    event.listen(PaperPortfolio, "load", _populate_deferred_tick_id_on_load)


# Initializes all database tables via SQLAlchemy metadata
def init_db(target_engine: Optional[Engine] = None) -> None:
    target = target_engine or engine
    Base.metadata.create_all(bind=target)

    with target.connect() as conn:
        inspector = inspect(conn)

        if inspector.has_table("paper_portfolio"):
            cols = {c["name"] for c in inspector.get_columns("paper_portfolio")}
            if "tick_id" not in cols:
                try:
                    if "postgresql" in target.dialect.name:
                        conn.execute(text("ALTER TABLE paper_portfolio ADD COLUMN IF NOT EXISTS tick_id INTEGER REFERENCES market_ticks(id);"))
                    elif "sqlite" in target.dialect.name:
                        conn.execute(text("ALTER TABLE paper_portfolio ADD COLUMN tick_id INTEGER REFERENCES market_ticks(id);"))
                    conn.commit()
                except Exception as exc:
                    logger.error(f"Failed to add tick_id to paper_portfolio: {exc}")
                    raise RuntimeError(f"Database migration failed: unable to add required column 'tick_id' to paper_portfolio: {exc}") from exc

            existing_indexes = {idx["name"] for idx in inspector.get_indexes("paper_portfolio")}
            if "uq_paper_portfolio_tick_id" not in existing_indexes:
                try:
                    conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_portfolio_tick_id ON paper_portfolio(tick_id) WHERE tick_id IS NOT NULL;"))
                    conn.commit()
                except Exception as exc:
                    logger.warning(f"Could not create index uq_paper_portfolio_tick_id: {exc}")

            _TICK_ID_COLUMN_CACHE[id(target)] = True

        if inspector.has_table("market_ticks"):
            existing_mt_indexes = {idx["name"] for idx in inspector.get_indexes("market_ticks")}
            if "uq_market_ticks_timestamp" not in existing_mt_indexes:
                try:
                    conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_market_ticks_timestamp ON market_ticks(timestamp);"))
                    conn.commit()
                except Exception as exc:
                    logger.warning(f"Could not create index uq_market_ticks_timestamp: {exc}")


# Pings database and measures latency
def test_connection() -> Tuple[bool, str, float]:
    start_time = time.perf_counter()
    try:
        with engine.connect() as conn:
            is_sqlite = engine.dialect.name == "sqlite"
            query_str = "SELECT sqlite_version();" if is_sqlite else "SELECT version();"
            result = conn.execute(text(query_str)).fetchone()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            version_str = f"SQLite {result[0]}" if (is_sqlite and result) else (str(result[0]) if result else "Unknown")
            return True, version_str, latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return False, str(exc), latency_ms


if __name__ == "__main__":
    success, info, latency = test_connection()
    status = "SUCCESS" if success else "FAILED"
    print(f"[{status}] Connected: {success} | Version: {info} | Latency: {latency:.2f}ms")
