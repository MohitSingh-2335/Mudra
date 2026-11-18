import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import tomllib

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session, attributes, defer, sessionmaker

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.models import Base, PaperPortfolio

logger = logging.getLogger(__name__)
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


CANONICAL_TABLE_COLUMNS: Dict[str, List[Tuple[str, str, bool]]] = {
    "market_ticks": [("id", "integer", False), ("timestamp", "timestamp", False), ("open", "float", False), ("high", "float", False), ("low", "float", False), ("close", "float", False), ("volume", "float", False), ("model_version", "varchar", False), ("feature_schema_version", "varchar", False), ("market_regime", "varchar", True), ("predicted_return_1h", "float", True), ("predicted_price_1h", "float", True), ("model_confidence", "float", True), ("features", "json", True)],
    "api_telemetry": [("id", "integer", False), ("timestamp", "timestamp", False), ("service_name", "varchar", False), ("endpoint", "varchar", False), ("latency_ms", "float", False), ("status_code", "integer", False), ("retry_count", "integer", False), ("rate_limit_remaining", "integer", True), ("tokens_used", "integer", True), ("cost_estimate_usd", "float", True), ("error_message", "text", True)],
    "council_audits": [("id", "integer", False), ("timestamp", "timestamp", False), ("tick_id", "integer", True), ("deliberation_latency_ms", "float", True), ("strategist_action", "varchar", False), ("strategist_conviction", "integer", False), ("rag_context_summary", "text", True), ("risk_verdict", "varchar", False), ("veto_reasons", "json", True), ("kelly_fraction", "float", False), ("final_decision", "varchar", False), ("veto_attribution", "varchar", True), ("execution_notes", "text", True)],
    "reconciliations": [("id", "integer", False), ("timestamp", "timestamp", False), ("tick_id", "integer", False), ("predicted_price", "float", False), ("actual_price_at_t_plus_1", "float", False), ("predicted_return_pct", "float", False), ("actual_return_pct", "float", False), ("residual_error", "float", False), ("absolute_error_usd", "float", False), ("percentage_error", "float", False), ("direction_predicted", "varchar", False), ("direction_realized", "varchar", False), ("direction_correct", "boolean", False), ("error_attribution_tag", "varchar", False), ("is_black_swan", "boolean", False)],
    "paper_portfolio": [("id", "integer", False), ("tick_id", "integer", True), ("timestamp", "timestamp", False), ("action", "varchar", False), ("btc_amount", "float", False), ("execution_price", "float", False), ("fee_deducted_usd", "float", False), ("slippage_usd", "float", False), ("cash_balance", "float", False), ("btc_balance", "float", False), ("total_equity_usd", "float", False), ("realized_pnl_usd", "float", False), ("unrealized_pnl_usd", "float", False), ("drawdown_pct", "float", False), ("cumulative_return_pct", "float", False), ("benchmark_btc_equity", "float", False), ("alpha_usd", "float", False), ("alpha_pct", "float", False), ("notes", "text", True)],
    "model_registry": [("id", "integer", False), ("model_version", "varchar", False), ("regressor_artifact_path", "varchar", False), ("regressor_sha256", "varchar", False), ("classifier_artifact_path", "varchar", False), ("classifier_sha256", "varchar", False), ("feature_schema_version", "varchar", False), ("target_definition", "varchar", False), ("training_dataset_path", "varchar", True), ("training_dataset_sha256", "varchar", True), ("training_row_count", "integer", True), ("split_metadata", "json", True), ("evaluation_metrics", "json", True), ("code_revision_sha", "varchar", True), ("current_status", "varchar", False), ("lifecycle_seq", "integer", False), ("created_at", "timestamp", False)],
    "model_lifecycle_events": [("id", "integer", False), ("model_version", "varchar", False), ("lifecycle_seq", "integer", False), ("event_type", "varchar", False), ("from_status", "varchar", True), ("to_status", "varchar", False), ("related_model_version", "varchar", True), ("actor", "varchar", False), ("reason", "text", False), ("evidence_json", "json", True), ("created_at", "timestamp", False)],
    "model_predictions": [("id", "integer", False), ("tick_id", "integer", False), ("model_version", "varchar", False), ("run_mode", "varchar", False), ("is_canonical", "boolean", False), ("is_primary", "boolean", False), ("parent_prediction_id", "integer", True), ("feature_schema_version", "varchar", False), ("predicted_return_pct", "float", False), ("predicted_price_1h", "float", False), ("direction_predicted", "varchar", False), ("direction_prob_up", "float", False), ("clears_hurdle", "boolean", False), ("created_at", "timestamp", False), ("outcome_tick_id", "integer", True), ("evaluation_policy_version", "varchar", True), ("actual_price_at_t_plus_1", "float", True), ("actual_return_pct", "float", True), ("direction_realized", "varchar", True), ("direction_correct", "boolean", True), ("residual_error", "float", True), ("absolute_error_usd", "float", True), ("percentage_error", "float", True), ("is_black_swan", "boolean", True), ("error_attribution_tag", "varchar", True), ("reconciled_at", "timestamp", True)],
    "prediction_investigations": [("id", "integer", False), ("tick_id", "integer", False), ("prediction_id", "integer", True), ("model_version", "varchar", False), ("headline", "varchar", False), ("headline_source_url", "text", True), ("headline_published_at", "timestamp", True), ("suspected_cause_category", "varchar", False), ("hypothesis_notes", "text", False), ("supporting_evidence_json", "json", True), ("diagnostic_confidence", "float", True), ("unresolved_questions", "text", True), ("verification_status", "varchar", False), ("verified_ground_truth_label", "varchar", True), ("verified_by", "varchar", True), ("verified_at", "timestamp", True), ("revision_history_json", "json", False), ("created_at", "timestamp", False), ("updated_at", "timestamp", False)],
}

CANONICAL_FOREIGN_KEYS = [
    ("fk_council_audits_tick_id", "council_audits", "tick_id", "market_ticks", "id"),
    ("fk_reconciliations_tick_id", "reconciliations", "tick_id", "market_ticks", "id"),
    ("fk_paper_portfolio_tick_id", "paper_portfolio", "tick_id", "market_ticks", "id"),
]

CANONICAL_CONSTRAINTS = [
    ("chk_council_audits_action", "council_audits", "c", None, "strategist_action in ('BUY', 'SELL', 'HOLD')"),
    ("chk_council_audits_verdict", "council_audits", "c", None, "risk_verdict in ('APPROVED', 'VETOED')"),
    ("chk_model_registry_status", "model_registry", "c", None, "current_status in ('CANDIDATE', 'ACTIVE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK')"),
]

CANONICAL_INDEXES = [
    ("uq_model_registry_single_active", "model_registry", True, ["current_status"], "WHERE current_status = 'ACTIVE'"),
    ("uq_paper_portfolio_tick_id", "paper_portfolio", True, ["tick_id"], "WHERE tick_id IS NOT NULL"),
]

CANONICAL_FUNCTIONS: List[FunctionSpec] = [
    FunctionSpec("public", "mudra_prevent_mutation", "BEGIN RAISE EXCEPTION 'Table % is append-only.', TG_TABLE_NAME; END;", "CREATE OR REPLACE FUNCTION public.mudra_prevent_mutation() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'Table % is append-only.', TG_TABLE_NAME; END; $$;"),
]

CANONICAL_TRIGGERS: List[TriggerSpec] = [
    TriggerSpec("trg_mr_no_delete", "model_registry", "mudra_prevent_mutation", 29, False, False, "", "CREATE TRIGGER trg_mr_no_delete BEFORE DELETE ON public.model_registry FOR EACH ROW EXECUTE FUNCTION public.mudra_prevent_mutation();"),
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
        inspector = inspect(conn)
        existing_tables = set(inspector.get_table_names())
        for tbl_name in CANONICAL_TABLE_COLUMNS:
            if tbl_name not in existing_tables:
                raise SchemaDriftError(f"Required table '{tbl_name}' is missing from database schema.")
        return {"status": "verified", "dialect": conn.dialect.name, "tables": len(CANONICAL_TABLE_COLUMNS)}


def _normalize_db_url(url: str) -> str:
    clean = url.strip()
    if clean.startswith("postgres://"):
        return clean.replace("postgres://", "postgresql+psycopg2://", 1)
    if clean.startswith("postgresql://") and not clean.startswith("postgresql+"):
        return clean.replace("postgresql://", "postgresql+psycopg2://", 1)
    return clean


def get_database_url() -> str:
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return _normalize_db_url(db_url)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
            return _normalize_db_url(st.secrets["DATABASE_URL"])
    except Exception:
        pass
    if SECRETS_FILE.exists():
        try:
            data = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            if "DATABASE_URL" in data:
                return _normalize_db_url(data["DATABASE_URL"])
        except Exception:
            pass
    fallback = Path("data/mudra_operational.db")
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{fallback.as_posix()}"


def create_db_engine(db_url: Optional[str] = None) -> Engine:
    url = db_url or get_database_url()
    if "sqlite" in url:
        return create_engine(url, connect_args={"check_same_thread": False}, echo=False)
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10, pool_recycle=300, connect_args={"connect_timeout": 10}, echo=False)


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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


def has_tick_id_column(session: Session) -> bool:
    try:
        bind = session.get_bind()
        bind_id = id(bind)
        if bind_id in _TICK_ID_COLUMN_CACHE:
            return _TICK_ID_COLUMN_CACHE[bind_id]
        inspector = inspect(session.connection() if hasattr(session, "connection") else bind)
        if inspector.has_table("paper_portfolio"):
            cols = {c["name"] for c in inspector.get_columns("paper_portfolio")}
            present = "tick_id" in cols
            _TICK_ID_COLUMN_CACHE[bind_id] = present
            return present
    except Exception:
        pass
    return False


def _defer_missing_tick_id_on_legacy_portfolio(orm_execute_state: Any) -> None:
    if orm_execute_state.is_select and not orm_execute_state.is_column_load and not orm_execute_state.is_relationship_load:
        try:
            if any(m.class_ is PaperPortfolio for m in orm_execute_state.all_mappers):
                if not has_tick_id_column(orm_execute_state.session):
                    orm_execute_state.statement = orm_execute_state.statement.options(defer(PaperPortfolio.tick_id))
        except Exception:
            pass


def _populate_deferred_tick_id_on_load(target: PaperPortfolio, context: Any) -> None:
    if "tick_id" not in target.__dict__:
        attributes.set_committed_value(target, "tick_id", None)


if __name__ == "src.database.connection":
    event.listen(Session, "do_orm_execute", _defer_missing_tick_id_on_legacy_portfolio)
    event.listen(PaperPortfolio, "load", _populate_deferred_tick_id_on_load)


def init_db(target_engine: Optional[Engine] = None) -> None:
    target = target_engine or engine
    Base.metadata.create_all(bind=target)
    _TICK_ID_COLUMN_CACHE[id(target)] = True


def test_connection() -> Tuple[bool, str, float]:
    start = time.perf_counter()
    try:
        with engine.connect() as conn:
            is_sqlite = engine.dialect.name == "sqlite"
            q = "SELECT sqlite_version();" if is_sqlite else "SELECT version();"
            res = conn.execute(text(q)).fetchone()
            latency = (time.perf_counter() - start) * 1000.0
            ver = f"SQLite {res[0]}" if (is_sqlite and res) else (str(res[0]) if res else "Unknown")
            return True, ver, latency
    except Exception as exc:
        return False, str(exc), (time.perf_counter() - start) * 1000.0
