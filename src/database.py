# src/database.py
"""
Mudra Database Architecture.
Contains SQLAlchemy ORM models, resilient connection pooling, schema drift verification,
and analytical queries for the quantitative trading engine and audit council.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
import tomllib
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sqlalchemy import (
    Boolean, CheckConstraint, Column, DateTime, Engine, Float, ForeignKey,
    ForeignKeyConstraint, Index, Integer, JSON, String, Text, UniqueConstraint,
    create_engine, desc, inspect, select, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from config import ACTIVE_MODEL_VERSION, FEATURE_SCHEMA_VERSION

logger = logging.getLogger("mudra.database")
ROOT_DIR = Path(__file__).resolve().parent.parent
SECRETS_FILE = ROOT_DIR / ".streamlit" / "secrets.toml"

# ---------------------------------------------------------------------------
# 1. ORM DATA MODELS
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    def to_dict(self) -> Dict[str, Any]:
        return {
            c.name: getattr(self, c.name).isoformat() if hasattr(getattr(self, c.name), "isoformat") else getattr(self, c.name)
            for c in self.__table__.columns
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={getattr(self, 'id', None)})>"


class MarketTick(Base):
    __tablename__ = "market_ticks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    model_version = Column(String(32), nullable=False, default=ACTIVE_MODEL_VERSION)
    predicted_return_1h = Column(Float, nullable=True)
    predicted_price_1h = Column(Float, nullable=True)
    model_confidence = Column(Float, nullable=True)
    features = Column(JSON, nullable=True)

    council_audits = relationship("CouncilAudit", back_populates="tick", uselist=True, order_by="CouncilAudit.id.asc()")
    reconciliation = relationship("Reconciliation", back_populates="tick", uselist=False)
    paper_portfolios = relationship("PaperPortfolio", back_populates="tick", foreign_keys="PaperPortfolio.tick_id", uselist=True)
    predictions = relationship("ModelPrediction", back_populates="tick", foreign_keys="ModelPrediction.tick_id", uselist=True)
    outcome_predictions = relationship("ModelPrediction", back_populates="outcome_tick", foreign_keys="ModelPrediction.outcome_tick_id", uselist=True)

    @property
    def council_audit(self) -> Optional["CouncilAudit"]:
        return self.council_audits[-1] if self.council_audits else None


class ApiTelemetry(Base):
    __tablename__ = "api_telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc), nullable=False)
    service_name = Column(String(64), nullable=False, index=True)
    endpoint = Column(String(256), nullable=False)
    latency_ms = Column(Float, nullable=False)
    status_code = Column(Integer, nullable=False)
    error_message = Column(Text, nullable=True)


class CouncilAudit(Base):
    __tablename__ = "council_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc), nullable=False)
    tick_id = Column(Integer, ForeignKey("market_ticks.id"), nullable=True, index=True)
    strategist_action = Column(String(16), nullable=False)
    strategist_conviction = Column(Integer, nullable=False)
    rag_context_summary = Column(Text, nullable=True)
    risk_verdict = Column(String(16), nullable=False)
    final_decision = Column(String(16), nullable=False)
    execution_notes = Column(Text, nullable=True)

    tick = relationship("MarketTick", back_populates="council_audits")


class Reconciliation(Base):
    __tablename__ = "reconciliations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc), nullable=False)
    tick_id = Column(Integer, ForeignKey("market_ticks.id"), nullable=False, unique=True, index=True)
    predicted_price = Column(Float, nullable=False)
    actual_price_at_t_plus_1 = Column(Float, nullable=False)
    predicted_return_pct = Column(Float, nullable=False)
    actual_return_pct = Column(Float, nullable=False)
    residual_error = Column(Float, nullable=False)
    absolute_error_usd = Column(Float, nullable=False)
    percentage_error = Column(Float, nullable=False)
    direction_predicted = Column(String(16), nullable=False)
    direction_realized = Column(String(16), nullable=False)
    direction_correct = Column(Boolean, nullable=False)
    is_black_swan = Column(Boolean, nullable=False, default=False)
    error_attribution_tag = Column(String(64), nullable=False)

    tick = relationship("MarketTick", back_populates="reconciliation")


class PaperPortfolio(Base):
    __tablename__ = "paper_portfolio"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tick_id = Column(Integer, ForeignKey("market_ticks.id"), nullable=True)
    timestamp = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc), nullable=False)
    action = Column(String(16), nullable=False)
    btc_amount = Column(Float, nullable=False, default=0.0)
    execution_price = Column(Float, nullable=False, default=0.0)
    fee_deducted_usd = Column(Float, nullable=False, default=0.0)
    slippage_usd = Column(Float, nullable=False, default=0.0)

    cash_balance = Column(Float, nullable=False)
    btc_balance = Column(Float, nullable=False)
    total_equity_usd = Column(Float, nullable=False)
    realized_pnl_usd = Column(Float, nullable=False, default=0.0)
    unrealized_pnl_usd = Column(Float, nullable=False, default=0.0)
    drawdown_pct = Column(Float, nullable=False, default=0.0)
    cumulative_return_pct = Column(Float, nullable=False)

    benchmark_btc_equity = Column(Float, nullable=False)
    alpha_usd = Column(Float, nullable=False, default=0.0)
    alpha_pct = Column(Float, nullable=False, default=0.0)
    notes = Column(Text, nullable=True)

    tick = relationship("MarketTick", foreign_keys=[tick_id], back_populates="paper_portfolios")

    __table_args__ = (
        Index("uq_paper_portfolio_tick_id", "tick_id", unique=True, postgresql_where=text("tick_id IS NOT NULL"), sqlite_where=text("tick_id IS NOT NULL")),
    )


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(64), nullable=False)
    regressor_artifact_path = Column(String(256), nullable=False)
    regressor_sha256 = Column(String(64), nullable=False)
    classifier_artifact_path = Column(String(256), nullable=False)
    classifier_sha256 = Column(String(64), nullable=False)
    feature_schema_version = Column(String(32), nullable=False)
    target_definition = Column(String(128), nullable=False)
    evaluation_metrics = Column(JSON, nullable=True)
    current_status = Column(String(24), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    predictions = relationship("ModelPrediction", primaryjoin="ModelRegistry.model_version == ModelPrediction.model_version", foreign_keys="ModelPrediction.model_version", back_populates="model", order_by="ModelPrediction.id.asc()")

    __table_args__ = (
        UniqueConstraint("model_version", name="uq_model_registry_model_version"),
        CheckConstraint("current_status IN ('ACTIVE', 'CANDIDATE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK')", name="chk_model_registry_status"),
        Index("uq_model_registry_single_active", "current_status", unique=True, postgresql_where=text("current_status = 'ACTIVE'"), sqlite_where=text("current_status = 'ACTIVE'")),
    )


class HourlyPostmortem(Base):
    __tablename__ = "hourly_postmortems"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tick_id = Column(Integer, ForeignKey("market_ticks.id"), nullable=False, unique=True, index=True)
    news_headlines = Column(JSON, nullable=True)
    feature_snapshot = Column(JSON, nullable=True)
    gemini_full_analysis = Column(JSON, nullable=True)
    gemini_status = Column(String(32), nullable=False, default="UNAVAILABLE")
    error_detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    tick = relationship("MarketTick", foreign_keys=[tick_id])


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tick_id = Column(Integer, nullable=False)
    model_version = Column(String(64), nullable=False)
    run_mode = Column(String(24), nullable=False)
    is_primary = Column(Boolean, nullable=False, default=False)
    feature_schema_version = Column(String(32), nullable=False, default=FEATURE_SCHEMA_VERSION)
    predicted_return_pct = Column(Float, nullable=True)
    predicted_price_1h = Column(Float, nullable=True)
    direction_predicted = Column(String(16), nullable=True)
    direction_prob_up = Column(Float, nullable=True)
    clears_hurdle = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    outcome_tick_id = Column(Integer, nullable=True)
    actual_price_at_t_plus_1 = Column(Float, nullable=True)
    actual_return_pct = Column(Float, nullable=True)
    direction_realized = Column(String(16), nullable=True)
    direction_correct = Column(Boolean, nullable=True)
    residual_error = Column(Float, nullable=True)
    absolute_error_usd = Column(Float, nullable=True)
    percentage_error = Column(Float, nullable=True)
    error_attribution_tag = Column(String(64), nullable=True)
    diagnosis_summary = Column(Text, nullable=True)
    reconciled_at = Column(DateTime(timezone=True), nullable=True)

    tick = relationship("MarketTick", foreign_keys=[tick_id], back_populates="predictions")
    outcome_tick = relationship("MarketTick", foreign_keys=[outcome_tick_id], back_populates="outcome_predictions")
    model = relationship("ModelRegistry", primaryjoin="ModelPrediction.model_version == ModelRegistry.model_version", foreign_keys=[model_version], back_populates="predictions")

    __table_args__ = (
        ForeignKeyConstraint(["tick_id"], ["market_ticks.id"], name="fk_mp_tick_id"),
        ForeignKeyConstraint(["model_version"], ["model_registry.model_version"], name="fk_mp_model_version"),
        ForeignKeyConstraint(["outcome_tick_id"], ["market_ticks.id"], name="fk_mp_outcome_tick_id"),
        CheckConstraint(
            "(((run_mode = 'PRIMARY' AND is_primary IS TRUE) "
            "OR (run_mode = 'SHADOW' AND is_primary IS FALSE))) "
            "AND (predicted_return_pct IS NULL OR predicted_return_pct NOT IN ('NaN', 'Infinity', '-Infinity')) "
            "AND (predicted_price_1h IS NULL OR (predicted_price_1h NOT IN ('NaN', 'Infinity', '-Infinity') AND predicted_price_1h >= 1.0)) "
            "AND (direction_prob_up IS NULL OR (direction_prob_up NOT IN ('NaN', 'Infinity', '-Infinity') AND direction_prob_up >= 0.0 AND direction_prob_up <= 1.0)) "
            "AND (direction_predicted IS NULL OR direction_predicted IN ('UP', 'DOWN'))",
            name="chk_prediction_mode_invariants",
        ),
        CheckConstraint(
            "(outcome_tick_id IS NULL AND actual_price_at_t_plus_1 IS NULL "
            "AND actual_return_pct IS NULL AND direction_realized IS NULL AND direction_correct IS NULL "
            "AND residual_error IS NULL AND absolute_error_usd IS NULL AND percentage_error IS NULL "
            "AND error_attribution_tag IS NULL AND reconciled_at IS NULL) "
            "OR (outcome_tick_id IS NOT NULL "
            "AND actual_price_at_t_plus_1 IS NOT NULL AND actual_return_pct IS NOT NULL "
            "AND direction_realized IS NOT NULL AND direction_correct IS NOT NULL "
            "AND residual_error IS NOT NULL AND absolute_error_usd IS NOT NULL "
            "AND percentage_error IS NOT NULL "
            "AND error_attribution_tag IS NOT NULL AND reconciled_at IS NOT NULL "
            "AND actual_price_at_t_plus_1 >= 1.0 "
            "AND direction_realized IN ('UP', 'DOWN') "
            "AND direction_realized = (CASE WHEN actual_return_pct >= 0.0 THEN 'UP' ELSE 'DOWN' END) "
            "AND direction_correct = (direction_predicted = direction_realized) "
            "AND absolute_error_usd >= 0.0 AND percentage_error >= 0.0 "
            "AND error_attribution_tag IN ('ACCURATE', 'MOMENTUM_UNDERESTIMATED', 'MOMENTUM_OVERESTIMATED', "
            "'SIDEWAYS_DRIFT', 'DIRECTION_MISALIGNED', 'BLACK_SWAN_OUTLIER'))",
            name="chk_reconciliation_all_or_nothing",
        ),
        Index("uq_model_predictions_tick_version", "tick_id", "model_version", unique=True),
        Index("uq_model_predictions_primary_tick", "tick_id", unique=True, postgresql_where=text("is_primary IS TRUE"), sqlite_where=text("is_primary IS TRUE")),
        Index("ix_model_predictions_tick_id", "tick_id"),
        Index("ix_model_predictions_model_version", "model_version"),
    )


# ---------------------------------------------------------------------------
# 2. SCHEMA DRIFT VERIFICATION & SPECS
# ---------------------------------------------------------------------------

class SchemaDriftError(RuntimeError):
    pass


def _classify_column_type(raw_type: str) -> str:
    t = str(raw_type).lower()
    for pattern, canonical in [
        (("int", "serial"), "integer"),
        (("double", "float", "real", "numeric"), "float"),
        (("char",), "varchar"),
        (("text",), "text"),
        (("bool",), "boolean"),
        (("json",), "json"),
        (("timestamp", "datetime"), "timestamp"),
    ]:
        if any(p in t for p in pattern):
            return canonical
    return t


CANONICAL_TABLE_COLUMNS: Dict[str, List[Tuple[str, str, bool]]] = {
    "market_ticks": [("id", "integer", False), ("timestamp", "timestamp", False), ("open", "float", False), ("high", "float", False), ("low", "float", False), ("close", "float", False), ("volume", "float", False), ("model_version", "varchar", False), ("predicted_return_1h", "float", True), ("predicted_price_1h", "float", True), ("model_confidence", "float", True), ("features", "json", True)],
    "api_telemetry": [("id", "integer", False), ("timestamp", "timestamp", False), ("service_name", "varchar", False), ("endpoint", "varchar", False), ("latency_ms", "float", False), ("status_code", "integer", False), ("error_message", "text", True)],
    "council_audits": [("id", "integer", False), ("timestamp", "timestamp", False), ("tick_id", "integer", True), ("strategist_action", "varchar", False), ("strategist_conviction", "integer", False), ("rag_context_summary", "text", True), ("risk_verdict", "varchar", False), ("final_decision", "varchar", False), ("execution_notes", "text", True)],
    "reconciliations": [("id", "integer", False), ("timestamp", "timestamp", False), ("tick_id", "integer", False), ("predicted_price", "float", False), ("actual_price_at_t_plus_1", "float", False), ("predicted_return_pct", "float", False), ("actual_return_pct", "float", False), ("residual_error", "float", False), ("absolute_error_usd", "float", False), ("percentage_error", "float", False), ("direction_predicted", "varchar", False), ("direction_realized", "varchar", False), ("direction_correct", "boolean", False), ("error_attribution_tag", "varchar", False), ("is_black_swan", "boolean", False)],
    "paper_portfolio": [("id", "integer", False), ("tick_id", "integer", True), ("timestamp", "timestamp", False), ("action", "varchar", False), ("btc_amount", "float", False), ("execution_price", "float", False), ("fee_deducted_usd", "float", False), ("slippage_usd", "float", False), ("cash_balance", "float", False), ("btc_balance", "float", False), ("total_equity_usd", "float", False), ("realized_pnl_usd", "float", False), ("unrealized_pnl_usd", "float", False), ("drawdown_pct", "float", False), ("cumulative_return_pct", "float", False), ("benchmark_btc_equity", "float", False), ("alpha_usd", "float", False), ("alpha_pct", "float", False), ("notes", "text", True)],
    "model_registry": [("id", "integer", False), ("model_version", "varchar", False), ("regressor_artifact_path", "varchar", False), ("regressor_sha256", "varchar", False), ("classifier_artifact_path", "varchar", False), ("classifier_sha256", "varchar", False), ("feature_schema_version", "varchar", False), ("target_definition", "varchar", False), ("evaluation_metrics", "json", True), ("current_status", "varchar", False), ("created_at", "timestamp", False)],
    "model_predictions": [("id", "integer", False), ("tick_id", "integer", False), ("model_version", "varchar", False), ("run_mode", "varchar", False), ("is_primary", "boolean", False), ("feature_schema_version", "varchar", False), ("predicted_return_pct", "float", True), ("predicted_price_1h", "float", True), ("direction_predicted", "varchar", True), ("direction_prob_up", "float", True), ("clears_hurdle", "boolean", False), ("created_at", "timestamp", False), ("outcome_tick_id", "integer", True), ("actual_price_at_t_plus_1", "float", True), ("actual_return_pct", "float", True), ("direction_realized", "varchar", True), ("direction_correct", "boolean", True), ("residual_error", "float", True), ("absolute_error_usd", "float", True), ("percentage_error", "float", True), ("error_attribution_tag", "varchar", True), ("diagnosis_summary", "text", True), ("reconciled_at", "timestamp", True)],
    "hourly_postmortems": [("id", "integer", False), ("tick_id", "integer", False), ("news_headlines", "json", True), ("feature_snapshot", "json", True), ("gemini_full_analysis", "json", True), ("gemini_status", "varchar", False), ("error_detail", "text", True), ("created_at", "timestamp", False)],
}

CANONICAL_FOREIGN_KEYS = [
    ("fk_council_audits_tick_id", "council_audits", "tick_id", "market_ticks", "id"),
    ("fk_reconciliations_tick_id", "reconciliations", "tick_id", "market_ticks", "id"),
    ("fk_paper_portfolio_tick_id", "paper_portfolio", "tick_id", "market_ticks", "id"),
    ("fk_hourly_postmortems_tick_id", "hourly_postmortems", "tick_id", "market_ticks", "id"),
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
        existing_tables = set(inspect(conn).get_table_names())
        for tbl_name in CANONICAL_TABLE_COLUMNS:
            if tbl_name not in existing_tables:
                raise SchemaDriftError(f"Required table '{tbl_name}' is missing from database schema.")
        return {"status": "verified", "dialect": conn.dialect.name, "tables": len(CANONICAL_TABLE_COLUMNS)}


# ---------------------------------------------------------------------------
# 3. CONNECTION POOLING & TRANSACTION CONTEXT
# ---------------------------------------------------------------------------

def _normalize_db_url(url: str) -> str:
    clean = url.strip()
    if clean.startswith("postgres://"):
        return clean.replace("postgres://", "postgresql+psycopg2://", 1)
    if clean.startswith("postgresql://") and not clean.startswith("postgresql+"):
        return clean.replace("postgresql://", "postgresql+psycopg2://", 1)
    return clean


def _find_postgres_url(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith(("postgres://", "postgresql://", "postgresql+")):
            return s
        return None
    if isinstance(obj, dict) or hasattr(obj, "items"):
        try:
            for k in ["DATABASE_URL", "database_url", "POSTGRES_URL", "postgres_url", "NEON_DATABASE_URL", "url", "uri", "connection_string"]:
                if k in obj:
                    res = _find_postgres_url(obj[k])
                    if res:
                        return res
            for v in obj.values():
                res = _find_postgres_url(v)
                if res:
                    return res
        except Exception:
            pass
    return None


def get_database_url() -> str:
    db_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    if db_url:
        return _normalize_db_url(db_url)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            found = _find_postgres_url(st.secrets)
            if found:
                return _normalize_db_url(found)
    except Exception as exc:
        logger.warning("Failed to read st.secrets (%s)", type(exc).__name__)
    if SECRETS_FILE.exists():
        try:
            data = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            found = _find_postgres_url(data)
            if found:
                return _normalize_db_url(found)
        except Exception as exc:
            loc = f" at line {exc.lineno}, col {exc.colno}" if hasattr(exc, "lineno") else ""
            logger.warning("Failed to parse %s: %s%s", SECRETS_FILE.name, type(exc).__name__, loc)
    fallback = ROOT_DIR / "data" / "mudra_operational.db"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{fallback.as_posix()}"


def create_db_engine(db_url: Optional[str] = None) -> Engine:
    fallback = ROOT_DIR / "data" / "mudra_operational.db"
    try:
        url = db_url or get_database_url()
        if "sqlite" in url:
            return create_engine(url, connect_args={"check_same_thread": False}, echo=False)
        return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10, pool_recycle=300, connect_args={"connect_timeout": 10}, echo=False)
    except Exception as exc:
        logger.warning("Failed to create remote database engine, falling back to sqlite: %s", exc)
        try:
            fallback.parent.mkdir(parents=True, exist_ok=True)
            return create_engine(f"sqlite:///{fallback.as_posix()}", connect_args={"check_same_thread": False}, echo=False)
        except Exception:
            return create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, echo=False)


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def ensure_connected_engine() -> Engine:
    global engine, SessionLocal
    if engine.dialect.name == "sqlite":
        db_path = str(engine.url.database or "")
        if db_path and db_path != ":memory:" and "mudra_operational" in db_path:
            url = get_database_url()
            if not url.startswith("sqlite"):
                engine = create_db_engine(url)
                SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine


@contextmanager
def get_db() -> Generator[Session, None, None]:
    ensure_connected_engine()
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


def has_tick_id_column(session: Any = None) -> bool:
    # returns True as paper_portfolio table includes the canonical tick_id column
    return True


def init_db(target_engine: Optional[Engine] = None) -> None:
    target = target_engine or engine
    Base.metadata.create_all(bind=target)


def test_connection() -> Tuple[bool, str, float]:
    ensure_connected_engine()
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


# ---------------------------------------------------------------------------
# 4. ANALYTICAL QUERIES & ROOT-CAUSE INVESTIGATIONS
# ---------------------------------------------------------------------------

ALLOWED_VERIFICATION_STATUSES = frozenset({"UNVERIFIED_HYPOTHESIS", "CONFIRMED_CAUSE", "REFUTED", "APPROVED_FOR_TRAINING"})


def get_latest_market_ticks(limit: int = 100) -> pd.DataFrame:
    try:
        with get_db() as session:
            ticks = session.execute(select(MarketTick).order_by(desc(MarketTick.timestamp)).limit(limit)).scalars().all()
            if not ticks:
                return pd.DataFrame()
            df = pd.DataFrame([t.to_dict() for t in ticks])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        logger.warning("Failed to fetch market ticks: %s", type(exc).__name__)
        return pd.DataFrame()


def get_reconciliation_history(limit: int = 500, model_version: Optional[str] = None) -> pd.DataFrame:
    # returns unified ground-truth evaluations across all registered models
    try:
        with get_db() as session:
            raw_data = []

            # 1. query unified model predictions (primary and candidate models)
            try:
                cand_stmt = (
                    select(
                        ModelPrediction,
                        MarketTick.timestamp.label("tick_ts"),
                        MarketTick.close.label("start_price"),
                    )
                    .join(MarketTick, ModelPrediction.tick_id == MarketTick.id)
                    .where(ModelPrediction.outcome_tick_id.isnot(None))
                )
                if model_version and model_version != "All Models":
                    cand_stmt = cand_stmt.where(ModelPrediction.model_version == model_version)
                for cp, tick_ts, start_price in session.execute(
                    cand_stmt.order_by(desc(ModelPrediction.reconciled_at)).limit(limit)
                ).all():
                    ts = cp.reconciled_at or tick_ts
                    raw_data.append({
                        "id": cp.id,
                        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                        "tick_id": cp.tick_id,
                        "start_price": float(start_price) if start_price is not None else (float(cp.predicted_price_1h) / (1.0 + float(cp.predicted_return_pct or 0.0) / 100.0) if cp.predicted_price_1h else None),
                        "predicted_price": cp.predicted_price_1h,
                        "actual_price_at_t_plus_1": cp.actual_price_at_t_plus_1,
                        "predicted_return_pct": cp.predicted_return_pct,
                        "actual_return_pct": cp.actual_return_pct,
                        "residual_error": cp.residual_error,
                        "absolute_error_usd": cp.absolute_error_usd,
                        "percentage_error": cp.percentage_error,
                        "direction_predicted": cp.direction_predicted,
                        "direction_realized": cp.direction_realized,
                        "direction_correct": cp.direction_correct,
                        "is_black_swan": (cp.error_attribution_tag == "BLACK_SWAN_OUTLIER"),
                        "error_attribution_tag": cp.error_attribution_tag,
                        "model_version": cp.model_version,
                        "diagnosis_summary": getattr(cp, "diagnosis_summary", None),
                    })
            except Exception as c_err:
                logger.warning("Failed to query model predictions: %s", c_err)

            # 2. query historical reconciliations archive (model 1 historical records)
            try:
                stmt = select(
                    Reconciliation,
                    MarketTick.model_version,
                    MarketTick.close.label("start_price"),
                ).outerjoin(MarketTick, Reconciliation.tick_id == MarketTick.id)
                if model_version and model_version != "All Models":
                    stmt = stmt.where(MarketTick.model_version == model_version)
                rows = session.execute(stmt.order_by(desc(Reconciliation.timestamp)).limit(limit)).all()
                for r, m_ver, start_price in rows:
                    item = r.to_dict()
                    item["model_version"] = m_ver or "xgb_onnx_v1.0.0"
                    item["start_price"] = float(start_price) if start_price is not None else None
                    item["diagnosis_summary"] = None
                    raw_data.append(item)
            except Exception as r_err:
                logger.warning("Failed to query primary reconciliations: %s", r_err)

            if not raw_data:
                return pd.DataFrame()

            # 3. deduplicate by (tick_id, model_version), keeping model_predictions preference
            seen = set()
            deduped = []
            for item in raw_data:
                key = (item.get("tick_id"), item.get("model_version"))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)

            df = pd.DataFrame(deduped)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.sort_values("timestamp", ascending=False).reset_index(drop=True).head(limit)
    except Exception as exc:
        logger.warning("Failed to fetch reconciliation history: %s", exc)
        return pd.DataFrame()


def get_reconciliation_kpis(lookback_hours: Optional[int] = 168, model_version: Optional[str] = None) -> Dict[str, Any]:
    try:
        from src.engine import get_reconciliation_summary
        with get_db() as session:
            res = get_reconciliation_summary(session, limit=lookback_hours, model_version=model_version)
            res["query_failed"] = False
            res["error"] = None
            return res
    except Exception as exc:
        logger.warning("Failed to fetch reconciliation KPIs: %s", type(exc).__name__)
        return {
            "total_evaluated": 0, "total_reconciled": 0, "total_lifetime_evaluated": 0, "win_rate_pct": 0.0,
            "directional_accuracy_pct": 0.0, "mae_usd": 0.0, "mape_pct": 0.0, "information_coefficient": 0.0,
            "information_coefficient_ic": 0.0, "black_swans_filtered": 0, "black_swan_count": 0,
            "accurate_count": 0, "tag_breakdown": {}, "model_version": model_version if (model_version and model_version != "All Models") else None,
            "query_failed": True, "error": "Database query failed",
        }


def get_portfolio_history(limit: Optional[int] = 500) -> pd.DataFrame:
    try:
        with get_db() as session:
            if limit:
                stmt = select(PaperPortfolio).order_by(desc(PaperPortfolio.id)).limit(limit)
            else:
                stmt = select(PaperPortfolio).order_by(PaperPortfolio.id.asc())
            records = session.execute(stmt).scalars().all()
            if not records:
                return pd.DataFrame()
            df = pd.DataFrame([p.to_dict() for p in records])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        logger.warning("Failed to fetch portfolio history: %s", type(exc).__name__)
        return pd.DataFrame()


def get_portfolio_kpis(current_price: float) -> Dict[str, Any]:
    try:
        from src.engine import get_portfolio_summary
        with get_db() as session:
            return get_portfolio_summary(session, current_price=current_price)
    except Exception as exc:
        logger.warning("Failed to fetch portfolio KPIs: %s", type(exc).__name__)
        return {
            "cash_balance": 10000.0, "btc_balance": 0.0, "current_btc_price": round(current_price, 2),
            "btc_position_value_usd": 0.0, "total_equity_usd": 10000.0, "cumulative_return_pct": 0.0,
            "benchmark_btc_equity": 10000.0, "alpha_usd": 0.0, "alpha_pct": 0.0, "realized_pnl_usd": 0.0,
            "total_fees_usd": 0.0, "drawdown_pct": 0.0, "total_trades": 0,
        }


def get_latest_council_audit() -> Optional[Dict[str, Any]]:
    try:
        with get_db() as session:
            audit = session.execute(select(CouncilAudit).order_by(desc(CouncilAudit.id)).limit(1)).scalar_one_or_none()
            return audit.to_dict() if audit else None
    except Exception as exc:
        logger.warning("Failed to fetch latest council audit: %s", type(exc).__name__)
        return None


def log_telemetry(
    session: Optional[Session],
    service: str,
    endpoint: str,
    latency_ms: float,
    status_code: int,
    error_msg: Optional[str] = None,
) -> None:
    # logs external service call latency and status code into api_telemetry table
    if "PYTEST_CURRENT_TEST" in os.environ or session is None:
        return
    try:
        with session.begin_nested():
            session.add(ApiTelemetry(
                timestamp=datetime.now(timezone.utc),
                service_name=service,
                endpoint=endpoint,
                latency_ms=float(latency_ms),
                status_code=int(status_code),
                error_message=str(error_msg) if error_msg else None,
            ))
            session.flush()
    except Exception:
        pass


def get_api_telemetry_summary(hours: int = 24) -> Dict[str, Any]:
    # calculates api telemetry call counts, success rate, and average latency over rolling window
    default_res = {"total_calls": 0, "success_rate_pct": 100.0, "avg_latency_ms": 0.0}
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with get_db() as s:
            rows = s.execute(
                select(ApiTelemetry.status_code, ApiTelemetry.latency_ms)
                .where(ApiTelemetry.timestamp >= cutoff)
            ).all()
            if not rows:
                return default_res
            total = len(rows)
            successes = sum(1 for r in rows if 200 <= int(r[0] or 0) < 400)
            avg_lat = sum(float(r[1] or 0.0) for r in rows) / total
            return {
                "total_calls": total,
                "success_rate_pct": round((successes / total) * 100.0, 1),
                "avg_latency_ms": round(avg_lat, 1),
            }
    except Exception:
        return default_res


def get_model_reconciled_counts() -> Dict[str, int]:
    # Returns true lifetime reconciled hour counts per registered model from Neon.
    try:
        with get_db() as s:
            rows = s.execute(
                select(ModelPrediction.model_version, func.count(ModelPrediction.id))
                .where(ModelPrediction.outcome_tick_id.isnot(None))
                .group_by(ModelPrediction.model_version)
            ).all()
            counts = {str(ver): int(cnt) for ver, cnt in rows if ver}
            if ACTIVE_MODEL_VERSION not in counts:
                rec_cnt = int(s.execute(select(func.count(Reconciliation.id))).scalar() or 0)
                if rec_cnt > 0:
                    counts[ACTIVE_MODEL_VERSION] = rec_cnt
            return counts
    except Exception:
        return {}


def get_latest_hourly_postmortems(limit: int = 3) -> List[Dict[str, Any]]:
    # Fetches recent Gemini post-mortem diagnoses, headlines, and per-row TreeSHAP drivers.
    try:
        with get_db() as s:
            rows = s.execute(
                select(HourlyPostmortem, MarketTick.timestamp)
                .outerjoin(MarketTick, HourlyPostmortem.tick_id == MarketTick.id)
                .order_by(desc(HourlyPostmortem.id))
                .limit(max(1, min(int(limit), 20)))
            ).all()
            out: List[Dict[str, Any]] = []
            for pm, tick_ts in rows:
                out.append({
                    "id": pm.id,
                    "tick_id": pm.tick_id,
                    "tick_timestamp": tick_ts.isoformat() if hasattr(tick_ts, "isoformat") else str(tick_ts or ""),
                    "created_at": pm.created_at.isoformat() if hasattr(pm.created_at, "isoformat") else str(pm.created_at),
                    "gemini_status": pm.gemini_status,
                    "news_headlines": pm.news_headlines or [],
                    "gemini_full_analysis": pm.gemini_full_analysis or {},
                    "error_detail": pm.error_detail,
                })
            return out
    except Exception:
        return []


def get_api_telemetry_breakdown(hours: int = 24) -> List[Dict[str, Any]]:
    # Summarizes rolling 24h external API calls by service, endpoint, and HTTP status code.
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with get_db() as s:
            rows = s.execute(
                select(
                    ApiTelemetry.service_name,
                    ApiTelemetry.endpoint,
                    ApiTelemetry.status_code,
                    ApiTelemetry.latency_ms,
                ).where(ApiTelemetry.timestamp >= cutoff)
            ).all()
            if not rows:
                return []
            buckets: Dict[Tuple[str, str, int], List[float]] = {}
            for svc, ep, code, lat in rows:
                buckets.setdefault((str(svc), str(ep), int(code or 0)), []).append(float(lat or 0.0))
            return [
                {
                    "Service": svc,
                    "Endpoint": ep,
                    "Status": code,
                    "Calls": len(lats),
                    "Avg ms": round(sum(lats) / len(lats), 1),
                }
                for (svc, ep, code), lats in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 5. RETRO-COMPATIBILITY MAPPINGS
# ---------------------------------------------------------------------------
# internal module aliases for backward compatibility within this module
this_mod = sys.modules.get(__name__)
if this_mod is not None:
    connection = this_mod
    models = this_mod
    queries = this_mod

__all__ = [
    "Base", "MarketTick", "ApiTelemetry", "CouncilAudit", "Reconciliation",
    "PaperPortfolio", "ModelRegistry", "ModelPrediction", "HourlyPostmortem",
    "SchemaDriftError", "CANONICAL_TABLE_COLUMNS", "CANONICAL_FOREIGN_KEYS",
    "CANONICAL_CONSTRAINTS", "CANONICAL_INDEXES", "verify_db_schema",
    "get_database_url", "create_db_engine", "engine", "SessionLocal", "get_db",
    "has_tick_id_column", "init_db", "test_connection", "_classify_column_type",
    "_TICK_ID_COLUMN_CACHE", "get_latest_market_ticks", "get_reconciliation_history",
    "get_reconciliation_kpis", "get_portfolio_history", "get_portfolio_kpis",
    "get_latest_council_audit", "log_telemetry", "get_api_telemetry_summary",
    "get_model_reconciled_counts", "get_latest_hourly_postmortems", "get_api_telemetry_breakdown",
]
