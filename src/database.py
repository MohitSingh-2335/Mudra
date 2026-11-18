# src/database.py
"""
Mudra Database Architecture.
Contains SQLAlchemy ORM models, resilient connection pooling, schema drift verification,
and analytical queries for the quantitative trading engine and audit council.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
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
    create_engine, desc, event, inspect, select, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, attributes, defer, relationship, sessionmaker

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
    feature_schema_version = Column(String(16), nullable=False, default=FEATURE_SCHEMA_VERSION)
    market_regime = Column(String(32), nullable=True)

    predicted_return_1h = Column(Float, nullable=True)
    predicted_price_1h = Column(Float, nullable=True)
    model_confidence = Column(Float, nullable=True)
    features = Column(JSON, nullable=True)

    council_audits = relationship("CouncilAudit", back_populates="tick", uselist=True, order_by="CouncilAudit.id.asc()")
    reconciliation = relationship("Reconciliation", back_populates="tick", uselist=False)
    paper_portfolios = relationship("PaperPortfolio", back_populates="tick", foreign_keys="PaperPortfolio.tick_id", uselist=True)
    predictions = relationship("ModelPrediction", back_populates="tick", foreign_keys="ModelPrediction.tick_id", uselist=True)
    outcome_predictions = relationship("ModelPrediction", back_populates="outcome_tick", foreign_keys="ModelPrediction.outcome_tick_id", uselist=True)
    investigations = relationship("PredictionInvestigation", back_populates="tick", foreign_keys="PredictionInvestigation.tick_id", uselist=True)

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
    retry_count = Column(Integer, nullable=False, default=0)
    rate_limit_remaining = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=True, default=0)
    cost_estimate_usd = Column(Float, nullable=True, default=0.0)
    error_message = Column(Text, nullable=True)


class CouncilAudit(Base):
    __tablename__ = "council_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), index=True, default=lambda: datetime.now(timezone.utc), nullable=False)
    tick_id = Column(Integer, ForeignKey("market_ticks.id"), nullable=True, index=True)
    deliberation_latency_ms = Column(Float, nullable=True)
    strategist_action = Column(String(16), nullable=False)
    strategist_conviction = Column(Integer, nullable=False)
    rag_context_summary = Column(Text, nullable=True)
    risk_verdict = Column(String(16), nullable=False)
    veto_reasons = Column(JSON, nullable=True)
    kelly_fraction = Column(Float, nullable=False)
    final_decision = Column(String(16), nullable=False)
    veto_attribution = Column(String(32), nullable=True)
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
    training_dataset_path = Column(String(256), nullable=True)
    training_dataset_sha256 = Column(String(64), nullable=True)
    training_row_count = Column(Integer, nullable=True)
    split_metadata = Column(JSON, nullable=True)
    evaluation_metrics = Column(JSON, nullable=True)
    code_revision_sha = Column(String(64), nullable=True)
    current_status = Column(String(24), nullable=False)
    lifecycle_seq = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    lifecycle_events = relationship("ModelLifecycleEvent", primaryjoin="ModelRegistry.model_version == ModelLifecycleEvent.model_version", foreign_keys="ModelLifecycleEvent.model_version", back_populates="model", order_by="ModelLifecycleEvent.lifecycle_seq.asc()")
    predictions = relationship("ModelPrediction", primaryjoin="ModelRegistry.model_version == ModelPrediction.model_version", foreign_keys="ModelPrediction.model_version", back_populates="model", order_by="ModelPrediction.id.asc()")
    investigations = relationship("PredictionInvestigation", primaryjoin="ModelRegistry.model_version == PredictionInvestigation.model_version", foreign_keys="PredictionInvestigation.model_version", back_populates="model", order_by="PredictionInvestigation.id.asc()")

    __table_args__ = (
        UniqueConstraint("model_version", name="uq_model_registry_model_version"),
        CheckConstraint("current_status IN ('ACTIVE', 'CANDIDATE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK') AND lifecycle_seq >= 1", name="chk_model_registry_status"),
        Index("uq_model_registry_single_active", "current_status", unique=True, postgresql_where=text("current_status = 'ACTIVE'"), sqlite_where=text("current_status = 'ACTIVE'")),
    )


class ModelLifecycleEvent(Base):
    __tablename__ = "model_lifecycle_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(64), nullable=False)
    lifecycle_seq = Column(Integer, nullable=False)
    event_type = Column(String(32), nullable=False)
    from_status = Column(String(24), nullable=True)
    to_status = Column(String(24), nullable=False)
    related_model_version = Column(String(64), nullable=True)
    actor = Column(String(64), nullable=False)
    reason = Column(Text, nullable=False)
    evidence_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    model = relationship("ModelRegistry", primaryjoin="ModelLifecycleEvent.model_version == ModelRegistry.model_version", foreign_keys=[model_version], back_populates="lifecycle_events")
    related_model = relationship("ModelRegistry", primaryjoin="ModelLifecycleEvent.related_model_version == ModelRegistry.model_version", foreign_keys=[related_model_version])

    __table_args__ = (
        ForeignKeyConstraint(["model_version"], ["model_registry.model_version"], name="fk_mle_model_version"),
        ForeignKeyConstraint(["related_model_version"], ["model_registry.model_version"], name="fk_mle_related_model_version"),
        UniqueConstraint("model_version", "lifecycle_seq", name="uq_mle_model_version_seq"),
        CheckConstraint(
            "lifecycle_seq >= 1 AND event_type IN ('REGISTERED', 'EVALUATED', 'REJECTED', 'PROMOTED', 'SUPERSEDED', 'ROLLED_BACK', 'ARCHIVED') "
            "AND to_status IN ('ACTIVE', 'CANDIDATE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK') "
            "AND (from_status IS NULL OR from_status IN ('ACTIVE', 'CANDIDATE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK'))",
            name="chk_lifecycle_event_vocabulary",
        ),
        Index("ix_model_lifecycle_events_model_version", "model_version"),
        Index("uq_mle_single_registered_per_model", "model_version", unique=True, postgresql_where=text("event_type = 'REGISTERED'"), sqlite_where=text("event_type = 'REGISTERED'")),
        Index("uq_mle_single_evaluated_per_model", "model_version", unique=True, postgresql_where=text("event_type = 'EVALUATED'"), sqlite_where=text("event_type = 'EVALUATED'")),
    )


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tick_id = Column(Integer, nullable=False)
    model_version = Column(String(64), nullable=False)
    run_mode = Column(String(24), nullable=False)
    is_canonical = Column(Boolean, nullable=False, default=True)
    is_primary = Column(Boolean, nullable=False, default=False)
    parent_prediction_id = Column(Integer, nullable=True)
    feature_schema_version = Column(String(32), nullable=False, default=FEATURE_SCHEMA_VERSION)
    predicted_return_pct = Column(Float, nullable=False)
    predicted_price_1h = Column(Float, nullable=False)
    direction_predicted = Column(String(16), nullable=False)
    direction_prob_up = Column(Float, nullable=False)
    clears_hurdle = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    outcome_tick_id = Column(Integer, nullable=True)
    evaluation_policy_version = Column(String(32), nullable=True)
    actual_price_at_t_plus_1 = Column(Float, nullable=True)
    actual_return_pct = Column(Float, nullable=True)
    direction_realized = Column(String(16), nullable=True)
    direction_correct = Column(Boolean, nullable=True)
    residual_error = Column(Float, nullable=True)
    absolute_error_usd = Column(Float, nullable=True)
    percentage_error = Column(Float, nullable=True)
    is_black_swan = Column(Boolean, nullable=True)
    error_attribution_tag = Column(String(64), nullable=True)
    reconciled_at = Column(DateTime(timezone=True), nullable=True)

    tick = relationship("MarketTick", foreign_keys=[tick_id], back_populates="predictions")
    outcome_tick = relationship("MarketTick", foreign_keys=[outcome_tick_id], back_populates="outcome_predictions")
    model = relationship("ModelRegistry", primaryjoin="ModelPrediction.model_version == ModelRegistry.model_version", foreign_keys=[model_version], back_populates="predictions")
    parent_prediction = relationship("ModelPrediction", remote_side=[id], foreign_keys=[parent_prediction_id], backref="child_predictions")
    investigations = relationship("PredictionInvestigation", foreign_keys="PredictionInvestigation.prediction_id", back_populates="prediction")

    __table_args__ = (
        ForeignKeyConstraint(["tick_id"], ["market_ticks.id"], name="fk_mp_tick_id"),
        ForeignKeyConstraint(["model_version"], ["model_registry.model_version"], name="fk_mp_model_version"),
        ForeignKeyConstraint(["parent_prediction_id"], ["model_predictions.id"], name="fk_mp_parent_prediction_id"),
        ForeignKeyConstraint(["outcome_tick_id"], ["market_ticks.id"], name="fk_mp_outcome_tick_id"),
        CheckConstraint(
            "((run_mode = 'PRIMARY' AND is_primary IS TRUE AND is_canonical IS TRUE AND parent_prediction_id IS NULL) "
            "OR (run_mode = 'SHADOW' AND is_primary IS FALSE AND is_canonical IS TRUE AND parent_prediction_id IS NULL) "
            "OR (run_mode = 'FORCED_REANALYSIS' AND is_primary IS FALSE AND is_canonical IS FALSE AND parent_prediction_id IS NOT NULL)) "
            "AND predicted_return_pct NOT IN ('NaN', 'Infinity', '-Infinity') "
            "AND predicted_price_1h NOT IN ('NaN', 'Infinity', '-Infinity') "
            "AND direction_prob_up NOT IN ('NaN', 'Infinity', '-Infinity') "
            "AND predicted_price_1h >= 1.0 "
            "AND direction_prob_up >= 0.0 AND direction_prob_up <= 1.0 "
            "AND direction_predicted IN ('UP', 'DOWN') "
            "AND direction_predicted = (CASE WHEN predicted_return_pct >= 0.0 THEN 'UP' ELSE 'DOWN' END)",
            name="chk_prediction_mode_invariants",
        ),
        CheckConstraint(
            "(outcome_tick_id IS NULL AND evaluation_policy_version IS NULL AND actual_price_at_t_plus_1 IS NULL "
            "AND actual_return_pct IS NULL AND direction_realized IS NULL AND direction_correct IS NULL "
            "AND residual_error IS NULL AND absolute_error_usd IS NULL AND percentage_error IS NULL "
            "AND is_black_swan IS NULL AND error_attribution_tag IS NULL AND reconciled_at IS NULL) "
            "OR (outcome_tick_id IS NOT NULL AND evaluation_policy_version = 'binary_exact_t1_v2' "
            "AND actual_price_at_t_plus_1 IS NOT NULL AND actual_return_pct IS NOT NULL "
            "AND direction_realized IS NOT NULL AND direction_correct IS NOT NULL "
            "AND residual_error IS NOT NULL AND absolute_error_usd IS NOT NULL "
            "AND percentage_error IS NOT NULL AND is_black_swan IS NOT NULL "
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
        Index("uq_model_predictions_canonical_tick_version", "tick_id", "model_version", unique=True, postgresql_where=text("is_canonical IS TRUE"), sqlite_where=text("is_canonical IS TRUE")),
        Index("uq_model_predictions_primary_tick", "tick_id", unique=True, postgresql_where=text("is_primary IS TRUE"), sqlite_where=text("is_primary IS TRUE")),
        Index("ix_model_predictions_tick_id", "tick_id"),
        Index("ix_model_predictions_model_version", "model_version"),
    )


class PredictionInvestigation(Base):
    __tablename__ = "prediction_investigations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tick_id = Column(Integer, nullable=False)
    prediction_id = Column(Integer, nullable=True)
    model_version = Column(String(64), nullable=False)
    headline = Column(String(512), nullable=False)
    headline_source_url = Column(Text, nullable=True)
    headline_published_at = Column(DateTime(timezone=True), nullable=True)
    suspected_cause_category = Column(String(64), nullable=False)
    hypothesis_notes = Column(Text, nullable=False)
    supporting_evidence_json = Column(JSON, nullable=True)
    diagnostic_confidence = Column(Float, nullable=True)
    unresolved_questions = Column(Text, nullable=True)
    verification_status = Column(String(32), nullable=False, default="UNVERIFIED_HYPOTHESIS")
    verified_ground_truth_label = Column(String(64), nullable=True)
    verified_by = Column(String(64), nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    revision_history_json = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    tick = relationship("MarketTick", foreign_keys=[tick_id], back_populates="investigations")
    prediction = relationship("ModelPrediction", foreign_keys=[prediction_id], back_populates="investigations")
    model = relationship("ModelRegistry", primaryjoin="PredictionInvestigation.model_version == ModelRegistry.model_version", foreign_keys=[model_version], back_populates="investigations")

    __table_args__ = (
        ForeignKeyConstraint(["tick_id"], ["market_ticks.id"], name="fk_pi_tick_id"),
        ForeignKeyConstraint(["prediction_id"], ["model_predictions.id"], name="fk_pi_prediction_id"),
        ForeignKeyConstraint(["model_version"], ["model_registry.model_version"], name="fk_pi_model_version"),
        CheckConstraint(
            "verification_status IN ('UNVERIFIED_HYPOTHESIS', 'CONFIRMED_CAUSE', 'REFUTED', 'APPROVED_FOR_TRAINING') "
            "AND (diagnostic_confidence IS NULL OR (diagnostic_confidence NOT IN ('NaN', 'Infinity', '-Infinity') "
            "AND diagnostic_confidence >= 0.0 AND diagnostic_confidence <= 1.0))",
            name="chk_pi_verification_status",
        ),
        Index("ix_prediction_investigations_tick_id", "tick_id"),
        Index("ix_prediction_investigations_model_version", "model_version"),
    )


# ---------------------------------------------------------------------------
# 2. SCHEMA DRIFT VERIFICATION & SPECS
# ---------------------------------------------------------------------------

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
        logger.warning("Failed to ensure canonical function %s: %s", spec.name, exc)
        return False


def ensure_canonical_trigger(conn: Any, spec: TriggerSpec) -> bool:
    try:
        conn.execute(text(spec.create_sql))
        return True
    except Exception as exc:
        logger.warning("Failed to ensure canonical trigger %s: %s", spec.name, exc)
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
    except Exception:
        pass
    if SECRETS_FILE.exists():
        try:
            data = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            found = _find_postgres_url(data)
            if found:
                return _normalize_db_url(found)
        except Exception:
            pass
    fallback = ROOT_DIR / "data" / "mudra_operational.db"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{fallback.as_posix()}"


def create_db_engine(db_url: Optional[str] = None) -> Engine:
    url = db_url or get_database_url()
    if "sqlite" in url:
        return create_engine(url, connect_args={"check_same_thread": False}, echo=False)
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10, pool_recycle=300, connect_args={"connect_timeout": 10}, echo=False)


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


def has_tick_id_column(session: Session) -> bool:
    try:
        bind = session.get_bind()
        bind_id = id(bind)
        if bind_id in _TICK_ID_COLUMN_CACHE:
            return _TICK_ID_COLUMN_CACHE[bind_id]
        inspector = inspect(session.connection() if hasattr(session, "connection") else bind)
        if inspector.has_table("paper_portfolio"):
            present = "tick_id" in {c["name"] for c in inspector.get_columns("paper_portfolio")}
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


event.listen(Session, "do_orm_execute", _defer_missing_tick_id_on_legacy_portfolio)
event.listen(PaperPortfolio, "load", _populate_deferred_tick_id_on_load)


def init_db(target_engine: Optional[Engine] = None) -> None:
    target = target_engine or engine
    Base.metadata.create_all(bind=target)
    _TICK_ID_COLUMN_CACHE[id(target)] = True


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
    try:
        with get_db() as session:
            stmt = select(Reconciliation, MarketTick.model_version).outerjoin(MarketTick, Reconciliation.tick_id == MarketTick.id)
            if model_version and model_version != "All Models":
                stmt = stmt.where(MarketTick.model_version == model_version)
            rows = session.execute(stmt.order_by(desc(Reconciliation.timestamp)).limit(limit)).all()

            data = []
            for r, m_ver in rows:
                item = r.to_dict()
                item["model_version"] = m_ver
                data.append(item)

            cand_stmt = select(ModelPrediction, MarketTick.timestamp.label("tick_ts")).join(MarketTick, ModelPrediction.tick_id == MarketTick.id).where(ModelPrediction.outcome_tick_id.isnot(None), ModelPrediction.is_primary.is_(False))
            if model_version and model_version != "All Models":
                cand_stmt = cand_stmt.where(ModelPrediction.model_version == model_version)
            for cp, tick_ts in session.execute(cand_stmt.order_by(desc(ModelPrediction.reconciled_at)).limit(limit)).all():
                ts = cp.reconciled_at or tick_ts
                data.append({
                    "id": cp.id, "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "tick_id": cp.tick_id, "predicted_price": cp.predicted_price_1h,
                    "actual_price_at_t_plus_1": cp.actual_price_at_t_plus_1, "predicted_return_pct": cp.predicted_return_pct,
                    "actual_return_pct": cp.actual_return_pct, "residual_error": cp.residual_error,
                    "absolute_error_usd": cp.absolute_error_usd, "percentage_error": cp.percentage_error,
                    "direction_predicted": cp.direction_predicted, "direction_realized": cp.direction_realized,
                    "direction_correct": cp.direction_correct, "is_black_swan": cp.is_black_swan,
                    "error_attribution_tag": cp.error_attribution_tag, "model_version": cp.model_version,
                })

            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.sort_values("timestamp", ascending=False).reset_index(drop=True).head(limit)
    except Exception as exc:
        logger.warning("Failed to fetch reconciliation history: %s", type(exc).__name__)
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


def _validate_status_and_conf(status: str, confidence: Optional[float] = None) -> None:
    if not isinstance(status, str) or status.strip() not in ALLOWED_VERIFICATION_STATUSES:
        raise ValueError(f"Invalid verification_status '{status}'.")
    if confidence is not None:
        try:
            val = float(confidence)
            if not math.isfinite(val) or val < 0.0 or val > 1.0:
                raise ValueError
        except Exception as err:
            raise ValueError(f"Diagnostic confidence must be between 0.0 and 1.0, got {confidence}.") from err


def _normalize_datetime(dt_val: Optional[Union[str, datetime]]) -> Optional[datetime]:
    if dt_val is None or isinstance(dt_val, datetime):
        return dt_val.replace(tzinfo=timezone.utc) if isinstance(dt_val, datetime) and not dt_val.tzinfo else dt_val
    try:
        parsed = datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception as err:
        raise ValueError(f"Invalid ISO datetime string: {dt_val}") from err


def record_prediction_investigation(
    tick_id: int, model_version: str, headline: str, suspected_cause_category: str,
    hypothesis_notes: str, supporting_evidence_json: Optional[Dict[str, Any]] = None,
    diagnostic_confidence: Optional[float] = None, unresolved_questions: Optional[str] = None,
    verification_status: str = "UNVERIFIED_HYPOTHESIS", headline_source_url: Optional[str] = None,
    headline_published_at: Optional[Union[str, datetime]] = None, prediction_id: Optional[int] = None,
    author: str = "SYSTEM", session: Optional[Session] = None,
) -> Dict[str, Any]:
    _validate_status_and_conf(verification_status, diagnostic_confidence)
    for name, val in [("headline", headline), ("suspected_cause_category", suspected_cause_category), ("hypothesis_notes", hypothesis_notes), ("model_version", model_version)]:
        if not val or not val.strip():
            raise ValueError(f"{name} cannot be empty.")

    pub_at = _normalize_datetime(headline_published_at)

    def _execute(sess: Session) -> Dict[str, Any]:
        if not sess.execute(select(MarketTick.id).where(MarketTick.id == tick_id)).scalar_one_or_none():
            raise ValueError(f"MarketTick {tick_id} does not exist.")
        if prediction_id is not None and not sess.execute(select(ModelPrediction.id).where(ModelPrediction.id == prediction_id)).scalar_one_or_none():
            raise ValueError(f"ModelPrediction {prediction_id} does not exist.")

        now_utc = datetime.now(timezone.utc)
        inv = PredictionInvestigation(
            tick_id=tick_id, prediction_id=prediction_id, model_version=model_version.strip(),
            headline=headline.strip(), headline_source_url=headline_source_url.strip() if headline_source_url else None,
            headline_published_at=pub_at, suspected_cause_category=suspected_cause_category.strip(),
            hypothesis_notes=hypothesis_notes.strip(), supporting_evidence_json=supporting_evidence_json or {},
            diagnostic_confidence=float(diagnostic_confidence) if diagnostic_confidence is not None else None,
            unresolved_questions=unresolved_questions.strip() if unresolved_questions else None,
            verification_status=verification_status, created_at=now_utc, updated_at=now_utc,
            revision_history_json=[{
                "timestamp": now_utc.isoformat(), "author": author, "action": "RECORDED",
                "verification_status": verification_status, "diagnostic_confidence": diagnostic_confidence,
                "suspected_cause_category": suspected_cause_category, "notes": hypothesis_notes[:250],
            }],
        )
        sess.add(inv)
        sess.flush()
        return inv.to_dict()

    if session:
        return _execute(session)
    with get_db() as new_session:
        return _execute(new_session)


def get_prediction_investigations(
    limit: int = 50, model_version: Optional[str] = None, verification_status: Optional[str] = None,
    headline_query: Optional[str] = None, tick_id: Optional[int] = None, session: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    def _execute(sess: Session) -> List[Dict[str, Any]]:
        stmt = select(PredictionInvestigation).order_by(desc(PredictionInvestigation.id))
        if model_version and model_version != "All Models":
            stmt = stmt.where(PredictionInvestigation.model_version == model_version)
        if verification_status and verification_status != "ALL":
            stmt = stmt.where(PredictionInvestigation.verification_status == verification_status)
        if headline_query and headline_query.strip():
            stmt = stmt.where(PredictionInvestigation.headline.ilike(f"%{headline_query.strip()}%"))
        if tick_id is not None:
            stmt = stmt.where(PredictionInvestigation.tick_id == tick_id)
        return [row.to_dict() for row in sess.execute(stmt.limit(limit)).scalars().all()]

    if session:
        return _execute(session)
    with get_db() as s:
        return _execute(s)


def get_prediction_investigation_details(investigation_id: int, session: Optional[Session] = None) -> Optional[Dict[str, Any]]:
    def _execute(sess: Session) -> Optional[Dict[str, Any]]:
        inv = sess.execute(select(PredictionInvestigation).where(PredictionInvestigation.id == investigation_id)).scalar_one_or_none()
        return inv.to_dict() if inv is not None else None

    if session:
        return _execute(session)
    with get_db() as s:
        return _execute(s)


def update_prediction_investigation_status(
    investigation_id: int, new_status: str, verified_by: str, notes: Optional[str] = None,
    ground_truth_label: Optional[str] = None, new_confidence: Optional[float] = None, session: Optional[Session] = None,
) -> Dict[str, Any]:
    _validate_status_and_conf(new_status, new_confidence)
    if not verified_by or not verified_by.strip():
        raise ValueError("verified_by cannot be empty.")

    def _execute(sess: Session) -> Dict[str, Any]:
        inv = sess.execute(select(PredictionInvestigation).where(PredictionInvestigation.id == investigation_id)).scalar_one_or_none()
        if inv is None:
            raise ValueError(f"Investigation {investigation_id} not found.")

        now_utc = datetime.now(timezone.utc)
        history = list(inv.revision_history_json or [])
        history.append({
            "timestamp": now_utc.isoformat(), "author": verified_by.strip(),
            "action": f"STATUS_CHANGED_{new_status}", "from_status": inv.verification_status,
            "to_status": new_status, "notes": notes[:250] if notes else None,
        })
        inv.verification_status = new_status
        inv.verified_by = verified_by.strip()
        inv.verified_at = now_utc
        if ground_truth_label:
            inv.verified_ground_truth_label = ground_truth_label.strip()
        if new_confidence is not None:
            inv.diagnostic_confidence = float(new_confidence)
        inv.revision_history_json = history
        inv.updated_at = now_utc
        sess.flush()
        return inv.to_dict()

    if session:
        return _execute(session)
    with get_db() as s:
        return _execute(s)


def get_training_approved_investigations(limit: int = 100, session: Optional[Session] = None) -> List[Dict[str, Any]]:
    return get_prediction_investigations(limit=limit, verification_status="APPROVED_FOR_TRAINING", session=session)


# ---------------------------------------------------------------------------
# 5. RETRO-COMPATIBILITY MAPPINGS
# ---------------------------------------------------------------------------
# Allows existing callers to reference submodules during migration
connection = sys.modules[__name__]
models = sys.modules[__name__]
queries = sys.modules[__name__]

sys.modules["src.database.models"] = sys.modules[__name__]
sys.modules["src.database.connection"] = sys.modules[__name__]
sys.modules["src.database.queries"] = sys.modules[__name__]

__all__ = [
    "Base", "MarketTick", "ApiTelemetry", "CouncilAudit", "Reconciliation",
    "PaperPortfolio", "ModelRegistry", "ModelLifecycleEvent", "ModelPrediction",
    "PredictionInvestigation", "SchemaDriftError", "FunctionSpec", "TriggerSpec",
    "CANONICAL_TABLE_COLUMNS", "CANONICAL_FOREIGN_KEYS", "CANONICAL_CONSTRAINTS",
    "CANONICAL_INDEXES", "CANONICAL_FUNCTIONS", "CANONICAL_TRIGGERS", "normalize_sql",
    "normalize_pg_predicate", "ensure_canonical_function", "ensure_canonical_trigger",
    "verify_db_schema", "get_database_url", "create_db_engine", "engine",
    "SessionLocal", "get_db", "has_tick_id_column", "init_db",
    "test_connection", "_classify_column_type", "_TICK_ID_COLUMN_CACHE",
    "get_latest_market_ticks", "get_reconciliation_history", "get_reconciliation_kpis",
    "get_portfolio_history", "get_portfolio_kpis", "get_latest_council_audit",
    "ALLOWED_VERIFICATION_STATUSES", "record_prediction_investigation", "get_prediction_investigations",
    "get_prediction_investigation_details", "update_prediction_investigation_status",
    "get_training_approved_investigations",
]
