from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from config import ACTIVE_MODEL_VERSION, FEATURE_SCHEMA_VERSION

__all__ = [
    "Base",
    "MarketTick",
    "ApiTelemetry",
    "CouncilAudit",
    "Reconciliation",
    "PaperPortfolio",
    "ModelRegistry",
    "ModelLifecycleEvent",
    "ModelPrediction",
    "PredictionInvestigation",
]


class Base(DeclarativeBase):
    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for col in self.__table__.columns:
            val = getattr(self, col.name)
            res[col.name] = val.isoformat() if hasattr(val, "isoformat") else val
        return res

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
        Index(
            "uq_paper_portfolio_tick_id",
            "tick_id",
            unique=True,
            postgresql_where=text("tick_id IS NOT NULL"),
            sqlite_where=text("tick_id IS NOT NULL"),
        ),
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

    lifecycle_events = relationship(
        "ModelLifecycleEvent",
        primaryjoin="ModelRegistry.model_version == ModelLifecycleEvent.model_version",
        foreign_keys="ModelLifecycleEvent.model_version",
        back_populates="model",
        order_by="ModelLifecycleEvent.lifecycle_seq.asc()",
    )
    predictions = relationship(
        "ModelPrediction",
        primaryjoin="ModelRegistry.model_version == ModelPrediction.model_version",
        foreign_keys="ModelPrediction.model_version",
        back_populates="model",
        order_by="ModelPrediction.id.asc()",
    )
    investigations = relationship(
        "PredictionInvestigation",
        primaryjoin="ModelRegistry.model_version == PredictionInvestigation.model_version",
        foreign_keys="PredictionInvestigation.model_version",
        back_populates="model",
        order_by="PredictionInvestigation.id.asc()",
    )

    __table_args__ = (
        UniqueConstraint("model_version", name="uq_model_registry_model_version"),
        CheckConstraint(
            "current_status IN ('ACTIVE', 'CANDIDATE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK') AND lifecycle_seq >= 1",
            name="chk_model_registry_status",
        ),
        Index(
            "uq_model_registry_single_active",
            "current_status",
            unique=True,
            postgresql_where=text("current_status = 'ACTIVE'"),
            sqlite_where=text("current_status = 'ACTIVE'"),
        ),
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

    model = relationship(
        "ModelRegistry",
        primaryjoin="ModelLifecycleEvent.model_version == ModelRegistry.model_version",
        foreign_keys=[model_version],
        back_populates="lifecycle_events",
    )
    related_model = relationship(
        "ModelRegistry",
        primaryjoin="ModelLifecycleEvent.related_model_version == ModelRegistry.model_version",
        foreign_keys=[related_model_version],
    )

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
    model = relationship(
        "ModelRegistry",
        primaryjoin="ModelPrediction.model_version == ModelRegistry.model_version",
        foreign_keys=[model_version],
        back_populates="predictions",
    )
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
    model = relationship(
        "ModelRegistry",
        primaryjoin="PredictionInvestigation.model_version == ModelRegistry.model_version",
        foreign_keys=[model_version],
        back_populates="investigations",
    )

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
