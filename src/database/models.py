"""
SQLAlchemy ORM models for Mudra Cloud Ledger, Observability, Model Governance,
Prediction Invariants, and Post-Mortem Diagnostics on Neon PostgreSQL.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    ForeignKeyConstraint,
    CheckConstraint,
    UniqueConstraint,
    Text,
    Index,
    text,
)
from sqlalchemy.orm import DeclarativeBase, relationship

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
    """Base declarative class for Mudra SQLAlchemy models."""
    pass


class MarketTick(Base):
    """
    Stores raw OHLCV market candles, model inference outputs,
    model version lineage, and all 39 engineered features as JSON.
    """
    __tablename__ = "market_ticks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # Model Lineage & Invariants
    model_version = Column(String(32), nullable=False, default="xgb_onnx_v1.0.0")
    feature_schema_version = Column(String(16), nullable=False, default="v1_39features")
    market_regime = Column(String(32), nullable=True)  # e.g., HIGH_VOL_CHOP, BULL_TREND

    # Model Inference
    predicted_return_1h = Column(Float, nullable=True)
    predicted_price_1h = Column(Float, nullable=True)
    model_confidence = Column(Float, nullable=True)
    features = Column(JSON, nullable=True)

    # Relationships
    council_audits = relationship(
        "CouncilAudit",
        back_populates="tick",
        uselist=True,
        order_by="CouncilAudit.id.asc()",
    )
    reconciliation = relationship("Reconciliation", back_populates="tick", uselist=False)
    paper_portfolios = relationship(
        "PaperPortfolio",
        back_populates="tick",
        foreign_keys="PaperPortfolio.tick_id",
        uselist=True,
        order_by="PaperPortfolio.id.asc()",
    )
    predictions = relationship(
        "ModelPrediction",
        back_populates="tick",
        foreign_keys="ModelPrediction.tick_id",
        uselist=True,
        order_by="ModelPrediction.id.asc()",
    )
    outcome_predictions = relationship(
        "ModelPrediction",
        back_populates="outcome_tick",
        foreign_keys="ModelPrediction.outcome_tick_id",
        uselist=True,
        order_by="ModelPrediction.id.asc()",
    )
    investigations = relationship(
        "PredictionInvestigation",
        back_populates="tick",
        foreign_keys="PredictionInvestigation.tick_id",
        uselist=True,
        order_by="PredictionInvestigation.id.asc()",
    )

    @property
    def council_audit(self) -> Optional["CouncilAudit"]:
        """Backward-compatible single-audit accessor returning the latest audit for this tick."""
        return self.council_audits[-1] if self.council_audits else None

    def __repr__(self) -> str:
        return f"<MarketTick(id={self.id}, time='{self.timestamp}', close={self.close}, pred_return={self.predicted_return_1h})>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else (str(self.timestamp) if self.timestamp else None),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "market_regime": self.market_regime,
            "predicted_return_1h": self.predicted_return_1h,
            "predicted_price_1h": self.predicted_price_1h,
            "model_confidence": self.model_confidence,
            "features": self.features,
        }


class ApiTelemetry(Base):
    """
    Observability table tracking all external API calls, latency, rate limits, and cost quotas.
    """
    __tablename__ = "api_telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True),
        index=True,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    service_name = Column(String(64), nullable=False, index=True)
    endpoint = Column(String(256), nullable=False)
    latency_ms = Column(Float, nullable=False)
    status_code = Column(Integer, nullable=False)
    retry_count = Column(Integer, nullable=False, default=0)
    rate_limit_remaining = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=True, default=0)
    cost_estimate_usd = Column(Float, nullable=True, default=0.0)
    error_message = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<ApiTelemetry(id={self.id}, service='{self.service_name}', endpoint='{self.endpoint}', latency={self.latency_ms:.1f}ms)>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else (str(self.timestamp) if self.timestamp else None),
            "service_name": self.service_name,
            "endpoint": self.endpoint,
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "retry_count": self.retry_count,
            "rate_limit_remaining": self.rate_limit_remaining,
            "tokens_used": self.tokens_used,
            "cost_estimate_usd": self.cost_estimate_usd,
            "error_message": self.error_message,
        }


class CouncilAudit(Base):
    """
    Immutable audit trail for Multi-Agent deliberations with Veto Quality attribution.
    """
    __tablename__ = "council_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True),
        index=True,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    tick_id = Column(Integer, ForeignKey("market_ticks.id"), nullable=True, index=True)
    deliberation_latency_ms = Column(Float, nullable=True)
    strategist_action = Column(String(16), nullable=False)
    strategist_conviction = Column(Integer, nullable=False)
    rag_context_summary = Column(Text, nullable=True)
    risk_verdict = Column(String(16), nullable=False)
    veto_reasons = Column(JSON, nullable=True)
    kelly_fraction = Column(Float, nullable=False)
    final_decision = Column(String(16), nullable=False)
    veto_attribution = Column(String(32), nullable=True)  # SAVED_CAPITAL, OPPORTUNITY_COST, N/A
    execution_notes = Column(Text, nullable=True)

    # Relationships
    tick = relationship("MarketTick", back_populates="council_audits")

    def __repr__(self) -> str:
        return f"<CouncilAudit(id={self.id}, tick_id={self.tick_id}, action='{self.strategist_action}', verdict='{self.risk_verdict}', final='{self.final_decision}')>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else (str(self.timestamp) if self.timestamp else None),
            "tick_id": self.tick_id,
            "deliberation_latency_ms": self.deliberation_latency_ms,
            "strategist_action": self.strategist_action,
            "strategist_conviction": self.strategist_conviction,
            "rag_context_summary": self.rag_context_summary,
            "risk_verdict": self.risk_verdict,
            "veto_reasons": self.veto_reasons,
            "kelly_fraction": self.kelly_fraction,
            "final_decision": self.final_decision,
            "veto_attribution": self.veto_attribution,
            "execution_notes": self.execution_notes,
        }


class Reconciliation(Base):
    """
    Delayed ground-truth reconciliation ledger with residual error decomposition
    and Black Swan outlier isolation for clean retraining datasets.
    """
    __tablename__ = "reconciliations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True),
        index=True,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    tick_id = Column(Integer, ForeignKey("market_ticks.id"), nullable=False, unique=True, index=True)
    predicted_price = Column(Float, nullable=False)
    actual_price_at_t_plus_1 = Column(Float, nullable=False)
    predicted_return_pct = Column(Float, nullable=False)
    actual_return_pct = Column(Float, nullable=False)
    residual_error = Column(Float, nullable=False)  # actual - predicted
    absolute_error_usd = Column(Float, nullable=False)
    percentage_error = Column(Float, nullable=False)
    direction_predicted = Column(String(16), nullable=False)
    direction_realized = Column(String(16), nullable=False)
    direction_correct = Column(Boolean, nullable=False)
    is_black_swan = Column(Boolean, nullable=False, default=False)  # |error| > 3-sigma
    error_attribution_tag = Column(String(64), nullable=False)

    # Relationships
    tick = relationship("MarketTick", back_populates="reconciliation")

    def __repr__(self) -> str:
        return f"<Reconciliation(id={self.id}, tick_id={self.tick_id}, dir_correct={self.direction_correct}, tag='{self.error_attribution_tag}')>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else (str(self.timestamp) if self.timestamp else None),
            "tick_id": self.tick_id,
            "predicted_price": self.predicted_price,
            "actual_price_at_t_plus_1": self.actual_price_at_t_plus_1,
            "predicted_return_pct": self.predicted_return_pct,
            "actual_return_pct": self.actual_return_pct,
            "residual_error": self.residual_error,
            "absolute_error_usd": self.absolute_error_usd,
            "percentage_error": self.percentage_error,
            "direction_predicted": self.direction_predicted,
            "direction_realized": self.direction_realized,
            "direction_correct": self.direction_correct,
            "is_black_swan": self.is_black_swan,
            "error_attribution_tag": self.error_attribution_tag,
        }


class PaperPortfolio(Base):
    """
    Forward $10,000 Paper Trading portfolio ledger tracking Realized/Unrealized PnL,
    drawdowns, transaction fees, slippage, and Alpha vs. Buy & Hold benchmark.
    """
    __tablename__ = "paper_portfolio"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tick_id = Column(Integer, ForeignKey("market_ticks.id"), nullable=True)
    timestamp = Column(
        DateTime(timezone=True),
        index=True,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    action = Column(String(16), nullable=False)  # INITIAL_DEPOSIT, BUY, SELL, HOLD
    btc_amount = Column(Float, nullable=False, default=0.0)
    execution_price = Column(Float, nullable=False, default=0.0)
    fee_deducted_usd = Column(Float, nullable=False, default=0.0)
    slippage_usd = Column(Float, nullable=False, default=0.0)

    # Portfolio Balances & PnL
    cash_balance = Column(Float, nullable=False)
    btc_balance = Column(Float, nullable=False)
    total_equity_usd = Column(Float, nullable=False)
    realized_pnl_usd = Column(Float, nullable=False, default=0.0)
    unrealized_pnl_usd = Column(Float, nullable=False, default=0.0)
    drawdown_pct = Column(Float, nullable=False, default=0.0)
    cumulative_return_pct = Column(Float, nullable=False)

    # Institutional Benchmark & Alpha Attribution
    benchmark_btc_equity = Column(Float, nullable=False)  # Hypothetical Buy & Hold equity
    alpha_usd = Column(Float, nullable=False, default=0.0)  # total_equity - benchmark_btc_equity
    alpha_pct = Column(Float, nullable=False, default=0.0)

    notes = Column(Text, nullable=True)

    # Relationships
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

    def __repr__(self) -> str:
        return f"<PaperPortfolio(id={self.id}, tick_id={self.tick_id}, action='{self.action}', equity=${self.total_equity_usd:.2f}, alpha=${self.alpha_usd:+.2f})>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tick_id": self.tick_id,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else (str(self.timestamp) if self.timestamp else None),
            "action": self.action,
            "btc_amount": self.btc_amount,
            "execution_price": self.execution_price,
            "fee_deducted_usd": self.fee_deducted_usd,
            "slippage_usd": self.slippage_usd,
            "cash_balance": self.cash_balance,
            "btc_balance": self.btc_balance,
            "total_equity_usd": self.total_equity_usd,
            "realized_pnl_usd": self.realized_pnl_usd,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "drawdown_pct": self.drawdown_pct,
            "cumulative_return_pct": self.cumulative_return_pct,
            "benchmark_btc_equity": self.benchmark_btc_equity,
            "alpha_usd": self.alpha_usd,
            "alpha_pct": self.alpha_pct,
            "notes": self.notes,
        }


class ModelRegistry(Base):
    """
    Authoritative model registry tracking model versions, cryptographic artifact digests,
    feature schemas, targets, training provenance, and state-machine lifecycle status.
    """
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
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
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

    def __repr__(self) -> str:
        return f"<ModelRegistry(id={self.id}, version='{self.model_version}', status='{self.current_status}', seq={self.lifecycle_seq})>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_version": self.model_version,
            "regressor_artifact_path": self.regressor_artifact_path,
            "regressor_sha256": self.regressor_sha256,
            "classifier_artifact_path": self.classifier_artifact_path,
            "classifier_sha256": self.classifier_sha256,
            "feature_schema_version": self.feature_schema_version,
            "target_definition": self.target_definition,
            "training_dataset_path": self.training_dataset_path,
            "training_dataset_sha256": self.training_dataset_sha256,
            "training_row_count": self.training_row_count,
            "split_metadata": self.split_metadata,
            "evaluation_metrics": self.evaluation_metrics,
            "code_revision_sha": self.code_revision_sha,
            "current_status": self.current_status,
            "lifecycle_seq": self.lifecycle_seq,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else (str(self.created_at) if self.created_at else None),
        }


class ModelLifecycleEvent(Base):
    """
    Append-only audit trail tracking every model lifecycle state transition,
    evaluator decisions, promotions, rollbacks, and retirement reasons.
    """
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
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
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
        ForeignKeyConstraint(
            ["model_version"],
            ["model_registry.model_version"],
            name="fk_mle_model_version",
            onupdate="NO ACTION",
            ondelete="NO ACTION",
        ),
        ForeignKeyConstraint(
            ["related_model_version"],
            ["model_registry.model_version"],
            name="fk_mle_related_model_version",
            onupdate="NO ACTION",
            ondelete="NO ACTION",
        ),
        UniqueConstraint("model_version", "lifecycle_seq", name="uq_mle_model_version_seq"),
        CheckConstraint(
            "lifecycle_seq >= 1 AND event_type IN ('REGISTERED', 'EVALUATED', 'REJECTED', 'PROMOTED', 'SUPERSEDED', 'ROLLED_BACK', 'ARCHIVED') "
            "AND to_status IN ('ACTIVE', 'CANDIDATE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK') "
            "AND (from_status IS NULL OR from_status IN ('ACTIVE', 'CANDIDATE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK'))",
            name="chk_lifecycle_event_vocabulary",
        ),
        Index("ix_model_lifecycle_events_model_version", "model_version"),
        Index(
            "uq_mle_single_registered_per_model",
            "model_version",
            unique=True,
            postgresql_where=text("event_type = 'REGISTERED'"),
            sqlite_where=text("event_type = 'REGISTERED'"),
        ),
        Index(
            "uq_mle_single_evaluated_per_model",
            "model_version",
            unique=True,
            postgresql_where=text("event_type = 'EVALUATED'"),
            sqlite_where=text("event_type = 'EVALUATED'"),
        ),
    )

    def __repr__(self) -> str:
        return f"<ModelLifecycleEvent(id={self.id}, model='{self.model_version}', seq={self.lifecycle_seq}, event='{self.event_type}', {self.from_status}->{self.to_status})>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_version": self.model_version,
            "lifecycle_seq": self.lifecycle_seq,
            "event_type": self.event_type,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "related_model_version": self.related_model_version,
            "actor": self.actor,
            "reason": self.reason,
            "evidence_json": self.evidence_json,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else (str(self.created_at) if self.created_at else None),
        }


class ModelPrediction(Base):
    """
    Granular prediction ledger tracking point-in-time forecasts, run modes (PRIMARY, SHADOW,
    FORCED_REANALYSIS), parent lineage, and immutable delayed reconciliation outcomes.
    """
    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tick_id = Column(Integer, nullable=False)
    model_version = Column(String(64), nullable=False)
    run_mode = Column(String(24), nullable=False)
    is_canonical = Column(Boolean, nullable=False, default=True)
    is_primary = Column(Boolean, nullable=False, default=False)
    parent_prediction_id = Column(Integer, nullable=True)
    feature_schema_version = Column(String(32), nullable=False, default="v1_39features")
    predicted_return_pct = Column(Float, nullable=False)
    predicted_price_1h = Column(Float, nullable=False)
    direction_predicted = Column(String(16), nullable=False)
    direction_prob_up = Column(Float, nullable=False)
    clears_hurdle = Column(Boolean, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Reconciliation Outcomes (Populated atomically strictly at T+1h)
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

    # Relationships
    tick = relationship("MarketTick", foreign_keys=[tick_id], back_populates="predictions")
    outcome_tick = relationship("MarketTick", foreign_keys=[outcome_tick_id], back_populates="outcome_predictions")
    model = relationship(
        "ModelRegistry",
        primaryjoin="ModelPrediction.model_version == ModelRegistry.model_version",
        foreign_keys=[model_version],
        back_populates="predictions",
    )
    parent_prediction = relationship(
        "ModelPrediction",
        remote_side=[id],
        foreign_keys=[parent_prediction_id],
        backref="child_predictions",
    )
    investigations = relationship(
        "PredictionInvestigation",
        foreign_keys="PredictionInvestigation.prediction_id",
        back_populates="prediction",
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tick_id"],
            ["market_ticks.id"],
            name="fk_mp_tick_id",
            onupdate="NO ACTION",
            ondelete="NO ACTION",
        ),
        ForeignKeyConstraint(
            ["model_version"],
            ["model_registry.model_version"],
            name="fk_mp_model_version",
            onupdate="NO ACTION",
            ondelete="NO ACTION",
        ),
        ForeignKeyConstraint(
            ["parent_prediction_id"],
            ["model_predictions.id"],
            name="fk_mp_parent_prediction_id",
            onupdate="NO ACTION",
            ondelete="NO ACTION",
        ),
        ForeignKeyConstraint(
            ["outcome_tick_id"],
            ["market_ticks.id"],
            name="fk_mp_outcome_tick_id",
            onupdate="NO ACTION",
            ondelete="NO ACTION",
        ),
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
            "AND actual_price_at_t_plus_1 NOT IN ('NaN', 'Infinity', '-Infinity') "
            "AND actual_return_pct NOT IN ('NaN', 'Infinity', '-Infinity') "
            "AND residual_error NOT IN ('NaN', 'Infinity', '-Infinity') "
            "AND absolute_error_usd NOT IN ('NaN', 'Infinity', '-Infinity') "
            "AND percentage_error NOT IN ('NaN', 'Infinity', '-Infinity') "
            "AND actual_price_at_t_plus_1 >= 1.0 "
            "AND direction_realized IN ('UP', 'DOWN') "
            "AND direction_realized = (CASE WHEN actual_return_pct >= 0.0 THEN 'UP' ELSE 'DOWN' END) "
            "AND direction_correct = (direction_predicted = direction_realized) "
            "AND absolute_error_usd >= 0.0 AND percentage_error >= 0.0 "
            "AND error_attribution_tag IN ('ACCURATE', 'MOMENTUM_UNDERESTIMATED', 'MOMENTUM_OVERESTIMATED', "
            "'SIDEWAYS_DRIFT', 'DIRECTION_MISALIGNED', 'BLACK_SWAN_OUTLIER'))",
            name="chk_reconciliation_all_or_nothing",
        ),
        Index(
            "uq_model_predictions_canonical_tick_version",
            "tick_id",
            "model_version",
            unique=True,
            postgresql_where=text("is_canonical IS TRUE"),
            sqlite_where=text("is_canonical IS TRUE"),
        ),
        Index(
            "uq_model_predictions_primary_tick",
            "tick_id",
            unique=True,
            postgresql_where=text("is_primary IS TRUE"),
            sqlite_where=text("is_primary IS TRUE"),
        ),
        Index("ix_model_predictions_tick_id", "tick_id"),
        Index("ix_model_predictions_model_version", "model_version"),
    )

    def __repr__(self) -> str:
        return f"<ModelPrediction(id={self.id}, tick_id={self.tick_id}, model='{self.model_version}', mode='{self.run_mode}', pred_ret={self.predicted_return_pct:.4f}%, dir='{self.direction_predicted}')>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tick_id": self.tick_id,
            "model_version": self.model_version,
            "run_mode": self.run_mode,
            "is_canonical": self.is_canonical,
            "is_primary": self.is_primary,
            "parent_prediction_id": self.parent_prediction_id,
            "feature_schema_version": self.feature_schema_version,
            "predicted_return_pct": self.predicted_return_pct,
            "predicted_price_1h": self.predicted_price_1h,
            "direction_predicted": self.direction_predicted,
            "direction_prob_up": self.direction_prob_up,
            "clears_hurdle": self.clears_hurdle,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else (str(self.created_at) if self.created_at else None),
            "outcome_tick_id": self.outcome_tick_id,
            "evaluation_policy_version": self.evaluation_policy_version,
            "actual_price_at_t_plus_1": self.actual_price_at_t_plus_1,
            "actual_return_pct": self.actual_return_pct,
            "direction_realized": self.direction_realized,
            "direction_correct": self.direction_correct,
            "residual_error": self.residual_error,
            "absolute_error_usd": self.absolute_error_usd,
            "percentage_error": self.percentage_error,
            "is_black_swan": self.is_black_swan,
            "error_attribution_tag": self.error_attribution_tag,
            "reconciled_at": self.reconciled_at.isoformat() if hasattr(self.reconciled_at, "isoformat") else (str(self.reconciled_at) if self.reconciled_at else None),
        }


class PredictionInvestigation(Base):
    """
    Qualitative post-mortem ledger preserving news headlines, macro events,
    operator hypotheses, and verified causal explanations for prediction failures.
    """
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
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    tick = relationship("MarketTick", foreign_keys=[tick_id], back_populates="investigations")
    prediction = relationship("ModelPrediction", foreign_keys=[prediction_id], back_populates="investigations")
    model = relationship(
        "ModelRegistry",
        primaryjoin="PredictionInvestigation.model_version == ModelRegistry.model_version",
        foreign_keys=[model_version],
        back_populates="investigations",
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tick_id"],
            ["market_ticks.id"],
            name="fk_pi_tick_id",
            onupdate="NO ACTION",
            ondelete="NO ACTION",
        ),
        ForeignKeyConstraint(
            ["prediction_id"],
            ["model_predictions.id"],
            name="fk_pi_prediction_id",
            onupdate="NO ACTION",
            ondelete="NO ACTION",
        ),
        ForeignKeyConstraint(
            ["model_version"],
            ["model_registry.model_version"],
            name="fk_pi_model_version",
            onupdate="NO ACTION",
            ondelete="NO ACTION",
        ),
        CheckConstraint(
            "verification_status IN ('UNVERIFIED_HYPOTHESIS', 'CONFIRMED_CAUSE', 'REFUTED', 'APPROVED_FOR_TRAINING') "
            "AND (diagnostic_confidence IS NULL OR (diagnostic_confidence NOT IN ('NaN', 'Infinity', '-Infinity') "
            "AND diagnostic_confidence >= 0.0 AND diagnostic_confidence <= 1.0))",
            name="chk_pi_verification_status",
        ),
        Index("ix_prediction_investigations_tick_id", "tick_id"),
        Index("ix_prediction_investigations_model_version", "model_version"),
    )

    def __repr__(self) -> str:
        return f"<PredictionInvestigation(id={self.id}, tick_id={self.tick_id}, model='{self.model_version}', status='{self.verification_status}')>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tick_id": self.tick_id,
            "prediction_id": self.prediction_id,
            "model_version": self.model_version,
            "headline": self.headline,
            "headline_source_url": self.headline_source_url,
            "headline_published_at": self.headline_published_at.isoformat() if hasattr(self.headline_published_at, "isoformat") else (str(self.headline_published_at) if self.headline_published_at else None),
            "suspected_cause_category": self.suspected_cause_category,
            "hypothesis_notes": self.hypothesis_notes,
            "supporting_evidence_json": self.supporting_evidence_json,
            "diagnostic_confidence": self.diagnostic_confidence,
            "unresolved_questions": self.unresolved_questions,
            "verification_status": self.verification_status,
            "verified_ground_truth_label": self.verified_ground_truth_label,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if hasattr(self.verified_at, "isoformat") else (str(self.verified_at) if self.verified_at else None),
            "revision_history_json": self.revision_history_json,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else (str(self.created_at) if self.created_at else None),
            "updated_at": self.updated_at.isoformat() if hasattr(self.updated_at, "isoformat") else (str(self.updated_at) if self.updated_at else None),
        }