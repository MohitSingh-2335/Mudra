"""
SQLAlchemy ORM models for Mudra Cloud Ledger & Quantitative Observability on Neon PostgreSQL.
"""
from datetime import datetime, timezone
from typing import Any, Dict
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    Text,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


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
    council_audit = relationship("CouncilAudit", back_populates="tick", uselist=False)
    reconciliation = relationship("Reconciliation", back_populates="tick", uselist=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
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
    tick = relationship("MarketTick", back_populates="council_audit")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
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
