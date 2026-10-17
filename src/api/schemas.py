# src/api/schemas.py

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from src.agents.market_analyst_agent import MarketAnalystReport

from config import ACTIVE_MODEL_VERSION

__all__ = [
    "ACTIVE_MODEL_VERSION",
    "HealthResponse",
    "PredictRequest",
    "FeatureAttribution",
    "PredictResponse",
    "ExplainResponse",
    "FullAnalysisResponse",
    "AgentTradeRequest",
    "AgentTradeResponse",
    "ReconciliationSummaryResponse",
    "PortfolioSummaryResponse",
]


class HealthResponse(BaseModel):
    """Health check status response."""
    model_config = ConfigDict(protected_namespaces=())
    status: str = Field(default="ok", examples=["ok"])
    model_version: str = Field(default=ACTIVE_MODEL_VERSION)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PredictRequest(BaseModel):
    """
    Inbound prediction request.
    Requires current price and a dictionary mapping feature names to numerical values.
    """
    current_price: float = Field(..., gt=0, description="Latest BTC/USDT price in USD", examples=[62904.50])
    features: Dict[str, float] = Field(..., description="Mapping of feature names to float values")


class FeatureAttribution(BaseModel):
    """Local TreeSHAP feature contribution."""
    feature: str
    shap_value: float
    actual_value: float


class PredictResponse(BaseModel):
    """Response containing numeric forecast, directional classification, and model provenance."""
    model_config = ConfigDict(protected_namespaces=())
    current_price: float
    predicted_price: float
    predicted_return: float
    directional_movement: str
    movement_code: int
    model_version: str = Field(default=ACTIVE_MODEL_VERSION)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExplainResponse(BaseModel):
    """Local TreeSHAP attributions for a given prediction."""
    base_value: float
    top_bullish_drivers: List[FeatureAttribution]
    top_bearish_drivers: List[FeatureAttribution]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FullAnalysisResponse(BaseModel):
    """Complete quantitative analysis fusing ML predictions, SHAP, and Gemini AI agent."""
    prediction: PredictResponse
    explanation: ExplainResponse
    analyst_commentary: MarketAnalystReport


class AgentTradeRequest(BaseModel):
    """Inbound request payload for multi-agent trading graph deliberation."""
    symbol: str = Field(default="BTCUSDT", description="Target trading asset")
    current_price: Optional[float] = Field(default=None, description="Current spot price (auto-fetched if None)")
    market_data: Optional[Dict[str, Any]] = Field(default=None, description="Optional raw bar data")
    engineered_features: Optional[Dict[str, float]] = Field(default=None, description="Optional feature dict")
    ml_forecast: Optional[Dict[str, Any]] = Field(default=None, description="Optional precomputed forecast")


class AgentTradeResponse(BaseModel):
    """Response payload from LangGraph 2-node Strategist -> Reflexion Risk Critic pipeline."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    symbol: str
    current_price: float
    trade_action: str
    conviction_score: int
    executive_summary: str
    strategist_thesis: str
    risk_approved: bool
    veto_reason: Optional[str] = None
    recommended_position_pct: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    risk_critic_critique: str
    circuit_breaker_active: bool
    tools_called: List[str]
    rag_context: str
    timestamp: str
    final_decision: Optional[str] = None
    risk_verdict: Optional[str] = None


class ReconciliationSummaryResponse(BaseModel):
    """Rolling ground truth reconciliation metrics."""
    model_config = ConfigDict(extra="ignore")
    total_reconciled: int = Field(default=0)
    directional_accuracy_pct: float = Field(default=0.0)
    mae_usd: float = Field(default=0.0)
    mape_pct: float = Field(default=0.0)
    information_coefficient_ic: float = Field(default=0.0)
    black_swan_count: int = Field(default=0)
    tag_breakdown: Dict[str, int] = Field(default_factory=dict)
    model_version: Optional[str] = Field(default=None, description="Optional model version filter applied")
    query_failed: bool = Field(default=False, description="Flag indicating whether underlying database query failed")
    error: Optional[str] = Field(default=None, description="Explicit failure description if query failed")


class PortfolioSummaryResponse(BaseModel):
    """Paper portfolio execution metrics and excess Alpha."""
    model_config = ConfigDict(extra="ignore")
    cash_balance: float = Field(default=10000.0)
    btc_balance: float = Field(default=0.0)
    current_btc_price: float = Field(default=0.0)
    btc_position_value_usd: float = Field(default=0.0)
    total_equity_usd: float = Field(default=10000.0)
    cumulative_return_pct: float = Field(default=0.0)
    benchmark_btc_equity: float = Field(default=10000.0)
    alpha_usd: float = Field(default=0.0)
    alpha_pct: float = Field(default=0.0)
    realized_pnl_usd: float = Field(default=0.0)
    total_fees_usd: float = Field(default=0.0)
    drawdown_pct: float = Field(default=0.0)
    total_trades: int = Field(default=0)