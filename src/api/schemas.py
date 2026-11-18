from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from config import ACTIVE_MODEL_VERSION
from src.agents.market_analyst_agent import MarketAnalystReport

__all__ = [
    "ACTIVE_MODEL_VERSION", "HealthResponse", "PredictRequest", "FeatureAttribution",
    "PredictResponse", "ExplainResponse", "FullAnalysisResponse", "AgentTradeRequest",
    "AgentTradeResponse", "ReconciliationSummaryResponse", "PortfolioSummaryResponse",
]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str = "ok"
    model_version: str = ACTIVE_MODEL_VERSION
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PredictRequest(BaseModel):
    current_price: float = Field(..., gt=0)
    features: Dict[str, float]


class FeatureAttribution(BaseModel):
    feature: str
    shap_value: float
    actual_value: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    current_price: float
    predicted_price: float
    predicted_return: float
    directional_movement: str
    movement_code: int
    model_version: str = ACTIVE_MODEL_VERSION
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExplainResponse(BaseModel):
    base_value: float
    top_bullish_drivers: List[FeatureAttribution]
    top_bearish_drivers: List[FeatureAttribution]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FullAnalysisResponse(BaseModel):
    prediction: PredictResponse
    explanation: ExplainResponse
    analyst_commentary: MarketAnalystReport


class AgentTradeRequest(BaseModel):
    symbol: str = "BTCUSDT"
    current_price: Optional[float] = None
    market_data: Optional[Dict[str, Any]] = None
    engineered_features: Optional[Dict[str, float]] = None
    ml_forecast: Optional[Dict[str, Any]] = None


class AgentTradeResponse(BaseModel):
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
    model_config = ConfigDict(extra="ignore")
    total_reconciled: int = 0
    directional_accuracy_pct: float = 0.0
    mae_usd: float = 0.0
    mape_pct: float = 0.0
    information_coefficient_ic: float = 0.0
    black_swan_count: int = 0
    tag_breakdown: Dict[str, int] = Field(default_factory=dict)
    model_version: Optional[str] = None
    query_failed: bool = False
    error: Optional[str] = None


class PortfolioSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cash_balance: float = 10000.0
    btc_balance: float = 0.0
    current_btc_price: float = 0.0
    btc_position_value_usd: float = 0.0
    total_equity_usd: float = 10000.0
    cumulative_return_pct: float = 0.0
    benchmark_btc_equity: float = 10000.0
    alpha_usd: float = 0.0
    alpha_pct: float = 0.0
    realized_pnl_usd: float = 0.0
    total_fees_usd: float = 0.0
    drawdown_pct: float = 0.0
    total_trades: int = 0
