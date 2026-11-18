from typing import TypedDict, List, Dict, Any, Optional, Literal
from datetime import datetime, timezone


class AgentState(TypedDict, total=False):
    # market snapshot and telemetry
    symbol: str
    current_price: float
    timestamp: str
    market_data: Dict[str, Any]
    engineered_features: Dict[str, Any]
    ml_forecast: Dict[str, Any]
    rag_context: str
    tools_called: List[str]

    # strategist decision
    strategist_thesis: Optional[str]
    trade_action: Optional[Literal["BUY", "SELL", "HOLD"]]
    conviction_score: Optional[int]
    executive_summary: Optional[str]

    # risk critic decision and sizing
    risk_critic_critique: Optional[str]
    risk_approved: Optional[bool]
    veto_reason: Optional[str]
    recommended_position_pct: Optional[float]
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    circuit_breaker_active: bool
    error: Optional[str]


def create_initial_agent_state(
    symbol: str = "BTCUSDT",
    current_price: float = 0.0,
    market_data: Optional[Dict[str, Any]] = None,
    engineered_features: Optional[Dict[str, Any]] = None,
    ml_forecast: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None
) -> AgentState:
    # returns an empty baseline state dict for agent evaluation
    return {
        "symbol": symbol,
        "current_price": float(current_price),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "market_data": market_data or {},
        "engineered_features": engineered_features or {},
        "ml_forecast": ml_forecast or {},
        "rag_context": "",
        "tools_called": [],
        "strategist_thesis": None,
        "trade_action": None,
        "conviction_score": None,
        "executive_summary": None,
        "risk_critic_critique": None,
        "risk_approved": None,
        "veto_reason": None,
        "recommended_position_pct": 0.0,
        "stop_loss_price": None,
        "take_profit_price": None,
        "circuit_breaker_active": False,
        "error": None,
    }
