# src/agents/graph_state.py
"""
Shared State Schema for the Mudra Multi-Agent LangGraph System.
Defines the state passed across the Strategist and Risk Critic nodes.
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """
    State dictionary passed between nodes in the Mudra Trading Graph.
    Follows LangGraph StateGraph schema conventions.
    """
    # Core identifiers
    symbol: str
    current_price: float
    timestamp: str

    # Quantitative Telemetry & Inputs
    market_data: Dict[str, Any]
    engineered_features: Dict[str, Any]
    ml_forecast: Dict[str, Any]
    rag_context: str
    tools_called: List[str]

    # Node 1: Quantitative Strategist Outputs
    strategist_thesis: Optional[str]
    trade_action: Optional[str]  # "BUY", "SELL", or "HOLD"
    conviction_score: Optional[int]  # 1 to 10
    executive_summary: Optional[str]

    # Node 2: Reflexion Risk Critic Outputs
    risk_critic_critique: Optional[str]
    risk_approved: Optional[bool]  # True = Approved, False = Vetoed
    veto_reason: Optional[str]
    recommended_position_pct: Optional[float]  # e.g., 0.15 = 15% of capital
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]

    # System Status & Telemetry
    circuit_breaker_active: bool
    error: Optional[str]
