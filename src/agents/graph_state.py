# src/agents/graph_state.py
"""
Shared State Schema for the Mudra Multi-Agent LangGraph System.

Defines the central state dictionary (`AgentState`) passed across nodes in
the LangGraph StateGraph pipeline:
  • Node 1 (Quantitative Strategist): Ingests telemetry, executes tools,
    computes returns, and synthesizes structured trade thesis.
  • Node 2 (Reflexion Risk Critic): Audits thesis against hurdle rates,
    volatility spreads, and assigns stop-loss / position sizing with absolute veto power.

Uses `TypedDict(total=False)` to allow nodes to return partial state diffs
that LangGraph automatically merges into the shared graph state.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from datetime import datetime, timezone


class AgentState(TypedDict, total=False):
    """
    Central blackboard state schema for the Mudra Multi-Agent Trading Graph.
    Follows LangGraph StateGraph schema conventions.
    """
    # -------------------------------------------------------------------------
    # 1. Core Asset Identifiers & Snapshots
    # -------------------------------------------------------------------------
    symbol: str                          # Ticker pair, e.g., "BTCUSDT"
    current_price: float                 # Latest mark/spot price in USD
    timestamp: str                       # ISO-8601 UTC timestamp of market snapshot

    # -------------------------------------------------------------------------
    # 2. Quantitative Telemetry & Tool Ingestion
    # -------------------------------------------------------------------------
    market_data: Dict[str, Any]          # OHLCV, volume, taker metrics from Binance
    engineered_features: Dict[str, Any]  # Spreads, order-flow biases, technical indicators
    ml_forecast: Dict[str, Any]          # expected_return_pct, confidence_score, clears_hurdle
    rag_context: str                     # Retrieved macro narrative from ChromaDB vector store
    tools_called: List[str]              # Chronological list of tool function invocations

    # -------------------------------------------------------------------------
    # 3. Node 1: Quantitative Strategist Outputs
    # -------------------------------------------------------------------------
    strategist_thesis: Optional[str]     # Multi-factor qualitative and quantitative thesis
    trade_action: Optional[Literal["BUY", "SELL", "HOLD"]]  # Directional order recommendation
    conviction_score: Optional[int]      # Model conviction on integer scale [1, 10]
    executive_summary: Optional[str]     # One-line executive order summary

    # -------------------------------------------------------------------------
    # 4. Node 2: Reflexion Risk Critic Outputs
    # -------------------------------------------------------------------------
    risk_critic_critique: Optional[str]  # Detailed risk audit analysis and critique
    risk_approved: Optional[bool]        # True = Order authorized, False = Vetoed
    veto_reason: Optional[str]           # Quantitative violation triggering veto, else None
    recommended_position_pct: Optional[float]  # Volatility-scaled capital fraction [0.0, 1.0]
    stop_loss_price: Optional[float]     # Volatility-adjusted stop-loss exit level in USD
    take_profit_price: Optional[float]   # Risk-reward calibrated profit target in USD

    # -------------------------------------------------------------------------
    # 5. System Status, Resilience & Fallbacks
    # -------------------------------------------------------------------------
    circuit_breaker_active: bool         # True if deterministic fallback rule engaged
    error: Optional[str]                 # Unhandled exception trace or error message


def create_initial_agent_state(
    symbol: str = "BTCUSDT",
    current_price: float = 0.0,
    market_data: Optional[Dict[str, Any]] = None,
    engineered_features: Optional[Dict[str, Any]] = None,
    ml_forecast: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None
) -> AgentState:
    """
    Standardized factory function to initialize an AgentState blackboard
    prior to invoking the compiled LangGraph StateGraph.
    """
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
