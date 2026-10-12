# src/agents/trading_graph.py
"""
Mudra Multi-Agent LangGraph Deliberation Engine.

Compiles the acyclic StateGraph connecting:
1. Node 1: Quantitative Strategist (Directional thesis & conviction generation)
2. Node 2: Reflexion Risk Critic (Invariant auditing, veto authority & Kelly sizing)

Blackboard State Pattern:
Both nodes read and write incrementally to a shared `AgentState` blackboard.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from src.agents.graph_state import AgentState, create_initial_agent_state
from src.agents.strategist_node import strategist_node
from src.agents.risk_critic_node import risk_critic_node


def build_trading_graph() -> Any:
    """
    Constructs and compiles the Mudra Multi-Agent LangGraph workflow.

    Graph Topology:
        START -> [Node 1: Strategist] -> [Node 2: Risk Critic] -> END

    Returns:
        CompiledStateGraph: Executable runnable LangGraph instance.
    """
    workflow = StateGraph(AgentState)

    # 1. Register Graph Nodes
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("risk_critic", risk_critic_node)

    # 2. Define Execution Edges
    workflow.set_entry_point("strategist")
    workflow.add_edge("strategist", "risk_critic")
    workflow.add_edge("risk_critic", END)

    # 3. Compile Runnable Graph
    app = workflow.compile()
    return app


# Pre-compiled global graph instance
trading_graph_app = build_trading_graph()


def run_trading_graph(
    symbol: str = "BTCUSDT",
    current_price: Optional[float] = None,
    market_data: Optional[Dict[str, Any]] = None,
    engineered_features: Optional[Dict[str, Any]] = None,
    ml_forecast: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    graph_app: Optional[Any] = None
) -> AgentState:
    """
    Executes the full Mudra multi-agent pipeline for a market snapshot.

    Args:
        symbol: Target trading pair (e.g. "BTCUSDT").
        current_price: Current spot/mark price in USD.
        market_data: OHLCV, volume, taker metrics from Binance.
        engineered_features: Spreads, order-flow biases, technical indicators.
        ml_forecast: expected_return_pct, confidence, clears_hurdle.
        timestamp: Optional ISO-8601 UTC timestamp string.
        graph_app: Optional compiled StateGraph runnable (defaults to trading_graph_app).

    Returns:
        AgentState: Final deliberated state enriched with council telemetry aliases.
    """
    initial_state = create_initial_agent_state(
        symbol=symbol,
        current_price=float(current_price or 0.0),
        market_data=market_data,
        engineered_features=engineered_features,
        ml_forecast=ml_forecast,
        timestamp=timestamp
    )

    runner = graph_app or trading_graph_app
    final_state: AgentState = runner.invoke(initial_state)

    # -------------------------------------------------------------------------
    # Cross-Component Telemetry Synchronization:
    # Populate legacy/council aliases required by Database, Worker, and UI
    # -------------------------------------------------------------------------
    raw_action = final_state.get("trade_action") or "HOLD"
    is_approved = bool(final_state.get("risk_approved", False))
    veto_reason = final_state.get("veto_reason")

    final_state["final_decision"] = raw_action if is_approved else "HOLD"
    final_state["risk_verdict"] = "APPROVED" if is_approved else "VETOED"
    final_state["strategist_action"] = raw_action
    final_state["strategist_conviction"] = final_state.get("conviction_score", 5)
    final_state["kelly_fraction"] = float(final_state.get("recommended_position_pct", 0.0))
    final_state["veto_reasons"] = [veto_reason] if veto_reason else []
    final_state["thesis"] = (
        final_state.get("strategist_thesis")
        or final_state.get("executive_summary")
        or "Thesis unavailable."
    )

    return final_state

