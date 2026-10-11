# src/agents/trading_graph.py
"""
Mudra Multi-Agent LangGraph Trading Engine.
Compiles the cyclical StateGraph connecting the Quantitative Strategist,
Reflexion Risk Critic, and autonomous tool calling interfaces.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from src.agents.graph_state import AgentState
from src.agents.strategist_node import strategist_node
from src.agents.risk_critic_node import risk_critic_node


def build_trading_graph():
    """
    Constructs and compiles the Mudra Multi-Agent LangGraph workflow.
    Flow: START -> Strategist -> Risk Critic -> END
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
    ml_forecast: Optional[Dict[str, Any]] = None
) -> AgentState:
    """
    Executes the full Mudra multi-agent pipeline for a market snapshot.
    Returns the comprehensive AgentState containing tool outputs,
    the Strategist thesis, and the Risk Critic's order approval/veto.
    """
    initial_state: AgentState = {
        "symbol": symbol,
        "current_price": current_price or 0.0,
        "market_data": market_data or {},
        "engineered_features": engineered_features or {},
        "ml_forecast": ml_forecast or {},
        "rag_context": "",
        "tools_called": [],
        "circuit_breaker_active": False
    }

    final_state = trading_graph_app.invoke(initial_state)
    return final_state
