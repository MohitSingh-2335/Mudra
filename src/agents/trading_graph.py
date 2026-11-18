from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from src.agents.graph_state import AgentState, create_initial_agent_state
from src.agents.strategist_node import strategist_node
from src.agents.risk_critic_node import risk_critic_node


def build_trading_graph() -> Any:
    # Build LangGraph workflow connecting strategist to risk critic
    workflow = StateGraph(AgentState)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("risk_critic", risk_critic_node)
    workflow.set_entry_point("strategist")
    workflow.add_edge("strategist", "risk_critic")
    workflow.add_edge("risk_critic", END)
    return workflow.compile()


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
    # Run multi-agent pipeline and return enriched deliberation state
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
