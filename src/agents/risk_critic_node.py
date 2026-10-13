# src/agents/risk_critic_node.py
"""
Node 2: Reflexion Risk Critic Agent for Mudra LangGraph System.

Serves as an autonomous risk supervisor with absolute veto authority over
the Quantitative Strategist's proposed orders. Implements 3 deterministic gates:
  • Gate 1 (Friction Hurdle): Rejects orders failing to clear the round-trip break-even hurdle.
  • Gate 2 (Volatility Anomaly): Vetoes orders during high-spread liquidation flushes (>3.8%).
  • Gate 3 (Conviction Floor): Rejects orders with model conviction scores < 4/10.

When approved, dynamically computes:
  • Volatility-Scaled Sizing: Fractional Kelly approximation pinned to [5%, 20%] of bankroll.
  • Institutional Risk Levels: Asymmetric stop-loss (-1.5%) and profit target (+2.5%),
    enforcing a minimum 1.67:1 risk-reward profile.
"""

from typing import Dict, Any, Optional
from config import (
    MIN_HURDLE_PCT,
    TOTAL_FRICTION_PER_SIDE_PCT,
    ROUND_TRIP_FRICTION_PCT,
)
from src.agents.graph_state import AgentState
from src.agents.market_analyst_agent import _get_gemini_api_key, _clean_json_response


def risk_critic_node(state: AgentState) -> AgentState:
    """
    Node 2 Execution Function:
    Evaluates Strategist thesis against strict quantitative risk gates.
    Either authorizes the trade order with risk parameters or executes a VETO.
    """
    action = state.get("trade_action", "HOLD")
    current_price = max(1.0, float(state.get("current_price", 65000.0)))
    ml_forecast = state.get("ml_forecast", {})
    features = state.get("engineered_features", {})
    conviction = int(state.get("conviction_score", 5))

    expected_return = float(ml_forecast.get("expected_return_pct", 0.0))
    clears_hurdle = bool(ml_forecast.get("clears_hurdle", False))
    vol_spread_pct = float(features.get("volatility_range_pct", 1.2))

    # 1. HOLD Case: Capital Preservation Already Advised
    if action == "HOLD":
        state["risk_approved"] = True
        state["veto_reason"] = None
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = (
            "RISK CRITIC CONCURS: Capital preservation authorized. "
            f"Market setup lacks sufficient directional momentum to clear estimated {ROUND_TRIP_FRICTION_PCT:.2f}% "
            f"round-trip friction ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side). "
            "Recommended allocation: 0.0% (Cash)."
        )
        return state

    # 2. Gate 1: Transaction Cost Hurdle Audit (For Active BUY/SELL signals)
    if action in ("BUY", "SELL") and not clears_hurdle:
        state["risk_approved"] = False
        state["veto_reason"] = f"Sub-Hurdle Fee Drag: Forecasted return < {MIN_HURDLE_PCT:.2f}% break-even threshold"
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = (
            f"VETOED BY RISK CRITIC: The Strategist recommended {action}, but expected return "
            f"({expected_return:+.4f}%) fails to clear the {MIN_HURDLE_PCT:.2f}% estimated round-trip break-even hurdle "
            f"({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side). Executing this trade would incur fee drag without statistical edge. Order aborted."
        )
        return state

    # 3. Gate 2: Volatility Spike Anomaly Audit
    if vol_spread_pct > 3.8:
        state["risk_approved"] = False
        state["veto_reason"] = f"Volatility Spike Anomaly: Single-hour spread ({vol_spread_pct:.2f}%) > 3.8% limit"
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = (
            f"VETOED BY RISK CRITIC: Single-bar high-low spread of {vol_spread_pct:.2f}% indicates "
            f"an abnormal liquidation cascade or extreme chop. Risk of adverse stop-out exceeds "
            f"expected return. Order aborted for downside protection."
        )
        return state

    # 4. Gate 3: Minimum Conviction Hurdle
    if conviction < 4:
        state["risk_approved"] = False
        state["veto_reason"] = f"Low Conviction Rating: Conviction ({conviction}/10) < 4 minimum threshold"
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = (
            f"VETOED BY RISK CRITIC: Strategist conviction score ({conviction}/10) is below the "
            f"mandatory threshold of 4. Model uncertainty is too elevated to risk portfolio capital."
        )
        return state

    # 5. ALL RISK HURDLES PASSED: Authorize Trade & Compute Position Metrics
    state["risk_approved"] = True
    state["veto_reason"] = None

    # Volatility-Scaled Sizing (Kelly Fraction approximation: 5% to 20% of bankroll)
    base_size = 0.10 + ((conviction - 5) * 0.02)
    position_pct = max(0.05, min(0.20, base_size))
    state["recommended_position_pct"] = round(position_pct, 4)

    # Calculate Institutional Stop-Loss (-1.5%) and Take-Profit (+2.5%) -> 1.67:1 Risk/Reward
    stop_distance = current_price * 0.015
    take_distance = current_price * 0.025

    if action == "BUY":
        state["stop_loss_price"] = round(current_price - stop_distance, 2)
        state["take_profit_price"] = round(current_price + take_distance, 2)
    else:  # SELL / SHORT
        state["stop_loss_price"] = round(current_price + stop_distance, 2)
        state["take_profit_price"] = round(current_price - take_distance, 2)

    # 6. Generate Detailed Risk Rationale (LLM Gateway or Deterministic Fallback)
    api_key = _get_gemini_api_key()
    if api_key and not state.get("circuit_breaker_active", False):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""You are the Chief Risk Officer at Mudra.
The Quantitative Strategist proposed a {action} order with conviction {conviction}/10 at ${current_price:,.2f}.
Risk Parameters:
- Expected Return: {expected_return:+.4f}% (Cleared 0.05% hurdle)
- Approved Position Size: {position_pct:.1%} of capital
- Hard Stop-Loss: ${state['stop_loss_price']:,.2f}
- Take-Profit Target: ${state['take_profit_price']:,.2f}
- Volatility Spread: {vol_spread_pct:.2f}%

Provide a concise 2-sentence formal risk critique authorizing this trade."""

            resp = model.generate_content(prompt)
            critique_text = resp.text.strip()
            state["risk_critic_critique"] = critique_text
            return state
        except Exception:
            pass

    # Deterministic Risk Critique Fallback
    state["risk_critic_critique"] = (
        f"TRADE AUTHORIZED BY RISK CRITIC: {action} order cleared all empirical risk gates. "
        f"Expected return of {expected_return:+.4f}% satisfies transaction friction constraints. "
        f"Position sized at {position_pct:.1%} of bankroll with hard stop-loss pinned at "
        f"${state['stop_loss_price']:,.2f} (-1.5%) and target at ${state['take_profit_price']:,.2f} (+2.5%)."
    )

    return state

