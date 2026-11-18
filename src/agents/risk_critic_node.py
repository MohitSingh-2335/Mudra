import math
from typing import Dict, Any, Tuple, Optional
from config import MIN_HURDLE_PCT, TOTAL_FRICTION_PER_SIDE_PCT, ROUND_TRIP_FRICTION_PCT
from src.agents.graph_state import AgentState
from src.agents.market_analyst_agent import _get_gemini_api_key, generate_with_gemini_fallback

ALLOWED_TRADE_ACTIONS = ("HOLD", "BUY", "SELL")


def _parse_float(val: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[bool, float]:
    # Parse a finite float with optional min/max bounds
    if val is None or isinstance(val, bool):
        return False, 0.0
    try:
        f = float(val)
        if not math.isfinite(f):
            return False, 0.0
        if min_val is not None and f < min_val:
            return False, 0.0
        if max_val is not None and f > max_val:
            return False, 0.0
        return True, f
    except (TypeError, ValueError, OverflowError):
        return False, 0.0


def risk_critic_node(state: AgentState) -> AgentState:
    # Reflexion Risk Critic: audits trade invariants, validates empirical fee hurdle, and bounds position size
    raw_action = state.get("trade_action") if isinstance(state, dict) else None
    action = raw_action.strip().upper() if isinstance(raw_action, str) else ""

    # Gate 0: Trade Action Contract
    if action not in ALLOWED_TRADE_ACTIONS:
        state["risk_approved"] = False
        state["veto_reason"] = f"Unsupported Trade Action: Action ({raw_action!r}) is not in allowed set ('HOLD', 'BUY', 'SELL')"
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: Proposed trade_action ({raw_action!r}) is invalid or unsupported."
        return state

    state["trade_action"] = action

    # Gate 1: HOLD Capital Preservation
    if action == "HOLD":
        state["risk_approved"] = True
        state["veto_reason"] = None
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = (
            f"RISK CRITIC CONCURS: Capital preservation authorized. "
            f"Market setup lacks sufficient directional momentum to clear estimated {ROUND_TRIP_FRICTION_PCT:.2f}% round-trip friction. "
            f"Recommended allocation: 0.0% (Cash)."
        )
        return state

    # Gate 2: Directional Friction Hurdle
    ml_forecast = state.get("ml_forecast") if isinstance(state.get("ml_forecast"), dict) else {}
    has_valid_return, expected_return = _parse_float(ml_forecast.get("expected_return_pct"))
    raw_clears = ml_forecast.get("clears_hurdle")
    has_valid_flag = isinstance(raw_clears, bool)
    caller_clears = raw_clears is True
    dir_match = has_valid_return and ((action == "BUY" and expected_return > 0.0) or (action == "SELL" and expected_return < 0.0))
    numeric_clears = has_valid_return and (abs(expected_return) >= MIN_HURDLE_PCT)

    if not (numeric_clears and dir_match and has_valid_flag and caller_clears):
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        if not has_valid_return:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Missing or non-finite forecasted return fails {MIN_HURDLE_PCT:.2f}% break-even threshold"
        elif not dir_match:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Directional Hurdle Mismatch ({action} with return {expected_return:+.4f}%)"
        elif not numeric_clears:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Forecasted return < {MIN_HURDLE_PCT:.2f}% break-even threshold"
        else:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Upstream clears_hurdle contract did not authorize active trade"
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: {action} order fails empirical friction hurdle ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side)."
        return state

    # Gate 3: Volatility Spike Anomaly
    features = state.get("engineered_features") if isinstance(state.get("engineered_features"), dict) else {}
    has_valid_vol, vol_spread_pct = _parse_float(features.get("volatility_range_pct"), min_val=0.0)
    if not has_valid_vol or vol_spread_pct > 3.8:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["veto_reason"] = f"Volatility Spike Anomaly: Single-hour spread ({vol_spread_pct:.2f}%) > 3.8% limit" if has_valid_vol else "Volatility Spike Anomaly: Invalid spread"
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: Excessive volatility risk ({vol_spread_pct:.2f}%)."
        return state

    # Gate 4: Conviction Floor
    has_valid_conv, conviction_raw = _parse_float(state.get("conviction_score"), min_val=1.0, max_val=10.0)
    conviction = int(conviction_raw) if has_valid_conv else 0
    if not has_valid_conv or conviction < 4:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["veto_reason"] = f"Low Conviction Rating: Conviction ({conviction}/10) < 4 minimum threshold"
        state["risk_critic_critique"] = "VETOED BY RISK CRITIC: Conviction below threshold."
        return state

    # Gate 5: Market Price Guard
    has_valid_price, current_price = _parse_float(state.get("current_price"), min_val=1e-6)
    if not has_valid_price:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["veto_reason"] = "Invalid Market Price: Invalid current_price"
        state["risk_critic_critique"] = "VETOED BY RISK CRITIC: Invalid market price."
        return state

    # Sizing and stop-loss / take-profit calculations
    state["risk_approved"] = True
    state["veto_reason"] = None
    position_pct = max(0.05, min(0.20, 0.10 + ((conviction - 5) * 0.02)))
    state["recommended_position_pct"] = round(position_pct, 4)

    stop_dist = current_price * 0.015
    take_dist = current_price * 0.025
    if action == "BUY":
        state["stop_loss_price"] = round(current_price - stop_dist, 2)
        state["take_profit_price"] = round(current_price + take_dist, 2)
    elif action == "SELL":
        state["stop_loss_price"] = round(current_price + stop_dist, 2)
        state["take_profit_price"] = round(current_price - take_dist, 2)

    api_key = _get_gemini_api_key()
    if api_key and not state.get("circuit_breaker_active", False):
        try:
            prompt = f"Chief Risk Officer: {action} order, conviction {conviction}/10, price ${current_price:,.2f}. Position: {position_pct:.1%}. Authorize concisely."
            state["risk_critic_critique"] = generate_with_gemini_fallback(api_key, prompt)
            return state
        except Exception:
            pass

    state["risk_critic_critique"] = (
        f"TRADE AUTHORIZED BY RISK CRITIC: {action} order cleared all empirical risk gates. "
        f"Expected return {expected_return:+.4f}% clears {MIN_HURDLE_PCT:.2f}% hurdle. "
        f"Position sized at {position_pct:.1%} with stop-loss ${state['stop_loss_price']:,.2f} and target ${state['take_profit_price']:,.2f}."
    )
    return state
