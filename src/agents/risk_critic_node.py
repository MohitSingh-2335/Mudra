# src/agents/risk_critic_node.py
import math
from typing import Dict, Any, Tuple
from config import MIN_HURDLE_PCT, TOTAL_FRICTION_PER_SIDE_PCT, ROUND_TRIP_FRICTION_PCT
from src.agents.graph_state import AgentState
from src.agents.market_analyst_agent import _get_gemini_api_key, generate_with_gemini_fallback

ALLOWED_TRADE_ACTIONS = ("HOLD", "BUY", "SELL")


def _parse_finite_expected_return_pct(ml_forecast: Any) -> Tuple[bool, float, Any]:
    if not isinstance(ml_forecast, dict) or "expected_return_pct" not in ml_forecast:
        return False, 0.0, None
    raw = ml_forecast.get("expected_return_pct")
    if raw is None or isinstance(raw, bool):
        return False, 0.0, raw
    try:
        val = float(raw)
        return (math.isfinite(val), val if math.isfinite(val) else 0.0, raw)
    except (TypeError, ValueError, OverflowError):
        return False, 0.0, raw


def _parse_strict_clears_hurdle_flag(ml_forecast: Any) -> Tuple[bool, bool, Any]:
    if not isinstance(ml_forecast, dict) or "clears_hurdle" not in ml_forecast:
        return False, False, None
    raw = ml_forecast.get("clears_hurdle")
    return (isinstance(raw, bool), raw is True, raw)


def _parse_finite_volatility_pct(features: Any) -> Tuple[bool, float, Any]:
    if not isinstance(features, dict) or "volatility_range_pct" not in features:
        return False, 0.0, None
    raw = features.get("volatility_range_pct")
    if raw is None or isinstance(raw, bool):
        return False, 0.0, raw
    try:
        val = float(raw)
        return (math.isfinite(val) and val >= 0.0, val if math.isfinite(val) and val >= 0.0 else 0.0, raw)
    except (TypeError, ValueError, OverflowError):
        return False, 0.0, raw


def _parse_valid_conviction_score(state: Any) -> Tuple[bool, int, Any]:
    if not isinstance(state, dict) or "conviction_score" not in state:
        return False, 0, None
    raw = state.get("conviction_score")
    if raw is None or isinstance(raw, bool):
        return False, 0, raw
    if isinstance(raw, int):
        return True, raw, raw
    if isinstance(raw, float) and math.isfinite(raw) and raw.is_integer():
        return True, int(raw), raw
    return False, 0, raw


def _parse_positive_finite_price(state: Any) -> Tuple[bool, float, Any]:
    if not isinstance(state, dict) or "current_price" not in state:
        return False, 0.0, None
    raw = state.get("current_price")
    if raw is None or isinstance(raw, bool):
        return False, 0.0, raw
    try:
        val = float(raw)
        return (math.isfinite(val) and val > 0.0, val if math.isfinite(val) and val > 0.0 else 0.0, raw)
    except (TypeError, ValueError, OverflowError):
        return False, 0.0, raw


def risk_critic_node(state: AgentState) -> AgentState:
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

    # Gate 1: HOLD Preservation
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
    ml_forecast = state.get("ml_forecast")
    has_valid_return, expected_return, raw_return = _parse_finite_expected_return_pct(ml_forecast)
    has_valid_flag, caller_clears, raw_clears_flag = _parse_strict_clears_hurdle_flag(ml_forecast)
    dir_match = has_valid_return and ((action == "BUY" and expected_return > 0.0) or (action == "SELL" and expected_return < 0.0))
    numeric_clears = has_valid_return and (abs(expected_return) >= MIN_HURDLE_PCT)

    if not (numeric_clears and dir_match and has_valid_flag and caller_clears):
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        if not has_valid_return:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Missing or non-finite forecasted return ({raw_return!r}) fails {MIN_HURDLE_PCT:.2f}% break-even threshold"
        elif not dir_match:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Directional Hurdle Mismatch ({action} with return {expected_return:+.4f}%)"
        elif not numeric_clears:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Forecasted return < {MIN_HURDLE_PCT:.2f}% break-even threshold"
        else:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Upstream clears_hurdle contract ({raw_clears_flag!r}) did not authorize active trade"
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: {action} order fails empirical friction hurdle ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side)."
        return state

    # Gate 3: Volatility Spike Anomaly
    has_valid_vol, vol_spread_pct, raw_vol = _parse_finite_volatility_pct(state.get("engineered_features"))
    if not has_valid_vol or vol_spread_pct > 3.8:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["veto_reason"] = f"Volatility Spike Anomaly: Single-hour spread ({vol_spread_pct:.2f}%) > 3.8% limit" if has_valid_vol else f"Volatility Spike Anomaly: Invalid spread ({raw_vol!r})"
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: Excessive volatility risk ({vol_spread_pct:.2f}%)."
        return state

    # Gate 4: Conviction Floor
    has_valid_conviction, conviction, raw_conviction = _parse_valid_conviction_score(state)
    if not has_valid_conviction or conviction < 4:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["veto_reason"] = f"Low Conviction Rating: Conviction ({conviction}/10) < 4 minimum threshold" if has_valid_conviction else f"Low Conviction Rating: Invalid conviction ({raw_conviction!r})"
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: Conviction below threshold."
        return state

    # Gate 5: Market Price Guard
    has_valid_price, current_price, raw_price = _parse_positive_finite_price(state)
    if not has_valid_price:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["veto_reason"] = f"Invalid Market Price: Invalid current_price ({raw_price!r})"
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: Invalid market price."
        return state

    # All gates cleared: Sizing and SL/TP computation
    state["risk_approved"] = True
    state["veto_reason"] = None
    position_pct = max(0.05, min(0.20, 0.10 + ((conviction - 5) * 0.02)))
    state["recommended_position_pct"] = round(position_pct, 4)

    stop_distance = current_price * 0.015
    take_distance = current_price * 0.025
    if action == "BUY":
        state["stop_loss_price"] = round(current_price - stop_distance, 2)
        state["take_profit_price"] = round(current_price + take_distance, 2)
    elif action == "SELL":
        state["stop_loss_price"] = round(current_price + stop_distance, 2)
        state["take_profit_price"] = round(current_price - take_distance, 2)

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
