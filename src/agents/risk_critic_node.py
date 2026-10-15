# src/agents/risk_critic_node.py
"""
Node 2: Reflexion Risk Critic Agent for Mudra LangGraph System.

Serves as an autonomous risk supervisor with absolute veto authority over
the Quantitative Strategist's proposed orders. Implements deterministic gates:
  • Gate 0 (Action Contract): Allows only 'HOLD', 'BUY', and 'SELL'; rejects unsupported actions fail-closed
    before evaluating or converting any ancillary state fields.
  • HOLD Preservation: Immediately authorizes 0.0% cash preservation for 'HOLD' without requiring valid
    forecast returns or active-trade sizing parameters.
  • Gate 1 (Directional Friction Hurdle & Upstream Contract): Rejects active orders unless expected_return_pct
    is a finite number whose sign agrees with the proposed action (BUY requires expected_return_pct >= MIN_HURDLE_PCT;
    SELL requires expected_return_pct <= -MIN_HURDLE_PCT) AND ml_forecast['clears_hurdle'] is strictly boolean True.
  • Gate 2 (Volatility Anomaly): Vetoes active orders when volatility_range_pct is missing, malformed,
    negative, non-finite, or exceeds the 3.8% hourly liquidation flush limit.
  • Gate 3 (Conviction Floor): Rejects active orders when conviction_score is missing, malformed,
    boolean, non-integral, non-finite, or < 4/10.
  • Gate 4 (Market Price Guard): Rejects active orders when current_price is missing, malformed,
    boolean, non-finite, or <= 0.0 before computing stop-loss and take-profit levels.

When approved, dynamically computes:
  • Volatility-Scaled Sizing: Fractional Kelly approximation pinned to [5%, 20%] of bankroll.
  • Institutional Risk Levels: Asymmetric stop-loss (-1.5%) and profit target (+2.5%),
    enforcing a minimum 1.67:1 risk-reward profile.
"""

import math
from typing import Dict, Any, Optional, Tuple
from config import (
    MIN_HURDLE_PCT,
    TOTAL_FRICTION_PER_SIDE_PCT,
    ROUND_TRIP_FRICTION_PCT,
)
from src.agents.graph_state import AgentState
from src.agents.market_analyst_agent import _get_gemini_api_key, _clean_json_response

ALLOWED_TRADE_ACTIONS = ("HOLD", "BUY", "SELL")


def _parse_finite_expected_return_pct(ml_forecast: Any) -> Tuple[bool, float, Any]:
    """
    Safely parses expected_return_pct from ml_forecast without raising on missing,
    None, boolean, malformed string, or non-finite (NaN, +inf, -inf) inputs.

    Returns:
        (is_valid_finite, parsed_float_or_zero, raw_value)
    """
    if not isinstance(ml_forecast, dict) or "expected_return_pct" not in ml_forecast:
        return False, 0.0, None

    raw_val = ml_forecast.get("expected_return_pct")
    if raw_val is None or isinstance(raw_val, bool):
        return False, 0.0, raw_val

    try:
        val = float(raw_val)
    except (TypeError, ValueError, OverflowError):
        return False, 0.0, raw_val

    if not math.isfinite(val):
        return False, 0.0, raw_val

    return True, val, raw_val


def _parse_strict_clears_hurdle_flag(ml_forecast: Any) -> Tuple[bool, bool, Any]:
    """
    Validates the authoritative upstream clears_hurdle contract in ml_forecast.
    Requires an explicit boolean True (`raw_flag is True`) to authorize an active trade;
    missing keys, None, False, numeric (0/1), string ('true'/'false'), or container values
    fail closed as False without raising or truthy coercion.

    Returns:
        (is_valid_bool_flag, caller_clears, raw_flag)
    """
    if not isinstance(ml_forecast, dict) or "clears_hurdle" not in ml_forecast:
        return False, False, None

    raw_flag = ml_forecast.get("clears_hurdle")
    if not isinstance(raw_flag, bool):
        return False, False, raw_flag

    return True, raw_flag, raw_flag


def _parse_finite_volatility_pct(features: Any) -> Tuple[bool, float, Any]:
    """
    Safely parses volatility_range_pct from engineered_features for active trades.
    Rejects non-dict containers, missing keys, None, booleans, malformed strings,
    non-finite floats (NaN, +inf, -inf), and negative values (< 0.0).

    Returns:
        (is_valid_non_negative_finite, parsed_float_or_zero, raw_value)
    """
    if not isinstance(features, dict) or "volatility_range_pct" not in features:
        return False, 0.0, None

    raw_val = features.get("volatility_range_pct")
    if raw_val is None or isinstance(raw_val, bool):
        return False, 0.0, raw_val

    try:
        val = float(raw_val)
    except (TypeError, ValueError, OverflowError):
        return False, 0.0, raw_val

    if not math.isfinite(val) or val < 0.0:
        return False, 0.0, raw_val

    return True, val, raw_val


def _parse_valid_conviction_score(state: Any) -> Tuple[bool, int, Any]:
    """
    Safely parses conviction_score from state for active trades.
    Accepts int (excluding bool) or finite integral float; rejects missing keys,
    None, booleans, strings, fractional floats, NaN, and +-inf.

    Returns:
        (is_valid_int, parsed_int_or_zero, raw_value)
    """
    if not isinstance(state, dict) or "conviction_score" not in state:
        return False, 0, None

    raw_val = state.get("conviction_score")
    if raw_val is None or isinstance(raw_val, bool):
        return False, 0, raw_val

    if isinstance(raw_val, int):
        return True, int(raw_val), raw_val

    if isinstance(raw_val, float):
        if math.isfinite(raw_val) and raw_val.is_integer():
            return True, int(raw_val), raw_val
        return False, 0, raw_val

    return False, 0, raw_val


def _parse_positive_finite_price(state: Any) -> Tuple[bool, float, Any]:
    """
    Safely parses current_price from state for active BUY/SELL sizing and SL/TP anchoring.
    Rejects missing keys, None, booleans, malformed strings, non-finite floats (NaN, +inf, -inf),
    and non-positive prices (<= 0.0).

    Returns:
        (is_valid_positive_finite, parsed_float_or_zero, raw_value)
    """
    if not isinstance(state, dict) or "current_price" not in state:
        return False, 0.0, None

    raw_val = state.get("current_price")
    if raw_val is None or isinstance(raw_val, bool):
        return False, 0.0, raw_val

    try:
        val = float(raw_val)
    except (TypeError, ValueError, OverflowError):
        return False, 0.0, raw_val

    if not math.isfinite(val) or val <= 0.0:
        return False, 0.0, raw_val

    return True, val, raw_val


def risk_critic_node(state: AgentState) -> AgentState:
    """
    Node 2 Execution Function:
    Evaluates Strategist thesis against strict quantitative risk gates.
    Either authorizes the trade order with risk parameters or executes a VETO.
    """
    raw_action = state.get("trade_action") if isinstance(state, dict) else None
    action = raw_action.strip().upper() if isinstance(raw_action, str) else ""

    # 0. Gate 0: Trade Action Contract Validation (Allow only HOLD, BUY, SELL)
    # Executed BEFORE reading or converting any ancillary state fields so malformed
    # current_price, conviction_score, or engineered_features cannot raise before Gate 0.
    if action not in ALLOWED_TRADE_ACTIONS:
        state["risk_approved"] = False
        state["veto_reason"] = (
            f"Unsupported Trade Action: Action ({raw_action!r}) is not in allowed set ('HOLD', 'BUY', 'SELL')"
        )
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = (
            f"VETOED BY RISK CRITIC: Proposed trade_action ({raw_action!r}) is invalid or unsupported. "
            "Only 'HOLD', 'BUY', and 'SELL' actions are permitted. Order aborted."
        )
        return state

    state["trade_action"] = action

    # 1. HOLD Case: Capital Preservation Already Advised
    # Evaluated before active-trade input parsing so a HOLD never becomes a risk-gate
    # failure or raises if ml_forecast, current_price, conviction_score, or engineered_features
    # are absent, None, or malformed.
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

    # 2. Gate 1: Directional Transaction Cost Hurdle & Upstream Contract Audit (For Active BUY/SELL signals)
    ml_forecast = state.get("ml_forecast")
    has_valid_return, expected_return, raw_return = _parse_finite_expected_return_pct(ml_forecast)
    has_valid_flag, caller_clears, raw_clears_flag = _parse_strict_clears_hurdle_flag(ml_forecast)
    direction_matches_action = has_valid_return and (
        (action == "BUY" and expected_return > 0.0)
        or (action == "SELL" and expected_return < 0.0)
    )
    numeric_clears = has_valid_return and (abs(expected_return) >= MIN_HURDLE_PCT)
    clears_hurdle = (
        numeric_clears
        and direction_matches_action
        and has_valid_flag
        and caller_clears
    )

    if not clears_hurdle:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        if not has_valid_return:
            state["veto_reason"] = (
                f"Sub-Hurdle Fee Drag: Missing or non-finite forecasted return ({raw_return!r}) "
                f"fails {MIN_HURDLE_PCT:.2f}% break-even threshold"
            )
            state["risk_critic_critique"] = (
                f"VETOED BY RISK CRITIC: The Strategist recommended {action}, but forecasted return "
                f"({raw_return!r}) is missing, malformed, or non-finite and cannot clear the "
                f"{MIN_HURDLE_PCT:.2f}% estimated round-trip break-even hurdle "
                f"({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side). Order aborted."
            )
        elif not direction_matches_action:
            state["veto_reason"] = (
                f"Sub-Hurdle Fee Drag: Directional Hurdle Mismatch ({action} with forecasted return "
                f"{expected_return:+.4f}% fails directional {MIN_HURDLE_PCT:.2f}% break-even threshold)"
            )
            state["risk_critic_critique"] = (
                f"VETOED BY RISK CRITIC: The Strategist recommended {action}, but expected return "
                f"({expected_return:+.4f}%) is directionally opposite to the proposed order and fails the "
                f"directional {MIN_HURDLE_PCT:.2f}% estimated round-trip break-even hurdle "
                f"({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side). Order aborted."
            )
        elif not numeric_clears:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Forecasted return < {MIN_HURDLE_PCT:.2f}% break-even threshold"
            state["risk_critic_critique"] = (
                f"VETOED BY RISK CRITIC: The Strategist recommended {action}, but expected return "
                f"({expected_return:+.4f}%) fails to clear the {MIN_HURDLE_PCT:.2f}% estimated round-trip break-even hurdle "
                f"({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side). Executing this trade would incur fee drag without statistical edge. Order aborted."
            )
        else:
            state["veto_reason"] = (
                f"Sub-Hurdle Fee Drag: Upstream clears_hurdle contract ({raw_clears_flag!r}) "
                f"did not authorize active trade at {MIN_HURDLE_PCT:.2f}% break-even threshold"
            )
            state["risk_critic_critique"] = (
                f"VETOED BY RISK CRITIC: The Strategist recommended {action} with expected return "
                f"({expected_return:+.4f}%), but upstream clears_hurdle contract ({raw_clears_flag!r}) "
                f"is not boolean True for the {MIN_HURDLE_PCT:.2f}% estimated round-trip break-even hurdle "
                f"({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side). Order aborted."
            )
        return state

    # 3. Gate 2: Volatility Spread Validation & Spike Anomaly Audit
    features = state.get("engineered_features")
    has_valid_vol, vol_spread_pct, raw_vol = _parse_finite_volatility_pct(features)
    if not has_valid_vol or vol_spread_pct > 3.8:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        if not has_valid_vol:
            state["veto_reason"] = (
                f"Volatility Spike Anomaly: Missing, negative, or non-finite volatility spread ({raw_vol!r})"
            )
            state["risk_critic_critique"] = (
                f"VETOED BY RISK CRITIC: Volatility spread ({raw_vol!r}) is missing, malformed, negative, "
                "or non-finite. Order aborted for downside protection."
            )
        else:
            state["veto_reason"] = f"Volatility Spike Anomaly: Single-hour spread ({vol_spread_pct:.2f}%) > 3.8% limit"
            state["risk_critic_critique"] = (
                f"VETOED BY RISK CRITIC: Single-bar high-low spread of {vol_spread_pct:.2f}% indicates "
                f"an abnormal liquidation cascade or extreme chop. Risk of adverse stop-out exceeds "
                f"expected return. Order aborted for downside protection."
            )
        return state

    # 4. Gate 3: Conviction Score Validation & Minimum Hurdle
    has_valid_conviction, conviction, raw_conviction = _parse_valid_conviction_score(state)
    if not has_valid_conviction or conviction < 4:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        if not has_valid_conviction:
            state["veto_reason"] = (
                f"Low Conviction Rating: Missing or malformed conviction score ({raw_conviction!r}) < 4 minimum threshold"
            )
            state["risk_critic_critique"] = (
                f"VETOED BY RISK CRITIC: Strategist conviction score ({raw_conviction!r}) is missing, "
                "malformed, or non-finite. Model uncertainty is too elevated to risk portfolio capital."
            )
        else:
            state["veto_reason"] = f"Low Conviction Rating: Conviction ({conviction}/10) < 4 minimum threshold"
            state["risk_critic_critique"] = (
                f"VETOED BY RISK CRITIC: Strategist conviction score ({conviction}/10) is below the "
                f"mandatory threshold of 4. Model uncertainty is too elevated to risk portfolio capital."
            )
        return state

    # 5. Gate 4: Market Price Validation (Required to anchor Stop-Loss and Take-Profit)
    has_valid_price, current_price, raw_price = _parse_positive_finite_price(state)
    if not has_valid_price:
        state["risk_approved"] = False
        state["veto_reason"] = (
            f"Invalid Market Price: Missing, non-positive, or non-finite current_price ({raw_price!r})"
        )
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = (
            f"VETOED BY RISK CRITIC: Current market price ({raw_price!r}) is missing, non-positive, "
            "or non-finite; cannot safely anchor stop-loss or take-profit levels. Order aborted."
        )
        return state

    # 6. ALL RISK HURDLES PASSED: Authorize Trade & Compute Position Metrics
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
    elif action == "SELL":
        state["stop_loss_price"] = round(current_price + stop_distance, 2)
        state["take_profit_price"] = round(current_price - take_distance, 2)

    # 7. Generate Detailed Risk Rationale (LLM Gateway or Deterministic Fallback)
    api_key = _get_gemini_api_key()
    if api_key and not state.get("circuit_breaker_active", False):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""You are the Chief Risk Officer at Mudra.
The Quantitative Strategist proposed a {action} order with conviction {conviction}/10 at ${current_price:,.2f}.
Risk Parameters:
- Expected Return: {expected_return:+.4f}% (Cleared {MIN_HURDLE_PCT:.2f}% hurdle)
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
