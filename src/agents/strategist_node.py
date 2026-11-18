import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config import (
    MIN_HURDLE_PCT,
    ROUND_TRIP_FRICTION_PCT,
    TOTAL_FRICTION_PER_SIDE_PCT,
)
from src.agents.graph_state import AgentState
from src.agents.market_analyst_agent import _clean_json_response, _get_gemini_api_key, generate_with_gemini_fallback
from src.agents.tools import (
    calculate_technical_signals,
    get_live_market_data,
    get_model_return_forecast,
    query_macro_sentiment_rag,
)


# Node 1: Quantitative Strategist Agent synthesizes trade thesis from models and RAG
def strategist_node(state: AgentState) -> AgentState:
    state["tools_called"] = state.get("tools_called", [])
    symbol = state.get("symbol", "BTCUSDT")

    # 1. Market Telemetry
    if not state.get("market_data") or state.get("current_price", 0.0) <= 0.0:
        market_res = get_live_market_data.invoke({"symbol": symbol})
        state["market_data"] = market_res
        state["current_price"] = float(market_res.get("close", 65000.0))
        state["tools_called"].append("get_live_market_data")
    else:
        market_res = state["market_data"]

    # 2. Microstructure Signals
    if not state.get("engineered_features"):
        tech_input = {
            "close": state["current_price"], "high": float(market_res.get("high", state["current_price"] * 1.01)),
            "low": float(market_res.get("low", state["current_price"] * 0.99)), "volume": float(market_res.get("volume", 1500.0)),
        }
        if isinstance(market_res, dict) and market_res.get("taker_buy_ratio") is not None:
            tech_input["taker_buy_ratio"] = float(market_res["taker_buy_ratio"])
        if isinstance(market_res, dict) and market_res.get("candles") is not None:
            tech_input["candles"] = market_res["candles"]
        state["engineered_features"] = calculate_technical_signals.invoke(tech_input)
        state["tools_called"].append("calculate_technical_signals")

    # 3. Model Forecasts
    if not state.get("ml_forecast"):
        state["ml_forecast"] = get_model_return_forecast.invoke({"features": {**state.get("engineered_features", {}), "close": state["current_price"]}})
        state["tools_called"].append("get_model_return_forecast")

    # 4. Regime-Conditioned RAG Context
    if not state.get("rag_context"):
        vol_spread = state.get("engineered_features", {}).get("volatility_range_pct", 1.0)
        taker_ratio = state.get("engineered_features", {}).get("taker_buy_ratio", 0.50)
        clears_hurdle = state.get("ml_forecast", {}).get("clears_hurdle", False)

        if vol_spread > 3.8:
            query = f"{symbol} extreme volatility liquidation flush cascade risk"
        elif clears_hurdle and taker_ratio >= 0.52:
            query = f"{symbol} bullish momentum spot ETF inflows institutional accumulation"
        elif taker_ratio < 0.48:
            query = f"{symbol} distribution selling pressure retail fear capitulation"
        else:
            query = f"{symbol} sideways chop mean reversion range regime"

        state["rag_context"] = query_macro_sentiment_rag.invoke({"query": query})
        state["tools_called"].append("query_macro_sentiment_rag")

    if not state.get("timestamp"):
        state["timestamp"] = datetime.now(timezone.utc).isoformat()

    pred_return = state["ml_forecast"].get("expected_return_pct", 0.0)
    pred_move = state["ml_forecast"].get("predicted_dollar_move", 0.0)
    clears_hurdle = state["ml_forecast"].get("clears_hurdle", False)
    taker_ratio = state["market_data"].get("taker_buy_ratio", 0.50)
    direction = state["ml_forecast"].get("direction", "SIDEWAYS")
    direction_badge = state["ml_forecast"].get("direction_badge", "NEUTRAL")
    prob_up = state["ml_forecast"].get("direction_probability_up", 50.0)

    # 5. Gemini API Call with Circuit Breaker Fallback
    api_key = _get_gemini_api_key()
    if api_key:
        try:
            prompt = f"""You are the Lead Quantitative Strategist at Mudra. Return strict JSON:
- Symbol: {symbol} | Price: ${state['current_price']:,.2f}
- XGBoost Return: {pred_return:+.4f}% (${pred_move:+,.2f}) | Direction: {direction} ({direction_badge}) | Up Prob: {prob_up:.1f}%
- Fee Hurdle: {'CLEARED' if clears_hurdle else 'SUB-HURDLE'} | Taker Ratio: {taker_ratio:.4f}
- Context: {state['rag_context'][:450]}
JSON Keys: trade_action (BUY/SELL/HOLD), conviction_score (1-10), executive_summary, strategist_thesis"""

            parsed = json.loads(_clean_json_response(generate_with_gemini_fallback(api_key, prompt)))
            action = str(parsed.get("trade_action", "HOLD")).strip().upper()
            state["trade_action"] = action if action in ("BUY", "SELL", "HOLD") else "HOLD"
            state["conviction_score"] = max(1, min(10, int(parsed.get("conviction_score", 5))))
            state["executive_summary"] = str(parsed.get("executive_summary", ""))
            state["strategist_thesis"] = str(parsed.get("strategist_thesis", ""))
            state["circuit_breaker_active"] = False
            return state
        except Exception:
            pass

    # 6. Deterministic Rule-Engine Fallback
    state["circuit_breaker_active"] = True
    if clears_hurdle and (pred_return > 0 or direction == "UP") and taker_ratio >= 0.50:
        action, conviction = "BUY", min(9, int(5 + (pred_return * 20) + (taker_ratio - 0.5) * 10))
        summary = f"Bullish setup: Return {pred_return:+.4f}% clears {MIN_HURDLE_PCT:.2f}% estimated round-trip hurdle ({direction_badge}, Up: {prob_up:.1f}%) with taker buying {taker_ratio:.2%}."
    elif clears_hurdle and (pred_return < 0 or direction == "DOWN") and taker_ratio < 0.50:
        action, conviction = "SELL", min(9, int(5 + abs(pred_return * 20)))
        summary = f"Bearish momentum: Return {pred_return:+.4f}% clears {MIN_HURDLE_PCT:.2f}% estimated round-trip hurdle ({direction_badge}) with taker selling {taker_ratio:.2%}."
    else:
        action, conviction = "HOLD", 4
        summary = f"Chop / sub-hurdle: Expected move ({pred_return:+.4f}%) does not justify {ROUND_TRIP_FRICTION_PCT:.2f}% friction ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side). Cash preserved."

    state["trade_action"] = action
    state["conviction_score"] = conviction
    state["executive_summary"] = summary
    state["strategist_thesis"] = f"Strategist Rule Signal: {action} ({conviction}/10). XGBoost forecasts ${pred_move:+,.2f} ({pred_return:+.4f}%). Taker ratio: {taker_ratio:.4f}. Fee-hurdle: {clears_hurdle}."
    return state
