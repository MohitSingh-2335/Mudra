# src/agents/strategist_node.py
"""
Node 1: Quantitative Strategist Agent for Mudra LangGraph System.
Gathers market data, triggers ML inference, queries RAG context,
and synthesizes a structured trade thesis with circuit breaker fallback.
"""

import os
import json
from typing import Dict, Any
from datetime import datetime, timezone

from src.agents.graph_state import AgentState
from src.agents.tools import (
    get_live_market_data,
    calculate_technical_signals,
    get_model_return_forecast,
    query_macro_sentiment_rag
)
from src.agents.market_analyst_agent import _get_gemini_api_key, _clean_json_response


def strategist_node(state: AgentState) -> AgentState:
    """
    Node 1 Execution Function:
    Orchestrates quantitative tools, generates numerical forecasts,
    and constructs a comprehensive trade thesis.
    """
    state["tools_called"] = state.get("tools_called", [])
    symbol = state.get("symbol", "BTCUSDT")

    # 1. Gather Market Telemetry via Tool if missing
    if not state.get("market_data") or state.get("current_price", 0.0) <= 0.0:
        market_res = get_live_market_data.invoke({"symbol": symbol})
        state["market_data"] = market_res
        state["current_price"] = float(market_res.get("close", 65000.0))
        state["tools_called"].append("get_live_market_data")
    else:
        market_res = state["market_data"]

    # 2. Gather Technical Signals via Tool if missing
    if not state.get("engineered_features"):
        tech_res = calculate_technical_signals.invoke({
            "close": state["current_price"],
            "high": float(market_res.get("high", state["current_price"] * 1.01)),
            "low": float(market_res.get("low", state["current_price"] * 0.99)),
            "volume": float(market_res.get("volume", 1500.0)),
            "taker_buy_ratio": float(market_res.get("taker_buy_ratio", 0.50))
        })
        state["engineered_features"] = tech_res
        state["tools_called"].append("calculate_technical_signals")

    # 3. Gather ML Return Forecast via Tool if missing
    if not state.get("ml_forecast"):
        forecast_input = {**state.get("engineered_features", {}), "close": state["current_price"]}
        ml_res = get_model_return_forecast.invoke({"features": forecast_input})
        state["ml_forecast"] = ml_res
        state["tools_called"].append("get_model_return_forecast")

    # 4. Gather RAG Macro Context via Tool if missing
    if not state.get("rag_context"):
        # PROPOSED DYNAMIC QUERY:
        vol_spread = state.get("engineered_features", {}).get("volatility_range_pct", 1.0)
        taker_ratio = state.get("engineered_features", {}).get("taker_buy_ratio", 0.50)
        clears_hurdle = state.get("ml_forecast", {}).get("clears_hurdle", False)

        if vol_spread > 3.8:
            rag_query = f"{symbol} extreme volatility liquidation flush cascade risk"
        elif clears_hurdle and taker_ratio >= 0.52:
            rag_query = f"{symbol} bullish momentum spot ETF inflows institutional accumulation"
        elif taker_ratio < 0.48:
            rag_query = f"{symbol} distribution selling pressure retail fear capitulation"
        else:
            rag_query = f"{symbol} sideways chop mean reversion range regime"
        rag_res = query_macro_sentiment_rag.invoke({"query": rag_query})
        state["rag_context"] = rag_res
        state["tools_called"].append("query_macro_sentiment_rag")

    if not state.get("timestamp"):
        state["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Extract key values for reasoning
    pred_return = state["ml_forecast"].get("expected_return_pct", 0.0)
    pred_move = state["ml_forecast"].get("predicted_dollar_move", 0.0)
    clears_hurdle = state["ml_forecast"].get("clears_hurdle", False)
    taker_ratio = state["market_data"].get("taker_buy_ratio", 0.50)

    direction = state["ml_forecast"].get("direction", "SIDEWAYS")
    direction_badge = state["ml_forecast"].get("direction_badge", "NEUTRAL ⚪")
    prob_up = state["ml_forecast"].get("direction_probability_up", 50.0)

    # 5. Synthesize Thesis via Gemini API or Circuit Breaker
    api_key = _get_gemini_api_key()

    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""You are the Lead Quantitative Strategist at Mudra.
Analyze the following telemetry and return strict JSON:
- Symbol: {symbol}
- Current Price: ${state['current_price']:,.2f}
- XGBoost Return Forecast: {pred_return:+.4f}% (${pred_move:+,.2f})
- Directional Prediction: {direction} ({direction_badge}) | Up Probability: {prob_up:.1f}%
- Fee Hurdle Status: {'CLEARED (>=0.05%)' if clears_hurdle else 'SUB-HURDLE'}
- Taker Buy Ratio: {taker_ratio:.4f}
- Macro RAG Context: {state['rag_context'][:300]}

Respond ONLY with valid JSON with these keys:
{{
  "trade_action": "BUY" or "SELL" or "HOLD",
  "conviction_score": integer 1 to 10,
  "executive_summary": "2 sentence strategic summary",
  "strategist_thesis": "Full quantitative rationale"
}}"""

            response = model.generate_content(prompt)
            clean_text = _clean_json_response(response.text)
            parsed = json.loads(clean_text)

            state["trade_action"] = str(parsed.get("trade_action", "HOLD")).upper()
            state["conviction_score"] = int(parsed.get("conviction_score", 5))
            state["executive_summary"] = str(parsed.get("executive_summary", ""))
            state["strategist_thesis"] = str(parsed.get("strategist_thesis", ""))
            state["circuit_breaker_active"] = False
            return state
        except Exception:
            # Trip circuit breaker on any LLM gateway failure
            pass

    # 6. Deterministic Heuristic Engine Fallback (Circuit Breaker)
    state["circuit_breaker_active"] = True

    if clears_hurdle and (pred_return > 0 or direction == "UP") and taker_ratio >= 0.50:
        action = "BUY"
        conviction = min(9, int(5 + (pred_return * 20) + (taker_ratio - 0.5) * 10))
        summary = (
            f"Bullish quantitative setup: Expected return of {pred_return:+.4f}% clears the "
            f"0.05% fee hurdle ({direction_badge}, Up Prob: {prob_up:.1f}%) with favorable taker buying ({taker_ratio:.2%})."
        )
    elif clears_hurdle and (pred_return < 0 or direction == "DOWN") and taker_ratio < 0.50:
        action = "SELL"
        conviction = min(9, int(5 + abs(pred_return * 20)))
        summary = (
            f"Bearish momentum: Expected return of {pred_return:+.4f}% clears fee hurdle "
            f"({direction_badge}) with aggressive taker sell dominance ({taker_ratio:.2%})."
        )
    else:
        action = "HOLD"
        conviction = 4
        summary = (
            f"Chop / sub-hurdle regime: Expected move ({pred_return:+.4f}%) does not justify "
            f"0.30% round-trip exchange friction. Capital preservation recommended."
        )

    state["trade_action"] = action
    state["conviction_score"] = conviction
    state["executive_summary"] = summary
    state["strategist_thesis"] = (
        f"Quantitative Strategist Rule-Engine Signal: {action} (Conviction: {conviction}/10). "
        f"XGBoost forecasts 1-hour price delta of ${pred_move:+,.2f} ({pred_return:+.4f}%). "
        f"Taker buy ratio sits at {taker_ratio:.4f}. "
        f"Fee-hurdle clearance: {clears_hurdle}."
    )

    return state
