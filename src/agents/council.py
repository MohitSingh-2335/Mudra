from datetime import datetime, timezone
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from config import (
    MIN_HURDLE_PCT,
    ROUND_TRIP_FRICTION_PCT,
    TOTAL_FRICTION_PER_SIDE_PCT,
)
from src.agents.tools import (
    calculate_technical_signals,
    get_live_market_data,
    get_model_return_forecast,
    query_macro_sentiment_rag,
)

load_dotenv('.env')
ALLOWED_TRADE_ACTIONS = ("HOLD", "BUY", "SELL")
GEMINI_MODEL_CANDIDATES = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash")


# typeddict schema holding all deliberation state across agents
class AgentState(TypedDict, total=False):
    symbol: str
    current_price: float
    timestamp: str
    market_data: Dict[str, Any]
    engineered_features: Dict[str, Any]
    ml_forecast: Dict[str, Any]
    rag_context: str
    tools_called: List[str]
    strategist_thesis: Optional[str]
    trade_action: Optional[Literal["BUY", "SELL", "HOLD"]]
    conviction_score: Optional[int]
    executive_summary: Optional[str]
    risk_critic_critique: Optional[str]
    risk_approved: Optional[bool]
    veto_reason: Optional[str]
    recommended_position_pct: Optional[float]
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    circuit_breaker_active: bool
    error: Optional[str]
    final_decision: Optional[str]
    risk_verdict: Optional[str]
    strategist_action: Optional[str]
    strategist_conviction: Optional[int]
    kelly_fraction: Optional[float]
    veto_reasons: Optional[List[str]]
    thesis: Optional[str]


# returns an empty baseline state dict for agent evaluation
def create_initial_agent_state(
    symbol: str = "BTCUSDT",
    current_price: float = 0.0,
    market_data: Optional[Dict[str, Any]] = None,
    engineered_features: Optional[Dict[str, Any]] = None,
    ml_forecast: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None
) -> AgentState:
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


# pydantic schema for market commentary output
class MarketAnalystReport(BaseModel):
    trade_thesis: str = Field(description="STRONG BULLISH, MODERATE BULLISH, NEUTRAL, MODERATE BEARISH, STRONG BEARISH")
    conviction_score: int = Field(description="Conviction rating 1-10")
    executive_summary: str = Field(description="Strategic executive summary")
    quantitative_driver_analysis: str = Field(description="Breakdown of top SHAP features")
    macro_market_context: str = Field(description="Retrieved domain knowledge")
    key_risks: List[str] = Field(description="Risk factors that invalidate thesis")


# reads gemini api key from environment or secrets
def _read_raw_gemini_api_key() -> Optional[str]:
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.getenv(key_name)
        if val and val.strip() and val.strip() != "your_gemini_api_key_here":
            return val.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for sec in ("Google", "google", "gemini", "Gemini"):
                if sec in st.secrets and "api_key" in st.secrets[sec]:
                    v = str(st.secrets[sec]["api_key"]).strip()
                    if v and v != "your_gemini_api_key_here":
                        return v
            for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                if var in st.secrets and str(st.secrets[var]).strip() != "your_gemini_api_key_here":
                    return str(st.secrets[var]).strip()
    except Exception:
        pass
    return None


# returns api key only if not running under pytest
def _get_gemini_api_key() -> Optional[str]:
    return None if os.getenv("PYTEST_CURRENT_TEST") else _read_raw_gemini_api_key()


# returns status dictionary for gemini model availability
def get_gemini_status() -> Dict[str, Any]:
    raw_key = _read_raw_gemini_api_key()
    preferred = os.getenv("GEMINI_MODEL_NAME", GEMINI_MODEL_CANDIDATES[0])
    masked = f"{raw_key[:4]}...{raw_key[-4:]}" if raw_key and len(raw_key) > 8 else ("Configured" if raw_key else None)
    return {
        "configured": bool(raw_key), "masked_key": masked, "preferred_model": preferred,
        "fallback_models": list(GEMINI_MODEL_CANDIDATES),
        "label": f"Gemini AI: Active ({preferred})" if raw_key else "Gemini AI: Rule Engine Fallback (No API Key)",
    }


# calls gemini with tiered model fallback across 2.5, 2.0, and 1.5 flash
def generate_with_gemini_fallback(api_key: str, prompt: str, system_instruction: Optional[str] = None) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    custom_model = os.getenv("GEMINI_MODEL_NAME")
    candidates = (custom_model,) + tuple(m for m in GEMINI_MODEL_CANDIDATES if m != custom_model) if custom_model else GEMINI_MODEL_CANDIDATES

    last_exc: Optional[Exception] = None
    for idx, model_name in enumerate(candidates):
        try:
            if system_instruction and idx == 0:
                try:
                    model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
                except TypeError:
                    model = genai.GenerativeModel(model_name)
            else:
                model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            text_out = getattr(resp, "text", None)
            if text_out and str(text_out).strip():
                return str(text_out).strip()
        except AssertionError:
            raise
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"All Gemini model candidates failed. Last error: {last_exc}")


# strips markdown code fences from json responses
def _clean_json_response(raw_text: str) -> str:
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    return match.group(1).strip() if match else text


# fallback report generator when api key is absent
def _generate_fallback_report(
    current_price: float, pred_price: float, pred_return: float, pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]], top_neg_drivers: List[Dict[str, Any]], rag_context: List[str]
) -> MarketAnalystReport:
    is_bullish = "Upward" in pred_move_text or pred_return > 0
    thesis = "MODERATE BULLISH" if is_bullish else "MODERATE BEARISH"
    conviction = 7 if abs(pred_return) > 0.002 else 6
    pos_summary = ", ".join([f"{d['feature']} (+{d['shap']:.3f})" for d in top_pos_drivers[:2]]) or "steady order book flow"
    neg_summary = ", ".join([f"{d['feature']} ({d['shap']:.3f})" for d in top_neg_drivers[:2]]) or "minor overhead resistance"

    return MarketAnalystReport(
        trade_thesis=thesis, conviction_score=conviction,
        executive_summary=f"Mudra ensemble forecasts {thesis.lower()} toward ${pred_price:,.2f} ({pred_return:+.2%}).",
        quantitative_driver_analysis=f"Upside catalysts: {pos_summary}. Downward drag: {neg_summary}.",
        macro_market_context=f"Market principles: {rag_context[0] if rag_context else 'Order flow imbalance dominates intraday.'}",
        key_risks=["Volatility expansion causing stop-out.", "Rapid mean-reversion at local resistance."],
    )


# synthesizes model forecast, shap drivers, and rag into analyst commentary
def generate_market_commentary(
    current_price: float, pred_price: float, pred_return: float, pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]], top_neg_drivers: List[Dict[str, Any]],
    market_snapshot: Optional[Dict[str, Any]] = None
) -> MarketAnalystReport:
    from src.agents.rag_agent import get_rag_agent
    rag = get_rag_agent()
    query_terms = [pred_move_text] + ([top_pos_drivers[0].get("feature", "")] if top_pos_drivers else []) + ([top_neg_drivers[0].get("feature", "")] if top_neg_drivers else [])
    rag_docs = rag.retrieve_market_context(" ".join(query_terms).strip(), n_results=2)

    api_key = _get_gemini_api_key()
    if not api_key:
        return _generate_fallback_report(current_price, pred_price, pred_return, pred_move_text, top_pos_drivers, top_neg_drivers, rag_docs)

    try:
        pos_str = ", ".join([f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_pos_drivers]) or "None"
        neg_str = ", ".join([f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_neg_drivers]) or "None"
        rag_context_str = "\n".join([f"- {d}" for d in rag_docs])
        prompt = (
            f"You are Mudra AI's Lead Quantitative Strategist.\n"
            f"Current BTC: ${current_price:,.2f}, Predicted: ${pred_price:,.2f} ({pred_return:+.4%}, {pred_move_text})\n"
            f"Bullish drivers: {pos_str}\nBearish drivers: {neg_str}\nRAG context: {rag_context_str}\n"
            f"Return strictly JSON with keys: trade_thesis, conviction_score, executive_summary, quantitative_driver_analysis, macro_market_context, key_risks."
        )
        raw_text = generate_with_gemini_fallback(api_key, prompt)
        clean_text = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        report_data = json.loads(match.group(1).strip() if match else clean_text)

        valid_theses = {"STRONG BULLISH", "MODERATE BULLISH", "NEUTRAL", "MODERATE BEARISH", "STRONG BEARISH"}
        raw_thesis = str(report_data.get("trade_thesis", "NEUTRAL")).upper().strip()
        report_data["trade_thesis"] = raw_thesis if raw_thesis in valid_theses else "NEUTRAL"
        report_data["conviction_score"] = max(1, min(10, int(report_data.get("conviction_score", 5))))
        return MarketAnalystReport(**report_data)
    except Exception:
        return _generate_fallback_report(current_price, pred_price, pred_return, pred_move_text, top_pos_drivers, top_neg_drivers, rag_docs)


# strategist node evaluates model forecast, technical signals, and rag context to propose trade action
def strategist_node(state: AgentState) -> AgentState:
    state["tools_called"] = state.get("tools_called", [])
    symbol = state.get("symbol", "BTCUSDT")

    if not state.get("market_data") or state.get("current_price", 0.0) <= 0.0:
        market_res = get_live_market_data.invoke({"symbol": symbol})
        state["market_data"] = market_res
        state["current_price"] = float(market_res.get("close", 65000.0))
        state["tools_called"].append("get_live_market_data")
    else:
        market_res = state["market_data"]

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

    if not state.get("ml_forecast"):
        state["ml_forecast"] = get_model_return_forecast.invoke({"features": {**state.get("engineered_features", {}), "close": state["current_price"]}})
        state["tools_called"].append("get_model_return_forecast")

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


# validates float values with finite and boundary checks
def _parse_float(val: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[bool, float]:
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


# risk critic node audits trades against empirical fee hurdles and bounds position size
def risk_critic_node(state: AgentState) -> AgentState:
    raw_action = state.get("trade_action") if isinstance(state, dict) else None
    action = raw_action.strip().upper() if isinstance(raw_action, str) else ""

    if action not in ALLOWED_TRADE_ACTIONS:
        state["risk_approved"] = False
        state["veto_reason"] = f"Unsupported Trade Action: Action ({raw_action!r}) is not in allowed set ('HOLD', 'BUY', 'SELL')"
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: Proposed trade_action ({raw_action!r}) is invalid or unsupported."
        return state

    state["trade_action"] = action

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

    has_valid_price, current_price = _parse_float(state.get("current_price"), min_val=1e-6)
    if not has_valid_price:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["veto_reason"] = "Invalid Market Price: Invalid current_price"
        state["risk_critic_critique"] = "VETOED BY RISK CRITIC: Invalid market price."
        return state

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


# compiles langgraph dag linking strategist to risk critic
def build_trading_graph() -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("risk_critic", risk_critic_node)
    workflow.set_entry_point("strategist")
    workflow.add_edge("strategist", "risk_critic")
    workflow.add_edge("risk_critic", END)
    return workflow.compile()


trading_graph_app = build_trading_graph()


# executes multi-agent pipeline and returns enriched deliberation state
def run_trading_graph(
    symbol: str = "BTCUSDT",
    current_price: Optional[float] = None,
    market_data: Optional[Dict[str, Any]] = None,
    engineered_features: Optional[Dict[str, Any]] = None,
    ml_forecast: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    graph_app: Optional[Any] = None
) -> AgentState:
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
