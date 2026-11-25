"""Mudra Multi-Agent LangGraph Council & Quant Workflow."""

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import pandas as pd
import requests
from langgraph.graph import END, StateGraph
from langsmith import traceable

from config import (
    ACTIVE_MODEL_VERSION,
    BTCUSDT_1H_CSV,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_MODEL_SHA256,
    FEATURED_BTC_DATA_PATH,
    LIVE_FETCH_LIMIT,
    MIN_HURDLE_PCT,
    REGRESSOR_MODEL_PATH,
    REGRESSOR_MODEL_SHA256,
    ROUND_TRIP_FRICTION_PCT,
    TOTAL_FRICTION_PER_SIDE_PCT,
)
from src.feature_engineering import (
    extract_features_for_inference,
    predict_proba_with_validation,
    predict_with_validation,
    validate_inference_features,
)
from src.model_loader import load_verified_model
from src.ai.llm import _clean_json_response, _get_gemini_api_key, generate_with_gemini_fallback
from src.ai.mcp_client import fetch_mcp_context
from src.ai.rag import get_rag_agent, query_similar_error_memories

logger = logging.getLogger("mudra.ai.workflow")
ALLOWED_TRADE_ACTIONS = ("HOLD", "BUY", "SELL")


def fetch_historical_candles(symbol: str = "BTCUSDT", limit: int = LIVE_FETCH_LIMIT) -> pd.DataFrame:
    """Fetches confirmed 1h candles from Binance (preserving c[9] volume), Coinbase, or local cache."""
    from src.data_preprocessing import enrich_candles_with_macro

    try:
        resp = requests.get(f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit={limit + 2}", timeout=5)
        if resp.status_code == 200:
            now_ms = datetime.now(timezone.utc).timestamp() * 1000.0
            closed = [c for c in resp.json() if float(c[6]) <= now_ms]
            if len(closed) >= 35:
                df_b = pd.DataFrame([
                    {
                        "timestamp": pd.to_datetime(c[0], unit="ms", utc=True),
                        "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]),
                        "volume": float(c[5]), "number_of_trades": int(c[8]),
                        "taker_buy_base_volume": float(c[9]),
                        "taker_buy_ratio": round(float(c[9]) / float(c[5]), 4) if float(c[5]) > 0 else 0.50,
                    }
                    for c in closed[-limit:]
                ])
                try:
                    return enrich_candles_with_macro(df_b)
                except Exception:
                    return df_b
    except Exception:
        pass

    try:
        cb_symbol = "BTC-USD" if "BTC" in symbol else symbol
        resp = requests.get(f"https://api.exchange.coinbase.com/products/{cb_symbol}/candles?granularity=3600", headers={"User-Agent": "Mudra-Quant/1.0"}, timeout=5)
        if resp.status_code == 200:
            now_sec = datetime.now(timezone.utc).timestamp()
            closed = [c for c in resp.json() if (float(c[0]) + 3600.0) <= now_sec]
            if len(closed) >= 35:
                df_cb = pd.DataFrame([
                    {
                        "timestamp": pd.to_datetime(c[0], unit="s", utc=True),
                        "low": float(c[1]), "high": float(c[2]), "open": float(c[3]), "close": float(c[4]),
                        "volume": float(c[5]), "number_of_trades": None,
                    }
                    for c in reversed(closed[:limit])
                ])
                try:
                    return enrich_candles_with_macro(df_cb)
                except Exception:
                    return df_cb
    except Exception:
        pass

    path = FEATURED_BTC_DATA_PATH if os.path.exists(FEATURED_BTC_DATA_PATH) else BTCUSDT_1H_CSV
    if os.path.exists(path):
        from src.data_preprocessing import load_and_clean_data
        df_local = pd.read_csv(path) if path == FEATURED_BTC_DATA_PATH else load_and_clean_data(path)
        if len(df_local) >= 35:
            df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], utc=True)
            df_out = df_local.tail(limit).reset_index(drop=True)
            df_out.attrs["source"] = "local_fallback"
            return df_out

    raise ValueError(f"Unable to fetch historical candlesticks for {symbol} from any market source.")


def get_live_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """Fetches latest confirmed candlestick and order-flow telemetry."""
    df_candles = fetch_historical_candles(symbol=symbol, limit=LIVE_FETCH_LIMIT)
    latest = df_candles.iloc[-1]
    raw_trades = latest.get("number_of_trades")
    ratio_val = latest.get("taker_buy_ratio")
    return {
        "symbol": symbol,
        "close": float(latest["close"]),
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "volume": float(latest["volume"]),
        "taker_buy_ratio": float(ratio_val) if pd.notna(ratio_val) and ratio_val is not None else 0.50,
        "number_of_trades": int(raw_trades) if pd.notna(raw_trades) and raw_trades is not None else None,
        "timestamp": latest.get("timestamp"),
        "source": "Market Data Feed",
        "status": "online",
        "candles": df_candles.to_dict(orient="records"),
    }


def calculate_technical_signals(
    close: Optional[float] = None, high: Optional[float] = None, low: Optional[float] = None,
    volume: Optional[float] = None, taker_buy_ratio: Optional[float] = None, open_price: Optional[float] = None,
    timestamp: Optional[Union[str, datetime]] = None, candles: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
    symbol: str = "BTCUSDT",
) -> Dict[str, Any]:
    """Computes technical indicators with strict training parity."""
    candles_df = pd.DataFrame(candles) if isinstance(candles, list) else (candles.copy() if candles is not None else fetch_historical_candles(symbol=symbol, limit=LIVE_FETCH_LIMIT))
    if not candles_df.empty:
        idx = candles_df.index[-1]
        if close and close > 0:
            candles_df.at[idx, "close"] = float(close)
        if high and high > 0:
            candles_df.at[idx, "high"] = float(high)
        if low and low > 0:
            candles_df.at[idx, "low"] = float(low)
        if volume is not None and volume >= 0:
            candles_df.at[idx, "volume"] = float(volume)
        if open_price and open_price > 0:
            candles_df.at[idx, "open"] = float(open_price)
        if taker_buy_ratio is not None:
            candles_df.at[idx, "taker_buy_ratio"] = float(taker_buy_ratio)

        # Fill taker_buy_ratio and base volume only where missing (never overwriting real Binance c[9])
        if "taker_buy_ratio" not in candles_df.columns:
            candles_df["taker_buy_ratio"] = 0.50
        else:
            candles_df["taker_buy_ratio"] = pd.to_numeric(candles_df["taker_buy_ratio"], errors="coerce").fillna(0.50)

        vol_s = pd.to_numeric(candles_df["volume"], errors="coerce").fillna(0.0)
        ratio_s = candles_df["taker_buy_ratio"]
        if "taker_buy_base_volume" not in candles_df.columns:
            candles_df["taker_buy_base_volume"] = (vol_s * ratio_s).round(4)
        else:
            existing_tbv = pd.to_numeric(candles_df["taker_buy_base_volume"], errors="coerce")
            candles_df["taker_buy_base_volume"] = existing_tbv.fillna((vol_s * ratio_s).round(4))

        if timestamp:
            candles_df.at[idx, "timestamp"] = pd.to_datetime(timestamp)
        if "fng_value" not in candles_df.columns or "onchain_num_tx" not in candles_df.columns:
            try:
                from src.data_preprocessing import enrich_candles_with_macro
                candles_df = enrich_candles_with_macro(candles_df)
            except Exception:
                pass
    return extract_features_for_inference(candles_df)


def get_model_return_forecast(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Runs dual inference using SHA-256 verified XGBoost regressor and classifier."""
    if not features or not isinstance(features, dict):
        raise ValueError("Invalid features dictionary passed to get_model_return_forecast.")

    df_reg = validate_inference_features(features, model_type="regressor")
    df_clf = validate_inference_features(features, model_type="classifier")
    close_raw = features.get("close", features.get("close_price", features.get("current_price")))
    try:
        close = float(close_raw) if close_raw is not None else 0.0
    except (TypeError, ValueError):
        close = 0.0
    if close <= 0.0:
        raise ValueError("Missing valid current BTC price")

    model_reg = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
    model_clf = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
    raw_pred_return = float(predict_with_validation(model_reg, df_reg, model_type="regressor")[0])
    move_code = int(predict_with_validation(model_clf, df_clf, model_type="classifier")[0])

    prob_up, prob_down = 0.50, 0.50
    if hasattr(model_clf, "predict_proba"):
        probs = predict_proba_with_validation(model_clf, df_clf)[0]
        prob_down, prob_up = float(probs[0]), float(probs[1])

    expected_return_pct = raw_pred_return * 100.0
    pred_dollar_move = close * raw_pred_return
    clears_hurdle = abs(expected_return_pct) >= MIN_HURDLE_PCT

    return {
        "expected_return_pct": round(expected_return_pct, 4),
        "predicted_dollar_move": round(pred_dollar_move, 2),
        "predicted_target_price": round(close + pred_dollar_move, 2),
        "clears_hurdle": clears_hurdle,
        "hurdle_threshold_pct": round(MIN_HURDLE_PCT, 4),
        "direction": "UP" if move_code == 1 else "DOWN",
        "direction_badge": "SIDEWAYS / CHOP" if not clears_hurdle else ("BULLISH" if move_code == 1 else "BEARISH"),
        "direction_probability_up": round(prob_up * 100.0, 2),
        "direction_probability_down": round(prob_down * 100.0, 2),
        "confidence": round(max(prob_up, prob_down), 4),
        "model_version": ACTIVE_MODEL_VERSION,
        "status": "success",
    }


def query_macro_sentiment_rag(query: str = "Bitcoin macroeconomic market sentiment and regime") -> str:
    """Queries ChromaDB vector store for market regime rules and episodic error memory."""
    try:
        docs = get_rag_agent().retrieve_market_context(query, n_results=2) or []
        err_memories = [m["document"] for m in query_similar_error_memories(query_text=query, n_results=2) if "document" in m]
        parts: List[str] = []
        if docs:
            parts.append(" | ".join(docs))
        if err_memories:
            parts.append("Recent Error Memory: " + " ; ".join(err_memories))
        return " || ".join(parts) if parts else "Market regime: Standard spot BTC liquidity."
    except Exception as e:
        return f"Macro context unavailable (ChromaDB offline): {e}"


def get_multi_model_consensus(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Computes direction consensus across registered models using classifier probabilities."""
    if not features or not isinstance(features, dict):
        return {"summary": "Consensus unavailable", "up_count": 0, "down_count": 0, "total": 0}
    try:
        from src.model_loader import get_registered_models
        votes: Dict[str, str] = {}
        for ver, info in get_registered_models().items():
            req_f = info.get("features", [])
            if not req_f:
                continue
            row_feats = dict(features)
            if ver.startswith("lgbm_v3") and "sma_10_true_dist_pct" in row_feats:
                row_feats["sma_10_dist_pct"] = row_feats["sma_10_true_dist_pct"]
            if not all(k in row_feats for k in req_f):
                continue
            inp = pd.DataFrame([{k: row_feats[k] for k in req_f}])
            clf_m, reg_m = info.get("classifier"), info.get("regressor")
            if clf_m is not None:
                if hasattr(clf_m, "predict_proba"):
                    votes[ver] = "UP" if float(clf_m.predict_proba(inp)[0][1]) >= 0.50 else "DOWN"
                else:
                    votes[ver] = "UP" if int(clf_m.predict(inp)[0]) == 1 else "DOWN"
            elif reg_m is not None:
                votes[ver] = "UP" if float(reg_m.predict(inp)[0]) >= 0 else "DOWN"
        if not votes:
            return {"summary": "Model consensus unavailable", "up_count": 0, "down_count": 0, "total": 0}
        up = sum(1 for v in votes.values() if v == "UP")
        detail_str = ", ".join(f"{v}: {d}" for v, d in sorted(votes.items()))
        return {"summary": f"{up}/{len(votes)} Models Bullish ({detail_str})", "up_count": up, "down_count": len(votes) - up, "total": len(votes), "votes": votes}
    except Exception:
        return {"summary": "Model consensus offline", "up_count": 0, "down_count": 0, "total": 0}


class AgentState(TypedDict, total=False):
    use_llm: bool
    symbol: str
    current_price: float
    timestamp: str
    market_data: Dict[str, Any]
    engineered_features: Dict[str, Any]
    ml_forecast: Dict[str, Any]
    rag_context: str
    tools_called: List[str]
    mcp_context: Optional[Dict[str, Any]]
    model_consensus: Optional[str]
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
    final_decision: Optional[str]
    risk_verdict: Optional[str]
    strategist_action: Optional[str]
    strategist_conviction: Optional[int]
    kelly_fraction: Optional[float]
    thesis: Optional[str]


def create_initial_agent_state(
    symbol: str = "BTCUSDT", current_price: float = 0.0,
    market_data: Optional[Dict[str, Any]] = None, engineered_features: Optional[Dict[str, Any]] = None,
    ml_forecast: Optional[Dict[str, Any]] = None, timestamp: Optional[str] = None, use_llm: bool = True,
) -> AgentState:
    return {
        "use_llm": bool(use_llm),
        "symbol": symbol,
        "current_price": float(current_price),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "market_data": market_data or {},
        "engineered_features": engineered_features or {},
        "ml_forecast": ml_forecast or {},
        "rag_context": "",
        "tools_called": [],
        "mcp_context": {},
        "model_consensus": "",
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
    }


def _resolve_valid_price(market_res: Any, current_price: float) -> float:
    if current_price > 0.0:
        return float(current_price)
    close_raw = market_res.get("close") if isinstance(market_res, dict) else None
    if close_raw is None or float(close_raw) <= 0.0:
        raise ValueError("Missing valid current BTC price")
    return float(close_raw)


def strategist_node(state: AgentState) -> AgentState:
    state["tools_called"] = state.get("tools_called", [])
    symbol = state.get("symbol", "BTCUSDT")

    if not state.get("market_data") or state.get("current_price", 0.0) <= 0.0:
        if not state.get("market_data"):
            state["market_data"] = get_live_market_data(symbol=symbol)
            state["tools_called"].append("get_live_market_data")
    market_res = state["market_data"]
    state["current_price"] = _resolve_valid_price(market_res, float(state.get("current_price") or 0.0))

    if not state.get("engineered_features"):
        tech_kwargs: Dict[str, Any] = {
            "close": state["current_price"],
            "high": float(market_res.get("high", state["current_price"] * 1.01)),
            "low": float(market_res.get("low", state["current_price"] * 0.99)),
            "volume": float(market_res.get("volume", 1500.0)),
        }
        if isinstance(market_res, dict) and market_res.get("taker_buy_ratio") is not None:
            tech_kwargs["taker_buy_ratio"] = float(market_res["taker_buy_ratio"])
        if isinstance(market_res, dict) and market_res.get("candles") is not None:
            tech_kwargs["candles"] = market_res["candles"]
        state["engineered_features"] = calculate_technical_signals(**tech_kwargs)
        state["tools_called"].append("calculate_technical_signals")

    if not state.get("ml_forecast"):
        state["ml_forecast"] = get_model_return_forecast(features={**state.get("engineered_features", {}), "close": state["current_price"]})
        state["tools_called"].append("get_model_return_forecast")

    if not state.get("rag_context"):
        vol_spread = state.get("engineered_features", {}).get("volatility_range_pct", 1.0)
        taker_ratio_feat = state.get("engineered_features", {}).get("taker_buy_ratio", 0.50)
        clears_h = state.get("ml_forecast", {}).get("clears_hurdle", False)
        if vol_spread > 3.8:
            query = f"{symbol} extreme volatility liquidation flush cascade risk"
        elif clears_h and taker_ratio_feat >= 0.52:
            query = f"{symbol} bullish momentum spot ETF inflows liquidity expansion"
        elif taker_ratio_feat < 0.48:
            query = f"{symbol} distribution selling pressure retail fear capitulation"
        else:
            query = f"{symbol} sideways chop mean reversion range regime"
        state["rag_context"] = query_macro_sentiment_rag(query=query)
        state["tools_called"].append("query_macro_sentiment_rag")

    if not state.get("mcp_context"):
        state["mcp_context"] = fetch_mcp_context(limit=3)
        state["tools_called"].append("fetch_mcp_context")

    if not state.get("model_consensus"):
        mc = get_multi_model_consensus(state.get("engineered_features", {}))
        state["model_consensus"] = mc.get("summary", "Consensus unavailable")

    if not state.get("timestamp"):
        state["timestamp"] = datetime.now(timezone.utc).isoformat()

    pred_return = state["ml_forecast"].get("expected_return_pct", 0.0)
    pred_move = state["ml_forecast"].get("predicted_dollar_move", 0.0)
    clears_hurdle = state["ml_forecast"].get("clears_hurdle", False)
    raw_taker = state["market_data"].get("taker_buy_ratio")
    taker_ratio = float(raw_taker) if raw_taker is not None else 0.50
    direction = state["ml_forecast"].get("direction", "SIDEWAYS")
    direction_badge = state["ml_forecast"].get("direction_badge", "NEUTRAL")
    prob_up = state["ml_forecast"].get("direction_probability_up", 50.0)

    api_key = _get_gemini_api_key() if state.get("use_llm", True) else None
    if api_key:
        try:
            mcp_data = state.get("mcp_context") or {}
            fund_rate = mcp_data.get("funding_rate_8h_pct")
            fund_str = f"{float(fund_rate):.4f}%" if fund_rate is not None else "Unavailable"
            deriv_regime = mcp_data.get("derivatives_regime", "Unavailable")
            consensus = state.get("model_consensus", "N/A")
            prompt = (
                f"You are the Lead Quantitative Strategist at Mudra. Return strict JSON:\n"
                f"- Symbol: {symbol} | Price: ${state['current_price']:,.2f}\n"
                f"- XGBoost Return: {pred_return:+.4f}% (${pred_move:+,.2f}) | Direction: {direction} ({direction_badge}) | Up Prob: {prob_up:.1f}%\n"
                f"- Multi-Model Consensus: {consensus}\n"
                f"- Derivatives Pulse: Funding {fund_str} ({deriv_regime})\n"
                f"- Fee Hurdle: {'CLEARED' if clears_hurdle else 'SUB-HURDLE'} | Taker Ratio: {taker_ratio:.4f}\n"
                f"- Context: {state['rag_context'][:450]}\n"
                f"JSON Keys: trade_action (BUY/SELL/HOLD), conviction_score (1-10), executive_summary, strategist_thesis"
            )
            parsed = json.loads(
                _clean_json_response(str(generate_with_gemini_fallback(api_key, prompt, response_mime_type="application/json", caller="strategist"))),
                strict=False,
            )
            action = str(parsed.get("trade_action", "HOLD")).strip().upper()
            state["trade_action"] = action if action in ALLOWED_TRADE_ACTIONS else "HOLD"  # type: ignore[assignment]
            state["conviction_score"] = max(1, min(10, int(parsed.get("conviction_score", 5))))
            state["executive_summary"] = str(parsed.get("executive_summary", ""))
            state["strategist_thesis"] = str(parsed.get("strategist_thesis", ""))
            state["circuit_breaker_active"] = False
            return state
        except Exception as exc:
            logger.warning("Strategist Gemini call failed, falling back to rules: %s", exc)

    # Deterministic rule fallback (reserves LLM quota in hourly cron when use_llm=False)
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

    consensus_txt = state.get("model_consensus", "N/A")
    mcp_txt = (state.get("mcp_context") or {}).get("derivatives_regime", "Unavailable")
    state["trade_action"] = action  # type: ignore[assignment]
    state["conviction_score"] = conviction
    state["executive_summary"] = summary
    state["strategist_thesis"] = f"Strategist Rule Signal: {action} ({conviction}/10). XGBoost: ${pred_move:+,.2f} ({pred_return:+.4f}%). Consensus: {consensus_txt}. Derivatives: {mcp_txt}. Taker: {taker_ratio:.4f}. Hurdle: {clears_hurdle}."
    return state


def _parse_float(val: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[bool, float]:
    if val is None or isinstance(val, bool):
        return False, 0.0
    try:
        f = float(val)
        if not math.isfinite(f) or (min_val is not None and f < min_val) or (max_val is not None and f > max_val):
            return False, 0.0
        return True, f
    except (TypeError, ValueError, OverflowError):
        return False, 0.0


def _apply_veto(state: AgentState, reason: str, critique: str) -> AgentState:
    state["risk_approved"] = False
    state["veto_reason"] = reason
    state["recommended_position_pct"] = 0.0
    state["stop_loss_price"] = None
    state["take_profit_price"] = None
    state["risk_critic_critique"] = critique
    return state


def risk_critic_node(state: AgentState) -> AgentState:
    raw_action = state.get("trade_action") if isinstance(state, dict) else None
    action = raw_action.strip().upper() if isinstance(raw_action, str) else ""

    if action not in ALLOWED_TRADE_ACTIONS:
        return _apply_veto(
            state,
            f"Unsupported Trade Action: Action ({raw_action!r}) is not in allowed set ('HOLD', 'BUY', 'SELL')",
            f"VETOED BY RISK CRITIC: Proposed trade_action ({raw_action!r}) is invalid or unsupported.",
        )

    state["trade_action"] = action  # type: ignore[assignment]
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
    dir_match = has_valid_return and ((action == "BUY" and expected_return > 0.0) or (action == "SELL" and expected_return < 0.0))
    numeric_clears = has_valid_return and (abs(expected_return) >= MIN_HURDLE_PCT)

    if not (numeric_clears and dir_match and isinstance(raw_clears, bool) and raw_clears is True):
        if not has_valid_return:
            reason = f"Sub-Hurdle Fee Drag: Missing or non-finite forecasted return fails {MIN_HURDLE_PCT:.2f}% break-even threshold"
        elif not dir_match:
            reason = f"Sub-Hurdle Fee Drag: Directional Hurdle Mismatch ({action} with return {expected_return:+.4f}%)"
        elif not numeric_clears:
            reason = f"Sub-Hurdle Fee Drag: Forecasted return < {MIN_HURDLE_PCT:.2f}% break-even threshold"
        else:
            reason = "Sub-Hurdle Fee Drag: Upstream clears_hurdle contract did not authorize active trade"
        return _apply_veto(state, reason, f"VETOED BY RISK CRITIC: {action} order fails empirical friction hurdle ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side).")

    features = state.get("engineered_features") if isinstance(state.get("engineered_features"), dict) else {}
    has_valid_vol, vol_spread_pct = _parse_float(features.get("volatility_range_pct"), min_val=0.0)
    if not has_valid_vol or vol_spread_pct > 3.8:
        reason = f"Volatility Spike Anomaly: Single-hour spread ({vol_spread_pct:.2f}%) > 3.8% limit" if has_valid_vol else "Volatility Spike Anomaly: Invalid spread"
        return _apply_veto(state, reason, f"VETOED BY RISK CRITIC: Excessive volatility risk ({vol_spread_pct:.2f}%).")

    has_valid_conv, conviction_raw = _parse_float(state.get("conviction_score"), min_val=1.0, max_val=10.0)
    conviction = int(conviction_raw) if has_valid_conv else 0
    if not has_valid_conv or conviction < 4:
        return _apply_veto(state, f"Low Conviction Rating: Conviction ({conviction}/10) < 4 minimum threshold", "VETOED BY RISK CRITIC: Conviction below threshold.")

    has_valid_price, current_price = _parse_float(state.get("current_price"), min_val=1e-6)
    if not has_valid_price:
        return _apply_veto(state, "Invalid Market Price: Invalid current_price", "VETOED BY RISK CRITIC: Invalid market price.")

    state["risk_approved"] = True
    state["veto_reason"] = None
    position_pct = max(0.05, min(0.20, 0.10 + ((conviction - 5) * 0.02)))
    state["recommended_position_pct"] = round(position_pct, 4)

    stop_dist, take_dist = current_price * 0.015, current_price * 0.025
    if action == "BUY":
        state["stop_loss_price"] = round(current_price - stop_dist, 2)
        state["take_profit_price"] = round(current_price + take_dist, 2)
    else:
        state["stop_loss_price"] = round(current_price + stop_dist, 2)
        state["take_profit_price"] = round(current_price - take_dist, 2)

    api_key = _get_gemini_api_key() if state.get("use_llm", True) else None
    if api_key and not state.get("circuit_breaker_active", False):
        try:
            prompt = f"Chief Risk Officer: {action} order, conviction {conviction}/10, price ${current_price:,.2f}. Position: {position_pct:.1%}. Authorize concisely."
            state["risk_critic_critique"] = str(generate_with_gemini_fallback(api_key, prompt, caller="risk_critic"))
            return state
        except Exception as exc:
            logger.warning("Risk critic Gemini call failed: %s", exc)

    state["risk_critic_critique"] = (
        f"TRADE AUTHORIZED BY RISK CRITIC: {action} order cleared all empirical risk gates. "
        f"Expected return {expected_return:+.4f}% clears {MIN_HURDLE_PCT:.2f}% hurdle. "
        f"Position sized at {position_pct:.1%} with stop-loss ${state['stop_loss_price']:,.2f} and target ${state['take_profit_price']:,.2f}."
    )
    return state


def build_trading_graph() -> Any:
    """Builds and compiles the 2-node LangGraph council StateGraph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("risk_critic", risk_critic_node)
    workflow.set_entry_point("strategist")
    workflow.add_edge("strategist", "risk_critic")
    workflow.add_edge("risk_critic", END)
    return workflow.compile()


trading_graph_app = build_trading_graph()


@traceable(name="mudra_langgraph_council")
def run_trading_graph(
    symbol: str = "BTCUSDT", current_price: Optional[float] = None,
    market_data: Optional[Dict[str, Any]] = None, engineered_features: Optional[Dict[str, Any]] = None,
    ml_forecast: Optional[Dict[str, Any]] = None, timestamp: Optional[str] = None,
    graph_app: Optional[Any] = None, use_llm: bool = True,
) -> AgentState:
    """Executes the 2-node LangGraph council deliberation cycle."""
    initial_state = create_initial_agent_state(
        symbol=symbol, current_price=float(current_price or 0.0),
        market_data=market_data, engineered_features=engineered_features,
        ml_forecast=ml_forecast, timestamp=timestamp, use_llm=use_llm,
    )
    final_state: AgentState = (graph_app or trading_graph_app).invoke(initial_state)

    raw_action = final_state.get("trade_action") or "HOLD"
    is_approved = bool(final_state.get("risk_approved", False))
    final_state["final_decision"] = raw_action if is_approved else "HOLD"
    final_state["risk_verdict"] = "APPROVED" if is_approved else "VETOED"
    final_state["strategist_action"] = raw_action
    final_state["strategist_conviction"] = final_state.get("conviction_score", 5)
    final_state["kelly_fraction"] = float(final_state.get("recommended_position_pct", 0.0))
    final_state["thesis"] = final_state.get("strategist_thesis") or final_state.get("executive_summary") or "Thesis unavailable."
    final_state.setdefault("mcp_context", initial_state.get("mcp_context", {}))
    final_state.setdefault("model_consensus", initial_state.get("model_consensus", ""))
    return final_state
