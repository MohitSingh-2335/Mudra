# src/agents/tools.py
import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timezone
from langchain_core.tools import tool

from config import (
    ACTIVE_MODEL_VERSION, REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256,
    CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256, FEATURED_BTC_DATA_PATH,
    BTCUSDT_1H_CSV, LIVE_FETCH_LIMIT, MIN_HURDLE_PCT,
)
from src.model_loader import load_verified_model
from src.feature_engineering import (
    extract_features_for_inference, validate_inference_features,
    predict_with_validation, predict_proba_with_validation,
)


def fetch_historical_candles(symbol: str = "BTCUSDT", limit: int = LIVE_FETCH_LIMIT) -> pd.DataFrame:
    # 1. Binance Dedicated Vision
    try:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit={limit + 2}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            now_ms = datetime.now(timezone.utc).timestamp() * 1000.0
            closed = [c for c in resp.json() if float(c[6]) <= now_ms]
            if len(closed) >= 35:
                records = [{
                    "timestamp": pd.to_datetime(c[0], unit="ms", utc=True),
                    "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]),
                    "volume": float(c[5]), "number_of_trades": int(c[8]),
                    "taker_buy_ratio": round(float(c[9]) / float(c[5]), 4) if float(c[5]) > 0 else 0.50,
                } for c in closed[-limit:]]
                df_out = pd.DataFrame(records)
                df_out.attrs["source"] = "binance"
                return df_out
    except Exception:
        pass

    # 2. Coinbase Fallback
    try:
        cb_symbol = "BTC-USD" if "BTC" in symbol else symbol
        resp = requests.get(f"https://api.exchange.coinbase.com/products/{cb_symbol}/candles?granularity=3600", headers={"User-Agent": "Mudra-Quant/1.0"}, timeout=5)
        if resp.status_code == 200:
            now_sec = datetime.now(timezone.utc).timestamp()
            closed = [c for c in resp.json() if (float(c[0]) + 3600.0) <= now_sec]
            if len(closed) >= 35:
                records = [{
                    "timestamp": pd.to_datetime(c[0], unit="s", utc=True),
                    "low": float(c[1]), "high": float(c[2]), "open": float(c[3]), "close": float(c[4]),
                    "volume": float(c[5]), "number_of_trades": None, "taker_buy_ratio": 0.50,
                } for c in reversed(closed[:limit])]
                df_out = pd.DataFrame(records)
                df_out.attrs["source"] = "coinbase"
                return df_out
    except Exception:
        pass

    # 3. Local Dataset Fallback
    try:
        path = FEATURED_BTC_DATA_PATH if os.path.exists(FEATURED_BTC_DATA_PATH) else BTCUSDT_1H_CSV
        if os.path.exists(path):
            from src.data_preprocessing import load_and_clean_data
            df_local = pd.read_csv(path) if path == FEATURED_BTC_DATA_PATH else load_and_clean_data(path)
            if len(df_local) >= 35:
                df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], utc=True)
                df_out = df_local.tail(limit).reset_index(drop=True)
                df_out.attrs["source"] = "local_fallback"
                return df_out
    except Exception:
        pass

    raise ValueError(f"Unable to fetch historical candlesticks for {symbol} from any market source.")


@tool
def get_live_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """Fetches the latest live market candlestick and order-flow telemetry."""
    df_candles = fetch_historical_candles(symbol=symbol, limit=LIVE_FETCH_LIMIT)
    latest = df_candles.iloc[-1]
    raw_trades = latest.get("number_of_trades")
    return {
        "symbol": symbol, "close": float(latest["close"]), "open": float(latest["open"]),
        "high": float(latest["high"]), "low": float(latest["low"]), "volume": float(latest["volume"]),
        "taker_buy_ratio": float(latest.get("taker_buy_ratio", 0.50)),
        "number_of_trades": int(raw_trades) if pd.notna(raw_trades) and raw_trades is not None else None,
        "timestamp": latest.get("timestamp"), "source": "Market Data Feed with HistoricalParity",
        "status": "online", "candles": df_candles.to_dict(orient="records"),
    }


@tool
def calculate_technical_signals(
    close: Optional[float] = None, high: Optional[float] = None, low: Optional[float] = None,
    volume: Optional[float] = None, taker_buy_ratio: Optional[float] = None, open_price: Optional[float] = None,
    timestamp: Optional[Union[str, datetime]] = None, candles: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
    symbol: str = "BTCUSDT"
) -> Dict[str, Any]:
    """Computes technical and microstructural indicators with strict training parity."""
    if candles is not None:
        candles_df = pd.DataFrame(candles) if isinstance(candles, list) else candles.copy()
    else:
        candles_df = fetch_historical_candles(symbol=symbol, limit=LIVE_FETCH_LIMIT)

    if not candles_df.empty:
        idx = candles_df.index[-1]
        if close is not None and close > 0:
            candles_df.at[idx, "close"] = float(close)
        if high is not None and high > 0:
            candles_df.at[idx, "high"] = float(high)
        if low is not None and low > 0:
            candles_df.at[idx, "low"] = float(low)
        if volume is not None and volume >= 0:
            candles_df.at[idx, "volume"] = float(volume)
        if open_price is not None and open_price > 0:
            candles_df.at[idx, "open"] = float(open_price)
        if taker_buy_ratio is not None:
            candles_df.at[idx, "taker_buy_ratio"] = float(taker_buy_ratio)
        elif "taker_buy_ratio" not in candles_df.columns:
            candles_df["taker_buy_ratio"] = 0.50
        if timestamp is not None:
            candles_df.at[idx, "timestamp"] = pd.to_datetime(timestamp)

    return extract_features_for_inference(candles_df)


@tool
def get_model_return_forecast(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Runs dual inference using the Optuna-tuned XGBoost Regressor and Classifier."""
    if not features or not isinstance(features, dict):
        raise ValueError("Invalid features dictionary passed to get_model_return_forecast.")

    df_reg = validate_inference_features(features, model_type="regressor")
    df_clf = validate_inference_features(features, model_type="classifier")
    close = float(features.get("close", features.get("close_price", features.get("current_price", 65000.0))))

    model_reg = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
    if hasattr(model_reg, "set_params"):
        try:
            model_reg.set_params(device="cpu")
        except Exception:
            pass
    raw_pred_return = float(predict_with_validation(model_reg, df_reg, model_type="regressor")[0])

    model_clf = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
    if hasattr(model_clf, "set_params"):
        try:
            model_clf.set_params(device="cpu")
        except Exception:
            pass
    move_code = int(predict_with_validation(model_clf, df_clf, model_type="classifier")[0])

    prob_up, prob_down = 0.50, 0.50
    if hasattr(model_clf, "predict_proba"):
        probs = predict_proba_with_validation(model_clf, df_clf)[0]
        prob_down, prob_up = float(probs[0]), float(probs[1])

    expected_return_pct = raw_pred_return * 100.0
    pred_dollar_move = close * raw_pred_return
    target_price = close + pred_dollar_move
    clears_hurdle = abs(expected_return_pct) >= MIN_HURDLE_PCT
    direction = "UP" if move_code == 1 else "DOWN"
    direction_badge = "SIDEWAYS / CHOP" if not clears_hurdle else ("BULLISH" if move_code == 1 else "BEARISH")

    return {
        "expected_return_pct": round(expected_return_pct, 4),
        "predicted_dollar_move": round(pred_dollar_move, 2),
        "predicted_target_price": round(target_price, 2),
        "clears_hurdle": clears_hurdle,
        "hurdle_threshold_pct": round(MIN_HURDLE_PCT, 4),
        "direction": direction,
        "direction_badge": direction_badge,
        "direction_probability_up": round(prob_up * 100.0, 2),
        "direction_probability_down": round(prob_down * 100.0, 2),
        "confidence": round(max(prob_up, prob_down), 4),
        "model_version": ACTIVE_MODEL_VERSION,
        "status": "success",
    }


@tool
def query_macro_sentiment_rag(query: str = "Bitcoin macroeconomic market sentiment and regime") -> str:
    """Queries the ChromaDB vector database to retrieve foundational market regime knowledge."""
    try:
        from src.agents.rag_agent import get_rag_agent, query_similar_error_memories
        rag = get_rag_agent()
        docs = rag.retrieve_market_context(query, n_results=2) or []
        err_memories = [m["document"] for m in query_similar_error_memories(query_text=query, n_results=2) if "document" in m]
        parts: List[str] = []
        if docs:
            parts.append(" | ".join(docs))
        if err_memories:
            parts.append("Recent Error Memory: " + " ; ".join(err_memories))
        return " || ".join(parts) if parts else "Market regime: Standard institutional spot ETF liquidity."
    except Exception as e:
        return f"Macro context unavailable (ChromaDB offline): {str(e)}"
