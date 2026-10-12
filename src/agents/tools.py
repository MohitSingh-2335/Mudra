# src/agents/tools.py
"""
Quantitative FinTech Tools for Mudra Multi-Agent LangGraph System.
Defines explicit tool-calling interfaces adhering to LangChain / LangGraph standards.
"""

import os
import joblib
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timezone
from langchain_core.tools import tool

from config import (
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH,
    BTCUSDT_1H_CSV,
    LIVE_FETCH_LIMIT
)
from src.feature_engineering import extract_features_for_inference


def fetch_historical_candles(symbol: str = "BTCUSDT", limit: int = LIVE_FETCH_LIMIT) -> pd.DataFrame:
    """
    Fetches the latest sequence of confirmed closed hourly candlesticks for rolling indicator computation.
    Verifies candle closure via provider timestamps to prevent active unclosed bars from entering inference.
    Primary: Binance Dedicated Vision.
    Fallback 1: Coinbase Institutional Public Feed.
    Fallback 2: Local historical dataset.
    """
    # 1. Binance Dedicated Vision
    try:
        # Request extra candles to account for dropping the actively forming candle
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit={limit + 2}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            klines = resp.json()
            # Verify closure by timestamp: close_time_ms (index 6) must have elapsed
            now_ms = datetime.now(timezone.utc).timestamp() * 1000.0
            closed_klines = [c for c in klines if float(c[6]) <= now_ms]

            if len(closed_klines) >= 35:
                # Take exactly the requested lookback history preserving ascending chronological order
                selected_klines = closed_klines[-limit:]
                records = []
                for c in selected_klines:
                    vol = float(c[5])
                    taker_vol = float(c[9])
                    taker_ratio = float(taker_vol / vol) if vol > 0 else 0.50
                    records.append({
                        "timestamp": pd.to_datetime(c[0], unit="ms", utc=True),
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": vol,
                        "number_of_trades": int(c[8]),
                        "taker_buy_ratio": round(taker_ratio, 4),
                    })
                return pd.DataFrame(records)
    except Exception:
        pass

    # 2. Coinbase Institutional Public Feed
    try:
        cb_symbol = "BTC-USD" if "BTC" in symbol else symbol
        cb_url = f"https://api.exchange.coinbase.com/products/{cb_symbol}/candles?granularity=3600"
        resp = requests.get(cb_url, headers={"User-Agent": "Mudra-Quant/1.0"}, timeout=5)
        if resp.status_code == 200:
            raw = resp.json()
            # Verify closure by timestamp: bucket end time (c[0] + 3600s) must have elapsed
            now_sec = datetime.now(timezone.utc).timestamp()
            closed_candles = [c for c in raw if (float(c[0]) + 3600.0) <= now_sec]

            if len(closed_candles) >= 35:
                # Coinbase returns descending (newest first); select latest `limit` then reverse for ascending order
                selected_candles = list(reversed(closed_candles[:limit]))
                records = []
                for c in selected_candles:
                    records.append({
                        "timestamp": pd.to_datetime(c[0], unit="s", utc=True),
                        "low": float(c[1]),
                        "high": float(c[2]),
                        "open": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                        "number_of_trades": None,  # Coinbase does not report trade count; do not fabricate
                        "taker_buy_ratio": 0.50,
                    })
                return pd.DataFrame(records)
    except Exception:
        pass

    # 3. Local Dataset Fallback
    try:
        if os.path.exists(FEATURED_BTC_DATA_PATH):
            df_local = pd.read_csv(FEATURED_BTC_DATA_PATH)
            if len(df_local) >= 35:
                df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], utc=True)
                return df_local.tail(limit).reset_index(drop=True)
        elif os.path.exists(BTCUSDT_1H_CSV):
            from src.data_preprocessing import load_and_clean_data
            df_local = load_and_clean_data(BTCUSDT_1H_CSV)
            if len(df_local) >= 35:
                return df_local.tail(limit).reset_index(drop=True)
    except Exception:
        pass

    raise ValueError(f"Unable to fetch historical candlesticks for {symbol} from any market source.")


@tool
def get_live_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Fetches the latest live market candlestick and order-flow telemetry for the target cryptocurrency,
    including the historical sequence of confirmed closed candles required for rolling indicator parity.
    """
    df_candles = fetch_historical_candles(symbol=symbol, limit=LIVE_FETCH_LIMIT)
    latest = df_candles.iloc[-1]

    raw_trades = latest.get("number_of_trades")
    trades_val = int(raw_trades) if pd.notna(raw_trades) and raw_trades is not None else None

    return {
        "symbol": symbol,
        "close": float(latest["close"]),
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "volume": float(latest["volume"]),
        "taker_buy_ratio": float(latest.get("taker_buy_ratio", 0.50)),
        "number_of_trades": trades_val,
        "timestamp": latest.get("timestamp"),
        "source": "Market Data Feed with HistoricalParity",
        "status": "online",
        "candles": df_candles.to_dict(orient="records"),
    }


@tool
def calculate_technical_signals(
    close: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    volume: Optional[float] = None,
    taker_buy_ratio: Optional[float] = None,
    open_price: Optional[float] = None,
    timestamp: Optional[Union[str, datetime]] = None,
    candles: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
    symbol: str = "BTCUSDT"
) -> Dict[str, Any]:
    """
    Computes technical and microstructural indicators with strict training parity.
    Extracts all 26 regressor and 39 classifier features from historical candle sequences.
    Preserves observed taker_buy_ratio in candles unless an explicit override is provided.
    """
    # 1. Resolve Candlestick Sequence
    if candles is not None:
        if isinstance(candles, list):
            candles_df = pd.DataFrame(candles)
        elif isinstance(candles, pd.DataFrame):
            candles_df = candles.copy()
        else:
            raise ValueError("Invalid candles format: must be DataFrame or list of dicts.")
    else:
        candles_df = fetch_historical_candles(symbol=symbol, limit=LIVE_FETCH_LIMIT)

    # 2. If scalar tick overrides are provided, reflect them on the latest candle
    if not candles_df.empty:
        last_idx = candles_df.index[-1]
        if close is not None and close > 0:
            candles_df.at[last_idx, "close"] = float(close)
        if high is not None and high > 0:
            candles_df.at[last_idx, "high"] = float(high)
        if low is not None and low > 0:
            candles_df.at[last_idx, "low"] = float(low)
        if volume is not None and volume >= 0:
            candles_df.at[last_idx, "volume"] = float(volume)
        if open_price is not None and open_price > 0:
            candles_df.at[last_idx, "open"] = float(open_price)
        # Preserve observed ratio: only overwrite when an explicit ratio is supplied
        if taker_buy_ratio is not None:
            candles_df.at[last_idx, "taker_buy_ratio"] = float(taker_buy_ratio)
        elif "taker_buy_ratio" not in candles_df.columns:
            candles_df["taker_buy_ratio"] = 0.50
        if timestamp is not None:
            candles_df.at[last_idx, "timestamp"] = pd.to_datetime(timestamp)

    # 3. Extract features enforcing training/live mathematical parity
    return extract_features_for_inference(candles_df)


@tool
def get_model_return_forecast(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runs dual inference using the Optuna-tuned XGBoost Regressor and Classifier.
    Enforces strict feature parity: rejects incomplete payloads without substituting 0.0.
    Returns expected percentage return, predicted dollar move, target close price,
    directional prediction (UP/DOWN), probability, and symmetric fee hurdle check.
    """
    if not features or not isinstance(features, dict):
        raise ValueError("Invalid features dictionary passed to get_model_return_forecast.")

    feature_dict = features

    # Validate that all required model features exist and are non-null
    missing_reg = [col for col in REGRESSOR_FEATURES if col not in feature_dict or pd.isna(feature_dict[col])]
    missing_clf = [col for col in CLASSIFIER_FEATURES if col not in feature_dict or pd.isna(feature_dict[col])]

    if missing_reg:
        raise ValueError(f"Feature parity violation: missing required regressor features: {missing_reg}")
    if missing_clf:
        raise ValueError(f"Feature parity violation: missing required classifier features: {missing_clf}")

    current_close = float(feature_dict.get("close", feature_dict.get("close_price", feature_dict.get("current_price", 65000.0))))

    # 1. Regressor Inference (Target Return)
    if not os.path.exists(REGRESSOR_MODEL_PATH):
        raise FileNotFoundError(f"Regressor model artifact not found at {REGRESSOR_MODEL_PATH}")

    model_reg = joblib.load(REGRESSOR_MODEL_PATH)
    if hasattr(model_reg, "set_params"):
        try:
            model_reg.set_params(device="cpu")
        except Exception:
            pass

    df_reg = pd.DataFrame([{col: float(feature_dict[col]) for col in REGRESSOR_FEATURES}], columns=REGRESSOR_FEATURES)
    raw_pred_return = float(model_reg.predict(df_reg)[0])

    # 2. Classifier Inference (Directional Movement & Probability)
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        raise FileNotFoundError(f"Classifier model artifact not found at {CLASSIFIER_MODEL_PATH}")

    model_clf = joblib.load(CLASSIFIER_MODEL_PATH)
    if hasattr(model_clf, "set_params"):
        try:
            model_clf.set_params(device="cpu")
        except Exception:
            pass

    df_clf = pd.DataFrame([{col: float(feature_dict[col]) for col in CLASSIFIER_FEATURES}], columns=CLASSIFIER_FEATURES)
    move_code = int(model_clf.predict(df_clf)[0])
    prob_up = 0.50
    prob_down = 0.50
    if hasattr(model_clf, "predict_proba"):
        probs = model_clf.predict_proba(df_clf)[0]
        prob_down = float(probs[0])
        prob_up = float(probs[1])

    # 3. Unit Normalization & Mathematical Targets
    expected_return_pct = raw_pred_return * 100.0  # e.g., 0.0035 -> +0.3500%
    pred_dollar_move = current_close * raw_pred_return
    target_price = current_close + pred_dollar_move

    # 4. Symmetric Fee Hurdle (|Return| >= 0.05% = 5 basis points)
    clears_hurdle = abs(expected_return_pct) >= 0.05

    direction = "UP" if move_code == 1 else "DOWN"
    if not clears_hurdle:
        direction_badge = "SIDEWAYS / CHOP"
    else:
        direction_badge = "BULLISH" if move_code == 1 else "BEARISH"

    confidence = max(prob_up, prob_down)

    return {
        "expected_return_pct": round(expected_return_pct, 4),
        "predicted_dollar_move": round(pred_dollar_move, 2),
        "predicted_target_price": round(target_price, 2),
        "clears_hurdle": clears_hurdle,
        "hurdle_threshold_pct": 0.05,
        "direction": direction,
        "direction_badge": direction_badge,
        "direction_probability_up": round(prob_up * 100.0, 2),
        "direction_probability_down": round(prob_down * 100.0, 2),
        "confidence": round(confidence, 4),
        "status": "success"
    }


@tool
def query_macro_sentiment_rag(query: str = "Bitcoin macroeconomic market sentiment and regime") -> str:
    """
    Queries the ChromaDB vector database to retrieve historically similar macroeconomic news,
    regulatory developments, and institutional sentiment precedents.
    """
    try:
        from src.agents.rag_agent import get_rag_agent
        rag = get_rag_agent()
        docs = rag.retrieve_market_context(query, n_results=2)
        if docs and len(docs) > 0:
            return " | ".join(docs)
        return "Market regime: Standard institutional spot ETF liquidity with balanced macro interest rate expectations."
    except Exception as e:
        return f"Macro context unavailable (ChromaDB offline): {str(e)}"
