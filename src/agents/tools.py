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
from typing import Dict, Any, Optional
from langchain_core.tools import tool

from datetime import datetime, timezone

from config import (
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH
)


@tool
def get_live_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Fetches the latest live market candlestick and order-flow telemetry for the target cryptocurrency.
    Primary source: Binance Dedicated Vision (data-api.binance.vision).
    Fallback source: Coinbase Institutional Public Feed (api.exchange.coinbase.com).
    """
    # 1. Try Binance Dedicated Vision (No auth, institutional taker buy volume)
    try:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit=5"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            klines = resp.json()
            if klines:
                latest = klines[-1]
                close = float(latest[4])
                high = float(latest[2])
                low = float(latest[3])
                vol = float(latest[5])
                taker_base = float(latest[9])
                taker_ratio = float(taker_base / vol) if vol > 0 else 0.5
                trades = int(latest[8])

                return {
                    "symbol": symbol,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": vol,
                    "taker_buy_ratio": round(taker_ratio, 4),
                    "number_of_trades": trades,
                    "source": "Binance Dedicated Vision (Live)",
                    "status": "online"
                }
    except Exception:
        pass

    # 2. Try Coinbase Institutional Public Feed
    try:
        cb_symbol = "BTC-USD" if "BTC" in symbol else symbol
        cb_url = f"https://api.exchange.coinbase.com/products/{cb_symbol}/candles?granularity=3600"
        cb_resp = requests.get(cb_url, timeout=5)
        if cb_resp.status_code == 200:
            candles = cb_resp.json()
            if candles:
                latest = candles[0]  # [time, low, high, open, close, volume]
                close = float(latest[4])
                low = float(latest[1])
                high = float(latest[2])
                vol = float(latest[5])

                return {
                    "symbol": symbol,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": vol,
                    "taker_buy_ratio": 0.50,  # Coinbase omits taker breakdown; neutral default
                    "number_of_trades": 0,
                    "source": "Coinbase Institutional Public (Fallback)",
                    "status": "online"
                }
    except Exception:
        pass

    # 3. Fallback to Local Parquet / CSV Simulation
    try:
        if os.path.exists(FEATURED_BTC_DATA_PATH):
            df = pd.read_csv(FEATURED_BTC_DATA_PATH)
            last_row = df.iloc[-1]
            close = float(last_row.get("close", 65000.0))
            return {
                "symbol": symbol,
                "close": close,
                "high": float(last_row.get("high", close * 1.01)),
                "low": float(last_row.get("low", close * 0.99)),
                "volume": float(last_row.get("volume", 1500.0)),
                "taker_buy_ratio": float(last_row.get("taker_buy_ratio", 0.50)),
                "number_of_trades": int(last_row.get("number_of_trades", 15000)),
                "source": "Local Historical Parquet Replay",
                "status": "simulation"
            }
    except Exception:
        pass

    # Absolute safety fallback
    return {
        "symbol": symbol,
        "close": 65000.0,
        "high": 65500.0,
        "low": 64800.0,
        "volume": 1200.0,
        "taker_buy_ratio": 0.50,
        "number_of_trades": 10000,
        "source": "Default Neutral Fallback",
        "status": "offline_default"
    }


@tool
def calculate_technical_signals(
    close: float,
    high: float,
    low: float,
    volume: float,
    taker_buy_ratio: float = 0.5,
    open_price: Optional[float] = None,
    timestamp: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Computes key microstructural and technical momentum indicators from price and order-flow inputs.
    Calculates volatility range, order-flow buyer dominance, and trend bias with aligned feature keys.
    """
    spread = high - low
    spread_pct = (spread / close) * 100.0 if close > 0 else 0.0
    hl_ratio = (high / low) if low > 0 else 1.0
    op = open_price if open_price and open_price > 0 else close
    price_change = ((close - op) / op) * 100.0

    # Determine order-flow imbalance
    if taker_buy_ratio > 0.55:
        flow_bias = "Bullish Aggressive Buying (Takers Sweeping Book)"
    elif taker_buy_ratio < 0.45:
        flow_bias = "Bearish Aggressive Selling (Inventory Dumping)"
    else:
        flow_bias = "Neutral Balanced Liquidity"

    # Synthetic RSI approximation for single bar relative to spread
    simulated_rsi = 50.0 + (taker_buy_ratio - 0.5) * 60.0
    simulated_rsi = max(10.0, min(90.0, simulated_rsi))

    # Time-based cyclical features
    ts = pd.to_datetime(timestamp, utc=True) if timestamp else datetime.now(timezone.utc)
    hour = int(ts.hour)
    dayofweek = int(ts.weekday() if hasattr(ts, "weekday") else getattr(ts, "dayofweek", 0))
    day = int(ts.day)

    return {
        # Raw & Model Feature Keys
        "close": float(close),
        "open": float(op),
        "high": float(high),
        "low": float(low),
        "volume": float(volume),
        "taker_buy_ratio": round(float(taker_buy_ratio), 4),
        "Price Change": round(float(price_change), 4),
        "high_low_ratio": round(float(hl_ratio), 6),
        "close_lag_1": float(close),
        "volume_lag_1": float(volume),
        "rsi": round(float(simulated_rsi), 2),
        "hour": hour,
        "dayofweek": dayofweek,
        "day": day,
        "hour_sin": round(float(np.sin(2 * np.pi * hour / 24.0)), 6),
        "hour_cos": round(float(np.cos(2 * np.pi * hour / 24.0)), 6),
        "day_sin": round(float(np.sin(2 * np.pi * day / 31.0)), 6),
        "day_cos": round(float(np.cos(2 * np.pi * day / 31.0)), 6),
        "vol_1h": float(volume),
        "vol_mean_6h": float(volume),
        "vol_std_6h": 0.0,
        "vol_max_6h": float(volume),
        "vol_min_6h": float(volume),
        "Rolling_Std_Close": round(float(spread / 2.0), 2),
        "Rolling_Mean_Close": float(close),
        "taker_buy_ratio_mean_6h": round(float(taker_buy_ratio), 4),
        "trades_mean_6h": 10000.0,
        "fng_value": 50.0,
        "fng_mean_3d": 50.0,
        "onchain_num_tx_change": 0.0,
        "onchain_hash_rate_change": 0.0,
        "onchain_miners_revenue_change": 0.0,

        # Microstructural Telemetry
        "close_price": float(close),
        "high_low_spread_usd": round(spread, 2),
        "volatility_range_pct": round(spread_pct, 3),
        "order_flow_bias": flow_bias,
        "rsi_estimate": round(simulated_rsi, 1),
        "volume_usd": round(volume, 2)
    }


@tool
def get_model_return_forecast(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runs dual inference using the Optuna-tuned XGBoost Regressor and Classifier.
    Returns expected percentage return, predicted dollar move, target close price,
    directional prediction (UP/DOWN), probability, and symmetric fee hurdle check.
    """
    try:
        feature_dict = features or {}
        current_close = float(feature_dict.get("close", feature_dict.get("close_price", feature_dict.get("current_price", 65000.0))))

        # 1. Regressor Inference (Target Return)
        raw_pred_return = 0.0
        if os.path.exists(REGRESSOR_MODEL_PATH):
            model_reg = joblib.load(REGRESSOR_MODEL_PATH)
            model_reg.set_params(device="cpu")

            row_reg = {}
            for col in REGRESSOR_FEATURES:
                val = feature_dict.get(col, 0.0)
                if col == "taker_buy_ratio" and val == 0.0:
                    val = 0.50
                elif col == "fng_value" and val == 0.0:
                    val = 50.0
                elif col == "close_lag_1" and val == 0.0:
                    val = current_close
                row_reg[col] = float(val)

            df_reg = pd.DataFrame([row_reg], columns=REGRESSOR_FEATURES)
            # Model outputs decimal fraction: 0.0035 = +0.35% return
            raw_pred_return = float(model_reg.predict(df_reg)[0])

        # 2. Classifier Inference (Directional Movement & Probability)
        move_code = 1 if raw_pred_return >= 0 else 0
        prob_up = 0.50
        prob_down = 0.50

        if os.path.exists(CLASSIFIER_MODEL_PATH):
            model_clf = joblib.load(CLASSIFIER_MODEL_PATH)
            model_clf.set_params(device="cpu")

            row_clf = {}
            for col in CLASSIFIER_FEATURES:
                val = feature_dict.get(col, 0.0)
                if col == "taker_buy_ratio" and val == 0.0:
                    val = 0.50
                elif col == "fng_value" and val == 0.0:
                    val = 50.0
                elif col == "close_lag_1" and val == 0.0:
                    val = current_close
                row_clf[col] = float(val)

            df_clf = pd.DataFrame([row_clf], columns=CLASSIFIER_FEATURES)
            move_code = int(model_clf.predict(df_clf)[0])
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
            direction_badge = "SIDEWAYS / CHOP ⚪"
        else:
            direction_badge = "BULLISH 🟢" if move_code == 1 else "BEARISH 🔴"

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
    except Exception as e:
        return {
            "expected_return_pct": 0.0,
            "predicted_dollar_move": 0.0,
            "predicted_target_price": round(features.get("close", 65000.0) if features else 65000.0, 2),
            "clears_hurdle": False,
            "hurdle_threshold_pct": 0.05,
            "direction": "SIDEWAYS",
            "direction_badge": "NEUTRAL ⚪",
            "direction_probability_up": 50.0,
            "direction_probability_down": 50.0,
            "confidence": 0.50,
            "status": f"error: {str(e)}"
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
