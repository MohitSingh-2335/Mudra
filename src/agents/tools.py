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

from config import (
    REGRESSOR_MODEL_PATH,
    REGRESSOR_FEATURES,
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
    taker_buy_ratio: float = 0.5
) -> Dict[str, Any]:
    """
    Computes key microstructural and technical momentum indicators from price and order-flow inputs.
    Calculates volatility range, order-flow buyer dominance, and trend bias.
    """
    spread = high - low
    spread_pct = (spread / close) * 100.0 if close > 0 else 0.0

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

    return {
        "close_price": close,
        "high_low_spread_usd": round(spread, 2),
        "volatility_range_pct": round(spread_pct, 3),
        "taker_buy_ratio": round(taker_buy_ratio, 4),
        "order_flow_bias": flow_bias,
        "rsi_estimate": round(simulated_rsi, 1),
        "volume_usd": round(volume, 2)
    }


@tool
def get_model_return_forecast(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runs inference using the Optuna-tuned XGBoost production regressor.
    Returns the expected 1-hour percentage return (R), predicted dollar move,
    and whether the forecast clears the 0.05% institutional fee hurdle.
    """
    try:
        if not os.path.exists(REGRESSOR_MODEL_PATH):
            return {
                "expected_return_pct": 0.0,
                "predicted_dollar_move": 0.0,
                "clears_hurdle": False,
                "hurdle_threshold_pct": 0.05,
                "status": "model_artifact_missing"
            }

        model = joblib.load(REGRESSOR_MODEL_PATH)
        model.set_params(device="cpu")

        # Construct input vector matching REGRESSOR_FEATURES
        feature_dict = features or {}
        row_data = {}
        for col in REGRESSOR_FEATURES:
            val = feature_dict.get(col, 0.0)
            # Impute sensible neutral defaults
            if col == "taker_buy_ratio" and val == 0.0:
                val = 0.5
            elif col == "fng_value" and val == 0.0:
                val = 50.0
            row_data[col] = float(val)

        df_input = pd.DataFrame([row_data], columns=REGRESSOR_FEATURES)
        pred_return = float(model.predict(df_input)[0])

        current_close = float(feature_dict.get("close", feature_dict.get("current_price", 65000.0)))
        pred_dollar_move = current_close * (pred_return / 100.0)
        target_price = current_close + pred_dollar_move

        clears_hurdle = pred_return >= 0.05

        return {
            "expected_return_pct": round(pred_return, 4),
            "predicted_dollar_move": round(pred_dollar_move, 2),
            "predicted_target_price": round(target_price, 2),
            "clears_hurdle": clears_hurdle,
            "hurdle_threshold_pct": 0.05,
            "status": "success"
        }
    except Exception as e:
        return {
            "expected_return_pct": 0.0,
            "predicted_dollar_move": 0.0,
            "clears_hurdle": False,
            "hurdle_threshold_pct": 0.05,
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
