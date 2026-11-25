import os
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict

BASE_DIR = Path(__file__).resolve().parent


def _parse_execution_cost_rate(rate_var: str, pct_var: str, default_rate: Optional[float] = None, max_rate: float = 0.01, max_pct: float = 5.0) -> float:
    # parses execution cost rates from environment variables with bounds checking
    rate_str, pct_str = os.getenv(rate_var), os.getenv(pct_var)
    rate_val: Optional[float] = None
    if rate_str and rate_str.strip():
        try:
            rate_val = float(rate_str)
        except ValueError as err:
            raise ValueError(f"Invalid non-numeric value for {rate_var}='{rate_str}'.") from err
        if not math.isfinite(rate_val) or rate_val < 0.0 or rate_val > max_rate:
            raise ValueError(f"{rate_var} must be a non-negative finite rate <= {max_rate}.")

    pct_val: Optional[float] = None
    if pct_str and pct_str.strip():
        try:
            pct_val = float(pct_str)
        except ValueError as err:
            raise ValueError(f"Invalid non-numeric value for {pct_var}='{pct_str}'.") from err
        if not math.isfinite(pct_val) or pct_val < 0.0 or pct_val > max_pct:
            raise ValueError(f"{pct_var} must be a finite percentage between 0 and {max_pct}.")

    if rate_val is not None and pct_val is not None:
        if abs(rate_val - pct_val / 100.0) > 1e-7:
            raise ValueError(f"Conflicting environment configuration between {rate_var} and {pct_var}.")
        return rate_val
    if rate_val is not None:
        return rate_val
    if pct_val is not None:
        return pct_val / 100.0
    if default_rate is not None:
        return default_rate
    raise ValueError(f"Neither {rate_var} nor {pct_var} was provided.")


def _parse_positive_int(var_name: str, default_val: int) -> int:
    val_str = os.getenv(var_name)
    if not val_str or not val_str.strip():
        return default_val
    val = int(val_str.strip())
    if val <= 0:
        raise ValueError(f"{var_name} must be a positive integer, got {val}")
    return val


def _parse_positive_float(var_name: str, default_val: float) -> float:
    val_str = os.getenv(var_name)
    if not val_str or not val_str.strip():
        return default_val
    val = float(val_str.strip())
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{var_name} must be a finite positive float, got {val}")
    return val


# Execution friction settings
INITIAL_CAPITAL_USD = float(os.getenv("INITIAL_CAPITAL_USD", 10000.0))
PORTFOLIO_ADVISORY_LOCK_ID: int = int(os.getenv("PORTFOLIO_ADVISORY_LOCK_ID", "888001"))
TAKER_FEE_RATE: float = _parse_execution_cost_rate("TAKER_FEE_RATE", "TAKER_FEE_PCT", default_rate=0.0005)
SLIPPAGE_RATE: float = _parse_execution_cost_rate("SLIPPAGE_RATE", "SLIPPAGE_PCT", default_rate=0.0002)
TOTAL_FRICTION_PER_SIDE_RATE: float = TAKER_FEE_RATE + SLIPPAGE_RATE
ROUND_TRIP_FRICTION_RATE: float = 1.0 - (1.0 - TOTAL_FRICTION_PER_SIDE_RATE) ** 2

_hurdle_rate_env, _hurdle_pct_env = os.getenv("MIN_HURDLE_RATE"), os.getenv("MIN_HURDLE_PCT")
if (_hurdle_rate_env and _hurdle_rate_env.strip()) or (_hurdle_pct_env and _hurdle_pct_env.strip()):
    MIN_HURDLE_RATE: float = _parse_execution_cost_rate("MIN_HURDLE_RATE", "MIN_HURDLE_PCT", default_rate=ROUND_TRIP_FRICTION_RATE, max_rate=0.05)
    _floor_dec = Decimal("1") - (Decimal("1") - (Decimal(str(TAKER_FEE_RATE)) + Decimal(str(SLIPPAGE_RATE)))) ** 2
    _configured_dec = Decimal(_hurdle_rate_env.strip()) if _hurdle_rate_env and _hurdle_rate_env.strip() else Decimal(_hurdle_pct_env.strip()) / Decimal("100")
    if _configured_dec < _floor_dec:
        raise ValueError("Configured MIN_HURDLE_RATE cannot be less than estimated ROUND_TRIP_FRICTION_RATE.")
else:
    MIN_HURDLE_RATE: float = ROUND_TRIP_FRICTION_RATE

TAKER_FEE_PCT: float = TAKER_FEE_RATE * 100.0
SLIPPAGE_PCT: float = SLIPPAGE_RATE * 100.0
TOTAL_FRICTION_PER_SIDE_PCT: float = TOTAL_FRICTION_PER_SIDE_RATE * 100.0
ROUND_TRIP_FRICTION_PCT: float = ROUND_TRIP_FRICTION_RATE * 100.0
MIN_HURDLE_PCT: float = MIN_HURDLE_RATE * 100.0

# Artifact paths and hashes
MODELS_DIR = str(BASE_DIR / "artifacts" / "models")
REGRESSOR_MODEL_PATH = str(BASE_DIR / "artifacts" / "models" / "best_regressor_model.pkl")
CLASSIFIER_MODEL_PATH = str(BASE_DIR / "artifacts" / "models" / "best_classifier_model.pkl")
MODEL_MANIFEST_PATH = str(BASE_DIR / "artifacts" / "models" / "model_manifest.json")
ACTIVE_MODEL_VERSION: str = "xgb_onnx_v1.0.0"
FEATURE_SCHEMA_VERSION: str = "v1_39features"
REGRESSOR_MODEL_SHA256: str = "b44fba18cf2c7b51e3297e83a946d869392803a267e951a5e85ea052c63eb854"
CLASSIFIER_MODEL_SHA256: str = "df139c1018ed5025b1444d6597c3d98b4d7c377a898487268d18e58d253ae684"

THRESHOLD_SWEEP = str(BASE_DIR / "data" / "results" / "threshold_sweep.csv")
BTCUSDT_1H_CSV = str(BASE_DIR / "data" / "raw" / "BTCUSDT-1H.csv")
FEATURED_BTC_DATA_PATH = str(BASE_DIR / "data" / "processed" / "featured_btc_data.csv")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{BASE_DIR / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = "Mudra_BTC_Models"
SHAP_REGRESSOR_SUMMARY_PATH = str(BASE_DIR / "artifacts" / "shap_regressor_summary.png")
SHAP_CLASSIFIER_SUMMARY_PATH = str(BASE_DIR / "artifacts" / "shap_classifier_summary.png")
SHAP_IMPORTANCE_CSV_PATH = str(BASE_DIR / "artifacts" / "shap_feature_importance.csv")
CHROMADB_DIR = str(BASE_DIR / "data" / "chromadb")
CHROMA_COLLECTION_NAME = "mudra_rag_minilm_v1"

FETCH_DAYS = 59
LIVE_FETCH_LIMIT = 100
LOOKBACK_WINDOW = 48
DEFAULT_WALK_FORWARD_WINDOWS: int = _parse_positive_int("DEFAULT_WALK_FORWARD_WINDOWS", 5)
DEFAULT_WALK_FORWARD_TRAIN_FRAC: float = _parse_positive_float("DEFAULT_WALK_FORWARD_TRAIN_FRAC", 0.60)
DEFAULT_SIGNIFICANCE_ALPHA: float = _parse_positive_float("DEFAULT_SIGNIFICANCE_ALPHA", 0.05)

EXCHANGE_RETRY_MAX_ATTEMPTS: int = _parse_positive_int("EXCHANGE_RETRY_MAX_ATTEMPTS", 3)
EXCHANGE_CONNECT_TIMEOUT: float = _parse_positive_float("EXCHANGE_CONNECT_TIMEOUT", 3.5)
EXCHANGE_READ_TIMEOUT: float = _parse_positive_float("EXCHANGE_READ_TIMEOUT", 7.0)
EXCHANGE_RETRY_BASE_BACKOFF: float = _parse_positive_float("EXCHANGE_RETRY_BASE_BACKOFF", 1.0)
EXCHANGE_RETRY_BACKOFF_FACTOR: float = _parse_positive_float("EXCHANGE_RETRY_BACKOFF_FACTOR", 2.0)
EXCHANGE_RETRY_MAX_BACKOFF: float = _parse_positive_float("EXCHANGE_RETRY_MAX_BACKOFF", 6.0)
EXCHANGE_MAX_RETRY_AFTER_CAP: float = _parse_positive_float("EXCHANGE_MAX_RETRY_AFTER_CAP", 15.0)
EXCHANGE_RETRY_TOTAL_DEADLINE: float = _parse_positive_float("EXCHANGE_RETRY_TOTAL_DEADLINE", 25.0)

REGRESSOR_FEATURES = [
    'volume', 'Price Change', 'Rolling_Std_Close',
    'vol_1h', 'vol_mean_6h', 'vol_std_6h', 'vol_max_6h', 'vol_min_6h',
    'hour', 'dayofweek', 'day', 'rsi',
    'high_low_ratio', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'close_lag_1',
    'taker_buy_ratio', 'taker_buy_ratio_mean_6h', 'trades_mean_6h',
    'fng_value', 'fng_mean_3d',
    'onchain_num_tx_change', 'onchain_hash_rate_change', 'onchain_miners_revenue_change'
]

CLASSIFIER_FEATURES = [
    'volume', 'Price Change', 'Volatility',
    'Rolling_Mean_Close', 'Rolling_Std_Close',
    'vol_mean_6h', 'vol_std_6h', 'vol_max_6h', 'vol_min_6h',
    'return_mean_6h', 'return_std_6h',
    'hour', 'dayofweek', 'day',
    'rsi', 'macd', 'bb_high', 'bb_low', 'ema_10', 'ema_30',
    'high_low_ratio', 'close_open_diff',
    'close_lag_1', 'volume_lag_1',
    'rolling_max_6h', 'rolling_min_6h',
    'price_volatility_interaction',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'taker_buy_ratio', 'taker_buy_ratio_mean_6h', 'trades_mean_6h',
    'fng_value', 'fng_mean_3d',
    'onchain_num_tx_change', 'onchain_hash_rate_change', 'onchain_miners_revenue_change'
]

if os.path.exists(MODEL_MANIFEST_PATH):
    try:
        with open(MODEL_MANIFEST_PATH, "r", encoding="utf-8") as _mf:
            _manifest = json.load(_mf)
        ACTIVE_MODEL_VERSION = _manifest.get("model_version", ACTIVE_MODEL_VERSION)
        FEATURE_SCHEMA_VERSION = _manifest.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
        REGRESSOR_MODEL_SHA256 = _manifest.get("regressor_sha256", REGRESSOR_MODEL_SHA256)
        CLASSIFIER_MODEL_SHA256 = _manifest.get("classifier_sha256", CLASSIFIER_MODEL_SHA256)
        if "regressor_features" in _manifest:
            REGRESSOR_FEATURES = _manifest["regressor_features"]
        if "classifier_features" in _manifest:
            CLASSIFIER_FEATURES = _manifest["classifier_features"]
    except Exception:
        pass

APP_TITLE: str = "Mudra — Bitcoin Price Prediction & Model Monitoring"
APP_SUBTITLE: str = "Live BTC/USDT Price, Next-Hour ML Forecast & Multi-Model Evaluation"

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "taker_buy_ratio": "Buyer vs. Seller Volume", "taker_buy_ratio_mean_6h": "6-Hour Average Buyer Ratio",
    "vol_mean_6h": "6-Hour Average Price Swing ($)", "vol_std_6h": "6-Hour Swing Change ($)",
    "vol_max_6h": "Largest 1h Swing in Last 6h ($)", "vol_min_6h": "Smallest 1h Swing in Last 6h ($)",
    "vol_1h": "1-Hour High–Low Price Range ($)", "Volatility": "Hourly Price Volatility ($)",
    "high_low_spread": "1-Hour High–Low Price Range ($)", "Price Change": "1-Hour Price Change ($)",
    "close_open_diff": "1-Hour Open-to-Close Difference ($)", "close_lag_1": "Previous Hour Closing Price ($)",
    "volume_lag_1": "Previous Hour Trading Volume", "Rolling_Mean_Close": "10-Hour Moving Average Price ($)",
    "Rolling_Std_Close": "10-Hour Price Volatility ($)", "return_mean_6h": "6-Hour Average Return %",
    "return_std_6h": "6-Hour Return Volatility %", "return_1h": "1-Hour Price Trend %",
    "return_3h": "3-Hour Price Trend %", "return_6h": "6-Hour Price Trend %",
    "rolling_max_6h": "6-Hour Rolling Highest Price ($)", "rolling_min_6h": "6-Hour Rolling Lowest Price ($)",
    "price_volatility_interaction": "Price x Volatility Interaction", "rsi": "Price Momentum (RSI)",
    "macd": "Trend Direction (MACD)", "macd_signal": "MACD Signal Line", "macd_hist": "MACD Histogram Divergence",
    "bb_high": "Upper Bollinger Band ($)", "bb_low": "Lower Bollinger Band ($)", "bb_mid": "Bollinger Middle Basis ($)",
    "ema_10": "10-Hour Exponential Moving Average ($)", "ema_20": "20-Hour Exponential Moving Average ($)",
    "ema_30": "30-Hour Exponential Moving Average ($)", "ema_50": "50-Hour Exponential Moving Average ($)",
    "ema_10_30_diff": "Fast vs. Slow EMA Trend Spread ($)", "volume": "Trading Volume",
    "Volume_Change": "Hourly Volume Growth Rate", "trades_mean_6h": "6-Hour Average Number of Trades",
    "number_of_trades": "Number of Trades", "fng_value": "Market Sentiment (Fear & Greed)",
    "fng_mean_3d": "3-Day Average Sentiment (Fear & Greed)", "onchain_num_tx_change": "On-Chain Transaction Growth Rate %",
    "onchain_hash_rate_change": "Bitcoin Network Hashrate Change %", "onchain_miners_revenue_change": "Miner Revenue Growth Rate %",
    "hour": "Hour of Day (UTC)", "dayofweek": "Day of Week", "day": "Day of Month",
    "hour_sin": "Cyclical Hour (Sine)", "hour_cos": "Cyclical Hour (Cosine)",
    "day_sin": "Cyclical Day (Sine)", "day_cos": "Cyclical Day (Cosine)",
    "high_low_ratio": "High-to-Low Ratio",
}

__all__ = [
    "BASE_DIR", "INITIAL_CAPITAL_USD", "PORTFOLIO_ADVISORY_LOCK_ID", "TAKER_FEE_RATE",
    "SLIPPAGE_RATE", "TOTAL_FRICTION_PER_SIDE_RATE", "ROUND_TRIP_FRICTION_RATE", "MIN_HURDLE_RATE",
    "TAKER_FEE_PCT", "SLIPPAGE_PCT", "TOTAL_FRICTION_PER_SIDE_PCT", "ROUND_TRIP_FRICTION_PCT",
    "MIN_HURDLE_PCT", "_parse_execution_cost_rate", "MODELS_DIR", "REGRESSOR_MODEL_PATH",
    "CLASSIFIER_MODEL_PATH", "MODEL_MANIFEST_PATH", "ACTIVE_MODEL_VERSION", "FEATURE_SCHEMA_VERSION",
    "REGRESSOR_MODEL_SHA256", "CLASSIFIER_MODEL_SHA256", "THRESHOLD_SWEEP", "BTCUSDT_1H_CSV",
    "FEATURED_BTC_DATA_PATH", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME",
    "SHAP_REGRESSOR_SUMMARY_PATH", "SHAP_CLASSIFIER_SUMMARY_PATH", "SHAP_IMPORTANCE_CSV_PATH",
    "CHROMADB_DIR", "CHROMA_COLLECTION_NAME", "FETCH_DAYS", "LIVE_FETCH_LIMIT",
    "LOOKBACK_WINDOW", "DEFAULT_WALK_FORWARD_WINDOWS", "DEFAULT_WALK_FORWARD_TRAIN_FRAC",
    "DEFAULT_SIGNIFICANCE_ALPHA", "EXCHANGE_RETRY_MAX_ATTEMPTS", "EXCHANGE_CONNECT_TIMEOUT",
    "EXCHANGE_READ_TIMEOUT", "EXCHANGE_RETRY_BASE_BACKOFF", "EXCHANGE_RETRY_BACKOFF_FACTOR",
    "EXCHANGE_RETRY_MAX_BACKOFF", "EXCHANGE_MAX_RETRY_AFTER_CAP", "EXCHANGE_RETRY_TOTAL_DEADLINE",
    "REGRESSOR_FEATURES", "CLASSIFIER_FEATURES", "APP_TITLE", "APP_SUBTITLE", "FEATURE_DISPLAY_NAMES",
]
