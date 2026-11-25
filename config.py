import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
_logger = logging.getLogger("mudra.config")


def _parse_bounded_float(var_name: str, max_val: float, min_val: float = 0.0, msg_label: str = "rate") -> Optional[float]:
    raw = os.getenv(var_name)
    if not raw or not raw.strip():
        return None
    try:
        val = float(raw)
    except ValueError as err:
        raise ValueError(f"Invalid non-numeric value for {var_name}='{raw}'.") from err
    if not math.isfinite(val) or val < min_val or val > max_val:
        raise ValueError(f"{var_name} must be a finite {msg_label} between {min_val} and {max_val}.")
    return val


def _parse_execution_cost_rate(
    rate_var: str, pct_var: str, default_rate: Optional[float] = None,
    max_rate: float = 0.01, max_pct: float = 5.0, min_rate: float = 0.0,
) -> float:
    # Parses execution cost rates or percentages from env vars with bounds and cross-checks
    rate_val = _parse_bounded_float(rate_var, max_rate, msg_label="rate")
    pct_val = _parse_bounded_float(pct_var, max_pct, msg_label="percentage")

    if rate_val is not None and pct_val is not None:
        if abs(rate_val - pct_val / 100.0) > 1e-7:
            raise ValueError(f"Conflicting environment configuration between {rate_var} and {pct_var}.")
        chosen = rate_val
    elif rate_val is not None:
        chosen = rate_val
    elif pct_val is not None:
        chosen = pct_val / 100.0
    elif default_rate is not None:
        chosen = default_rate
    else:
        raise ValueError(f"Neither {rate_var} nor {pct_var} was provided.")

    if chosen + 1e-12 < min_rate:
        raise ValueError(f"Configured {rate_var} cannot be less than floor {min_rate:.6f}.")
    return chosen


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


# Execution friction settings (0.05% taker + 0.02% slippage per side)
INITIAL_CAPITAL_USD = float(os.getenv("INITIAL_CAPITAL_USD", 10000.0))
PORTFOLIO_ADVISORY_LOCK_ID: int = int(os.getenv("PORTFOLIO_ADVISORY_LOCK_ID", "888001"))
TAKER_FEE_RATE: float = _parse_execution_cost_rate("TAKER_FEE_RATE", "TAKER_FEE_PCT", default_rate=0.0005)
SLIPPAGE_RATE: float = _parse_execution_cost_rate("SLIPPAGE_RATE", "SLIPPAGE_PCT", default_rate=0.0002)
TOTAL_FRICTION_PER_SIDE_RATE: float = TAKER_FEE_RATE + SLIPPAGE_RATE
ROUND_TRIP_FRICTION_RATE: float = 1.0 - (1.0 - TOTAL_FRICTION_PER_SIDE_RATE) ** 2
MIN_HURDLE_RATE: float = _parse_execution_cost_rate(
    "MIN_HURDLE_RATE", "MIN_HURDLE_PCT", default_rate=ROUND_TRIP_FRICTION_RATE,
    max_rate=0.05, min_rate=ROUND_TRIP_FRICTION_RATE,
)

TAKER_FEE_PCT: float = TAKER_FEE_RATE * 100.0
SLIPPAGE_PCT: float = SLIPPAGE_RATE * 100.0
TOTAL_FRICTION_PER_SIDE_PCT: float = TOTAL_FRICTION_PER_SIDE_RATE * 100.0
ROUND_TRIP_FRICTION_PCT: float = ROUND_TRIP_FRICTION_RATE * 100.0
MIN_HURDLE_PCT: float = MIN_HURDLE_RATE * 100.0

# Artifact paths and cryptographic manifest defaults
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

LIVE_FETCH_LIMIT = 100
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
    except Exception as exc:
        _logger.warning("Could not parse model manifest %s: %s", MODEL_MANIFEST_PATH, exc)

__all__ = [
    "BASE_DIR", "INITIAL_CAPITAL_USD", "PORTFOLIO_ADVISORY_LOCK_ID", "TAKER_FEE_RATE",
    "SLIPPAGE_RATE", "TOTAL_FRICTION_PER_SIDE_RATE", "ROUND_TRIP_FRICTION_RATE", "MIN_HURDLE_RATE",
    "TAKER_FEE_PCT", "SLIPPAGE_PCT", "TOTAL_FRICTION_PER_SIDE_PCT", "ROUND_TRIP_FRICTION_PCT",
    "MIN_HURDLE_PCT", "_parse_execution_cost_rate", "MODELS_DIR", "REGRESSOR_MODEL_PATH",
    "CLASSIFIER_MODEL_PATH", "MODEL_MANIFEST_PATH", "ACTIVE_MODEL_VERSION", "FEATURE_SCHEMA_VERSION",
    "REGRESSOR_MODEL_SHA256", "CLASSIFIER_MODEL_SHA256", "THRESHOLD_SWEEP", "BTCUSDT_1H_CSV",
    "FEATURED_BTC_DATA_PATH", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME",
    "SHAP_REGRESSOR_SUMMARY_PATH", "SHAP_CLASSIFIER_SUMMARY_PATH", "SHAP_IMPORTANCE_CSV_PATH",
    "CHROMADB_DIR", "CHROMA_COLLECTION_NAME", "LIVE_FETCH_LIMIT",
    "EXCHANGE_RETRY_MAX_ATTEMPTS", "EXCHANGE_CONNECT_TIMEOUT", "EXCHANGE_READ_TIMEOUT",
    "EXCHANGE_RETRY_BASE_BACKOFF", "EXCHANGE_RETRY_BACKOFF_FACTOR", "EXCHANGE_RETRY_MAX_BACKOFF",
    "EXCHANGE_MAX_RETRY_AFTER_CAP", "EXCHANGE_RETRY_TOTAL_DEADLINE",
    "REGRESSOR_FEATURES", "CLASSIFIER_FEATURES",
]
