# config.py
"""
Mudra Institutional Quantitative Configuration Registry.
Central source of truth for file paths, feature schemas, model artifacts,
and trading parameters.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Institutional Paper Portfolio & Execution Invariants
INITIAL_CAPITAL_USD = float(os.getenv("INITIAL_CAPITAL_USD", 10000.0))
TAKER_FEE_PCT = float(os.getenv("TAKER_FEE_PCT", 0.0005))        # 0.05% Binance VIP-0 Taker Fee
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", 0.0002))          # 0.02% Conservative Execution Slippage
MIN_HURDLE_PCT = float(os.getenv("MIN_HURDLE_PCT", 0.05))        # 0.05% Hurdle Rate for Directional Signals

# Active Model Artifact Paths
MODELS_DIR = str(BASE_DIR / "artifacts" / "models")
REGRESSOR_MODEL_PATH = str(BASE_DIR / "artifacts" / "models" / "best_regressor_model.pkl")
CLASSIFIER_MODEL_PATH = str(BASE_DIR / "artifacts" / "models" / "best_classifier_model.pkl")

# Auxiliary / Checkpoint Artifact Paths
XGB_MODEL_PATH = str(BASE_DIR / "artifacts" / "models" / "best_xgb_model.pkl")
LSTM_MODEL_PATH = str(BASE_DIR / "artifacts" / "models" / "patchtst_weights.pt")
SVC_MODEL_PATH = str(BASE_DIR / "artifacts" / "models" / "best_svc_model.pkl")
SCALER_PATH = str(BASE_DIR / "artifacts" / "models" / "scaler.pkl")

# Data File Paths
THRESHOLD_SWEEP = str(BASE_DIR / "data" / "results" / "threshold_sweep.csv")
BTCUSDT_1H_CSV = str(BASE_DIR / "data" / "raw" / "BTCUSDT-1H.csv")
FEATURED_BTC_DATA_PATH = str(BASE_DIR / "data" / "processed" / "featured_btc_data.csv")

# MLflow Tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{BASE_DIR / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = "Mudra_BTC_Models"

# SHAP Explainability Paths
SHAP_REGRESSOR_SUMMARY_PATH = str(BASE_DIR / "artifacts" / "shap_regressor_summary.png")
SHAP_CLASSIFIER_SUMMARY_PATH = str(BASE_DIR / "artifacts" / "shap_classifier_summary.png")
SHAP_IMPORTANCE_CSV_PATH = str(BASE_DIR / "artifacts" / "shap_feature_importance.csv")

# ChromaDB Vector Store
CHROMADB_DIR = str(BASE_DIR / "data" / "chromadb")
CHROMA_COLLECTION_NAME = "mudra_market_knowledge_v3"

# Asset Registry
ASSETS = {
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "SOL/USDT": "SOL-USD",
    "BNB/USDT": "BNB-USD",
    "XRP/USDT": "XRP-USD",
    "Apple (AAPL)": "AAPL",
    "Nvidia (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
}

# Ingestion / Pipeline Constants
FETCH_DAYS = 59
LIVE_FETCH_LIMIT = 100
LOOKBACK_WINDOW = 48

# Regressor Feature Schema (26 features)
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

# Classifier Feature Schema (39 features)
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

__all__ = [
    "BASE_DIR",
    "INITIAL_CAPITAL_USD",
    "TAKER_FEE_PCT",
    "SLIPPAGE_PCT",
    "MIN_HURDLE_PCT",
    "MODELS_DIR",
    "REGRESSOR_MODEL_PATH",
    "CLASSIFIER_MODEL_PATH",
    "XGB_MODEL_PATH",
    "LSTM_MODEL_PATH",
    "SVC_MODEL_PATH",
    "SCALER_PATH",
    "THRESHOLD_SWEEP",
    "BTCUSDT_1H_CSV",
    "FEATURED_BTC_DATA_PATH",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "SHAP_REGRESSOR_SUMMARY_PATH",
    "SHAP_CLASSIFIER_SUMMARY_PATH",
    "SHAP_IMPORTANCE_CSV_PATH",
    "CHROMADB_DIR",
    "CHROMA_COLLECTION_NAME",
    "ASSETS",
    "FETCH_DAYS",
    "LIVE_FETCH_LIMIT",
    "LOOKBACK_WINDOW",
    "REGRESSOR_FEATURES",
    "CLASSIFIER_FEATURES",
]