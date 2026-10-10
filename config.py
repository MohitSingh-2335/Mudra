#config.py

# This is where we can change all the features and path 
# so that every where this setting can be changes 
# and we don't have to change every place like hardcoded codes

#Artifacts paths

#Active
REGRESSOR_MODEL_PATH = 'artifacts/models/best_regressor_model.pkl'
CLASSIFIER_MODEL_PATH = 'artifacts/models/best_classifier_model.pkl'
MODELS_DIR = 'artifacts/models'

#Non-active
XGB_MODEL_PATH = 'artifacts/models/best_xgb_model.pkl'
LSTM_MODEL_PATH = 'artifacts/models/patchtst_weights.pt'
SVC_MODEL_PATH = 'artifacts/models/best_svc_model.pkl'
SCALER_PATH = 'artifacts/models/scaler.pkl'

#Data paths

THRESHOLD_SWEEP = 'data/results/threshold_sweep.csv'
BTCUSDT_1H_CSV = 'data/raw/BTCUSDT-1H.csv'
FEATURED_BTC_DATA_PATH = 'data/processed/featured_btc_data.csv'


# MLflow Tracking

MLFLOW_TRACKING_URI = 'file:./mlruns'
MLFLOW_EXPERIMENT_NAME = 'Mudra_BTC_Models'


# SHAP Explainability Paths
SHAP_REGRESSOR_SUMMARY_PATH = 'artifacts/shap_regressor_summary.png'
SHAP_CLASSIFIER_SUMMARY_PATH = 'artifacts/shap_classifier_summary.png'
SHAP_IMPORTANCE_CSV_PATH = 'artifacts/shap_feature_importance.csv'


# ChromaDB Vector Store
CHROMADB_DIR = 'data/chromadb'
CHROMA_COLLECTION_NAME = 'mudra_market_knowledge'


#Assets (for now we need to increase this later)

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

#Data setting

FETCH_DAYS = 59
LIVE_FETCH_LIMIT = 100
LOOKBACK_WINDOW = 48

#Features for price prediction

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

#Features for direction prediction

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