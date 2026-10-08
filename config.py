#config.py

# This is where we can change all the features and path 
# so that every where this setting can be changes 
# and we don't have to change every place like hardcoded codes

#Path

XGB_MODEL_PATH = 'artifacts/models/best_xgb_model.pkl'
LSTM_MODEL_PATH = 'artifacts/models/patchtst_weights.pt'
SVC_MODEL_PATH = 'artifacts/models/best_svc_model.pkl'
SCALER_PATH = 'artifacts/models/scaler.pkl'

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

#XGBoost features for price prediction

XGB_FEATURES = [
    'volume', 'Price Change', 'Rolling_Std_Close',
    'vol_1h', 'vol_mean_6h', 'vol_std_6h', 'vol_max_6h', 'vol_min_6h',
    'hour', 'dayofweek', 'day', 'rsi',
    'high_low_ratio', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'close_lag_1',  # price-level anchor — was missing, caused RMSE worse than naive baseline
    'taker_buy_ratio', 'taker_buy_ratio_mean_6h', 'trades_mean_6h',
    'fng_value', 'fng_mean_3d',
    'onchain_num_tx_change', 'onchain_hash_rate_change', 'onchain_miners_revenue_change',
    'sentiment_score', 'sentiment_mean_3d'
]

#SVC features for direction prediction

SVC_FEATURES = [
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
    'onchain_num_tx_change', 'onchain_hash_rate_change', 'onchain_miners_revenue_change',
    'sentiment_score', 'sentiment_mean_3d'
]

#App settings

APP_TITLE = "AI Trading Suite 🤖"
APP_VERSION = "1.0.0"
APP_ICON = "🤖"