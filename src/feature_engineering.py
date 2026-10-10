# src/feature_engineering.py

import pandas as pd
import numpy as np
import ta

def create_features(df):
    """
    Engineers a comprehensive set of features for the BTCUSDT dataset.

    Args:
        df (pandas.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume', 'timestamp'.

    Returns:
        pandas.DataFrame: The DataFrame with all new features.
    """
    # Basic price and volatility features
    df['Price Change'] = df['close'] - df['open']
    df['Volatility'] = (df['high'] - df['low']) / df['open']
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_diff'] = df['close'] - df['open']

    # Rolling window features
    df['Rolling_Mean_Close'] = df['close'].rolling(window=5).mean()
    df['Rolling_Std_Close'] = df['close'].rolling(window=5).std()
    df['rolling_max_6h'] = df['close'].rolling(window=6).max()
    df['rolling_min_6h'] = df['close'].rolling(window=6).min()

    # Target variables for prediction
    df['Target_Close'] = df['close'].shift(-1)
    df['Target_Movement'] = (df['close'].shift(-1) > df['close']).astype(int)
    # Return-based target: percentage change to next hour's close, instead of
    # the absolute price. Predicting absolute price lost to a do-nothing
    # baseline in testing across every model tried (Linear/RF/XGBoost/SVR).
    # Predicting the small change instead is the standard fix — it bakes in
    # "price mostly persists, only adjust slightly" instead of hoping the
    # model discovers that on its own.
    df['Target_Return'] = df['close'].shift(-1) / df['close'] - 1

    # Volatility and return features
    window = 6
    df['vol_1h'] = df['high'] - df['low']
    df['vol_mean_6h'] = df['vol_1h'].rolling(window).mean()
    df['vol_std_6h']  = df['vol_1h'].rolling(window).std()
    df['vol_max_6h']  = df['vol_1h'].rolling(window).max()
    df['vol_min_6h']  = df['vol_1h'].rolling(window).min()
    # FIX: the old return_1h was computed with .shift(-1), making it a
    # FORWARD-looking value (next hour's return — same sign as
    # Target_Movement). The rolling mean/std below were built FROM that
    # column, so they leaked next-hour's return into the current row's
    # features. return_1h_lag below is backward-looking only (safe to use
    # as a feature); nothing here looks past the current row anymore.
    df['return_1h_lag'] = df['close'].pct_change()
    df['return_mean_6h'] = df['return_1h_lag'].rolling(window).mean()
    df['return_std_6h'] = df['return_1h_lag'].rolling(window).std()

    # Time-based and cyclical features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['day'] = df['timestamp'].dt.day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Technical indicators using the 'ta' library
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    bb = ta.volatility.BollingerBands(close=df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_30'] = df['close'].ewm(span=30).mean()

    # Lag and interaction features
    df['close_lag_1'] = df['close'].shift(1)
    df['volume_lag_1'] = df['volume'].shift(1)
    df['price_volatility_interaction'] = df['close'] * df['Volatility']

    # Order-flow features — default to neutral 50% ratio and 0 trades if missing
    if 'taker_buy_ratio' in df.columns:
        df['taker_buy_ratio_mean_6h'] = df['taker_buy_ratio'].rolling(window=6).mean()
    else:
        df['taker_buy_ratio'] = 0.5
        df['taker_buy_ratio_mean_6h'] = 0.5

    if 'number_of_trades' in df.columns:
        df['trades_mean_6h'] = df['number_of_trades'].rolling(window=6).mean()
    else:
        df['number_of_trades'] = 0.0
        df['trades_mean_6h'] = 0.0

    if 'fng_value' in df.columns:
        df['fng_mean_3d'] = df['fng_value'].rolling(window=72).mean()
    else:
        df['fng_value'] = 50.0
        df['fng_mean_3d'] = 50.0

    # On-chain metrics — daily values broadcast from merge_onchain(). Using
    # day-over-day % change rather than raw level: raw level is highly
    # autocorrelated/non-stationary (hash rate trends up for years), so
    # change is more likely to carry hourly-relevant signal. Same
    # daily-into-hourly granularity caveat as fng_value applies here too.
    if 'onchain_num_tx' in df.columns:
        df['onchain_num_tx_change'] = df['onchain_num_tx'].pct_change(fill_method=None)
    if 'onchain_hash_rate' in df.columns:
        df['onchain_hash_rate_change'] = df['onchain_hash_rate'].pct_change(fill_method=None)
    if 'onchain_miners_revenue_usd' in df.columns:
        df['onchain_miners_revenue_change'] = df['onchain_miners_revenue_usd'].pct_change(fill_method=None)

    # Sentiment — daily FinBERT score broadcast from merge_sentiment().
    # Same daily-into-hourly granularity caveat as fng_value/onchain applies.
    if 'sentiment_score' in df.columns:
        df['sentiment_mean_3d'] = df['sentiment_score'].rolling(window=72).mean()


    # Drop rows with NaN values created by feature engineering
    df.dropna(inplace=True)

    return df