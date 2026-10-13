# src/feature_engineering.py
"""
Feature engineering module for Mudra quantitative trading system.
Transforms raw OHLCV and market microstructure into 39 technical features
and generates continuous return and binary movement target variables.
"""

from typing import Optional, Union, Any, List, Dict
import numpy as np
import pandas as pd
import ta


def create_features(
    df: pd.DataFrame,
    drop_na: bool = True,
    is_training: bool = True
) -> pd.DataFrame:
    """
    Engineers technical indicators, cyclical time embeddings, order flow microstructure,
    macro sentiment, on-chain dynamics, and target labels.

    Args:
        df (pd.DataFrame): Raw candlestick DataFrame with OHLCV and timestamp.
        drop_na (bool): Whether to drop initial rolling window NaNs (default: True).
        is_training (bool): Whether to generate forward-shifted targets for training (default: True).

    Returns:
        pd.DataFrame: Engineered DataFrame containing all model features and targets.
    """
    df = df.copy()

    # Ensure timestamp column is datetime object
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y:%m:%d %H:%M:%S")
        except (ValueError, TypeError):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Basic price and volatility features
    df["Price Change"] = df["close"] - df["open"]
    df["Volatility"] = (df["high"] - df["low"]) / df["open"]
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_diff"] = df["close"] - df["open"]

    # Rolling window features
    df["Rolling_Mean_Close"] = df["close"].rolling(window=5).mean()
    df["Rolling_Std_Close"] = df["close"].rolling(window=5).std()
    df["rolling_max_6h"] = df["close"].rolling(window=6).max()
    df["rolling_min_6h"] = df["close"].rolling(window=6).min()

    # Target variables for model training (strictly disabled during live inference)
    if is_training:
        df["Target_Close"] = df["close"].shift(-1)
        df["Target_Movement"] = (df["close"].shift(-1) > df["close"]).astype(int)
        # Continuous fractional return target: (Close_{t+1} - Close_t) / Close_t
        df["Target_Return"] = df["close"].shift(-1) / df["close"] - 1.0

    # Volatility and historical return features
    window = 6
    df["vol_1h"] = df["high"] - df["low"]
    df["vol_mean_6h"] = df["vol_1h"].rolling(window).mean()
    df["vol_std_6h"] = df["vol_1h"].rolling(window).std()
    df["vol_max_6h"] = df["vol_1h"].rolling(window).max()
    df["vol_min_6h"] = df["vol_1h"].rolling(window).min()

    # Backward-looking historical lag return (strictly preserves temporal causality)
    df["return_1h_lag"] = df["close"].pct_change(fill_method=None)
    df["return_mean_6h"] = df["return_1h_lag"].rolling(window).mean()
    df["return_std_6h"] = df["return_1h_lag"].rolling(window).std()

    # Time-based and cyclical sinusoidal encodings
    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["day"] = df["timestamp"].dt.day
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    else:
        for col in ["hour", "dayofweek", "day", "hour_sin", "hour_cos", "day_sin", "day_cos"]:
            df[col] = 0.0

    # Technical indicators
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(close=df["close"]).macd()
    bb = ta.volatility.BollingerBands(close=df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["ema_30"] = df["close"].ewm(span=30).mean()

    # Lag and interaction features
    df["close_lag_1"] = df["close"].shift(1)
    df["volume_lag_1"] = df["volume"].shift(1)
    df["price_volatility_interaction"] = df["close"] * df["Volatility"]

    # Order-flow microstructure features (neutral defaults if feed lacks taker/trades data)
    if "taker_buy_ratio" in df.columns and df["taker_buy_ratio"].notna().any():
        df["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce").fillna(0.5)
        df["taker_buy_ratio_mean_6h"] = df["taker_buy_ratio"].rolling(window=6).mean().fillna(0.5)
    else:
        df["taker_buy_ratio"] = 0.5
        df["taker_buy_ratio_mean_6h"] = 0.5

    if "number_of_trades" in df.columns and df["number_of_trades"].notna().any():
        df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0.0)
        df["trades_mean_6h"] = df["number_of_trades"].rolling(window=6).mean().fillna(0.0)
    else:
        df["number_of_trades"] = 0.0
        df["trades_mean_6h"] = 0.0

    # Fear & Greed sentiment features (neutral 50.0 default if feed lacks sentiment)
    if "fng_value" in df.columns:
        df["fng_mean_3d"] = df["fng_value"].rolling(window=72).mean()
    else:
        df["fng_value"] = 50.0
        df["fng_mean_3d"] = 50.0

    # On-chain network dynamics (day-over-day changes, preserve upstream daily changes if present)
    if "onchain_num_tx_change" not in df.columns:
        if "onchain_num_tx" in df.columns:
            df["onchain_num_tx_change"] = df["onchain_num_tx"].pct_change(fill_method=None)
        else:
            df["onchain_num_tx_change"] = 0.0

    if "onchain_hash_rate_change" not in df.columns:
        if "onchain_hash_rate" in df.columns:
            df["onchain_hash_rate_change"] = df["onchain_hash_rate"].pct_change(fill_method=None)
        else:
            df["onchain_hash_rate_change"] = 0.0

    if "onchain_miners_revenue_change" not in df.columns:
        if "onchain_miners_revenue_usd" in df.columns:
            df["onchain_miners_revenue_change"] = df["onchain_miners_revenue_usd"].pct_change(fill_method=None)
        else:
            df["onchain_miners_revenue_change"] = 0.0

    # Clean rolling window NaNs and reset index
    if drop_na:
        df = df.dropna().reset_index(drop=True)

    return df


def extract_features_for_inference(df: pd.DataFrame) -> dict:
    """
    Extracts engineered features for the latest confirmed candlestick in df.
    Strictly enforces training/live parity by running create_features on the historical sequence.

    Args:
        df (pd.DataFrame): Candlestick sequence with OHLCV columns.
                           Must contain at least 35 candles to ensure rolling window convergence.

    Returns:
        dict: Comprehensive dictionary containing all 26 REGRESSOR_FEATURES,
              all 39 CLASSIFIER_FEATURES, raw OHLCV metrics, and telemetry indicators.

    Raises:
        ValueError: If df is None, empty, contains fewer than 35 candles, or has NaN values.
        KeyError: If required OHLCV columns are missing.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Market candle sequence cannot be empty.")

    required_cols = {"open", "high", "low", "close", "volume"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"Candle sequence missing required OHLCV columns: {missing_cols}")

    min_lookback = 35
    if len(df) < min_lookback:
        raise ValueError(
            f"Insufficient historical candles for feature extraction: received {len(df)}, "
            f"minimum required is {min_lookback}."
        )

    # Compute features over entire history without shifting future targets or dropping rows
    featured_df = create_features(df, drop_na=False, is_training=False)
    latest = featured_df.iloc[-1]

    # Lazy import to avoid circular dependency
    from config import REGRESSOR_FEATURES, CLASSIFIER_FEATURES

    # Validate that all required model features are non-null
    all_required_features = sorted(list(set(REGRESSOR_FEATURES + CLASSIFIER_FEATURES)))
    nan_features = [col for col in all_required_features if pd.isna(latest.get(col))]
    if nan_features:
        raise ValueError(f"Feature parity violation: NaNs encountered in computed features: {nan_features}")

    # Build comprehensive output dictionary
    out: dict = {}
    for col in all_required_features:
        val = latest[col]
        out[col] = int(val) if col in {"hour", "dayofweek", "day"} else float(val)

    # Raw candlestick values
    out["close"] = float(latest["close"])
    out["open"] = float(latest["open"])
    out["high"] = float(latest["high"])
    out["low"] = float(latest["low"])
    out["volume"] = float(latest["volume"])
    if "timestamp" in latest and pd.notna(latest["timestamp"]):
        out["timestamp"] = latest["timestamp"]
    if "number_of_trades" in latest and pd.notna(latest["number_of_trades"]) and latest["number_of_trades"] > 0:
        out["number_of_trades"] = int(latest["number_of_trades"])
    else:
        out["number_of_trades"] = None

    # Microstructural Telemetry
    spread = float(latest["high"] - latest["low"])
    close_price = float(latest["close"])
    spread_pct = (spread / close_price) * 100.0 if close_price > 0 else 0.0
    taker_ratio = float(out.get("taker_buy_ratio", 0.50))

    if taker_ratio > 0.55:
        flow_bias = "Bullish Aggressive Buying (Takers Sweeping Book)"
    elif taker_ratio < 0.45:
        flow_bias = "Bearish Aggressive Selling (Inventory Dumping)"
    else:
        flow_bias = "Neutral Balanced Liquidity"

    out["close_price"] = close_price
    out["high_low_spread_usd"] = round(spread, 2)
    out["volatility_range_pct"] = round(spread_pct, 3)
    out["order_flow_bias"] = flow_bias
    out["rsi_estimate"] = round(float(out["rsi"]), 1)
    out["volume_usd"] = round(float(latest["volume"]), 2)

    return out


def validate_inference_features(
    features: Union[pd.DataFrame, dict],
    model_type: str = "regressor"
) -> pd.DataFrame:
    """
    Validates feature contracts and enforces strict canonical column ordering
    prior to model inference. Prevents silent feature transposition, missing
    columns, NaN/infinite propagation, and schema mismatch.

    Args:
        features: Input feature mapping (dict) or DataFrame.
        model_type: "regressor" (26 features) or "classifier" (39 features).

    Returns:
        pd.DataFrame: A validated DataFrame with columns strictly ordered
                      according to config.REGRESSOR_FEATURES or config.CLASSIFIER_FEATURES.

    Raises:
        ValueError: If model_type is unsupported, if required features are missing,
                    if input DataFrame columns are shuffled or out of order,
                    or if feature values contain NaNs or infinite values.
    """
    from config import REGRESSOR_FEATURES, CLASSIFIER_FEATURES

    model_type_normalized = model_type.strip().lower()
    if model_type_normalized == "regressor":
        expected_features = REGRESSOR_FEATURES
    elif model_type_normalized == "classifier":
        expected_features = CLASSIFIER_FEATURES
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Expected 'regressor' or 'classifier'."
        )

    if isinstance(features, dict):
        missing = [col for col in expected_features if col not in features]
        if missing:
            raise ValueError(
                f"Feature contract violation: Input dictionary missing required {model_type_normalized} features: {missing}"
            )
        # Build DataFrame with exact expected order
        df = pd.DataFrame([{col: features[col] for col in expected_features}], columns=expected_features)
    elif isinstance(features, pd.DataFrame):
        if features.empty:
            raise ValueError(f"Input DataFrame for {model_type_normalized} inference cannot be empty.")

        missing = [col for col in expected_features if col not in features.columns]
        if missing:
            raise ValueError(
                f"Feature contract violation: Input DataFrame missing required {model_type_normalized} features: {missing}"
            )

        # Enforce exact column order on the first len(expected_features) columns
        actual_cols = list(features.columns)
        if actual_cols[:len(expected_features)] != expected_features:
            raise ValueError(
                f"Feature ordering contract violation for {model_type_normalized}: "
                f"Input columns are shuffled or out of order. "
                f"Expected canonical order: {expected_features}, "
                f"got: {actual_cols[:len(expected_features)]}"
            )

        extra_cols = [c for c in actual_cols if c not in expected_features]
        if extra_cols:
            raise ValueError(
                f"Feature schema violation for {model_type_normalized}: "
                f"Input DataFrame contains unexpected extra columns: {extra_cols}"
            )

        df = features[expected_features].copy()
    else:
        raise ValueError(
            f"Unsupported input type for features: {type(features).__name__}. Expected dict or pd.DataFrame."
        )

    # Validate NaN values
    if df.isna().any().any():
        nan_cols = [c for c in expected_features if df[c].isna().any()]
        raise ValueError(
            f"Feature integrity violation: NaNs encountered in {model_type_normalized} input features: {nan_cols}"
        )

    # Validate Infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if np.isinf(numeric_df).any().any():
        inf_cols = [c for c in numeric_df.columns if np.isinf(numeric_df[c]).any()]
        raise ValueError(
            f"Feature integrity violation: Infinite values encountered in {model_type_normalized} input features: {inf_cols}"
        )

    return df


def predict_with_validation(
    model: Any,
    features: Union[pd.DataFrame, dict],
    model_type: str = "regressor"
) -> np.ndarray:
    """
    Validates feature contracts and executes model.predict() on strictly ordered columns.

    Args:
        model: Trained model artifact with a .predict() method.
        features: Input feature mapping (dict) or DataFrame.
        model_type: "regressor" or "classifier".

    Returns:
        np.ndarray: Model predictions.
    """
    if model is None or not hasattr(model, "predict"):
        raise ValueError("Invalid model provided: must implement .predict().")
    validated_df = validate_inference_features(features, model_type=model_type)
    return model.predict(validated_df)


def predict_proba_with_validation(
    model: Any,
    features: Union[pd.DataFrame, dict]
) -> np.ndarray:
    """
    Validates classifier feature contracts and executes model.predict_proba().

    Args:
        model: Trained classifier artifact with a .predict_proba() method.
        features: Input feature mapping (dict) or DataFrame.

    Returns:
        np.ndarray: Predicted class probabilities.
    """
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("Invalid classifier provided: must implement .predict_proba().")
    validated_df = validate_inference_features(features, model_type="classifier")
    return model.predict_proba(validated_df)