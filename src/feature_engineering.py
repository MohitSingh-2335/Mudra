# src/feature_engineering.py
from typing import Any, Union
import numpy as np
import pandas as pd
import ta


def create_features(df: pd.DataFrame, drop_na: bool = True, is_training: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Parse timestamp
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y:%m:%d %H:%M:%S")
        except (ValueError, TypeError):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Price and range features
    df["Price Change"] = df["close"] - df["open"]
    df["Volatility"] = (df["high"] - df["low"]) / df["open"]
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_diff"] = df["close"] - df["open"]

    # Rolling price statistics
    df["Rolling_Mean_Close"] = df["close"].rolling(5).mean()
    df["Rolling_Std_Close"] = df["close"].rolling(5).std()
    df["rolling_max_6h"] = df["close"].rolling(6).max()
    df["rolling_min_6h"] = df["close"].rolling(6).min()

    # Shifted target variables for supervised learning
    if is_training:
        df["Target_Close"] = df["close"].shift(-1)
        df["Target_Movement"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df["Target_Return"] = df["close"].shift(-1) / df["close"] - 1.0

    # Volatility and historical return features
    df["vol_1h"] = df["high"] - df["low"]
    df["vol_mean_6h"] = df["vol_1h"].rolling(6).mean()
    df["vol_std_6h"] = df["vol_1h"].rolling(6).std()
    df["vol_max_6h"] = df["vol_1h"].rolling(6).max()
    df["vol_min_6h"] = df["vol_1h"].rolling(6).min()
    df["return_1h_lag"] = df["close"].pct_change(fill_method=None)
    df["return_mean_6h"] = df["return_1h_lag"].rolling(6).mean()
    df["return_std_6h"] = df["return_1h_lag"].rolling(6).std()

    # Cyclical hour and day sinusoidal encodings
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

    # Technical momentum and volatility indicators
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(close=df["close"]).macd()
    bb = ta.volatility.BollingerBands(close=df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["ema_30"] = df["close"].ewm(span=30).mean()

    # Stationary scale-invariant percentage distance ratios
    df["candle_body_pct"] = (df["close"] - df["open"]) / df["open"]
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["sma_10_dist_pct"] = (df["close"] - df["Rolling_Mean_Close"]) / df["Rolling_Mean_Close"]
    df["ema_10_dist_pct"] = (df["close"] - df["ema_10"]) / df["ema_10"]
    df["ema_30_dist_pct"] = (df["close"] - df["ema_30"]) / df["ema_30"]
    df["bb_width_pct"] = (df["bb_high"] - df["bb_low"]) / df["close"]
    bb_diff = (df["bb_high"] - df["bb_low"]).replace(0, np.nan)
    df["bb_pct_b"] = ((df["close"] - df["bb_low"]) / bb_diff).fillna(0.5)

    # Lags and interaction features
    df["close_lag_1"] = df["close"].shift(1)
    df["volume_lag_1"] = df["volume"].shift(1)
    df["price_volatility_interaction"] = df["close"] * df["Volatility"]

    # Order flow microstructure metrics
    if "taker_buy_ratio" in df.columns and df["taker_buy_ratio"].notna().any():
        df["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce").fillna(0.5)
        df["taker_buy_ratio_mean_6h"] = df["taker_buy_ratio"].rolling(6).mean().fillna(0.5)
    else:
        df["taker_buy_ratio"] = 0.5
        df["taker_buy_ratio_mean_6h"] = 0.5

    if "taker_buy_base_volume" not in df.columns:
        df["taker_buy_base_volume"] = 0.0

    if "number_of_trades" in df.columns and df["number_of_trades"].notna().any():
        df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0.0)
        df["trades_mean_6h"] = df["number_of_trades"].rolling(6).mean().fillna(0.0)
    else:
        df["number_of_trades"] = 0.0
        df["trades_mean_6h"] = 0.0

    # Macro sentiment features
    if "fng_value" in df.columns:
        df["fng_mean_3d"] = df["fng_value"].rolling(72).mean()
    else:
        df["fng_value"] = 50.0
        df["fng_mean_3d"] = 50.0

    # On-chain network dynamics
    for col, src in [("onchain_num_tx_change", "onchain_num_tx"),
                     ("onchain_hash_rate_change", "onchain_hash_rate"),
                     ("onchain_miners_revenue_change", "onchain_miners_revenue_usd")]:
        if col not in df.columns:
            df[col] = df[src].pct_change(fill_method=None).fillna(0.0) if src in df.columns else 0.0
        else:
            df[col] = df[col].fillna(0.0)

    for col in ["onchain_num_tx", "onchain_hash_rate"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna().reset_index(drop=True) if drop_na else df


def extract_features_for_inference(df: pd.DataFrame) -> dict:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Market candle sequence cannot be empty.")
    missing_cols = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing_cols:
        raise KeyError(f"Candle sequence missing required OHLCV columns: {missing_cols}")
    if len(df) < 35:
        raise ValueError(f"Insufficient historical candles for feature extraction: received {len(df)}, minimum required is 35.")

    featured_df = create_features(df, drop_na=False, is_training=False)
    latest = featured_df.iloc[-1]

    from config import REGRESSOR_FEATURES, CLASSIFIER_FEATURES
    all_required = sorted(list(set(REGRESSOR_FEATURES + CLASSIFIER_FEATURES)))
    nan_features = [col for col in all_required if pd.isna(latest.get(col))]
    if nan_features:
        raise ValueError(f"Feature parity violation: NaNs encountered in computed features: {nan_features}")

    out: dict = {}
    for col in all_required:
        val = latest[col]
        out[col] = int(val) if col in {"hour", "dayofweek", "day"} else float(val)

    for col in featured_df.columns:
        if col not in out and col in latest and pd.notna(latest[col]):
            val = latest[col]
            if isinstance(val, (int, np.integer)):
                out[col] = int(val)
            elif isinstance(val, (float, np.floating)):
                out[col] = float(val)

    for k in ["close", "open", "high", "low", "volume"]:
        out[k] = float(latest[k])

    if "timestamp" in latest and pd.notna(latest["timestamp"]):
        out["timestamp"] = pd.to_datetime(latest["timestamp"]).isoformat()
    out["number_of_trades"] = int(latest["number_of_trades"]) if "number_of_trades" in latest and pd.notna(latest["number_of_trades"]) and latest["number_of_trades"] > 0 else None

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


def validate_inference_features(features: Union[pd.DataFrame, dict], model_type: str = "regressor") -> pd.DataFrame:
    from config import REGRESSOR_FEATURES, CLASSIFIER_FEATURES
    model_type_norm = model_type.strip().lower()
    if model_type_norm == "regressor":
        expected = REGRESSOR_FEATURES
    elif model_type_norm == "classifier":
        expected = CLASSIFIER_FEATURES
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected 'regressor' or 'classifier'.")

    if isinstance(features, dict):
        missing = [col for col in expected if col not in features]
        if missing:
            raise ValueError(f"Feature parity violation: missing required {model_type_norm} features: {missing}")
        df = pd.DataFrame([{col: features[col] for col in expected}], columns=expected)
    elif isinstance(features, pd.DataFrame):
        if features.empty:
            raise ValueError(f"Input DataFrame for {model_type_norm} inference cannot be empty.")
        missing = [col for col in expected if col not in features.columns]
        if missing:
            raise ValueError(f"Feature parity violation: missing required {model_type_norm} features: {missing}")
        actual_cols = list(features.columns)
        if actual_cols[:len(expected)] != expected:
            raise ValueError(
                f"Feature ordering contract violation for {model_type_norm}: "
                f"Input columns are shuffled or out of order. "
                f"Expected canonical order: {expected}, got: {actual_cols[:len(expected)]}"
            )
        extra = [c for c in actual_cols if c not in expected]
        if extra:
            raise ValueError(f"Feature schema violation for {model_type_norm}: Input DataFrame contains unexpected extra columns: {extra}")
        df = features[expected].copy()
    else:
        raise ValueError(f"Unsupported input type for features: {type(features).__name__}. Expected dict or pd.DataFrame.")

    for col in expected:
        if pd.api.types.is_bool_dtype(df[col]) or (df[col].dtype == object and df[col].map(lambda v: isinstance(v, (bool, np.bool_))).any()):
            raise ValueError(f"Feature integrity violation: Boolean values not permitted in {model_type_norm} input feature '{col}'.")
        if pd.api.types.is_complex_dtype(df[col]) or (df[col].dtype == object and df[col].map(lambda v: isinstance(v, (complex, np.complexfloating))).any()):
            raise ValueError(f"Feature integrity violation: Complex values not permitted in {model_type_norm} input feature '{col}'.")
        if not (pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col])):
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                raise ValueError(f"Feature integrity violation: Non-numeric values encountered in {model_type_norm} input feature '{col}'.")
            if not (pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col])):
                raise ValueError(f"Feature integrity violation: Non-numeric values encountered in {model_type_norm} input feature '{col}'.")

    if df.isna().any().any():
        nan_cols = [c for c in expected if df[c].isna().any()]
        raise ValueError(f"Feature integrity violation: NaNs encountered in {model_type_norm} input features: {nan_cols}")
    if np.isinf(df[expected]).any().any():
        inf_cols = [c for c in expected if np.isinf(df[c]).any()]
        raise ValueError(f"Feature integrity violation: Infinite values encountered in {model_type_norm} input features: {inf_cols}")
    return df


def predict_with_validation(model: Any, features: Union[pd.DataFrame, dict], model_type: str = "regressor") -> np.ndarray:
    if model is None or not hasattr(model, "predict"):
        raise ValueError("Invalid model provided: must implement .predict().")
    validated_df = validate_inference_features(features, model_type=model_type)
    preds = np.asarray(model.predict(validated_df))
    if preds.size == 0:
        raise ValueError(f"Model inference integrity violation: {model_type} model returned empty predictions.")
    if np.isnan(preds).any() or np.isinf(preds).any():
        raise ValueError(f"Model inference integrity violation: Non-finite values (NaN or Inf) returned by {model_type} model.")
    return preds


def predict_proba_with_validation(model: Any, features: Union[pd.DataFrame, dict]) -> np.ndarray:
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("Invalid classifier provided: must implement .predict_proba().")
    validated_df = validate_inference_features(features, model_type="classifier")
    probs = np.asarray(model.predict_proba(validated_df))
    if probs.size == 0:
        raise ValueError("Classifier predict_proba integrity violation: returned empty probabilities.")
    if np.isnan(probs).any() or np.isinf(probs).any():
        raise ValueError("Classifier predict_proba integrity violation: Probabilities contain non-finite values (NaN or Inf).")
    if (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError("Classifier predict_proba integrity violation: Probabilities must fall strictly within [0.0, 1.0].")
    return probs
