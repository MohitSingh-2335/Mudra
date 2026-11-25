from typing import Any, Union
import numpy as np
import pandas as pd
import ta

from config import CLASSIFIER_FEATURES, REGRESSOR_FEATURES


def create_features(df: pd.DataFrame, drop_na: bool = True, is_training: bool = True) -> pd.DataFrame:
    """Computes technical indicators, rolling statistics, cyclical encodings, and microstructure features."""
    df = df.copy()

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y:%m:%d %H:%M:%S")
        except (ValueError, TypeError):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["Price Change"] = df["close"] - df["open"]
    df["Volatility"] = (df["high"] - df["low"]) / df["open"]
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_diff"] = df["close"] - df["open"]

    df["Rolling_Mean_Close"] = df["close"].rolling(5).mean()
    df["Rolling_Std_Close"] = df["close"].rolling(5).std()
    df["rolling_max_6h"] = df["close"].rolling(6).max()
    df["rolling_min_6h"] = df["close"].rolling(6).min()

    # Forward targets disabled during live inference to enforce zero leakage
    if is_training:
        df["Target_Close"] = df["close"].shift(-1)
        df["Target_Movement"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df["Target_Return"] = df["close"].shift(-1) / df["close"] - 1.0

    df["vol_1h"] = df["high"] - df["low"]
    df["vol_mean_6h"] = df["vol_1h"].rolling(6).mean()
    df["vol_std_6h"] = df["vol_1h"].rolling(6).std()
    df["vol_max_6h"] = df["vol_1h"].rolling(6).max()
    df["vol_min_6h"] = df["vol_1h"].rolling(6).min()
    df["return_1h_lag"] = df["close"].pct_change(fill_method=None)
    df["return_mean_6h"] = df["return_1h_lag"].rolling(6).mean()
    df["return_std_6h"] = df["return_1h_lag"].rolling(6).std()

    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["day"] = df["timestamp"].dt.day
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    else:
        for c in ("hour", "dayofweek", "day", "hour_sin", "hour_cos", "day_sin", "day_cos"):
            df[c] = 0.0

    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(close=df["close"]).macd()
    bb = ta.volatility.BollingerBands(close=df["close"])
    df["bb_high"], df["bb_low"] = bb.bollinger_hband(), bb.bollinger_lband()
    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["ema_30"] = df["close"].ewm(span=30).mean()

    df["candle_body_pct"] = (df["close"] - df["open"]) / df["open"]
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["sma_10_dist_pct"] = (df["close"] - df["Rolling_Mean_Close"]) / df["Rolling_Mean_Close"]
    # True 10-hour SMA distance for Model 3 parity without altering Model 2's 5-hour feature
    sma_10_true = df["close"].rolling(10, min_periods=1).mean()
    df["sma_10_true_dist_pct"] = (df["close"] - sma_10_true) / (sma_10_true + 1e-8)
    df["ema_10_dist_pct"] = (df["close"] - df["ema_10"]) / df["ema_10"]
    df["ema_30_dist_pct"] = (df["close"] - df["ema_30"]) / df["ema_30"]
    df["bb_width_pct"] = (df["bb_high"] - df["bb_low"]) / df["close"]
    bb_diff = (df["bb_high"] - df["bb_low"]).replace(0, np.nan)
    df["bb_pct_b"] = ((df["close"] - df["bb_low"]) / bb_diff).fillna(0.5)

    df["close_lag_1"] = df["close"].shift(1)
    df["volume_lag_1"] = df["volume"].shift(1)
    df["price_volatility_interaction"] = df["close"] * df["Volatility"]

    has_taker = "taker_buy_ratio" in df.columns and df["taker_buy_ratio"].notna().any()
    df["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce").fillna(0.5) if has_taker else 0.5
    df["taker_buy_ratio_mean_6h"] = df["taker_buy_ratio"].rolling(6).mean().fillna(0.5) if has_taker else 0.5
    if "taker_buy_base_volume" not in df.columns:
        df["taker_buy_base_volume"] = 0.0

    has_trades = "number_of_trades" in df.columns and df["number_of_trades"].notna().any()
    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0.0) if has_trades else 0.0
    df["trades_mean_6h"] = df["number_of_trades"].rolling(6).mean().fillna(0.0) if has_trades else 0.0

    if "fng_value" not in df.columns:
        df["fng_value"] = 50.0
    df["fng_mean_3d"] = df["fng_value"].rolling(72, min_periods=1).mean()

    for col, src in (
        ("onchain_num_tx_change", "onchain_num_tx"),
        ("onchain_hash_rate_change", "onchain_hash_rate"),
        ("onchain_miners_revenue_change", "onchain_miners_revenue_usd"),
    ):
        df[col] = df[src].pct_change(fill_method=None).fillna(0.0) if (col not in df.columns and src in df.columns) else df.get(col, pd.Series(0.0, index=df.index)).fillna(0.0)
    for c in ("onchain_num_tx", "onchain_hash_rate"):
        if c not in df.columns:
            df[c] = 0.0

    # Model 3 (v3_29stationary) scale-invariant features
    df["return_1h"] = df["close"].pct_change(1, fill_method=None).fillna(0.0)
    df["return_4h"] = df["close"].pct_change(4, fill_method=None).fillna(0.0)
    df["return_24h"] = df["close"].pct_change(24, fill_method=None).fillna(0.0)
    df["hl_range_mean_6h"] = df["hl_range_pct"].rolling(6).mean().fillna(0.0)
    df["hl_range_std_6h"] = df["hl_range_pct"].rolling(6).std().fillna(0.0)
    df["hl_range_max_6h"] = df["hl_range_pct"].rolling(6).max().fillna(0.0)
    df["hl_range_min_6h"] = df["hl_range_pct"].rolling(6).min().fillna(0.0)
    ret_std_6 = df["return_1h"].rolling(6).std()
    ret_std_24 = df["return_1h"].rolling(24).std()
    df["vol_regime_ratio_6h_24h"] = (ret_std_6 / (ret_std_24 + 1e-8)).fillna(1.0)
    df["macd_pct"] = (df["macd"] / (df["close"] + 1e-8)).fillna(0.0)
    macd_sig = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_signal_diff_pct"] = ((df["macd"] - macd_sig) / (df["close"] + 1e-8)).fillna(0.0)

    for raw_c, new_pref in (("volume", "vol"), ("taker_buy_base_volume", "taker"), ("number_of_trades", "trades")):
        log_v = np.log1p(df[raw_c])
        df[f"{new_pref}_log_z24"] = ((log_v - log_v.rolling(24).mean()) / (log_v.rolling(24).std() + 1e-8)).fillna(0.0)
    df["volume_ratio_6h_24h"] = (df["volume"].rolling(6).mean() / (df["volume"].rolling(24).mean() + 1e-8)).fillna(1.0)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna().reset_index(drop=True) if drop_na else df


def extract_features_for_inference(df: pd.DataFrame) -> dict:
    """Extracts features from latest closed candle for live inference."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Market candle sequence cannot be empty.")
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise KeyError(f"Candle sequence missing required OHLCV columns: {missing}")
    if len(df) < 35:
        raise ValueError(f"Insufficient historical candles for feature extraction: received {len(df)}, minimum required is 35.")

    featured_df = create_features(df, drop_na=False, is_training=False)
    latest = featured_df.iloc[-1]

    all_required = sorted(set(REGRESSOR_FEATURES + CLASSIFIER_FEATURES))
    nan_features = [col for col in all_required if pd.isna(latest.get(col))]
    if nan_features:
        raise ValueError(f"Feature parity violation: NaNs encountered in computed features: {nan_features}")

    out: dict = {col: (int(latest[col]) if col in {"hour", "dayofweek", "day"} else float(latest[col])) for col in all_required}
    for col in featured_df.columns:
        if col not in out and pd.notna(latest[col]):
            val = latest[col]
            if isinstance(val, (int, np.integer)):
                out[col] = int(val)
            elif isinstance(val, (float, np.floating)):
                out[col] = float(val)

    for k in ("close", "open", "high", "low", "volume"):
        out[k] = float(latest[k])

    if "timestamp" in latest and pd.notna(latest["timestamp"]):
        out["timestamp"] = pd.to_datetime(latest["timestamp"]).isoformat()
    out["number_of_trades"] = int(latest["number_of_trades"]) if "number_of_trades" in latest and pd.notna(latest["number_of_trades"]) and latest["number_of_trades"] > 0 else None

    spread = float(latest["high"] - latest["low"])
    close_price = float(latest["close"])
    taker_ratio = float(out.get("taker_buy_ratio", 0.50))
    flow_bias = "Bullish Aggressive Buying (Takers Sweeping Book)" if taker_ratio > 0.55 else ("Bearish Aggressive Selling (Inventory Dumping)" if taker_ratio < 0.45 else "Neutral Balanced Liquidity")

    out.update({
        "close_price": close_price,
        "high_low_spread_usd": round(spread, 2),
        "volatility_range_pct": round((spread / close_price) * 100.0 if close_price > 0 else 0.0, 3),
        "order_flow_bias": flow_bias,
        "rsi_estimate": round(float(out["rsi"]), 1),
        "volume_usd": round(float(latest["volume"]), 2),
    })
    return out


def validate_inference_features(features: Union[pd.DataFrame, dict], model_type: str = "regressor") -> pd.DataFrame:
    """Enforces column presence, canonical ordering, numeric types, and zero NaN/Inf before inference."""
    mtype = model_type.strip().lower()
    if mtype == "regressor":
        expected = REGRESSOR_FEATURES
    elif mtype == "classifier":
        expected = CLASSIFIER_FEATURES
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected 'regressor' or 'classifier'.")

    if isinstance(features, dict):
        missing = [col for col in expected if col not in features]
        if missing:
            raise ValueError(f"Feature parity violation: missing required {mtype} features: {missing}")
        df = pd.DataFrame([{col: features[col] for col in expected}], columns=expected)
    elif isinstance(features, pd.DataFrame):
        if features.empty:
            raise ValueError(f"Input DataFrame for {mtype} inference cannot be empty.")
        missing = [col for col in expected if col not in features.columns]
        if missing:
            raise ValueError(f"Feature parity violation: missing required {mtype} features: {missing}")
        if list(features.columns)[:len(expected)] != expected:
            raise ValueError(f"Feature ordering contract violation for {mtype}: Expected canonical order: {expected}, got: {list(features.columns)[:len(expected)]}")
        extra = [c for c in features.columns if c not in expected]
        if extra:
            raise ValueError(f"Feature schema violation for {mtype}: Input DataFrame contains unexpected extra columns: {extra}")
        df = features[expected].copy()
    else:
        raise ValueError(f"Unsupported input type for features: {type(features).__name__}. Expected dict or pd.DataFrame.")

    for col in expected:
        if pd.api.types.is_bool_dtype(df[col]) or (df[col].dtype == object and df[col].map(lambda v: isinstance(v, (bool, np.bool_))).any()):
            raise ValueError(f"Feature integrity violation: Boolean values not permitted in {mtype} input feature '{col}'.")
        if pd.api.types.is_complex_dtype(df[col]) or (df[col].dtype == object and df[col].map(lambda v: isinstance(v, (complex, np.complexfloating))).any()):
            raise ValueError(f"Feature integrity violation: Complex values not permitted in {mtype} input feature '{col}'.")
        if not (pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col])):
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                raise ValueError(f"Feature integrity violation: Non-numeric values encountered in {mtype} input feature '{col}'.")

    if df.isna().any().any():
        raise ValueError(f"Feature integrity violation: NaNs encountered in {mtype} input features: {[c for c in expected if df[c].isna().any()]}")
    if np.isinf(df[expected]).any().any():
        raise ValueError(f"Feature integrity violation: Infinite values encountered in {mtype} input features: {[c for c in expected if np.isinf(df[c]).any()]}")
    return df


def predict_with_validation(model: Any, features: Union[pd.DataFrame, dict], model_type: str = "regressor") -> np.ndarray:
    if model is None or not hasattr(model, "predict"):
        raise ValueError("Invalid model provided: must implement .predict().")
    validated_df = validate_inference_features(features, model_type=model_type)
    preds = np.asarray(model.predict(validated_df))
    if preds.size == 0 or np.isnan(preds).any() or np.isinf(preds).any():
        raise ValueError(f"Model inference integrity violation: Invalid predictions returned by {model_type} model.")
    return preds


def predict_proba_with_validation(model: Any, features: Union[pd.DataFrame, dict]) -> np.ndarray:
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("Invalid classifier provided: must implement .predict_proba().")
    validated_df = validate_inference_features(features, model_type="classifier")
    probs = np.asarray(model.predict_proba(validated_df))
    if probs.size == 0 or np.isnan(probs).any() or np.isinf(probs).any():
        raise ValueError("Classifier predict_proba integrity violation: Probabilities contain non-finite values (NaN or Inf).")
    if (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError("Classifier predict_proba integrity violation: Probabilities must fall strictly within [0.0, 1.0].")
    return probs
