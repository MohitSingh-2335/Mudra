# tests/test_feature_parity.py
"""
Regression test suite for Training vs. Live Feature Parity.
Verifies that feature engineering for live inference produces identical values,
units, column order, and semantics as the training pipeline on historical candlestick sequences.
"""

import pytest
import numpy as np
import pandas as pd
import ta
import joblib

import config
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features, extract_features_for_inference
from src.agents.tools import (
    calculate_technical_signals,
    get_model_return_forecast,
    fetch_historical_candles
)


@pytest.fixture(scope="module")
def sample_historical_candles():
    """Loads 100 real-shaped historical candles from the canonical dataset."""
    df = load_and_clean_data(config.BTCUSDT_1H_CSV)
    return df.tail(100).reset_index(drop=True)


def test_feature_parity_all_regressor_and_classifier_features(sample_historical_candles):
    """
    Assert that extract_features_for_inference produces identical feature values
    to create_features (training pipeline) for all 26 regressor and 39 classifier features.
    """
    df = sample_historical_candles.copy()

    # 1. Compute via training transformation pipeline
    df_train = create_features(df, drop_na=False, is_training=True)
    latest_train = df_train.iloc[-1]

    # 2. Compute via shared inference extraction path
    live_features = extract_features_for_inference(df)

    # 3. Check all 26 Regressor Features
    for col in config.REGRESSOR_FEATURES:
        expected_val = float(latest_train[col])
        actual_val = float(live_features[col])
        assert actual_val == pytest.approx(expected_val, rel=1e-5), (
            f"Regressor feature parity failed for '{col}': expected {expected_val}, got {actual_val}"
        )

    # 4. Check all 39 Classifier Features
    for col in config.CLASSIFIER_FEATURES:
        expected_val = float(latest_train[col])
        actual_val = float(live_features[col])
        assert actual_val == pytest.approx(expected_val, rel=1e-5), (
            f"Classifier feature parity failed for '{col}': expected {expected_val}, got {actual_val}"
        )


def test_feature_parity_rolling_window_indicators(sample_historical_candles):
    """Verify rolling window close and volatility statistics match pandas definitions exactly."""
    df = sample_historical_candles.copy()
    features = extract_features_for_inference(df)

    expected_mean_close = float(df["close"].rolling(5).mean().iloc[-1])
    expected_std_close = float(df["close"].rolling(5).std().iloc[-1])
    expected_max_6h = float(df["close"].rolling(6).max().iloc[-1])
    expected_min_6h = float(df["close"].rolling(6).min().iloc[-1])

    vol_1h = df["high"] - df["low"]
    expected_vol_mean_6h = float(vol_1h.rolling(6).mean().iloc[-1])
    expected_vol_std_6h = float(vol_1h.rolling(6).std().iloc[-1])

    assert features["Rolling_Mean_Close"] == pytest.approx(expected_mean_close, rel=1e-5)
    assert features["Rolling_Std_Close"] == pytest.approx(expected_std_close, rel=1e-5)
    assert features["rolling_max_6h"] == pytest.approx(expected_max_6h, rel=1e-5)
    assert features["rolling_min_6h"] == pytest.approx(expected_min_6h, rel=1e-5)
    assert features["vol_mean_6h"] == pytest.approx(expected_vol_mean_6h, rel=1e-5)
    assert features["vol_std_6h"] == pytest.approx(expected_vol_std_6h, rel=1e-5)


def test_feature_parity_rsi(sample_historical_candles):
    """Verify RSI is the true 14-period indicator, not a fabricated taker-ratio approximation."""
    df = sample_historical_candles.copy()
    features = extract_features_for_inference(df)

    expected_rsi = float(ta.momentum.RSIIndicator(close=df["close"], window=14).rsi().iloc[-1])
    assert features["rsi"] == pytest.approx(expected_rsi, rel=1e-5)

    # Confirm it does not match the old broken formula: 50 + (taker_ratio - 0.5) * 60
    taker_ratio = float(df.get("taker_buy_ratio", pd.Series([0.5]*len(df))).iloc[-1])
    old_fake_rsi = 50.0 + (taker_ratio - 0.5) * 60.0
    if abs(expected_rsi - old_fake_rsi) > 1.0:
        assert features["rsi"] != pytest.approx(old_fake_rsi, abs=0.5)


def test_feature_parity_lagged_values(sample_historical_candles):
    """Verify lagged close and volume features strictly reference row t-1, preserving causality."""
    df = sample_historical_candles.copy()
    features = extract_features_for_inference(df)

    expected_close_lag_1 = float(df["close"].iloc[-2])
    expected_vol_lag_1 = float(df["volume"].iloc[-2])

    assert features["close_lag_1"] == pytest.approx(expected_close_lag_1, rel=1e-5)
    assert features["volume_lag_1"] == pytest.approx(expected_vol_lag_1, rel=1e-5)
    # Ensure close_lag_1 is NOT equal to the current candle close (no zero lag)
    if df["close"].iloc[-1] != df["close"].iloc[-2]:
        assert features["close_lag_1"] != features["close"]


def test_feature_parity_time_encodings(sample_historical_candles):
    """Verify cyclical time embeddings use 24h and 7-day periods (dayofweek/7, NOT day/31)."""
    df = sample_historical_candles.copy()
    features = extract_features_for_inference(df)

    ts = df["timestamp"].iloc[-1]
    h = ts.hour
    w = ts.dayofweek

    expected_hour_sin = float(np.sin(2 * np.pi * h / 24.0))
    expected_hour_cos = float(np.cos(2 * np.pi * h / 24.0))
    expected_day_sin = float(np.sin(2 * np.pi * w / 7.0))
    expected_day_cos = float(np.cos(2 * np.pi * w / 7.0))

    assert features["hour_sin"] == pytest.approx(expected_hour_sin, rel=1e-5)
    assert features["hour_cos"] == pytest.approx(expected_hour_cos, rel=1e-5)
    assert features["day_sin"] == pytest.approx(expected_day_sin, rel=1e-5)
    assert features["day_cos"] == pytest.approx(expected_day_cos, rel=1e-5)


def test_model_feature_columns_and_ordering(sample_historical_candles):
    """
    Verify that trained models strictly require REGRESSOR_FEATURES (26)
    and CLASSIFIER_FEATURES (39) in the exact order, and dual inference succeeds.
    """
    reg_model = joblib.load(config.REGRESSOR_MODEL_PATH)
    clf_model = joblib.load(config.CLASSIFIER_MODEL_PATH)

    assert list(reg_model.feature_names_in_) == config.REGRESSOR_FEATURES
    assert list(clf_model.feature_names_in_) == config.CLASSIFIER_FEATURES
    assert len(reg_model.feature_names_in_) == 26
    assert len(clf_model.feature_names_in_) == 39

    # Test tool invocation with parity-compliant features
    df = sample_historical_candles.copy()
    signals = calculate_technical_signals.invoke({"candles": df})
    forecast = get_model_return_forecast.invoke({"features": signals})

    assert forecast["status"] == "success"
    assert "expected_return_pct" in forecast
    assert "direction" in forecast
    assert forecast["direction"] in ["UP", "DOWN"]
    assert 0.0 <= forecast["confidence"] <= 1.0


def test_explicit_failure_on_insufficient_history(sample_historical_candles):
    """
    Verify that extract_features_for_inference explicitly raises ValueError
    when fewer than 35 historical candles are provided, refusing to fabricate data.
    """
    df = sample_historical_candles.copy()

    # Test short sequences: 5, 20, 34 rows
    for short_len in [5, 20, 34]:
        with pytest.raises(ValueError, match="Insufficient historical candles"):
            extract_features_for_inference(df.head(short_len))

    # Test empty DataFrame
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_features_for_inference(pd.DataFrame())


def test_rejection_of_missing_features_in_forecast():
    """
    Verify that get_model_return_forecast explicitly raises ValueError
    when required features are missing rather than substituting 0.0 defaults.
    """
    incomplete_features = {
        "close": 65000.0,
        "rsi": 55.0,
        # Missing remaining 38 required features
    }

    with pytest.raises(ValueError, match="Feature parity violation"):
        get_model_return_forecast.invoke({"features": incomplete_features})
