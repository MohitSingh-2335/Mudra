import pytest
import numpy as np
import pandas as pd
import ta

import config
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import (
    create_features, extract_features_for_inference, validate_inference_features,
    predict_with_validation, predict_proba_with_validation,
)
from src.agents.onchain_agent import merge_onchain
from src.agents.fear_greed_agent import merge_fear_greed


@pytest.fixture(scope="module")
def sample_historical_candles():
    return load_and_clean_data(config.BTCUSDT_1H_CSV).tail(100).reset_index(drop=True)


def test_feature_parity_all_regressor_and_classifier_features(sample_historical_candles):
    df = sample_historical_candles.copy()
    latest_train = create_features(df, drop_na=False, is_training=True).iloc[-1]
    live_features = extract_features_for_inference(df)

    for col in config.REGRESSOR_FEATURES:
        assert float(live_features[col]) == pytest.approx(float(latest_train[col]), rel=1e-5)
    for col in config.CLASSIFIER_FEATURES:
        assert float(live_features[col]) == pytest.approx(float(latest_train[col]), rel=1e-5)


def test_feature_parity_technical_indicators(sample_historical_candles):
    df = sample_historical_candles.copy()
    features = extract_features_for_inference(df)
    assert features["rsi"] == pytest.approx(float(ta.momentum.RSIIndicator(close=df["close"], window=14).rsi().iloc[-1]), rel=1e-5)
    assert features["Rolling_Mean_Close"] == pytest.approx(float(df["close"].rolling(5).mean().iloc[-1]), rel=1e-5)
    assert features["close_lag_1"] == pytest.approx(float(df["close"].iloc[-2]), rel=1e-5)
    ts = df["timestamp"].iloc[-1]
    assert features["hour_sin"] == pytest.approx(np.sin(2 * np.pi * ts.hour / 24.0), rel=1e-5)
    assert features["day_sin"] == pytest.approx(np.sin(2 * np.pi * ts.dayofweek / 7.0), rel=1e-5)


def test_insufficient_history_safely_suppresses_forecast(sample_historical_candles):
    with pytest.raises(ValueError, match="Insufficient historical candles"):
        extract_features_for_inference(sample_historical_candles.tail(20).reset_index(drop=True))


def test_candle_closure_timestamp_excludes_forming_candle(monkeypatch):
    from src.engine.sync import fetch_latest_confirmed_candle
    mock_raw = [[1706781600000 + i * 3600000, "60000", "60100", "59900", "60050", "10", 1706781600000 + (i + 1) * 3600000 - 1, "0", 100, "5", "0", "0"] for i in range(40)]
    monkeypatch.setattr("src.engine.sync._fetch_feed_with_retry", lambda *a, **k: mock_raw)
    c = fetch_latest_confirmed_candle()
    assert c["close"] == 60050.0 and len(c["candles"]) == 40


def test_merge_onchain_prevents_intraday_leakage():
    candle_df = pd.DataFrame({"timestamp": ["2026-10-10 12:00:00", "2026-10-11 05:00:00"], "close": [60000.0, 60200.0]})
    mock_onchain = pd.DataFrame({
        "date": [pd.to_datetime("2026-10-09").date(), pd.to_datetime("2026-10-10").date(), pd.to_datetime("2026-10-11").date()],
        "onchain_num_tx": [100000.0, 200000.0, 300000.0], "onchain_hash_rate": [100.0, 200.0, 300.0], "onchain_miners_revenue_usd": [1000.0, 2000.0, 3000.0],
    })
    merged = merge_onchain(candle_df, onchain_df=mock_onchain)
    assert merged.loc[merged["timestamp"] == "2026-10-11 05:00:00", "onchain_num_tx"].iloc[0] == 200000.0


def test_merge_fear_greed_prevents_leakage():
    candle_df = pd.DataFrame({"timestamp": ["2026-10-11 08:00:00"], "close": [60000.0]})
    mock_fg = pd.DataFrame({"date": [pd.to_datetime("2026-10-10").date(), pd.to_datetime("2026-10-11").date()], "fng_value": [45.0, 75.0]})
    assert merge_fear_greed(candle_df, fng_df=mock_fg)["fng_value"].iloc[0] == 45.0


def test_validate_inference_features_success():
    reg_dict = {col: float(i + 1) for i, col in enumerate(config.REGRESSOR_FEATURES)}
    assert list(validate_inference_features(reg_dict, model_type="regressor").columns) == config.REGRESSOR_FEATURES
    clf_dict = {col: float(i + 1) for i, col in enumerate(config.CLASSIFIER_FEATURES)}
    assert list(validate_inference_features(clf_dict, model_type="classifier").columns) == config.CLASSIFIER_FEATURES


def test_validate_inference_features_schema_violations():
    with pytest.raises(ValueError, match="missing required regressor features"):
        validate_inference_features({col: 1.0 for col in config.REGRESSOR_FEATURES[:-1]}, model_type="regressor")

    shuffled = list(config.REGRESSOR_FEATURES)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    with pytest.raises(ValueError, match="Feature ordering contract violation"):
        validate_inference_features(pd.DataFrame([{col: 1.0 for col in shuffled}], columns=shuffled), model_type="regressor")

    with pytest.raises(ValueError, match="unexpected extra columns"):
        validate_inference_features(pd.DataFrame([{col: 1.0 for col in list(config.REGRESSOR_FEATURES) + ["extra"]}]), model_type="regressor")

    nan_dict = {col: 1.0 for col in config.REGRESSOR_FEATURES}
    nan_dict["rsi"] = float("nan")
    with pytest.raises(ValueError, match="NaNs encountered"):
        validate_inference_features(nan_dict, model_type="regressor")

    inf_dict = {col: 1.0 for col in config.REGRESSOR_FEATURES}
    inf_dict["rsi"] = float("inf")
    with pytest.raises(ValueError, match="Infinite values encountered"):
        validate_inference_features(inf_dict, model_type="regressor")

    str_dict = {col: 1.0 for col in config.REGRESSOR_FEATURES}
    str_dict["rsi"] = "invalid_string"
    with pytest.raises(ValueError, match="Non-numeric"):
        validate_inference_features(str_dict, model_type="regressor")


def test_predict_with_validation():
    class MockModel:
        def predict(self, X):
            return np.array([0.005])

        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])

    m = MockModel()
    assert predict_with_validation(m, {col: 1.0 for col in config.REGRESSOR_FEATURES}, "regressor")[0] == 0.005
    assert predict_proba_with_validation(m, {col: 1.0 for col in config.CLASSIFIER_FEATURES})[0, 1] == 0.7
