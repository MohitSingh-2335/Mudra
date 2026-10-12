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


def test_taker_buy_ratio_preserved_when_supplied_in_candles(sample_historical_candles):
    """
    Verify that calculate_technical_signals preserves the observed taker_buy_ratio
    in supplied candles, and only overwrites when an explicit ratio is provided.
    """
    df = sample_historical_candles.copy()
    # Set known non-default taker_buy_ratio on the latest candle
    target_ratio = 0.6854
    df.at[df.index[-1], "taker_buy_ratio"] = target_ratio

    # 1. Invoke without passing taker_buy_ratio override -> MUST PRESERVE 0.6854
    res = calculate_technical_signals.invoke({"candles": df})
    assert res["taker_buy_ratio"] == pytest.approx(target_ratio, abs=1e-4), (
        f"Actual taker_buy_ratio was overwritten! Expected {target_ratio}, got {res['taker_buy_ratio']}"
    )

    # 2. Invoke WITH explicit override -> MUST APPLY explicit override
    explicit_override = 0.7421
    res_override = calculate_technical_signals.invoke({
        "candles": df,
        "taker_buy_ratio": explicit_override
    })
    assert res_override["taker_buy_ratio"] == pytest.approx(explicit_override, abs=1e-4)


def test_candle_closure_timestamp_excludes_forming_candle(monkeypatch):
    """
    Test A: Verify that fetch_historical_candles excludes actively forming candles
    by evaluating close_time against the current time, maintaining ascending chronological order.
    """
    from datetime import datetime, timezone
    import requests

    # Anchor to dynamic current UTC time
    now_dt = datetime.now(timezone.utc)
    now_ms = now_dt.timestamp() * 1000.0

    # Build 36 completed closed candles (closed in the past relative to now_ms)
    mock_klines = []
    for i in range(36):
        open_time = now_ms - (38 - i) * 3600000
        close_time = open_time + 3600000 - 1  # Completed in the past
        close_price = 60000.0 + i * 10.0
        mock_klines.append([
            open_time, "60000.0", "60500.0", "59800.0", str(close_price), "100.0",
            close_time, "6020000.0", 1500, "55.0", "3311000.0", "0"
        ])

    # 37th candle is actively forming: close_time is 30 minutes in the future
    forming_open = now_ms - 1800000  # Started 30 min ago
    forming_close = forming_open + 3600000 - 1  # Closes 30 min in the future
    mock_klines.append([
        forming_open, "60200.0", "60800.0", "60100.0", "75000.0", "45.0",
        forming_close, "2727000.0", 800, "20.0", "1212000.0", "0"
    ])

    class MockResponse:
        status_code = 200
        def json(self):
            return mock_klines

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())

    # Fetch historical candles (requesting 35 lookback)
    candles_df = fetch_historical_candles(symbol="BTCUSDT", limit=35)

    # Verify that the unclosed forming candle (close=75000.0) was EXCLUDED strictly by timestamp
    latest_bar_close = candles_df["close"].iloc[-1]
    expected_closed_price = 60000.0 + 35 * 10.0  # 60350.0
    assert latest_bar_close == expected_closed_price, (
        f"Active unclosed candle was not excluded! Got {latest_bar_close}, expected closed bar {expected_closed_price}"
    )
    assert len(candles_df) == 35
    assert candles_df["timestamp"].is_monotonic_increasing


def test_candle_closure_retains_all_closed_candles_without_blind_slice(monkeypatch):
    """
    Test B: Verify that when all returned candles are already closed,
    NO candle is discarded by a blind slice fallback, and all requested closed bars are retained.
    """
    from datetime import datetime, timezone
    import requests

    now_dt = datetime.now(timezone.utc)
    now_ms = now_dt.timestamp() * 1000.0

    # Build 36 completed closed candles (0 forming candles)
    mock_klines = []
    for i in range(36):
        open_time = now_ms - (38 - i) * 3600000
        close_time = open_time + 3600000 - 1
        close_price = 60000.0 + i * 10.0
        mock_klines.append([
            open_time, "60000.0", "60500.0", "59800.0", str(close_price), "100.0",
            close_time, "6020000.0", 1500, "55.0", "3311000.0", "0"
        ])

    class MockResponse:
        status_code = 200
        def json(self):
            return mock_klines

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())

    candles_df = fetch_historical_candles(symbol="BTCUSDT", limit=35)

    # The 36th candle (index 35, price 60350.0) MUST NOT be discarded by a fallback
    latest_bar_close = candles_df["close"].iloc[-1]
    expected_closed_price = 60000.0 + 35 * 10.0  # 60350.0
    assert latest_bar_close == expected_closed_price, (
        f"Valid closed candle was discarded! Got {latest_bar_close}, expected {expected_closed_price}"
    )
    assert len(candles_df) == 35
    assert candles_df["timestamp"].is_monotonic_increasing


def test_sync_fetch_latest_confirmed_candle_binance_and_coinbase(monkeypatch):
    """
    Verify that src.engine.sync.fetch_latest_confirmed_candle parses timestamps correctly
    using pandas without raising NameError or RuntimeError on both Binance and Coinbase paths.
    """
    from datetime import datetime, timezone
    import requests
    from src.engine.sync import fetch_latest_confirmed_candle

    now_dt = datetime.now(timezone.utc)
    now_ms = now_dt.timestamp() * 1000.0
    now_sec = now_dt.timestamp()

    # 1. Test Binance Path
    mock_binance_klines = []
    for i in range(102):
        open_time = now_ms - (103 - i) * 3600000
        close_time = open_time + 3600000 - 1
        mock_binance_klines.append([
            open_time, "60000.0", "60500.0", "59800.0", "60200.0", "100.0",
            close_time, "6020000.0", 1500, "55.0", "3311000.0", "0"
        ])

    class MockBinanceResponse:
        status_code = 200
        def json(self):
            return mock_binance_klines

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockBinanceResponse())

    res_binance = fetch_latest_confirmed_candle("BTCUSDT")
    assert res_binance["close"] == 60200.0
    assert isinstance(res_binance["timestamp"], datetime)
    assert len(res_binance["candles"]) == 100
    # Crucially verify pd.to_datetime executed cleanly and populated pd.Timestamp
    assert isinstance(res_binance["candles"][-1]["timestamp"], pd.Timestamp)
    assert res_binance["number_of_trades"] == 1500

    # 2. Test Coinbase Fallback Path (Binance fails with 500, Coinbase succeeds)
    # Coinbase returns [time, low, high, open, close, volume] descending
    mock_cb_candles = []
    for i in range(100):
        # time is in seconds
        candle_sec = now_sec - (i + 1) * 3600  # Closed in the past
        mock_cb_candles.append([
            candle_sec, 59800.0, 60500.0, 60000.0, 60250.0, 100.0
        ])

    def mock_get_dispatcher(url, *args, **kwargs):
        class MockResp:
            def __init__(self, code, data):
                self.status_code = code
                self._data = data
            def json(self):
                return self._data

        if "binance" in url:
            return MockResp(500, {})
        return MockResp(200, mock_cb_candles)

    monkeypatch.setattr(requests, "get", mock_get_dispatcher)

    res_cb = fetch_latest_confirmed_candle("BTCUSDT")
    assert res_cb["close"] == 60250.0
    assert isinstance(res_cb["timestamp"], datetime)
    assert len(res_cb["candles"]) == 100
    assert isinstance(res_cb["candles"][-1]["timestamp"], pd.Timestamp)
    # Coinbase unobserved trade count must be None, not 10000
    assert res_cb["number_of_trades"] is None
    assert res_cb["candles"][-1]["number_of_trades"] is None


def test_coinbase_fallback_unobserved_trade_count_handling(sample_historical_candles):
    """
    Verify that when trade count is unobserved (None/NaN, as in Coinbase public feed),
    create_features cleanly evaluates trades_mean_6h = 0.0 with zero NaNs,
    and extract_features_for_inference returns number_of_trades as None without fabricating 10,000.
    """
    df = sample_historical_candles.copy()

    # Simulate Coinbase feed: number_of_trades is None for all bars
    df["number_of_trades"] = None

    # Should succeed without raising "Feature parity violation: NaNs encountered in trades_mean_6h"
    features = extract_features_for_inference(df)

    assert features["trades_mean_6h"] == 0.0
    assert features["number_of_trades"] is None
    # Crucially verify it did NOT fabricate 10,000
    assert features["number_of_trades"] != 10000


def test_fetch_live_market_candles_binance_excludes_forming_candle(monkeypatch):
    """
    Verify that app.py:fetch_live_market_candles strictly excludes actively forming Binance candles
    under a frozen clock and retains exactly 100 confirmed closed candles in chronological order.
    """
    from datetime import datetime, timezone
    import requests
    from app import fetch_live_market_candles

    now_dt = datetime.now(timezone.utc)
    now_ms = now_dt.timestamp() * 1000.0

    # Mock Binance: 101 completed closed candles + 1 currently forming candle (12:00 to 12:59)
    mock_binance_klines = []
    for i in range(101):
        open_time = now_ms - (102 - i) * 3600000
        close_time = open_time + 3600000 - 1
        mock_binance_klines.append([
            open_time, "80000.0", "80500.0", "79800.0", str(80000.0 + i * 10.0), "100.0",
            close_time, "8000000.0", 1200, "55.0", "4400000.0", "0"
        ])

    # 102nd candle: actively forming (ends 30 minutes in the future)
    forming_open = now_ms - 1800000
    forming_close = forming_open + 3600000 - 1  # > now_ms
    mock_binance_klines.append([
        forming_open, "85000.0", "86000.0", "84000.0", "85500.0", "20.0",
        forming_close, "1700000.0", 200, "10.0", "850000.0", "0"
    ])

    class MockBinanceResp:
        status_code = 200
        def json(self):
            return mock_binance_klines

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockBinanceResp())

    # Clear streamlit cached data to prevent cache reuse
    fetch_live_market_candles.clear()
    candles_df = fetch_live_market_candles(symbol="BTCUSDT")

    assert len(candles_df) == 100
    assert candles_df["timestamp"].is_monotonic_increasing
    # Forming candle with close=85500.0 MUST NOT be present
    assert float(candles_df["close"].iloc[-1]) == 80000.0 + 100 * 10.0  # 81000.0
    assert float(candles_df["close"].iloc[-1]) != 85500.0
    assert candles_df.attrs.get("source") == "binance"


def test_fetch_live_market_candles_coinbase_excludes_forming_candle(monkeypatch):
    """
    Verify that app.py:fetch_live_market_candles strictly excludes actively forming Coinbase candles
    when Binance fails and Coinbase succeeds under a frozen clock.
    """
    from datetime import datetime, timezone
    import requests
    from app import fetch_live_market_candles

    now_dt = datetime.now(timezone.utc)
    now_sec = now_dt.timestamp()

    # Coinbase returns candles descending (newest first)
    # Candle 0 is forming: started 30 mins ago, bucket end is 30 mins in future
    mock_cb_candles = [[now_sec - 1800, 79000.0, 81000.0, 79500.0, 80500.0, 10.0]]
    for i in range(102):
        candle_sec = now_sec - (i + 1) * 3600  # Closed in the past
        mock_cb_candles.append([
            candle_sec, 78000.0, 79000.0, 78200.0, 78500.0 + i * 5.0, 80.0
        ])

    def mock_get_dispatcher(url, *args, **kwargs):
        class MockResp:
            def __init__(self, code, data):
                self.status_code = code
                self._data = data
            def json(self):
                return self._data

        if "binance" in url:
            return MockResp(500, {})
        return MockResp(200, mock_cb_candles)

    monkeypatch.setattr(requests, "get", mock_get_dispatcher)

    fetch_live_market_candles.clear()
    candles_df = fetch_live_market_candles(symbol="BTCUSDT")

    assert len(candles_df) == 100
    assert candles_df["timestamp"].is_monotonic_increasing
    # Forming candle with close=80500.0 MUST NOT be present
    assert float(candles_df["close"].iloc[-1]) != 80500.0
    assert candles_df.attrs.get("source") == "coinbase"


def test_fetch_live_market_candles_stale_local_fallback_flagged(monkeypatch):
    """
    Verify that when all live feeds fail, fetch_live_market_candles safely falls back
    to the local dataset, tags df.attrs['source'] = 'local_fallback', and flags staleness.
    """
    import requests
    from app import fetch_live_market_candles
    from datetime import datetime, timezone

    # Simulate both live providers failing (500 Server Error)
    class FailingResp:
        status_code = 500
        def json(self):
            return {}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FailingResp())

    fetch_live_market_candles.clear()
    df_fallback = fetch_live_market_candles(symbol="BTCUSDT")

    assert not df_fallback.empty
    assert len(df_fallback) == 100
    assert df_fallback.attrs.get("source") == "local_fallback"

    # Verify staleness evaluation logic
    from app import check_feed_staleness, generate_live_forecast, generate_live_shap_attribution
    is_stale, stale_age_hours = check_feed_staleness(df_fallback)
    assert is_stale is True
    assert stale_age_hours >= 0.0


def test_stale_fallback_skips_forecast_inference(sample_historical_candles):
    """
    Verify that when is_stale is True, generate_live_forecast strictly skips model inference,
    does not invoke reg_model.predict or clf_model.predict, and returns None.
    """
    from unittest.mock import MagicMock
    from app import generate_live_forecast

    mock_reg = MagicMock()
    mock_clf = MagicMock()

    # Pass DataFrame with is_stale=True
    res = generate_live_forecast(
        live_df=sample_historical_candles,
        reg_model=mock_reg,
        clf_model=mock_clf,
        is_stale=True,
    )

    assert res is None
    assert mock_reg.predict.called is False
    assert mock_clf.predict.called is False


def test_stale_fallback_skips_shap_inference(sample_historical_candles, monkeypatch):
    """
    Verify that when is_stale is True, generate_live_shap_attribution strictly skips TreeSHAP
    attribution calculations, does not invoke explain_single_prediction, and returns None.
    """
    from unittest.mock import MagicMock
    import app
    from app import generate_live_shap_attribution

    mock_explainer = MagicMock()
    explain_mock = MagicMock()
    monkeypatch.setattr(app, "explain_single_prediction", explain_mock)

    res = generate_live_shap_attribution(
        live_df=sample_historical_candles,
        explainer_clf=mock_explainer,
        is_stale=True,
    )

    assert res is None
    assert explain_mock.called is False


def test_fresh_feed_executes_forecast_and_shap_inference(sample_historical_candles, monkeypatch):
    """
    Verify that when the market feed is confirmed fresh (is_stale=False), generate_live_forecast
    and generate_live_shap_attribution execute cleanly along the live inference path and return valid outputs.
    """
    import numpy as np
    import pytest
    from unittest.mock import MagicMock
    from datetime import datetime, timezone, timedelta
    import app
    from app import check_feed_staleness, generate_live_forecast, generate_live_shap_attribution

    # Create fresh feed ending 1 hour ago
    fresh_df = sample_historical_candles.copy()
    now_dt = datetime.now(timezone.utc)
    fresh_timestamps = [now_dt - timedelta(hours=len(fresh_df) - i) for i in range(len(fresh_df))]
    fresh_df["timestamp"] = fresh_timestamps
    fresh_df.attrs["source"] = "binance"

    is_stale, age_h = check_feed_staleness(fresh_df)
    assert is_stale is False

    mock_reg = MagicMock()
    mock_reg.predict.return_value = np.array([0.002])  # +0.20% predicted return (clears 0.05% hurdle)

    mock_clf = MagicMock()
    mock_clf.predict.return_value = np.array([1])      # UP
    mock_clf.predict_proba.return_value = np.array([[0.20, 0.80]])

    forecast = generate_live_forecast(
        live_df=fresh_df,
        reg_model=mock_reg,
        clf_model=mock_clf,
        is_stale=False,
    )

    assert forecast is not None
    assert mock_reg.predict.called is True
    assert mock_clf.predict.called is True
    assert forecast["cleared_hurdle"] is True
    assert forecast["direction_badge"] == "BULLISH (UP)"
    assert forecast["pred_ret_pct"] == pytest.approx(0.20, abs=1e-3)
    assert forecast["prob_up"] == pytest.approx(80.0, abs=1e-1)

    # Test fresh TreeSHAP path
    mock_explainer = MagicMock()
    expected_drivers = [{"feature": "return_mean_6h", "shap": 0.065}]
    monkeypatch.setattr(
        app,
        "explain_single_prediction",
        lambda explainer, df, top_n: (0.50, expected_drivers, [], None),
    )

    shap_res = generate_live_shap_attribution(
        live_df=fresh_df,
        explainer_clf=mock_explainer,
        is_stale=False,
        top_n=4,
    )

    assert shap_res is not None
    base_val, top_pos, top_neg, fig = shap_res
    assert base_val == 0.50
    assert len(top_pos) == 1
    assert top_pos[0]["feature"] == "return_mean_6h"


def test_live_shap_attribution_uses_canonical_features_and_correct_lag(sample_historical_candles, monkeypatch):
    """
    Verify that generate_live_shap_attribution passes the genuine 39-feature contract
    from extract_features_for_inference, does NOT zero-pad indicators, and strictly references
    the true prior candle close (t-1) for close_lag_1.
    """
    from unittest.mock import MagicMock
    import app
    from app import generate_live_shap_attribution

    captured_df = []

    def spy_explain(explainer, input_df, top_n=4):
        captured_df.append(input_df.copy())
        return (0.50, [{"feature": "rsi", "shap": 0.05}], [], None)

    monkeypatch.setattr(app, "explain_single_prediction", spy_explain)

    mock_explainer = MagicMock()
    live_df = sample_historical_candles.copy()

    res = generate_live_shap_attribution(
        live_df=live_df,
        explainer_clf=mock_explainer,
        is_stale=False,
        top_n=4,
    )

    assert res is not None
    assert len(captured_df) == 1
    input_df = captured_df[0]

    # Verify column schema and order
    assert list(input_df.columns) == config.CLASSIFIER_FEATURES
    assert len(input_df.columns) == 39

    # Verify technical indicators are not synthetic zeros
    assert float(input_df["rsi"].iloc[0]) > 0.0
    assert float(input_df["Rolling_Mean_Close"].iloc[0]) > 0.0
    assert float(input_df["vol_mean_6h"].iloc[0]) > 0.0

    # Verify temporal causality for close_lag_1
    expected_lag_1 = float(live_df["close"].iloc[-2])
    current_close = float(live_df["close"].iloc[-1])
    assert float(input_df["close_lag_1"].iloc[0]) == pytest.approx(expected_lag_1, rel=1e-5)
    if current_close != expected_lag_1:
        assert float(input_df["close_lag_1"].iloc[0]) != current_close


def test_live_forecast_classifier_uses_correct_lag_and_full_features(sample_historical_candles):
    """
    Verify that generate_live_forecast passes the canonical 39-feature vector to clf_model
    with close_lag_1 strictly referencing candle t-1.
    """
    from unittest.mock import MagicMock
    from app import generate_live_forecast

    captured_clf_df = []

    mock_reg = MagicMock()
    mock_reg.predict.return_value = np.array([0.001])

    mock_clf = MagicMock()

    def spy_clf_predict(clf_df):
        captured_clf_df.append(clf_df.copy())
        return np.array([1])

    mock_clf.predict.side_effect = spy_clf_predict
    mock_clf.predict_proba.return_value = np.array([[0.30, 0.70]])

    live_df = sample_historical_candles.copy()

    forecast = generate_live_forecast(
        live_df=live_df,
        reg_model=mock_reg,
        clf_model=mock_clf,
        is_stale=False,
    )

    assert forecast is not None
    assert len(captured_clf_df) == 1
    clf_df = captured_clf_df[0]

    assert list(clf_df.columns) == config.CLASSIFIER_FEATURES
    expected_lag_1 = float(live_df["close"].iloc[-2])
    current_close = float(live_df["close"].iloc[-1])
    assert float(clf_df["close_lag_1"].iloc[0]) == pytest.approx(expected_lag_1, rel=1e-5)
    if current_close != expected_lag_1:
        assert float(clf_df["close_lag_1"].iloc[0]) != current_close


def test_insufficient_history_safely_suppresses_forecast_and_shap(sample_historical_candles):
    """
    Verify that when live_df contains fewer than 35 candles, both generate_live_forecast
    and generate_live_shap_attribution safely return None without executing predictions on fake zeros.
    """
    from unittest.mock import MagicMock
    from app import generate_live_forecast, generate_live_shap_attribution

    short_df = sample_historical_candles.head(20).copy()  # Only 20 candles (< 35)

    mock_reg = MagicMock()
    mock_clf = MagicMock()
    mock_explainer = MagicMock()

    forecast = generate_live_forecast(
        live_df=short_df,
        reg_model=mock_reg,
        clf_model=mock_clf,
        is_stale=False,
    )
    assert forecast is None
    assert mock_reg.predict.called is False
    assert mock_clf.predict.called is False

    shap_res = generate_live_shap_attribution(
        live_df=short_df,
        explainer_clf=mock_explainer,
        is_stale=False,
    )
    assert shap_res is None


def test_merge_onchain_prevents_same_day_intraday_leakage():
    """
    Verify that an intraday candle at date D does NOT receive date D's completed aggregate,
    preventing lookahead leakage. It must strictly receive the fully completed day D-1 metrics.
    """
    from src.agents.onchain_agent import merge_onchain

    # Synthetic candles spanning 2026-10-10 and 2026-10-11
    timestamps = [
        "2026-10-10 12:00:00",
        "2026-10-10 23:00:00",
        "2026-10-11 05:00:00",
        "2026-10-11 23:00:00",
    ]
    candle_df = pd.DataFrame({"timestamp": timestamps, "close": [60000.0, 60100.0, 60200.0, 60300.0]})

    # Onchain daily observations
    mock_onchain = pd.DataFrame({
        "date": [pd.to_datetime("2026-10-09").date(), pd.to_datetime("2026-10-10").date(), pd.to_datetime("2026-10-11").date()],
        "onchain_num_tx": [100000.0, 200000.0, 300000.0],
        "onchain_hash_rate": [100.0, 200.0, 300.0],
        "onchain_miners_revenue_usd": [1000.0, 2000.0, 3000.0],
    })

    merged = merge_onchain(candle_df, onchain_df=mock_onchain)

    # At 2026-10-10 (any hour): latest completed day is 2026-10-09 (100000.0)
    assert merged.loc[merged["timestamp"] == "2026-10-10 12:00:00", "onchain_num_tx"].iloc[0] == 100000.0
    assert merged.loc[merged["timestamp"] == "2026-10-10 23:00:00", "onchain_num_tx"].iloc[0] == 100000.0

    # At 2026-10-11 05:00:00: latest completed day is 2026-10-10 (200000.0), NOT same-day 2026-10-11 (300000.0)
    leakage_candle = merged.loc[merged["timestamp"] == "2026-10-11 05:00:00"]
    assert leakage_candle["onchain_num_tx"].iloc[0] == 200000.0, (
        f"Temporal leakage detected! Intraday candle received {leakage_candle['onchain_num_tx'].iloc[0]} "
        f"instead of completed prior day's value 200000.0."
    )
    assert leakage_candle["onchain_num_tx"].iloc[0] != 300000.0


def test_merge_onchain_prevents_backward_fill_leakage():
    """
    Verify that historical candles preceding the earliest onchain observation
    receive 0.0 (unobserved) and are NEVER backward-filled with future observations.
    """
    from src.agents.onchain_agent import merge_onchain

    timestamps = [
        "2026-10-08 10:00:00",
        "2026-10-09 10:00:00",
        "2026-10-11 10:00:00",
    ]
    candle_df = pd.DataFrame({"timestamp": timestamps, "close": [58000.0, 59000.0, 60000.0]})

    # Onchain data only starts on 2026-10-10
    mock_onchain = pd.DataFrame({
        "date": [pd.to_datetime("2026-10-10").date()],
        "onchain_num_tx": [500000.0],
        "onchain_hash_rate": [500.0],
        "onchain_miners_revenue_usd": [5000.0],
    })

    merged = merge_onchain(candle_df, onchain_df=mock_onchain)

    # 2026-10-08 and 2026-10-09 must NOT be backward-filled with 500000.0
    val_08 = merged.loc[merged["timestamp"] == "2026-10-08 10:00:00", "onchain_num_tx"].iloc[0]
    val_09 = merged.loc[merged["timestamp"] == "2026-10-09 10:00:00", "onchain_num_tx"].iloc[0]
    assert val_08 == 0.0, f"Backward-fill leakage! Expected 0.0, got {val_08}"
    assert val_09 == 0.0, f"Backward-fill leakage! Expected 0.0, got {val_09}"

    # 2026-10-11 looks up completed day 2026-10-10 -> 500000.0
    val_11 = merged.loc[merged["timestamp"] == "2026-10-11 10:00:00", "onchain_num_tx"].iloc[0]
    assert val_11 == 500000.0


def test_merge_onchain_computes_daily_change_without_hourly_zero_repetition():
    """
    Verify that day-over-day percentage changes are computed at the daily level before broadcast,
    preventing the 96% zero-repetition artifact across hourly candles.
    Also verify that create_features preserves this upstream calculation.
    """
    from src.agents.onchain_agent import merge_onchain
    from src.feature_engineering import create_features

    # 24 hourly candles on 2026-10-11
    timestamps = [f"2026-10-11 {h:02d}:00:00" for h in range(24)]
    candle_df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [60000.0] * 24,
        "high": [60500.0] * 24,
        "low": [59500.0] * 24,
        "close": [60000.0] * 24,
        "volume": [100.0] * 24,
    })

    # Daily data: 2026-10-09 = 100,000; 2026-10-10 = 120,000 (+20% change)
    mock_onchain = pd.DataFrame({
        "date": [pd.to_datetime("2026-10-09").date(), pd.to_datetime("2026-10-10").date()],
        "onchain_num_tx": [100000.0, 120000.0],
        "onchain_hash_rate": [100.0, 120.0],
        "onchain_miners_revenue_usd": [1000.0, 1200.0],
    })

    merged = merge_onchain(candle_df, onchain_df=mock_onchain)

    # All 24 hours of 2026-10-11 look up completed day 2026-10-10, where daily change was (120k - 100k)/100k = 0.20
    assert len(merged) == 24
    for h in range(24):
        val = merged["onchain_num_tx_change"].iloc[h]
        assert val == pytest.approx(0.20, abs=1e-4), (
            f"Hour {h} did not receive daily day-over-day change! Got {val}, expected 0.20"
        )

    # In create_features, verify this change is NOT overwritten by hourly pct_change (which would yield 0.0)
    feat_df = create_features(merged, drop_na=False)
    for h in range(1, 24):
        feat_val = feat_df["onchain_num_tx_change"].iloc[h]
        assert feat_val == pytest.approx(0.20, abs=1e-4), (
            f"create_features overwrote daily change with hourly zero at hour {h}! Got {feat_val}"
        )


def test_merge_onchain_timestamp_and_boundary_semantics():
    """
    Verify timezone handling (UTC awareness, string formatting) and midnight boundary transitions.
    At 2026-10-11 00:00:00 UTC, target date is 2026-10-10.
    At 2026-10-10 23:00:00 UTC, target date is 2026-10-09.
    """
    from src.agents.onchain_agent import merge_onchain

    # 1. Test timezone-aware UTC timestamps
    tz_aware_ts = pd.date_range("2026-10-10 22:00:00", periods=4, freq="h", tz="UTC")
    df_tz = pd.DataFrame({"timestamp": tz_aware_ts, "close": [60000.0] * 4})

    mock_onchain = pd.DataFrame({
        "date": [pd.to_datetime("2026-10-09").date(), pd.to_datetime("2026-10-10").date()],
        "onchain_num_tx": [111.0, 222.0],
        "onchain_hash_rate": [11.0, 22.0],
        "onchain_miners_revenue_usd": [1.0, 2.0],
    })

    merged_tz = merge_onchain(df_tz, onchain_df=mock_onchain)
    # Row 0: 2026-10-10 22:00 -> completed day 2026-10-09 -> 111.0
    # Row 1: 2026-10-10 23:00 -> completed day 2026-10-09 -> 111.0
    # Row 2: 2026-10-11 00:00 -> completed day 2026-10-10 -> 222.0
    # Row 3: 2026-10-11 01:00 -> completed day 2026-10-10 -> 222.0
    assert merged_tz["onchain_num_tx"].iloc[0] == 111.0
    assert merged_tz["onchain_num_tx"].iloc[1] == 111.0
    assert merged_tz["onchain_num_tx"].iloc[2] == 222.0
    assert merged_tz["onchain_num_tx"].iloc[3] == 222.0

    # 2. Test timezone-naive timestamps (should localize cleanly to UTC)
    naive_ts = [
        pd.Timestamp("2026-10-10 23:00:00"),
        pd.Timestamp("2026-10-11 00:00:00"),
    ]
    df_naive = pd.DataFrame({"timestamp": naive_ts, "close": [60000.0, 60100.0]})
    merged_naive = merge_onchain(df_naive, onchain_df=mock_onchain)
    assert merged_naive["onchain_num_tx"].iloc[0] == 111.0
    assert merged_naive["onchain_num_tx"].iloc[1] == 222.0


def test_merge_fear_greed_prevents_same_day_leakage():
    """
    Verify that an intraday candle at date D does NOT receive date D's sentiment index,
    preventing lookahead leakage. It must strictly receive the fully completed day D-1 metrics.
    """
    from src.agents.fear_greed_agent import merge_fear_greed

    timestamps = [
        "2026-10-10 12:00:00",
        "2026-10-10 23:00:00",
        "2026-10-11 05:00:00",
        "2026-10-11 23:00:00",
    ]
    candle_df = pd.DataFrame({"timestamp": timestamps, "close": [60000.0, 60100.0, 60200.0, 60300.0]})

    mock_fng = pd.DataFrame({
        "date": [pd.to_datetime("2026-10-09").date(), pd.to_datetime("2026-10-10").date(), pd.to_datetime("2026-10-11").date()],
        "fng_value": [30.0, 42.0, 88.0],
    })

    merged = merge_fear_greed(candle_df, fng_df=mock_fng)

    # At 2026-10-10 (any hour): latest completed day is 2026-10-09 (30.0)
    assert merged.loc[merged["timestamp"] == "2026-10-10 12:00:00", "fng_value"].iloc[0] == 30.0
    assert merged.loc[merged["timestamp"] == "2026-10-10 23:00:00", "fng_value"].iloc[0] == 30.0

    # At 2026-10-11 05:00:00: latest completed day is 2026-10-10 (42.0), NOT same-day 2026-10-11 (88.0)
    leakage_candle = merged.loc[merged["timestamp"] == "2026-10-11 05:00:00"]
    assert leakage_candle["fng_value"].iloc[0] == 42.0, (
        f"Temporal leakage detected! Intraday candle received {leakage_candle['fng_value'].iloc[0]} "
        f"instead of completed prior day's value 42.0."
    )
    assert leakage_candle["fng_value"].iloc[0] != 88.0


def test_merge_fear_greed_prevents_backward_fill_and_sets_pre_observation_default():
    """
    Verify that historical candles preceding the earliest FNG observation receive NEUTRAL_DEFAULT (50.0)
    and are NEVER backward-filled with future sentiment observations.
    """
    from src.agents.fear_greed_agent import merge_fear_greed

    timestamps = [
        "2026-10-07 10:00:00",
        "2026-10-08 10:00:00",
        "2026-10-11 10:00:00",
    ]
    candle_df = pd.DataFrame({"timestamp": timestamps, "close": [58000.0, 59000.0, 60000.0]})

    # FNG data only begins on 2026-10-10
    mock_fng = pd.DataFrame({
        "date": [pd.to_datetime("2026-10-10").date()],
        "fng_value": [78.0],
    })

    merged = merge_fear_greed(candle_df, fng_df=mock_fng)

    # 2026-10-07 and 2026-10-08 must receive 50.0 (NEUTRAL_DEFAULT), not backward-filled 78.0
    val_07 = merged.loc[merged["timestamp"] == "2026-10-07 10:00:00", "fng_value"].iloc[0]
    val_08 = merged.loc[merged["timestamp"] == "2026-10-08 10:00:00", "fng_value"].iloc[0]
    assert val_07 == 50.0, f"Backward-fill leakage! Expected 50.0, got {val_07}"
    assert val_08 == 50.0, f"Backward-fill leakage! Expected 50.0, got {val_08}"

    # 2026-10-11 looks up completed day 2026-10-10 -> 78.0
    val_11 = merged.loc[merged["timestamp"] == "2026-10-11 10:00:00", "fng_value"].iloc[0]
    assert val_11 == 78.0


def test_merge_fear_greed_forward_fills_missing_observation_dates():
    """
    Verify that when FNG data contains missing observation dates (API gaps or delayed releases),
    the previous observed sentiment is forward-filled across the gap rather than dropping to 50.0.
    """
    from src.agents.fear_greed_agent import merge_fear_greed

    timestamps = [
        "2026-10-10 12:00:00",  # targets 2026-10-09 (observed: 25.0)
        "2026-10-11 12:00:00",  # targets 2026-10-10 (missing date: should forward-fill 25.0)
        "2026-10-12 12:00:00",  # targets 2026-10-11 (observed: 65.0)
    ]
    candle_df = pd.DataFrame({"timestamp": timestamps, "close": [60000.0, 60100.0, 60200.0]})

    # FNG data has an intentional gap on 2026-10-10
    mock_fng = pd.DataFrame({
        "date": [pd.to_datetime("2026-10-09").date(), pd.to_datetime("2026-10-11").date()],
        "fng_value": [25.0, 65.0],
    })

    merged = merge_fear_greed(candle_df, fng_df=mock_fng)

    # Row 0: targets 2026-10-09 -> 25.0
    assert merged["fng_value"].iloc[0] == 25.0
    # Row 1: targets 2026-10-10 (missing) -> must forward-fill 25.0, NOT drop to 50.0
    assert merged["fng_value"].iloc[1] == 25.0, (
        f"Missing observation was not forward-filled! Got {merged['fng_value'].iloc[1]}, expected 25.0"
    )
    # Row 2: targets 2026-10-11 -> 65.0
    assert merged["fng_value"].iloc[2] == 65.0


def test_merge_fear_greed_midnight_utc_and_timezone_boundaries():
    """
    Verify timezone handling (UTC awareness, non-UTC conversion, string parsing)
    and clean transitions across midnight boundaries.
    """
    from src.agents.fear_greed_agent import merge_fear_greed

    # 1. Midnight boundary in UTC
    tz_aware_ts = pd.date_range("2026-10-10 22:00:00", periods=4, freq="h", tz="UTC")
    df_tz = pd.DataFrame({"timestamp": tz_aware_ts, "close": [60000.0] * 4})

    mock_fng = pd.DataFrame({
        "date": [pd.to_datetime("2026-10-09").date(), pd.to_datetime("2026-10-10").date()],
        "fng_value": [33.0, 66.0],
    })

    merged_tz = merge_fear_greed(df_tz, fng_df=mock_fng)
    # 2026-10-10 22:00 UTC -> targets 2026-10-09 -> 33.0
    # 2026-10-10 23:00 UTC -> targets 2026-10-09 -> 33.0
    # 2026-10-11 00:00 UTC -> targets 2026-10-10 -> 66.0
    # 2026-10-11 01:00 UTC -> targets 2026-10-10 -> 66.0
    assert merged_tz["fng_value"].iloc[0] == 33.0
    assert merged_tz["fng_value"].iloc[1] == 33.0
    assert merged_tz["fng_value"].iloc[2] == 66.0
    assert merged_tz["fng_value"].iloc[3] == 66.0

    # 2. Non-UTC timezone conversion (e.g. +05:30)
    # 2026-10-11 05:00:00+05:30 corresponds to 2026-10-10 23:30:00 UTC -> targets 2026-10-09 -> 33.0
    # 2026-10-11 06:00:00+05:30 corresponds to 2026-10-11 00:30:00 UTC -> targets 2026-10-10 -> 66.0
    ts_non_utc = pd.to_datetime(["2026-10-11 05:00:00+05:30", "2026-10-11 06:00:00+05:30"])
    df_non_utc = pd.DataFrame({"timestamp": ts_non_utc, "close": [60000.0, 60100.0]})
    merged_non_utc = merge_fear_greed(df_non_utc, fng_df=mock_fng)
    assert merged_non_utc["fng_value"].iloc[0] == 33.0
    assert merged_non_utc["fng_value"].iloc[1] == 66.0


def test_merge_fear_greed_rolling_mean_consistency():
    """
    Verify downstream consistency of fng_mean_3d (72-hour rolling window)
    when computed via create_features after causal merge_fear_greed.
    """
    from src.agents.fear_greed_agent import merge_fear_greed
    from src.feature_engineering import create_features

    # 100 hourly candles across 4+ days
    timestamps = pd.date_range("2026-10-10 00:00:00", periods=100, freq="h", tz="UTC")
    candle_df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [60000.0] * 100,
        "high": [60500.0] * 100,
        "low": [59500.0] * 100,
        "close": [60000.0] * 100,
        "volume": [100.0] * 100,
    })

    # FNG data: 2026-10-09 through 2026-10-13 constant at 40.0
    dates = pd.date_range("2026-10-09", periods=6, freq="D").date
    mock_fng = pd.DataFrame({
        "date": dates,
        "fng_value": [40.0] * len(dates),
    })

    merged = merge_fear_greed(candle_df, fng_df=mock_fng)
    assert (merged["fng_value"] == 40.0).all()

    # Pass through feature engineering
    feat_df = create_features(merged, drop_na=False)
    assert "fng_mean_3d" in feat_df.columns
    # Row 71 (the 72nd candle) should have rolling mean of 40.0 exactly
    assert feat_df["fng_mean_3d"].iloc[71] == pytest.approx(40.0, abs=1e-5)
    assert feat_df["fng_mean_3d"].iloc[-1] == pytest.approx(40.0, abs=1e-5)





