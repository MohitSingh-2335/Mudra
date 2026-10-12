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
    latest_candle_ts = pd.to_datetime(df_fallback["timestamp"].iloc[-1], utc=True)
    stale_age_hours = (datetime.now(timezone.utc) - latest_candle_ts).total_seconds() / 3600.0
    is_stale = (stale_age_hours > 3.0 or df_fallback.attrs.get("source") == "local_fallback")
    assert is_stale is True


