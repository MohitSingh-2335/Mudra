from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import ta

import app
import config
from config import (
    ACTIVE_MODEL_VERSION,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_MODEL_SHA256,
    FEATURE_SCHEMA_VERSION,
    REGRESSOR_MODEL_PATH,
    REGRESSOR_MODEL_SHA256,
)
from src.data_preprocessing import (
    load_and_clean_data,
    merge_fear_greed,
    merge_onchain,
)
from src.database.models import Base, MarketTick, Reconciliation
from src.database.queries import get_reconciliation_history, get_reconciliation_kpis
from src.feature_engineering import (
    create_features,
    extract_features_for_inference,
    predict_proba_with_validation,
    predict_with_validation,
    validate_inference_features,
)
from src.model_loader import (
    compute_file_sha256,
    load_production_models,
    load_verified_model,
    validate_sha256_digest,
)
from scripts.migrate_db import BASELINE_MODEL_SPEC


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


def test_canonical_configuration_constants():
    assert ACTIVE_MODEL_VERSION == "xgb_onnx_v1.0.0"
    assert FEATURE_SCHEMA_VERSION == "v1_39features"
    assert BASELINE_MODEL_SPEC["model_version"] == ACTIVE_MODEL_VERSION
    assert BASELINE_MODEL_SPEC["regressor_sha256"] == REGRESSOR_MODEL_SHA256
    assert MarketTick.model_version.nullable is False
    assert MarketTick.feature_schema_version.nullable is False
    assert compute_file_sha256(REGRESSOR_MODEL_PATH) == REGRESSOR_MODEL_SHA256
    assert compute_file_sha256(CLASSIFIER_MODEL_PATH) == CLASSIFIER_MODEL_SHA256


def test_artifact_hash_verification_success_and_tampering():
    reg, clf = load_production_models()
    assert reg is not None and hasattr(reg, "predict")
    assert clf is not None and hasattr(clf, "predict_proba")
    with pytest.raises(RuntimeError, match="Cryptographic integrity verification failed"):
        load_verified_model(REGRESSOR_MODEL_PATH, "0" * 64)
    with pytest.raises(FileNotFoundError):
        load_verified_model("nonexistent.pkl", "0" * 64)


def test_sha256_digest_format():
    validate_sha256_digest("a" * 64)
    with pytest.raises(ValueError):
        validate_sha256_digest("short")
    with pytest.raises(ValueError):
        validate_sha256_digest("g" * 64)


@pytest.fixture
def rec_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tick = MarketTick(
        timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60000.0, volume=10.0,
        predicted_price_1h=60300.0, predicted_return_1h=0.005, model_version=ACTIVE_MODEL_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    session.add(tick)
    session.commit()

    rec = Reconciliation(
        timestamp=t0, tick_id=tick.id, predicted_price=60300.0, actual_price_at_t_plus_1=60250.0,
        predicted_return_pct=0.5, actual_return_pct=0.416, residual_error=-0.084,
        absolute_error_usd=50.0, percentage_error=0.083, direction_predicted="UP",
        direction_realized="UP", direction_correct=True, error_attribution_tag="ACCURATE",
    )
    session.add(rec)
    session.commit()
    yield session
    session.close()


def test_reconciliation_filtering_and_kpis(rec_db, monkeypatch):
    @contextmanager
    def mock_get_db():
        yield rec_db

    monkeypatch.setattr("src.database.queries.get_db", mock_get_db)
    assert len(get_reconciliation_history(limit=50, model_version=None)) == 1
    assert len(get_reconciliation_history(limit=50, model_version=ACTIVE_MODEL_VERSION)) == 1
    assert len(get_reconciliation_history(limit=50, model_version="candidate_v2")) == 0

    kpis = get_reconciliation_kpis(model_version=ACTIVE_MODEL_VERSION)
    assert kpis["total_reconciled"] == 1
    assert kpis["directional_accuracy_pct"] == 100.0


def test_dashboard_tab3_helpers(rec_db, monkeypatch):
    @contextmanager
    def mock_get_db():
        yield rec_db

    monkeypatch.setattr("src.database.queries.get_db", mock_get_db)
    records = get_reconciliation_history(limit=50, model_version=None)
    filtered = app.filter_reconciliation_records(records, selected_model=ACTIVE_MODEL_VERSION)
    assert len(filtered) == 1
    df_display = app.prepare_reconciliation_table_display(filtered)
    assert not df_display.empty
    assert "Model Version" in df_display.columns
