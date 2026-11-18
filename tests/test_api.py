from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from config import CLASSIFIER_FEATURES, FEATURED_BTC_DATA_PATH, REGRESSOR_FEATURES
from src.api.main import ACTIVE_MODEL_VERSION, app
from src.api.schemas import ACTIVE_MODEL_VERSION as SCHEMA_ACTIVE_MODEL_VERSION, HealthResponse, PredictResponse
from src.database.models import Base, MarketTick, Reconciliation
from src.engine.paper_portfolio import execute_paper_order


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_features_payload():
    all_features = sorted(list(set(REGRESSOR_FEATURES + CLASSIFIER_FEATURES)))
    if os.path.exists(FEATURED_BTC_DATA_PATH):
        df = pd.read_csv(FEATURED_BTC_DATA_PATH, nrows=1)
        price = float(df["close"].iloc[0])
        feats = {f: float(df[f].iloc[0]) for f in all_features if f in df.columns}
    else:
        price = 65000.0
        feats = {f: 1.0 for f in all_features}
    return {"current_price": price, "features": feats}


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200 and res.json()["status"] == "ok" and res.json()["model_version"] == ACTIVE_MODEL_VERSION


def test_predict_valid_features(client, sample_features_payload):
    res = client.post("/predict", json=sample_features_payload)
    assert res.status_code == 200
    d = res.json()
    assert d["current_price"] == pytest.approx(sample_features_payload["current_price"]) and d["predicted_price"] > 0
    assert d["directional_movement"] in ["Upward 📈", "Downward 📉"] and d["model_version"] == ACTIVE_MODEL_VERSION


def test_health_and_predict_model_version_provenance_parity(client, sample_features_payload):
    assert ACTIVE_MODEL_VERSION == SCHEMA_ACTIVE_MODEL_VERSION == "xgb_onnx_v1.0.0"
    assert client.get("/health").json()["model_version"] == "xgb_onnx_v1.0.0"
    assert client.post("/predict", json=sample_features_payload).json()["model_version"] == "xgb_onnx_v1.0.0"
    assert HealthResponse().model_version == "xgb_onnx_v1.0.0"
    assert PredictResponse(current_price=65000.0, predicted_price=65100.0, predicted_return=0.0015, directional_movement="Upward 📈", movement_code=1).model_version == "xgb_onnx_v1.0.0"


def test_predict_missing_and_invalid_features(client, sample_features_payload):
    assert client.post("/predict", json={"current_price": 65000.0, "features": {"rsi": 55.0}}).status_code == 400
    bad_p = sample_features_payload.copy()
    bad_p["current_price"] = -100.0
    assert client.post("/predict", json=bad_p).status_code == 422
    bad_nan = {"current_price": 65000.0, "features": dict(sample_features_payload["features"])}
    bad_nan["features"]["rsi"] = float("nan")
    assert client.post("/predict", json=bad_nan).status_code == 400


def test_explain_endpoint(client, sample_features_payload):
    res = client.post("/explain", json=sample_features_payload)
    assert res.status_code == 200
    d = res.json()
    assert isinstance(d["base_value"], float) and isinstance(d["top_bullish_drivers"], list) and isinstance(d["top_bearish_drivers"], list)


def test_agents_trade_endpoint(client):
    payload = {
        "symbol": "BTCUSDT", "current_price": 65000.0,
        "market_data": {"close": 65000.0, "high": 65400.0, "low": 64800.0, "volume": 1500.0, "taker_buy_ratio": 0.55},
        "engineered_features": {"volatility_range_pct": 0.92, "taker_buy_ratio": 0.55},
        "ml_forecast": {"expected_return_pct": 0.25, "predicted_dollar_move": 162.5, "clears_hurdle": True},
    }
    res = client.post("/agents/trade", json=payload)
    assert res.status_code == 200 and res.json()["trade_action"] in ["BUY", "SELL", "HOLD"] and isinstance(res.json()["risk_approved"], bool)


def test_reconciliation_summary_endpoint(client):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    @contextmanager
    def _test_db():
        s = TestingSession()
        try:
            yield s
        finally:
            s.close()

    with patch("src.database.connection.get_db", _test_db), patch("src.database.queries.get_db", _test_db):
        assert client.get("/reconciliation/summary?lookback_hours=24").json()["total_reconciled"] == 0
        now = datetime.now(timezone.utc)
        with _test_db() as s:
            t1 = MarketTick(timestamp=now - timedelta(hours=2), open=60000.0, high=60200.0, low=59900.0, close=60100.0, volume=100.0)
            t2 = MarketTick(timestamp=now - timedelta(hours=1), open=60100.0, high=60300.0, low=57000.0, close=57200.0, volume=100.0)
            s.add_all([t1, t2])
            s.flush()
            r1 = Reconciliation(timestamp=now - timedelta(hours=1), tick_id=t1.id, predicted_price=60200.0, actual_price_at_t_plus_1=60200.0, predicted_return_pct=0.0017, actual_return_pct=0.0017, residual_error=0.0, absolute_error_usd=0.0, percentage_error=0.0, direction_predicted="UP", direction_realized="UP", direction_correct=True, is_black_swan=False, error_attribution_tag="ACCURATE")
            r2 = Reconciliation(timestamp=now, tick_id=t2.id, predicted_price=60400.0, actual_price_at_t_plus_1=57200.0, predicted_return_pct=0.002, actual_return_pct=-0.051, residual_error=-3200.0, absolute_error_usd=3200.0, percentage_error=5.3, direction_predicted="UP", direction_realized="DOWN", direction_correct=False, is_black_swan=True, error_attribution_tag="BLACK_SWAN_OUTLIER")
            s.add_all([r1, r2])
            s.commit()

        res = client.get("/reconciliation/summary?lookback_hours=24")
        assert res.status_code == 200 and res.json()["total_reconciled"] == 2 and res.json()["black_swan_count"] == 1


def test_portfolio_summary_endpoint(client):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    @contextmanager
    def _test_db():
        s = TestingSession()
        try:
            yield s
        finally:
            s.close()

    with patch("src.database.connection.get_db", _test_db), patch("src.database.queries.get_db", _test_db):
        assert client.get("/portfolio/summary?current_price=65000.0").json()["cash_balance"] == 10000.0
        with _test_db() as s:
            execute_paper_order(s, "BUY", 60000.0, kelly_fraction=0.10)
            execute_paper_order(s, "SELL", 63000.0)
            s.commit()

        res = client.get("/portfolio/summary?current_price=64000.0")
        assert res.status_code == 200 and res.json()["total_trades"] == 2 and res.json()["realized_pnl_usd"] > 40.0


def test_endpoints_defensive_fallbacks(client):
    with patch("src.engine.reconciliation.get_reconciliation_summary", side_effect=RuntimeError("DB disconnected")):
        assert client.get("/reconciliation/summary").status_code == 200
    with patch("src.engine.paper_portfolio.get_portfolio_summary", side_effect=RuntimeError("DB disconnected")):
        assert client.get("/portfolio/summary").status_code == 200
