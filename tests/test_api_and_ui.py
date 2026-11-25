import os
from unittest.mock import patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app
from app import format_economic_result
from config import CLASSIFIER_FEATURES, FEATURED_BTC_DATA_PATH, REGRESSOR_FEATURES
from src.api import ACTIVE_MODEL_VERSION, app as fastapi_app


@pytest.fixture(scope="session")
def client():
    with TestClient(fastapi_app) as test_client:
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


def test_predict_endpoint(client, sample_features_payload):
    res = client.post("/predict", json=sample_features_payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_price" in data and "predicted_return" in data and data["model_version"] == ACTIVE_MODEL_VERSION


def test_predict_missing_feature(client):
    res = client.post("/predict", json={"current_price": 65000.0, "features": {"rsi": 50.0}})
    assert res.status_code == 400 and "Missing required feature fields" in res.json()["detail"]


def test_explain_endpoint(client, sample_features_payload):
    res = client.post("/explain", json=sample_features_payload)
    assert res.status_code == 200
    data = res.json()
    assert "base_value" in data and "top_bullish_drivers" in data and "top_bearish_drivers" in data


def test_analyze_endpoint(client, sample_features_payload):
    res = client.post("/analyze", json=sample_features_payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data and "explanation" in data and "analyst_commentary" in data


def test_agent_trade_endpoint(client):
    res = client.post("/agents/trade", json={"symbol": "BTCUSDT", "current_price": 65000.0, "market_data": {"close": 65000.0}})
    assert res.status_code == 200 and "trade_action" in res.json() and "conviction_score" in res.json()


def test_reconciliation_summary_endpoint(client):
    with patch("src.database.get_reconciliation_kpis", return_value={"total_reconciled": 5, "directional_accuracy_pct": 60.0}):
        res = client.get("/reconciliation/summary")
        assert res.status_code == 200 and res.json()["total_reconciled"] == 5 and res.json()["directional_accuracy_pct"] == 60.0


def test_portfolio_summary_endpoint(client):
    with patch("src.database.get_portfolio_kpis", return_value={"total_equity_usd": 10500.0, "realized_pnl_usd": 500.0}):
        res = client.get("/portfolio/summary?current_price=65000.0")
        assert res.status_code == 200 and res.json()["total_equity_usd"] == 10500.0 and res.json()["realized_pnl_usd"] == 500.0

    with patch("src.database.get_portfolio_kpis", return_value={"query_failed": True, "error": "Database query failed"}):
        res_err = client.get("/portfolio/summary?current_price=65000.0")
        assert res_err.status_code == 503


def test_docs_and_openapi_endpoints(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_portfolio_summary_requires_positive_current_price(client):
    assert client.get("/portfolio/summary").status_code == 422
    assert client.get("/portfolio/summary?current_price=0").status_code == 422


def test_app_generate_registered_model_forecast_respects_stale_and_m3_sma10():
    live_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01", periods=40, freq="1h", tz="UTC"),
        "open": [65000.0] * 40, "high": [65200.0] * 40, "low": [64800.0] * 40,
        "close": [65000.0] * 40, "volume": [100.0] * 40,
    })

    seen_cols = {}

    class _Reg:
        def predict(self, df):
            seen_cols["reg"] = float(df["sma_10_dist_pct"].iloc[0])
            return [0.002]

    class _Clf:
        def predict_proba(self, df):
            seen_cols["clf"] = float(df["sma_10_dist_pct"].iloc[0])
            return [[0.4, 0.6]]

    spec = {"features": ["sma_10_dist_pct"], "regressor": _Reg(), "classifier": _Clf()}
    feats = {"sma_10_dist_pct": -0.02, "sma_10_true_dist_pct": 0.08}

    assert app.generate_registered_model_forecast(live_df, "lgbm_v3.0.0", spec, precomputed_features=feats, is_stale=True) is None
    res = app.generate_registered_model_forecast(live_df, "lgbm_v3.0.0", spec, precomputed_features=feats, is_stale=False)
    assert res is not None and res["direction_label"] == "UP"
    assert seen_cols["reg"] == pytest.approx(0.08) and seen_cols["clf"] == pytest.approx(0.08)


def test_format_economic_result_fee_hurdle():
    # verifies fee friction turns small directional wins into economic losses and filters sub-hurdle moves
    # 1. Traded move (+0.20% predicted UP, clears 0.14% hurdle), actual move was only +0.05%
    # gross = +0.05%, net = 0.05% - 0.14% = -0.09% -> LOSS
    row_traded = {
        "predicted_return_pct": 0.20,
        "actual_return_pct": 0.05,
        "direction_predicted": "UP",
    }
    res_traded = format_economic_result(row_traded, friction_pct=0.14)
    assert "LOSS" in res_traded
    assert "-0.09% net" in res_traded

    # 2. Sub-hurdle move (+0.05% predicted, below 0.14% hurdle), actual was +0.05% -> HELD CASH
    row_below_hurdle = {
        "predicted_return_pct": 0.05,
        "actual_return_pct": 0.05,
        "direction_predicted": "UP",
    }
    res_cash = format_economic_result(row_below_hurdle, friction_pct=0.14)
    assert res_cash == "HELD CASH (0.00%)"

    # 3. Traded winning move (+0.40% predicted UP, actual +0.70%) -> WIN (+0.56% net)
    row_win = {
        "predicted_return_pct": 0.40,
        "actual_return_pct": 0.70,
        "direction_predicted": "UP",
    }
    res_win = format_economic_result(row_win, friction_pct=0.14)
    assert "WIN" in res_win
    assert "+0.56% net" in res_win

