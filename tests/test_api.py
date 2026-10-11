# tests/test_api.py - Mudra Automated REST API Test Suite
from src.api.main import app

import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from config import (
    FEATURED_BTC_DATA_PATH,
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES
)

from unittest.mock import patch


@pytest.fixture(scope="session")
def client():
    """Initializes the FastAPI TestClient with startup/shutdown lifespans."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_features_payload():
    """Generates a complete, valid feature dictionary for test inference."""
    all_features = sorted(list(set(REGRESSOR_FEATURES + CLASSIFIER_FEATURES)))
    
    if os.path.exists(FEATURED_BTC_DATA_PATH):
        df = pd.read_csv(FEATURED_BTC_DATA_PATH, nrows=1)
        current_price = float(df["close"].iloc[0])
        feature_dict = {f: float(df[f].iloc[0]) for f in all_features if f in df.columns}
    else:
        # Fallback values if dataset is absent
        current_price = 65000.0
        feature_dict = {f: 1.0 for f in all_features}

    return {
        "current_price": current_price,
        "features": feature_dict
    }


def test_health_check(client):
    """Verify that the health check probe returns 200 and correct model version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "optuna_tuned_xgboost_v1"


def test_predict_valid_features(client, sample_features_payload):
    """Verify that /predict produces valid numeric and directional forecasts."""
    response = client.post("/predict", json=sample_features_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_price"] > 0
    assert isinstance(data["predicted_return"], float)
    assert data["directional_movement"] in ["Upward 📈", "Downward 📉"]
    assert data["movement_code"] in [0, 1]


def test_predict_missing_features(client):
    """Verify that /predict rejects incomplete payloads with HTTP 400."""
    bad_payload = {
        "current_price": 65000.0,
        "features": {
            "rsi": 55.0
            # Missing other 38 required features
        }
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
    assert "Missing required feature fields" in response.json()["detail"]


def test_predict_invalid_price(client, sample_features_payload):
    """Verify that Pydantic rejects negative or zero price with HTTP 422."""
    bad_payload = sample_features_payload.copy()
    bad_payload["current_price"] = -100.0  # Invalid price

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_explain_endpoint(client, sample_features_payload):
    """Verify that /explain produces local TreeSHAP feature attributions."""
    response = client.post("/explain", json=sample_features_payload)
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["base_value"], float)
    assert isinstance(data["top_bullish_drivers"], list)
    assert isinstance(data["top_bearish_drivers"], list)
    assert len(data["top_bullish_drivers"]) <= 4
    assert len(data["top_bearish_drivers"]) <= 4


def test_agents_trade_endpoint(client):
    """Verify that /agents/trade runs the LangGraph 2-node pipeline and returns valid decision schema."""
    payload = {
        "symbol": "BTCUSDT",
        "current_price": 65000.0,
        "market_data": {
            "close": 65000.0,
            "high": 65400.0,
            "low": 64800.0,
            "volume": 1500.0,
            "taker_buy_ratio": 0.55
        },
        "engineered_features": {
            "volatility_range_pct": 0.92,
            "taker_buy_ratio": 0.55
        },
        "ml_forecast": {
            "expected_return_pct": 0.12,
            "predicted_dollar_move": 78.0,
            "clears_hurdle": True
        }
    }
    response = client.post("/agents/trade", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["symbol"] == "BTCUSDT"
    assert data["trade_action"] in ["BUY", "SELL", "HOLD"]
    assert isinstance(data["risk_approved"], bool)
    assert isinstance(data["recommended_position_pct"], float)
    assert isinstance(data["conviction_score"], int)
    assert len(data["tools_called"]) >= 0
    assert "risk_critic_critique" in data

def test_reconciliation_summary_endpoint(client):
    """Verify that /reconciliation/summary returns accuracy KPIs with expected schema."""
    mock_summary = {
        "total_reconciled": 24,
        "directional_accuracy_pct": 62.5,
        "mae_usd": 145.20,
        "mape_pct": 0.22,
        "information_coefficient_ic": 0.18,
        "black_swan_count": 0,
        "tag_breakdown": {"ACCURATE": 15, "DIRECTION_MISMATCH": 9}
    }
    with patch("src.database.queries.get_reconciliation_kpis", return_value=mock_summary):
        response = client.get("/reconciliation/summary?lookback_hours=24")
        assert response.status_code == 200
        data = response.json()
        assert data["total_reconciled"] == 24
        assert data["directional_accuracy_pct"] == 62.5
        assert "information_coefficient_ic" in data
        assert data["tag_breakdown"]["ACCURATE"] == 15
def test_portfolio_summary_endpoint(client):
    """Verify that /portfolio/summary returns paper portfolio state and Alpha metrics."""
    mock_summary = {
        "cash_balance": 9500.0,
        "btc_balance": 0.006,
        "current_btc_price": 83000.0,
        "btc_position_value_usd": 498.0,
        "total_equity_usd": 9998.0,
        "benchmark_btc_equity": 10000.0,
        "alpha_usd": -2.0,
        "alpha_pct": -0.02,
        "realized_pnl_usd": 0.0,
        "total_fees_usd": 2.0,
        "total_trades": 1
    }
    with patch("src.database.queries.get_portfolio_kpis", return_value=mock_summary):
        response = client.get("/portfolio/summary?current_price=83000.0")
        assert response.status_code == 200
        data = response.json()
        assert data["total_equity_usd"] == 9998.0
        assert data["cash_balance"] == 9500.0
        assert data["total_trades"] == 1
        assert "alpha_usd" in data