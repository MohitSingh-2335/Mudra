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
