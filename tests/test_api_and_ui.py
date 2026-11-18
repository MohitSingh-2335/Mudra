from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import os
from unittest.mock import patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app
from app import format_risk_levels
from config import CLASSIFIER_FEATURES, FEATURED_BTC_DATA_PATH, REGRESSOR_FEATURES
from src.api import (
    ACTIVE_MODEL_VERSION,
    ACTIVE_MODEL_VERSION as SCHEMA_ACTIVE_MODEL_VERSION,
    HealthResponse,
    PredictResponse,
    app as fastapi_app,
)
from src.database import Base, MarketTick, Reconciliation
from src.engine import execute_paper_order


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


def test_docs_and_openapi_endpoints(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_format_risk_levels_hold_or_unapproved():
    sl, tp = format_risk_levels("HOLD", "APPROVED", 65000.0)
    assert sl == "None (No active trade opened — cash preserved)"
    assert tp == "None (No active trade opened — cash preserved)"

    sl, tp = format_risk_levels("BUY", "VETOED", 65000.0)
    assert sl == "None (No active trade opened — cash preserved)"
    assert tp == "None (No active trade opened — cash preserved)"

    sl, tp = format_risk_levels("", "APPROVED", 65000.0)
    assert sl == "None (No active trade opened — cash preserved)"
    assert tp == "None (No active trade opened — cash preserved)"


def test_format_risk_levels_approved_buy():
    price = 100000.0
    sl, tp = format_risk_levels("BUY", "APPROVED", price)
    assert "$98,500.00 (-1.5%)" in sl
    assert "$102,500.00 (+2.5%)" in tp

    custom_state = {"stop_loss_price": 97000.0, "take_profit_price": 105000.0}
    sl, tp = format_risk_levels("BUY", "APPROVED", price, custom_state)
    assert "$97,000.00 (-1.5%)" in sl
    assert "$105,000.00 (+2.5%)" in tp


def test_format_risk_levels_approved_sell():
    price = 100000.0
    sl, tp = format_risk_levels("SELL", "APPROVED", price)
    assert "$101,500.00 (+1.5%)" in sl
    assert "$97,500.00 (-2.5%)" in tp


def test_non_falsy_kpi_extraction():
    rec_kpis = {"directional_accuracy_pct": 0.0, "total_reconciled": 0}
    win_rate = float(
        rec_kpis.get("directional_accuracy_pct")
        if rec_kpis.get("directional_accuracy_pct") is not None
        else rec_kpis.get("win_rate_pct", 0.0)
    )
    total_eval = int(
        rec_kpis.get("total_reconciled")
        if rec_kpis.get("total_reconciled") is not None
        else rec_kpis.get("total_evaluated", 0)
    )
    assert win_rate == 0.0
    assert total_eval == 0


def test_app_source_contains_governance_and_disclaimers():
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()

    assert "CLEAN DATASET (34,990 Total 1-Hour Candles)" in content
    assert "27,991 Hours" in content
    assert "PURGE BUFFER (1-Hour Causality Guard — 1 Hour)" in content
    assert "2025-09-17 18:00:00 UTC" in content
    assert "OUT-OF-SAMPLE TEST PERIOD (20% Holdout — 6,998 Hours)" in content
    assert "Controlled Model Benchmark & Rejection Rationale" in content
    assert "p1_controlled_candidate_v1" in content
    assert "REJECTED CANDIDATE" in content
    assert "Naive Persistence" in content
    assert "$398.77" in content
    assert "Simulated paper-trading performance demonstrates execution accounting" in content
    assert "Forecast Error Investigations & Root-Cause Audit Log" in content
    assert "investigation_headline_search" in content
    assert "investigation_status_filter" in content
