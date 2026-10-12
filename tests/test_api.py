# tests/test_api.py - Mudra Automated REST API Test Suite
from src.api.main import app

import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from config import (
    FEATURED_BTC_DATA_PATH,
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES
)
from src.database.models import Base, MarketTick, Reconciliation
from src.engine.paper_portfolio import execute_paper_order
from src.api.schemas import ReconciliationSummaryResponse, PortfolioSummaryResponse


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

def test_reconciliation_summary_endpoint_real_engine_contract(client):
    """
    Verify that /reconciliation/summary executes the real engine/query layer against an
    isolated test database, proving that calculated IC, black swan counts, and tag breakdown
    serialize faithfully into ReconciliationSummaryResponse.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    @contextmanager
    def _test_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    with patch("src.database.connection.get_db", _test_db), \
         patch("src.database.queries.get_db", _test_db):

        # 1. Empty database verification
        resp_empty = client.get("/reconciliation/summary?lookback_hours=24")
        assert resp_empty.status_code == 200
        data_empty = resp_empty.json()
        assert data_empty["total_reconciled"] == 0
        assert data_empty["directional_accuracy_pct"] == 0.0
        assert data_empty["information_coefficient_ic"] == 0.0
        assert data_empty["black_swan_count"] == 0
        assert data_empty["tag_breakdown"] == {}

        # 2. Populated records verification
        now = datetime.now(timezone.utc)
        with _test_db() as session:
            t1 = MarketTick(timestamp=now - timedelta(hours=3), open=60000.0, high=60200.0, low=59900.0, close=60100.0, volume=100.0)
            t2 = MarketTick(timestamp=now - timedelta(hours=2), open=60100.0, high=60300.0, low=60000.0, close=60200.0, volume=100.0)
            t3 = MarketTick(timestamp=now - timedelta(hours=1), open=60200.0, high=60400.0, low=57000.0, close=57200.0, volume=100.0)
            session.add_all([t1, t2, t3])
            session.flush()

            r1 = Reconciliation(
                timestamp=now - timedelta(hours=2),
                tick_id=t1.id,
                predicted_price=60200.0,
                actual_price_at_t_plus_1=60200.0,
                predicted_return_pct=0.0017,
                actual_return_pct=0.0017,
                residual_error=0.0,
                absolute_error_usd=0.0,
                percentage_error=0.0,
                direction_predicted="UP",
                direction_realized="UP",
                direction_correct=True,
                is_black_swan=False,
                error_attribution_tag="ACCURATE",
            )
            r2 = Reconciliation(
                timestamp=now - timedelta(hours=1),
                tick_id=t2.id,
                predicted_price=60100.0,
                actual_price_at_t_plus_1=60150.0,
                predicted_return_pct=-0.0010,
                actual_return_pct=-0.0008,
                residual_error=50.0,
                absolute_error_usd=50.0,
                percentage_error=0.08,
                direction_predicted="DOWN",
                direction_realized="DOWN",
                direction_correct=True,
                is_black_swan=False,
                error_attribution_tag="ACCURATE",
            )
            r3 = Reconciliation(
                timestamp=now,
                tick_id=t3.id,
                predicted_price=60400.0,
                actual_price_at_t_plus_1=57200.0,
                predicted_return_pct=0.002,
                actual_return_pct=-0.051,
                residual_error=-3200.0,
                absolute_error_usd=3200.0,
                percentage_error=5.3,
                direction_predicted="UP",
                direction_realized="DOWN",
                direction_correct=False,
                is_black_swan=True,
                error_attribution_tag="BLACK_SWAN_OUTLIER",
            )
            session.add_all([r1, r2, r3])
            session.commit()

        resp_populated = client.get("/reconciliation/summary?lookback_hours=24")
        assert resp_populated.status_code == 200
        data = resp_populated.json()
        assert data["total_reconciled"] == 3
        assert data["directional_accuracy_pct"] == pytest.approx(66.67, abs=1e-2)
        assert data["black_swan_count"] == 1
        assert isinstance(data["information_coefficient_ic"], float)
        assert -1.0 <= data["information_coefficient_ic"] <= 1.0
        assert data["information_coefficient_ic"] != 0.0
        assert data["tag_breakdown"] == {"ACCURATE": 2, "BLACK_SWAN_OUTLIER": 1}


def test_portfolio_summary_endpoint_real_engine_contract(client):
    """
    Verify that /portfolio/summary executes the real engine/query layer against an
    isolated test database with BUY, SELL, and subsequent HOLD orders.
    Confirms that cumulative realized PnL and total fees are preserved and total equity
    strictly matches cash plus BTC position value.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    @contextmanager
    def _test_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    with patch("src.database.connection.get_db", _test_db), \
         patch("src.database.queries.get_db", _test_db):

        # 1. Empty ledger verification
        resp_empty = client.get("/portfolio/summary?current_price=65000.0")
        assert resp_empty.status_code == 200
        data_empty = resp_empty.json()
        assert data_empty["cash_balance"] == 10000.0
        assert data_empty["btc_balance"] == 0.0
        assert data_empty["current_btc_price"] == 65000.0
        assert data_empty["btc_position_value_usd"] == 0.0
        assert data_empty["total_equity_usd"] == 10000.0
        assert data_empty["cumulative_return_pct"] == 0.0
        assert data_empty["realized_pnl_usd"] == 0.0
        assert data_empty["total_fees_usd"] == 0.0
        assert data_empty["drawdown_pct"] == 0.0
        assert data_empty["total_trades"] == 0

        # 2. Populated ledger: BUY -> SELL -> HOLD
        with _test_db() as session:
            # BUY with 10% Kelly
            execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10)
            # SELL all BTC at $63,000 (+5% profit)
            execute_paper_order(session, "SELL", 63000.0)
            # Subsequent HOLD at $64,000
            execute_paper_order(session, "HOLD", 64000.0)
            session.commit()

        resp_populated = client.get("/portfolio/summary?current_price=64000.0")
        assert resp_populated.status_code == 200
        data = resp_populated.json()
        assert data["total_trades"] == 2
        assert data["current_btc_price"] == 64000.0
        assert data["btc_balance"] == 0.0
        assert data["btc_position_value_usd"] == 0.0
        assert data["cash_balance"] > 10040.0
        assert data["total_equity_usd"] == round(data["cash_balance"] + data["btc_position_value_usd"], 2)
        assert data["realized_pnl_usd"] > 40.0
        assert data["total_fees_usd"] > 0.80  # Cumulative BUY + SELL fees
        assert data["cumulative_return_pct"] > 0.0
        assert data["drawdown_pct"] >= 0.0


def test_endpoints_defensive_fallbacks_and_error_handling(client):
    """
    Verifies that query-level defensive fallbacks return consistent schemas,
    and unhandled server exceptions propagate standard HTTP 500 errors.
    """
    # 1. Query-level fallback returns 200 with structured default schema
    with patch("src.engine.reconciliation.get_reconciliation_summary", side_effect=RuntimeError("DB disconnected")):
        resp_rec = client.get("/reconciliation/summary")
        assert resp_rec.status_code == 200
        assert resp_rec.json()["total_reconciled"] == 0
        assert resp_rec.json()["information_coefficient_ic"] == 0.0

    with patch("src.engine.paper_portfolio.get_portfolio_summary", side_effect=RuntimeError("DB disconnected")):
        resp_port = client.get("/portfolio/summary")
        assert resp_port.status_code == 200
        assert resp_port.json()["total_equity_usd"] == 10000.0
        assert resp_port.json()["cumulative_return_pct"] == 0.0

    # 2. Unhandled exception in endpoint returns HTTP 500
    with patch("src.database.queries.get_reconciliation_kpis", side_effect=Exception("Fatal internal error")):
        resp_rec_err = client.get("/reconciliation/summary")
        assert resp_rec_err.status_code == 500
        assert "Database query failed" in resp_rec_err.json()["detail"]

    with patch("src.database.queries.get_portfolio_kpis", side_effect=Exception("Fatal internal error")):
        resp_port_err = client.get("/portfolio/summary")
        assert resp_port_err.status_code == 500
        assert "Database query failed" in resp_port_err.json()["detail"]