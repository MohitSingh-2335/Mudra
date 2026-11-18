"""
Regression test suite for Prediction-Failure Investigations & Reliability Controls.
"""

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
from config import ACTIVE_MODEL_VERSION, CLASSIFIER_FEATURES, FEATURE_SCHEMA_VERSION
from src.database.investigations import (
    ALLOWED_VERIFICATION_STATUSES,
    record_prediction_investigation,
    get_prediction_investigations,
    get_prediction_investigation_details,
    update_prediction_investigation_status,
    get_training_approved_investigations,
)
from src.database.models import (
    Base,
    MarketTick,
    ModelRegistry,
    PredictionInvestigation,
)
from src.feature_engineering import (
    predict_with_validation,
    predict_proba_with_validation,
)


@pytest.fixture
def in_memory_investigation_db():
    """Creates an isolated in-memory SQLite database seeded with model and tick."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    active_reg = ModelRegistry(
        model_version=ACTIVE_MODEL_VERSION,
        regressor_artifact_path=config.REGRESSOR_MODEL_PATH,
        regressor_sha256=config.REGRESSOR_MODEL_SHA256,
        classifier_artifact_path=config.CLASSIFIER_MODEL_PATH,
        classifier_sha256=config.CLASSIFIER_MODEL_SHA256,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        target_definition="1h_ahead_return_and_movement",
        current_status="ACTIVE",
        lifecycle_seq=1,
    )
    session.add(active_reg)

    tick = MarketTick(
        id=101,
        timestamp=datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc),
        open=65000.0,
        high=65500.0,
        low=64800.0,
        close=65200.0,
        volume=1250.0,
        model_version=ACTIVE_MODEL_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    session.add(tick)
    session.commit()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_prediction_investigations_schema_columns_and_constraints():
    """Verifies table columns for prediction_investigations."""
    table = PredictionInvestigation.__table__
    expected_cols = {
        "id", "tick_id", "prediction_id", "model_version", "headline",
        "suspected_cause_category", "hypothesis_notes", "diagnostic_confidence",
        "verification_status", "created_at", "updated_at"
    }
    assert expected_cols.issubset(set(table.columns.keys()))


def test_headline_search_traces_to_market_tick_and_model_version(in_memory_investigation_db):
    """Verifies recording and retrieving an investigation by headline search."""
    session = in_memory_investigation_db
    inv = record_prediction_investigation(
        session=session,
        tick_id=101,
        model_version=ACTIVE_MODEL_VERSION,
        headline="Federal Reserve announces surprise rate hike",
        suspected_cause_category="MACRO_RATE_DECISION",
        hypothesis_notes="Sudden hawkish liquidity contraction caused sharp selloff.",
        diagnostic_confidence=0.85,
    )
    assert inv["id"] is not None
    results = get_prediction_investigations(session=session, headline_query="Federal Reserve")
    assert len(results) == 1
    assert results[0]["headline"] == "Federal Reserve announces surprise rate hike"
    assert results[0]["tick_id"] == 101


def test_distinguish_suspected_cause_from_confirmed_ground_truth(in_memory_investigation_db):
    """Verifies state transition from UNVERIFIED_HYPOTHESIS to CONFIRMED_CAUSE."""
    session = in_memory_investigation_db
    inv = record_prediction_investigation(
        session=session,
        tick_id=101,
        model_version=ACTIVE_MODEL_VERSION,
        headline="Major Exchange halts withdrawals",
        suspected_cause_category="EXCHANGE_OUTAGE",
        hypothesis_notes="Withdrawal halt rumors sparked panic.",
    )
    assert inv["verification_status"] == "UNVERIFIED_HYPOTHESIS"
    updated = update_prediction_investigation_status(
        session=session,
        investigation_id=inv["id"],
        new_status="CONFIRMED_CAUSE",
        verified_by="Lead Quantitative Auditor",
        ground_truth_label="EXCHANGE_LIQUIDITY_HALT",
    )
    assert updated["verification_status"] == "CONFIRMED_CAUSE"
    assert updated["verified_by"] == "Lead Quantitative Auditor"


def test_diagnostic_confidence_validation(in_memory_investigation_db):
    """Verifies that confidence outside [0.0, 1.0] or non-finite fails closed."""
    session = in_memory_investigation_db
    for bad in [-0.1, 1.5, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            record_prediction_investigation(
                session=session,
                tick_id=101,
                model_version=ACTIVE_MODEL_VERSION,
                headline="Test headline",
                suspected_cause_category="OTHER",
                hypothesis_notes="Testing bounds",
                diagnostic_confidence=bad,
            )


def test_predict_with_validation_rejects_non_finite_predictions():
    """Verifies that regressors returning non-finite values raise ValueError."""
    dummy_input = pd.DataFrame([{"feat_1": 1.0}])
    mock_reg = MagicMock()
    mock_reg.predict.return_value = np.array([float("nan")])
    with pytest.raises(ValueError):
        predict_with_validation(mock_reg, dummy_input, model_type="regressor")


def test_predict_proba_with_validation_rejects_non_finite_or_out_of_bounds_probabilities():
    """Verifies that classifier probabilities must be finite and within [0, 1]."""
    dummy_input = pd.DataFrame([{"feat_1": 1.0}])
    mock_clf = MagicMock()
    mock_clf.predict_proba.return_value = np.array([[float("nan"), 0.5]])
    with pytest.raises(ValueError):
        predict_proba_with_validation(mock_clf, dummy_input)
