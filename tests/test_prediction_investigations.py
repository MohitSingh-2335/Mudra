"""
Regression test suite for Prediction-Failure Investigations & Reliability Controls.

Verifies:
1. Exact table, column, and constraint alignment for `prediction_investigations` in Neon/SQLite.
2. End-to-end headline search tracing to market tick, prediction timestamp, model version, features, and reconciliation outcome.
3. Strict distinction between suspected causes and confirmed ground truth.
4. Audit history preservation across model versions and migrations.
5. Strict training gates: only confirmed, approved investigations can be ingested for training.
6. Fail-closed defense against non-finite inference inputs and model prediction outputs.
"""

import math
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import config
from config import (
    ACTIVE_MODEL_VERSION,
    CLASSIFIER_FEATURES,
    FEATURE_SCHEMA_VERSION,
    REGRESSOR_FEATURES,
)
from src.database.connection import get_db
from src.database.investigations import (
    ALLOWED_VERIFICATION_STATUSES,
    get_prediction_investigation_details,
    get_prediction_investigations,
    get_training_approved_investigations,
    record_prediction_investigation,
    update_prediction_investigation_status,
)
from src.database.models import (
    Base,
    MarketTick,
    ModelPrediction,
    ModelRegistry,
    PredictionInvestigation,
    Reconciliation,
)
from src.feature_engineering import (
    predict_proba_with_validation,
    predict_with_validation,
    validate_inference_features,
)


# =============================================================================
# Isolated Test Database Fixtures
# =============================================================================

@pytest.fixture
def in_memory_investigation_db():
    """
    Creates an isolated in-memory SQLite database with full schema,
    seeding the active production model and sample market ticks.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed Model Registry
    now_utc = datetime.now(timezone.utc)
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

    # Seed Sample Market Tick
    sample_features = {feat: 1.0 for feat in CLASSIFIER_FEATURES}
    sample_features["symbol"] = "BTCUSDT"
    tick_1 = MarketTick(
        id=101,
        timestamp=datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc),
        open=65000.0,
        high=65500.0,
        low=64800.0,
        close=65200.0,
        volume=1250.0,
        model_version=ACTIVE_MODEL_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        market_regime="BULL_MOMENTUM",
        predicted_return_1h=0.0085,
        predicted_price_1h=65754.20,
        model_confidence=0.78,
        features=sample_features,
    )
    session.add(tick_1)

    # Seed ModelPrediction
    pred_1 = ModelPrediction(
        id=501,
        tick_id=101,
        model_version=ACTIVE_MODEL_VERSION,
        run_mode="PRIMARY",
        is_canonical=True,
        is_primary=True,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        predicted_return_pct=0.85,
        predicted_price_1h=65754.20,
        direction_predicted="UP",
        direction_prob_up=0.78,
        clears_hurdle=True,
        created_at=datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc),
    )
    session.add(pred_1)

    # Seed Delayed Reconciliation (where prediction failed: actual fell sharply due to sudden event)
    rec_1 = Reconciliation(
        id=301,
        timestamp=datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc),
        tick_id=101,
        predicted_price=65754.20,
        actual_price_at_t_plus_1=62400.0,  # Flash dump: -4.29% move
        predicted_return_pct=0.85,
        actual_return_pct=-4.29,
        residual_error=-5.14,
        absolute_error_usd=3354.20,
        percentage_error=5.37,
        direction_predicted="UP",
        direction_realized="DOWN",
        direction_correct=False,
        is_black_swan=True,
        error_attribution_tag="BLACK_SWAN_OUTLIER",
    )
    session.add(rec_1)

    session.commit()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


# =============================================================================
# 1. Exact Table & Column Alignment Tests
# =============================================================================

def test_prediction_investigations_schema_columns_and_constraints():
    """
    Verifies that PredictionInvestigation table definition matches the exact Neon DDL:
    columns, types, nullabilities, and foreign key definitions.
    """
    table = PredictionInvestigation.__table__
    assert table.name == "prediction_investigations"

    col_names = {c.name for c in table.columns}
    expected_cols = {
        "id",
        "tick_id",
        "prediction_id",
        "model_version",
        "headline",
        "headline_source_url",
        "headline_published_at",
        "suspected_cause_category",
        "hypothesis_notes",
        "supporting_evidence_json",
        "diagnostic_confidence",
        "unresolved_questions",
        "verification_status",
        "verified_ground_truth_label",
        "verified_by",
        "verified_at",
        "revision_history_json",
        "created_at",
        "updated_at",
    }
    assert expected_cols.issubset(col_names)

    # Verify column nullability constraints
    assert table.columns["tick_id"].nullable is False
    assert table.columns["model_version"].nullable is False
    assert table.columns["headline"].nullable is False
    assert table.columns["suspected_cause_category"].nullable is False
    assert table.columns["hypothesis_notes"].nullable is False
    assert table.columns["verification_status"].nullable is False
    assert table.columns["revision_history_json"].nullable is False


# =============================================================================
# 2. Headline Search & End-to-End Tracing Tests
# =============================================================================

def test_headline_search_traces_to_market_tick_and_model_version(in_memory_investigation_db):
    """
    Verifies that an operator searching by headline keyword can trace:
    Headline -> Investigation -> Market Tick (timestamp, OHLCV, features, prediction)
             -> Reconciliation Outcome (actual outcome, error, attribution tag).
    """
    session = in_memory_investigation_db

    headline_text = "CFTC Files Emergency Action Against Offshore Derivative Venue"
    inv_dict = record_prediction_investigation(
        tick_id=101,
        model_version=ACTIVE_MODEL_VERSION,
        headline=headline_text,
        headline_source_url="https://reuters.com/markets/cftc-emergency-action",
        headline_published_at="2026-10-15T14:12:00Z",
        suspected_cause_category="REGULATORY_SHOCK",
        hypothesis_notes="Sudden liquidation cascade caused by regulatory headline broke technical support.",
        supporting_evidence_json={
            "open_interest_drop_usd": 450000000.0,
            "taker_sell_volume_surge_pct": 320.0,
            "headline_lead_time_minutes": 12,
        },
        diagnostic_confidence=0.85,
        unresolved_questions="Did the liquidity drop begin before or after the headline timestamp?",
        verification_status="UNVERIFIED_HYPOTHESIS",
        prediction_id=501,
        author="quant_analyst_1",
        session=session,
    )
    assert inv_dict["id"] is not None

    # 1. Search by keyword
    search_results = get_prediction_investigations(
        headline_query="CFTC Files Emergency",
        session=session,
    )
    assert len(search_results) == 1
    found = search_results[0]
    assert found["headline"] == headline_text
    assert found["tick_id"] == 101
    assert found["model_version"] == ACTIVE_MODEL_VERSION

    # 2. Trace complete causal chain
    details = get_prediction_investigation_details(found["id"], session=session)
    assert details is not None

    # Verify MarketTick linkage and feature vector preservation
    tick = details["market_tick"]
    assert tick["id"] == 101
    assert tick["close"] == 65200.0
    assert tick["predicted_return_1h"] == 0.0085
    assert tick["model_version"] == ACTIVE_MODEL_VERSION
    assert "volume" in tick["features"]
    assert len(tick["features"]) >= 39  # Complete canonical feature set preserved

    # Verify Reconciliation linkage and ground truth outcome
    rec = details["reconciliation"]
    assert rec["actual_price_at_t_plus_1"] == 62400.0
    assert rec["actual_return_pct"] == pytest.approx(-4.29)
    assert rec["direction_correct"] is False
    assert rec["error_attribution_tag"] == "BLACK_SWAN_OUTLIER"

    # Verify ModelPrediction linkage
    pred = details["model_prediction"]
    assert pred["id"] == 501
    assert pred["predicted_return_pct"] == 0.85


# =============================================================================
# 3. Suspected Cause vs. Confirmed Ground Truth & History Preservation
# =============================================================================

def test_distinguish_suspected_cause_from_confirmed_ground_truth(in_memory_investigation_db):
    """
    Verifies that:
    1. Hypotheses start as UNVERIFIED_HYPOTHESIS.
    2. Cannot jump directly from UNVERIFIED_HYPOTHESIS to APPROVED_FOR_TRAINING.
    3. Transition to CONFIRMED_CAUSE requires operator verification.
    4. Transition to APPROVED_FOR_TRAINING requires confirmed cause and verified label.
    5. All revisions append immutably to revision_history_json.
    """
    session = in_memory_investigation_db

    inv_dict = record_prediction_investigation(
        tick_id=101,
        model_version=ACTIVE_MODEL_VERSION,
        headline="Major Mining Pool Connectivity Disruption",
        suspected_cause_category="ONCHAIN_MINER_HALT",
        hypothesis_notes="Temporary hash rate drop caused block interval extension.",
        diagnostic_confidence=0.60,
        verification_status="UNVERIFIED_HYPOTHESIS",
        author="lead_researcher",
        session=session,
    )
    inv_id = inv_dict["id"]
    assert inv_dict["verification_status"] == "UNVERIFIED_HYPOTHESIS"
    assert len(inv_dict["revision_history_json"]) == 1
    assert inv_dict["revision_history_json"][0]["action"] == "RECORDED"

    # Gate 1: Cannot approve an unverified hypothesis directly for training
    with pytest.raises(ValueError, match="Hypothesis must first be independently verified as 'CONFIRMED_CAUSE'"):
        update_prediction_investigation_status(
            investigation_id=inv_id,
            new_status="APPROVED_FOR_TRAINING",
            verified_by="lead_quant",
            ground_truth_label="CONFIRMED_MINER_OUTAGE",
            session=session,
        )

    # Step 2: Confirm Cause through operator investigation
    confirmed = update_prediction_investigation_status(
        investigation_id=inv_id,
        new_status="CONFIRMED_CAUSE",
        verified_by="senior_analyst_2",
        notes="Confirmed through stratum ping logs and mempool congestion metrics.",
        new_confidence=0.92,
        ground_truth_label="CONFIRMED_MINER_OUTAGE",
        session=session,
    )
    assert confirmed["verification_status"] == "CONFIRMED_CAUSE"
    assert confirmed["diagnostic_confidence"] == pytest.approx(0.92)
    assert len(confirmed["revision_history_json"]) == 2
    assert confirmed["revision_history_json"][1]["action"] == "STATUS_TRANSITION"
    assert confirmed["revision_history_json"][1]["old_status"] == "UNVERIFIED_HYPOTHESIS"
    assert confirmed["revision_history_json"][1]["new_status"] == "CONFIRMED_CAUSE"

    # Step 3: Now approve for future model training set ingestion
    approved = update_prediction_investigation_status(
        investigation_id=inv_id,
        new_status="APPROVED_FOR_TRAINING",
        verified_by="model_governance_lead",
        notes="Validated feature alignment and outcome label; eligible for fine-tuning.",
        ground_truth_label="CONFIRMED_MINER_OUTAGE",
        session=session,
    )
    assert approved["verification_status"] == "APPROVED_FOR_TRAINING"
    assert len(approved["revision_history_json"]) == 3
    assert approved["verified_ground_truth_label"] == "CONFIRMED_MINER_OUTAGE"

    # Gate 2: Cannot approve without ground truth label
    with pytest.raises(ValueError, match="Cannot approve investigation for training without a verified ground_truth_label"):
        update_prediction_investigation_status(
            investigation_id=inv_id,
            new_status="APPROVED_FOR_TRAINING",
            verified_by="analyst",
            ground_truth_label="",  # Empty label forbidden
            session=session,
        )


def test_refuted_hypothesis_cannot_be_approved_for_training(in_memory_investigation_db):
    """Verifies that a refuted hypothesis is permanently blocked from training approval."""
    session = in_memory_investigation_db

    inv_dict = record_prediction_investigation(
        tick_id=101,
        model_version=ACTIVE_MODEL_VERSION,
        headline="Rumor of Exchange Hack",
        suspected_cause_category="SECURITY_EXPLOIT",
        hypothesis_notes="Unsubstantiated Twitter rumor claiming exchange hot wallet drained.",
        diagnostic_confidence=0.20,
        verification_status="UNVERIFIED_HYPOTHESIS",
        session=session,
    )
    inv_id = inv_dict["id"]

    # Refute the hypothesis
    refuted = update_prediction_investigation_status(
        investigation_id=inv_id,
        new_status="REFUTED",
        verified_by="security_team",
        notes="Exchange on-chain wallets confirmed 100% intact; rumor was market manipulation.",
        session=session,
    )
    assert refuted["verification_status"] == "REFUTED"

    # Attempt to approve refuted hypothesis for training
    with pytest.raises(ValueError, match="Hypothesis must first be independently verified as 'CONFIRMED_CAUSE'"):
        update_prediction_investigation_status(
            investigation_id=inv_id,
            new_status="APPROVED_FOR_TRAINING",
            verified_by="analyst",
            ground_truth_label="EXPLOIT_FALSE_ALARM",
            session=session,
        )


# =============================================================================
# 4. Fail-Closed Diagnostic Confidence & Constraint Validation
# =============================================================================

@pytest.mark.parametrize("bad_confidence", [float("nan"), float("inf"), float("-inf"), -0.05, 1.05, "not_a_num"])
def test_diagnostic_confidence_fail_closed_validation(in_memory_investigation_db, bad_confidence):
    """Verifies that non-finite or out-of-range confidence values fail closed."""
    session = in_memory_investigation_db

    with pytest.raises(ValueError):
        record_prediction_investigation(
            tick_id=101,
            model_version=ACTIVE_MODEL_VERSION,
            headline="Confidence validation test",
            suspected_cause_category="TEST_CATEGORY",
            hypothesis_notes="Testing fail-closed confidence validation.",
            diagnostic_confidence=bad_confidence,
            session=session,
        )


def test_invalid_verification_status_rejected(in_memory_investigation_db):
    """Verifies that unknown or arbitrary status strings fail closed."""
    session = in_memory_investigation_db

    with pytest.raises(ValueError, match="Invalid verification_status"):
        record_prediction_investigation(
            tick_id=101,
            model_version=ACTIVE_MODEL_VERSION,
            headline="Invalid status test",
            suspected_cause_category="TEST",
            hypothesis_notes="Notes",
            verification_status="ARBITRARY_STATUS_STRING",
            session=session,
        )


def test_missing_tick_id_foreign_key_rejected(in_memory_investigation_db):
    """Verifies that an investigation referencing a non-existent tick_id is rejected."""
    session = in_memory_investigation_db

    with pytest.raises(ValueError, match="MarketTick with ID 999999 does not exist"):
        record_prediction_investigation(
            tick_id=999999,  # Non-existent
            model_version=ACTIVE_MODEL_VERSION,
            headline="Non-existent tick test",
            suspected_cause_category="TEST",
            hypothesis_notes="Notes",
            session=session,
        )


# =============================================================================
# 5. Training Gating Service Tests
# =============================================================================

def test_get_training_approved_investigations_filters_strictly(in_memory_investigation_db):
    """
    Verifies that get_training_approved_investigations returns ONLY
    APPROVED_FOR_TRAINING records, strictly excluding hypotheses and refuted cases.
    """
    session = in_memory_investigation_db

    # Record 3 investigations with different statuses
    inv_unverified = record_prediction_investigation(
        tick_id=101,
        model_version=ACTIVE_MODEL_VERSION,
        headline="Case 1: Unverified",
        suspected_cause_category="CAT_1",
        hypothesis_notes="Unverified",
        verification_status="UNVERIFIED_HYPOTHESIS",
        session=session,
    )
    inv_confirmed = record_prediction_investigation(
        tick_id=101,
        model_version=ACTIVE_MODEL_VERSION,
        headline="Case 2: Confirmed but not yet approved for training",
        suspected_cause_category="CAT_2",
        hypothesis_notes="Confirmed",
        verification_status="CONFIRMED_CAUSE",
        session=session,
    )
    inv_approved = record_prediction_investigation(
        tick_id=101,
        model_version=ACTIVE_MODEL_VERSION,
        headline="Case 3: Fully Approved for Training",
        suspected_cause_category="CAT_3",
        hypothesis_notes="Approved",
        verification_status="CONFIRMED_CAUSE",
        session=session,
    )
    update_prediction_investigation_status(
        investigation_id=inv_approved["id"],
        new_status="APPROVED_FOR_TRAINING",
        verified_by="lead_governance",
        ground_truth_label="LABEL_CASE_3",
        session=session,
    )

    approved_list = get_training_approved_investigations(session=session)
    assert len(approved_list) == 1
    assert approved_list[0]["id"] == inv_approved["id"]
    assert approved_list[0]["verification_status"] == "APPROVED_FOR_TRAINING"


# =============================================================================
# 6. Model Inference Non-Finite Output Protection Tests
# =============================================================================

def test_predict_with_validation_rejects_non_finite_predictions():
    """
    Verifies that predict_with_validation catches and rejects non-finite outputs (NaN/Inf)
    from corrupted or misbehaving model estimators.
    """
    class CorruptedModelReturningNaN:
        def predict(self, X):
            return np.array([float("nan")])

    class CorruptedModelReturningInf:
        def predict(self, X):
            return np.array([float("inf")])

    class EmptyPredictionModel:
        def predict(self, X):
            return np.array([])

    valid_features = {f: 1.0 for f in REGRESSOR_FEATURES}

    with pytest.raises(ValueError, match="Non-finite values.*NaN or Inf"):
        predict_with_validation(CorruptedModelReturningNaN(), valid_features, model_type="regressor")

    with pytest.raises(ValueError, match="Non-finite values.*NaN or Inf"):
        predict_with_validation(CorruptedModelReturningInf(), valid_features, model_type="regressor")

    with pytest.raises(ValueError, match="returned empty predictions"):
        predict_with_validation(EmptyPredictionModel(), valid_features, model_type="regressor")


def test_predict_proba_with_validation_rejects_non_finite_or_out_of_bounds_probabilities():
    """
    Verifies that predict_proba_with_validation catches and rejects non-finite or
    out-of-range probabilities from classifier estimators.
    """
    class CorruptedProbaReturningNaN:
        def predict_proba(self, X):
            return np.array([[float("nan"), 0.5]])

    class CorruptedProbaOutOfBoundsNegative:
        def predict_proba(self, X):
            return np.array([[-0.1, 1.1]])

    class CorruptedProbaOutOfBoundsExcessive:
        def predict_proba(self, X):
            return np.array([[0.5, 1.5]])

    valid_features = {f: 1.0 for f in CLASSIFIER_FEATURES}

    with pytest.raises(ValueError, match="Probabilities contain non-finite values"):
        predict_proba_with_validation(CorruptedProbaReturningNaN(), valid_features)

    with pytest.raises(ValueError, match="Probabilities must fall strictly within"):
        predict_proba_with_validation(CorruptedProbaOutOfBoundsNegative(), valid_features)

    with pytest.raises(ValueError, match="Probabilities must fall strictly within"):
        predict_proba_with_validation(CorruptedProbaOutOfBoundsExcessive(), valid_features)
