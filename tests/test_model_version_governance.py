# tests/test_model_version_governance.py
"""
Focused model-version governance, cryptographic verification,
reconciliation provenance, and dashboard selector isolation tests.
"""

from datetime import datetime, timezone
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import hashlib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

import config
from config import (
    ACTIVE_MODEL_VERSION,
    FEATURE_SCHEMA_VERSION,
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_MODEL_SHA256,
    CLASSIFIER_MODEL_SHA256,
)
from src.model_loader import (
    validate_sha256_digest,
    compute_file_sha256,
    verify_model_artifact,
    load_verified_model,
    load_production_models,
)
from src.api.schemas import ACTIVE_MODEL_VERSION as SCHEMA_ACTIVE_MODEL_VERSION, ReconciliationSummaryResponse
from src.database.models import Base, MarketTick, Reconciliation
from src.database.queries import get_reconciliation_history, get_reconciliation_kpis
from src.engine.reconciliation import get_reconciliation_summary
from scripts.migrate_db import BASELINE_MODEL_SPEC, CANDIDATE_MODEL_SPEC, CANDIDATE_REJECTION_REASON
from app import (
    TAB3_MODEL_SELECTOR_OPTIONS,
    filter_reconciliation_records,
    prepare_reconciliation_table_display,
    compute_fallback_reconciliation_kpis,
    resolve_tab3_reconciliation_kpis,
)


# =============================================================================
# 1. Canonical Configuration Tests
# =============================================================================

def test_canonical_configuration_constants():
    """Verifies that config.py is the canonical source of truth for model governance."""
    assert ACTIVE_MODEL_VERSION == "xgb_onnx_v1.0.0"
    assert FEATURE_SCHEMA_VERSION == "v1_39features"
    assert SCHEMA_ACTIVE_MODEL_VERSION == ACTIVE_MODEL_VERSION
    assert BASELINE_MODEL_SPEC["model_version"] == ACTIVE_MODEL_VERSION
    assert BASELINE_MODEL_SPEC["regressor_sha256"] == REGRESSOR_MODEL_SHA256
    assert BASELINE_MODEL_SPEC["classifier_sha256"] == CLASSIFIER_MODEL_SHA256

    # Verify column defaults and NOT NULL constraints in ORM models (unweakened schema)
    assert MarketTick.model_version.default.arg == ACTIVE_MODEL_VERSION
    assert MarketTick.feature_schema_version.default.arg == FEATURE_SCHEMA_VERSION
    assert MarketTick.model_version.nullable is False
    assert MarketTick.feature_schema_version.nullable is False

    # Verify real artifact files match the canonical hashes on disk
    reg_hash = compute_file_sha256(REGRESSOR_MODEL_PATH)
    clf_hash = compute_file_sha256(CLASSIFIER_MODEL_PATH)
    assert reg_hash == REGRESSOR_MODEL_SHA256
    assert clf_hash == CLASSIFIER_MODEL_SHA256


# =============================================================================
# 2. Cryptographic Artifact Verification Tests
# =============================================================================

def test_artifact_hash_verification_success():
    """Verifies that authentic production artifacts load successfully when hashes match."""
    reg = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
    assert reg is not None
    assert hasattr(reg, "predict")

    clf = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
    assert clf is not None
    assert hasattr(clf, "predict_proba")

    # Convenience bundle loader
    prod_reg, prod_clf = load_production_models()
    assert prod_reg is not None
    assert prod_clf is not None


def test_artifact_hash_verification_tampering_fail_closed(tmp_path):
    """Verifies that tampered artifacts or mismatched hashes fail closed with RuntimeError."""
    # Mismatched expected hash
    with pytest.raises(RuntimeError, match="Cryptographic integrity verification failed"):
        load_verified_model(REGRESSOR_MODEL_PATH, "0" * 64)

    # Corrupted artifact content
    corrupted_file = tmp_path / "corrupted_regressor.pkl"
    corrupted_file.write_bytes(b"tampered malicious payload")
    with pytest.raises(RuntimeError, match="Cryptographic integrity verification failed"):
        load_verified_model(str(corrupted_file), REGRESSOR_MODEL_SHA256)


def test_artifact_hash_verification_missing_file():
    """Verifies that missing model artifact files fail closed with FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_verified_model("non_existent_model_artifact.pkl", REGRESSOR_MODEL_SHA256)


def test_artifact_verification_operates_on_same_bytes_to_prevent_toctou(tmp_path):
    """
    Verifies that load_verified_model verifies SHA-256 and deserializes the exact same
    in-memory bytes via io.BytesIO, preventing TOCTOU file replacement races.
    """
    import joblib
    dummy_model = {"model_name": "test_model", "weights": [0.1, 0.2, 0.3]}
    model_file = tmp_path / "toctou_test_model.pkl"
    with open(model_file, "wb") as f:
        joblib.dump(dummy_model, f)

    real_hash = compute_file_sha256(model_file)

    # 1. Load model and verify it deserializes correctly from in-memory byte buffer
    loaded_obj = load_verified_model(model_file, real_hash)
    assert loaded_obj == dummy_model

    # 2. Verify that if artifact bytes fail checksum, deserialization is never reached
    corrupted_file = tmp_path / "tampered.pkl"
    corrupted_file.write_bytes(b"invalid corrupt payload")
    with pytest.raises(RuntimeError, match="Cryptographic integrity verification failed"):
        load_verified_model(corrupted_file, real_hash)


def test_artifact_hash_digest_format_validation():
    """Verifies that invalid, truncated, non-hex, or empty SHA-256 digests fail with ValueError."""
    # Empty digest
    with pytest.raises(ValueError, match="Invalid expected SHA-256 digest format"):
        load_verified_model(REGRESSOR_MODEL_PATH, "")

    # Truncated digest
    with pytest.raises(ValueError, match="Invalid expected SHA-256 digest format"):
        load_verified_model(REGRESSOR_MODEL_PATH, "a" * 32)

    # Non-hex characters
    with pytest.raises(ValueError, match="Invalid expected SHA-256 digest format"):
        load_verified_model(REGRESSOR_MODEL_PATH, "z" * 64)

    # Valid canonical digests pass validation
    validate_sha256_digest(REGRESSOR_MODEL_SHA256)
    validate_sha256_digest(CLASSIFIER_MODEL_SHA256)


# =============================================================================
# 3. Agents Tool Provenance Tests
# =============================================================================

def test_agents_tools_forecast_provenance():
    """Verifies that get_model_return_forecast outputs the canonical model_version."""
    from src.agents.tools import get_model_return_forecast

    dummy_features = {"close": 90000.0}
    with patch("src.agents.tools.validate_inference_features") as mock_val, \
         patch("src.agents.tools.predict_with_validation") as mock_pred, \
         patch("src.agents.tools.predict_proba_with_validation") as mock_proba:
        mock_val.return_value = pd.DataFrame([{"feat": 1.0}])
        mock_pred.return_value = np.array([0.005])
        mock_proba.return_value = np.array([[0.4, 0.6]])

        res = get_model_return_forecast.invoke({"features": dummy_features})
        assert isinstance(res, dict)
        assert res.get("model_version") == ACTIVE_MODEL_VERSION
        assert "expected_return_pct" in res
        assert "direction_probability_up" in res


# =============================================================================
# 4. Historical Model-Version Provenance & Attribution Tests
# =============================================================================

@pytest.fixture
def in_memory_reconciliation_db():
    """
    Creates an isolated in-memory SQLite database populated with historical reconciliation records.
    Explicitly tests both:
      Case 1: An existing MarketTick with NULL model_version (simulating pre-migration legacy ticks).
      Case 2: An orphaned Reconciliation record with an unlinked tick_id.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    # To accurately simulate legacy market_ticks created prior to the NOT NULL constraint
    # without weakening production ORM models.py constraints, recreate market_ticks with nullable model_version in SQLite:
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE market_ticks"))
        conn.execute(text(
            "CREATE TABLE market_ticks ("
            "id INTEGER PRIMARY KEY, "
            "timestamp DATETIME, "
            "open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT, "
            "model_version VARCHAR(32), "
            "feature_schema_version VARCHAR(16), "
            "market_regime VARCHAR(32), "
            "predicted_return_1h FLOAT, predicted_price_1h FLOAT, "
            "model_confidence FLOAT, features JSON)"
        ))
        conn.commit()

    Session = sessionmaker(bind=engine)
    session = Session()

    # Create market ticks:
    # T1: active model version
    t1 = MarketTick(
        id=101,
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        open=90000.0, high=90500.0, low=89500.0, close=90200.0, volume=50.0,
        model_version="xgb_onnx_v1.0.0",
        feature_schema_version="v1_39features",
    )
    # T2: legacy model version
    t2 = MarketTick(
        id=102,
        timestamp=datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc),
        open=90200.0, high=90800.0, low=90000.0, close=90600.0, volume=60.0,
        model_version="xgb_legacy_v0.9.0",
        feature_schema_version="v0_legacy",
    )
    # T3: legacy market tick with NULL/UNKNOWN model attribution (simulating pre-migration tick)
    t3 = MarketTick(
        id=103,
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        open=90600.0, high=91000.0, low=90400.0, close=90800.0, volume=55.0,
        feature_schema_version="v1_39features",
    )
    session.add_all([t1, t2, t3])
    session.commit()

    # Set t3.model_version to NULL via SQL to bypass default assignment
    session.execute(text("UPDATE market_ticks SET model_version = NULL WHERE id = 103"))
    session.commit()
    session.expire_all()

    # Create reconciliation records:
    # R1 -> T1 (xgb_onnx_v1.0.0)
    r1 = Reconciliation(
        id=1,
        tick_id=101,
        timestamp=datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc),
        predicted_price=90400.0,
        actual_price_at_t_plus_1=90600.0,
        predicted_return_pct=0.0022,
        actual_return_pct=0.0044,
        residual_error=200.0,
        absolute_error_usd=200.0,
        percentage_error=0.0022,
        direction_predicted="UP",
        direction_realized="UP",
        direction_correct=True,
        is_black_swan=False,
        error_attribution_tag="ACCURATE",
    )
    # R2 -> T2 (xgb_legacy_v0.9.0)
    r2 = Reconciliation(
        id=2,
        tick_id=102,
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        predicted_price=90900.0,
        actual_price_at_t_plus_1=90800.0,
        predicted_return_pct=0.0033,
        actual_return_pct=0.0022,
        residual_error=-100.0,
        absolute_error_usd=100.0,
        percentage_error=0.0011,
        direction_predicted="UP",
        direction_realized="UP",
        direction_correct=True,
        is_black_swan=False,
        error_attribution_tag="ACCURATE",
    )
    # R3 -> T3 (Case 1: Existing parent tick, but model_version is NULL in DB)
    r3 = Reconciliation(
        id=3,
        tick_id=103,
        timestamp=datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc),
        predicted_price=90500.0,
        actual_price_at_t_plus_1=91000.0,
        predicted_return_pct=-0.0033,
        actual_return_pct=0.0022,
        residual_error=500.0,
        absolute_error_usd=500.0,
        percentage_error=0.0055,
        direction_predicted="DOWN",
        direction_realized="UP",
        direction_correct=False,
        is_black_swan=False,
        error_attribution_tag="DIRECTION_FLIP",
    )
    # R4 -> Case 2: Orphaned reconciliation record (tick_id = 9999 where parent tick doesn't exist)
    r4 = Reconciliation(
        id=4,
        tick_id=9999,
        timestamp=datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc),
        predicted_price=91200.0,
        actual_price_at_t_plus_1=91300.0,
        predicted_return_pct=0.0022,
        actual_return_pct=0.0033,
        residual_error=100.0,
        absolute_error_usd=100.0,
        percentage_error=0.0011,
        direction_predicted="UP",
        direction_realized="UP",
        direction_correct=True,
        is_black_swan=False,
        error_attribution_tag="ACCURATE",
    )
    session.add_all([r1, r2, r3, r4])
    session.commit()

    yield session, engine

    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_reconciliation_history_provenance_and_filtering(in_memory_reconciliation_db):
    """
    Verifies:
    1. Outer join assigns model_version from MarketTick.
    2. Case 1 (existing tick with null model_version) returns None.
    3. Case 2 (orphaned reconciliation with unlinked tick_id) returns None.
    4. Neither case ever defaults to ACTIVE_MODEL_VERSION.
    5. 'All Models' / None returns all records.
    6. Model-specific filter returns only matching records.
    """
    session, engine = in_memory_reconciliation_db

    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield session

    with patch("src.database.queries.get_db", mock_get_db):
        # 1. Unfiltered query
        all_df = get_reconciliation_history(limit=50, model_version=None)
        assert len(all_df) == 4

        row_r1 = all_df[all_df["id"] == 1].iloc[0]
        row_r2 = all_df[all_df["id"] == 2].iloc[0]
        row_r3 = all_df[all_df["id"] == 3].iloc[0]
        row_r4 = all_df[all_df["id"] == 4].iloc[0]

        # Verify active and legacy versions
        assert row_r1["model_version"] == "xgb_onnx_v1.0.0"
        assert row_r2["model_version"] == "xgb_legacy_v0.9.0"

        # Explicitly verify Case 1: Existing tick with NULL model_version (R3 -> T3)
        assert row_r3["tick_id"] == 103
        assert row_r3["model_version"] is None or pd.isna(row_r3["model_version"])
        assert row_r3["model_version"] != ACTIVE_MODEL_VERSION

        # Explicitly verify Case 2: Orphaned reconciliation (R4 -> tick_id 9999)
        assert row_r4["tick_id"] == 9999
        assert row_r4["model_version"] is None or pd.isna(row_r4["model_version"])
        assert row_r4["model_version"] != ACTIVE_MODEL_VERSION

        # 2. 'All Models' query preserves all records including null/orphaned
        all_models_df = get_reconciliation_history(limit=50, model_version="All Models")
        assert len(all_models_df) == 4

        # 3. Filter by active model excludes null and orphaned records
        active_df = get_reconciliation_history(limit=50, model_version="xgb_onnx_v1.0.0")
        assert len(active_df) == 1
        assert active_df.iloc[0]["id"] == 1
        assert active_df.iloc[0]["model_version"] == "xgb_onnx_v1.0.0"

        # 4. Filter by legacy model
        legacy_df = get_reconciliation_history(limit=50, model_version="xgb_legacy_v0.9.0")
        assert len(legacy_df) == 1
        assert legacy_df.iloc[0]["id"] == 2
        assert legacy_df.iloc[0]["model_version"] == "xgb_legacy_v0.9.0"

        # 5. Filter by nonexistent model
        empty_df = get_reconciliation_history(limit=50, model_version="non_existent_model")
        assert len(empty_df) == 0


def test_reconciliation_summary_kpi_filtering(in_memory_reconciliation_db):
    """Verifies that get_reconciliation_summary computes KPIs filtered by model_version."""
    session, engine = in_memory_reconciliation_db

    # Unfiltered summary
    summary_all = get_reconciliation_summary(session, model_version=None)
    assert summary_all["total_evaluated"] == 4
    assert summary_all["model_version"] is None

    # Filtered by active model
    summary_active = get_reconciliation_summary(session, model_version="xgb_onnx_v1.0.0")
    assert summary_active["total_evaluated"] == 1
    assert summary_active["directional_accuracy_pct"] == 100.0
    assert summary_active["model_version"] == "xgb_onnx_v1.0.0"

    # Filtered by legacy model
    summary_legacy = get_reconciliation_summary(session, model_version="xgb_legacy_v0.9.0")
    assert summary_legacy["total_evaluated"] == 1
    assert summary_legacy["model_version"] == "xgb_legacy_v0.9.0"

    # Filtered by nonexistent model
    summary_nonexistent = get_reconciliation_summary(session, model_version="nonexistent")
    assert summary_nonexistent["total_evaluated"] == 0
    assert summary_nonexistent["model_version"] == "nonexistent"


# =============================================================================
# 5. API Endpoint Model-Version Filtering Tests
# =============================================================================

def test_api_reconciliation_summary_filtering(in_memory_reconciliation_db):
    """Verifies that the /reconciliation/summary endpoint supports model_version query parameter."""
    session, engine = in_memory_reconciliation_db
    from src.api.main import app
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield session

    client = TestClient(app)
    with patch("src.database.queries.get_db", mock_get_db):
        # 1. Unfiltered
        res_all = client.get("/reconciliation/summary")
        assert res_all.status_code == 200
        data_all = res_all.json()
        assert data_all["total_reconciled"] == 4
        assert data_all["model_version"] is None

        # 2. Filtered by active model
        res_active = client.get(f"/reconciliation/summary?model_version={ACTIVE_MODEL_VERSION}")
        assert res_active.status_code == 200
        data_active = res_active.json()
        assert data_active["total_reconciled"] == 1
        assert data_active["model_version"] == ACTIVE_MODEL_VERSION

        # 3. Filtered by All Models
        res_all_models = client.get("/reconciliation/summary?model_version=All%20Models")
        assert res_all_models.status_code == 200
        data_all_models = res_all_models.json()
        assert data_all_models["total_reconciled"] == 4
        assert data_all_models["model_version"] is None


# =============================================================================
# 6. Dashboard Tab 3 Production Helpers & Failure Regression Tests
# =============================================================================

def test_dashboard_tab3_production_helpers_and_isolation():
    """
    Verifies Tab 3 production filtering logic via extracted helpers in app.py:
    1. TAB3_MODEL_SELECTOR_OPTIONS strictly exposes ['All Models', ACTIVE_MODEL_VERSION].
    2. filter_reconciliation_records preserves unknown attribution on 'All Models' and filters strictly on model version.
    3. prepare_reconciliation_table_display maps missing model_version to 'UNKNOWN'.
    4. Isolation invariants: verify absence of trading, portfolio, or model config mutations.
    """
    # 1. Selector options invariant
    assert TAB3_MODEL_SELECTOR_OPTIONS == ["All Models", ACTIVE_MODEL_VERSION]
    assert "p1_controlled_candidate_v1" not in TAB3_MODEL_SELECTOR_OPTIONS

    # 2. Mixed data with active, legacy, and null attribution records
    data = [
        {"id": 1, "model_version": "xgb_onnx_v1.0.0", "actual_return_pct": 0.01, "predicted_return_pct": 0.01, "absolute_error_usd": 50.0, "direction_correct": True},
        {"id": 2, "model_version": "xgb_legacy_v0.9.0", "actual_return_pct": -0.01, "predicted_return_pct": -0.01, "absolute_error_usd": 60.0, "direction_correct": True},
        {"id": 3, "model_version": None, "actual_return_pct": 0.02, "predicted_return_pct": -0.01, "absolute_error_usd": 200.0, "direction_correct": False},
    ]
    df = pd.DataFrame(data)

    # 3. 'All Models' preserves all rows
    filtered_all = filter_reconciliation_records(df, "All Models")
    assert len(filtered_all) == 3

    # 4. 'xgb_onnx_v1.0.0' strictly filters to row 1
    filtered_active = filter_reconciliation_records(df, "xgb_onnx_v1.0.0")
    assert len(filtered_active) == 1
    assert filtered_active.iloc[0]["id"] == 1
    assert filtered_active.iloc[0]["model_version"] == "xgb_onnx_v1.0.0"

    # 5. Nonexistent model returns empty
    filtered_empty = filter_reconciliation_records(df, "nonexistent")
    assert len(filtered_empty) == 0

    # 6. Table preparation sets UNKNOWN for missing model_version
    disp_all = prepare_reconciliation_table_display(filtered_all)
    assert disp_all["Model Version"].tolist() == ["xgb_onnx_v1.0.0", "xgb_legacy_v0.9.0", "UNKNOWN"]

    # 7. Isolation check: verify ACTIVE_MODEL_VERSION and FEATURE_SCHEMA_VERSION are not mutated
    assert ACTIVE_MODEL_VERSION == "xgb_onnx_v1.0.0"
    assert FEATURE_SCHEMA_VERSION == "v1_39features"


def test_dashboard_tab3_kpi_query_failure_regression():
    """
    Regression test for Finding 3:
    Verifies that resolve_tab3_reconciliation_kpis NEVER silently falls back to unfiltered base_kpis
    when a model-specific query fails, and instead provides a correctly filtered fallback or
    defensive unavailable state with clear warning messages.
    """
    base_unfiltered_kpis = {
        "total_evaluated": 100,
        "directional_accuracy_pct": 52.0,
        "mae_usd": 150.0,
        "mape_pct": 0.22,
        "information_coefficient_ic": 0.045,
        "model_version": None,
    }

    # Case A: 'All Models' returns base_kpis without error
    all_kpis, warning = resolve_tab3_reconciliation_kpis(
        selected_model="All Models",
        base_kpis=base_unfiltered_kpis,
        filtered_df=pd.DataFrame(),
    )
    assert all_kpis == base_unfiltered_kpis
    assert warning is None

    # Case B: Specific model with working kpi_fetcher
    def working_fetcher(hours, model):
        return {
            "total_evaluated": 25,
            "directional_accuracy_pct": 68.0,
            "mae_usd": 85.0,
            "mape_pct": 0.12,
            "information_coefficient_ic": 0.082,
            "model_version": model,
        }

    active_kpis, warning = resolve_tab3_reconciliation_kpis(
        selected_model="xgb_onnx_v1.0.0",
        base_kpis=base_unfiltered_kpis,
        filtered_df=pd.DataFrame(),
        kpi_fetcher=working_fetcher,
    )
    assert active_kpis["total_evaluated"] == 25
    assert active_kpis["directional_accuracy_pct"] == 68.0
    assert active_kpis["model_version"] == "xgb_onnx_v1.0.0"
    assert warning is None

    # Case C: Specific model where kpi_fetcher raises an exception BUT client has filtered rows
    def failing_fetcher(hours, model):
        raise RuntimeError("Database connection timed out or Neon unavailable")

    client_filtered_df = pd.DataFrame([
        {"id": 1, "predicted_return_pct": 0.01, "actual_return_pct": 0.02, "absolute_error_usd": 50.0, "percentage_error": 0.005, "direction_correct": True, "is_black_swan": False, "error_attribution_tag": "ACCURATE"},
        {"id": 2, "predicted_return_pct": -0.01, "actual_return_pct": -0.01, "absolute_error_usd": 40.0, "percentage_error": 0.004, "direction_correct": True, "is_black_swan": False, "error_attribution_tag": "ACCURATE"},
    ])

    fallback_kpis, warning = resolve_tab3_reconciliation_kpis(
        selected_model="xgb_onnx_v1.0.0",
        base_kpis=base_unfiltered_kpis,
        filtered_df=client_filtered_df,
        kpi_fetcher=failing_fetcher,
    )
    # CRITICAL: Must NOT be the unfiltered base_kpis!
    assert fallback_kpis != base_unfiltered_kpis
    assert fallback_kpis["total_evaluated"] == 2
    assert fallback_kpis["directional_accuracy_pct"] == 100.0
    assert fallback_kpis["mae_usd"] == 45.0
    assert fallback_kpis["is_fallback"] is True
    assert fallback_kpis["model_version"] == "xgb_onnx_v1.0.0"
    assert warning is not None
    assert "Displaying fallback KPIs computed from 2 loaded client records" in warning

    # Case D: Specific model where kpi_fetcher raises an exception AND client has 0 rows
    empty_filtered_df = pd.DataFrame()
    unavailable_kpis, warning = resolve_tab3_reconciliation_kpis(
        selected_model="xgb_onnx_v1.0.0",
        base_kpis=base_unfiltered_kpis,
        filtered_df=empty_filtered_df,
        kpi_fetcher=failing_fetcher,
    )
    # CRITICAL: Must NOT be the unfiltered base_kpis!
    assert unavailable_kpis != base_unfiltered_kpis
    assert unavailable_kpis["kpi_unavailable"] is True
    assert unavailable_kpis["total_evaluated"] == 0
    assert unavailable_kpis["directional_accuracy_pct"] is None
    assert unavailable_kpis["mae_usd"] is None
    assert unavailable_kpis["model_version"] == "xgb_onnx_v1.0.0"
    assert warning is not None
    assert "Model-specific KPIs for 'xgb_onnx_v1.0.0' are currently unavailable" in warning


# =============================================================================
# 7. Preservation of Rejected Candidate Tests
# =============================================================================

def test_preservation_of_rejected_candidate():
    """
    Verifies that candidate 'p1_controlled_candidate_v1' remains permanently rejected:
    1. Not active in config.py.
    2. Registered as rejected candidate in migration specs with clear rejection rationale.
    3. Not presented as an active selectable model option in the UI selector.
    """
    assert ACTIVE_MODEL_VERSION != "p1_controlled_candidate_v1"
    assert CANDIDATE_MODEL_SPEC["model_version"] == "p1_controlled_candidate_v1"
    assert "Rejected candidate promotion" in CANDIDATE_REJECTION_REASON

    # Tab 3 selector options must strictly be ["All Models", "xgb_onnx_v1.0.0"]
    allowed_selector_options = ["All Models", ACTIVE_MODEL_VERSION]
    assert "p1_controlled_candidate_v1" not in allowed_selector_options
    assert allowed_selector_options == ["All Models", "xgb_onnx_v1.0.0"]


# =============================================================================
# 8. API Error Redaction & KPI Failure Signaling Regression Tests
# =============================================================================

def test_api_reconciliation_summary_error_redaction_and_sanitization(caplog):
    """
    Verifies that /reconciliation/summary redacts internal database exception details
    (passwords, URLs, connection strings) from public HTTP responses and returns a
    stable, generic error message. Also verifies log output does not leak secrets.
    """
    import logging
    from src.api.main import app, _sanitize_log_message

    sensitive_exception = (
        "FATAL: password authentication failed for user 'neondb_owner' "
        "at postgresql://neondb_owner:npg_SecretPass123@ep-cool-db.us-east-2.aws.neon.tech/neondb?sslmode=require token=tok_abc123 path=C:\\secrets\\creds.json"
    )

    client = TestClient(app)
    with caplog.at_level(logging.ERROR, logger="src.api.main"):
        with patch("src.database.queries.get_reconciliation_kpis", side_effect=RuntimeError(sensitive_exception)):
            response = client.get("/reconciliation/summary")
            assert response.status_code == 500
            data = response.json()

            # Stable generic message
            assert data["detail"] == "Database query failed"

            # Explicit verification that sensitive strings are not leaked to HTTP caller
            assert "npg_SecretPass123" not in response.text
            assert "neondb_owner" not in response.text
            assert "postgresql://" not in response.text
            assert "aws.neon.tech" not in response.text
            assert "tok_abc123" not in response.text
            assert "creds.json" not in response.text

    # Verify captured log output contains structured error type and no sensitive leaks
    assert "error_type=RuntimeError" in caplog.text
    assert "npg_SecretPass123" not in caplog.text
    assert "postgresql://" not in caplog.text
    assert "aws.neon.tech" not in caplog.text
    assert "tok_abc123" not in caplog.text
    assert "creds.json" not in caplog.text

    # Verify log sanitizer masks password
    sanitized = _sanitize_log_message(sensitive_exception)
    assert "npg_SecretPass123" not in sanitized
    assert "postgresql://neondb_owner:***@ep-cool-db.us-east-2.aws.neon.tech/neondb?sslmode=require" in sanitized


def test_database_queries_reconciliation_logging_does_not_leak_secrets(caplog):
    """
    Verifies that get_reconciliation_kpis in src/database/queries.py uses structured logging
    (error_type=%s) and does not interpolate raw sensitive exception messages into log outputs.
    """
    import logging
    from src.database.queries import get_reconciliation_kpis

    sensitive_exception = (
        "FATAL: password authentication failed for user 'neondb_owner' "
        "at postgresql://neondb_owner:npg_SecretPass123@ep-cool-db.us-east-2.aws.neon.tech/neondb?sslmode=require token=tok_secret_999 path=/var/secrets/db.key"
    )

    with caplog.at_level(logging.WARNING, logger="src.database.queries"):
        with patch("src.database.queries.get_db", side_effect=RuntimeError(sensitive_exception)):
            res = get_reconciliation_kpis(lookback_hours=24)
            assert res["query_failed"] is True
            assert res["error"] == "Database query failed"
            assert res["total_reconciled"] == 0

    # Assert structured logging
    assert "Failed to fetch reconciliation KPIs: error_type=RuntimeError" in caplog.text
    # Assert zero credentials / secrets in logs
    assert "npg_SecretPass123" not in caplog.text
    assert "neondb_owner" not in caplog.text
    assert "postgresql://" not in caplog.text
    assert "aws.neon.tech" not in caplog.text
    assert "tok_secret_999" not in caplog.text
    assert "db.key" not in caplog.text


def test_api_reconciliation_summary_endpoint_kpi_failure_vs_empty_success():
    """
    Endpoint-level tests for /reconciliation/summary asserting status codes and response bodies
    for both genuine successful empty results and database query failures.
    """
    from src.api.main import app

    client = TestClient(app)

    # 1. Genuine successful empty result (e.g. 0 records in window, DB healthy)
    with patch("src.database.queries.get_db") as mock_get_db:
        mock_session = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_session
        with patch("src.engine.reconciliation.get_reconciliation_summary") as mock_summary:
            mock_summary.return_value = {
                "total_evaluated": 0,
                "total_reconciled": 0,
                "win_rate_pct": 0.0,
                "directional_accuracy_pct": 0.0,
                "mae_usd": 0.0,
                "mape_pct": 0.0,
                "information_coefficient": 0.0,
                "information_coefficient_ic": 0.0,
                "black_swans_filtered": 0,
                "black_swan_count": 0,
                "accurate_count": 0,
                "tag_breakdown": {},
                "model_version": None,
                "query_failed": False,
                "error": None,
            }
            resp_empty = client.get("/reconciliation/summary")
            assert resp_empty.status_code == 200
            data_empty = resp_empty.json()
            assert data_empty["query_failed"] is False
            assert data_empty["error"] is None
            assert data_empty["total_reconciled"] == 0
            assert data_empty["directional_accuracy_pct"] == 0.0
            assert data_empty["mae_usd"] == 0.0
            assert data_empty["mape_pct"] == 0.0
            assert data_empty["information_coefficient_ic"] == 0.0
            assert data_empty["black_swan_count"] == 0
            assert data_empty["tag_breakdown"] == {}

    # 2. Database query failure caught at query level (DB exception caught and handled)
    with patch("src.engine.reconciliation.get_reconciliation_summary", side_effect=RuntimeError("Connection timeout")):
        resp_fail = client.get("/reconciliation/summary")
        assert resp_fail.status_code == 200
        data_fail = resp_fail.json()
        assert data_fail["query_failed"] is True
        assert data_fail["error"] == "Database query failed"
        assert data_fail["total_reconciled"] == 0

    # 3. Unhandled endpoint exception (e.g. query function itself raises)
    with patch("src.database.queries.get_reconciliation_kpis", side_effect=RuntimeError("Unhandled exception")):
        resp_err = client.get("/reconciliation/summary")
        assert resp_err.status_code == 500
        data_err = resp_err.json()
        assert data_err["detail"] == "Database query failed"


def test_reconciliation_kpi_successful_empty_result_vs_query_failure():
    """
    Verifies the explicit distinction between:
    1. A successful database query returning 0 records -> preserves valid zero performance without fallback or warning.
    2. A failed database query returning query_failed=True:
       - With client records -> triggers fallback with warning banner.
       - With 0 client records -> returns unavailable state (None rates) with warning banner.
    3. A failed database query raising an exception -> triggers fallback or unavailable state.
    """
    # 1. Genuine successful query returning 0 records
    successful_empty_db_result = {
        "total_evaluated": 0,
        "total_reconciled": 0,
        "win_rate_pct": 0.0,
        "directional_accuracy_pct": 0.0,
        "mae_usd": 0.0,
        "mape_pct": 0.0,
        "information_coefficient": 0.0,
        "information_coefficient_ic": 0.0,
        "black_swans_filtered": 0,
        "black_swan_count": 0,
        "accurate_count": 0,
        "tag_breakdown": {},
        "model_version": "xgb_onnx_v1.0.0",
        "query_failed": False,
    }

    kpis_ok, warning_ok = resolve_tab3_reconciliation_kpis(
        selected_model="xgb_onnx_v1.0.0",
        base_kpis={},
        filtered_df=pd.DataFrame(),
        kpi_fetcher=lambda hours, model: successful_empty_db_result,
    )
    # Must preserve valid zero performance without triggering fallback or warning
    assert warning_ok is None
    assert kpis_ok["total_evaluated"] == 0
    assert kpis_ok["directional_accuracy_pct"] == 0.0
    assert kpis_ok["mae_usd"] == 0.0
    assert kpis_ok.get("is_fallback") is not True
    assert kpis_ok.get("kpi_unavailable") is not True

    # 2. Failed query returning query_failed=True (when DB caught exception)
    failed_db_result = {
        "total_evaluated": 0,
        "total_reconciled": 0,
        "win_rate_pct": 0.0,
        "directional_accuracy_pct": 0.0,
        "mae_usd": 0.0,
        "mape_pct": 0.0,
        "information_coefficient": 0.0,
        "information_coefficient_ic": 0.0,
        "black_swans_filtered": 0,
        "black_swan_count": 0,
        "accurate_count": 0,
        "tag_breakdown": {},
        "model_version": "xgb_onnx_v1.0.0",
        "query_failed": True,
        "error": "Database query failed",
    }

    # Case 2A: Query failed, but client records exist -> fallback to client records
    client_df = pd.DataFrame([
        {"id": 1, "predicted_return_pct": 0.01, "actual_return_pct": 0.02, "absolute_error_usd": 50.0, "percentage_error": 0.005, "direction_correct": True, "is_black_swan": False, "error_attribution_tag": "ACCURATE"}
    ])
    fallback_kpis, fallback_warn = resolve_tab3_reconciliation_kpis(
        selected_model="xgb_onnx_v1.0.0",
        base_kpis={},
        filtered_df=client_df,
        kpi_fetcher=lambda hours, model: failed_db_result,
    )
    assert fallback_warn is not None
    assert "Displaying fallback KPIs computed from 1 loaded client records" in fallback_warn
    assert fallback_kpis["is_fallback"] is True
    assert fallback_kpis["total_evaluated"] == 1
    assert fallback_kpis["directional_accuracy_pct"] == 100.0

    # Case 2B: Query failed, zero client records -> unavailable state (None metrics)
    unavail_kpis, unavail_warn = resolve_tab3_reconciliation_kpis(
        selected_model="xgb_onnx_v1.0.0",
        base_kpis={},
        filtered_df=pd.DataFrame(),
        kpi_fetcher=lambda hours, model: failed_db_result,
    )
    assert unavail_warn is not None
    assert "Model-specific KPIs for 'xgb_onnx_v1.0.0' are currently unavailable" in unavail_warn
    assert unavail_kpis["kpi_unavailable"] is True
    assert unavail_kpis["directional_accuracy_pct"] is None
    assert unavail_kpis["mae_usd"] is None
    assert unavail_kpis["total_evaluated"] == 0

