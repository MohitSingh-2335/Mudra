# tests/test_model_version_governance.py
"""Unit tests for model governance, cryptographic verification, and reconciliation provenance."""

import pytest
from pathlib import Path
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

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
    load_verified_model,
    load_production_models,
)
from src.database.models import Base, MarketTick, Reconciliation
from src.database.queries import get_reconciliation_history, get_reconciliation_kpis
from scripts.migrate_db import BASELINE_MODEL_SPEC, CANDIDATE_MODEL_SPEC
from app import (
    filter_reconciliation_records,
    prepare_reconciliation_table_display,
    resolve_tab3_reconciliation_kpis,
)


def test_canonical_configuration_constants():
    """Verify config.py constants and ORM schema contracts."""
    assert ACTIVE_MODEL_VERSION == "xgb_onnx_v1.0.0"
    assert FEATURE_SCHEMA_VERSION == "v1_39features"
    assert BASELINE_MODEL_SPEC["model_version"] == ACTIVE_MODEL_VERSION
    assert BASELINE_MODEL_SPEC["regressor_sha256"] == REGRESSOR_MODEL_SHA256

    assert MarketTick.model_version.nullable is False
    assert MarketTick.feature_schema_version.nullable is False

    # Artifact hashes on disk
    assert compute_file_sha256(REGRESSOR_MODEL_PATH) == REGRESSOR_MODEL_SHA256
    assert compute_file_sha256(CLASSIFIER_MODEL_PATH) == CLASSIFIER_MODEL_SHA256


def test_artifact_hash_verification_success_and_tampering(tmp_path):
    """Verify authentic models load and corrupted ones fail closed."""
    reg, clf = load_production_models()
    assert reg is not None and hasattr(reg, "predict")
    assert clf is not None and hasattr(clf, "predict_proba")

    # Tampered hash fails closed
    with pytest.raises(RuntimeError, match="Cryptographic integrity verification failed"):
        load_verified_model(REGRESSOR_MODEL_PATH, "0" * 64)

    # Missing file fails closed
    with pytest.raises(FileNotFoundError):
        load_verified_model("nonexistent.pkl", "0" * 64)


def test_sha256_digest_format():
    """Verify digest validator checks 64-char lowercase hex."""
    validate_sha256_digest("a" * 64)
    with pytest.raises(ValueError):
        validate_sha256_digest("short")
    with pytest.raises(ValueError):
        validate_sha256_digest("g" * 64)  # non-hex character


@pytest.fixture
def rec_db():
    """In-memory SQLite DB with ticks and reconciliations."""
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
    """Verify database queries filter by model_version and return valid KPIs."""
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield rec_db

    monkeypatch.setattr("src.database.queries.get_db", mock_get_db)

    history_all = get_reconciliation_history(limit=50, model_version=None)
    assert len(history_all) == 1

    history_active = get_reconciliation_history(limit=50, model_version=ACTIVE_MODEL_VERSION)
    assert len(history_active) == 1

    history_candidate = get_reconciliation_history(limit=50, model_version="candidate_v2")
    assert len(history_candidate) == 0

    kpis = get_reconciliation_kpis(model_version=ACTIVE_MODEL_VERSION)
    assert kpis["total_reconciled"] == 1
    assert kpis["directional_accuracy_pct"] == 100.0


def test_dashboard_tab3_helpers(rec_db, monkeypatch):
    """Verify Tab 3 display preparation and fallback resolution."""
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield rec_db

    monkeypatch.setattr("src.database.queries.get_db", mock_get_db)

    records = get_reconciliation_history(limit=50, model_version=None)
    filtered = filter_reconciliation_records(records, selected_model=ACTIVE_MODEL_VERSION)
    assert len(filtered) == 1

    df_display = prepare_reconciliation_table_display(filtered)
    assert not df_display.empty
    assert "Model Version" in df_display.columns
