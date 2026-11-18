import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

import config
from config import ACTIVE_MODEL_VERSION, FEATURE_SCHEMA_VERSION
from src.database.models import Base, MarketTick, ModelRegistry, PredictionInvestigation
from src.database.connection import init_db, CANONICAL_TABLE_COLUMNS
from src.database.queries import (
    record_prediction_investigation,
    get_prediction_investigations,
    update_prediction_investigation_status,
)
from src.engine.sync import reconcile_pending_ticks
from src.engine.paper_portfolio import execute_paper_order, get_portfolio_summary
from scripts.migrate_db import BASELINE_MODEL_SPEC, run_migration


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    init_db(engine)
    session = sessionmaker(bind=engine)()
    yield session, engine
    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_table_creation_and_inspection(in_memory_db):
    tables = inspect(in_memory_db[1]).get_table_names()
    for expected_table in CANONICAL_TABLE_COLUMNS.keys():
        assert expected_table in tables


def test_market_tick_lifecycle_and_dict(in_memory_db):
    session = in_memory_db[0]
    tick = MarketTick(
        timestamp=datetime.now(timezone.utc), open=50000.0, high=51000.0, low=49500.0, close=50500.0, volume=1200.5,
        model_version="xgb_v1", feature_schema_version="v1_39features",
        predicted_return_1h=0.015, predicted_price_1h=51257.5, model_confidence=0.72,
        features={"rsi": 55.4},
    )
    session.add(tick)
    session.commit()

    saved = session.query(MarketTick).filter_by(id=tick.id).first()
    assert saved.close == 50500.0 and saved.to_dict()["close"] == 50500.0 and saved.features["rsi"] == 55.4


def test_reconciliation_math_and_accurate_tagging(in_memory_db):
    session = in_memory_db[0]
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    tick0 = MarketTick(timestamp=t0, open=60000.0, high=60500.0, low=59500.0, close=60000.0, volume=100.0, model_version="xgb_v1", feature_schema_version="v1_39features", predicted_return_1h=0.01, predicted_price_1h=60600.0)
    tick1 = MarketTick(timestamp=t0 + timedelta(hours=1), open=60000.0, high=60700.0, low=59900.0, close=60550.0, volume=120.0, model_version="xgb_v1", feature_schema_version="v1_39features")
    session.add_all([tick0, tick1])
    session.commit()

    reconciled = reconcile_pending_ticks(session, tick1)
    assert len(reconciled) == 1
    r = reconciled[0]
    assert r.direction_correct is True and r.error_attribution_tag == "ACCURATE"
    assert r.absolute_error_usd == pytest.approx(50.0)
    assert r.residual_error == pytest.approx(-0.083333, abs=1e-4)


def test_reconciliation_sideways_drift_tagging(in_memory_db):
    session = in_memory_db[0]
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    tick0 = MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60000.0, volume=100.0, predicted_return_1h=0.0008, predicted_price_1h=60048.0, model_version="xgb_v1", feature_schema_version="v1_39features")
    tick1 = MarketTick(timestamp=t0 + timedelta(hours=1), open=60000.0, high=60050.0, low=59920.0, close=59960.0, volume=110.0, model_version="xgb_v1", feature_schema_version="v1_39features")
    session.add_all([tick0, tick1])
    session.commit()

    reconciled = reconcile_pending_ticks(session, tick1)
    assert len(reconciled) == 1 and reconciled[0].direction_correct is False and reconciled[0].error_attribution_tag == "SIDEWAYS_DRIFT"


def test_reconciliation_black_swan_outlier(in_memory_db):
    session = in_memory_db[0]
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    tick0 = MarketTick(timestamp=t0, open=60000.0, high=60500.0, low=59500.0, close=60000.0, volume=100.0, predicted_return_1h=0.002, predicted_price_1h=60120.0, model_version="xgb_v1", feature_schema_version="v1_39features")
    tick1 = MarketTick(timestamp=t0 + timedelta(hours=1), open=60000.0, high=63500.0, low=59800.0, close=63000.0, volume=500.0, model_version="xgb_v1", feature_schema_version="v1_39features")
    session.add_all([tick0, tick1])
    session.commit()

    reconciled = reconcile_pending_ticks(session, tick1)
    assert len(reconciled) == 1 and reconciled[0].is_black_swan is True and reconciled[0].error_attribution_tag == "BLACK_SWAN_OUTLIER"


def test_reconciliation_one_hour_horizon_alignment(in_memory_db):
    session = in_memory_db[0]
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tick0 = MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60000.0, volume=10.0, predicted_price_1h=60200.0, predicted_return_1h=0.0033)
    tick3 = MarketTick(timestamp=t0 + timedelta(hours=3), open=60500.0, high=60600.0, low=60400.0, close=60500.0, volume=10.0)
    session.add_all([tick0, tick3])
    session.commit()

    assert len(reconcile_pending_ticks(session, tick3)) == 0
    session.add(MarketTick(timestamp=t0 + timedelta(hours=1), open=60100.0, high=60250.0, low=60000.0, close=60200.0, volume=10.0))
    session.commit()

    reconciled_after = reconcile_pending_ticks(session, tick3)
    assert len(reconciled_after) == 1 and reconciled_after[0].actual_price_at_t_plus_1 == pytest.approx(60200.0)


def test_paper_portfolio_full_lifecycle(in_memory_db):
    session = in_memory_db[0]
    b_order = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10)
    assert b_order.action == "BUY" and b_order.btc_amount > 0.0

    s_order = execute_paper_order(session, "SELL", 63000.0)
    assert s_order.action == "SELL" and s_order.btc_balance == pytest.approx(0.0, abs=1e-8) and s_order.realized_pnl_usd > 0.0

    summary = get_portfolio_summary(session, current_price=63000.0)
    assert summary["total_trades"] == 2 and summary["realized_pnl_usd"] == pytest.approx(s_order.realized_pnl_usd, abs=1e-2)


def test_migrate_db_defaults_to_dry_run(in_memory_db):
    plan = run_migration(target_engine=in_memory_db[1], execute=False)
    assert plan["mode"] == "DRY_RUN" and plan["writes_executed"] is False and "market_ticks" in plan["existing_tables"]


def test_migrate_db_seed_hashes():
    expected_reg_sha = "b44fba18cf2c7b51e3297e83a946d869392803a267e951a5e85ea052c63eb854"
    expected_clf_sha = "df139c1018ed5025b1444d6597c3d98b4d7c377a898487268d18e58d253ae684"

    assert BASELINE_MODEL_SPEC["regressor_sha256"] == expected_reg_sha and BASELINE_MODEL_SPEC["classifier_sha256"] == expected_clf_sha
    reg_path, clf_path = Path(BASELINE_MODEL_SPEC["regressor_artifact_path"]), Path(BASELINE_MODEL_SPEC["classifier_artifact_path"])
    if reg_path.exists():
        assert hashlib.sha256(reg_path.read_bytes()).hexdigest() == expected_reg_sha
    if clf_path.exists():
        assert hashlib.sha256(clf_path.read_bytes()).hexdigest() == expected_clf_sha


def test_prediction_investigations_schema_columns_and_constraints():
    table = PredictionInvestigation.__table__
    expected_cols = {
        "id", "tick_id", "prediction_id", "model_version", "headline",
        "suspected_cause_category", "hypothesis_notes", "diagnostic_confidence",
        "verification_status", "created_at", "updated_at",
    }
    assert expected_cols.issubset(set(table.columns.keys()))


def test_headline_search_traces_to_market_tick_and_model_version(in_memory_db):
    session = in_memory_db[0]
    active_reg = ModelRegistry(
        model_version=ACTIVE_MODEL_VERSION,
        regressor_artifact_path=config.REGRESSOR_MODEL_PATH, regressor_sha256=config.REGRESSOR_MODEL_SHA256,
        classifier_artifact_path=config.CLASSIFIER_MODEL_PATH, classifier_sha256=config.CLASSIFIER_MODEL_SHA256,
        feature_schema_version=FEATURE_SCHEMA_VERSION, target_definition="1h_ahead_return_and_movement",
        current_status="ACTIVE", lifecycle_seq=1,
    )
    session.add(active_reg)
    tick = MarketTick(
        id=101, timestamp=datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc),
        open=65000.0, high=65500.0, low=64800.0, close=65200.0, volume=1250.0,
        model_version=ACTIVE_MODEL_VERSION, feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    session.add(tick)
    session.commit()

    inv = record_prediction_investigation(
        session=session, tick_id=101, model_version=ACTIVE_MODEL_VERSION,
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


def test_distinguish_suspected_cause_from_confirmed_ground_truth(in_memory_db):
    session = in_memory_db[0]
    tick = MarketTick(
        id=102, timestamp=datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc),
        open=65000.0, high=65500.0, low=64800.0, close=65200.0, volume=1250.0,
        model_version=ACTIVE_MODEL_VERSION, feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    session.add(tick)
    session.commit()

    inv = record_prediction_investigation(
        session=session, tick_id=102, model_version=ACTIVE_MODEL_VERSION,
        headline="Major Exchange halts withdrawals", suspected_cause_category="EXCHANGE_OUTAGE",
        hypothesis_notes="Withdrawal halt rumors sparked panic.",
    )
    assert inv["verification_status"] == "UNVERIFIED_HYPOTHESIS"
    updated = update_prediction_investigation_status(
        session=session, investigation_id=inv["id"], new_status="CONFIRMED_CAUSE",
        verified_by="Lead Quantitative Auditor", ground_truth_label="EXCHANGE_LIQUIDITY_HALT",
    )
    assert updated["verification_status"] == "CONFIRMED_CAUSE"
    assert updated["verified_by"] == "Lead Quantitative Auditor"


def test_diagnostic_confidence_validation(in_memory_db):
    session = in_memory_db[0]
    tick = MarketTick(
        id=103, timestamp=datetime(2026, 10, 15, 16, 0, tzinfo=timezone.utc),
        open=65000.0, high=65500.0, low=64800.0, close=65200.0, volume=1250.0,
        model_version=ACTIVE_MODEL_VERSION, feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    session.add(tick)
    session.commit()

    for bad in [-0.1, 1.5, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            record_prediction_investigation(
                session=session, tick_id=103, model_version=ACTIVE_MODEL_VERSION,
                headline="Test headline", suspected_cause_category="OTHER",
                hypothesis_notes="Testing bounds", diagnostic_confidence=bad,
            )
