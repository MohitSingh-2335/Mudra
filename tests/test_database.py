import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

import config
from config import ACTIVE_MODEL_VERSION, FEATURE_SCHEMA_VERSION
from src.database import (
    CANONICAL_TABLE_COLUMNS, Base, HourlyPostmortem, MarketTick, ModelPrediction, ModelRegistry, init_db,
)
from src.engine import execute_paper_order, get_portfolio_summary, reconcile_pending_ticks
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
        model_version="xgb_v1",
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
    tick0 = MarketTick(timestamp=t0, open=60000.0, high=60500.0, low=59500.0, close=60000.0, volume=100.0, model_version="xgb_v1", predicted_return_1h=0.01, predicted_price_1h=60600.0)
    tick1 = MarketTick(timestamp=t0 + timedelta(hours=1), open=60000.0, high=60700.0, low=59900.0, close=60550.0, volume=120.0, model_version="xgb_v1")
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
    tick0 = MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60000.0, volume=100.0, predicted_return_1h=0.0008, predicted_price_1h=60048.0, model_version="xgb_v1")
    tick1 = MarketTick(timestamp=t0 + timedelta(hours=1), open=60000.0, high=60050.0, low=59920.0, close=59960.0, volume=110.0, model_version="xgb_v1")
    session.add_all([tick0, tick1])
    session.commit()

    reconciled = reconcile_pending_ticks(session, tick1)
    assert len(reconciled) == 1 and reconciled[0].direction_correct is False and reconciled[0].error_attribution_tag == "SIDEWAYS_DRIFT"


def test_reconciliation_black_swan_outlier(in_memory_db):
    session = in_memory_db[0]
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    tick0 = MarketTick(timestamp=t0, open=60000.0, high=60500.0, low=59500.0, close=60000.0, volume=100.0, predicted_return_1h=0.002, predicted_price_1h=60120.0, model_version="xgb_v1")
    tick1 = MarketTick(timestamp=t0 + timedelta(hours=1), open=60000.0, high=63500.0, low=59800.0, close=63000.0, volume=500.0, model_version="xgb_v1")
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


def test_direction_and_return_independence(in_memory_db):
    # verifies classifier direction and regressor return can diverge freely
    session = in_memory_db[0]
    tick = MarketTick(
        timestamp=datetime.now(timezone.utc), open=65000.0, high=65500.0, low=64800.0, close=65100.0,
        volume=100.0, model_version="xgb_candidate_v2.0.0",
    )
    session.add(tick)
    session.flush()

    pred = ModelPrediction(
        tick_id=tick.id,
        model_version="xgb_candidate_v2.0.0",
        run_mode="SHADOW",
        is_primary=False,
        feature_schema_version="v2_36stationary",
        predicted_return_pct=-0.045,
        predicted_price_1h=65070.0,
        direction_predicted="UP",
        direction_prob_up=0.523,
        clears_hurdle=False,
        diagnosis_summary="Classifier bullish but expected move sub-hurdle.",
    )
    session.add(pred)
    session.commit()

    saved = session.query(ModelPrediction).filter_by(tick_id=tick.id).first()
    assert saved is not None
    assert saved.direction_predicted == "UP"
    assert saved.predicted_return_pct == -0.045
    assert saved.diagnosis_summary == "Classifier bullish but expected move sub-hurdle."


def test_hourly_postmortem_lifecycle(in_memory_db):
    # verifies hourly postmortem table persistence and retrieval
    session = in_memory_db[0]
    tick = MarketTick(
        timestamp=datetime.now(timezone.utc), open=65000.0, high=65500.0, low=64800.0, close=65100.0,
        volume=100.0, model_version="xgb_onnx_v1.0.0",
    )
    session.add(tick)
    session.flush()

    pm = HourlyPostmortem(
        tick_id=tick.id,
        news_headlines=["Bitcoin holds support amid calm derivatives funding."],
        feature_snapshot={"close": 65100.0, "volatility_range_pct": 1.05},
        gemini_full_analysis={"market_summary": "Sideways chop with neutral funding."},
        gemini_status="OK",
        error_detail=None,
        created_at=datetime.now(timezone.utc),
    )
    session.add(pm)
    session.commit()

    retrieved = session.query(HourlyPostmortem).filter_by(tick_id=tick.id).first()
    assert retrieved is not None
    assert retrieved.gemini_status == "OK"
    assert len(retrieved.news_headlines) == 1
    assert retrieved.feature_snapshot["close"] == 65100.0



