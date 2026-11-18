import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import Engine, inspect, text

from config import (
    ACTIVE_MODEL_VERSION, CLASSIFIER_MODEL_SHA256, FEATURE_SCHEMA_VERSION,
    MIN_HURDLE_PCT, PORTFOLIO_ADVISORY_LOCK_ID, REGRESSOR_MODEL_SHA256,
)
from src.database.connection import (
    CANONICAL_FUNCTIONS, CANONICAL_TABLE_COLUMNS, CANONICAL_TRIGGERS,
    SchemaDriftError, _TICK_ID_COLUMN_CACHE, _classify_column_type,
    create_db_engine, engine as default_engine, ensure_canonical_function,
    ensure_canonical_trigger, verify_db_schema,
)
from src.database.models import Base

MIGRATION_ADVISORY_LOCK_ID = 888002

BASELINE_MODEL_SPEC: Dict[str, Any] = {
    "model_version": ACTIVE_MODEL_VERSION,
    "regressor_artifact_path": "artifacts/models/best_regressor_model.pkl",
    "regressor_sha256": REGRESSOR_MODEL_SHA256,
    "classifier_artifact_path": "artifacts/models/best_classifier_model.pkl",
    "classifier_sha256": CLASSIFIER_MODEL_SHA256,
    "feature_schema_version": FEATURE_SCHEMA_VERSION,
    "target_definition": "next_1h_close_return_pct_and_binary_direction",
    "training_dataset_path": "data/processed/featured_btc_data.csv",
    "training_dataset_sha256": "fdff3cb57bbad1ed3fd4f3bc44cc26a34efba3d72a62b5a25b78b73e4c36cf68",
    "training_row_count": 27992,
    "split_metadata": {"train_rows": 27992, "test_rows": 6999, "note": "Historical baseline production model."},
    "evaluation_metrics": {
        "reg_price_rmse_usd": 399.4728, "naive_persistence_price_rmse_usd": 398.7661,
        "rmse_outperformance_usd": -0.7067, "reg_rmse": 0.00487042, "reg_mae": 0.00322086,
        "directional_hit_rate": 0.505001, "clf_accuracy_pct": 53.2581, "clf_macro_f1": 0.529665,
        "clf_log_loss": 0.691365, "benchmark_tag": "REUSED HISTORICAL BENCHMARK",
    },
    "code_revision_sha": "e46a586",
    "current_status": "ACTIVE",
}

CANDIDATE_MODEL_SPEC: Dict[str, Any] = {
    "model_version": "p1_controlled_candidate_v1",
    "regressor_artifact_path": "artifacts/experiments/p1_controlled_retrain/candidate_regressor.pkl",
    "regressor_sha256": "5cb9e6f263a97a0d807f7a1a04a6ce48f5e801317cff56d5cb6f644703440011",
    "classifier_artifact_path": "artifacts/experiments/p1_controlled_retrain/candidate_classifier.pkl",
    "classifier_sha256": "80139690e79804a29a9b1cd11e83999530c1b5c013023a740bb975873625eb7f",
    "feature_schema_version": "v1_39features",
    "target_definition": "next_1h_close_return_pct_and_binary_direction",
    "training_dataset_path": "data/processed/featured_btc_data.csv",
    "training_dataset_sha256": "0b819589346b03cbb1a830db1e10235f79179da89c030a6d55d60b09585e2648",
    "training_row_count": 27991,
    "split_metadata": {"clean_rows": 34990, "train_rows": 27991, "purge_rows": 1, "test_rows": 6998},
    "evaluation_metrics": {
        "reg_price_rmse_usd": 399.3687, "naive_persistence_price_rmse_usd": 398.7661,
        "rmse_outperformance_usd": -0.6027, "reg_rmse": 0.00486872, "reg_mae": 0.00321975,
        "directional_hit_rate": 0.503572, "clf_accuracy_pct": 53.0294, "clf_macro_f1": 0.528027,
        "clf_log_loss": 0.691404, "benchmark_tag": "REUSED HISTORICAL BENCHMARK",
    },
    "code_revision_sha": "99a27e297d3f54c91fdc2d170a05de536ec97fed",
}


def _verify_new_tables_subset(conn: Any) -> None:
    inspector = inspect(conn)
    existing_tables = set(inspector.get_table_names())
    for tbl_name in ("model_registry", "model_lifecycle_events", "model_predictions", "prediction_investigations"):
        if tbl_name not in existing_tables:
            raise SchemaDriftError(f"Table '{tbl_name}' missing.")
        live_cols = {c["name"]: c for c in inspector.get_columns(tbl_name)}
        for col_name, exp_type, exp_nullable in CANONICAL_TABLE_COLUMNS[tbl_name]:
            if col_name not in live_cols:
                raise SchemaDriftError(f"Column '{tbl_name}.{col_name}' missing.")
            c_info = live_cols[col_name]
            if _classify_column_type(str(c_info["type"])) != exp_type or bool(c_info.get("nullable", True)) != exp_nullable:
                raise SchemaDriftError(f"Column drift on '{tbl_name}.{col_name}'.")


def run_tx1a(target_engine: Engine, bootstrap_base_tables: bool = True) -> Dict[str, Any]:
    with target_engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            conn.execute(text(f"SELECT pg_advisory_xact_lock({MIGRATION_ADVISORY_LOCK_ID});"))
        if bootstrap_base_tables:
            Base.metadata.create_all(bind=conn)
        _verify_new_tables_subset(conn)
        created_funcs = sum(1 for f in CANONICAL_FUNCTIONS if ensure_canonical_function(conn, f))
        created_trigs = sum(1 for t in CANONICAL_TRIGGERS if ensure_canonical_trigger(conn, t))
    return {"tx": "1A", "functions_created": created_funcs, "triggers_created": created_trigs}


def run_tx1b(target_engine: Engine) -> Dict[str, Any]:
    with target_engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            conn.execute(text(f"SELECT pg_advisory_xact_lock({MIGRATION_ADVISORY_LOCK_ID});"))
        seeded: List[str] = []

        if not conn.execute(text("SELECT model_version FROM public.model_registry WHERE model_version = :mv"), {"mv": BASELINE_MODEL_SPEC["model_version"]}).first():
            conn.execute(
                text("""
                    INSERT INTO public.model_registry (
                        model_version, regressor_artifact_path, regressor_sha256,
                        classifier_artifact_path, classifier_sha256, feature_schema_version,
                        target_definition, training_dataset_path, training_dataset_sha256,
                        training_row_count, split_metadata, evaluation_metrics,
                        code_revision_sha, current_status
                    ) VALUES (
                        :model_version, :regressor_artifact_path, :regressor_sha256,
                        :classifier_artifact_path, :classifier_sha256, :feature_schema_version,
                        :target_definition, :training_dataset_path, :training_dataset_sha256,
                        :training_row_count, CAST(:split_metadata AS json), CAST(:evaluation_metrics AS json),
                        :code_revision_sha, :current_status
                    )
                """),
                {**BASELINE_MODEL_SPEC, "split_metadata": json.dumps(BASELINE_MODEL_SPEC["split_metadata"]), "evaluation_metrics": json.dumps(BASELINE_MODEL_SPEC["evaluation_metrics"])},
            )
            conn.execute(
                text("""
                    INSERT INTO public.model_lifecycle_events (
                        model_version, lifecycle_seq, event_type, from_status, to_status,
                        actor, reason, evidence_json
                    ) VALUES (
                        :mv, 1, 'REGISTERED', NULL, 'ACTIVE',
                        'scripts/migrate_db.py', 'Historical production baseline model registered.',
                        CAST(:ev_json AS json)
                    )
                """),
                {"mv": BASELINE_MODEL_SPEC["model_version"], "ev_json": json.dumps(BASELINE_MODEL_SPEC["evaluation_metrics"])},
            )
            seeded.append(BASELINE_MODEL_SPEC["model_version"])

        if not conn.execute(text("SELECT model_version FROM public.model_registry WHERE model_version = :mv"), {"mv": CANDIDATE_MODEL_SPEC["model_version"]}).first():
            conn.execute(
                text("""
                    INSERT INTO public.model_registry (
                        model_version, regressor_artifact_path, regressor_sha256,
                        classifier_artifact_path, classifier_sha256, feature_schema_version,
                        target_definition, training_dataset_path, training_dataset_sha256,
                        training_row_count, split_metadata, evaluation_metrics,
                        code_revision_sha, current_status
                    ) VALUES (
                        :model_version, :regressor_artifact_path, :regressor_sha256,
                        :classifier_artifact_path, :classifier_sha256, :feature_schema_version,
                        :target_definition, :training_dataset_path, :training_dataset_sha256,
                        :training_row_count, CAST(:split_metadata AS json), CAST(:eval_json AS json),
                        :code_revision_sha, 'REJECTED'
                    )
                """),
                {**CANDIDATE_MODEL_SPEC, "split_metadata": json.dumps(CANDIDATE_MODEL_SPEC["split_metadata"]), "eval_json": json.dumps(CANDIDATE_MODEL_SPEC["evaluation_metrics"])},
            )
            seeded.append(CANDIDATE_MODEL_SPEC["model_version"])

    return {"tx": "1B", "seeded_models": seeded}


def run_tx1c(target_engine: Engine) -> Dict[str, Any]:
    with target_engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            conn.execute(text(f"SELECT pg_advisory_xact_lock({MIGRATION_ADVISORY_LOCK_ID});"))
        ins_res = conn.execute(
            text("""
                INSERT INTO public.model_predictions (
                    tick_id, model_version, run_mode, is_canonical, is_primary,
                    parent_prediction_id, feature_schema_version, predicted_return_pct,
                    predicted_price_1h, direction_predicted, direction_prob_up, clears_hurdle, created_at
                )
                SELECT
                    mt.id AS tick_id, mt.model_version, 'PRIMARY', TRUE, TRUE, NULL,
                    mt.feature_schema_version,
                    ((mt.predicted_price_1h - mt.close) / mt.close) * 100.0,
                    mt.predicted_price_1h,
                    CASE WHEN ((mt.predicted_price_1h - mt.close) / mt.close) >= 0.0 THEN 'UP' ELSE 'DOWN' END,
                    COALESCE(mt.model_confidence, 0.5),
                    (abs(((mt.predicted_price_1h - mt.close) / mt.close) * 100.0) >= :hurdle_pct),
                    mt.timestamp
                FROM public.market_ticks mt
                WHERE mt.predicted_price_1h IS NOT NULL AND mt.close >= 1.0 AND mt.predicted_price_1h >= 1.0
                ON CONFLICT (tick_id, model_version) WHERE (is_canonical IS TRUE) DO NOTHING
            """),
            {"hurdle_pct": float(MIN_HURDLE_PCT)},
        )
        inserted = ins_res.rowcount if ins_res.rowcount is not None and ins_res.rowcount >= 0 else 0
    return {"tx": "1C", "inserted_predictions": inserted}


def run_tx2(target_engine: Engine) -> Dict[str, Any]:
    with target_engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            conn.execute(text(f"SELECT pg_advisory_xact_lock({MIGRATION_ADVISORY_LOCK_ID});"))
            conn.execute(text(f"SELECT pg_advisory_xact_lock({PORTFOLIO_ADVISORY_LOCK_ID});"))
            conn.execute(text("ALTER TABLE public.paper_portfolio ADD COLUMN IF NOT EXISTS tick_id INTEGER;"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_portfolio_tick_id ON public.paper_portfolio (tick_id) WHERE (tick_id IS NOT NULL);"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_market_ticks_timestamp ON public.market_ticks (timestamp);"))
    _TICK_ID_COLUMN_CACHE.clear()
    return {"tx": "2", "status": "completed"}


def inspect_dry_run_plan(target_engine: Engine) -> Dict[str, Any]:
    with target_engine.connect() as conn:
        inspector = inspect(conn)
        existing = set(inspector.get_table_names())
        missing = [t for t in CANONICAL_TABLE_COLUMNS if t not in existing]
        has_pp_tick_id = "paper_portfolio" in existing and "tick_id" in {c["name"] for c in inspector.get_columns("paper_portfolio")}
        tick_count = int(conn.execute(text("SELECT COUNT(*) FROM market_ticks WHERE predicted_price_1h IS NOT NULL")).scalar() or 0) if "market_ticks" in existing else 0
        rec_count = int(conn.execute(text("SELECT COUNT(*) FROM reconciliations")).scalar() or 0) if "reconciliations" in existing else 0
        return {
            "mode": "DRY_RUN", "writes_executed": False, "existing_tables": sorted(existing),
            "missing_tables": missing, "paper_portfolio_has_tick_id": has_pp_tick_id,
            "eligible_market_ticks_for_backfill": tick_count, "eligible_reconciliations_for_backfill": rec_count,
            "duplicate_market_tick_timestamps": 0,
        }


def run_migration(target_engine: Optional[Engine] = None, execute: bool = False, bootstrap_base_tables: bool = True) -> Dict[str, Any]:
    eng = target_engine or default_engine
    if not execute:
        return inspect_dry_run_plan(eng)
    res_1a = run_tx1a(eng, bootstrap_base_tables=bootstrap_base_tables)
    res_1b = run_tx1b(eng)
    res_1c = run_tx1c(eng)
    res_2 = run_tx2(eng)
    verification = verify_db_schema(eng, require_seed_models=True)
    return {"mode": "EXECUTE", "writes_executed": True, "tx1a": res_1a, "tx1b": res_1b, "tx1c": res_1c, "tx2": res_2, "verification": verification}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mudra Database Migration Runner")
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--db-url", type=str, default=None)
    args = parser.parse_args(argv)
    target_eng = create_db_engine(args.db_url) if args.db_url else default_engine
    try:
        result = run_migration(target_engine=target_eng, execute=bool(args.execute))
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Migration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
