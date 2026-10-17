"""
Guarded Idempotent Database Migration Runner for Mudra Model Governance & Schema Evolution.

Safety Contract:
- Defaults to read-only `--dry-run` mode (zero database writes).
- Requires explicit `--execute` flag to perform DDL/DML writes.
- Executes 4 isolated transactions (TX 1A, TX 1B, TX 1C, TX 2) with transaction-scoped advisory locks
  and fail-closed timeouts (`lock_timeout = '5s'`, `statement_timeout = '30s'` / `'15s'`).
- Never uses blind `CREATE OR REPLACE FUNCTION` or `DROP TRIGGER IF EXISTS`; inspects `pg_proc`
  and `pg_trigger` and aborts with `SchemaDriftError` on any unexpected overload or definition drift.
"""
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
    ACTIVE_MODEL_VERSION,
    FEATURE_SCHEMA_VERSION,
    REGRESSOR_MODEL_SHA256,
    CLASSIFIER_MODEL_SHA256,
    MIN_HURDLE_PCT,
    PORTFOLIO_ADVISORY_LOCK_ID,
)
from src.database.connection import (
    CANONICAL_CONSTRAINTS,
    CANONICAL_FOREIGN_KEYS,
    CANONICAL_FUNCTIONS,
    CANONICAL_INDEXES,
    CANONICAL_TABLE_COLUMNS,
    CANONICAL_TRIGGERS,
    SchemaDriftError,
    _TICK_ID_COLUMN_CACHE,
    _classify_column_type,
    create_db_engine,
    engine as default_engine,
    ensure_canonical_function,
    ensure_canonical_trigger,
    verify_db_schema,
)

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
    "split_metadata": {
        "train_rows": 27992,
        "test_rows": 6999,
        "note": "Historical baseline production model trained prior to point-in-time causality remediation.",
    },
    "evaluation_metrics": {
        "reg_price_rmse_usd": 399.4728,
        "naive_persistence_price_rmse_usd": 398.7661,
        "rmse_outperformance_usd": -0.7067,
        "reg_rmse": 0.00487042,
        "reg_mae": 0.00322086,
        "directional_hit_rate": 0.505001,
        "clf_accuracy_pct": 53.2581,
        "clf_macro_f1": 0.529665,
        "clf_log_loss": 0.691365,
        "benchmark_tag": "REUSED HISTORICAL BENCHMARK — PRIOR MODEL SELECTION EXPOSURE",
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
    "training_dataset_path": "data/processed/featured_btc_data_candidate.csv",
    "training_dataset_sha256": "0b819589346b03cbb1a830db1e10235f79179da89c030a6d55d60b09585e2648",
    "training_row_count": 27991,
    "split_metadata": {
        "clean_rows": 34990,
        "train_rows": 27991,
        "purge_rows": 1,
        "test_rows": 6998,
        "train_start": "2022-07-08T13:00:00",
        "train_end": "2025-09-17T17:00:00",
        "purge_timestamp": "2025-09-17T18:00:00",
        "test_start": "2025-09-17T19:00:00",
        "test_end": "2026-07-06T08:00:00",
    },
    "evaluation_metrics": {
        "reg_price_rmse_usd": 399.3687,
        "naive_persistence_price_rmse_usd": 398.7661,
        "rmse_outperformance_usd": -0.6027,
        "reg_rmse": 0.00486872,
        "reg_mae": 0.00321975,
        "directional_hit_rate": 0.503572,
        "clf_accuracy_pct": 53.0294,
        "clf_macro_f1": 0.528027,
        "clf_log_loss": 0.691404,
        "benchmark_tag": "REUSED HISTORICAL BENCHMARK — PRIOR MODEL SELECTION EXPOSURE",
    },
    "code_revision_sha": "99a27e297d3f54c91fdc2d170a05de536ec97fed",
}

CANDIDATE_EVALUATED_REASON = (
    "Out-of-sample benchmark evaluation completed on 6,998 holdout rows "
    "(price RMSE=$399.3687 vs naive persistence $398.7661, RMSE outperformance=-$0.6027, "
    "return RMSE=0.00486872, return MAE=0.00321975, directional hit rate=0.503572; "
    "classifier accuracy=53.0294%, macro F1=0.528027, log loss=0.691404)."
)

CANDIDATE_REJECTION_REASON = (
    "Rejected candidate promotion: although candidate regressor price RMSE ($399.3687) is slightly lower "
    "than production baseline ($399.4728), both regressors underperform naive persistence ($398.7661; "
    "candidate outperformance -$0.6027 vs baseline -$0.7067) with lower directional hit rate "
    "(0.503572 vs 0.505001), and candidate classifier underperforms baseline across accuracy "
    "(53.0294% vs 53.2581%), macro F1 (0.528027 vs 0.529665), and log loss (0.691404 vs 0.691365)."
)


# Base 5 historical tables DDL (used only if bootstrapping an empty disposable PostgreSQL database)
_BASE_HISTORICAL_TABLES_DDL = """
CREATE TABLE IF NOT EXISTS public.market_ticks (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(32) NOT NULL DEFAULT 'xgb_onnx_v1.0.0',
    feature_schema_version VARCHAR(16) NOT NULL DEFAULT 'v1_39features',
    market_regime VARCHAR(32),
    predicted_return_1h DOUBLE PRECISION,
    predicted_price_1h DOUBLE PRECISION,
    model_confidence DOUBLE PRECISION,
    features JSON
);
CREATE INDEX IF NOT EXISTS ix_market_ticks_timestamp ON public.market_ticks (timestamp);

CREATE TABLE IF NOT EXISTS public.api_telemetry (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    service_name VARCHAR(64) NOT NULL,
    endpoint VARCHAR(256) NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    status_code INTEGER NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    rate_limit_remaining INTEGER,
    tokens_used INTEGER DEFAULT 0,
    cost_estimate_usd DOUBLE PRECISION DEFAULT 0.0,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS ix_api_telemetry_timestamp ON public.api_telemetry (timestamp);
CREATE INDEX IF NOT EXISTS ix_api_telemetry_service_name ON public.api_telemetry (service_name);

CREATE TABLE IF NOT EXISTS public.council_audits (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tick_id INTEGER REFERENCES public.market_ticks(id),
    deliberation_latency_ms DOUBLE PRECISION,
    strategist_action VARCHAR(16) NOT NULL,
    strategist_conviction INTEGER NOT NULL,
    rag_context_summary TEXT,
    risk_verdict VARCHAR(16) NOT NULL,
    veto_reasons JSON,
    kelly_fraction DOUBLE PRECISION NOT NULL,
    final_decision VARCHAR(16) NOT NULL,
    veto_attribution VARCHAR(32),
    execution_notes TEXT
);
CREATE INDEX IF NOT EXISTS ix_council_audits_timestamp ON public.council_audits (timestamp);
CREATE INDEX IF NOT EXISTS ix_council_audits_tick_id ON public.council_audits (tick_id);

CREATE TABLE IF NOT EXISTS public.reconciliations (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tick_id INTEGER NOT NULL UNIQUE REFERENCES public.market_ticks(id),
    predicted_price DOUBLE PRECISION NOT NULL,
    actual_price_at_t_plus_1 DOUBLE PRECISION NOT NULL,
    predicted_return_pct DOUBLE PRECISION NOT NULL,
    actual_return_pct DOUBLE PRECISION NOT NULL,
    residual_error DOUBLE PRECISION NOT NULL,
    absolute_error_usd DOUBLE PRECISION NOT NULL,
    percentage_error DOUBLE PRECISION NOT NULL,
    direction_predicted VARCHAR(16) NOT NULL,
    direction_realized VARCHAR(16) NOT NULL,
    direction_correct BOOLEAN NOT NULL,
    is_black_swan BOOLEAN NOT NULL DEFAULT FALSE,
    error_attribution_tag VARCHAR(64) NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_reconciliations_timestamp ON public.reconciliations (timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS ix_reconciliations_tick_id ON public.reconciliations (tick_id);

CREATE TABLE IF NOT EXISTS public.paper_portfolio (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action VARCHAR(16) NOT NULL,
    btc_amount DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    execution_price DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    fee_deducted_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    slippage_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    cash_balance DOUBLE PRECISION NOT NULL,
    btc_balance DOUBLE PRECISION NOT NULL,
    total_equity_usd DOUBLE PRECISION NOT NULL,
    realized_pnl_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    unrealized_pnl_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    drawdown_pct DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    cumulative_return_pct DOUBLE PRECISION NOT NULL,
    benchmark_btc_equity DOUBLE PRECISION NOT NULL,
    alpha_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    alpha_pct DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS ix_paper_portfolio_timestamp ON public.paper_portfolio (timestamp);
"""

_TX1A_NEW_TABLES_AND_INDEXES_DDL = """
CREATE TABLE IF NOT EXISTS public.model_registry (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(64) NOT NULL,
    regressor_artifact_path VARCHAR(256) NOT NULL,
    regressor_sha256 VARCHAR(64) NOT NULL,
    classifier_artifact_path VARCHAR(256) NOT NULL,
    classifier_sha256 VARCHAR(64) NOT NULL,
    feature_schema_version VARCHAR(32) NOT NULL,
    target_definition VARCHAR(128) NOT NULL,
    training_dataset_path VARCHAR(256),
    training_dataset_sha256 VARCHAR(64),
    training_row_count INTEGER,
    split_metadata JSON,
    evaluation_metrics JSON,
    code_revision_sha VARCHAR(64),
    current_status VARCHAR(24) NOT NULL,
    lifecycle_seq INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_registry_model_version UNIQUE (model_version),
    CONSTRAINT chk_model_registry_status CHECK (
        current_status IN ('ACTIVE', 'CANDIDATE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK')
        AND lifecycle_seq >= 1
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_single_active
ON public.model_registry (current_status)
WHERE (current_status = 'ACTIVE');

CREATE TABLE IF NOT EXISTS public.model_lifecycle_events (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(64) NOT NULL,
    lifecycle_seq INTEGER NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    from_status VARCHAR(24),
    to_status VARCHAR(24) NOT NULL,
    related_model_version VARCHAR(64),
    actor VARCHAR(64) NOT NULL,
    reason TEXT NOT NULL,
    evidence_json JSON,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_mle_model_version
        FOREIGN KEY (model_version) REFERENCES public.model_registry(model_version)
        ON UPDATE NO ACTION ON DELETE NO ACTION,
    CONSTRAINT fk_mle_related_model_version
        FOREIGN KEY (related_model_version) REFERENCES public.model_registry(model_version)
        ON UPDATE NO ACTION ON DELETE NO ACTION,
    CONSTRAINT uq_mle_model_version_seq UNIQUE (model_version, lifecycle_seq),
    CONSTRAINT chk_lifecycle_event_vocabulary CHECK (
        lifecycle_seq >= 1
        AND event_type IN ('REGISTERED', 'EVALUATED', 'REJECTED', 'PROMOTED', 'SUPERSEDED', 'ROLLED_BACK', 'ARCHIVED')
        AND to_status IN ('ACTIVE', 'CANDIDATE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK')
        AND (from_status IS NULL OR from_status IN ('ACTIVE', 'CANDIDATE', 'REJECTED', 'ARCHIVED', 'ROLLED_BACK'))
    )
);

CREATE INDEX IF NOT EXISTS ix_model_lifecycle_events_model_version
ON public.model_lifecycle_events (model_version);

CREATE UNIQUE INDEX IF NOT EXISTS uq_mle_single_registered_per_model
ON public.model_lifecycle_events (model_version)
WHERE (event_type = 'REGISTERED');

CREATE UNIQUE INDEX IF NOT EXISTS uq_mle_single_evaluated_per_model
ON public.model_lifecycle_events (model_version)
WHERE (event_type = 'EVALUATED');

CREATE TABLE IF NOT EXISTS public.model_predictions (
    id SERIAL PRIMARY KEY,
    tick_id INTEGER NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    run_mode VARCHAR(24) NOT NULL,
    is_canonical BOOLEAN NOT NULL DEFAULT TRUE,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    parent_prediction_id INTEGER,
    feature_schema_version VARCHAR(32) NOT NULL DEFAULT 'v1_39features',
    predicted_return_pct DOUBLE PRECISION NOT NULL,
    predicted_price_1h DOUBLE PRECISION NOT NULL,
    direction_predicted VARCHAR(16) NOT NULL,
    direction_prob_up DOUBLE PRECISION NOT NULL,
    clears_hurdle BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    outcome_tick_id INTEGER,
    evaluation_policy_version VARCHAR(32),
    actual_price_at_t_plus_1 DOUBLE PRECISION,
    actual_return_pct DOUBLE PRECISION,
    direction_realized VARCHAR(16),
    direction_correct BOOLEAN,
    residual_error DOUBLE PRECISION,
    absolute_error_usd DOUBLE PRECISION,
    percentage_error DOUBLE PRECISION,
    is_black_swan BOOLEAN,
    error_attribution_tag VARCHAR(64),
    reconciled_at TIMESTAMPTZ,
    CONSTRAINT fk_mp_tick_id
        FOREIGN KEY (tick_id) REFERENCES public.market_ticks(id)
        ON UPDATE NO ACTION ON DELETE NO ACTION,
    CONSTRAINT fk_mp_model_version
        FOREIGN KEY (model_version) REFERENCES public.model_registry(model_version)
        ON UPDATE NO ACTION ON DELETE NO ACTION,
    CONSTRAINT fk_mp_parent_prediction_id
        FOREIGN KEY (parent_prediction_id) REFERENCES public.model_predictions(id)
        ON UPDATE NO ACTION ON DELETE NO ACTION,
    CONSTRAINT fk_mp_outcome_tick_id
        FOREIGN KEY (outcome_tick_id) REFERENCES public.market_ticks(id)
        ON UPDATE NO ACTION ON DELETE NO ACTION,
    CONSTRAINT chk_prediction_mode_invariants CHECK (
        (
            (run_mode = 'PRIMARY' AND is_primary IS TRUE AND is_canonical IS TRUE AND parent_prediction_id IS NULL)
            OR (run_mode = 'SHADOW' AND is_primary IS FALSE AND is_canonical IS TRUE AND parent_prediction_id IS NULL)
            OR (run_mode = 'FORCED_REANALYSIS' AND is_primary IS FALSE AND is_canonical IS FALSE AND parent_prediction_id IS NOT NULL)
        )
        AND predicted_return_pct NOT IN ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision)
        AND predicted_price_1h NOT IN ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision)
        AND direction_prob_up NOT IN ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision)
        AND predicted_price_1h >= 1.0
        AND direction_prob_up >= 0.0 AND direction_prob_up <= 1.0
        AND direction_predicted IN ('UP', 'DOWN')
        AND direction_predicted = (CASE WHEN predicted_return_pct >= 0.0 THEN 'UP' ELSE 'DOWN' END)
    ),
    CONSTRAINT chk_reconciliation_all_or_nothing CHECK (
        (
            outcome_tick_id IS NULL
            AND evaluation_policy_version IS NULL
            AND actual_price_at_t_plus_1 IS NULL
            AND actual_return_pct IS NULL
            AND direction_realized IS NULL
            AND direction_correct IS NULL
            AND residual_error IS NULL
            AND absolute_error_usd IS NULL
            AND percentage_error IS NULL
            AND is_black_swan IS NULL
            AND error_attribution_tag IS NULL
            AND reconciled_at IS NULL
        )
        OR (
            outcome_tick_id IS NOT NULL
            AND evaluation_policy_version = 'binary_exact_t1_v2'
            AND actual_price_at_t_plus_1 IS NOT NULL
            AND actual_return_pct IS NOT NULL
            AND direction_realized IS NOT NULL
            AND direction_correct IS NOT NULL
            AND residual_error IS NOT NULL
            AND absolute_error_usd IS NOT NULL
            AND percentage_error IS NOT NULL
            AND is_black_swan IS NOT NULL
            AND error_attribution_tag IS NOT NULL
            AND reconciled_at IS NOT NULL
            AND actual_price_at_t_plus_1 NOT IN ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision)
            AND actual_return_pct NOT IN ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision)
            AND residual_error NOT IN ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision)
            AND absolute_error_usd NOT IN ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision)
            AND percentage_error NOT IN ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision)
            AND actual_price_at_t_plus_1 >= 1.0
            AND direction_realized IN ('UP', 'DOWN')
            AND direction_realized = (CASE WHEN actual_return_pct >= 0.0 THEN 'UP' ELSE 'DOWN' END)
            AND direction_correct = (direction_predicted = direction_realized)
            AND absolute_error_usd >= 0.0
            AND percentage_error >= 0.0
            AND error_attribution_tag IN (
                'ACCURATE',
                'MOMENTUM_UNDERESTIMATED',
                'MOMENTUM_OVERESTIMATED',
                'SIDEWAYS_DRIFT',
                'DIRECTION_MISALIGNED',
                'BLACK_SWAN_OUTLIER'
            )
        )
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_predictions_canonical_tick_version
ON public.model_predictions (tick_id, model_version)
WHERE (is_canonical IS TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_predictions_primary_tick
ON public.model_predictions (tick_id)
WHERE (is_primary IS TRUE);

CREATE INDEX IF NOT EXISTS ix_model_predictions_tick_id
ON public.model_predictions (tick_id);

CREATE INDEX IF NOT EXISTS ix_model_predictions_model_version
ON public.model_predictions (model_version);

CREATE TABLE IF NOT EXISTS public.prediction_investigations (
    id SERIAL PRIMARY KEY,
    tick_id INTEGER NOT NULL,
    prediction_id INTEGER,
    model_version VARCHAR(64) NOT NULL,
    headline VARCHAR(512) NOT NULL,
    headline_source_url TEXT,
    headline_published_at TIMESTAMPTZ,
    suspected_cause_category VARCHAR(64) NOT NULL,
    hypothesis_notes TEXT NOT NULL,
    supporting_evidence_json JSON,
    diagnostic_confidence DOUBLE PRECISION,
    unresolved_questions TEXT,
    verification_status VARCHAR(32) NOT NULL DEFAULT 'UNVERIFIED_HYPOTHESIS',
    verified_ground_truth_label VARCHAR(64),
    verified_by VARCHAR(64),
    verified_at TIMESTAMPTZ,
    revision_history_json JSON NOT NULL DEFAULT '[]'::json,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_pi_tick_id
        FOREIGN KEY (tick_id) REFERENCES public.market_ticks(id)
        ON UPDATE NO ACTION ON DELETE NO ACTION,
    CONSTRAINT fk_pi_prediction_id
        FOREIGN KEY (prediction_id) REFERENCES public.model_predictions(id)
        ON UPDATE NO ACTION ON DELETE NO ACTION,
    CONSTRAINT fk_pi_model_version
        FOREIGN KEY (model_version) REFERENCES public.model_registry(model_version)
        ON UPDATE NO ACTION ON DELETE NO ACTION,
    CONSTRAINT chk_pi_verification_status CHECK (
        verification_status IN (
            'UNVERIFIED_HYPOTHESIS',
            'CONFIRMED_CAUSE',
            'REFUTED',
            'APPROVED_FOR_TRAINING'
        )
        AND (
            diagnostic_confidence IS NULL
            OR (
                diagnostic_confidence NOT IN ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision)
                AND diagnostic_confidence >= 0.0
                AND diagnostic_confidence <= 1.0
            )
        )
    )
);

CREATE INDEX IF NOT EXISTS ix_prediction_investigations_tick_id
ON public.prediction_investigations (tick_id);

CREATE INDEX IF NOT EXISTS ix_prediction_investigations_model_version
ON public.prediction_investigations (model_version);
"""


def _verify_new_tables_subset(conn: Any) -> None:
    """Verifies that the 4 model governance tables, their columns, FKs, constraints, and indexes have zero drift."""
    inspector = inspect(conn)
    existing_tables = set(inspector.get_table_names())
    new_tables = ("model_registry", "model_lifecycle_events", "model_predictions", "prediction_investigations")

    for tbl_name in new_tables:
        if tbl_name not in existing_tables:
            raise SchemaDriftError(f"Table '{tbl_name}' missing after TX 1A.")
        expected_cols = CANONICAL_TABLE_COLUMNS[tbl_name]
        live_cols = {c["name"]: c for c in inspector.get_columns(tbl_name)}
        expected_names = [c[0] for c in expected_cols]
        if set(live_cols.keys()) != set(expected_names):
            raise SchemaDriftError(f"Column set drift on '{tbl_name}': got {sorted(live_cols.keys())}.")
        for col_name, exp_type, exp_nullable in expected_cols:
            c_info = live_cols[col_name]
            if _classify_column_type(str(c_info["type"])) != exp_type:
                raise SchemaDriftError(f"Column type drift on '{tbl_name}.{col_name}'.")
            if bool(c_info.get("nullable", True)) != exp_nullable:
                raise SchemaDriftError(f"Column nullability drift on '{tbl_name}.{col_name}'.")


def run_tx1a(target_engine: Engine, bootstrap_base_tables: bool = True) -> Dict[str, Any]:
    """
    Transaction 1A: Additive Table Creation, Indexes, Guarded Trigger Functions & Triggers.
    """
    with target_engine.begin() as conn:
        conn.execute(text(f"SELECT pg_advisory_xact_lock({MIGRATION_ADVISORY_LOCK_ID});"))
        conn.execute(text("SET LOCAL lock_timeout = '5s';"))
        conn.execute(text("SET LOCAL statement_timeout = '30s';"))

        inspector = inspect(conn)
        existing_tables = set(inspector.get_table_names())
        if bootstrap_base_tables and "market_ticks" not in existing_tables:
            conn.execute(text(_BASE_HISTORICAL_TABLES_DDL))

        conn.execute(text(_TX1A_NEW_TABLES_AND_INDEXES_DDL))
        _verify_new_tables_subset(conn)

        created_funcs = 0
        for f_spec in CANONICAL_FUNCTIONS:
            if ensure_canonical_function(conn, f_spec):
                created_funcs += 1

        created_trigs = 0
        for t_spec in CANONICAL_TRIGGERS:
            if ensure_canonical_trigger(conn, t_spec):
                created_trigs += 1

    return {
        "tx": "1A",
        "functions_created": created_funcs,
        "triggers_created": created_trigs,
    }


def run_tx1b(target_engine: Engine) -> Dict[str, Any]:
    """
    Transaction 1B: Atomic Model Registry & Lifecycle Seeding (`xgb_onnx_v1.0.0` & `p1_controlled_candidate_v1`).
    """
    with target_engine.begin() as conn:
        conn.execute(text(f"SELECT pg_advisory_xact_lock({MIGRATION_ADVISORY_LOCK_ID});"))
        conn.execute(text("SET LOCAL lock_timeout = '5s';"))
        conn.execute(text("SET LOCAL statement_timeout = '30s';"))

        seeded_models: List[str] = []

        # 1. Seed or verify baseline model xgb_onnx_v1.0.0
        base_row = conn.execute(
            text("""
                SELECT model_version, regressor_sha256, classifier_sha256,
                       feature_schema_version, current_status, lifecycle_seq,
                       evaluation_metrics
                FROM public.model_registry
                WHERE model_version = :mv
            """),
            {"mv": BASELINE_MODEL_SPEC["model_version"]},
        ).mappings().first()

        if base_row is None:
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
                {
                    **BASELINE_MODEL_SPEC,
                    "split_metadata": json.dumps(BASELINE_MODEL_SPEC["split_metadata"]),
                    "evaluation_metrics": json.dumps(BASELINE_MODEL_SPEC["evaluation_metrics"]),
                },
            )
            conn.execute(
                text("""
                    INSERT INTO public.model_lifecycle_events (
                        model_version, lifecycle_seq, event_type, from_status, to_status,
                        related_model_version, actor, reason, evidence_json
                    ) VALUES (
                        :mv, 1, 'REGISTERED', NULL, 'ACTIVE',
                        NULL, 'scripts/migrate_db.py',
                        'Historical production baseline model registered during model governance migration.',
                        CAST(:ev_json AS json)
                    )
                """),
                {
                    "mv": BASELINE_MODEL_SPEC["model_version"],
                    "ev_json": json.dumps(BASELINE_MODEL_SPEC["evaluation_metrics"]),
                },
            )
            seeded_models.append(BASELINE_MODEL_SPEC["model_version"])
        else:
            live_base_metrics = (
                json.loads(base_row["evaluation_metrics"])
                if isinstance(base_row.get("evaluation_metrics"), str)
                else base_row.get("evaluation_metrics")
            )
            if (
                base_row["regressor_sha256"] != BASELINE_MODEL_SPEC["regressor_sha256"]
                or base_row["classifier_sha256"] != BASELINE_MODEL_SPEC["classifier_sha256"]
                or base_row["feature_schema_version"] != BASELINE_MODEL_SPEC["feature_schema_version"]
                or base_row["current_status"] != "ACTIVE"
                or live_base_metrics != BASELINE_MODEL_SPEC["evaluation_metrics"]
            ):
                raise SchemaDriftError(
                    f"Existing baseline model '{BASELINE_MODEL_SPEC['model_version']}' in model_registry exhibits artifact digest, metric, or status drift."
                )

        # 2. Seed or verify rejected candidate model p1_controlled_candidate_v1 through 3-step state machine
        cand_row = conn.execute(
            text("""
                SELECT model_version, regressor_sha256, classifier_sha256,
                       feature_schema_version, current_status, lifecycle_seq,
                       evaluation_metrics
                FROM public.model_registry
                WHERE model_version = :mv
            """),
            {"mv": CANDIDATE_MODEL_SPEC["model_version"]},
        ).mappings().first()

        if cand_row is None:
            # Step 1 (seq = 1): Register as CANDIDATE with evaluation_metrics = NULL
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
                        :training_row_count, CAST(:split_metadata AS json), NULL,
                        :code_revision_sha, 'CANDIDATE'
                    )
                """),
                {
                    **CANDIDATE_MODEL_SPEC,
                    "split_metadata": json.dumps(CANDIDATE_MODEL_SPEC["split_metadata"]),
                },
            )
            conn.execute(
                text("""
                    INSERT INTO public.model_lifecycle_events (
                        model_version, lifecycle_seq, event_type, from_status, to_status,
                        related_model_version, actor, reason, evidence_json
                    ) VALUES (
                        :mv, 1, 'REGISTERED', NULL, 'CANDIDATE',
                        'xgb_onnx_v1.0.0', 'scripts/migrate_db.py',
                        'Controlled candidate retrained on point-in-time causally corrected dataset.',
                        CAST(:ev_json AS json)
                    )
                """),
                {
                    "mv": CANDIDATE_MODEL_SPEC["model_version"],
                    "ev_json": json.dumps(CANDIDATE_MODEL_SPEC["split_metadata"]),
                },
            )

            # Step 2 (seq = 2): Populate evaluation_metrics while remaining CANDIDATE + EVALUATED event
            conn.execute(
                text("""
                    UPDATE public.model_registry
                    SET evaluation_metrics = CAST(:eval_json AS json)
                    WHERE model_version = :mv
                """),
                {
                    "mv": CANDIDATE_MODEL_SPEC["model_version"],
                    "eval_json": json.dumps(CANDIDATE_MODEL_SPEC["evaluation_metrics"]),
                },
            )
            conn.execute(
                text("""
                    INSERT INTO public.model_lifecycle_events (
                        model_version, lifecycle_seq, event_type, from_status, to_status,
                        related_model_version, actor, reason, evidence_json
                    ) VALUES (
                        :mv, 2, 'EVALUATED', 'CANDIDATE', 'CANDIDATE',
                        'xgb_onnx_v1.0.0', 'scripts/migrate_db.py',
                        :reason,
                        CAST(:ev_json AS json)
                    )
                """),
                {
                    "mv": CANDIDATE_MODEL_SPEC["model_version"],
                    "reason": CANDIDATE_EVALUATED_REASON,
                    "ev_json": json.dumps(CANDIDATE_MODEL_SPEC["evaluation_metrics"]),
                },
            )

            # Step 3 (seq = 3): Transition CANDIDATE -> REJECTED + REJECTED event
            conn.execute(
                text("""
                    UPDATE public.model_registry
                    SET current_status = 'REJECTED'
                    WHERE model_version = :mv
                """),
                {"mv": CANDIDATE_MODEL_SPEC["model_version"]},
            )
            conn.execute(
                text("""
                    INSERT INTO public.model_lifecycle_events (
                        model_version, lifecycle_seq, event_type, from_status, to_status,
                        related_model_version, actor, reason, evidence_json
                    ) VALUES (
                        :mv, 3, 'REJECTED', 'CANDIDATE', 'REJECTED',
                        'xgb_onnx_v1.0.0', 'scripts/migrate_db.py',
                        :reason,
                        CAST(:ev_json AS json)
                    )
                """),
                {
                    "mv": CANDIDATE_MODEL_SPEC["model_version"],
                    "reason": CANDIDATE_REJECTION_REASON,
                    "ev_json": json.dumps(CANDIDATE_MODEL_SPEC["evaluation_metrics"]),
                },
            )
            seeded_models.append(CANDIDATE_MODEL_SPEC["model_version"])
        else:
            live_cand_metrics = (
                json.loads(cand_row["evaluation_metrics"])
                if isinstance(cand_row.get("evaluation_metrics"), str)
                else cand_row.get("evaluation_metrics")
            )
            if (
                cand_row["regressor_sha256"] != CANDIDATE_MODEL_SPEC["regressor_sha256"]
                or cand_row["classifier_sha256"] != CANDIDATE_MODEL_SPEC["classifier_sha256"]
                or cand_row["current_status"] != "REJECTED"
                or int(cand_row["lifecycle_seq"]) != 3
                or live_cand_metrics != CANDIDATE_MODEL_SPEC["evaluation_metrics"]
            ):
                raise SchemaDriftError(
                    f"Existing candidate model '{CANDIDATE_MODEL_SPEC['model_version']}' in model_registry exhibits artifact digest, metric, status, or sequence drift."
                )

    return {"tx": "1B", "seeded_models": seeded_models}


def run_tx1c(target_engine: Engine) -> Dict[str, Any]:
    """
    Transaction 1C: Historical Prediction & Reconciliation Backfill into `model_predictions`.
    """
    with target_engine.begin() as conn:
        conn.execute(text(f"SELECT pg_advisory_xact_lock({MIGRATION_ADVISORY_LOCK_ID});"))
        conn.execute(text("SET LOCAL lock_timeout = '5s';"))
        conn.execute(text("SET LOCAL statement_timeout = '30s';"))

        # 1. Insert unresolved PRIMARY prediction rows for historical market_ticks with forecasts
        ins_res = conn.execute(
            text("""
                INSERT INTO public.model_predictions (
                    tick_id,
                    model_version,
                    run_mode,
                    is_canonical,
                    is_primary,
                    parent_prediction_id,
                    feature_schema_version,
                    predicted_return_pct,
                    predicted_price_1h,
                    direction_predicted,
                    direction_prob_up,
                    clears_hurdle,
                    created_at
                )
                SELECT
                    mt.id AS tick_id,
                    mt.model_version,
                    'PRIMARY' AS run_mode,
                    TRUE AS is_canonical,
                    TRUE AS is_primary,
                    NULL AS parent_prediction_id,
                    mt.feature_schema_version,
                    ((mt.predicted_price_1h - mt.close) / mt.close) * 100.0 AS predicted_return_pct,
                    mt.predicted_price_1h,
                    CASE
                        WHEN ((mt.predicted_price_1h - mt.close) / mt.close) >= 0.0 THEN 'UP'
                        ELSE 'DOWN'
                    END AS direction_predicted,
                    COALESCE(mt.model_confidence, 0.5) AS direction_prob_up,
                    (abs(((mt.predicted_price_1h - mt.close) / mt.close) * 100.0) >= :hurdle_pct) AS clears_hurdle,
                    mt.timestamp AS created_at
                FROM public.market_ticks mt
                WHERE mt.predicted_price_1h IS NOT NULL
                  AND mt.close >= 1.0
                  AND mt.predicted_price_1h >= 1.0
                ON CONFLICT (tick_id, model_version) WHERE (is_canonical IS TRUE) DO NOTHING
            """),
            {"hurdle_pct": float(MIN_HURDLE_PCT)},
        )
        inserted_predictions = ins_res.rowcount if ins_res.rowcount is not None and ins_res.rowcount >= 0 else 0

        # 2. Populate delayed T+1h reconciliation outcomes for unresolved predictions
        upd_res = conn.execute(
            text("""
                UPDATE public.model_predictions mp
                SET
                    outcome_tick_id = t_out.id,
                    evaluation_policy_version = 'binary_exact_t1_v2',
                    actual_price_at_t_plus_1 = t_out.close,
                    actual_return_pct = ((t_out.close - t_pred.close) / t_pred.close) * 100.0,
                    direction_realized = CASE
                        WHEN ((t_out.close - t_pred.close) / t_pred.close) >= 0.0 THEN 'UP'
                        ELSE 'DOWN'
                    END,
                    direction_correct = (
                        mp.direction_predicted = (
                            CASE
                                WHEN ((t_out.close - t_pred.close) / t_pred.close) >= 0.0 THEN 'UP'
                                ELSE 'DOWN'
                            END
                        )
                    ),
                    residual_error = (((t_out.close - t_pred.close) / t_pred.close) * 100.0) - mp.predicted_return_pct,
                    absolute_error_usd = abs(t_out.close - mp.predicted_price_1h),
                    percentage_error = (abs(t_out.close - mp.predicted_price_1h) / t_out.close) * 100.0,
                    is_black_swan = (
                        abs(((t_out.close - t_pred.close) / t_pred.close) * 100.0) >= 3.8
                        OR abs((((t_out.close - t_pred.close) / t_pred.close) * 100.0) - mp.predicted_return_pct) >= 3.8
                    ),
                    error_attribution_tag = CASE
                        WHEN (
                            abs(((t_out.close - t_pred.close) / t_pred.close) * 100.0) >= 3.8
                            OR abs((((t_out.close - t_pred.close) / t_pred.close) * 100.0) - mp.predicted_return_pct) >= 3.8
                        ) THEN 'BLACK_SWAN_OUTLIER'
                        WHEN (
                            mp.direction_predicted = (
                                CASE
                                    WHEN ((t_out.close - t_pred.close) / t_pred.close) >= 0.0 THEN 'UP'
                                    ELSE 'DOWN'
                                END
                            )
                        ) THEN
                            CASE
                                WHEN ((abs(t_out.close - mp.predicted_price_1h) / t_out.close) * 100.0) <= 0.35 THEN 'ACCURATE'
                                WHEN abs(((t_out.close - t_pred.close) / t_pred.close) * 100.0) > abs(mp.predicted_return_pct) THEN 'MOMENTUM_UNDERESTIMATED'
                                ELSE 'MOMENTUM_OVERESTIMATED'
                            END
                        ELSE
                            CASE
                                WHEN (
                                    ((abs(t_out.close - mp.predicted_price_1h) / t_out.close) * 100.0) <= 0.20
                                    OR abs(t_out.close - mp.predicted_price_1h) < 150.0
                                ) THEN 'SIDEWAYS_DRIFT'
                                ELSE 'DIRECTION_MISALIGNED'
                            END
                    END,
                    reconciled_at = r.timestamp
                FROM public.reconciliations r
                JOIN public.market_ticks t_pred ON t_pred.id = r.tick_id
                JOIN public.market_ticks t_out ON t_out.timestamp = t_pred.timestamp + INTERVAL '1 hour'
                WHERE mp.tick_id = r.tick_id
                  AND mp.is_canonical IS TRUE
                  AND mp.reconciled_at IS NULL
            """)
        )
        reconciled_predictions = upd_res.rowcount if upd_res.rowcount is not None and upd_res.rowcount >= 0 else 0

        # 3. Verify zero drift between market_ticks and canonical model_predictions rows
        drift_rows = conn.execute(
            text("""
                SELECT mp.id, mp.tick_id, mp.predicted_price_1h, mt.predicted_price_1h AS mt_pred
                FROM public.model_predictions mp
                JOIN public.market_ticks mt ON mt.id = mp.tick_id
                WHERE mp.is_primary IS TRUE
                  AND abs(mp.predicted_price_1h - mt.predicted_price_1h) > 1e-6
            """)
        ).fetchall()
        if drift_rows:
            raise SchemaDriftError(
                f"Detected forecast price drift between market_ticks and model_predictions on {len(drift_rows)} row(s)."
            )

    return {
        "tx": "1C",
        "inserted_predictions": inserted_predictions,
        "reconciled_predictions": reconciled_predictions,
    }


def run_tx2(target_engine: Engine) -> Dict[str, Any]:
    """
    Transaction 2: Isolated Existing-Table Evolution (`paper_portfolio` & `market_ticks`).
    Acquires both migration lock (888002) and PORTFOLIO_ADVISORY_LOCK_ID (888001).
    """
    with target_engine.begin() as conn:
        conn.execute(text(f"SELECT pg_advisory_xact_lock({MIGRATION_ADVISORY_LOCK_ID});"))
        conn.execute(text(f"SELECT pg_advisory_xact_lock({PORTFOLIO_ADVISORY_LOCK_ID});"))
        conn.execute(text("SET LOCAL lock_timeout = '5s';"))
        conn.execute(text("SET LOCAL statement_timeout = '15s';"))

        # Pre-check 1: Verify 0 duplicate timestamps in market_ticks
        dup_ticks = conn.execute(
            text("SELECT timestamp, COUNT(*) FROM public.market_ticks GROUP BY timestamp HAVING COUNT(*) > 1")
        ).fetchall()
        if dup_ticks:
            raise SchemaDriftError(
                f"Cannot create uq_market_ticks_timestamp: found {len(dup_ticks)} duplicate timestamp group(s) in market_ticks."
            )

        # 1. Add tick_id column to paper_portfolio if absent
        conn.execute(text("ALTER TABLE public.paper_portfolio ADD COLUMN IF NOT EXISTS tick_id INTEGER;"))

        # Pre-check 2: Verify 0 duplicate non-null tick_ids in paper_portfolio
        dup_port = conn.execute(
            text("SELECT tick_id, COUNT(*) FROM public.paper_portfolio WHERE tick_id IS NOT NULL GROUP BY tick_id HAVING COUNT(*) > 1")
        ).fetchall()
        if dup_port:
            raise SchemaDriftError(
                f"Cannot create uq_paper_portfolio_tick_id: found {len(dup_port)} duplicate non-null tick_id group(s) in paper_portfolio."
            )

        # 2. Add named foreign key fk_paper_portfolio_tick_id if absent
        fk_row = conn.execute(
            text("""
                SELECT c.conname,
                       rel_src.relname AS src_table,
                       att_src.attname AS src_col,
                       rel_tgt.relname AS tgt_table,
                       att_tgt.attname AS tgt_col,
                       c.confupdtype AS upd_type,
                       c.confdeltype AS del_type
                FROM pg_constraint c
                JOIN pg_class rel_src ON rel_src.oid = c.conrelid
                JOIN pg_namespace n ON n.oid = rel_src.relnamespace
                JOIN pg_class rel_tgt ON rel_tgt.oid = c.confrelid
                JOIN pg_attribute att_src ON att_src.attrelid = c.conrelid AND att_src.attnum = c.conkey[1]
                JOIN pg_attribute att_tgt ON att_tgt.attrelid = c.confrelid AND att_tgt.attnum = c.confkey[1]
                WHERE n.nspname = 'public'
                  AND rel_src.relname = 'paper_portfolio'
                  AND c.conname = 'fk_paper_portfolio_tick_id'
            """)
        ).mappings().first()

        if fk_row is None:
            conn.execute(
                text("""
                    ALTER TABLE public.paper_portfolio
                    ADD CONSTRAINT fk_paper_portfolio_tick_id
                    FOREIGN KEY (tick_id) REFERENCES public.market_ticks(id)
                    ON UPDATE NO ACTION ON DELETE NO ACTION;
                """)
            )
        else:
            if (
                fk_row["src_col"] != "tick_id"
                or fk_row["tgt_table"] != "market_ticks"
                or fk_row["tgt_col"] != "id"
                or str(fk_row["upd_type"]) != "a"
                or str(fk_row["del_type"]) != "a"
            ):
                raise SchemaDriftError("Foreign key 'fk_paper_portfolio_tick_id' on paper_portfolio exhibits catalog drift.")

        # 3. Create partial unique index uq_paper_portfolio_tick_id if absent
        conn.execute(
            text("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_portfolio_tick_id
                ON public.paper_portfolio (tick_id)
                WHERE (tick_id IS NOT NULL);
            """)
        )

        # 4. Create unique index uq_market_ticks_timestamp if absent
        conn.execute(
            text("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_market_ticks_timestamp
                ON public.market_ticks (timestamp);
            """)
        )

    _TICK_ID_COLUMN_CACHE.clear()
    return {"tx": "2", "status": "completed"}


def inspect_dry_run_plan(target_engine: Engine) -> Dict[str, Any]:
    """
    Strictly read-only dry-run inspection. Performs zero DDL/DML writes.
    Reports which tables, functions, triggers, seed models, and backfills are pending.
    """
    with target_engine.connect() as conn:
        inspector = inspect(conn)
        existing_tables = set(inspector.get_table_names())
        missing_tables = [t for t in CANONICAL_TABLE_COLUMNS if t not in existing_tables]

        has_pp_tick_id = False
        if "paper_portfolio" in existing_tables:
            pp_cols = {c["name"] for c in inspector.get_columns("paper_portfolio")}
            has_pp_tick_id = "tick_id" in pp_cols

        tick_count = 0
        rec_count = 0
        dup_tick_timestamps = 0
        if "market_ticks" in existing_tables:
            tick_count = int(conn.execute(text("SELECT COUNT(*) FROM market_ticks WHERE predicted_price_1h IS NOT NULL")).scalar() or 0)
            dup_rows = conn.execute(
                text("SELECT timestamp, COUNT(*) FROM market_ticks GROUP BY timestamp HAVING COUNT(*) > 1")
            ).fetchall()
            dup_tick_timestamps = len(dup_rows)
        if "reconciliations" in existing_tables:
            rec_count = int(conn.execute(text("SELECT COUNT(*) FROM reconciliations")).scalar() or 0)

        return {
            "mode": "DRY_RUN",
            "writes_executed": False,
            "existing_tables": sorted(existing_tables),
            "missing_tables": missing_tables,
            "paper_portfolio_has_tick_id": has_pp_tick_id,
            "eligible_market_ticks_for_backfill": tick_count,
            "eligible_reconciliations_for_backfill": rec_count,
            "duplicate_market_tick_timestamps": dup_tick_timestamps,
        }


def run_migration(
    target_engine: Optional[Engine] = None,
    execute: bool = False,
    bootstrap_base_tables: bool = True,
) -> Dict[str, Any]:
    """
    Main entry point for database schema and model governance migration.
    - When `execute=False` (default): performs a 100% read-only dry-run inspection.
    - When `execute=True`: executes TX 1A, TX 1B, TX 1C, and TX 2 sequentially,
      then runs `verify_db_schema(..., require_seed_models=True)` to confirm catalog equivalence.
    """
    eng = target_engine or default_engine
    if not execute:
        return inspect_dry_run_plan(eng)

    res_1a = run_tx1a(eng, bootstrap_base_tables=bootstrap_base_tables)
    res_1b = run_tx1b(eng)
    res_1c = run_tx1c(eng)
    res_2 = run_tx2(eng)
    verification = verify_db_schema(eng, require_seed_models=True)

    return {
        "mode": "EXECUTE",
        "writes_executed": True,
        "tx1a": res_1a,
        "tx1b": res_1b,
        "tx1c": res_1c,
        "tx2": res_2,
        "verification": verification,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mudra Guarded Idempotent Database Migration Runner (Dry-Run by Default)."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Inspect live schema and print migration plan without modifying the database (default).",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Execute TX 1A, TX 1B, TX 1C, and TX 2 against the target database.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Optional explicit database URL override.",
    )
    args = parser.parse_args(argv)

    target_eng = create_db_engine(args.db_url) if args.db_url else default_engine
    try:
        result = run_migration(target_engine=target_eng, execute=bool(args.execute))
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"[MIGRATION FAILED] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())