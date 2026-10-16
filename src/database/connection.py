"""
Database Connection Manager, Session Factory, and Read-Only Schema Verifier for Mudra.
Supports Neon Serverless PostgreSQL with SSL pooling and SQLite local fallback.
"""
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union
import tomllib
import logging

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session, attributes, defer, sessionmaker

from src.database.models import Base, PaperPortfolio

# Path to local Streamlit secrets
SECRETS_FILE = Path(".streamlit/secrets.toml")

__all__ = [
    "SchemaDriftError",
    "FunctionSpec",
    "TriggerSpec",
    "CANONICAL_TABLE_COLUMNS",
    "CANONICAL_FOREIGN_KEYS",
    "CANONICAL_CONSTRAINTS",
    "CANONICAL_INDEXES",
    "CANONICAL_FUNCTIONS",
    "CANONICAL_TRIGGERS",
    "normalize_sql",
    "normalize_pg_predicate",
    "ensure_canonical_function",
    "ensure_canonical_trigger",
    "verify_db_schema",
    "get_database_url",
    "create_db_engine",
    "engine",
    "SessionLocal",
    "get_db",
    "has_tick_id_column",
    "init_db",
    "test_connection",
]


class SchemaDriftError(RuntimeError):
    """Raised when the live database catalog is missing required objects or exhibits schema drift."""
    pass


@dataclass(frozen=True)
class FunctionSpec:
    schema_name: str
    name: str
    expected_prosrc: str
    create_sql: str


@dataclass(frozen=True)
class TriggerSpec:
    name: str
    table_name: str
    func_name: str
    expected_tgtype: int
    deferrable: bool
    initially_deferred: bool
    expected_triggerdef: str
    create_sql: str


def normalize_sql(sql_str: Optional[str]) -> str:
    """
    Normalizes SQL/PLpgSQL text for deterministic comparison by stripping comments,
    collapsing whitespace, and normalizing case outside single-quoted literals.
    """
    if not sql_str:
        return ""
    # Strip block comments and line comments
    no_block = re.sub(r"/\*.*?\*/", " ", sql_str, flags=re.DOTALL)
    no_comments = re.sub(r"--[^\r\n]*", " ", no_block)

    # Tokenize by single-quoted literals to preserve literal casing while lowercasing SQL tokens
    parts = re.split(r"('(?:''|[^'])*')", no_comments)
    normalized_parts: List[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            normalized_parts.append(part)
        else:
            cleaned = part.lower()
            cleaned = re.sub(r"\s+", " ", cleaned)
            normalized_parts.append(cleaned)

    joined = "".join(normalized_parts).strip()
    joined = re.sub(r"\s*([(),;])\s*", r"\1", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    if joined.endswith(";"):
        joined = joined[:-1].strip()
    return joined


def normalize_pg_predicate(expr: Optional[str]) -> str:
    """
    Normalizes PostgreSQL index WHERE predicates and constraint expressions by stripping
    PostgreSQL type casts (::text, ::character varying, ::double precision, etc.) and parentheses.
    """
    if not expr:
        return ""
    s = normalize_sql(expr)
    s = re.sub(
        r"::(?:character varying(?:\(\d+\))?|varchar(?:\(\d+\))?|text|double precision|float8|integer|int4|boolean|bool|json|timestamptz|timestamp with time zone)\b(?:\[\])?",
        "",
        s,
    )
    s = s.replace("(", " ").replace(")", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =============================================================================
# CANONICAL SCHEMA CONTRACT DEFINITIONS
# =============================================================================

CANONICAL_TABLE_COLUMNS: Dict[str, List[Tuple[str, str, bool]]] = {
    "market_ticks": [
        ("id", "integer", False),
        ("timestamp", "timestamp", False),
        ("open", "float", False),
        ("high", "float", False),
        ("low", "float", False),
        ("close", "float", False),
        ("volume", "float", False),
        ("model_version", "varchar", False),
        ("feature_schema_version", "varchar", False),
        ("market_regime", "varchar", True),
        ("predicted_return_1h", "float", True),
        ("predicted_price_1h", "float", True),
        ("model_confidence", "float", True),
        ("features", "json", True),
    ],
    "api_telemetry": [
        ("id", "integer", False),
        ("timestamp", "timestamp", False),
        ("service_name", "varchar", False),
        ("endpoint", "varchar", False),
        ("latency_ms", "float", False),
        ("status_code", "integer", False),
        ("retry_count", "integer", False),
        ("rate_limit_remaining", "integer", True),
        ("tokens_used", "integer", True),
        ("cost_estimate_usd", "float", True),
        ("error_message", "text", True),
    ],
    "council_audits": [
        ("id", "integer", False),
        ("timestamp", "timestamp", False),
        ("tick_id", "integer", True),
        ("deliberation_latency_ms", "float", True),
        ("strategist_action", "varchar", False),
        ("strategist_conviction", "integer", False),
        ("rag_context_summary", "text", True),
        ("risk_verdict", "varchar", False),
        ("veto_reasons", "json", True),
        ("kelly_fraction", "float", False),
        ("final_decision", "varchar", False),
        ("veto_attribution", "varchar", True),
        ("execution_notes", "text", True),
    ],
    "reconciliations": [
        ("id", "integer", False),
        ("timestamp", "timestamp", False),
        ("tick_id", "integer", False),
        ("predicted_price", "float", False),
        ("actual_price_at_t_plus_1", "float", False),
        ("predicted_return_pct", "float", False),
        ("actual_return_pct", "float", False),
        ("residual_error", "float", False),
        ("absolute_error_usd", "float", False),
        ("percentage_error", "float", False),
        ("direction_predicted", "varchar", False),
        ("direction_realized", "varchar", False),
        ("direction_correct", "boolean", False),
        ("is_black_swan", "boolean", False),
        ("error_attribution_tag", "varchar", False),
    ],
    "paper_portfolio": [
        ("id", "integer", False),
        ("tick_id", "integer", True),
        ("timestamp", "timestamp", False),
        ("action", "varchar", False),
        ("btc_amount", "float", False),
        ("execution_price", "float", False),
        ("fee_deducted_usd", "float", False),
        ("slippage_usd", "float", False),
        ("cash_balance", "float", False),
        ("btc_balance", "float", False),
        ("total_equity_usd", "float", False),
        ("realized_pnl_usd", "float", False),
        ("unrealized_pnl_usd", "float", False),
        ("drawdown_pct", "float", False),
        ("cumulative_return_pct", "float", False),
        ("benchmark_btc_equity", "float", False),
        ("alpha_usd", "float", False),
        ("alpha_pct", "float", False),
        ("notes", "text", True),
    ],
    "model_registry": [
        ("id", "integer", False),
        ("model_version", "varchar", False),
        ("regressor_artifact_path", "varchar", False),
        ("regressor_sha256", "varchar", False),
        ("classifier_artifact_path", "varchar", False),
        ("classifier_sha256", "varchar", False),
        ("feature_schema_version", "varchar", False),
        ("target_definition", "varchar", False),
        ("training_dataset_path", "varchar", True),
        ("training_dataset_sha256", "varchar", True),
        ("training_row_count", "integer", True),
        ("split_metadata", "json", True),
        ("evaluation_metrics", "json", True),
        ("code_revision_sha", "varchar", True),
        ("current_status", "varchar", False),
        ("lifecycle_seq", "integer", False),
        ("created_at", "timestamp", False),
    ],
    "model_lifecycle_events": [
        ("id", "integer", False),
        ("model_version", "varchar", False),
        ("lifecycle_seq", "integer", False),
        ("event_type", "varchar", False),
        ("from_status", "varchar", True),
        ("to_status", "varchar", False),
        ("related_model_version", "varchar", True),
        ("actor", "varchar", False),
        ("reason", "text", False),
        ("evidence_json", "json", True),
        ("created_at", "timestamp", False),
    ],
    "model_predictions": [
        ("id", "integer", False),
        ("tick_id", "integer", False),
        ("model_version", "varchar", False),
        ("run_mode", "varchar", False),
        ("is_canonical", "boolean", False),
        ("is_primary", "boolean", False),
        ("parent_prediction_id", "integer", True),
        ("feature_schema_version", "varchar", False),
        ("predicted_return_pct", "float", False),
        ("predicted_price_1h", "float", False),
        ("direction_predicted", "varchar", False),
        ("direction_prob_up", "float", False),
        ("clears_hurdle", "boolean", False),
        ("created_at", "timestamp", False),
        ("outcome_tick_id", "integer", True),
        ("evaluation_policy_version", "varchar", True),
        ("actual_price_at_t_plus_1", "float", True),
        ("actual_return_pct", "float", True),
        ("direction_realized", "varchar", True),
        ("direction_correct", "boolean", True),
        ("residual_error", "float", True),
        ("absolute_error_usd", "float", True),
        ("percentage_error", "float", True),
        ("is_black_swan", "boolean", True),
        ("error_attribution_tag", "varchar", True),
        ("reconciled_at", "timestamp", True),
    ],
    "prediction_investigations": [
        ("id", "integer", False),
        ("tick_id", "integer", False),
        ("prediction_id", "integer", True),
        ("model_version", "varchar", False),
        ("headline", "varchar", False),
        ("headline_source_url", "text", True),
        ("headline_published_at", "timestamp", True),
        ("suspected_cause_category", "varchar", False),
        ("hypothesis_notes", "text", False),
        ("supporting_evidence_json", "json", True),
        ("diagnostic_confidence", "float", True),
        ("unresolved_questions", "text", True),
        ("verification_status", "varchar", False),
        ("verified_ground_truth_label", "varchar", True),
        ("verified_by", "varchar", True),
        ("verified_at", "timestamp", True),
        ("revision_history_json", "json", False),
        ("created_at", "timestamp", False),
        ("updated_at", "timestamp", False),
    ],
}

# 10 Named Foreign Keys: (conname, source_table, source_col, target_table, target_col)
CANONICAL_FOREIGN_KEYS: List[Tuple[str, str, str, str, str]] = [
    ("fk_mle_model_version", "model_lifecycle_events", "model_version", "model_registry", "model_version"),
    ("fk_mle_related_model_version", "model_lifecycle_events", "related_model_version", "model_registry", "model_version"),
    ("fk_mp_tick_id", "model_predictions", "tick_id", "market_ticks", "id"),
    ("fk_mp_model_version", "model_predictions", "model_version", "model_registry", "model_version"),
    ("fk_mp_parent_prediction_id", "model_predictions", "parent_prediction_id", "model_predictions", "id"),
    ("fk_mp_outcome_tick_id", "model_predictions", "outcome_tick_id", "market_ticks", "id"),
    ("fk_pi_tick_id", "prediction_investigations", "tick_id", "market_ticks", "id"),
    ("fk_pi_prediction_id", "prediction_investigations", "prediction_id", "model_predictions", "id"),
    ("fk_pi_model_version", "prediction_investigations", "model_version", "model_registry", "model_version"),
    ("fk_paper_portfolio_tick_id", "paper_portfolio", "tick_id", "market_ticks", "id"),
]

# 7 Named Table Constraints: (conname, table_name, contype, required_columns, required_tokens)
CANONICAL_CONSTRAINTS: List[Tuple[str, str, str, List[str], List[str]]] = [
    (
        "uq_model_registry_model_version",
        "model_registry",
        "u",
        ["model_version"],
        ["unique", "model_version"],
    ),
    (
        "chk_model_registry_status",
        "model_registry",
        "c",
        ["current_status", "lifecycle_seq"],
        ["'ACTIVE'", "'CANDIDATE'", "'REJECTED'", "'ARCHIVED'", "'ROLLED_BACK'", "lifecycle_seq >= 1"],
    ),
    (
        "uq_mle_model_version_seq",
        "model_lifecycle_events",
        "u",
        ["model_version", "lifecycle_seq"],
        ["unique", "model_version", "lifecycle_seq"],
    ),
    (
        "chk_lifecycle_event_vocabulary",
        "model_lifecycle_events",
        "c",
        ["lifecycle_seq", "event_type", "from_status", "to_status"],
        [
            "lifecycle_seq >= 1",
            "'REGISTERED'",
            "'EVALUATED'",
            "'REJECTED'",
            "'PROMOTED'",
            "'SUPERSEDED'",
            "'ROLLED_BACK'",
            "'ARCHIVED'",
            "from_status is null",
        ],
    ),
    (
        "chk_prediction_mode_invariants",
        "model_predictions",
        "c",
        [
            "run_mode",
            "is_canonical",
            "is_primary",
            "parent_prediction_id",
            "predicted_return_pct",
            "predicted_price_1h",
            "direction_predicted",
            "direction_prob_up",
        ],
        [
            "'PRIMARY'",
            "'SHADOW'",
            "'FORCED_REANALYSIS'",
            "predicted_price_1h >= 1.0",
            "direction_prob_up >= 0.0",
            "direction_prob_up <= 1.0",
            "'UP'",
            "'DOWN'",
        ],
    ),
    (
        "chk_reconciliation_all_or_nothing",
        "model_predictions",
        "c",
        [
            "direction_predicted",
            "outcome_tick_id",
            "evaluation_policy_version",
            "actual_price_at_t_plus_1",
            "actual_return_pct",
            "direction_realized",
            "direction_correct",
            "residual_error",
            "absolute_error_usd",
            "percentage_error",
            "is_black_swan",
            "error_attribution_tag",
            "reconciled_at",
        ],
        [
            "'binary_exact_t1_v2'",
            "actual_price_at_t_plus_1 >= 1.0",
            "'ACCURATE'",
            "'MOMENTUM_UNDERESTIMATED'",
            "'MOMENTUM_OVERESTIMATED'",
            "'SIDEWAYS_DRIFT'",
            "'DIRECTION_MISALIGNED'",
            "'BLACK_SWAN_OUTLIER'",
        ],
    ),
    (
        "chk_pi_verification_status",
        "prediction_investigations",
        "c",
        ["diagnostic_confidence", "verification_status"],
        [
            "'UNVERIFIED_HYPOTHESIS'",
            "'CONFIRMED_CAUSE'",
            "'REFUTED'",
            "'APPROVED_FOR_TRAINING'",
            "diagnostic_confidence >= 0.0",
            "diagnostic_confidence <= 1.0",
        ],
    ),
]

# 12 Explicit Indexes: (index_name, table_name, is_unique, key_columns, normalized_predicate)
CANONICAL_INDEXES: List[Tuple[str, str, bool, List[str], str]] = [
    ("uq_model_registry_single_active", "model_registry", True, ["current_status"], "current_status = 'ACTIVE'"),
    ("ix_model_lifecycle_events_model_version", "model_lifecycle_events", False, ["model_version"], ""),
    ("uq_mle_single_registered_per_model", "model_lifecycle_events", True, ["model_version"], "event_type = 'REGISTERED'"),
    ("uq_mle_single_evaluated_per_model", "model_lifecycle_events", True, ["model_version"], "event_type = 'EVALUATED'"),
    ("uq_model_predictions_canonical_tick_version", "model_predictions", True, ["tick_id", "model_version"], "is_canonical is true"),
    ("uq_model_predictions_primary_tick", "model_predictions", True, ["tick_id"], "is_primary is true"),
    ("ix_model_predictions_tick_id", "model_predictions", False, ["tick_id"], ""),
    ("ix_model_predictions_model_version", "model_predictions", False, ["model_version"], ""),
    ("ix_prediction_investigations_tick_id", "prediction_investigations", False, ["tick_id"], ""),
    ("ix_prediction_investigations_model_version", "prediction_investigations", False, ["model_version"], ""),
    ("uq_paper_portfolio_tick_id", "paper_portfolio", True, ["tick_id"], "tick_id is not null"),
    ("uq_market_ticks_timestamp", "market_ticks", True, ["timestamp"], ""),
]

# 6 Canonical Trigger Functions
_FUNC_1_BODY = """
BEGIN
    RAISE EXCEPTION '% is strictly append-only; % operations are forbidden.',
        TG_TABLE_NAME, TG_OP;
END;
"""

_FUNC_2_BODY = """
DECLARE
    prev_event_to_status VARCHAR(24);
    eval_event_exists BOOLEAN;
BEGIN
    IF TG_OP = 'INSERT' THEN
        NEW.lifecycle_seq := 1;
        IF NEW.model_version = 'xgb_onnx_v1.0.0' THEN
            IF NEW.current_status != 'ACTIVE' THEN
                RAISE EXCEPTION 'Baseline model xgb_onnx_v1.0.0 must be registered with current_status = ACTIVE.';
            END IF;
        ELSE
            IF NEW.current_status != 'CANDIDATE' OR NEW.evaluation_metrics IS NOT NULL THEN
                RAISE EXCEPTION 'New model % must be registered with current_status = CANDIDATE and evaluation_metrics = NULL.',
                    NEW.model_version;
            END IF;
        END IF;
        RETURN NEW;
    END IF;

    IF (OLD.id IS DISTINCT FROM NEW.id OR
        OLD.model_version IS DISTINCT FROM NEW.model_version OR
        OLD.regressor_artifact_path IS DISTINCT FROM NEW.regressor_artifact_path OR
        OLD.regressor_sha256 IS DISTINCT FROM NEW.regressor_sha256 OR
        OLD.classifier_artifact_path IS DISTINCT FROM NEW.classifier_artifact_path OR
        OLD.classifier_sha256 IS DISTINCT FROM NEW.classifier_sha256 OR
        OLD.feature_schema_version IS DISTINCT FROM NEW.feature_schema_version OR
        OLD.target_definition IS DISTINCT FROM NEW.target_definition OR
        OLD.training_dataset_path IS DISTINCT FROM NEW.training_dataset_path OR
        OLD.training_dataset_sha256 IS DISTINCT FROM NEW.training_dataset_sha256 OR
        OLD.training_row_count IS DISTINCT FROM NEW.training_row_count OR
        OLD.split_metadata::text IS DISTINCT FROM NEW.split_metadata::text OR
        OLD.code_revision_sha IS DISTINCT FROM NEW.code_revision_sha OR
        OLD.created_at IS DISTINCT FROM NEW.created_at) THEN
        RAISE EXCEPTION 'Provenance fields in model_registry are immutable for model %.', OLD.model_version;
    END IF;

    SELECT to_status INTO prev_event_to_status
    FROM public.model_lifecycle_events
    WHERE model_version = OLD.model_version AND lifecycle_seq = OLD.lifecycle_seq;

    IF NOT FOUND OR prev_event_to_status IS DISTINCT FROM OLD.current_status THEN
        RAISE EXCEPTION 'Cannot update model_registry for %: prior state at lifecycle_seq=% (status=%) has no matching event in model_lifecycle_events.',
            OLD.model_version, OLD.lifecycle_seq, OLD.current_status;
    END IF;

    IF OLD.model_version != 'xgb_onnx_v1.0.0' THEN
        SELECT EXISTS (
            SELECT 1 FROM public.model_lifecycle_events
            WHERE model_version = OLD.model_version AND event_type = 'EVALUATED'
        ) INTO eval_event_exists;

        IF OLD.evaluation_metrics IS NOT NULL AND NOT eval_event_exists THEN
            RAISE EXCEPTION 'Cannot update model_registry for %: prior evaluation_metrics update lacks an EVALUATED lifecycle event.',
                OLD.model_version;
        END IF;

        IF NEW.current_status = 'ACTIVE' AND (OLD.evaluation_metrics IS NULL OR NEW.evaluation_metrics IS NULL OR NOT eval_event_exists) THEN
            RAISE EXCEPTION 'Cannot promote candidate model % to ACTIVE without non-null evaluation_metrics and a recorded EVALUATED lifecycle event.',
                OLD.model_version;
        END IF;
    END IF;

    IF OLD.evaluation_metrics::text IS DISTINCT FROM NEW.evaluation_metrics::text THEN
        IF OLD.evaluation_metrics IS NOT NULL OR OLD.current_status != 'CANDIDATE' OR NEW.current_status != 'CANDIDATE' THEN
            RAISE EXCEPTION 'evaluation_metrics in model_registry can only be populated once while current_status remains CANDIDATE.';
        END IF;
    ELSIF OLD.current_status = NEW.current_status THEN
        RAISE EXCEPTION 'No-op update on model_registry for model % is forbidden.', OLD.model_version;
    ELSE
        IF NOT (
            (OLD.current_status = 'CANDIDATE' AND NEW.current_status IN ('ACTIVE', 'REJECTED', 'ARCHIVED')) OR
            (OLD.current_status = 'ACTIVE' AND NEW.current_status IN ('ARCHIVED', 'ROLLED_BACK')) OR
            (OLD.current_status = 'ARCHIVED' AND NEW.current_status = 'ACTIVE') OR
            (OLD.current_status = 'ROLLED_BACK' AND NEW.current_status IN ('ACTIVE', 'ARCHIVED')) OR
            (OLD.current_status = 'REJECTED' AND NEW.current_status = 'ARCHIVED')
        ) THEN
            RAISE EXCEPTION 'Invalid model_registry status transition for %: % -> %.',
                OLD.model_version, OLD.current_status, NEW.current_status;
        END IF;
    END IF;

    NEW.lifecycle_seq := OLD.lifecycle_seq + 1;
    RETURN NEW;
END;
"""

_FUNC_3_BODY = """
DECLARE
    reg_status VARCHAR(24);
    reg_eval JSON;
    reg_seq INTEGER;
    prev_event_type VARCHAR(32);
    prev_to_status VARCHAR(24);
    eval_event_exists BOOLEAN;
BEGIN
    SELECT current_status, evaluation_metrics, lifecycle_seq
    INTO reg_status, reg_eval, reg_seq
    FROM public.model_registry
    WHERE model_version = NEW.model_version
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model version % does not exist in model_registry.', NEW.model_version;
    END IF;

    IF NEW.lifecycle_seq IS NULL THEN
        NEW.lifecycle_seq := reg_seq;
    ELSIF NEW.lifecycle_seq != reg_seq THEN
        RAISE EXCEPTION 'Lifecycle event sequence mismatch for %: event lifecycle_seq (%) != model_registry.lifecycle_seq (%).',
            NEW.model_version, NEW.lifecycle_seq, reg_seq;
    END IF;

    IF reg_seq = 1 THEN
        IF NEW.event_type != 'REGISTERED' OR NEW.from_status IS NOT NULL OR NEW.to_status != reg_status THEN
            RAISE EXCEPTION 'Lifecycle event at seq=1 for % must be REGISTERED with from_status IS NULL and to_status=% (got event_type=%, from=%, to=%).',
                NEW.model_version, reg_status, NEW.event_type, NEW.from_status, NEW.to_status;
        END IF;
    ELSE
        SELECT event_type, to_status INTO prev_event_type, prev_to_status
        FROM public.model_lifecycle_events
        WHERE model_version = NEW.model_version AND lifecycle_seq = reg_seq - 1;

        IF NOT FOUND THEN
            RAISE EXCEPTION 'Missing prior lifecycle event at seq=% for %.', reg_seq - 1, NEW.model_version;
        END IF;

        IF NEW.from_status IS NULL OR NEW.from_status != prev_to_status OR NEW.to_status != reg_status THEN
            RAISE EXCEPTION 'Lifecycle transition mismatch at seq=% for %: expected % -> %, got % -> %.',
                reg_seq, NEW.model_version, prev_to_status, reg_status, NEW.from_status, NEW.to_status;
        END IF;

        SELECT EXISTS (
            SELECT 1 FROM public.model_lifecycle_events
            WHERE model_version = NEW.model_version AND event_type = 'EVALUATED'
        ) INTO eval_event_exists;

        IF NEW.event_type = 'EVALUATED' THEN
            IF eval_event_exists OR NEW.from_status != 'CANDIDATE' OR NEW.to_status != 'CANDIDATE' OR reg_eval IS NULL THEN
                RAISE EXCEPTION 'Invalid or duplicate EVALUATED lifecycle event for %.', NEW.model_version;
            END IF;
        ELSIF NEW.event_type = 'REJECTED' THEN
            IF NEW.from_status != 'CANDIDATE' OR NEW.to_status != 'REJECTED' OR NEW.related_model_version IS NULL THEN
                RAISE EXCEPTION 'Invalid REJECTED lifecycle event for %.', NEW.model_version;
            END IF;
        ELSIF NEW.event_type = 'PROMOTED' THEN
            IF NEW.from_status NOT IN ('CANDIDATE', 'ARCHIVED', 'ROLLED_BACK') OR NEW.to_status != 'ACTIVE' OR NEW.related_model_version IS NULL THEN
                RAISE EXCEPTION 'Invalid PROMOTED lifecycle event for %.', NEW.model_version;
            END IF;
            IF NEW.model_version != 'xgb_onnx_v1.0.0' AND (reg_eval IS NULL OR NOT eval_event_exists) THEN
                RAISE EXCEPTION 'PROMOTED lifecycle event rejected for %: candidate must have non-null evaluation_metrics and a prior EVALUATED lifecycle event.',
                    NEW.model_version;
            END IF;
        ELSIF NEW.event_type = 'SUPERSEDED' THEN
            IF NEW.from_status != 'ACTIVE' OR NEW.to_status != 'ARCHIVED' OR NEW.related_model_version IS NULL THEN
                RAISE EXCEPTION 'Invalid SUPERSEDED lifecycle event for %.', NEW.model_version;
            END IF;
        ELSIF NEW.event_type = 'ROLLED_BACK' THEN
            IF NEW.from_status != 'ACTIVE' OR NEW.to_status != 'ROLLED_BACK' OR NEW.related_model_version IS NULL THEN
                RAISE EXCEPTION 'Invalid ROLLED_BACK lifecycle event for %.', NEW.model_version;
            END IF;
        ELSIF NEW.event_type = 'ARCHIVED' THEN
            IF NEW.from_status NOT IN ('CANDIDATE', 'REJECTED', 'ROLLED_BACK') OR NEW.to_status != 'ARCHIVED' THEN
                RAISE EXCEPTION 'Invalid ARCHIVED lifecycle event for %.', NEW.model_version;
            END IF;
        ELSE
            RAISE EXCEPTION 'Invalid lifecycle event_type % at seq=% for %.', NEW.event_type, reg_seq, NEW.model_version;
        END IF;
    END IF;

    RETURN NEW;
END;
"""

_FUNC_4_BODY = """
DECLARE
    live_seq INTEGER;
    ev_type VARCHAR(32);
    ev_from VARCHAR(24);
    ev_to VARCHAR(24);
BEGIN
    SELECT lifecycle_seq INTO live_seq
    FROM public.model_registry
    WHERE model_version = NEW.model_version;

    IF NOT FOUND OR live_seq < NEW.lifecycle_seq THEN
        RETURN NULL;
    END IF;

    SELECT event_type, from_status, to_status
    INTO ev_type, ev_from, ev_to
    FROM public.model_lifecycle_events
    WHERE model_version = NEW.model_version AND lifecycle_seq = NEW.lifecycle_seq;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Atomic lifecycle audit violation: model_registry operation for % at lifecycle_seq=% has no matching event in model_lifecycle_events.',
            NEW.model_version, NEW.lifecycle_seq;
    END IF;

    IF TG_OP = 'INSERT' THEN
        IF ev_type != 'REGISTERED' OR ev_from IS NOT NULL OR ev_to != NEW.current_status THEN
            RAISE EXCEPTION 'Atomic lifecycle audit violation on INSERT for % at seq=1.', NEW.model_version;
        END IF;
    ELSIF OLD.evaluation_metrics IS NULL AND NEW.evaluation_metrics IS NOT NULL THEN
        IF ev_type != 'EVALUATED' OR ev_from != 'CANDIDATE' OR ev_to != 'CANDIDATE' THEN
            RAISE EXCEPTION 'Atomic lifecycle audit violation on evaluation_metrics update for % at seq=%.',
                NEW.model_version, NEW.lifecycle_seq;
        END IF;
    ELSE
        IF ev_from IS DISTINCT FROM OLD.current_status OR ev_to IS DISTINCT FROM NEW.current_status THEN
            RAISE EXCEPTION 'Atomic lifecycle audit violation on status transition % -> % for % at seq=%.',
                OLD.current_status, NEW.current_status, NEW.model_version, NEW.lifecycle_seq;
        END IF;
    END IF;

    RETURN NULL;
END;
"""

_FUNC_5_BODY = """
DECLARE
    pred_close DOUBLE PRECISION;
    canonical_pred_ret DOUBLE PRECISION;
    canonical_dir_pred VARCHAR(16);
    model_status VARCHAR(24);
    parent_tick_id INTEGER;
    parent_model_version VARCHAR(64);
    parent_is_canonical BOOLEAN;
BEGIN
    IF (NEW.outcome_tick_id IS NOT NULL OR
        NEW.evaluation_policy_version IS NOT NULL OR
        NEW.actual_price_at_t_plus_1 IS NOT NULL OR
        NEW.actual_return_pct IS NOT NULL OR
        NEW.direction_realized IS NOT NULL OR
        NEW.direction_correct IS NOT NULL OR
        NEW.residual_error IS NOT NULL OR
        NEW.absolute_error_usd IS NOT NULL OR
        NEW.percentage_error IS NOT NULL OR
        NEW.is_black_swan IS NOT NULL OR
        NEW.error_attribution_tag IS NOT NULL OR
        NEW.reconciled_at IS NOT NULL) THEN
        RAISE EXCEPTION 'Initial prediction insert must be unresolved; all 12 outcome fields must be NULL.';
    END IF;

    SELECT close INTO pred_close FROM public.market_ticks WHERE id = NEW.tick_id;
    IF NOT FOUND OR pred_close < 1.0 THEN
        RAISE EXCEPTION 'Valid prediction tick_id % with close >= 1.0 not found in market_ticks.', NEW.tick_id;
    END IF;

    IF NEW.predicted_price_1h < 1.0 THEN
        RAISE EXCEPTION 'predicted_price_1h (%) must be >= 1.0 USD.', NEW.predicted_price_1h;
    END IF;

    canonical_pred_ret := ((NEW.predicted_price_1h - pred_close) / pred_close) * 100.0;
    canonical_dir_pred := CASE WHEN canonical_pred_ret >= 0.0 THEN 'UP' ELSE 'DOWN' END;

    IF abs(NEW.predicted_return_pct - canonical_pred_ret) > 1e-5 THEN
        RAISE EXCEPTION 'Forecast return/price mismatch: supplied predicted_return_pct (%) deviates from canonical price-implied return (%) by more than 1e-5.',
            NEW.predicted_return_pct, canonical_pred_ret;
    END IF;

    IF NEW.direction_predicted IS DISTINCT FROM canonical_dir_pred THEN
        RAISE EXCEPTION 'Forecast direction mismatch: supplied direction_predicted (%) != canonical direction (%) for price-implied return (%).',
            NEW.direction_predicted, canonical_dir_pred, canonical_pred_ret;
    END IF;

    NEW.predicted_return_pct := canonical_pred_ret;
    NEW.direction_predicted := canonical_dir_pred;

    SELECT current_status INTO model_status
    FROM public.model_registry
    WHERE model_version = NEW.model_version
    FOR SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model version % is not registered in model_registry.', NEW.model_version;
    END IF;

    IF NEW.run_mode = 'PRIMARY' AND model_status != 'ACTIVE' THEN
        RAISE EXCEPTION 'Primary prediction rejected: model % has status % (must be ACTIVE).',
            NEW.model_version, model_status;
    END IF;

    IF NEW.run_mode = 'SHADOW' AND model_status NOT IN ('CANDIDATE', 'ACTIVE') THEN
        RAISE EXCEPTION 'Shadow prediction rejected: model % has status % (must be CANDIDATE or ACTIVE).',
            NEW.model_version, model_status;
    END IF;

    IF NEW.run_mode = 'FORCED_REANALYSIS' THEN
        SELECT tick_id, model_version, is_canonical
        INTO parent_tick_id, parent_model_version, parent_is_canonical
        FROM public.model_predictions
        WHERE id = NEW.parent_prediction_id
        FOR SHARE;

        IF NOT FOUND THEN
            RAISE EXCEPTION 'Parent prediction ID % does not exist.', NEW.parent_prediction_id;
        END IF;

        IF parent_tick_id != NEW.tick_id THEN
            RAISE EXCEPTION 'Parent lineage violation: parent tick_id (%) != child tick_id (%).',
                parent_tick_id, NEW.tick_id;
        END IF;

        IF parent_model_version != NEW.model_version THEN
            RAISE EXCEPTION 'Parent lineage violation: parent model_version (%) != child model_version (%).',
                parent_model_version, NEW.model_version;
        END IF;

        IF parent_is_canonical IS NOT TRUE THEN
            RAISE EXCEPTION 'Parent lineage violation: parent prediction must have is_canonical=TRUE.';
        END IF;
    END IF;

    RETURN NEW;
END;
"""

_FUNC_6_BODY = """
DECLARE
    pred_ts TIMESTAMPTZ;
    pred_close DOUBLE PRECISION;
    outcome_ts TIMESTAMPTZ;
    outcome_close DOUBLE PRECISION;
    expected_outcome_ts TIMESTAMPTZ;
    canonical_actual_ret DOUBLE PRECISION;
    canonical_residual DOUBLE PRECISION;
    canonical_abs_err DOUBLE PRECISION;
    canonical_pct_err DOUBLE PRECISION;
    canonical_dir_real VARCHAR(16);
    canonical_dir_correct BOOLEAN;
    canonical_black_swan BOOLEAN;
    canonical_tag VARCHAR(64);
BEGIN
    IF (OLD.id IS DISTINCT FROM NEW.id OR
        OLD.tick_id IS DISTINCT FROM NEW.tick_id OR
        OLD.model_version IS DISTINCT FROM NEW.model_version OR
        OLD.run_mode IS DISTINCT FROM NEW.run_mode OR
        OLD.is_canonical IS DISTINCT FROM NEW.is_canonical OR
        OLD.is_primary IS DISTINCT FROM NEW.is_primary OR
        OLD.parent_prediction_id IS DISTINCT FROM NEW.parent_prediction_id OR
        OLD.feature_schema_version IS DISTINCT FROM NEW.feature_schema_version OR
        OLD.predicted_return_pct IS DISTINCT FROM NEW.predicted_return_pct OR
        OLD.predicted_price_1h IS DISTINCT FROM NEW.predicted_price_1h OR
        OLD.direction_predicted IS DISTINCT FROM NEW.direction_predicted OR
        OLD.direction_prob_up IS DISTINCT FROM NEW.direction_prob_up OR
        OLD.clears_hurdle IS DISTINCT FROM NEW.clears_hurdle OR
        OLD.created_at IS DISTINCT FROM NEW.created_at) THEN
        RAISE EXCEPTION 'Point-in-time forecast fields in model_predictions are immutable.';
    END IF;

    IF OLD.reconciled_at IS NOT NULL THEN
        RAISE EXCEPTION 'Finalized reconciliation outcome in model_predictions is permanently immutable.';
    END IF;

    IF (NEW.reconciled_at IS NULL OR
        NEW.outcome_tick_id IS NULL OR
        NEW.evaluation_policy_version IS DISTINCT FROM 'binary_exact_t1_v2') THEN
        RAISE EXCEPTION 'Update on unresolved prediction must populate all 12 reconciliation fields with evaluation_policy_version = binary_exact_t1_v2.';
    END IF;

    SELECT timestamp, close INTO pred_ts, pred_close FROM public.market_ticks WHERE id = NEW.tick_id;
    IF NOT FOUND OR pred_close < 1.0 THEN
        RAISE EXCEPTION 'Valid prediction tick_id % with close >= 1.0 not found in market_ticks.', NEW.tick_id;
    END IF;

    SELECT timestamp, close INTO outcome_ts, outcome_close
    FROM public.market_ticks WHERE id = NEW.outcome_tick_id;
    IF NOT FOUND OR outcome_close < 1.0 THEN
        RAISE EXCEPTION 'Valid outcome tick_id % with close >= 1.0 not found in market_ticks.', NEW.outcome_tick_id;
    END IF;

    expected_outcome_ts := pred_ts + INTERVAL '1 hour';

    IF outcome_ts != expected_outcome_ts THEN
        RAISE EXCEPTION 'Outcome timestamp mismatch: prediction tick (%) requires exact outcome at (%), but outcome_tick_id % has timestamp (%).',
            pred_ts, expected_outcome_ts, NEW.outcome_tick_id, outcome_ts;
    END IF;

    IF abs(NEW.actual_price_at_t_plus_1 - outcome_close) > 1e-6 THEN
        RAISE EXCEPTION 'Outcome close price mismatch: actual_price_at_t_plus_1 (%) != market_ticks.close (%).',
            NEW.actual_price_at_t_plus_1, outcome_close;
    END IF;

    canonical_actual_ret := ((outcome_close - pred_close) / pred_close) * 100.0;
    canonical_residual := canonical_actual_ret - NEW.predicted_return_pct;
    canonical_abs_err := abs(outcome_close - NEW.predicted_price_1h);
    canonical_pct_err := (canonical_abs_err / outcome_close) * 100.0;
    canonical_dir_real := CASE WHEN canonical_actual_ret >= 0.0 THEN 'UP' ELSE 'DOWN' END;
    canonical_dir_correct := (NEW.direction_predicted = canonical_dir_real);

    IF (abs(NEW.actual_return_pct - canonical_actual_ret) > 1e-5 OR
        abs(NEW.residual_error - canonical_residual) > 1e-5 OR
        abs(NEW.absolute_error_usd - canonical_abs_err) > 1e-4 OR
        abs(NEW.percentage_error - canonical_pct_err) > 1e-5 OR
        NEW.direction_realized IS DISTINCT FROM canonical_dir_real OR
        NEW.direction_correct IS DISTINCT FROM canonical_dir_correct) THEN
        RAISE EXCEPTION 'Reconciliation formula mismatch on prediction %: outcome metrics violate canonical T+1 equations.',
            NEW.id;
    END IF;

    NEW.actual_price_at_t_plus_1 := outcome_close;
    NEW.actual_return_pct := canonical_actual_ret;
    NEW.residual_error := canonical_residual;
    NEW.absolute_error_usd := canonical_abs_err;
    NEW.percentage_error := canonical_pct_err;
    NEW.direction_realized := canonical_dir_real;
    NEW.direction_correct := canonical_dir_correct;

    canonical_black_swan := (abs(canonical_actual_ret) >= 3.8 OR abs(canonical_residual) >= 3.8);

    IF canonical_black_swan THEN
        canonical_tag := 'BLACK_SWAN_OUTLIER';
    ELSIF canonical_dir_correct THEN
        IF canonical_pct_err <= 0.35 THEN
            canonical_tag := 'ACCURATE';
        ELSIF abs(canonical_actual_ret) > abs(NEW.predicted_return_pct) THEN
            canonical_tag := 'MOMENTUM_UNDERESTIMATED';
        ELSE
            canonical_tag := 'MOMENTUM_OVERESTIMATED';
        END IF;
    ELSE
        IF (canonical_pct_err <= 0.20 OR canonical_abs_err < 150.0) THEN
            canonical_tag := 'SIDEWAYS_DRIFT';
        ELSE
            canonical_tag := 'DIRECTION_MISALIGNED';
        END IF;
    END IF;

    IF (NEW.is_black_swan IS DISTINCT FROM canonical_black_swan OR
        NEW.error_attribution_tag IS DISTINCT FROM canonical_tag) THEN
        RAISE EXCEPTION 'Reconciliation attribution mismatch on prediction %: expected is_black_swan=% and tag=%, got is_black_swan=% and tag=%.',
            NEW.id, canonical_black_swan, canonical_tag, NEW.is_black_swan, NEW.error_attribution_tag;
    END IF;

    RETURN NEW;
END;
"""

CANONICAL_FUNCTIONS: List[FunctionSpec] = [
    FunctionSpec(
        schema_name="public",
        name="mudra_prevent_mutation",
        expected_prosrc=_FUNC_1_BODY,
        create_sql=f"CREATE FUNCTION public.mudra_prevent_mutation() RETURNS TRIGGER LANGUAGE plpgsql AS $${_FUNC_1_BODY}$$;",
    ),
    FunctionSpec(
        schema_name="public",
        name="mudra_guard_model_registry_write",
        expected_prosrc=_FUNC_2_BODY,
        create_sql=f"CREATE FUNCTION public.mudra_guard_model_registry_write() RETURNS TRIGGER LANGUAGE plpgsql AS $${_FUNC_2_BODY}$$;",
    ),
    FunctionSpec(
        schema_name="public",
        name="mudra_validate_lifecycle_event_insert",
        expected_prosrc=_FUNC_3_BODY,
        create_sql=f"CREATE FUNCTION public.mudra_validate_lifecycle_event_insert() RETURNS TRIGGER LANGUAGE plpgsql AS $${_FUNC_3_BODY}$$;",
    ),
    FunctionSpec(
        schema_name="public",
        name="mudra_verify_registry_lifecycle_sync",
        expected_prosrc=_FUNC_4_BODY,
        create_sql=f"CREATE FUNCTION public.mudra_verify_registry_lifecycle_sync() RETURNS TRIGGER LANGUAGE plpgsql AS $${_FUNC_4_BODY}$$;",
    ),
    FunctionSpec(
        schema_name="public",
        name="mudra_validate_prediction_insert",
        expected_prosrc=_FUNC_5_BODY,
        create_sql=f"CREATE FUNCTION public.mudra_validate_prediction_insert() RETURNS TRIGGER LANGUAGE plpgsql AS $${_FUNC_5_BODY}$$;",
    ),
    FunctionSpec(
        schema_name="public",
        name="mudra_guard_prediction_immutability",
        expected_prosrc=_FUNC_6_BODY,
        create_sql=f"CREATE FUNCTION public.mudra_guard_prediction_immutability() RETURNS TRIGGER LANGUAGE plpgsql AS $${_FUNC_6_BODY}$$;",
    ),
]

# pg_trigger.tgtype bitmasks:
# ROW(1) | BEFORE(2) | INSERT(4) | DELETE(8) | UPDATE(16) | TRUNCATE(32)
# BEFORE DELETE FOR EACH ROW = 1 | 2 | 8 = 11
# BEFORE TRUNCATE FOR EACH STATEMENT = 2 | 32 = 34
# BEFORE UPDATE OR DELETE FOR EACH ROW = 1 | 2 | 8 | 16 = 27
# BEFORE INSERT OR UPDATE FOR EACH ROW = 1 | 2 | 4 | 16 = 23
# BEFORE INSERT FOR EACH ROW = 1 | 2 | 4 = 7
# AFTER INSERT OR UPDATE FOR EACH ROW = 1 | 4 | 16 = 21
# BEFORE UPDATE FOR EACH ROW = 1 | 2 | 16 = 19
CANONICAL_TRIGGERS: List[TriggerSpec] = [
    TriggerSpec(
        name="trg_mr_no_delete",
        table_name="model_registry",
        func_name="mudra_prevent_mutation",
        expected_tgtype=11,
        deferrable=False,
        initially_deferred=False,
        expected_triggerdef="CREATE TRIGGER trg_mr_no_delete BEFORE DELETE ON public.model_registry FOR EACH ROW EXECUTE FUNCTION public.mudra_prevent_mutation()",
        create_sql="CREATE TRIGGER trg_mr_no_delete BEFORE DELETE ON public.model_registry FOR EACH ROW EXECUTE FUNCTION public.mudra_prevent_mutation();",
    ),
    TriggerSpec(
        name="trg_mr_no_truncate",
        table_name="model_registry",
        func_name="mudra_prevent_mutation",
        expected_tgtype=34,
        deferrable=False,
        initially_deferred=False,
        expected_triggerdef="CREATE TRIGGER trg_mr_no_truncate BEFORE TRUNCATE ON public.model_registry FOR EACH STATEMENT EXECUTE FUNCTION public.mudra_prevent_mutation()",
        create_sql="CREATE TRIGGER trg_mr_no_truncate BEFORE TRUNCATE ON public.model_registry FOR EACH STATEMENT EXECUTE FUNCTION public.mudra_prevent_mutation();",
    ),
    TriggerSpec(
        name="trg_mle_no_update_delete",
        table_name="model_lifecycle_events",
        func_name="mudra_prevent_mutation",
        expected_tgtype=27,
        deferrable=False,
        initially_deferred=False,
        expected_triggerdef="CREATE TRIGGER trg_mle_no_update_delete BEFORE DELETE OR UPDATE ON public.model_lifecycle_events FOR EACH ROW EXECUTE FUNCTION public.mudra_prevent_mutation()",
        create_sql="CREATE TRIGGER trg_mle_no_update_delete BEFORE UPDATE OR DELETE ON public.model_lifecycle_events FOR EACH ROW EXECUTE FUNCTION public.mudra_prevent_mutation();",
    ),
    TriggerSpec(
        name="trg_mle_no_truncate",
        table_name="model_lifecycle_events",
        func_name="mudra_prevent_mutation",
        expected_tgtype=34,
        deferrable=False,
        initially_deferred=False,
        expected_triggerdef="CREATE TRIGGER trg_mle_no_truncate BEFORE TRUNCATE ON public.model_lifecycle_events FOR EACH STATEMENT EXECUTE FUNCTION public.mudra_prevent_mutation()",
        create_sql="CREATE TRIGGER trg_mle_no_truncate BEFORE TRUNCATE ON public.model_lifecycle_events FOR EACH STATEMENT EXECUTE FUNCTION public.mudra_prevent_mutation();",
    ),
    TriggerSpec(
        name="trg_model_predictions_no_delete",
        table_name="model_predictions",
        func_name="mudra_prevent_mutation",
        expected_tgtype=11,
        deferrable=False,
        initially_deferred=False,
        expected_triggerdef="CREATE TRIGGER trg_model_predictions_no_delete BEFORE DELETE ON public.model_predictions FOR EACH ROW EXECUTE FUNCTION public.mudra_prevent_mutation()",
        create_sql="CREATE TRIGGER trg_model_predictions_no_delete BEFORE DELETE ON public.model_predictions FOR EACH ROW EXECUTE FUNCTION public.mudra_prevent_mutation();",
    ),
    TriggerSpec(
        name="trg_model_predictions_no_truncate",
        table_name="model_predictions",
        func_name="mudra_prevent_mutation",
        expected_tgtype=34,
        deferrable=False,
        initially_deferred=False,
        expected_triggerdef="CREATE TRIGGER trg_model_predictions_no_truncate BEFORE TRUNCATE ON public.model_predictions FOR EACH STATEMENT EXECUTE FUNCTION public.mudra_prevent_mutation()",
        create_sql="CREATE TRIGGER trg_model_predictions_no_truncate BEFORE TRUNCATE ON public.model_predictions FOR EACH STATEMENT EXECUTE FUNCTION public.mudra_prevent_mutation();",
    ),
    TriggerSpec(
        name="trg_mr_guard_write",
        table_name="model_registry",
        func_name="mudra_guard_model_registry_write",
        expected_tgtype=23,
        deferrable=False,
        initially_deferred=False,
        expected_triggerdef="CREATE TRIGGER trg_mr_guard_write BEFORE INSERT OR UPDATE ON public.model_registry FOR EACH ROW EXECUTE FUNCTION public.mudra_guard_model_registry_write()",
        create_sql="CREATE TRIGGER trg_mr_guard_write BEFORE INSERT OR UPDATE ON public.model_registry FOR EACH ROW EXECUTE FUNCTION public.mudra_guard_model_registry_write();",
    ),
    TriggerSpec(
        name="trg_mle_validate_insert",
        table_name="model_lifecycle_events",
        func_name="mudra_validate_lifecycle_event_insert",
        expected_tgtype=7,
        deferrable=False,
        initially_deferred=False,
        expected_triggerdef="CREATE TRIGGER trg_mle_validate_insert BEFORE INSERT ON public.model_lifecycle_events FOR EACH ROW EXECUTE FUNCTION public.mudra_validate_lifecycle_event_insert()",
        create_sql="CREATE TRIGGER trg_mle_validate_insert BEFORE INSERT ON public.model_lifecycle_events FOR EACH ROW EXECUTE FUNCTION public.mudra_validate_lifecycle_event_insert();",
    ),
    TriggerSpec(
        name="trg_mr_verify_lifecycle_sync",
        table_name="model_registry",
        func_name="mudra_verify_registry_lifecycle_sync",
        expected_tgtype=21,
        deferrable=True,
        initially_deferred=True,
        expected_triggerdef="CREATE CONSTRAINT TRIGGER trg_mr_verify_lifecycle_sync AFTER INSERT OR UPDATE ON public.model_registry DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION public.mudra_verify_registry_lifecycle_sync()",
        create_sql="CREATE CONSTRAINT TRIGGER trg_mr_verify_lifecycle_sync AFTER INSERT OR UPDATE ON public.model_registry DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION public.mudra_verify_registry_lifecycle_sync();",
    ),
    TriggerSpec(
        name="trg_model_predictions_validate_insert",
        table_name="model_predictions",
        func_name="mudra_validate_prediction_insert",
        expected_tgtype=7,
        deferrable=False,
        initially_deferred=False,
        expected_triggerdef="CREATE TRIGGER trg_model_predictions_validate_insert BEFORE INSERT ON public.model_predictions FOR EACH ROW EXECUTE FUNCTION public.mudra_validate_prediction_insert()",
        create_sql="CREATE TRIGGER trg_model_predictions_validate_insert BEFORE INSERT ON public.model_predictions FOR EACH ROW EXECUTE FUNCTION public.mudra_validate_prediction_insert();",
    ),
    TriggerSpec(
        name="trg_model_predictions_immutability",
        table_name="model_predictions",
        func_name="mudra_guard_prediction_immutability",
        expected_tgtype=19,
        deferrable=False,
        initially_deferred=False,
        expected_triggerdef="CREATE TRIGGER trg_model_predictions_immutability BEFORE UPDATE ON public.model_predictions FOR EACH ROW EXECUTE FUNCTION public.mudra_guard_prediction_immutability()",
        create_sql="CREATE TRIGGER trg_model_predictions_immutability BEFORE UPDATE ON public.model_predictions FOR EACH ROW EXECUTE FUNCTION public.mudra_guard_prediction_immutability();",
    ),
]


def _normalize_triggerdef_for_compare(trig_def: str) -> str:
    """Normalizes pg_get_triggerdef output across PG versions (EXECUTE FUNCTION vs EXECUTE PROCEDURE, schema qualification)."""
    s = normalize_sql(trig_def)
    s = s.replace("execute procedure ", "execute function ")
    s = s.replace(" on model_", " on public.model_")
    s = s.replace(" on prediction_", " on public.prediction_")
    s = s.replace("execute function mudra_", "execute function public.mudra_")
    return s


def ensure_canonical_function(session_or_conn: Any, spec: FunctionSpec) -> bool:
    """
    Identifies trigger function by schema ('public'), name, and exact argument signature ('').
    Fails closed with SchemaDriftError on unexpected overloads or signature/body drift.
    Returns True if created, False if already present and matching.
    """
    rows = session_or_conn.execute(
        text("""
            SELECT p.pronargs AS nargs,
                   pg_get_function_identity_arguments(p.oid) AS arg_sig,
                   p.prorettype::regtype::text AS rettype,
                   l.lanname AS lang,
                   p.prosrc AS prosrc
            FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            JOIN pg_language l ON l.oid = p.prolang
            WHERE n.nspname = :schema_name AND p.proname = :proname
        """),
        {"schema_name": spec.schema_name, "proname": spec.name},
    ).mappings().all()

    if len(rows) == 0:
        session_or_conn.execute(text(spec.create_sql))
        return True

    if len(rows) > 1:
        raise SchemaDriftError(
            f"Unexpected function overloads ({len(rows)} definitions) found for "
            f"{spec.schema_name}.{spec.name}."
        )

    row = rows[0]
    if (
        int(row["nargs"]) != 0
        or str(row["arg_sig"]).strip() != ""
        or str(row["rettype"]).lower() != "trigger"
        or str(row["lang"]).lower() != "plpgsql"
        or normalize_sql(row["prosrc"]) != normalize_sql(spec.expected_prosrc)
    ):
        raise SchemaDriftError(
            f"Function {spec.schema_name}.{spec.name}({row['arg_sig']}) exists with drifted "
            f"signature, return type ({row['rettype']}), language ({row['lang']}), or body."
        )
    return False


def ensure_canonical_trigger(session_or_conn: Any, spec: TriggerSpec) -> bool:
    """
    Checks pg_trigger before creating a trigger; skips if identical, aborts with SchemaDriftError on drift.
    Returns True if created, False if already present and matching.
    """
    row = session_or_conn.execute(
        text("""
            SELECT t.tgenabled AS tgenabled,
                   t.tgtype AS tgtype,
                   t.tgdeferrable AS tgdeferrable,
                   t.tginitdeferred AS tginitdeferred,
                   n_proc.nspname AS func_schema,
                   p.proname AS func_name,
                   pg_get_triggerdef(t.oid, true) AS trig_def
            FROM pg_trigger t
            JOIN pg_class c ON c.oid = t.tgrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_proc p ON p.oid = t.tgfoid
            JOIN pg_namespace n_proc ON n_proc.oid = p.pronamespace
            WHERE n.nspname = 'public'
              AND c.relname = :table_name
              AND t.tgname = :trig_name
              AND NOT t.tgisinternal
        """),
        {"table_name": spec.table_name, "trig_name": spec.name},
    ).mappings().first()

    if row is None:
        session_or_conn.execute(text(spec.create_sql))
        return True

    if (
        str(row["tgenabled"]) != "O"
        or int(row["tgtype"]) != spec.expected_tgtype
        or bool(row["tgdeferrable"]) != spec.deferrable
        or bool(row["tginitdeferred"]) != spec.initially_deferred
        or str(row["func_schema"]) != "public"
        or str(row["func_name"]) != spec.func_name
        or _normalize_triggerdef_for_compare(row["trig_def"])
        != _normalize_triggerdef_for_compare(spec.expected_triggerdef)
    ):
        raise SchemaDriftError(
            f"Trigger {spec.name} on public.{spec.table_name} exists with drifted definition."
        )
    return False


def _classify_column_type(raw_type: str) -> str:
    """Maps dialect-specific column type names to canonical type categories."""
    t = str(raw_type).lower()
    if "int" in t or "serial" in t:
        return "integer"
    if "double" in t or "float" in t or "real" in t or "numeric" in t:
        return "float"
    if "char" in t:
        return "varchar"
    if "text" in t:
        return "text"
    if "bool" in t:
        return "boolean"
    if "json" in t:
        return "json"
    if "timestamp" in t or "datetime" in t:
        return "timestamp"
    return t


def verify_db_schema(
    target_engine_or_conn: Optional[Any] = None,
    require_seed_models: bool = False,
) -> Dict[str, Any]:
    """
    Strictly read-only database schema verifier. Performs zero DDL/DML writes.
    Verifies all 9 tables, columns, foreign keys, constraints, explicit indexes,
    trigger functions (by schema, name, and zero-arg signature), and attached triggers.
    Raises SchemaDriftError on any missing object or definition drift.
    """
    target = target_engine_or_conn or engine

    @contextmanager
    def _acquire_conn():
        if isinstance(target, Engine):
            with target.connect() as c:
                yield c
        elif isinstance(target, Session):
            yield target.connection()
        else:
            yield target

    with _acquire_conn() as conn:
        dialect_name = conn.dialect.name
        inspector = inspect(conn)
        existing_tables = set(inspector.get_table_names())

        # 1. Verify all 9 tables and their exact column sets, types, and nullability
        for tbl_name, expected_cols in CANONICAL_TABLE_COLUMNS.items():
            if tbl_name not in existing_tables:
                raise SchemaDriftError(f"Required table '{tbl_name}' is missing from database schema.")

            live_cols = inspector.get_columns(tbl_name)
            live_col_map = {c["name"]: c for c in live_cols}
            expected_col_names = [c[0] for c in expected_cols]

            if set(live_col_map.keys()) != set(expected_col_names):
                missing = set(expected_col_names) - set(live_col_map.keys())
                extra = set(live_col_map.keys()) - set(expected_col_names)
                raise SchemaDriftError(
                    f"Column set drift on table '{tbl_name}': missing={sorted(missing)}, unexpected={sorted(extra)}."
                )

            for col_name, exp_type, exp_nullable in expected_cols:
                c_info = live_col_map[col_name]
                live_type = _classify_column_type(str(c_info["type"]))
                if live_type != exp_type:
                    raise SchemaDriftError(
                        f"Column type drift on '{tbl_name}.{col_name}': expected {exp_type}, got {live_type} ({c_info['type']})."
                    )
                if bool(c_info.get("nullable", True)) != exp_nullable:
                    raise SchemaDriftError(
                        f"Column nullability drift on '{tbl_name}.{col_name}': expected nullable={exp_nullable}, got {c_info.get('nullable')}."
                    )

        if "sqlite" in dialect_name:
            # SQLite unit-test mode: verify index existence on SQLite tables
            for idx_name, tbl_name, is_unique, key_cols, _ in CANONICAL_INDEXES:
                tbl_indexes = {i["name"]: i for i in inspector.get_indexes(tbl_name)}
                if idx_name not in tbl_indexes:
                    raise SchemaDriftError(f"Required index '{idx_name}' on '{tbl_name}' is missing.")
            return {"status": "verified", "dialect": "sqlite", "tables": len(CANONICAL_TABLE_COLUMNS)}

        # PostgreSQL Catalog Verification (Zero DDL)
        # 2. Verify all 10 named Foreign Keys
        fk_rows = conn.execute(
            text("""
                SELECT c.conname AS conname,
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
                WHERE n.nspname = 'public' AND c.contype = 'f'
            """)
        ).mappings().all()
        fk_map = {r["conname"]: r for r in fk_rows}

        for fk_name, src_tbl, src_col, tgt_tbl, tgt_col in CANONICAL_FOREIGN_KEYS:
            if fk_name not in fk_map:
                raise SchemaDriftError(f"Required foreign key '{fk_name}' on '{src_tbl}' is missing.")
            fk = fk_map[fk_name]
            if (
                fk["src_table"] != src_tbl
                or fk["src_col"] != src_col
                or fk["tgt_table"] != tgt_tbl
                or fk["tgt_col"] != tgt_col
                or str(fk["upd_type"]) != "a"
                or str(fk["del_type"]) != "a"
            ):
                raise SchemaDriftError(f"Foreign key '{fk_name}' on '{src_tbl}' exhibits catalog definition drift.")

        # 3. Verify all 7 named UNIQUE and CHECK constraints
        con_rows = conn.execute(
            text("""
                SELECT c.conname AS conname,
                       rel.relname AS table_name,
                       c.contype AS contype,
                       c.convalidated AS convalidated,
                       ARRAY(
                           SELECT a.attname
                           FROM unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord)
                           JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
                           ORDER BY k.ord
                       ) AS col_names,
                       pg_get_constraintdef(c.oid, true) AS condef
                FROM pg_constraint c
                JOIN pg_class rel ON rel.oid = c.conrelid
                JOIN pg_namespace n ON n.oid = rel.relnamespace
                WHERE n.nspname = 'public' AND c.contype IN ('c', 'u')
            """)
        ).mappings().all()
        con_map = {(r["table_name"], r["conname"]): r for r in con_rows}

        for con_name, tbl_name, exp_contype, exp_cols, req_tokens in CANONICAL_CONSTRAINTS:
            key = (tbl_name, con_name)
            if key not in con_map:
                raise SchemaDriftError(f"Required constraint '{con_name}' on '{tbl_name}' is missing.")
            con = con_map[key]
            if str(con["contype"]).strip() != exp_contype or not bool(con["convalidated"]):
                raise SchemaDriftError(f"Constraint '{con_name}' on '{tbl_name}' has invalid type or is unvalidated.")
            live_cols_list = list(con["col_names"] or [])
            if exp_contype == "u" and live_cols_list != exp_cols:
                raise SchemaDriftError(
                    f"Unique constraint '{con_name}' column drift: expected {exp_cols}, got {live_cols_list}."
                )
            if exp_contype == "c" and set(live_cols_list) != set(exp_cols):
                raise SchemaDriftError(
                    f"Check constraint '{con_name}' referenced column drift: expected {sorted(exp_cols)}, got {sorted(live_cols_list)}."
                )
            norm_condef = normalize_pg_predicate(con["condef"])
            for tok in req_tokens:
                if normalize_pg_predicate(tok) not in norm_condef:
                    raise SchemaDriftError(
                        f"Constraint '{con_name}' on '{tbl_name}' is missing required clause '{tok}' in '{con['condef']}'."
                    )

        # 4. Verify all 12 Explicit Indexes
        idx_rows = conn.execute(
            text("""
                SELECT c_idx.relname AS index_name,
                       c_tbl.relname AS table_name,
                       i.indisunique AS is_unique,
                       i.indisvalid AS is_valid,
                       ARRAY(
                           SELECT a.attname
                           FROM unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord)
                           JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
                           ORDER BY k.ord
                       ) AS key_cols,
                       pg_get_expr(i.indpred, i.indrelid) AS pred_expr
                FROM pg_index i
                JOIN pg_class c_idx ON c_idx.oid = i.indexrelid
                JOIN pg_class c_tbl ON c_tbl.oid = i.indrelid
                JOIN pg_namespace n ON n.oid = c_tbl.relnamespace
                WHERE n.nspname = 'public'
            """)
        ).mappings().all()
        idx_map = {r["index_name"]: r for r in idx_rows}

        for idx_name, tbl_name, exp_unique, exp_cols, exp_pred in CANONICAL_INDEXES:
            if idx_name not in idx_map:
                raise SchemaDriftError(f"Required index '{idx_name}' on '{tbl_name}' is missing.")
            idx = idx_map[idx_name]
            if (
                idx["table_name"] != tbl_name
                or not bool(idx["is_valid"])
                or bool(idx["is_unique"]) != exp_unique
                or list(idx["key_cols"] or []) != exp_cols
                or normalize_pg_predicate(idx["pred_expr"]) != normalize_pg_predicate(exp_pred)
            ):
                raise SchemaDriftError(f"Index '{idx_name}' on '{tbl_name}' exhibits catalog definition drift.")

        # 5. Verify all 6 Trigger Functions (by schema, name, zero-arg signature, and body)
        for f_spec in CANONICAL_FUNCTIONS:
            f_rows = conn.execute(
                text("""
                    SELECT p.pronargs AS nargs,
                           pg_get_function_identity_arguments(p.oid) AS arg_sig,
                           p.prorettype::regtype::text AS rettype,
                           l.lanname AS lang,
                           p.prosrc AS prosrc
                    FROM pg_proc p
                    JOIN pg_namespace n ON n.oid = p.pronamespace
                    JOIN pg_language l ON l.oid = p.prolang
                    WHERE n.nspname = :schema_name AND p.proname = :proname
                """),
                {"schema_name": f_spec.schema_name, "proname": f_spec.name},
            ).mappings().all()

            if len(f_rows) == 0:
                raise SchemaDriftError(f"Required trigger function '{f_spec.schema_name}.{f_spec.name}()' is missing.")
            if len(f_rows) > 1:
                raise SchemaDriftError(
                    f"Unexpected function overloads ({len(f_rows)}) found for '{f_spec.schema_name}.{f_spec.name}'."
                )
            f_row = f_rows[0]
            if (
                int(f_row["nargs"]) != 0
                or str(f_row["arg_sig"]).strip() != ""
                or str(f_row["rettype"]).lower() != "trigger"
                or str(f_row["lang"]).lower() != "plpgsql"
                or normalize_sql(f_row["prosrc"]) != normalize_sql(f_spec.expected_prosrc)
            ):
                raise SchemaDriftError(f"Trigger function '{f_spec.schema_name}.{f_spec.name}' exhibits body or signature drift.")

        # 6. Verify all 11 Attached Triggers
        for t_spec in CANONICAL_TRIGGERS:
            t_row = conn.execute(
                text("""
                    SELECT t.tgenabled AS tgenabled,
                           t.tgtype AS tgtype,
                           t.tgdeferrable AS tgdeferrable,
                           t.tginitdeferred AS tginitdeferred,
                           n_proc.nspname AS func_schema,
                           p.proname AS func_name,
                           pg_get_triggerdef(t.oid, true) AS trig_def
                    FROM pg_trigger t
                    JOIN pg_class c ON c.oid = t.tgrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    JOIN pg_proc p ON p.oid = t.tgfoid
                    JOIN pg_namespace n_proc ON n_proc.oid = p.pronamespace
                    WHERE n.nspname = 'public'
                      AND c.relname = :table_name
                      AND t.tgname = :trig_name
                      AND NOT t.tgisinternal
                """),
                {"table_name": t_spec.table_name, "trig_name": t_spec.name},
            ).mappings().first()

            if t_row is None:
                raise SchemaDriftError(f"Required trigger '{t_spec.name}' on '{t_spec.table_name}' is missing.")
            if (
                str(t_row["tgenabled"]) != "O"
                or int(t_row["tgtype"]) != t_spec.expected_tgtype
                or bool(t_row["tgdeferrable"]) != t_spec.deferrable
                or bool(t_row["tginitdeferred"]) != t_spec.initially_deferred
                or str(t_row["func_schema"]) != "public"
                or str(t_row["func_name"]) != t_spec.func_name
                or _normalize_triggerdef_for_compare(t_row["trig_def"])
                != _normalize_triggerdef_for_compare(t_spec.expected_triggerdef)
            ):
                raise SchemaDriftError(f"Trigger '{t_spec.name}' on '{t_spec.table_name}' exhibits catalog definition drift.")

        # 7. Optional seed model verification
        if require_seed_models:
            seed_rows = conn.execute(
                text("SELECT model_version, current_status, lifecycle_seq FROM public.model_registry")
            ).mappings().all()
            seed_map = {r["model_version"]: r for r in seed_rows}
            if "xgb_onnx_v1.0.0" not in seed_map or seed_map["xgb_onnx_v1.0.0"]["current_status"] != "ACTIVE":
                raise SchemaDriftError("Baseline model 'xgb_onnx_v1.0.0' is missing or not ACTIVE in model_registry.")
            if "p1_controlled_candidate_v1" not in seed_map or seed_map["p1_controlled_candidate_v1"]["current_status"] != "REJECTED":
                raise SchemaDriftError("Candidate model 'p1_controlled_candidate_v1' is missing or not REJECTED in model_registry.")

        return {
            "status": "verified",
            "dialect": "postgresql",
            "tables": len(CANONICAL_TABLE_COLUMNS),
            "foreign_keys": len(CANONICAL_FOREIGN_KEYS),
            "constraints": len(CANONICAL_CONSTRAINTS),
            "indexes": len(CANONICAL_INDEXES),
            "functions": len(CANONICAL_FUNCTIONS),
            "triggers": len(CANONICAL_TRIGGERS),
        }


# =============================================================================
# CONNECTION & SESSION MANAGEMENT
# =============================================================================

def get_database_url() -> str:
    """
    Resolves the active DATABASE_URL in order of precedence:
    1. OS environment variable (GitHub Actions / Cloud container)
    2. Streamlit st.secrets (if running inside Streamlit runtime)
    3. Direct read of .streamlit/secrets.toml via tomllib (CLI workers)
    4. Local SQLite file fallback
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return _normalize_db_url(db_url)

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "neondb" in st.secrets and "DATABASE_URL" in st.secrets["neondb"]:
                return _normalize_db_url(st.secrets["neondb"]["DATABASE_URL"])
            if "DATABASE_URL" in st.secrets:
                return _normalize_db_url(st.secrets["DATABASE_URL"])
    except Exception:
        pass

    if SECRETS_FILE.exists():
        try:
            data = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            if "neondb" in data and "DATABASE_URL" in data["neondb"]:
                return _normalize_db_url(data["neondb"]["DATABASE_URL"])
            if "DATABASE_URL" in data:
                return _normalize_db_url(data["DATABASE_URL"])
        except Exception:
            pass

    fallback_path = Path("data/mudra_operational.db")
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{fallback_path.as_posix()}"


def _normalize_db_url(url: str) -> str:
    """Normalizes PostgreSQL URL for SQLAlchemy + psycopg2 compatibility."""
    clean_url = url.strip()
    if clean_url.startswith("postgres://"):
        return clean_url.replace("postgres://", "postgresql+psycopg2://", 1)
    if clean_url.startswith("postgresql://") and not clean_url.startswith("postgresql+"):
        return clean_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return clean_url


def create_db_engine(db_url: Optional[str] = None) -> Engine:
    """Creates a configured SQLAlchemy Engine."""
    url = db_url or get_database_url()

    if "sqlite" in url:
        eng = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        try:
            Base.metadata.create_all(bind=eng)
        except Exception:
            pass
        return eng

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,
        connect_args={"connect_timeout": 10},
        echo=False,
    )


# Global Engine and SessionLocal
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager providing a transactional database session with auto-rollback.
    """
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


_TICK_ID_COLUMN_CACHE: Dict[int, bool] = {}


def has_tick_id_column(session: Session) -> bool:
    """
    Reliably verifies whether the underlying `paper_portfolio` table contains
    the `tick_id` column on disk, caching the result per engine instance.
    """
    try:
        bind = session.get_bind()
        bind_id = id(bind)
        if bind_id in _TICK_ID_COLUMN_CACHE:
            return _TICK_ID_COLUMN_CACHE[bind_id]

        inspector = inspect(bind)
        if inspector.has_table("paper_portfolio"):
            cols = {c["name"] for c in inspector.get_columns("paper_portfolio")}
            present = "tick_id" in cols
            _TICK_ID_COLUMN_CACHE[bind_id] = present
            return present
    except Exception as exc:
        logger.warning(f"[DB SCHEMA] Failed to inspect paper_portfolio table columns: {exc}")
    return False


# Backward-compatible 18-column paper_portfolio read hooks (defers tick_id when absent on disk)
@event.listens_for(Session, "do_orm_execute")
def _defer_missing_tick_id_on_legacy_portfolio(orm_execute_state: Any) -> None:
    if (
        orm_execute_state.is_select
        and not orm_execute_state.is_column_load
        and not orm_execute_state.is_relationship_load
    ):
        try:
            mappers = orm_execute_state.all_mappers
            if any(m.class_ is PaperPortfolio for m in mappers):
                if not has_tick_id_column(orm_execute_state.session):
                    orm_execute_state.statement = orm_execute_state.statement.options(
                        defer(PaperPortfolio.tick_id)
                    )
        except Exception:
            pass


@event.listens_for(PaperPortfolio, "load")
def _populate_deferred_tick_id_as_none(target: PaperPortfolio, context: Any) -> None:
    if "tick_id" not in target.__dict__:
        attributes.set_committed_value(target, "tick_id", None)


def init_db(target_engine: Optional[Engine] = None) -> None:
    """
    Creates tables in the database if they do not exist and applies non-destructive schema evolutions.
    Fails closed with actionable diagnostics if required migrations fail.
    Note: Explicit migration for PostgreSQL production/disposable environments is handled by scripts/migrate_db.py.
    """
    target = target_engine or engine
    Base.metadata.create_all(bind=target)

    with target.connect() as conn:
        inspector = inspect(conn)

        # 1. Non-destructive migration for paper_portfolio.tick_id
        if inspector.has_table("paper_portfolio"):
            cols = {c["name"] for c in inspector.get_columns("paper_portfolio")}
            if "tick_id" not in cols:
                try:
                    if "postgresql" in target.dialect.name:
                        conn.execute(text("ALTER TABLE paper_portfolio ADD COLUMN IF NOT EXISTS tick_id INTEGER REFERENCES market_ticks(id);"))
                    elif "sqlite" in target.dialect.name:
                        conn.execute(text("ALTER TABLE paper_portfolio ADD COLUMN tick_id INTEGER REFERENCES market_ticks(id);"))
                    conn.commit()
                    logger.info("[DB MIGRATION] Added 'tick_id' column to paper_portfolio.")
                except Exception as exc:
                    logger.error(f"[DB MIGRATION CRITICAL] Failed to add 'tick_id' column to paper_portfolio: {exc}")
                    raise RuntimeError(f"Database migration failed: unable to add required column 'tick_id' to paper_portfolio: {exc}") from exc

            existing_indexes = {idx["name"] for idx in inspector.get_indexes("paper_portfolio")}
            if "uq_paper_portfolio_tick_id" not in existing_indexes:
                try:
                    conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_portfolio_tick_id ON paper_portfolio(tick_id) WHERE tick_id IS NOT NULL;"))
                    conn.commit()
                except Exception as exc:
                    logger.warning(f"[DB MIGRATION] Could not create partial unique index 'uq_paper_portfolio_tick_id': {exc}")

            _TICK_ID_COLUMN_CACHE[id(target)] = True

        # 2. Safe duplicate-detection prior to creating unique index on market_ticks(timestamp)
        if inspector.has_table("market_ticks"):
            existing_mt_indexes = {idx["name"] for idx in inspector.get_indexes("market_ticks")}
            if "uq_market_ticks_timestamp" not in existing_mt_indexes:
                try:
                    dup_rows = conn.execute(
                        text("SELECT timestamp, COUNT(*) as cnt FROM market_ticks GROUP BY timestamp HAVING COUNT(*) > 1")
                    ).fetchall()
                    if dup_rows:
                        dup_summary = [(str(r[0]), r[1]) for r in dup_rows[:5]]
                        logger.warning(
                            f"[DB MIGRATION] Cannot create unique index 'uq_market_ticks_timestamp': found {len(dup_rows)} duplicate timestamp group(s) "
                            f"(examples: {dup_summary}). Preserving all existing records; uniqueness constraint installation skipped."
                        )
                    else:
                        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_market_ticks_timestamp ON market_ticks(timestamp);"))
                        conn.commit()
                        logger.info("[DB MIGRATION] Verified 0 duplicate timestamps; created unique index 'uq_market_ticks_timestamp'.")
                except Exception as exc:
                    logger.warning(f"[DB MIGRATION] Failed during market_ticks timestamp uniqueness check or index creation: {exc}")


def test_connection() -> Tuple[bool, str, float]:
    """
    Tests database connectivity and measures round-trip latency.
    Dynamically handles PostgreSQL (SELECT version();) and SQLite (SELECT sqlite_version();).
    Returns: (is_success, server_version_or_error, latency_ms)
    """
    start_time = time.perf_counter()
    try:
        with engine.connect() as conn:
            is_sqlite = engine.dialect.name == "sqlite"
            query_str = "SELECT sqlite_version();" if is_sqlite else "SELECT version();"
            result = conn.execute(text(query_str)).fetchone()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            version_str = f"SQLite {result[0]}" if (is_sqlite and result) else (str(result[0]) if result else "Unknown")
            return True, version_str, latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return False, str(exc), latency_ms


if __name__ == "__main__":
    print("[INFO] Testing Mudra database connectivity and latency...")
    success, info, latency = test_connection()
    status = "SUCCESS" if success else "FAILED"
    print(f"[{status}] Connected: {success} | Version: {info} | Latency: {latency:.2f}ms")