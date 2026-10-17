"""
Prediction-Failure Investigation Service for Mudra Quantitative Platform.

Provides structured diagnostic tracing, headline-to-tick association,
causal hypothesis tracking, and evidence-gated training approval.
Enforces fail-closed validation for diagnostic confidence, verification status,
and causal ground-truth confirmation before any example can be approved for training.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import (
    MarketTick,
    ModelPrediction,
    ModelRegistry,
    PredictionInvestigation,
    Reconciliation,
)

logger = logging.getLogger(__name__)

ALLOWED_VERIFICATION_STATUSES = frozenset(
    {
        "UNVERIFIED_HYPOTHESIS",
        "CONFIRMED_CAUSE",
        "REFUTED",
        "APPROVED_FOR_TRAINING",
    }
)

__all__ = [
    "ALLOWED_VERIFICATION_STATUSES",
    "record_prediction_investigation",
    "get_prediction_investigations",
    "get_prediction_investigation_details",
    "update_prediction_investigation_status",
    "get_training_approved_investigations",
]


def _validate_diagnostic_confidence(confidence: Optional[float]) -> None:
    """Validates that diagnostic confidence is None or a finite real float within [0.0, 1.0]."""
    if confidence is not None:
        try:
            val = float(confidence)
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Diagnostic confidence must be a numeric float, got {type(confidence).__name__}."
            ) from err

        if not math.isfinite(val):
            raise ValueError(
                f"Diagnostic confidence must be a finite float, got non-finite value: {val}."
            )
        if val < 0.0 or val > 1.0:
            raise ValueError(
                f"Diagnostic confidence must be between 0.0 and 1.0, got {val}."
            )


def _validate_verification_status(status: str) -> None:
    """Validates that verification_status belongs to the canonical enum set."""
    if not isinstance(status, str) or status.strip() not in ALLOWED_VERIFICATION_STATUSES:
        raise ValueError(
            f"Invalid verification_status '{status}'. Must be one of: "
            f"{sorted(list(ALLOWED_VERIFICATION_STATUSES))}."
        )


def _normalize_datetime(dt_val: Optional[Union[str, datetime]]) -> Optional[datetime]:
    """Parses and normalizes input datetimes to UTC timezone-aware datetime."""
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        return dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError) as err:
        raise ValueError(f"Invalid ISO datetime string for publication timestamp: {dt_val}") from err


def record_prediction_investigation(
    tick_id: int,
    model_version: str,
    headline: str,
    suspected_cause_category: str,
    hypothesis_notes: str,
    supporting_evidence_json: Optional[Dict[str, Any]] = None,
    diagnostic_confidence: Optional[float] = None,
    unresolved_questions: Optional[str] = None,
    verification_status: str = "UNVERIFIED_HYPOTHESIS",
    headline_source_url: Optional[str] = None,
    headline_published_at: Optional[Union[str, datetime]] = None,
    prediction_id: Optional[int] = None,
    author: str = "SYSTEM",
    session: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Records a structured prediction-failure investigation linked to a market tick.

    Validates inputs fail-closed: rejects non-finite confidence, unsupported status,
    and missing foreign key references.
    Initializes immutable revision history with the initial hypothesis creation event.
    """
    _validate_verification_status(verification_status)
    _validate_diagnostic_confidence(diagnostic_confidence)

    if not headline or not headline.strip():
        raise ValueError("Investigation headline cannot be empty.")
    if not suspected_cause_category or not suspected_cause_category.strip():
        raise ValueError("suspected_cause_category cannot be empty.")
    if not hypothesis_notes or not hypothesis_notes.strip():
        raise ValueError("hypothesis_notes cannot be empty.")
    if not model_version or not model_version.strip():
        raise ValueError("model_version cannot be empty.")

    pub_at = _normalize_datetime(headline_published_at)

    def _execute(sess: Session) -> Dict[str, Any]:
        # Verify tick_id exists
        tick = sess.execute(select(MarketTick).where(MarketTick.id == tick_id)).scalar_one_or_none()
        if tick is None:
            raise ValueError(f"Cannot record investigation: MarketTick with ID {tick_id} does not exist.")

        # Verify prediction_id exists if supplied
        if prediction_id is not None:
            pred = sess.execute(
                select(ModelPrediction).where(ModelPrediction.id == prediction_id)
            ).scalar_one_or_none()
            if pred is None:
                raise ValueError(
                    f"Cannot record investigation: ModelPrediction with ID {prediction_id} does not exist."
                )

        now_utc = datetime.now(timezone.utc)
        initial_history = [
            {
                "timestamp": now_utc.isoformat(),
                "author": author,
                "action": "RECORDED",
                "verification_status": verification_status,
                "diagnostic_confidence": diagnostic_confidence,
                "suspected_cause_category": suspected_cause_category,
                "notes": hypothesis_notes[:250],
            }
        ]

        inv = PredictionInvestigation(
            tick_id=tick_id,
            prediction_id=prediction_id,
            model_version=model_version,
            headline=headline.strip(),
            headline_source_url=headline_source_url.strip() if headline_source_url else None,
            headline_published_at=pub_at,
            suspected_cause_category=suspected_cause_category.strip(),
            hypothesis_notes=hypothesis_notes.strip(),
            supporting_evidence_json=supporting_evidence_json or {},
            diagnostic_confidence=float(diagnostic_confidence) if diagnostic_confidence is not None else None,
            unresolved_questions=unresolved_questions.strip() if unresolved_questions else None,
            verification_status=verification_status,
            revision_history_json=initial_history,
            created_at=now_utc,
            updated_at=now_utc,
        )
        sess.add(inv)
        sess.flush()
        return inv.to_dict()

    if session is not None:
        return _execute(session)
    with get_db() as new_session:
        return _execute(new_session)


def get_prediction_investigations(
    limit: int = 50,
    model_version: Optional[str] = None,
    verification_status: Optional[str] = None,
    headline_query: Optional[str] = None,
    tick_id: Optional[int] = None,
    session: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    """
    Fetches prediction investigations ordered chronologically descending.
    Supports headline keyword search, status filtering, and model version filtering.
    """
    def _execute(sess: Session) -> List[Dict[str, Any]]:
        stmt = select(PredictionInvestigation).order_by(desc(PredictionInvestigation.id))

        if model_version and model_version != "All Models":
            stmt = stmt.where(PredictionInvestigation.model_version == model_version)
        if verification_status:
            _validate_verification_status(verification_status)
            stmt = stmt.where(PredictionInvestigation.verification_status == verification_status)
        if headline_query and headline_query.strip():
            stmt = stmt.where(PredictionInvestigation.headline.ilike(f"%{headline_query.strip()}%"))
        if tick_id is not None:
            stmt = stmt.where(PredictionInvestigation.tick_id == tick_id)

        stmt = stmt.limit(limit)
        results = sess.execute(stmt).scalars().all()
        return [r.to_dict() for r in results]

    try:
        if session is not None:
            return _execute(session)
        with get_db() as new_session:
            return _execute(new_session)
    except Exception as exc:
        logger.warning("Failed to fetch prediction investigations: error_type=%s", type(exc).__name__)
        return []


def get_prediction_investigation_details(
    investigation_id: int,
    session: Optional[Session] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetches an investigation and traces its complete causal chain end-to-end:
    Investigation -> MarketTick (timestamp, OHLCV, original features, prediction)
                  -> Reconciliation (actual outcome at T+1, error, attribution tag)
                  -> ModelPrediction (if linked).
    """
    def _execute(sess: Session) -> Optional[Dict[str, Any]]:
        inv = sess.execute(
            select(PredictionInvestigation).where(PredictionInvestigation.id == investigation_id)
        ).scalar_one_or_none()
        if inv is None:
            return None

        inv_dict = inv.to_dict()

        # Load linked MarketTick
        tick = sess.execute(
            select(MarketTick).where(MarketTick.id == inv.tick_id)
        ).scalar_one_or_none()
        tick_dict = tick.to_dict() if tick else None

        # Load linked delayed Reconciliation (if reconciled)
        rec = sess.execute(
            select(Reconciliation).where(Reconciliation.tick_id == inv.tick_id)
        ).scalar_one_or_none()
        rec_dict = rec.to_dict() if rec else None

        # Load linked ModelPrediction (if available)
        pred_dict = None
        if inv.prediction_id is not None:
            pred = sess.execute(
                select(ModelPrediction).where(ModelPrediction.id == inv.prediction_id)
            ).scalar_one_or_none()
            pred_dict = pred.to_dict() if pred else None

        return {
            "investigation": inv_dict,
            "market_tick": tick_dict,
            "reconciliation": rec_dict,
            "model_prediction": pred_dict,
        }

    try:
        if session is not None:
            return _execute(session)
        with get_db() as new_session:
            return _execute(new_session)
    except Exception as exc:
        logger.warning(
            "Failed to fetch investigation details for ID %s: error_type=%s",
            investigation_id,
            type(exc).__name__,
        )
        return None


def update_prediction_investigation_status(
    investigation_id: int,
    new_status: str,
    verified_by: str,
    notes: Optional[str] = None,
    ground_truth_label: Optional[str] = None,
    new_confidence: Optional[float] = None,
    session: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Updates the verification status of an investigation with full audit history preservation.

    Enforces strict gates:
    - Promoting to 'APPROVED_FOR_TRAINING' requires confirmed cause and verified ground truth label.
    - An unverified hypothesis or refuted explanation cannot be approved for training.
    - Appends an immutable audit entry to revision_history_json.
    """
    _validate_verification_status(new_status)
    _validate_diagnostic_confidence(new_confidence)

    if not verified_by or not verified_by.strip():
        raise ValueError("verified_by cannot be empty.")

    def _execute(sess: Session) -> Dict[str, Any]:
        inv = sess.execute(
            select(PredictionInvestigation).where(PredictionInvestigation.id == investigation_id)
        ).scalar_one_or_none()
        if inv is None:
            raise ValueError(f"Investigation with ID {investigation_id} not found.")

        old_status = inv.verification_status

        # Strict Training Approval Gate:
        # Cannot approve for training without confirmed cause and non-empty ground truth label
        if new_status == "APPROVED_FOR_TRAINING":
            if not ground_truth_label or not ground_truth_label.strip():
                raise ValueError(
                    "Cannot approve investigation for training without a verified ground_truth_label."
                )
            if old_status not in ("CONFIRMED_CAUSE", "APPROVED_FOR_TRAINING"):
                raise ValueError(
                    f"Cannot approve investigation for training from status '{old_status}'. "
                    "Hypothesis must first be independently verified as 'CONFIRMED_CAUSE'."
                )

        now_utc = datetime.now(timezone.utc)
        current_history = list(inv.revision_history_json or [])
        history_entry = {
            "timestamp": now_utc.isoformat(),
            "author": verified_by.strip(),
            "action": "STATUS_TRANSITION",
            "old_status": old_status,
            "new_status": new_status,
            "ground_truth_label": ground_truth_label.strip() if ground_truth_label else inv.verified_ground_truth_label,
            "confidence": float(new_confidence) if new_confidence is not None else inv.diagnostic_confidence,
            "notes": notes.strip() if notes else None,
        }
        current_history.append(history_entry)

        inv.verification_status = new_status
        inv.verified_by = verified_by.strip()
        inv.verified_at = now_utc
        if ground_truth_label and ground_truth_label.strip():
            inv.verified_ground_truth_label = ground_truth_label.strip()
        if new_confidence is not None:
            inv.diagnostic_confidence = float(new_confidence)
        inv.revision_history_json = current_history
        inv.updated_at = now_utc

        sess.flush()
        return inv.to_dict()

    if session is not None:
        return _execute(session)
    with get_db() as new_session:
        return _execute(new_session)


def get_training_approved_investigations(
    limit: int = 100,
    model_version: Optional[str] = None,
    session: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    """
    Returns only verified, validated investigation records approved for future training sets.
    Strictly excludes unverified hypotheses, refuted records, or unconfirmed causes.
    """
    return get_prediction_investigations(
        limit=limit,
        model_version=model_version,
        verification_status="APPROVED_FOR_TRAINING",
        session=session,
    )
