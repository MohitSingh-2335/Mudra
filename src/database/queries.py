from datetime import datetime, timezone
import logging
import math
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import (
    CouncilAudit,
    MarketTick,
    ModelPrediction,
    PaperPortfolio,
    PredictionInvestigation,
    Reconciliation,
)

logger = logging.getLogger(__name__)
ALLOWED_VERIFICATION_STATUSES = frozenset({"UNVERIFIED_HYPOTHESIS", "CONFIRMED_CAUSE", "REFUTED", "APPROVED_FOR_TRAINING"})
__all__ = [
    "get_latest_market_ticks", "get_reconciliation_history", "get_reconciliation_kpis",
    "get_portfolio_history", "get_portfolio_kpis", "get_latest_council_audit",
    "ALLOWED_VERIFICATION_STATUSES", "record_prediction_investigation", "get_prediction_investigations",
    "get_prediction_investigation_details", "update_prediction_investigation_status", "get_training_approved_investigations",
]


def get_latest_market_ticks(limit: int = 100) -> pd.DataFrame:
    # returns recent market ticks ordered chronologically
    try:
        with get_db() as session:
            ticks = session.execute(select(MarketTick).order_by(desc(MarketTick.timestamp)).limit(limit)).scalars().all()
            if not ticks:
                return pd.DataFrame()
            df = pd.DataFrame([t.to_dict() for t in ticks])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        logger.warning("Failed to fetch market ticks: %s", type(exc).__name__)
        return pd.DataFrame()


def get_reconciliation_history(limit: int = 500, model_version: Optional[str] = None) -> pd.DataFrame:
    # returns reconciled prediction outcomes across primary and candidate models
    try:
        with get_db() as session:
            stmt = select(Reconciliation, MarketTick.model_version).outerjoin(MarketTick, Reconciliation.tick_id == MarketTick.id)
            if model_version and model_version != "All Models":
                stmt = stmt.where(MarketTick.model_version == model_version)
            rows = session.execute(stmt.order_by(desc(Reconciliation.timestamp)).limit(limit)).all()

            data = []
            for r, m_ver in rows:
                item = r.to_dict()
                item["model_version"] = m_ver
                data.append(item)

            cand_stmt = select(ModelPrediction, MarketTick.timestamp.label("tick_ts")).join(MarketTick, ModelPrediction.tick_id == MarketTick.id).where(ModelPrediction.outcome_tick_id.isnot(None), ModelPrediction.is_primary.is_(False))
            if model_version and model_version != "All Models":
                cand_stmt = cand_stmt.where(ModelPrediction.model_version == model_version)
            for cp, tick_ts in session.execute(cand_stmt.order_by(desc(ModelPrediction.reconciled_at)).limit(limit)).all():
                ts = cp.reconciled_at or tick_ts
                data.append({
                    "id": cp.id, "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "tick_id": cp.tick_id, "predicted_price": cp.predicted_price_1h,
                    "actual_price_at_t_plus_1": cp.actual_price_at_t_plus_1, "predicted_return_pct": cp.predicted_return_pct,
                    "actual_return_pct": cp.actual_return_pct, "residual_error": cp.residual_error,
                    "absolute_error_usd": cp.absolute_error_usd, "percentage_error": cp.percentage_error,
                    "direction_predicted": cp.direction_predicted, "direction_realized": cp.direction_realized,
                    "direction_correct": cp.direction_correct, "is_black_swan": cp.is_black_swan,
                    "error_attribution_tag": cp.error_attribution_tag, "model_version": cp.model_version,
                })

            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.sort_values("timestamp", ascending=False).reset_index(drop=True).head(limit)
    except Exception as exc:
        logger.warning("Failed to fetch reconciliation history: %s", type(exc).__name__)
        return pd.DataFrame()


def get_reconciliation_kpis(lookback_hours: Optional[int] = 168, model_version: Optional[str] = None) -> Dict[str, Any]:
    # returns aggregate win rate, directional accuracy, and MAE
    try:
        from src.engine import get_reconciliation_summary
        with get_db() as session:
            res = get_reconciliation_summary(session, limit=lookback_hours, model_version=model_version)
            res["query_failed"] = False
            res["error"] = None
            return res
    except Exception as exc:
        logger.warning("Failed to fetch reconciliation KPIs: %s", type(exc).__name__)
        return {
            "total_evaluated": 0, "total_reconciled": 0, "total_lifetime_evaluated": 0, "win_rate_pct": 0.0,
            "directional_accuracy_pct": 0.0, "mae_usd": 0.0, "mape_pct": 0.0, "information_coefficient": 0.0,
            "information_coefficient_ic": 0.0, "black_swans_filtered": 0, "black_swan_count": 0,
            "accurate_count": 0, "tag_breakdown": {}, "model_version": model_version if (model_version and model_version != "All Models") else None,
            "query_failed": True, "error": "Database query failed",
        }


def get_portfolio_history(limit: int = 500) -> pd.DataFrame:
    # returns chronological paper portfolio state snapshots
    try:
        with get_db() as session:
            records = session.execute(select(PaperPortfolio).order_by(PaperPortfolio.id.asc()).limit(limit)).scalars().all()
            if not records:
                return pd.DataFrame()
            df = pd.DataFrame([p.to_dict() for p in records])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
    except Exception as exc:
        logger.warning("Failed to fetch portfolio history: %s", type(exc).__name__)
        return pd.DataFrame()


def get_portfolio_kpis(current_price: float) -> Dict[str, Any]:
    # returns portfolio equity, alpha, and trade statistics
    try:
        from src.engine import get_portfolio_summary
        with get_db() as session:
            return get_portfolio_summary(session, current_price=current_price)
    except Exception as exc:
        logger.warning("Failed to fetch portfolio KPIs: %s", type(exc).__name__)
        return {
            "cash_balance": 10000.0, "btc_balance": 0.0, "current_btc_price": round(current_price, 2),
            "btc_position_value_usd": 0.0, "total_equity_usd": 10000.0, "cumulative_return_pct": 0.0,
            "benchmark_btc_equity": 10000.0, "alpha_usd": 0.0, "alpha_pct": 0.0, "realized_pnl_usd": 0.0,
            "total_fees_usd": 0.0, "drawdown_pct": 0.0, "total_trades": 0,
        }


def get_latest_council_audit() -> Optional[Dict[str, Any]]:
    # returns most recent multi-agent council audit decision
    try:
        with get_db() as session:
            audit = session.execute(select(CouncilAudit).order_by(desc(CouncilAudit.id)).limit(1)).scalar_one_or_none()
            return audit.to_dict() if audit else None
    except Exception as exc:
        logger.warning("Failed to fetch latest council audit: %s", type(exc).__name__)
        return None


# validates investigation status and confidence score bounds
def _validate_status_and_conf(status: str, confidence: Optional[float] = None) -> None:
    if not isinstance(status, str) or status.strip() not in ALLOWED_VERIFICATION_STATUSES:
        raise ValueError(f"Invalid verification_status '{status}'.")
    if confidence is not None:
        try:
            val = float(confidence)
            if not math.isfinite(val) or val < 0.0 or val > 1.0:
                raise ValueError
        except Exception as err:
            raise ValueError(f"Diagnostic confidence must be between 0.0 and 1.0, got {confidence}.") from err


# converts iso string or datetime to timezone-aware utc datetime
def _normalize_datetime(dt_val: Optional[Union[str, datetime]]) -> Optional[datetime]:
    if dt_val is None or isinstance(dt_val, datetime):
        return dt_val.replace(tzinfo=timezone.utc) if isinstance(dt_val, datetime) and not dt_val.tzinfo else dt_val
    try:
        parsed = datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception as err:
        raise ValueError(f"Invalid ISO datetime string: {dt_val}") from err


# records a new root-cause investigation hypothesis linked to a market tick
def record_prediction_investigation(
    tick_id: int, model_version: str, headline: str, suspected_cause_category: str,
    hypothesis_notes: str, supporting_evidence_json: Optional[Dict[str, Any]] = None,
    diagnostic_confidence: Optional[float] = None, unresolved_questions: Optional[str] = None,
    verification_status: str = "UNVERIFIED_HYPOTHESIS", headline_source_url: Optional[str] = None,
    headline_published_at: Optional[Union[str, datetime]] = None, prediction_id: Optional[int] = None,
    author: str = "SYSTEM", session: Optional[Session] = None,
) -> Dict[str, Any]:
    _validate_status_and_conf(verification_status, diagnostic_confidence)
    for name, val in [("headline", headline), ("suspected_cause_category", suspected_cause_category), ("hypothesis_notes", hypothesis_notes), ("model_version", model_version)]:
        if not val or not val.strip():
            raise ValueError(f"{name} cannot be empty.")

    pub_at = _normalize_datetime(headline_published_at)

    def _execute(sess: Session) -> Dict[str, Any]:
        if not sess.execute(select(MarketTick.id).where(MarketTick.id == tick_id)).scalar_one_or_none():
            raise ValueError(f"MarketTick {tick_id} does not exist.")
        if prediction_id is not None and not sess.execute(select(ModelPrediction.id).where(ModelPrediction.id == prediction_id)).scalar_one_or_none():
            raise ValueError(f"ModelPrediction {prediction_id} does not exist.")

        now_utc = datetime.now(timezone.utc)
        inv = PredictionInvestigation(
            tick_id=tick_id, prediction_id=prediction_id, model_version=model_version.strip(),
            headline=headline.strip(), headline_source_url=headline_source_url.strip() if headline_source_url else None,
            headline_published_at=pub_at, suspected_cause_category=suspected_cause_category.strip(),
            hypothesis_notes=hypothesis_notes.strip(), supporting_evidence_json=supporting_evidence_json or {},
            diagnostic_confidence=float(diagnostic_confidence) if diagnostic_confidence is not None else None,
            unresolved_questions=unresolved_questions.strip() if unresolved_questions else None,
            verification_status=verification_status, created_at=now_utc, updated_at=now_utc,
            revision_history_json=[{
                "timestamp": now_utc.isoformat(), "author": author, "action": "RECORDED",
                "verification_status": verification_status, "diagnostic_confidence": diagnostic_confidence,
                "suspected_cause_category": suspected_cause_category, "notes": hypothesis_notes[:250],
            }],
        )
        sess.add(inv)
        sess.flush()
        return inv.to_dict()

    if session:
        return _execute(session)
    with get_db() as new_session:
        return _execute(new_session)


# queries investigations with optional model version, status, or keyword filters
def get_prediction_investigations(
    limit: int = 50, model_version: Optional[str] = None, verification_status: Optional[str] = None,
    headline_query: Optional[str] = None, tick_id: Optional[int] = None, session: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    def _execute(sess: Session) -> List[Dict[str, Any]]:
        stmt = select(PredictionInvestigation).order_by(desc(PredictionInvestigation.id))
        if model_version and model_version != "All Models":
            stmt = stmt.where(PredictionInvestigation.model_version == model_version)
        if verification_status and verification_status != "ALL":
            stmt = stmt.where(PredictionInvestigation.verification_status == verification_status)
        if headline_query and headline_query.strip():
            stmt = stmt.where(PredictionInvestigation.headline.ilike(f"%{headline_query.strip()}%"))
        if tick_id is not None:
            stmt = stmt.where(PredictionInvestigation.tick_id == tick_id)
        return [row.to_dict() for row in sess.execute(stmt.limit(limit)).scalars().all()]

    if session:
        return _execute(session)
    with get_db() as s:
        return _execute(s)


# fetches full investigation detail by primary key id
def get_prediction_investigation_details(investigation_id: int, session: Optional[Session] = None) -> Optional[Dict[str, Any]]:
    def _execute(sess: Session) -> Optional[Dict[str, Any]]:
        inv = sess.execute(select(PredictionInvestigation).where(PredictionInvestigation.id == investigation_id)).scalar_one_or_none()
        return inv.to_dict() if inv is not None else None

    if session:
        return _execute(session)
    with get_db() as s:
        return _execute(s)


# updates investigation status, author, and appends to immutable audit trail
def update_prediction_investigation_status(
    investigation_id: int, new_status: str, verified_by: str, notes: Optional[str] = None,
    ground_truth_label: Optional[str] = None, new_confidence: Optional[float] = None, session: Optional[Session] = None,
) -> Dict[str, Any]:
    _validate_status_and_conf(new_status, new_confidence)
    if not verified_by or not verified_by.strip():
        raise ValueError("verified_by cannot be empty.")

    def _execute(sess: Session) -> Dict[str, Any]:
        inv = sess.execute(select(PredictionInvestigation).where(PredictionInvestigation.id == investigation_id)).scalar_one_or_none()
        if inv is None:
            raise ValueError(f"Investigation {investigation_id} not found.")

        now_utc = datetime.now(timezone.utc)
        history = list(inv.revision_history_json or [])
        history.append({
            "timestamp": now_utc.isoformat(), "author": verified_by.strip(),
            "action": f"STATUS_CHANGED_{new_status}", "from_status": inv.verification_status,
            "to_status": new_status, "notes": notes[:250] if notes else None,
        })
        inv.verification_status = new_status
        inv.verified_by = verified_by.strip()
        inv.verified_at = now_utc
        if ground_truth_label:
            inv.verified_ground_truth_label = ground_truth_label.strip()
        if new_confidence is not None:
            inv.diagnostic_confidence = float(new_confidence)
        inv.revision_history_json = history
        inv.updated_at = now_utc
        sess.flush()
        return inv.to_dict()

    if session:
        return _execute(session)
    with get_db() as s:
        return _execute(s)


# retrieves investigations explicitly approved for retraining datasets
def get_training_approved_investigations(limit: int = 100, session: Optional[Session] = None) -> List[Dict[str, Any]]:
    return get_prediction_investigations(limit=limit, verification_status="APPROVED_FOR_TRAINING", session=session)
