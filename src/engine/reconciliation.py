import logging
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sqlalchemy import and_, desc, func, select
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.models import CouncilAudit, MarketTick, Reconciliation

logger = logging.getLogger("mudra.reconciliation")

__all__ = [
    "reconcile_pending_ticks",
    "get_reconciliation_summary",
    "build_plain_english_reconciliation_story",
]


def reconcile_pending_ticks(session: Session, current_tick: MarketTick) -> List[Reconciliation]:
    # Finds unresolved predictions and grades them strictly against T+1h candle
    if not current_tick or current_tick.close <= 0:
        return []

    stmt = (
        select(MarketTick)
        .outerjoin(Reconciliation, MarketTick.id == Reconciliation.tick_id)
        .where(
            and_(
                MarketTick.predicted_price_1h.isnot(None),
                MarketTick.timestamp < current_tick.timestamp,
                Reconciliation.id.is_(None),
            )
        )
        .order_by(MarketTick.timestamp.asc())
    )
    pending_ticks = session.execute(stmt).scalars().all()
    reconciled_records: List[Reconciliation] = []

    for prior_tick in pending_ticks:
        expected_ts = prior_tick.timestamp + timedelta(hours=1)
        outcome_candle = (
            current_tick
            if current_tick.timestamp == expected_ts
            else session.execute(select(MarketTick).where(MarketTick.timestamp == expected_ts)).scalars().first()
        )

        if not outcome_candle or outcome_candle.close <= 0:
            continue

        actual_price = max(1.0, float(outcome_candle.close))
        predicted_price = float(prior_tick.predicted_price_1h)
        base_price = max(1.0, float(prior_tick.close))

        predicted_return_pct = ((predicted_price - base_price) / base_price) * 100.0
        actual_return_pct = ((actual_price - base_price) / base_price) * 100.0
        residual_error = actual_return_pct - predicted_return_pct

        absolute_error_usd = abs(actual_price - predicted_price)
        percentage_error = (absolute_error_usd / actual_price) * 100.0

        dir_pred = "UP" if predicted_return_pct >= 0.0 else "DOWN"
        dir_real = "UP" if actual_return_pct >= 0.0 else "DOWN"
        direction_correct = bool(dir_pred == dir_real)
        is_black_swan = bool(abs(actual_return_pct) >= 3.8 or abs(residual_error) >= 3.8)

        # Attribution tagging
        if is_black_swan:
            tag = "BLACK_SWAN_OUTLIER"
        elif direction_correct:
            if percentage_error <= 0.35:
                tag = "ACCURATE"
            elif abs(actual_return_pct) > abs(predicted_return_pct):
                tag = "MOMENTUM_UNDERESTIMATED"
            else:
                tag = "MOMENTUM_OVERESTIMATED"
        else:
            tag = "SIDEWAYS_DRIFT" if (percentage_error <= 0.20 or absolute_error_usd < 150.0) else "DIRECTION_MISALIGNED"

        rec = Reconciliation(
            timestamp=datetime.now(timezone.utc),
            tick_id=prior_tick.id,
            predicted_price=predicted_price,
            actual_price_at_t_plus_1=actual_price,
            predicted_return_pct=predicted_return_pct,
            actual_return_pct=actual_return_pct,
            residual_error=residual_error,
            absolute_error_usd=absolute_error_usd,
            percentage_error=percentage_error,
            direction_predicted=dir_pred,
            direction_realized=dir_real,
            direction_correct=direction_correct,
            is_black_swan=is_black_swan,
            error_attribution_tag=tag,
        )
        session.add(rec)
        reconciled_records.append(rec)

        # Update model_predictions rows if present
        try:
            from src.database.models import ModelPrediction
            unresolved = session.execute(
                select(ModelPrediction).where(ModelPrediction.tick_id == prior_tick.id, ModelPrediction.outcome_tick_id.is_(None))
            ).scalars().all()
            for u in unresolved:
                u.outcome_tick_id = outcome_candle.id
                u.evaluation_policy_version = "binary_exact_t1_v2"
                u.actual_price_at_t_plus_1 = actual_price
                u.actual_return_pct = actual_return_pct
                u.direction_realized = dir_real
                u.direction_correct = (u.direction_predicted == dir_real)
                u.residual_error = actual_return_pct - float(u.predicted_return_pct)
                u.absolute_error_usd = abs(actual_price - float(u.predicted_price_1h))
                u.percentage_error = (u.absolute_error_usd / actual_price) * 100.0
                u.is_black_swan = bool(abs(actual_return_pct) >= 3.8 or abs(u.residual_error) >= 3.8)
                u.error_attribution_tag = tag
                u.reconciled_at = datetime.now(timezone.utc)
        except Exception:
            pass

        # Veto quality attribution
        audit = session.execute(
            select(CouncilAudit).where(CouncilAudit.tick_id == prior_tick.id).order_by(CouncilAudit.id.desc())
        ).scalars().first()
        if audit:
            if audit.risk_verdict == "VETOED":
                if audit.strategist_action == "BUY" and actual_return_pct < 0:
                    audit.veto_attribution = "SAVED_CAPITAL"
                elif audit.strategist_action == "BUY" and actual_return_pct > 0.10:
                    audit.veto_attribution = "OPPORTUNITY_COST"
                elif audit.strategist_action == "SELL" and actual_return_pct > 0:
                    audit.veto_attribution = "SAVED_CAPITAL"
                elif audit.strategist_action == "SELL" and actual_return_pct < -0.10:
                    audit.veto_attribution = "OPPORTUNITY_COST"
                else:
                    audit.veto_attribution = "NEUTRAL"
            else:
                audit.veto_attribution = "N/A_APPROVED"

    session.flush()
    return reconciled_records


def build_plain_english_reconciliation_story(
    pred_tick: MarketTick,
    outcome_tick: MarketTick,
    pred_price: float,
    actual_price: float,
    pred_ret_pct: float,
    actual_ret_pct: float,
    dir_pred: str,
    dir_real: str,
    dir_correct: bool,
    abs_err_usd: float,
    tag: str,
) -> Dict[str, str]:
    start_price = float(pred_tick.close)
    pred_move = pred_price - start_price
    actual_move = actual_price - start_price
    candle_range = float(outcome_tick.high - outcome_tick.low)

    res_str = "CORRECT" if dir_correct else "WRONG"
    headline = f"[{res_str}] Pred {dir_pred} ({pred_move:+,.2f}$) | Actual {dir_real} ({actual_move:+,.2f}$)"

    if tag == "BLACK_SWAN_OUTLIER":
        why = f"Big Market Swing Alert: moved {actual_ret_pct:+.2f}% with ${candle_range:,.2f} range."
    elif dir_correct and tag == "ACCURATE":
        why = f"Spot-On Forecast: {dir_pred} accurate within ${abs_err_usd:,.2f}."
    elif dir_correct:
        why = f"Direction Right ({dir_pred}): Bitcoin moved ${actual_move:+,.2f} vs pred ${pred_move:+,.2f}."
    elif tag == "SIDEWAYS_DRIFT":
        why = f"Small Chop Flip: drifted by ${abs_err_usd:,.2f} inside normal noise."
    else:
        why = f"Trend Reversal Miss: reversed to {dir_real}, missing by ${abs_err_usd:,.2f}."

    notes = f"Pred: ${pred_price:,.2f} ({dir_pred}) | Actual: ${actual_price:,.2f} ({dir_real}) | Err: ${abs_err_usd:,.2f} | Diagnosis: {why}"
    return {"headline": headline[:500], "hypothesis_notes": notes, "suspected_cause_category": tag}


def get_reconciliation_summary(session: Session, limit: int = 168, model_version: Optional[str] = None) -> Dict[str, Any]:
    records: List[Any] = []
    total_lifetime = 0

    if model_version and model_version != "All Models":
        try:
            from src.database.models import ModelPrediction
            pred_stmt = select(ModelPrediction).where(ModelPrediction.model_version == model_version, ModelPrediction.outcome_tick_id.isnot(None)).order_by(desc(ModelPrediction.reconciled_at))
            if limit and limit > 0:
                pred_stmt = pred_stmt.limit(limit)
            records = session.execute(pred_stmt).scalars().all()
            if records:
                total_lifetime = session.execute(select(func.count(ModelPrediction.id)).where(ModelPrediction.model_version == model_version, ModelPrediction.outcome_tick_id.isnot(None))).scalar() or len(records)
        except Exception:
            pass

        if not records:
            stmt = select(Reconciliation).join(MarketTick, Reconciliation.tick_id == MarketTick.id).where(MarketTick.model_version == model_version).order_by(desc(Reconciliation.timestamp))
            if limit and limit > 0:
                stmt = stmt.limit(limit)
            records = session.execute(stmt).scalars().all()
            total_lifetime = session.execute(select(func.count(Reconciliation.id)).join(MarketTick, Reconciliation.tick_id == MarketTick.id).where(MarketTick.model_version == model_version)).scalar() or len(records) if records else 0
    else:
        total_lifetime = session.execute(select(func.count(Reconciliation.id))).scalar() or 0
        stmt = select(Reconciliation).order_by(desc(Reconciliation.timestamp))
        if limit and limit > 0:
            stmt = stmt.limit(limit)
        records = session.execute(stmt).scalars().all()

    if not records:
        return {
            "total_evaluated": 0, "total_reconciled": 0, "total_lifetime_evaluated": total_lifetime,
            "win_rate_pct": 0.0, "directional_accuracy_pct": 0.0, "mae_usd": 0.0, "mape_pct": 0.0,
            "information_coefficient": 0.0, "information_coefficient_ic": 0.0, "black_swans_filtered": 0,
            "black_swan_count": 0, "accurate_count": 0, "tag_breakdown": {}, "model_version": model_version if (model_version and model_version != "All Models") else None,
            "query_failed": False, "error": None,
        }

    total = len(records)
    wins = sum(1 for r in records if r.direction_correct)
    mae = float(np.mean([r.absolute_error_usd for r in records]))
    mape = float(np.mean([r.percentage_error for r in records]))
    ic = 0.0
    pred_rets = [r.predicted_return_pct for r in records]
    act_rets = [r.actual_return_pct for r in records]
    if len(records) >= 3 and np.std(pred_rets) > 1e-8 and np.std(act_rets) > 1e-8:
        corr = np.corrcoef(pred_rets, act_rets)[0, 1]
        ic = float(np.nan_to_num(corr, nan=0.0))

    win_rate = round((wins / total) * 100.0, 2)
    return {
        "total_evaluated": total, "total_reconciled": total, "total_lifetime_evaluated": total_lifetime,
        "win_rate_pct": win_rate, "directional_accuracy_pct": win_rate, "mae_usd": round(mae, 2), "mape_pct": round(mape, 3),
        "information_coefficient": round(ic, 4), "information_coefficient_ic": round(ic, 4),
        "black_swans_filtered": sum(1 for r in records if r.is_black_swan),
        "black_swan_count": sum(1 for r in records if r.is_black_swan),
        "accurate_count": sum(1 for r in records if r.error_attribution_tag == "ACCURATE"),
        "tag_breakdown": dict(Counter(r.error_attribution_tag for r in records if r.error_attribution_tag)),
        "model_version": model_version if (model_version and model_version != "All Models") else None,
        "query_failed": False, "error": None,
    }
