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


# Graded evaluation of pending predictions against verified T+1h confirmed candle
def reconcile_pending_ticks(session: Session, current_tick: MarketTick) -> List[Reconciliation]:
    if not current_tick or current_tick.close <= 0:
        return []

    stmt = (
        select(MarketTick)
        .outerjoin(Reconciliation, MarketTick.id == Reconciliation.tick_id)
        .where(and_(MarketTick.predicted_price_1h.isnot(None), MarketTick.timestamp < current_tick.timestamp, Reconciliation.id.is_(None)))
        .order_by(MarketTick.timestamp.asc())
    )
    pending_ticks = session.execute(stmt).scalars().all()
    reconciled_records: List[Reconciliation] = []

    for prior_tick in pending_ticks:
        expected_ts = prior_tick.timestamp + timedelta(hours=1)
        outcome = current_tick if current_tick.timestamp == expected_ts else session.execute(select(MarketTick).where(MarketTick.timestamp == expected_ts)).scalars().first()
        if not outcome or outcome.close <= 0:
            continue

        act_p, pred_p, base_p = max(1.0, float(outcome.close)), float(prior_tick.predicted_price_1h), max(1.0, float(prior_tick.close))
        pred_ret, act_ret = ((pred_p - base_p) / base_p) * 100.0, ((act_p - base_p) / base_p) * 100.0
        abs_err, pct_err = abs(act_p - pred_p), (abs(act_p - pred_p) / act_p) * 100.0
        dir_pred, dir_real = ("UP" if pred_ret >= 0 else "DOWN"), ("UP" if act_ret >= 0 else "DOWN")
        is_swan = bool(abs(act_ret) >= 3.8 or abs(act_ret - pred_ret) >= 3.8)

        if is_swan:
            tag = "BLACK_SWAN_OUTLIER"
        elif dir_pred == dir_real:
            tag = "ACCURATE" if pct_err <= 0.35 else ("MOMENTUM_UNDERESTIMATED" if abs(act_ret) > abs(pred_ret) else "MOMENTUM_OVERESTIMATED")
        else:
            tag = "SIDEWAYS_DRIFT" if (pct_err <= 0.20 or abs_err < 150.0) else "DIRECTION_MISALIGNED"

        rec = Reconciliation(
            timestamp=datetime.now(timezone.utc), tick_id=prior_tick.id, predicted_price=pred_p,
            actual_price_at_t_plus_1=act_p, predicted_return_pct=pred_ret, actual_return_pct=act_ret,
            residual_error=act_ret - pred_ret, absolute_error_usd=abs_err, percentage_error=pct_err,
            direction_predicted=dir_pred, direction_realized=dir_real, direction_correct=(dir_pred == dir_real),
            is_black_swan=is_swan, error_attribution_tag=tag,
        )
        session.add(rec)
        reconciled_records.append(rec)

        try:
            from src.database.models import ModelPrediction
            for u in session.execute(select(ModelPrediction).where(ModelPrediction.tick_id == prior_tick.id, ModelPrediction.outcome_tick_id.is_(None))).scalars().all():
                u.outcome_tick_id, u.evaluation_policy_version = outcome.id, "binary_exact_t1_v2"
                u.actual_price_at_t_plus_1, u.actual_return_pct = act_p, act_ret
                u.direction_realized, u.direction_correct = dir_real, (u.direction_predicted == dir_real)
                u.residual_error, u.absolute_error_usd = act_ret - float(u.predicted_return_pct), abs_err
                u.percentage_error, u.is_black_swan = pct_err, is_swan
                u.error_attribution_tag, u.reconciled_at = tag, datetime.now(timezone.utc)
        except Exception:
            pass

        audit = session.execute(select(CouncilAudit).where(CouncilAudit.tick_id == prior_tick.id).order_by(CouncilAudit.id.desc())).scalars().first()
        if audit:
            if audit.risk_verdict == "VETOED":
                if (audit.strategist_action == "BUY" and act_ret < 0) or (audit.strategist_action == "SELL" and act_ret > 0):
                    audit.veto_attribution = "SAVED_CAPITAL"
                elif (audit.strategist_action == "BUY" and act_ret > 0.10) or (audit.strategist_action == "SELL" and act_ret < -0.10):
                    audit.veto_attribution = "OPPORTUNITY_COST"
                else:
                    audit.veto_attribution = "NEUTRAL"
            else:
                audit.veto_attribution = "N/A_APPROVED"

    session.flush()
    return reconciled_records


# Diagnostic plain English headline and notes for investigations
def build_plain_english_reconciliation_story(
    pred_tick: MarketTick, outcome_tick: MarketTick, pred_price: float, actual_price: float,
    pred_ret_pct: float, actual_ret_pct: float, dir_pred: str, dir_real: str,
    dir_correct: bool, abs_err_usd: float, tag: str,
) -> Dict[str, str]:
    start = float(pred_tick.close)
    res_str = "CORRECT" if dir_correct else "WRONG"
    headline = f"[{res_str}] Pred {dir_pred} ({pred_price - start:+,.2f}$) | Actual {dir_real} ({actual_price - start:+,.2f}$)"
    if tag == "BLACK_SWAN_OUTLIER":
        why = f"Big Market Swing Alert: moved {actual_ret_pct:+.2f}% with ${float(outcome_tick.high - outcome_tick.low):,.2f} range."
    elif dir_correct and tag == "ACCURATE":
        why = f"Spot-On Forecast: {dir_pred} accurate within ${abs_err_usd:,.2f}."
    elif dir_correct:
        why = f"Direction Right ({dir_pred}): Bitcoin moved ${actual_price - start:+,.2f} vs pred ${pred_price - start:+,.2f}."
    elif tag == "SIDEWAYS_DRIFT":
        why = f"Small Chop Flip: drifted by ${abs_err_usd:,.2f} inside normal noise."
    else:
        why = f"Trend Reversal Miss: reversed to {dir_real}, missing by ${abs_err_usd:,.2f}."
    notes = f"Pred: ${pred_price:,.2f} ({dir_pred}) | Actual: ${actual_price:,.2f} ({dir_real}) | Err: ${abs_err_usd:,.2f} | Diagnosis: {why}"
    return {"headline": headline[:500], "hypothesis_notes": notes, "suspected_cause_category": tag}


# Summary KPI calculations across reconciled predictions
def get_reconciliation_summary(session: Session, limit: int = 168, model_version: Optional[str] = None) -> Dict[str, Any]:
    records: List[Any] = []
    total_lifetime = 0

    if model_version and model_version != "All Models":
        try:
            from src.database.models import ModelPrediction
            stmt = select(ModelPrediction).where(ModelPrediction.model_version == model_version, ModelPrediction.outcome_tick_id.isnot(None)).order_by(desc(ModelPrediction.reconciled_at))
            records = session.execute(stmt.limit(limit) if limit and limit > 0 else stmt).scalars().all()
            if records:
                total_lifetime = session.execute(select(func.count(ModelPrediction.id)).where(ModelPrediction.model_version == model_version, ModelPrediction.outcome_tick_id.isnot(None))).scalar() or len(records)
        except Exception:
            pass
        if not records:
            stmt = select(Reconciliation).join(MarketTick, Reconciliation.tick_id == MarketTick.id).where(MarketTick.model_version == model_version).order_by(desc(Reconciliation.timestamp))
            records = session.execute(stmt.limit(limit) if limit and limit > 0 else stmt).scalars().all()
            total_lifetime = session.execute(select(func.count(Reconciliation.id)).join(MarketTick, Reconciliation.tick_id == MarketTick.id).where(MarketTick.model_version == model_version)).scalar() or (len(records) if records else 0)
    else:
        total_lifetime = session.execute(select(func.count(Reconciliation.id))).scalar() or 0
        stmt = select(Reconciliation).order_by(desc(Reconciliation.timestamp))
        records = session.execute(stmt.limit(limit) if limit and limit > 0 else stmt).scalars().all()

    if not records:
        return {
            "total_evaluated": 0, "total_reconciled": 0, "total_lifetime_evaluated": total_lifetime,
            "win_rate_pct": 0.0, "directional_accuracy_pct": 0.0, "mae_usd": 0.0, "mape_pct": 0.0,
            "information_coefficient": 0.0, "information_coefficient_ic": 0.0, "black_swans_filtered": 0,
            "black_swan_count": 0, "accurate_count": 0, "tag_breakdown": {},
            "model_version": model_version if (model_version and model_version != "All Models") else None,
            "query_failed": False, "error": None,
        }

    total = len(records)
    wins = sum(1 for r in records if r.direction_correct)
    mae = float(np.mean([r.absolute_error_usd for r in records]))
    mape = float(np.mean([r.percentage_error for r in records]))
    ic = 0.0
    pred_rets, act_rets = [r.predicted_return_pct for r in records], [r.actual_return_pct for r in records]
    if len(records) >= 3 and np.std(pred_rets) > 1e-8 and np.std(act_rets) > 1e-8:
        ic = float(np.nan_to_num(np.corrcoef(pred_rets, act_rets)[0, 1], nan=0.0))

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
