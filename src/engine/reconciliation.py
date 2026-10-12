"""
Delayed Ground Truth Reconciliation Engine.
Grades predictions at T+1 against realized Binance market candles,
decomposes forecast residuals, isolates 3-sigma black-swan outliers,
and attributes Risk Critic veto quality.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, desc

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.models import MarketTick, Reconciliation, CouncilAudit

__all__ = [
    "reconcile_pending_ticks",
    "get_reconciliation_summary",
]


def reconcile_pending_ticks(session: Session, current_tick: MarketTick) -> List[Reconciliation]:
    """
    Finds all prior market ticks with unsettled predictions (lacking a Reconciliation entry)
    and evaluates them against the newly closed current_tick.
    
    Parameters:
        session: Active SQLAlchemy DB session
        current_tick: The latest confirmed closed candle (T)
        
    Returns:
        List of newly created Reconciliation records.
    """
    if not current_tick or current_tick.close <= 0:
        return []

    # Find candidate ticks that have a predicted price, are strictly older than current_tick,
    # and do not yet have a record in the reconciliations table.
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
        actual_price = max(1.0, float(current_tick.close))
        predicted_price = float(prior_tick.predicted_price_1h)
        base_price = max(1.0, float(prior_tick.close))

        # 1. Return Calculations
        predicted_return_pct = ((predicted_price - base_price) / base_price) * 100.0
        actual_return_pct = ((actual_price - base_price) / base_price) * 100.0
        residual_error = actual_return_pct - predicted_return_pct

        # 2. Dollar and Percentage Error
        absolute_error_usd = abs(actual_price - predicted_price)
        percentage_error = (absolute_error_usd / actual_price) * 100.0

        # 3. Directional Alignment
        dir_pred = "UP" if predicted_return_pct > 0.05 else ("DOWN" if predicted_return_pct < -0.05 else "FLAT")
        dir_real = "UP" if actual_return_pct > 0.05 else ("DOWN" if actual_return_pct < -0.05 else "FLAT")
        
        pred_sign = np.sign(predicted_return_pct) if abs(predicted_return_pct) >= 0.02 else 0
        real_sign = np.sign(actual_return_pct) if abs(actual_return_pct) >= 0.02 else 0
        direction_correct = bool(pred_sign == real_sign)

        # 4. 3-Sigma Black Swan Detection (BTC hourly 3-sigma is typically > 3.8% return)
        is_black_swan = bool(abs(actual_return_pct) >= 3.8 or abs(residual_error) >= 3.8)

        # 5. Institutional Error Attribution Tagging
        if is_black_swan:
            tag = "BLACK_SWAN_OUTLIER"
        elif percentage_error <= 0.35:
            tag = "ACCURATE"
        elif direction_correct and abs(actual_return_pct) > abs(predicted_return_pct):
            tag = "MOMENTUM_UNDERESTIMATED"
        elif direction_correct and abs(actual_return_pct) < abs(predicted_return_pct):
            tag = "MOMENTUM_OVERESTIMATED"
        else:
            tag = "DIRECTION_MISALIGNED"

        # 6. Instantiate Reconciliation Record
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

        # 7. Audit Risk Critic Veto Quality (if CouncilAudit exists for this prior tick)
        audit = session.execute(
            select(CouncilAudit)
            .where(CouncilAudit.tick_id == prior_tick.id)
            .order_by(CouncilAudit.id.desc())
        ).scalars().first()

        if audit:
            if audit.risk_verdict == "VETOED":
                if audit.strategist_action == "BUY" and actual_return_pct < 0:
                    audit.veto_attribution = "SAVED_CAPITAL"  # Critic avoided loss!
                elif audit.strategist_action == "BUY" and actual_return_pct > 0.10:
                    audit.veto_attribution = "OPPORTUNITY_COST"  # Critic was too cautious
                elif audit.strategist_action == "SELL" and actual_return_pct > 0:
                    audit.veto_attribution = "SAVED_CAPITAL"  # Critic avoided short loss on market rally
                elif audit.strategist_action == "SELL" and actual_return_pct < -0.10:
                    audit.veto_attribution = "OPPORTUNITY_COST"  # Critic missed profitable short drop
                else:
                    audit.veto_attribution = "NEUTRAL"
            else:
                audit.veto_attribution = "N/A_APPROVED"

    session.flush()
    return reconciled_records


def get_reconciliation_summary(session: Session, limit: int = 168) -> Dict[str, Any]:
    """
    Computes rolling performance metrics across the most recent N reconciled predictions.
    Used by Streamlit Terminal and FastAPI monitoring endpoints.
    
    Parameters:
        session: Active DB session
        limit: Rolling lookback window (default 168 hours = 7 days)
        
    Returns:
        Dictionary containing win rate, MAE, MAPE, IC, and sample counts.
    """
    stmt = (
        select(Reconciliation)
        .order_by(desc(Reconciliation.timestamp))
        .limit(limit)
    )
    records = session.execute(stmt).scalars().all()

    if not records:
        return {
            "total_evaluated": 0,
            "total_reconciled": 0,
            "win_rate_pct": 0.0,
            "directional_accuracy_pct": 0.0,
            "mae_usd": 0.0,
            "mape_pct": 0.0,
            "information_coefficient": 0.0,
            "black_swans_filtered": 0,
            "accurate_count": 0,
        }

    total = len(records)
    wins = sum(1 for r in records if r.direction_correct)
    mae = float(np.mean([r.absolute_error_usd for r in records]))
    mape = float(np.mean([r.percentage_error for r in records]))
    black_swans = sum(1 for r in records if r.is_black_swan)
    accurate = sum(1 for r in records if r.error_attribution_tag == "ACCURATE")

    # Information Coefficient (Pearson correlation between predicted return and actual return)
    pred_returns = [r.predicted_return_pct for r in records]
    actual_returns = [r.actual_return_pct for r in records]
    if len(records) >= 3 and np.std(pred_returns) > 1e-8 and np.std(actual_returns) > 1e-8:
        corr_val = np.corrcoef(pred_returns, actual_returns)[0, 1]
        ic = float(np.nan_to_num(corr_val, nan=0.0))
    else:
        ic = 0.0

    win_rate = round((wins / total) * 100.0, 2)
    return {
        "total_evaluated": total,
        "total_reconciled": total,
        "win_rate_pct": win_rate,
        "directional_accuracy_pct": win_rate,
        "mae_usd": round(mae, 2),
        "mape_pct": round(mape, 3),
        "information_coefficient": round(ic, 4),
        "black_swans_filtered": black_swans,
        "accurate_count": accurate,
    }


if __name__ == '__main__':
    from src.database.connection import get_db
    print("[INFO] Testing Ground Truth Reconciliation Engine...")
    with get_db() as session:
        summary = get_reconciliation_summary(session, limit=24)
        print(f"[RECONCILIATION] Total Evaluated: {summary['total_evaluated']}")
        print(f"[RECONCILIATION] Win Rate: {summary['win_rate_pct']}%")
        print(f"[RECONCILIATION] MAE: ${summary['mae_usd']:,.2f} | MAPE: {summary['mape_pct']:.3f}%")
        print(f"[RECONCILIATION] Information Coefficient: {summary['information_coefficient']:.4f}")
        print(f"[RECONCILIATION] Accurate Count: {summary['accurate_count']} | Black Swans: {summary['black_swans_filtered']}")
    print("[SUCCESS] Reconciliation Engine verification completed.")
