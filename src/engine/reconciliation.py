"""
Delayed Ground Truth Reconciliation Engine.
Grades predictions at T+1 against realized Binance market candles,
decomposes forecast residuals, isolates 3-sigma black-swan outliers,
and attributes Risk Critic veto quality.
"""
import sys
from pathlib import Path
import logging
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, desc

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.models import MarketTick, Reconciliation, CouncilAudit

logger = logging.getLogger("mudra.reconciliation")

__all__ = [
    "reconcile_pending_ticks",
    "get_reconciliation_summary",
    "build_plain_english_reconciliation_story",
]


def reconcile_pending_ticks(session: Session, current_tick: MarketTick) -> List[Reconciliation]:
    """
    Finds prior market ticks with unsettled predictions (lacking a Reconciliation entry)
    and evaluates each one strictly against its intended T+1 hour outcome candle.
    
    If the expected outcome candle at prediction_timestamp + 1h is unavailable in the database,
    the prediction is left unresolved with a diagnostic log and is never graded against a later candle.
    
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
        expected_ts = prior_tick.timestamp + timedelta(hours=1)

        # 1. Match outcome candle: use current_tick if it is the exact expected candle,
        # otherwise look up the historical candle in market_ticks matching expected_ts
        if current_tick.timestamp == expected_ts:
            outcome_candle = current_tick
        else:
            outcome_candle = session.execute(
                select(MarketTick).where(MarketTick.timestamp == expected_ts)
            ).scalars().first()

        # 2. Defend against market/symbol cross-contamination if metadata is present
        if outcome_candle and prior_tick.features and outcome_candle.features:
            if isinstance(prior_tick.features, dict) and isinstance(outcome_candle.features, dict):
                p_sym = prior_tick.features.get("symbol")
                o_sym = outcome_candle.features.get("symbol")
                if p_sym and o_sym and p_sym != o_sym:
                    outcome_candle = None

        # 3. If the expected outcome candle is unavailable, leave prediction unresolved
        if not outcome_candle or outcome_candle.close <= 0:
            p_ts_str = prior_tick.timestamp.isoformat() if hasattr(prior_tick.timestamp, "isoformat") else str(prior_tick.timestamp)
            exp_ts_str = expected_ts.isoformat() if hasattr(expected_ts, "isoformat") else str(expected_ts)
            logger.warning(
                "[RECONCILIATION_UNRESOLVED] Tick ID %s (%s): expected T+1h outcome candle at %s "
                "is missing from market_ticks. Prediction left unresolved.",
                prior_tick.id,
                p_ts_str,
                exp_ts_str,
            )
            continue

        actual_price = max(1.0, float(outcome_candle.close))
        predicted_price = float(prior_tick.predicted_price_1h)
        base_price = max(1.0, float(prior_tick.close))

        # 1. Return Calculations
        predicted_return_pct = ((predicted_price - base_price) / base_price) * 100.0
        actual_return_pct = ((actual_price - base_price) / base_price) * 100.0
        residual_error = actual_return_pct - predicted_return_pct

        # 2. Dollar and Percentage Error
        absolute_error_usd = abs(actual_price - predicted_price)
        percentage_error = (absolute_error_usd / actual_price) * 100.0

        # 3. Directional Alignment (Strict binary UP / DOWN: non-negative return is UP, negative is DOWN)
        dir_pred = "UP" if predicted_return_pct >= 0.0 else "DOWN"
        dir_real = "UP" if actual_return_pct >= 0.0 else "DOWN"
        direction_correct = bool(dir_pred == dir_real)

        # 4. 3-Sigma Black Swan Detection (BTC hourly 3-sigma is typically > 3.8% return)
        is_black_swan = bool(abs(actual_return_pct) >= 3.8 or abs(residual_error) >= 3.8)

        # 5. Institutional Error Attribution Tagging
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
            if percentage_error <= 0.20 or absolute_error_usd < 150.0:
                tag = "SIDEWAYS_DRIFT"
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

        # 7. Index plain-English outcome into ChromaDB episodic memory
        try:
            from src.agents.rag_agent import index_reconciled_outcome_memory
            story = build_plain_english_reconciliation_story(
                pred_tick=prior_tick,
                outcome_tick=outcome_candle,
                pred_price=predicted_price,
                actual_price=actual_price,
                pred_ret_pct=predicted_return_pct,
                actual_ret_pct=actual_return_pct,
                dir_pred=dir_pred,
                dir_real=dir_real,
                dir_correct=direction_correct,
                abs_err_usd=absolute_error_usd,
                tag=tag,
            )
            regime = "NORMAL"
            if prior_tick.features and isinstance(prior_tick.features, dict):
                regime = str(prior_tick.features.get("market_regime", "NORMAL"))

            t_iso = outcome_candle.timestamp.isoformat() if hasattr(outcome_candle.timestamp, "isoformat") else str(outcome_candle.timestamp)
            index_reconciled_outcome_memory(
                tick_id=prior_tick.id,
                timestamp_iso=t_iso,
                market_regime=regime,
                direction_predicted=dir_pred,
                direction_realized=dir_real,
                direction_correct=direction_correct,
                predicted_dollar_move=predicted_price - base_price,
                actual_dollar_move=actual_price - base_price,
                absolute_error_usd=absolute_error_usd,
                error_attribution_tag=tag,
                plain_english_notes=story["hypothesis_notes"],
            )
        except Exception as rag_err:
            logger.debug("ChromaDB episodic memory indexing skipped: %s", rag_err)

        # 8. Audit Risk Critic Veto Quality (if CouncilAudit exists for this prior tick)
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
    """
    Generates a clear, plain-English post-mortem explanation of what the model predicted,
    what Bitcoin actually did 1 hour later, and why the prediction succeeded or missed.
    """
    start_price = float(pred_tick.close)
    pred_dollar_move = pred_price - start_price
    actual_dollar_move = actual_price - start_price
    candle_range_usd = float(outcome_tick.high - outcome_tick.low)

    result_word = "CORRECT" if dir_correct else "WRONG"
    headline = (
        f"[{result_word}] Predicted {dir_pred} ({pred_dollar_move:+,.2f}$ to ${pred_price:,.2f}) | "
        f"Actual {dir_real} ({actual_dollar_move:+,.2f}$ to ${actual_price:,.2f})"
    )

    if tag == "BLACK_SWAN_OUTLIER":
        why_text = (
            f"Big Market Swing Alert: Bitcoin moved ${actual_dollar_move:+,.2f} ({actual_ret_pct:+.2f}%) "
            f"with a ${candle_range_usd:,.2f} high-low range in 1 hour, driven by a sudden macro shock or liquidation cascade."
        )
    elif dir_correct and tag == "ACCURATE":
        why_text = (
            f"Spot-On Forecast: Both direction ({dir_pred}) and target price were accurate within ${abs_err_usd:,.2f}."
        )
    elif dir_correct and tag == "MOMENTUM_UNDERESTIMATED":
        why_text = (
            f"Direction Right ({dir_pred}), Stronger Move Than Expected: Bitcoin moved ${actual_dollar_move:+,.2f} "
            f"vs predicted ${pred_dollar_move:+,.2f} (error ${abs_err_usd:,.2f})."
        )
    elif dir_correct:
        why_text = (
            f"Direction Right ({dir_pred}), Smaller Move Than Expected: Bitcoin moved ${actual_dollar_move:+,.2f} "
            f"vs predicted ${pred_dollar_move:+,.2f} (error ${abs_err_usd:,.2f})."
        )
    elif tag == "SIDEWAYS_DRIFT":
        why_text = (
            f"Small Chop Flip: Model predicted {dir_pred} (${pred_dollar_move:+,.2f}), but Bitcoin drifted "
            f"${actual_dollar_move:+,.2f} ({dir_real}). Price difference was only ${abs_err_usd:,.2f} inside normal noise."
        )
    else:
        why_text = (
            f"Trend Reversal Miss: Model predicted {dir_pred} (${pred_dollar_move:+,.2f}), but sellers/buyers reversed "
            f"the candle to {dir_real} (${actual_dollar_move:+,.2f}), missing by ${abs_err_usd:,.2f}."
        )

    notes = (
        f"Hour T Close: ${start_price:,.2f} -> Predicted T+1h: ${pred_price:,.2f} ({dir_pred}, {pred_dollar_move:+,.2f}$) | "
        f"Actual T+1h Close: ${actual_price:,.2f} ({dir_real}, {actual_dollar_move:+,.2f}$) | "
        f"Dollar Error: ${abs_err_usd:,.2f} | 1h High-Low Range: ${candle_range_usd:,.2f} | Diagnosis: {why_text}"
    )
    return {
        "headline": headline[:500],
        "hypothesis_notes": notes,
        "suspected_cause_category": tag,
    }


def get_reconciliation_summary(session: Session, limit: int = 168, model_version: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes rolling performance metrics across the most recent N reconciled predictions.
    Supports optional model_version filtering via joined MarketTick.
    Used by Streamlit Terminal and FastAPI monitoring endpoints.
    
    Parameters:
        session: Active DB session
        limit: Rolling lookback window (default 168 hours = 7 days)
        model_version: Optional model version filter (e.g. 'xgb_onnx_v1.0.0').
                      If None or 'All Models', aggregates across all records.
        
    Returns:
        Dictionary containing win rate, MAE, MAPE, IC, sample counts, and applied model_version filter.
    """
    if model_version is not None and model_version != "All Models":
        stmt = (
            select(Reconciliation)
            .join(MarketTick, Reconciliation.tick_id == MarketTick.id)
            .where(MarketTick.model_version == model_version)
            .order_by(desc(Reconciliation.timestamp))
            .limit(limit)
        )
    else:
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
            "information_coefficient_ic": 0.0,
            "black_swans_filtered": 0,
            "black_swan_count": 0,
            "accurate_count": 0,
            "tag_breakdown": {},
            "model_version": model_version if (model_version and model_version != "All Models") else None,
            "query_failed": False,
            "error": None,
        }

    total = len(records)
    wins = sum(1 for r in records if r.direction_correct)
    mae = float(np.mean([r.absolute_error_usd for r in records]))
    mape = float(np.mean([r.percentage_error for r in records]))
    black_swans = sum(1 for r in records if r.is_black_swan)
    accurate = sum(1 for r in records if r.error_attribution_tag == "ACCURATE")
    tag_counts = dict(Counter(r.error_attribution_tag for r in records if r.error_attribution_tag))

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
        "information_coefficient_ic": round(ic, 4),
        "black_swans_filtered": black_swans,
        "black_swan_count": black_swans,
        "accurate_count": accurate,
        "tag_breakdown": tag_counts,
        "model_version": model_version if (model_version and model_version != "All Models") else None,
        "query_failed": False,
        "error": None,
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
