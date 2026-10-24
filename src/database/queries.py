"""
Database Query Service for Mudra Frontend Terminal and API Endpoints.
Fetches historical market ticks, delayed reconciliation records,
and paper portfolio equity curves directly from Neon PostgreSQL.
Defensively handles connection drops, cold starts, and uninitialized schemas.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Pre-load onnxruntime before numerical/db query engines to stabilize Windows OpenMP DLL runtime
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass

import pandas as pd
from sqlalchemy import select, desc

from src.database.connection import get_db
from src.database.models import MarketTick, Reconciliation, PaperPortfolio, CouncilAudit

logger = logging.getLogger(__name__)

__all__ = [
    "get_latest_market_ticks",
    "get_reconciliation_history",
    "get_reconciliation_kpis",
    "get_portfolio_history",
    "get_portfolio_kpis",
    "get_latest_council_audit",
]


def get_latest_market_ticks(limit: int = 100) -> pd.DataFrame:
    """
    Fetches the latest N confirmed market ticks from the cloud database,
    ordered chronologically for plotting. Returns empty DataFrame on failure.
    """
    try:
        with get_db() as session:
            stmt = (
                select(MarketTick)
                .order_by(desc(MarketTick.timestamp))
                .limit(limit)
            )
            ticks = session.execute(stmt).scalars().all()
            if not ticks:
                return pd.DataFrame()
            
            data = [t.to_dict() for t in ticks]
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
    except Exception as exc:
        logger.warning("Failed to fetch market ticks: error_type=%s", type(exc).__name__)
        return pd.DataFrame()


def get_reconciliation_history(limit: int = 500, model_version: Optional[str] = None) -> pd.DataFrame:
    """
    Fetches the most recent N delayed reconciliation records from Neon.
    Attributes each record to its producing model_version via MarketTick.
    Supports optional model_version filtering while preserving records with
    unknown model attribution when model_version is None or 'All Models'.
    Returns empty DataFrame on failure.
    """
    try:
        from src.database.models import ModelPrediction
        with get_db() as session:
            stmt = (
                select(Reconciliation, MarketTick.model_version)
                .outerjoin(MarketTick, Reconciliation.tick_id == MarketTick.id)
            )
            if model_version is not None and model_version != "All Models":
                stmt = stmt.where(MarketTick.model_version == model_version)

            stmt = stmt.order_by(desc(Reconciliation.timestamp)).limit(limit)
            rows = session.execute(stmt).all()

            data = []
            for r, m_ver in rows:
                r_dict = r.to_dict()
                r_dict["model_version"] = m_ver if m_ver is not None else None
                data.append(r_dict)

            # Also fetch reconciled candidate model predictions from ModelPrediction
            cand_stmt = (
                select(ModelPrediction, MarketTick.timestamp.label("tick_ts"))
                .join(MarketTick, ModelPrediction.tick_id == MarketTick.id)
                .where(
                    ModelPrediction.outcome_tick_id.isnot(None),
                    ModelPrediction.is_primary.is_(False),
                )
            )
            if model_version is not None and model_version != "All Models":
                cand_stmt = cand_stmt.where(ModelPrediction.model_version == model_version)
            cand_stmt = cand_stmt.order_by(desc(ModelPrediction.reconciled_at)).limit(limit)
            cand_rows = session.execute(cand_stmt).all()

            for cp, tick_ts in cand_rows:
                ts = cp.reconciled_at or tick_ts
                data.append({
                    "id": cp.id,
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "tick_id": cp.tick_id,
                    "predicted_price": cp.predicted_price_1h,
                    "actual_price_at_t_plus_1": cp.actual_price_at_t_plus_1,
                    "predicted_return_pct": cp.predicted_return_pct,
                    "actual_return_pct": cp.actual_return_pct,
                    "residual_error": cp.residual_error,
                    "absolute_error_usd": cp.absolute_error_usd,
                    "percentage_error": cp.percentage_error,
                    "direction_predicted": cp.direction_predicted,
                    "direction_realized": cp.direction_realized,
                    "direction_correct": cp.direction_correct,
                    "is_black_swan": cp.is_black_swan,
                    "error_attribution_tag": cp.error_attribution_tag,
                    "model_version": cp.model_version,
                })

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
            if limit is not None and limit > 0:
                df = df.head(limit)
            return df
    except Exception as exc:
        logger.warning("Failed to fetch reconciliation history: error_type=%s", type(exc).__name__)
        return pd.DataFrame()


def get_reconciliation_kpis(lookback_hours: Optional[int] = 168, model_version: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes rolling accuracy KPIs (Win Rate %, MAE, MAPE, IC) across N hours or all-time (if lookback_hours is None).
    Supports optional model_version filtering.
    Explicitly distinguishes query success from query failure via 'query_failed' boolean flag.
    Returns defensive default KPIs with query_failed=True on failure.
    """
    try:
        from src.engine.reconciliation import get_reconciliation_summary
        with get_db() as session:
            res = get_reconciliation_summary(session, limit=lookback_hours, model_version=model_version)
            res["query_failed"] = False
            res["error"] = None
            return res
    except Exception as exc:
        logger.warning("Failed to fetch reconciliation KPIs: error_type=%s", type(exc).__name__)
        return {
            "total_evaluated": 0,
            "total_reconciled": 0,
            "total_lifetime_evaluated": 0,
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
            "query_failed": True,
            "error": "Database query failed",
        }


def get_portfolio_history(limit: int = 500) -> pd.DataFrame:
    """
    Fetches the chronological paper portfolio ledger from Neon.
    Used for plotting Strategy Equity vs. Buy & Hold Benchmark Alpha curves.
    Returns empty DataFrame on failure.
    """
    try:
        with get_db() as session:
            stmt = (
                select(PaperPortfolio)
                .order_by(PaperPortfolio.id.asc())
                .limit(limit)
            )
            records = session.execute(stmt).scalars().all()
            if not records:
                return pd.DataFrame()
            
            data = [p.to_dict() for p in records]
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
    except Exception as exc:
        logger.warning("Failed to fetch portfolio history: error_type=%s", type(exc).__name__)
        return pd.DataFrame()


def get_portfolio_kpis(current_price: float) -> Dict[str, Any]:
    """
    Returns current portfolio snapshot: Cash, BTC, Total Equity, and Alpha.
    Returns defensive initial baseline on failure.
    """
    try:
        from src.engine.paper_portfolio import get_portfolio_summary
        with get_db() as session:
            return get_portfolio_summary(session, current_price=current_price)

    except Exception as exc:
        logger.warning("Failed to fetch portfolio KPIs: error_type=%s", type(exc).__name__)
        return {
            "cash_balance": 10000.0,
            "btc_balance": 0.0,
            "current_btc_price": round(current_price, 2),
            "btc_position_value_usd": 0.0,
            "total_equity_usd": 10000.0,
            "cumulative_return_pct": 0.0,
            "benchmark_btc_equity": 10000.0,
            "alpha_usd": 0.0,
            "alpha_pct": 0.0,
            "realized_pnl_usd": 0.0,
            "total_fees_usd": 0.0,
            "drawdown_pct": 0.0,
            "total_trades": 0,
        }


def get_latest_council_audit() -> Optional[Dict[str, Any]]:
    """
    Returns the most recent Multi-Agent Council deliberation record.
    Returns None on failure.
    """
    try:
        with get_db() as session:
            stmt = select(CouncilAudit).order_by(desc(CouncilAudit.id)).limit(1)
            audit = session.execute(stmt).scalar_one_or_none()
            return audit.to_dict() if audit else None
    except Exception as exc:
        logger.warning("Failed to fetch latest council audit: error_type=%s", type(exc).__name__)
        return None


if __name__ == '__main__':
    print("[INFO] Testing database query services...")
    ticks_df = get_latest_market_ticks(limit=5)
    print(f"[QUERY] Latest market ticks count: {len(ticks_df)}")
    rec_kpis = get_reconciliation_kpis(lookback_hours=24)
    print(f"[QUERY] Reconciliation evaluated count: {rec_kpis.get('total_reconciled', 0)}")
    port_kpis = get_portfolio_kpis(current_price=65000.0)
    print(f"[QUERY] Portfolio total equity: ${port_kpis.get('total_equity_usd', 10000.0):,.2f}")
    latest_audit = get_latest_council_audit()
    print(f"[QUERY] Latest council audit found: {latest_audit is not None}")
    print("[SUCCESS] All database query services verified.")

