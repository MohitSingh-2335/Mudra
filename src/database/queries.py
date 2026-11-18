# src/database/queries.py
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from sqlalchemy import select, desc

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.connection import get_db
from src.database.models import MarketTick, Reconciliation, PaperPortfolio, CouncilAudit

logger = logging.getLogger(__name__)
__all__ = [
    "get_latest_market_ticks", "get_reconciliation_history", "get_reconciliation_kpis",
    "get_portfolio_history", "get_portfolio_kpis", "get_latest_council_audit",
]


def get_latest_market_ticks(limit: int = 100) -> pd.DataFrame:
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
    try:
        from src.database.models import ModelPrediction
        with get_db() as session:
            stmt = select(Reconciliation, MarketTick.model_version).outerjoin(MarketTick, Reconciliation.tick_id == MarketTick.id)
            if model_version and model_version != "All Models":
                stmt = stmt.where(MarketTick.model_version == model_version)
            rows = session.execute(stmt.order_by(desc(Reconciliation.timestamp)).limit(limit)).all()

            data = []
            for r, m_ver in rows:
                r_dict = r.to_dict()
                r_dict["model_version"] = m_ver
                data.append(r_dict)

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
            df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
            return df.head(limit) if limit and limit > 0 else df
    except Exception as exc:
        logger.warning("Failed to fetch reconciliation history: %s", type(exc).__name__)
        return pd.DataFrame()


def get_reconciliation_kpis(lookback_hours: Optional[int] = 168, model_version: Optional[str] = None) -> Dict[str, Any]:
    try:
        from src.engine.reconciliation import get_reconciliation_summary
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
    try:
        from src.engine.paper_portfolio import get_portfolio_summary
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
    try:
        with get_db() as session:
            audit = session.execute(select(CouncilAudit).order_by(desc(CouncilAudit.id)).limit(1)).scalar_one_or_none()
            return audit.to_dict() if audit else None
    except Exception as exc:
        logger.warning("Failed to fetch latest council audit: %s", type(exc).__name__)
        return None
