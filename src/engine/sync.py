"""
Mudra Quant Synchronization Engine.
Provides a unified, thread-safe sync pipeline for both the 24/7 headless cron worker
and on-demand Streamlit UI terminal triggers.
"""
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Pre-load onnxruntime before numerical engines to prevent Windows OpenMP runtime collision (0xC0000005)
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass

import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import requests
from sqlalchemy import select, desc

from src.database.connection import get_db, test_connection
from src.database.models import MarketTick, ApiTelemetry, CouncilAudit
from src.engine.reconciliation import reconcile_pending_ticks
from src.engine.paper_portfolio import execute_paper_order, get_portfolio_summary
from src.agents.trading_graph import run_trading_graph
from src.agents.tools import calculate_technical_signals, get_model_return_forecast

__all__ = ["sync_hourly_quant_cycle", "fetch_latest_confirmed_candle"]


def fetch_latest_confirmed_candle(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Fetches the most recently confirmed closed 1-hour candle alongside
    the preceding sequence of historical candles for rolling feature parity.
    Primary: Binance Dedicated Vision. Fallback: Coinbase Institutional.
    """
    # 1. Binance Dedicated Vision
    try:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit=102"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            klines = resp.json()
            now_ms = datetime.now(timezone.utc).timestamp() * 1000.0
            confirmed_klines = [k for k in klines if float(k[6]) <= now_ms]
            if len(confirmed_klines) == len(klines) and len(klines) > 1:
                confirmed_klines = klines[:-1]

            if len(confirmed_klines) >= 35:
                selected_klines = confirmed_klines[-100:]
                c = selected_klines[-1]
                ts = datetime.fromtimestamp(c[0] / 1000.0, tz=timezone.utc)
                vol = float(c[5])
                taker_base = float(c[9])
                taker_ratio = float(taker_base / vol) if vol > 0 else 0.50

                candle_records = []
                for k in selected_klines:
                    k_vol = float(k[5])
                    k_taker = float(k[9])
                    k_ratio = float(k_taker / k_vol) if k_vol > 0 else 0.50
                    candle_records.append({
                        "timestamp": pd.to_datetime(k[0], unit="ms", utc=True),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": k_vol,
                        "number_of_trades": int(k[8]),
                        "taker_buy_ratio": round(k_ratio, 4),
                    })

                return {
                    "timestamp": ts,
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": vol,
                    "taker_buy_ratio": round(taker_ratio, 4),
                    "number_of_trades": int(c[8]),
                    "candles": candle_records,
                }
    except Exception:
        pass

    # 2. Coinbase Institutional Fallback
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
        resp = requests.get(url, headers={"User-Agent": "Mudra-Sync/1.0"}, timeout=10)
        if resp.status_code == 200:
            candles = resp.json()
            now_sec = datetime.now(timezone.utc).timestamp()
            confirmed_candles = [k for k in candles if (float(k[0]) + 3600.0) <= now_sec]
            if len(confirmed_candles) == len(candles) and len(candles) > 1:
                confirmed_candles = candles[1:]

            if len(confirmed_candles) >= 35:
                selected_candles = list(reversed(confirmed_candles[:100]))
                c = selected_candles[-1]
                ts = datetime.fromtimestamp(c[0], tz=timezone.utc)

                candle_records = []
                for k in selected_candles:
                    candle_records.append({
                        "timestamp": pd.to_datetime(k[0], unit="s", utc=True),
                        "low": float(k[1]),
                        "high": float(k[2]),
                        "open": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "taker_buy_ratio": 0.50,
                        "number_of_trades": None,  # Coinbase does not report trade counts; unobserved
                    })

                return {
                    "timestamp": ts,
                    "open": float(c[3]),
                    "high": float(c[2]),
                    "low": float(c[1]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                    "taker_buy_ratio": 0.50,
                    "number_of_trades": None,  # Coinbase does not report trade counts; unobserved
                    "candles": candle_records,
                }
    except Exception:
        pass

    raise RuntimeError("Failed to fetch confirmed 1-hour candle from all market feeds.")


def sync_hourly_quant_cycle(force: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Executes the full quantitative pipeline:
    Ingestion -> Signals -> XGBoost Forecast -> Reconciliation -> Council Deliberation -> Paper Order -> Neon Commit.
    Returns: (was_synced, status_message, summary_dict)
    """
    db_ok, ver, _ = test_connection()
    if not db_ok:
        return False, f"Database unreachable: {ver}", {}

    try:
        candle = fetch_latest_confirmed_candle()
    except Exception as exc:
        return False, f"Market feed error: {exc}", {}

    with get_db() as session:
        existing = session.execute(
            select(MarketTick).where(MarketTick.timestamp == candle["timestamp"]).order_by(MarketTick.id.desc())
        ).scalars().first()

        if existing and not force:
            return False, f"Candle for {candle['timestamp'].strftime('%H:00 UTC')} already recorded in Neon.", {"tick_id": existing.id}

        # 1. Compute Signals and XGBoost Return Forecast
        signals_output = calculate_technical_signals.invoke({
            "close": candle["close"],
            "open_price": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "volume": candle["volume"],
            "taker_buy_ratio": candle.get("taker_buy_ratio"),
            "timestamp": candle["timestamp"],
            "candles": candle.get("candles"),
        })
        ml_output = get_model_return_forecast.invoke({"features": signals_output})
        pred_return = float(ml_output.get("expected_return_pct", 0.0))
        pred_price = float(ml_output.get("predicted_target_price", candle["close"]))
        confidence = float(ml_output.get("confidence", 0.50))

        clears_hurdle = ml_output.get("clears_hurdle", False)
        volatility = float(signals_output.get("volatility_range_pct", 0.02))
        regime = "HIGH_VOL_CHOP" if volatility > 3.8 else ("BULL_MOMENTUM" if (clears_hurdle and pred_return > 0) else ("BEAR_DISTRIBUTION" if (clears_hurdle and pred_return < 0) else "SIDEWAYS_RANGE"))

        # 2. Record or Update MarketTick
        if existing:
            current_tick = existing
            if force:
                current_tick.market_regime = regime
                current_tick.predicted_return_1h = pred_return
                current_tick.predicted_price_1h = pred_price
                current_tick.model_confidence = confidence
                current_tick.features = signals_output
                session.flush()
        else:
            current_tick = MarketTick(
                timestamp=candle["timestamp"],
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle["volume"],
                model_version="xgb_onnx_v1.0.0",
                feature_schema_version="v1_39features",
                market_regime=regime,
                predicted_return_1h=pred_return,
                predicted_price_1h=pred_price,
                model_confidence=confidence,
                features=signals_output,
            )
            session.add(current_tick)
            session.flush()

        # 3. Delayed Ground Truth Reconciliation (Evaluates T-1 against T)
        reconcile_pending_ticks(session, current_tick)

        # 4. Multi-Agent Council Deliberation
        start_delib = time.perf_counter()
        agent_state = run_trading_graph(
            symbol="BTCUSDT",
            current_price=candle["close"],
            market_data=candle,
            engineered_features=signals_output,
            ml_forecast=ml_output,
        )
        delib_ms = (time.perf_counter() - start_delib) * 1000.0

        decision = agent_state.get("final_decision", "HOLD")
        verdict = agent_state.get("risk_verdict", "APPROVED")
        action = agent_state.get("strategist_action", "HOLD")
        conviction = agent_state.get("strategist_conviction", 5)
        kelly = float(agent_state.get("kelly_fraction", 0.10))
        veto_reasons = agent_state.get("veto_reasons", [])

        # 5. Record Council Audit
        audit = CouncilAudit(
            timestamp=datetime.now(timezone.utc),
            tick_id=current_tick.id,
            deliberation_latency_ms=delib_ms,
            strategist_action=action,
            strategist_conviction=conviction,
            rag_context_summary=agent_state.get("rag_context", "")[:300],
            risk_verdict=verdict,
            veto_reasons=veto_reasons,
            kelly_fraction=kelly,
            final_decision=decision,
            execution_notes=agent_state.get("thesis", "")[:500],
        )
        session.add(audit)
        session.flush()

        # 6. Execute Forward Paper Order
        order_notes = f"Executed {decision} ({verdict}) at BTC ${candle['close']:,.2f}."
        execute_paper_order(
            session=session,
            action=decision,
            current_price=candle["close"],
            kelly_fraction=kelly,
            notes=order_notes,
        )

        summary = get_portfolio_summary(session, candle["close"])
        return True, f"Recorded tick #{current_tick.id} at ${candle['close']:,.2f} ({decision} | Kelly: {kelly*100:.1f}%)", summary


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    print(f"Triggering sync cycle (force={force_flag})...")
    success, msg, summ = sync_hourly_quant_cycle(force=force_flag)
    print(f"Status: {success} | Message: {msg}")
    if summ:
        print(f"Portfolio Summary: {summ}")
