"""
Mudra 24/7 Hourly Headless Cron Worker.
Executes the full quant lifecycle: market ingestion, API telemetry,
delayed ground truth reconciliation, multi-agent council deliberation,
paper portfolio execution, and outbox synchronization.
"""
# PRE-LOAD ONNXRUNTIME FIRST to prevent Windows OpenMP (0xC0000005) thread collision
try:
    import onnxruntime
except ImportError:
    pass
import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import pandas as pd
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import get_db, init_db, test_connection
from src.database.models import (
    MarketTick,
    ApiTelemetry,
    CouncilAudit,
    Reconciliation,
    PaperPortfolio,
)
from src.engine.reconciliation import reconcile_pending_ticks, get_reconciliation_summary
from src.engine.paper_portfolio import execute_paper_order, get_portfolio_summary
from src.agents.trading_graph import run_trading_graph
from src.agents.tools import calculate_technical_signals, get_model_return_forecast

OUTBOX_FILE = Path("data/sync_outbox.json")


def log_telemetry(session, service: str, endpoint: str, latency_ms: float, status_code: int, error_msg: Optional[str] = None):
    """Logs API observability record."""
    try:
        record = ApiTelemetry(
            timestamp=datetime.now(timezone.utc),
            service_name=service,
            endpoint=endpoint,
            latency_ms=latency_ms,
            status_code=status_code,
            error_message=error_msg,
        )
        session.add(record)
        session.flush()
    except Exception as e:
        print(f"[WARN] Failed to write API telemetry: {e}")


def fetch_latest_candle(symbol: str = "BTCUSDT", session=None) -> Dict[str, Any]:
    """
    Fetches the latest confirmed closed 1-hour candle from Binance Dedicated Vision
    with institutional order-flow attributes.
    """
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit=5"
    start_t = time.perf_counter()
    
    try:
        resp = requests.get(url, timeout=10)
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        
        if session:
            log_telemetry(session, "binance_vision", "/api/v3/klines", latency_ms, resp.status_code)
            
        if resp.status_code == 200:
            klines = resp.json()
            if len(klines) >= 2:
                # klines[-2] is the most recently confirmed CLOSED candle
                # klines[-1] is the currently active open candle
                c = klines[-2]
                ts = datetime.fromtimestamp(c[0] / 1000.0, tz=timezone.utc)
                o, h, l, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
                vol = float(c[5])
                taker_base = float(c[9])
                taker_ratio = float(taker_base / vol) if vol > 0 else 0.5
                trades = int(c[8])
                
                return {
                    "timestamp": ts,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": cl,
                    "volume": vol,
                    "taker_buy_ratio": round(taker_ratio, 4),
                    "number_of_trades": trades,
                }
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        if session:
            log_telemetry(session, "binance_vision", "/api/v3/klines", latency_ms, 500, str(exc))
        raise RuntimeError(f"Failed to fetch Binance candle: {exc}")

    raise RuntimeError("Received empty klines response from Binance.")


def flush_outbox(session) -> int:
    """Flushes cached records from data/sync_outbox.json if present."""
    if not OUTBOX_FILE.exists():
        return 0
    try:
        data = json.loads(OUTBOX_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            return 0
        
        flushed_count = 0
        for item in data:
            if item.get("type") == "tick":
                tick = MarketTick(**item["payload"])
                session.add(tick)
                flushed_count += 1
                
        session.flush()
        OUTBOX_FILE.unlink(missing_ok=True)
        print(f"[OUTBOX] Successfully flushed {flushed_count} cached items to Neon cloud.")
        return flushed_count
    except Exception as exc:
        print(f"[WARN] Failed to flush outbox: {exc}")
        return 0


def run_worker():
    """Executes the complete hourly quantitative pipeline."""
    print(f"\n{'='*70}")
    print(f"Mudra Hourly Worker Starting at {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*70}")

    # 1. Initialize Tables in Neon Cloud
    init_db()
    
    # 2. Test Connection
    ok, ver, lat = test_connection()
    if not ok:
        print(f"[FATAL] Cannot reach Database: {ver}")
        sys.exit(1)
    print(f"[DB] Connected to {ver[:35]}... (Latency: {lat:.1f}ms)")

    with get_db() as session:
        # 3. Flush any pending offline queue items
        flush_outbox(session)

        # 4. Ingest Latest Closed Candle
        print("[INGEST] Streaming latest confirmed 1-hour candle from Binance...")
        candle = fetch_latest_candle(session=session)
        print(f"[MARKET] BTC Price: ${candle['close']:,.2f} | Taker Ratio: {candle['taker_buy_ratio']} | Vol: {candle['volume']:,.2f}")

        # 5. Calculate Technical Indicators & ML Forecast
        print("[FEATURE] Computing technical signals & XGBoost forecast...")
        signals_output = calculate_technical_signals.invoke({
            "close": candle["close"],
            "open_price": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "volume": candle["volume"],
            "taker_buy_ratio": candle.get("taker_buy_ratio", 0.5),
            "timestamp": candle["timestamp"],
        })
        ml_output = get_model_return_forecast.invoke({"features": signals_output})
        
        pred_return = float(ml_output.get("expected_return_pct", 0.0))
        pred_price = float(ml_output.get("predicted_target_price", candle["close"]))
        confidence = float(ml_output.get("confidence", 0.50))
        
        clears_hurdle = ml_output.get("clears_hurdle", False)
        volatility = float(signals_output.get("volatility_range_pct", 0.02))
        regime = "HIGH_VOL_CHOP" if volatility > 3.8 else ("BULL_MOMENTUM" if (clears_hurdle and pred_return > 0) else ("BEAR_DISTRIBUTION" if (clears_hurdle and pred_return < 0) else "SIDEWAYS_RANGE"))

        # 6. Check if tick already recorded
        existing_tick = session.query(MarketTick).filter(MarketTick.timestamp == candle["timestamp"]).first()
        if existing_tick:
            current_tick = existing_tick
            print(f"[DB] Tick at {candle['timestamp']} already exists in database (ID: {current_tick.id}).")
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
            print(f"[DB] Recorded MarketTick ID: {current_tick.id} | Regime: {regime}")

        # 7. Delayed Ground Truth Reconciliation (Reconciles T-1 against current T)
        print("[RECONCILIATION] Evaluating prior predictions at T-1 against current reality...")
        reconciled = reconcile_pending_ticks(session, current_tick)
        if reconciled:
            for r in reconciled:
                result_icon = "WIN" if r.direction_correct else "LOSS"
                print(f"  └─ [{result_icon}] Prior Tick #{r.tick_id}: Pred ${r.predicted_price:,.2f} vs Actual ${r.actual_price_at_t_plus_1:,.2f} | Error: ${r.absolute_error_usd:,.2f} ({r.percentage_error:.2f}%) | Tag: {r.error_attribution_tag}")
        else:
            print("  └─ No pending unsettled predictions to evaluate.")

        # 8. Multi-Agent Council Deliberation
        print("[COUNCIL] Convening Multi-Agent Council (Strategist + Risk Critic)...")
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

        print(f"[COUNCIL] Strategist: {action} (Rating: {conviction}/10) | Critic Verdict: {verdict} | Final: {decision} | Kelly: {kelly*100:.1f}% (Deliberation: {delib_ms:.0f}ms)")
        if veto_reasons:
            print(f"  └─ Veto Reasons: {veto_reasons}")

        # Record Council Audit
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

        # 9. Execute in Paper Trading Portfolio Engine
        print(f"[PORTFOLIO] Executing Forward Paper Order ({decision})...")
        order_notes = f"Executed {decision} based on Council verdict ({verdict}) at BTC ${candle['close']:,.2f}."
        portfolio_record = execute_paper_order(
            session=session,
            action=decision,
            current_price=candle["close"],
            kelly_fraction=kelly,
            notes=order_notes,
        )

        summary = get_portfolio_summary(session, candle["close"])
        print(f"[PORTFOLIO] Total Equity: ${summary['total_equity_usd']:,.2f} | Cash: ${summary['cash_balance']:,.2f} | BTC: {summary['btc_balance']:.6f}")
        print(f"[ALPHA] Strategy Return: {summary['cumulative_return_pct']:+.2f}% | Benchmark: ${summary['benchmark_btc_equity']:,.2f} | Alpha: ${summary['alpha_usd']:+,.2f} ({summary['alpha_pct']:+.2f}%)")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    run_worker()
