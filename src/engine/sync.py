"""
Mudra Quant Synchronization Engine.
Provides a unified, thread-safe sync pipeline for both the 24/7 headless cron worker
and on-demand Streamlit UI terminal triggers.
"""
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


def fetch_latest_confirmed_candle(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Fetches the most recently confirmed closed 1-hour candle.
    Primary: Binance Dedicated Vision. Fallback: Coinbase Institutional.
    """
    # 1. Binance Dedicated Vision
    try:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit=5"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            klines = resp.json()
            if len(klines) >= 2:
                # klines[-2] is the most recently confirmed CLOSED candle
                c = klines[-2]
                ts = datetime.fromtimestamp(c[0] / 1000.0, tz=timezone.utc)
                vol = float(c[5])
                taker_base = float(c[9])
                taker_ratio = float(taker_base / vol) if vol > 0 else 0.5
                return {
                    "timestamp": ts,
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": vol,
                    "taker_buy_ratio": round(taker_ratio, 4),
                    "number_of_trades": int(c[8]),
                }
    except Exception:
        pass

    # 2. Coinbase Institutional Fallback
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
        resp = requests.get(url, headers={"User-Agent": "Mudra-Sync/1.0"}, timeout=10)
        if resp.status_code == 200:
            candles = resp.json()
            if len(candles) >= 2:
                # candles[1] is the previous closed hourly candle
                c = candles[1]
                ts = datetime.fromtimestamp(c[0], tz=timezone.utc)
                return {
                    "timestamp": ts,
                    "open": float(c[3]),
                    "high": float(c[2]),
                    "low": float(c[1]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                    "taker_buy_ratio": 0.50,
                    "number_of_trades": 10000,
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
            "high": candle["high"],
            "low": candle["low"],
            "volume": candle["volume"],
            "taker_buy_ratio": candle.get("taker_buy_ratio", 0.5),
        })
        ml_output = get_model_return_forecast.invoke({"features": signals_output})
        pred_return = float(ml_output.get("expected_return_pct", 0.0))
        pred_price = candle["close"] * (1.0 + (pred_return / 100.0))
        confidence = float(ml_output.get("confidence", 0.50))

        volatility = float(signals_output.get("volatility_range_pct", 0.02))
        regime = "HIGH_VOL_CHOP" if volatility > 3.8 else ("BULL_MOMENTUM" if pred_return > 0.05 else ("BEAR_DISTRIBUTION" if pred_return < -0.05 else "SIDEWAYS_RANGE"))

        # 2. Record or Update MarketTick
        if existing:
            current_tick = existing
        else:
            current_tick = MarketTick(
                timestamp=candle["timestamp"],
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle["volume"],
                taker_buy_ratio=candle["taker_buy_ratio"],
                number_of_trades=candle["number_of_trades"],
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
