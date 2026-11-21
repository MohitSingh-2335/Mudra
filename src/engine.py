# src/engine.py
"""
Mudra Execution & Hourly Sync Engine.
Handles paper portfolio accounting, Binance/Coinbase live ingestion, multi-agent execution, and delayed T+1h reconciliation.
"""
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from sqlalchemy import and_, desc, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import (
    ACTIVE_MODEL_VERSION,
    EXCHANGE_CONNECT_TIMEOUT,
    EXCHANGE_MAX_RETRY_AFTER_CAP,
    EXCHANGE_READ_TIMEOUT,
    EXCHANGE_RETRY_BACKOFF_FACTOR,
    EXCHANGE_RETRY_BASE_BACKOFF,
    EXCHANGE_RETRY_MAX_ATTEMPTS,
    EXCHANGE_RETRY_MAX_BACKOFF,
    EXCHANGE_RETRY_TOTAL_DEADLINE,
    FEATURE_SCHEMA_VERSION,
    INITIAL_CAPITAL_USD,
    MIN_HURDLE_PCT,
    PORTFOLIO_ADVISORY_LOCK_ID,
    ROUND_TRIP_FRICTION_PCT,
    SLIPPAGE_PCT,
    SLIPPAGE_RATE,
    TAKER_FEE_PCT,
    TAKER_FEE_RATE,
    TOTAL_FRICTION_PER_SIDE_PCT,
)
from src.agents import (
    calculate_technical_signals,
    get_model_return_forecast,
    run_trading_graph,
)
from src.database import (
    CouncilAudit, MarketTick, ModelPrediction, PaperPortfolio, Reconciliation,
    get_db, has_tick_id_column, test_connection,
)

logger = logging.getLogger("mudra.engine")
sync_logger = logging.getLogger("mudra.sync")

BINANCE_MIN_ROW_LEN: int = 10
COINBASE_MIN_ROW_LEN: int = 6
BACKFILL_LARGE_GAP_WARNING_HOURS: float = 24.0


# ---------------------------------------------------------------------------
# 1. PAPER PORTFOLIO & SPOT HOLD DISCIPLINE
# ---------------------------------------------------------------------------

def get_average_entry_price(session: Session, target_btc: float, include_fees: bool = True) -> float:
    if target_btc <= 1e-8:
        return 0.0
    last_sell = session.execute(
        select(PaperPortfolio.id).where(PaperPortfolio.action == "SELL").order_by(PaperPortfolio.id.desc()).limit(1)
    ).scalar_one_or_none()

    stmt = select(PaperPortfolio.execution_price, PaperPortfolio.btc_amount, PaperPortfolio.fee_deducted_usd).where(PaperPortfolio.action == "BUY")
    if last_sell is not None:
        stmt = stmt.where(PaperPortfolio.id > last_sell)

    records = session.execute(stmt.order_by(PaperPortfolio.id.desc())).all()
    accum_btc, accum_cost = 0.0, 0.0
    for price, amount, fee in records:
        if amount <= 0:
            continue
        needed = min(amount, target_btc - accum_btc)
        accum_cost += needed * ((price + (fee / amount)) if include_fees else price)
        accum_btc += needed
        if accum_btc >= target_btc - 1e-8:
            break
    return (accum_cost / accum_btc) if accum_btc > 0 else 0.0


def get_latest_portfolio_state(session: Session) -> Optional[PaperPortfolio]:
    return session.execute(select(PaperPortfolio).order_by(desc(PaperPortfolio.id)).limit(1)).scalar_one_or_none()


def get_baseline_btc_price(session: Session) -> Optional[float]:
    rec = session.execute(select(PaperPortfolio).order_by(PaperPortfolio.id.asc()).limit(1)).scalar_one_or_none()
    return float(rec.execution_price) if rec and rec.execution_price > 0 else None


def execute_paper_order(session: Session, action: str, current_price: float, kelly_fraction: float = 0.10, notes: Optional[str] = None, tick_id: Optional[int] = None) -> PaperPortfolio:
    bind = session.get_bind()
    if bind and "postgresql" in bind.dialect.name:
        session.execute(text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": PORTFOLIO_ADVISORY_LOCK_ID})

    if tick_id is not None and has_tick_id_column(session):
        existing = session.execute(select(PaperPortfolio).where(PaperPortfolio.tick_id == tick_id)).scalar_one_or_none()
        if existing is not None:
            return existing

    current_price = max(1.0, float(current_price))
    latest = get_latest_portfolio_state(session)
    baseline_price = get_baseline_btc_price(session) or current_price

    if latest is None:
        init_entry = PaperPortfolio(
            timestamp=datetime.now(timezone.utc), action="INITIAL_DEPOSIT", btc_amount=0.0,
            execution_price=current_price, fee_deducted_usd=0.0, slippage_usd=0.0,
            cash_balance=INITIAL_CAPITAL_USD, btc_balance=0.0, total_equity_usd=INITIAL_CAPITAL_USD,
            realized_pnl_usd=0.0, unrealized_pnl_usd=0.0, drawdown_pct=0.0, cumulative_return_pct=0.0,
            benchmark_btc_equity=INITIAL_CAPITAL_USD, alpha_usd=0.0, alpha_pct=0.0,
            notes="Initialized paper portfolio with $10,000.00 USDT.",
        )
        session.add(init_entry)
        session.flush()
        latest = init_entry

    cash, btc = float(latest.cash_balance), float(latest.btc_balance)
    action = action.upper().strip()
    btc_delta, fee_usd, slippage_usd, realized_pnl = 0.0, 0.0, 0.0, 0.0
    exec_price = current_price

    if action == "BUY" and cash >= 10.0 and kelly_fraction > 0:
        spend = cash * min(max(kelly_fraction, 0.05), 0.20)
        exec_price = current_price * (1.0 + SLIPPAGE_RATE)
        slippage_usd = spend * SLIPPAGE_RATE
        fee_usd = spend * TAKER_FEE_RATE
        btc_bought = (spend - fee_usd) / exec_price
        cash -= spend
        btc += btc_bought
        btc_delta = btc_bought
    elif action == "SELL" and btc > 1e-6:
        exec_price = current_price * (1.0 - SLIPPAGE_RATE)
        gross = btc * exec_price
        slippage_usd = (btc * current_price) * SLIPPAGE_RATE
        fee_usd = gross * TAKER_FEE_RATE
        net_proceeds = gross - fee_usd
        avg_entry = get_average_entry_price(session, btc, include_fees=True)
        realized_pnl = net_proceeds - (btc * avg_entry)
        cash += net_proceeds
        btc_delta = -btc
        btc = 0.0
    else:
        action = "HOLD"

    open_entry = get_average_entry_price(session, btc) if btc > 0 else 0.0
    unrealized_pnl = btc * (current_price - open_entry) if btc > 0 else 0.0
    total_equity = cash + (btc * current_price)
    cum_ret = ((total_equity - INITIAL_CAPITAL_USD) / INITIAL_CAPITAL_USD) * 100.0
    bench_eq = INITIAL_CAPITAL_USD * (current_price / baseline_price) if baseline_price > 0 else INITIAL_CAPITAL_USD
    alpha_usd = total_equity - bench_eq
    alpha_pct = ((total_equity - bench_eq) / bench_eq) * 100.0 if bench_eq > 0 else 0.0

    ath = session.execute(select(PaperPortfolio.total_equity_usd).order_by(desc(PaperPortfolio.total_equity_usd)).limit(1)).scalar() or INITIAL_CAPITAL_USD
    drawdown_pct = ((max(float(ath), total_equity) - total_equity) / max(float(ath), total_equity)) * 100.0

    record = PaperPortfolio(
        tick_id=tick_id, timestamp=datetime.now(timezone.utc), action=action, btc_amount=btc_delta,
        execution_price=exec_price, fee_deducted_usd=fee_usd, slippage_usd=slippage_usd,
        cash_balance=cash, btc_balance=btc, total_equity_usd=total_equity,
        realized_pnl_usd=realized_pnl, unrealized_pnl_usd=unrealized_pnl, drawdown_pct=drawdown_pct,
        cumulative_return_pct=cum_ret, benchmark_btc_equity=bench_eq, alpha_usd=alpha_usd,
        alpha_pct=alpha_pct, notes=notes,
    )
    try:
        with session.begin_nested():
            session.add(record)
            session.flush()
    except IntegrityError:
        if tick_id is not None and has_tick_id_column(session):
            existing = session.execute(select(PaperPortfolio).where(PaperPortfolio.tick_id == tick_id)).scalar_one_or_none()
            if existing is not None:
                return existing
        raise
    return record


def get_portfolio_summary(session: Session, current_price: float) -> Dict[str, Any]:
    current_price = max(1.0, float(current_price))
    latest = get_latest_portfolio_state(session)
    if latest is None:
        return {
            "cash_balance": INITIAL_CAPITAL_USD, "btc_balance": 0.0, "current_btc_price": round(current_price, 2),
            "btc_position_value_usd": 0.0, "total_equity_usd": INITIAL_CAPITAL_USD, "cumulative_return_pct": 0.0,
            "benchmark_btc_equity": INITIAL_CAPITAL_USD, "alpha_usd": 0.0, "alpha_pct": 0.0,
            "realized_pnl_usd": 0.0, "total_fees_usd": 0.0, "drawdown_pct": 0.0, "total_trades": 0,
        }

    trade_count = int(session.execute(select(func.count(PaperPortfolio.id)).where(PaperPortfolio.action.in_(["BUY", "SELL"]))).scalar() or 0)
    pnl, fees = session.execute(select(func.coalesce(func.sum(PaperPortfolio.realized_pnl_usd), 0.0), func.coalesce(func.sum(PaperPortfolio.fee_deducted_usd), 0.0))).one()

    cash_bal = round(float(latest.cash_balance), 2)
    btc_bal = round(float(latest.btc_balance), 6)
    btc_pos_val = round(float(latest.btc_balance) * current_price, 2)
    live_equity = round(cash_bal + btc_pos_val, 2)
    cum_ret = ((live_equity - INITIAL_CAPITAL_USD) / INITIAL_CAPITAL_USD) * 100.0

    baseline_price = get_baseline_btc_price(session) or current_price
    bench_eq = INITIAL_CAPITAL_USD * (current_price / baseline_price) if baseline_price > 0 else INITIAL_CAPITAL_USD

    return {
        "cash_balance": cash_bal, "btc_balance": btc_bal, "current_btc_price": round(current_price, 2),
        "btc_position_value_usd": btc_pos_val, "total_equity_usd": live_equity,
        "cumulative_return_pct": round(cum_ret, 2), "benchmark_btc_equity": round(bench_eq, 2),
        "alpha_usd": round(live_equity - bench_eq, 2),
        "alpha_pct": round(((live_equity - bench_eq) / bench_eq) * 100.0, 2) if bench_eq > 0 else 0.0,
        "realized_pnl_usd": round(float(pnl), 2), "total_fees_usd": round(float(fees), 2),
        "drawdown_pct": round(float(latest.drawdown_pct), 2), "total_trades": trade_count,
    }


def evaluate_spot_hold_until_profit_decision(
    session: Session, current_price: float, predicted_direction: str, proposed_action: str,
    clears_hurdle: bool = True, max_stop_loss_pct: float = 3.5,
) -> Dict[str, Any]:
    latest = get_latest_portfolio_state(session)
    btc_held = float(latest.btc_balance) if latest else 0.0
    norm_dir = str(predicted_direction or "").strip().upper()
    norm_act = str(proposed_action or "HOLD").strip().upper()
    net_exit = round(current_price * (1.0 - SLIPPAGE_RATE) * (1.0 - TAKER_FEE_RATE), 2)

    if btc_held <= 1e-8:
        if norm_act == "BUY" and clears_hurdle:
            return {"final_spot_action": "BUY", "position_state": "ENTERING_LONG_SPOT", "btc_held": 0.0, "avg_entry_with_fees": 0.0, "net_exit_price": net_exit, "estimated_net_pnl_usd": 0.0, "reason": "Spot Entry: Model predicts UP and clears fee hurdle."}
        return {"final_spot_action": "HOLD", "position_state": "FLAT_HOLDING_CASH", "btc_held": 0.0, "avg_entry_with_fees": 0.0, "net_exit_price": net_exit, "estimated_net_pnl_usd": 0.0, "reason": "Spot Flat: Keeping cash safe ($0 fee)."}

    avg_entry = get_average_entry_price(session, target_btc=btc_held, include_fees=True) or current_price
    est_pnl = (net_exit - avg_entry) * btc_held
    ret_pct = ((net_exit - avg_entry) / avg_entry) * 100.0

    if norm_dir == "UP" and norm_act != "SELL":
        return {"final_spot_action": "HOLD", "position_state": "HOLDING_BTC_ON_UP", "btc_held": btc_held, "avg_entry_with_fees": round(avg_entry, 2), "net_exit_price": net_exit, "estimated_net_pnl_usd": round(est_pnl, 2), "reason": "Holding BTC on UP prediction ($0 fee)."}
    if est_pnl > 0.0:
        return {"final_spot_action": "SELL", "position_state": "TAKING_NET_PROFIT_ON_DOWN", "btc_held": btc_held, "avg_entry_with_fees": round(avg_entry, 2), "net_exit_price": net_exit, "estimated_net_pnl_usd": round(est_pnl, 2), "reason": f"Spot Profit-Take SELL: locks in +${est_pnl:,.2f} net profit."}
    if ret_pct <= -abs(max_stop_loss_pct):
        return {"final_spot_action": "SELL", "position_state": "EMERGENCY_STOP_LOSS_EXIT", "btc_held": btc_held, "avg_entry_with_fees": round(avg_entry, 2), "net_exit_price": net_exit, "estimated_net_pnl_usd": round(est_pnl, 2), "reason": f"Emergency Stop-Loss SELL: breached -{abs(max_stop_loss_pct):.2f}% floor."}
    return {"final_spot_action": "HOLD", "position_state": "HOLDING_BTC_UNTIL_NET_PROFIT", "btc_held": btc_held, "avg_entry_with_fees": round(avg_entry, 2), "net_exit_price": net_exit, "estimated_net_pnl_usd": round(est_pnl, 2), "reason": "Spot Hold-Until-Profit: holding BTC to avoid locking in fee loss."}


def evaluate_futures_long_short_plan(
    current_price: float, predicted_direction: str, expected_return_pct: float, clears_hurdle: bool,
    volatility_range_pct: float, risk_approved: bool, conviction_score: int = 5,
    account_equity_usd: float = INITIAL_CAPITAL_USD, leverage: float = 3.0, maintenance_margin_rate: float = 0.004,
) -> Dict[str, Any]:
    if current_price <= 0.0 or not math.isfinite(current_price):
        current_price = 65000.0
    lev = max(1.0, min(float(leverage), 10.0))
    mmr = max(0.001, float(maintenance_margin_rate))
    norm_dir = str(predicted_direction or "").strip().upper()
    expected_range = round(current_price * (max(volatility_range_pct, 0.25) / 100.0), 2)

    if not risk_approved or not clears_hurdle or abs(expected_return_pct) < MIN_HURDLE_PCT or conviction_score < 5:
        return {
            "futures_Position": "FLAT (NO POSITION)", "leverage": f"{lev:.0f}x Isolated", "margin_allocated_usd": 0.0,
            "notional_exposure_usd": 0.0, "entry_price": round(current_price, 2), "liquidation_price": None,
            "stop_loss_price": None, "take_profit_price": None, "expected_1h_range_usd": expected_range,
            "summary": "FLAT (No Futures Trade) — Risk check or move magnitude does not justify leverage.",
        }

    margin_usd = round(account_equity_usd * min(0.20, max(0.05, (conviction_score - 4) * 0.04)), 2)
    notional_usd = round(margin_usd * lev, 2)
    dist = max(0.012, min(0.025, (volatility_range_pct / 100.0) * 1.2))

    if norm_dir == "UP" and expected_return_pct > 0:
        pos_side = "LONG (BUY FUTURES)"
        liq = round(current_price * (1.0 - (1.0 / lev) + mmr), 2)
        sl, tp = round(current_price * (1.0 - dist), 2), round(current_price * (1.0 + dist * 2.0), 2)
    else:
        pos_side = "SHORT (SELL FUTURES)"
        liq = round(current_price * (1.0 + (1.0 / lev) - mmr), 2)
        sl, tp = round(current_price * (1.0 + dist), 2), round(current_price * (1.0 - dist * 2.0), 2)

    return {
        "futures_Position": pos_side, "leverage": f"{lev:.0f}x Isolated", "margin_allocated_usd": margin_usd,
        "notional_exposure_usd": notional_usd, "entry_price": round(current_price, 2), "liquidation_price": liq,
        "stop_loss_price": sl, "take_profit_price": tp, "expected_1h_range_usd": expected_range,
        "summary": f"{pos_side} ({lev:.0f}x Isolated) | Margin: ${margin_usd:,.0f} | Entry: ${current_price:,.2f}",
    }


# ---------------------------------------------------------------------------
# 2. LIVE INGESTION & ROBUST EXCHANGE RETRY
# ---------------------------------------------------------------------------

class MarketFeedExhaustedError(RuntimeError):
    pass


def _normalize_utc_timestamp(ts: Any) -> datetime:
    if ts is None or isinstance(ts, bool) or (isinstance(ts, float) and math.isnan(ts)):
        raise ValueError("Invalid timestamp.")
    if hasattr(ts, "to_pydatetime"):
        if pd.isna(ts):
            raise ValueError("Timestamp cannot be NaT.")
        dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, str):
        raw = ts.strip()
        if not raw:
            raise ValueError("Empty timestamp string.")
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    else:
        raise TypeError(f"Unsupported timestamp type: {type(ts).__name__}")
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _validate_historical_candle_dict(raw_c: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_c, dict) or any(k not in raw_c for k in ("timestamp", "open", "high", "low", "close", "volume")):
        return None
    try:
        c_ts = _normalize_utc_timestamp(raw_c["timestamp"])
        ohlcv = {f: float(raw_c[f]) for f in ("open", "high", "low", "close", "volume") if not isinstance(raw_c[f], bool)}
        if len(ohlcv) < 5 or any(ohlcv[k] <= 0 for k in ("open", "high", "low", "close")) or ohlcv["volume"] < 0:
            return None
        if ohlcv["high"] < max(ohlcv["low"], ohlcv["open"], ohlcv["close"]) or ohlcv["low"] > min(ohlcv["open"], ohlcv["close"]):
            return None
        return {"timestamp": c_ts, **ohlcv}
    except Exception:
        return None


def backfill_missing_candles(session, candle: Dict[str, Any]) -> int:
    # inserts missing confirmed candles preceding the current sync tick
    if not isinstance(candle, dict) or "timestamp" not in candle:
        return 0
    raw_history = candle.get("candles")
    if not raw_history or not isinstance(raw_history, Sequence) or isinstance(raw_history, (str, bytes)):
        return 0
    try:
        latest_ts = _normalize_utc_timestamp(candle["timestamp"])
    except (ValueError, TypeError):
        return 0

    candidates = {v["timestamp"]: v for raw_c in raw_history if (v := _validate_historical_candle_dict(raw_c)) and v["timestamp"] < latest_ts}
    if not candidates:
        return 0

    sorted_candidates = [candidates[t] for t in sorted(candidates.keys())]
    prior = session.execute(select(MarketTick).where(MarketTick.timestamp < latest_ts).order_by(MarketTick.timestamp.desc()).limit(1)).scalars().first()
    if prior is None:
        return 0

    prior_ts = _normalize_utc_timestamp(prior.timestamp)
    gap_hours = (latest_ts - prior_ts).total_seconds() / 3600.0
    if gap_hours > BACKFILL_LARGE_GAP_WARNING_HOURS:
        sync_logger.warning("Detected large candle gap of %.1f hours (%s -> %s).", gap_hours, prior_ts, latest_ts)
    existing_rows = session.execute(select(MarketTick.timestamp).where(MarketTick.timestamp < latest_ts)).scalars().all()
    existing_ts = {_normalize_utc_timestamp(r) for r in existing_rows if r is not None}
    anchor_ts = min(existing_ts | {prior_ts})

    inserted = 0
    for c_data in sorted_candidates:
        c_ts = c_data["timestamp"]
        if c_ts <= anchor_ts or c_ts in existing_ts:
            continue
        tick = MarketTick(
            timestamp=c_ts, open=c_data["open"], high=c_data["high"], low=c_data["low"],
            close=c_data["close"], volume=c_data["volume"], model_version=ACTIVE_MODEL_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )
        try:
            with session.begin_nested():
                session.add(tick)
                session.flush()
                inserted += 1
                existing_ts.add(c_ts)
        except IntegrityError:
            pass
    return inserted


def _parse_trade_count(raw_trades: Any) -> Optional[int]:
    # validates non-negative integer trade counts
    if raw_trades is None:
        return None
    if isinstance(raw_trades, bool):
        raise ValueError("Boolean value not permitted for trade count")
    if isinstance(raw_trades, int):
        if raw_trades < 0:
            raise ValueError(f"Negative trade count: {raw_trades}")
        return raw_trades
    if isinstance(raw_trades, float):
        if not math.isfinite(raw_trades) or raw_trades < 0.0 or not raw_trades.is_integer():
            raise ValueError(f"Invalid float trade count: {raw_trades}")
        return int(raw_trades)
    if isinstance(raw_trades, str):
        s = raw_trades.strip()
        if not s or s.startswith("-"):
            raise ValueError(f"Invalid trade count: {raw_trades!r}")
        try:
            dec = Decimal(s)
            if not dec.is_finite() or dec < 0 or dec != int(dec):
                raise ValueError
            return int(dec)
        except Exception as exc:
            raise ValueError(f"Invalid trade count: {raw_trades!r}") from exc
    raise ValueError(f"Unsupported trade count type: {type(raw_trades).__name__}")


def parse_retry_after(retry_after_header: Optional[str], now_dt_fn: Optional[Callable[[], datetime]] = None) -> Optional[float]:
    # parses HTTP Retry-After header into delay seconds
    if not retry_after_header or not isinstance(retry_after_header, str) or not retry_after_header.strip():
        return None
    val = retry_after_header.strip()
    try:
        sec = float(val)
        return sec if (math.isfinite(sec) and sec >= 0.0) else None
    except ValueError:
        pass
    try:
        p_dt = parsedate_to_datetime(val)
        if p_dt is not None:
            now_dt = now_dt_fn() if now_dt_fn else datetime.now(timezone.utc)
            delta = ((p_dt.replace(tzinfo=timezone.utc) if p_dt.tzinfo is None else p_dt) - now_dt).total_seconds()
            return max(0.0, delta) if math.isfinite(delta) else None
    except Exception:
        pass
    return None


_parse_retry_after_header = parse_retry_after


def _validate_candle_payload(payload: Any, min_row_len: int = BINANCE_MIN_ROW_LEN) -> bool:
    # validates structure and numeric bounds of raw exchange klines
    if not isinstance(payload, list) or len(payload) == 0:
        return False
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) < min_row_len:
            return False
        for idx in range(1, 6):
            try:
                val = float(row[idx])
                if not math.isfinite(val) or isinstance(row[idx], bool):
                    return False
            except Exception:
                return False
    return True


def _fetch_feed_with_retry(
    url: str, headers: Optional[Dict[str, str]] = None, feed_name: str = "Binance Vision",
    deadline_ts: Optional[float] = None, max_attempts: int = EXCHANGE_RETRY_MAX_ATTEMPTS,
    connect_timeout: float = EXCHANGE_CONNECT_TIMEOUT, read_timeout: float = EXCHANGE_READ_TIMEOUT,
    base_backoff: float = EXCHANGE_RETRY_BASE_BACKOFF, backoff_factor: float = EXCHANGE_RETRY_BACKOFF_FACTOR,
    max_backoff: float = EXCHANGE_RETRY_MAX_BACKOFF, max_retry_after: float = EXCHANGE_MAX_RETRY_AFTER_CAP,
    total_deadline: float = EXCHANGE_RETRY_TOTAL_DEADLINE, min_row_len: Optional[int] = None,
    sleep_fn: Callable[[float], None] = time.sleep, jitter_fn: Callable[[float, float], float] = random.uniform,
    clock_fn: Callable[[], float] = time.perf_counter, now_dt_fn: Optional[Callable[[], datetime]] = None,
) -> Optional[List[Any]]:
    # resilient HTTP request execution with exponential backoff and rate limit handling
    if deadline_ts is None:
        deadline_ts = clock_fn() + total_deadline
    req_row_len = min_row_len or (BINANCE_MIN_ROW_LEN if "binance" in feed_name.lower() else COINBASE_MIN_ROW_LEN)

    def _sleep_backoff(attempt: int) -> bool:
        backoff = min(max_backoff, base_backoff * (backoff_factor ** (attempt - 1)))
        raw_j = jitter_fn(0.1, 0.5)
        delay = backoff + (raw_j if (math.isfinite(raw_j) and raw_j >= 0.0) else 0.0)
        if (clock_fn() + delay) > deadline_ts:
            return False
        sleep_fn(delay)
        return True

    for attempt in range(1, max_attempts + 1):
        rem = deadline_ts - clock_fn()
        if rem < 0.5:
            return None
        try:
            resp = requests.get(url, headers=headers, timeout=(min(connect_timeout, rem), min(read_timeout, rem)))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
            if attempt >= max_attempts or not _sleep_backoff(attempt):
                return None
            continue
        except Exception:
            return None

        status = resp.status_code
        if not isinstance(status, int) or clock_fn() > deadline_ts or (400 <= status < 500 and status not in (429, 418)):
            return None
        if status in (429, 418):
            r_str = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
            r_val = parse_retry_after(r_str, now_dt_fn=now_dt_fn)
            if r_val is None or r_val > max_retry_after or attempt >= max_attempts:
                return None
            delay = min(max_retry_after, r_val + max(0.0, jitter_fn(0.1, 0.5)))
            if (clock_fn() + delay) > deadline_ts:
                return None
            sleep_fn(delay)
            continue
        if status in (500, 502, 503, 504):
            if attempt >= max_attempts or not _sleep_backoff(attempt):
                return None
            continue
        if status == 200:
            try:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0 and len(data[0]) >= req_row_len:
                    return data
            except Exception:
                pass
            return None
    return None


def fetch_latest_confirmed_candle(
    symbol: str = "BTCUSDT", sleep_fn: Optional[Callable[[float], None]] = None,
    jitter_fn: Optional[Callable[[float, float], float]] = None, clock_fn: Optional[Callable[[], float]] = None,
    now_dt_fn: Optional[Callable[[], datetime]] = None, total_deadline: Optional[float] = None,
) -> Dict[str, Any]:
    # fetches confirmed 1-hour candle from Binance Vision with Coinbase fallback
    _sleep_fn, _jitter_fn = sleep_fn or time.sleep, jitter_fn or random.uniform
    _clock_fn, _budget = clock_fn or time.perf_counter, total_deadline or EXCHANGE_RETRY_TOTAL_DEADLINE

    b_url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit=102"
    binance_klines = _fetch_feed_with_retry(
        url=b_url, feed_name="Binance Vision", min_row_len=BINANCE_MIN_ROW_LEN,
        sleep_fn=_sleep_fn, jitter_fn=_jitter_fn, clock_fn=_clock_fn, now_dt_fn=now_dt_fn, total_deadline=_budget,
    )
    if binance_klines and _validate_candle_payload(binance_klines, min_row_len=BINANCE_MIN_ROW_LEN):
        now_ms = (now_dt_fn().timestamp() if now_dt_fn else datetime.now(timezone.utc).timestamp()) * 1000.0
        confirmed = [k for k in binance_klines if float(k[6]) < now_ms]
        if len(confirmed) >= 35:
            selected = confirmed[-100:]
            c = selected[-1]
            records = [
                {
                    "timestamp": pd.to_datetime(float(k[0]), unit="ms", utc=True),
                    "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]), "number_of_trades": _parse_trade_count(k[8]),
                    "taker_buy_ratio": round(float(k[9]) / float(k[5]), 4) if float(k[5]) > 0 else 0.5,
                }
                for k in selected
            ]
            return {
                "timestamp": datetime.fromtimestamp(float(c[0]) / 1000.0, tz=timezone.utc),
                "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]),
                "volume": float(c[5]), "taker_buy_ratio": records[-1]["taker_buy_ratio"],
                "number_of_trades": _parse_trade_count(c[8]), "candles": records,
            }

    c_url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
    cb_candles = _fetch_feed_with_retry(
        url=c_url, headers={"User-Agent": "Mudra-Sync/1.0"}, feed_name="Coinbase Institutional",
        min_row_len=COINBASE_MIN_ROW_LEN, sleep_fn=_sleep_fn, jitter_fn=_jitter_fn,
        clock_fn=_clock_fn, now_dt_fn=now_dt_fn, total_deadline=_budget,
    )
    if cb_candles and _validate_candle_payload(cb_candles, min_row_len=COINBASE_MIN_ROW_LEN):
        now_sec = now_dt_fn().timestamp() if now_dt_fn else datetime.now(timezone.utc).timestamp()
        confirmed = [k for k in cb_candles if (float(k[0]) + 3600.0) <= now_sec]
        if len(confirmed) >= 35:
            selected = list(reversed(confirmed[:100]))
            c = selected[-1]
            records = [
                {
                    "timestamp": pd.to_datetime(float(k[0]), unit="s", utc=True),
                    "low": float(k[1]), "high": float(k[2]), "open": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]), "taker_buy_ratio": 0.50, "number_of_trades": None,
                }
                for k in selected
            ]
            return {
                "timestamp": datetime.fromtimestamp(float(c[0]), tz=timezone.utc),
                "open": float(c[3]), "high": float(c[2]), "low": float(c[1]), "close": float(c[4]),
                "volume": float(c[5]), "taker_buy_ratio": 0.50, "number_of_trades": None, "candles": records,
            }

    raise MarketFeedExhaustedError("Failed to fetch confirmed 1-hour candle from all market feeds.")


def is_paper_trade_executed_for_tick(session: Session, tick_id: int) -> bool:
    # prevents duplicate paper trading execution on the same market tick
    if has_tick_id_column(session) and session.execute(select(PaperPortfolio.id).where(PaperPortfolio.tick_id == tick_id)).scalar_one_or_none():
        return True
    audit = session.execute(select(CouncilAudit).where(CouncilAudit.tick_id == tick_id).order_by(CouncilAudit.id.asc())).scalars().first()
    return audit is not None and (not audit.execution_notes or "[FORCED_REANALYSIS]" not in audit.execution_notes)


def sync_hourly_quant_cycle(force: bool = False, candle: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    # executes end-to-end quantitative cycle: candle ingestion, ML inference, multi-agent council, and ledger
    db_ok, ver, _ = test_connection()
    if not db_ok:
        return False, f"Database unreachable: {ver}", {}

    if candle is None:
        try:
            candle = fetch_latest_confirmed_candle()
        except MarketFeedExhaustedError as exc:
            return False, f"Market feed error: {exc}", {}

    with get_db() as session:
        latest_ts = _normalize_utc_timestamp(candle["timestamp"])
        backfill_missing_candles(session, candle)

        existing = session.execute(
            select(MarketTick).where(MarketTick.timestamp == latest_ts, MarketTick.model_version == ACTIVE_MODEL_VERSION).order_by(MarketTick.id.desc())
        ).scalars().first()

        if existing and not force:
            reconcile_pending_ticks(session, existing)
            return False, f"Candle for {latest_ts.strftime('%H:00 UTC')} already recorded in Neon.", {"tick_id": existing.id}

        is_existing = existing is not None
        already_executed = is_paper_trade_executed_for_tick(session, existing.id) if is_existing else False

        signals_output = calculate_technical_signals.invoke({
            "close": candle["close"], "open_price": candle["open"], "high": candle["high"], "low": candle["low"],
            "volume": candle["volume"], "taker_buy_ratio": candle.get("taker_buy_ratio"), "timestamp": latest_ts, "candles": candle.get("candles"),
        })
        ml_output = get_model_return_forecast.invoke({"features": signals_output})
        pred_return = float(ml_output.get("expected_return_pct", 0.0))
        pred_price = float(ml_output.get("predicted_target_price", candle["close"]))
        confidence = float(ml_output.get("confidence", 0.50))
        clears_hurdle = ml_output.get("clears_hurdle", False)
        volatility = float(signals_output.get("volatility_range_pct", 0.02))
        regime = "HIGH_VOL_CHOP" if volatility > 3.8 else ("BULL_MOMENTUM" if (clears_hurdle and pred_return > 0) else ("BEAR_DISTRIBUTION" if (clears_hurdle and pred_return < 0) else "SIDEWAYS_RANGE"))

        safe_features = json.loads(json.dumps(signals_output, default=str)) if signals_output else None

        if is_existing:
            current_tick = existing
            if force:
                current_tick.market_regime = regime
                current_tick.predicted_return_1h = pred_return
                current_tick.predicted_price_1h = pred_price
                current_tick.model_confidence = confidence
                current_tick.features = safe_features
                session.flush()
        else:
            current_tick = MarketTick(
                timestamp=latest_ts, open=candle["open"], high=candle["high"], low=candle["low"],
                close=candle["close"], volume=candle["volume"], model_version=ACTIVE_MODEL_VERSION,
                feature_schema_version=FEATURE_SCHEMA_VERSION, market_regime=regime,
                predicted_return_1h=pred_return, predicted_price_1h=pred_price,
                model_confidence=confidence, features=safe_features,
            )
            session.add(current_tick)
            session.flush()

        # evaluate and record shadow predictions for registered candidate models
        try:
            from src.model_loader import get_registered_models
            all_registered = get_registered_models()
            for cand_version, cand_spec in all_registered.items():
                if cand_version == ACTIVE_MODEL_VERSION:
                    continue
                c_feats = cand_spec.get("features", [])
                if c_feats and all(k in signals_output for k in c_feats):
                    c_input_df = pd.DataFrame([{k: signals_output[k] for k in c_feats}])
                    c_clf = cand_spec["classifier"]
                    c_reg = cand_spec["regressor"]
                    c_prob_up = float(c_clf.predict_proba(c_input_df)[0, 1]) if hasattr(c_clf, "predict_proba") else 0.5
                    c_ret = float(c_reg.predict(c_input_df)[0])
                    close_f = float(candle["close"])
                    c_price = float(close_f * (1.0 + c_ret))
                    c_ret_pct = float(((c_price - close_f) / close_f) * 100.0)
                    c_dir = "UP" if c_ret_pct >= 0.0 else "DOWN"

                    cand_pred = session.execute(
                        select(ModelPrediction).where(
                            ModelPrediction.tick_id == current_tick.id,
                            ModelPrediction.model_version == cand_version,
                        )
                    ).scalars().first()

                    if not cand_pred:
                        cand_pred = ModelPrediction(
                            tick_id=current_tick.id,
                            model_version=cand_version,
                            run_mode="SHADOW",
                            is_canonical=True,
                            is_primary=False,
                            feature_schema_version=cand_spec.get("manifest_data", {}).get("feature_schema_version", "v2_36stationary"),
                            predicted_return_pct=c_ret_pct,
                            predicted_price_1h=c_price,
                            direction_predicted=c_dir,
                            direction_prob_up=c_prob_up,
                            clears_hurdle=bool(abs(c_ret_pct) >= MIN_HURDLE_PCT),
                        )
                        session.add(cand_pred)
                        session.flush()
        except Exception as cand_exc:
            sync_logger.warning("Failed candidate model prediction: %s", type(cand_exc).__name__)

        reconcile_pending_ticks(session, current_tick)

        council_state = run_trading_graph(
            symbol="BTCUSDT", current_price=candle["close"], market_data=candle,
            engineered_features=signals_output, ml_forecast=ml_output, timestamp=latest_ts.isoformat(),
        )
        decision = council_state.get("final_decision", "HOLD")
        verdict = council_state.get("risk_verdict", "VETOED")
        kelly = float(council_state.get("kelly_fraction", 0.0))

        audit = CouncilAudit(
            tick_id=current_tick.id,
            deliberation_latency_ms=council_state.get("latency_ms", 120.0),
            strategist_action=council_state.get("strategist_action", decision),
            strategist_conviction=council_state.get("strategist_conviction", 5),
            rag_context_summary=council_state.get("rag_context", "Normal quant market regime"),
            risk_verdict=verdict,
            veto_reasons=council_state.get("veto_reasons", []),
            kelly_fraction=kelly,
            final_decision=decision,
            execution_notes=f"Council deliberated {decision} ({verdict}) at BTC ${candle['close']:,.2f}." + (" [FORCED_REANALYSIS]" if (is_existing and force) else ""),
        )
        session.add(audit)
        session.flush()

        if not (is_existing and already_executed):
            effective_action = decision
            order_notes = f"Executed {decision} ({verdict}) at BTC ${candle['close']:,.2f}."
            if str(decision).upper() == "SELL":
                try:
                    spot_guard = evaluate_spot_hold_until_profit_decision(
                        session=session, current_price=float(candle["close"]),
                        predicted_direction=str(ml_output.get("direction", "DOWN")),
                        proposed_action=decision, clears_hurdle=bool(ml_output.get("clears_hurdle", False)),
                    )
                    effective_action = spot_guard.get("final_spot_action", decision)
                    order_notes = spot_guard.get("reason", order_notes)
                except Exception:
                    pass
            execute_paper_order(
                session=session, action=effective_action, current_price=candle["close"],
                kelly_fraction=kelly, notes=order_notes, tick_id=current_tick.id,
            )

        summary = get_portfolio_summary(session, candle["close"])
        if is_existing and already_executed:
            return True, f"Updated forecast for existing tick #{current_tick.id} at ${candle['close']:,.2f} ({decision} | Trade deduplicated | No fees)", summary
        return True, f"Recorded tick #{current_tick.id} at ${candle['close']:,.2f} ({decision} | Kelly: {kelly*100:.1f}%)", summary


# ---------------------------------------------------------------------------
# 4. MULTI-MODEL CONSENSUS & PRICE RANGE ESTIMATION
# ---------------------------------------------------------------------------

def estimate_next_hour_price_range_usd(live_df: pd.DataFrame, current_price: float) -> Dict[str, float]:
    if live_df is None or live_df.empty or current_price <= 0:
        return {"predicted_range_usd": 0.0, "predicted_range_pct": 0.0, "predicted_high_1h": current_price, "predicted_low_1h": current_price}

    highs = pd.to_numeric(live_df["high"], errors="coerce")
    lows = pd.to_numeric(live_df["low"], errors="coerce")
    closes = pd.to_numeric(live_df["close"], errors="coerce").replace(0, np.nan)
    range_pct_series = ((highs - lows) / closes).dropna()

    if range_pct_series.empty:
        return {"predicted_range_usd": 0.0, "predicted_range_pct": 0.0, "predicted_high_1h": current_price, "predicted_low_1h": current_price}

    last_1h = float(range_pct_series.iloc[-1])
    ema_6h = float(range_pct_series.tail(6).ewm(span=6, adjust=False).mean().iloc[-1])
    mean_24h = float(range_pct_series.tail(24).mean())
    pred_range_rate = (0.45 * last_1h) + (0.35 * ema_6h) + (0.20 * mean_24h)
    pred_range_usd = float(current_price * pred_range_rate)
    half_range = pred_range_usd / 2.0

    return {
        "predicted_range_usd": round(pred_range_usd, 2), "predicted_range_pct": round(pred_range_rate * 100.0, 4),
        "predicted_high_1h": round(current_price + half_range, 2), "predicted_low_1h": round(max(1.0, current_price - half_range), 2),
    }


def evaluate_multi_angle_consensus(
    current_price: float, primary_direction: str, primary_prob_up: float, primary_pred_dollar_move: float,
    predicted_range_usd: float, shadow_direction: Optional[str] = None, shadow_prob_up: Optional[float] = None,
    model_votes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    hurdle_dollar_min = current_price * (MIN_HURDLE_PCT / 100.0)

    if model_votes and len(model_votes) > 0:
        total_models = len(model_votes)
        votes_up = sum(1 for m in model_votes if str(m.get("direction")).upper() == "UP")
        avg_prob_up = float(np.mean([float(m.get("prob_up", 50.0)) for m in model_votes]))
    else:
        votes_up = 1 if str(primary_direction).upper() == "UP" else 0
        total_models = 1
        if shadow_direction is not None and shadow_prob_up is not None:
            total_models = 2
            if str(shadow_direction).upper() == "UP":
                votes_up += 1
            avg_prob_up = (float(primary_prob_up) + float(shadow_prob_up)) / 2.0
        else:
            avg_prob_up = float(primary_prob_up)

    votes_down = total_models - votes_up
    if votes_up > votes_down:
        consensus_dir, agreement_text = "UP", f"{votes_up}/{total_models} Models Vote UP"
    elif votes_down > votes_up:
        consensus_dir, agreement_text = "DOWN", f"{votes_down}/{total_models} Models Vote DOWN"
    else:
        consensus_dir, agreement_text = "SPLIT", f"{votes_up}/{total_models} Models Split (No Consensus -> Hold)"

    return {
        "consensus_direction": consensus_dir, "agreement_badge": agreement_text, "total_models_voting": total_models,
        "avg_prob_up_pct": round(avg_prob_up, 2), "predicted_range_usd": round(predicted_range_usd, 2),
        "predicted_dollar_move": round(primary_pred_dollar_move, 2), "hurdle_dollar_min": round(hurdle_dollar_min, 2),
        "range_clears_fees": bool(predicted_range_usd >= (hurdle_dollar_min * 1.5)),
        "move_clears_fees": bool(abs(primary_pred_dollar_move) >= hurdle_dollar_min),
        "plain_summary": f"Direction Vote: {agreement_text} | Predicted 1h Move: ${primary_pred_dollar_move:+,.2f} (Fee Hurdle: ${hurdle_dollar_min:,.0f}) | Predicted 1h High-Low Range: ${predicted_range_usd:,.2f}",
    }


# ---------------------------------------------------------------------------
# 5. DELAYED T+1h GROUND-TRUTH RECONCILIATION
# ---------------------------------------------------------------------------

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

    for prior in pending_ticks:
        expected_ts = prior.timestamp + timedelta(hours=1)
        outcome = current_tick if current_tick.timestamp == expected_ts else session.execute(select(MarketTick).where(MarketTick.timestamp == expected_ts)).scalars().first()
        if not outcome or outcome.close <= 0:
            continue

        act_p, pred_p, base_p = max(1.0, float(outcome.close)), float(prior.predicted_price_1h), max(1.0, float(prior.close))
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
            timestamp=datetime.now(timezone.utc), tick_id=prior.id, predicted_price=pred_p,
            actual_price_at_t_plus_1=act_p, predicted_return_pct=pred_ret, actual_return_pct=act_ret,
            residual_error=act_ret - pred_ret, absolute_error_usd=abs_err, percentage_error=pct_err,
            direction_predicted=dir_pred, direction_realized=dir_real, direction_correct=(dir_pred == dir_real),
            is_black_swan=is_swan, error_attribution_tag=tag,
        )
        session.add(rec)
        reconciled_records.append(rec)

        # indexes reconciled prediction outcome into ChromaDB episodic error memory
        if not os.getenv("PYTEST_CURRENT_TEST"):
            try:
                from src.agents import index_reconciled_outcome_memory
                story = build_plain_english_reconciliation_story(
                    pred_tick=prior, outcome_tick=outcome, pred_price=pred_p, actual_price=act_p,
                    pred_ret_pct=pred_ret, actual_ret_pct=act_ret, dir_pred=dir_pred, dir_real=dir_real,
                    dir_correct=(dir_pred == dir_real), abs_err_usd=abs_err, tag=tag,
                )
                index_reconciled_outcome_memory(
                    tick_id=prior.id,
                    timestamp_iso=outcome.timestamp.isoformat() if hasattr(outcome.timestamp, "isoformat") else str(outcome.timestamp),
                    market_regime=str(prior.market_regime or "STANDARD_SPOT"),
                    direction_predicted=dir_pred,
                    direction_realized=dir_real,
                    direction_correct=(dir_pred == dir_real),
                    predicted_dollar_move=pred_p - base_p,
                    actual_dollar_move=act_p - base_p,
                    absolute_error_usd=abs_err,
                    error_attribution_tag=tag,
                    plain_english_notes=story.get("hypothesis_notes", f"Error: ${abs_err:,.2f}"),
                )
            except Exception:
                pass

        try:
            with session.begin_nested():
                for u in session.execute(select(ModelPrediction).where(ModelPrediction.tick_id == prior.id, ModelPrediction.outcome_tick_id.is_(None))).scalars().all():
                    u_pred_price = float(u.predicted_price_1h)
                    u_pred_ret = float(u.predicted_return_pct)
                    u_abs_err = abs(act_p - u_pred_price)
                    u_pct_err = (u_abs_err / act_p) * 100.0
                    u_residual = act_ret - u_pred_ret
                    u_dir_real = "UP" if act_ret >= 0.0 else "DOWN"
                    u_dir_correct = (u.direction_predicted == u_dir_real)
                    u_is_swan = bool(abs(act_ret) >= 3.8 or abs(u_residual) >= 3.8)

                    if u_is_swan:
                        u_tag = "BLACK_SWAN_OUTLIER"
                    elif u_dir_correct:
                        if u_pct_err <= 0.35:
                            u_tag = "ACCURATE"
                        elif abs(act_ret) > abs(u_pred_ret):
                            u_tag = "MOMENTUM_UNDERESTIMATED"
                        else:
                            u_tag = "MOMENTUM_OVERESTIMATED"
                    else:
                        if u_pct_err <= 0.20 or u_abs_err < 150.0:
                            u_tag = "SIDEWAYS_DRIFT"
                        else:
                            u_tag = "DIRECTION_MISALIGNED"

                    u.outcome_tick_id = outcome.id
                    u.evaluation_policy_version = "binary_exact_t1_v2"
                    u.actual_price_at_t_plus_1 = act_p
                    u.actual_return_pct = act_ret
                    u.direction_realized = u_dir_real
                    u.direction_correct = u_dir_correct
                    u.residual_error = u_residual
                    u.absolute_error_usd = u_abs_err
                    u.percentage_error = u_pct_err
                    u.is_black_swan = u_is_swan
                    u.error_attribution_tag = u_tag
                    u.reconciled_at = datetime.now(timezone.utc)
                session.flush()
        except Exception as exc:
            logger.warning("Could not update model_predictions for tick %s: %s", prior.id, exc)

        audit = session.execute(select(CouncilAudit).where(CouncilAudit.tick_id == prior.id).order_by(CouncilAudit.id.desc())).scalars().first()
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
    return {"headline": headline[:500], "hypothesis_notes": f"Pred: ${pred_price:,.2f} ({dir_pred}) | Actual: ${actual_price:,.2f} ({dir_real}) | Err: ${abs_err_usd:,.2f} | Diagnosis: {why}", "suspected_cause_category": tag}


def get_reconciliation_summary(session: Session, limit: int = 168, model_version: Optional[str] = None) -> Dict[str, Any]:
    records, total_lifetime = [], 0

    if model_version and model_version != "All Models":
        try:
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
