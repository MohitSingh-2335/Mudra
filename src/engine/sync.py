import json
import logging
import math
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
from sqlalchemy import and_, desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ACTIVE_MODEL_VERSION, EXCHANGE_CONNECT_TIMEOUT, EXCHANGE_MAX_RETRY_AFTER_CAP,
    EXCHANGE_READ_TIMEOUT, EXCHANGE_RETRY_BACKOFF_FACTOR, EXCHANGE_RETRY_BASE_BACKOFF,
    EXCHANGE_RETRY_MAX_ATTEMPTS, EXCHANGE_RETRY_MAX_BACKOFF, EXCHANGE_RETRY_TOTAL_DEADLINE,
    FEATURE_SCHEMA_VERSION, MIN_HURDLE_PCT,
)
from src.agents.council import calculate_technical_signals, get_model_return_forecast, run_trading_graph
from src.database.connection import get_db, has_tick_id_column, test_connection
from src.database.models import CouncilAudit, MarketTick, PaperPortfolio, Reconciliation
from src.engine.paper_portfolio import execute_paper_order, get_portfolio_summary

logger = logging.getLogger("mudra.sync")
BINANCE_MIN_ROW_LEN: int = 10
COINBASE_MIN_ROW_LEN: int = 6
BACKFILL_LARGE_GAP_WARNING_HOURS: float = 24.0


class MarketFeedExhaustedError(RuntimeError):
    pass


def _normalize_utc_timestamp(ts: Any) -> datetime:
    # normalizes timestamp input to UTC datetime
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
    # validates OHLCV values and boundary invariants
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
        logger.warning("[BACKFILL] Detected large candle gap of %.1f hours between %s and %s.", gap_hours, prior_ts.isoformat(), latest_ts.isoformat())
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
            existing_ts.add(c_ts)
            inserted += 1
        except IntegrityError:
            pass
    return inserted


def is_paper_trade_executed_for_tick(session, tick_id: int) -> bool:
    # prevents duplicate paper trading execution on the same market tick
    if has_tick_id_column(session) and session.execute(select(PaperPortfolio.id).where(PaperPortfolio.tick_id == tick_id)).scalar_one_or_none():
        return True
    audit = session.execute(select(CouncilAudit).where(CouncilAudit.tick_id == tick_id).order_by(CouncilAudit.id.asc())).scalars().first()
    return audit is not None and (not audit.execution_notes or "[FORCED_REANALYSIS]" not in audit.execution_notes)


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


def _validate_candle_payload(payload: Any, min_row_len: int = BINANCE_MIN_ROW_LEN, feed_name: str = "Market Feed") -> bool:
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
            except (TypeError, ValueError):
                return False
        if min_row_len >= BINANCE_MIN_ROW_LEN:
            try:
                _parse_trade_count(row[8])
            except ValueError:
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
                payload = resp.json()
            except Exception:
                if attempt >= max_attempts or not _sleep_backoff(attempt):
                    return None
                continue
            if clock_fn() > deadline_ts or not _validate_candle_payload(payload, min_row_len=req_row_len, feed_name=feed_name):
                return None
            return payload
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
    binance_klines = _fetch_feed_with_retry(url=b_url, feed_name="Binance Vision", min_row_len=BINANCE_MIN_ROW_LEN, sleep_fn=_sleep_fn, jitter_fn=_jitter_fn, clock_fn=_clock_fn, now_dt_fn=now_dt_fn, total_deadline=_budget)
    if binance_klines and _validate_candle_payload(binance_klines, min_row_len=BINANCE_MIN_ROW_LEN):
        now_ms = (now_dt_fn().timestamp() if now_dt_fn else datetime.now(timezone.utc).timestamp()) * 1000.0
        confirmed = [k for k in binance_klines if float(k[6]) < now_ms]
        if len(confirmed) >= 35:
            selected = confirmed[-100:]
            c = selected[-1]
            records = [{"timestamp": pd.to_datetime(float(k[0]), unit="ms", utc=True), "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]), "number_of_trades": _parse_trade_count(k[8]), "taker_buy_ratio": round(float(k[9]) / float(k[5]), 4) if float(k[5]) > 0 else 0.5} for k in selected]
            return {
                "timestamp": datetime.fromtimestamp(float(c[0]) / 1000.0, tz=timezone.utc), "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]), "volume": float(c[5]), "taker_buy_ratio": records[-1]["taker_buy_ratio"], "number_of_trades": _parse_trade_count(c[8]), "candles": records,
            }

    c_url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
    cb_candles = _fetch_feed_with_retry(url=c_url, headers={"User-Agent": "Mudra-Sync/1.0"}, feed_name="Coinbase Institutional", min_row_len=COINBASE_MIN_ROW_LEN, sleep_fn=_sleep_fn, jitter_fn=_jitter_fn, clock_fn=_clock_fn, now_dt_fn=now_dt_fn, total_deadline=_budget)
    if cb_candles and _validate_candle_payload(cb_candles, min_row_len=COINBASE_MIN_ROW_LEN):
        now_sec = now_dt_fn().timestamp() if now_dt_fn else datetime.now(timezone.utc).timestamp()
        confirmed = [k for k in cb_candles if (float(k[0]) + 3600.0) <= now_sec]
        if len(confirmed) >= 35:
            selected = list(reversed(confirmed[:100]))
            c = selected[-1]
            records = [{"timestamp": pd.to_datetime(float(k[0]), unit="s", utc=True), "low": float(k[1]), "high": float(k[2]), "open": float(c[3]), "close": float(k[4]), "volume": float(k[5]), "taker_buy_ratio": 0.50, "number_of_trades": None} for k in selected]
            return {
                "timestamp": datetime.fromtimestamp(float(c[0]), tz=timezone.utc), "open": float(c[3]), "high": float(c[2]), "low": float(c[1]), "close": float(c[4]), "volume": float(c[5]), "taker_buy_ratio": 0.50, "number_of_trades": None, "candles": records,
            }

    raise MarketFeedExhaustedError("Failed to fetch confirmed 1-hour candle from all market feeds.")


def sync_hourly_quant_cycle(force: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
    # executes end-to-end quantitative cycle: candle ingestion, ML inference, multi-agent council, and ledger
    db_ok, ver, _ = test_connection()
    if not db_ok:
        return False, f"Database unreachable: {ver}", {}

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

        reconcile_pending_ticks(session, current_tick)

        council_state = run_trading_graph(symbol="BTCUSDT", current_price=candle["close"], market_data=candle, engineered_features=signals_output, ml_forecast=ml_output, timestamp=latest_ts.isoformat())
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
                    from src.engine.paper_portfolio import evaluate_spot_hold_until_profit_decision
                    spot_guard = evaluate_spot_hold_until_profit_decision(session=session, current_price=float(candle["close"]), predicted_direction=str(ml_output.get("direction", "DOWN")), proposed_action=decision, clears_hurdle=bool(ml_output.get("clears_hurdle", False)))
                    effective_action = spot_guard.get("final_spot_action", decision)
                    order_notes = spot_guard.get("reason", order_notes)
                except Exception:
                    pass
            execute_paper_order(session=session, action=effective_action, current_price=candle["close"], kelly_fraction=kelly, notes=order_notes, tick_id=current_tick.id)

        summary = get_portfolio_summary(session, candle["close"])
        if is_existing and already_executed:
            return True, f"Updated forecast for existing tick #{current_tick.id} at ${candle['close']:,.2f} ({decision} | Trade deduplicated | No fees)", summary
        return True, f"Recorded tick #{current_tick.id} at ${candle['close']:,.2f} ({decision} | Kelly: {kelly*100:.1f}%)", summary


# predicts next 1-hour price range using volatility clustering across 1h, 6h, and 24h
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


# evaluates consensus agreement across multiple model votes and compares against fee hurdle
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

        try:
            from src.database.models import ModelPrediction
            for u in session.execute(select(ModelPrediction).where(ModelPrediction.tick_id == prior.id, ModelPrediction.outcome_tick_id.is_(None))).scalars().all():
                u.outcome_tick_id, u.evaluation_policy_version = outcome.id, "binary_exact_t1_v2"
                u.actual_price_at_t_plus_1, u.actual_return_pct = act_p, act_ret
                u.direction_realized, u.direction_correct = dir_real, (u.direction_predicted == dir_real)
                u.residual_error, u.absolute_error_usd = act_ret - float(u.predicted_return_pct), abs_err
                u.percentage_error, u.is_black_swan = pct_err, is_swan
                u.error_attribution_tag, u.reconciled_at = tag, datetime.now(timezone.utc)
        except Exception:
            pass

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
    return {"headline": headline[:500], "hypothesis_notes": f"Pred: ${pred_price:,.2f} ({dir_pred}) | Actual: ${actual_price:,.2f} ({dir_real}) | Err: ${abs_err_usd:,.2f} | Diagnosis: {why}", "suspected_cause_category": tag}


# Summary KPI calculations across reconciled predictions
def get_reconciliation_summary(session: Session, limit: int = 168, model_version: Optional[str] = None) -> Dict[str, Any]:
    records, total_lifetime = [], 0

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
