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
import random
import math
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Dict, Any, Tuple, Optional, Callable, List, Sequence
import requests
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.exc import IntegrityError

from config import (
    ACTIVE_MODEL_VERSION,
    FEATURE_SCHEMA_VERSION,
    EXCHANGE_RETRY_MAX_ATTEMPTS,
    EXCHANGE_CONNECT_TIMEOUT,
    EXCHANGE_READ_TIMEOUT,
    EXCHANGE_RETRY_BASE_BACKOFF,
    EXCHANGE_RETRY_BACKOFF_FACTOR,
    EXCHANGE_RETRY_MAX_BACKOFF,
    EXCHANGE_MAX_RETRY_AFTER_CAP,
    EXCHANGE_RETRY_TOTAL_DEADLINE,
)
from src.database.connection import get_db, test_connection, has_tick_id_column
from src.database.models import MarketTick, ApiTelemetry, CouncilAudit, PaperPortfolio
from src.engine.reconciliation import reconcile_pending_ticks
from src.engine.paper_portfolio import execute_paper_order, get_portfolio_summary
from src.agents.trading_graph import run_trading_graph
from src.agents.tools import calculate_technical_signals, get_model_return_forecast

logger = logging.getLogger("mudra.sync")

BINANCE_MIN_ROW_LEN: int = 10
COINBASE_MIN_ROW_LEN: int = 6
BACKFILL_LARGE_GAP_WARNING_HOURS: float = 24.0


class MarketFeedExhaustedError(RuntimeError):
    """Raised when all configured market feeds fail to provide a valid confirmed candle."""


__all__ = [
    "MarketFeedExhaustedError",
    "sync_hourly_quant_cycle",
    "fetch_latest_confirmed_candle",
    "backfill_missing_candles",
    "_normalize_utc_timestamp",
    "is_paper_trade_executed_for_tick",
    "parse_retry_after",
    "_parse_trade_count",
    "_validate_candle_payload",
    "_fetch_feed_with_retry",
]


def _normalize_utc_timestamp(ts: Any) -> datetime:
    """
    Normalizes a datetime, pandas Timestamp, or ISO-8601 timestamp string
    to a standard Python timezone-aware UTC datetime object.
    """
    if ts is None or isinstance(ts, bool) or (isinstance(ts, float) and math.isnan(ts)):
        raise ValueError("Timestamp cannot be None, boolean, or NaN.")
    if hasattr(ts, "to_pydatetime"):
        if pd.isna(ts):
            raise ValueError("Timestamp cannot be NaT.")
        dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, str):
        raw = ts.strip()
        if not raw:
            raise ValueError("Timestamp string cannot be empty.")
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    else:
        raise TypeError(f"Unsupported timestamp type: {type(ts).__name__}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _validate_historical_candle_dict(raw_c: Any) -> Optional[Dict[str, Any]]:
    """
    Validates a single historical candle dictionary from candle['candles'].
    Returns a normalized dictionary with a UTC-aware Python datetime 'timestamp'
    and finite float OHLCV values, or None if malformed.
    """
    if not isinstance(raw_c, dict):
        return None
    required_keys = ("timestamp", "open", "high", "low", "close", "volume")
    if any(k not in raw_c for k in required_keys):
        return None
    try:
        c_ts = _normalize_utc_timestamp(raw_c["timestamp"])
    except (ValueError, TypeError):
        return None

    ohlcv: Dict[str, float] = {}
    for field in ("open", "high", "low", "close", "volume"):
        val = raw_c[field]
        if isinstance(val, bool) or val is None:
            return None
        try:
            f_val = float(val)
        except (ValueError, TypeError):
            return None
        if not math.isfinite(f_val):
            return None
        ohlcv[field] = f_val

    if (
        ohlcv["open"] <= 0.0
        or ohlcv["high"] <= 0.0
        or ohlcv["low"] <= 0.0
        or ohlcv["close"] <= 0.0
        or ohlcv["volume"] < 0.0
    ):
        return None

    if (
        ohlcv["high"] < ohlcv["low"]
        or ohlcv["high"] < ohlcv["open"]
        or ohlcv["high"] < ohlcv["close"]
        or ohlcv["low"] > ohlcv["open"]
        or ohlcv["low"] > ohlcv["close"]
    ):
        return None

    return {
        "timestamp": c_ts,
        "open": ohlcv["open"],
        "high": ohlcv["high"],
        "low": ohlcv["low"],
        "close": ohlcv["close"],
        "volume": ohlcv["volume"],
    }


def backfill_missing_candles(session, candle: Dict[str, Any]) -> int:
    """
    Detects and backfills missing historical confirmed 1-hour candles into `market_ticks`
    using the `candle["candles"]` history already fetched from the exchange.

    Contract & Invariants:
    - If `candle["candles"]` is absent/empty or the database has zero existing ticks prior
      to `candle["timestamp"]` (initial empty-DB bootstrap), returns 0 without inserting
      historical bars.
    - Backfills missing timestamps strictly after the earliest prior database tick and
      before `latest_ts` (`anchor_ts < c_ts < latest_ts`), even if a later timestamp
      already exists inside the fetched window.
    - Populates OHLCV and schema version fields while setting `market_regime`,
      `predicted_return_1h`, `predicted_price_1h`, `model_confidence`, and `features` to None.
    - Never fabricates retrospective predictions, CouncilAudit records, or PaperPortfolio trades.
    - Wraps each insert in a nested transaction (SAVEPOINT) catching `IntegrityError` so
      duplicate timestamps are skipped safely without aborting the outer transaction.
    """
    if not isinstance(candle, dict) or "timestamp" not in candle:
        return 0

    raw_history = candle.get("candles")
    if not raw_history or not isinstance(raw_history, Sequence) or isinstance(raw_history, (str, bytes)):
        return 0

    try:
        latest_ts = _normalize_utc_timestamp(candle["timestamp"])
    except (ValueError, TypeError):
        return 0

    candidates_by_ts: Dict[datetime, Dict[str, Any]] = {}
    for raw_c in raw_history:
        validated = _validate_historical_candle_dict(raw_c)
        if validated is None:
            logger.warning("[BACKFILL] Skipping malformed historical candle entry: %r", raw_c)
            continue
        c_ts = validated["timestamp"]
        if c_ts < latest_ts:
            candidates_by_ts[c_ts] = validated

    if not candidates_by_ts:
        return 0

    sorted_candidates = [candidates_by_ts[ts] for ts in sorted(candidates_by_ts.keys())]
    window_start_ts = sorted_candidates[0]["timestamp"]

    prior_latest_tick = session.execute(
        select(MarketTick)
        .where(MarketTick.timestamp < latest_ts)
        .order_by(MarketTick.timestamp.desc())
        .limit(1)
    ).scalars().first()

    if prior_latest_tick is None:
        return 0

    prior_latest_ts = _normalize_utc_timestamp(prior_latest_tick.timestamp)
    gap_hours = (latest_ts - prior_latest_ts).total_seconds() / 3600.0

    if prior_latest_ts < window_start_ts:
        logger.warning(
            "[BACKFILL] Outage gap of %.1f hours (last tick %s, current %s) exceeds exchange "
            "history window starting at %s. Backfilling available window candles only.",
            gap_hours,
            prior_latest_ts.isoformat(),
            latest_ts.isoformat(),
            window_start_ts.isoformat(),
        )
    elif gap_hours > BACKFILL_LARGE_GAP_WARNING_HOURS:
        logger.warning(
            "[BACKFILL] Detected large candle gap of %.1f hours (> %.1fh threshold) between %s and %s.",
            gap_hours,
            BACKFILL_LARGE_GAP_WARNING_HOURS,
            prior_latest_ts.isoformat(),
            latest_ts.isoformat(),
        )

    existing_rows = session.execute(
        select(MarketTick.timestamp).where(
            MarketTick.timestamp < latest_ts,
        )
    ).scalars().all()
    existing_timestamps = {_normalize_utc_timestamp(r) for r in existing_rows if r is not None}

    anchor_ts = min(existing_timestamps | {prior_latest_ts})

    inserted_count = 0
    for c_data in sorted_candidates:
        c_ts = c_data["timestamp"]
        if c_ts <= anchor_ts or c_ts in existing_timestamps:
            continue

        backfilled_tick = MarketTick(
            timestamp=c_ts,
            open=c_data["open"],
            high=c_data["high"],
            low=c_data["low"],
            close=c_data["close"],
            volume=c_data["volume"],
            model_version=ACTIVE_MODEL_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            market_regime=None,
            predicted_return_1h=None,
            predicted_price_1h=None,
            model_confidence=None,
            features=None,
        )
        try:
            with session.begin_nested():
                session.add(backfilled_tick)
                session.flush()
            existing_timestamps.add(c_ts)
            inserted_count += 1
        except IntegrityError:
            logger.info(
                "[BACKFILL] Duplicate candle timestamp %s skipped via savepoint.",
                c_ts.isoformat(),
            )

    if inserted_count > 0:
        logger.info(
            "[BACKFILL] Inserted %d missing historical candle(s) prior to %s.",
            inserted_count,
            latest_ts.isoformat(),
        )

    return inserted_count


def is_paper_trade_executed_for_tick(session, tick_id: int) -> bool:
    """
    Checks if a paper order (BUY, SELL, or HOLD) has already been executed for tick_id.
    First checks PaperPortfolio.tick_id directly if present in the database schema.
    As a fallback for historical rows where tick_id was NULL, checks whether an initial
    CouncilAudit without [FORCED_REANALYSIS] already exists for this tick_id.
    """
    if has_tick_id_column(session):
        trade_id = session.execute(
            select(PaperPortfolio.id).where(PaperPortfolio.tick_id == tick_id)
        ).scalar_one_or_none()
        if trade_id is not None:
            return True

    audit = session.execute(
        select(CouncilAudit).where(CouncilAudit.tick_id == tick_id).order_by(CouncilAudit.id.asc())
    ).scalars().first()
    if audit is not None and (not audit.execution_notes or "[FORCED_REANALYSIS]" not in audit.execution_notes):
        return True

    return False


def parse_retry_after(
    retry_after_header: Optional[str],
    now_dt_fn: Optional[Callable[[], datetime]] = None,
) -> Optional[float]:
    """
    Parses Retry-After header per RFC 7231 Section 7.1.3 and RFC 9110 Section 10.2.3.
    Accepts either non-negative integer/float seconds, or an RFC 2822 / RFC 822 HTTP-date string.
    Returns the required delay in seconds as a float, or None if missing, malformed,
    negative, or non-finite.
    """
    if not retry_after_header or not isinstance(retry_after_header, str):
        return None
    val = retry_after_header.strip()
    if not val:
        return None

    # 1. Direct seconds representation (e.g. "5", "120")
    try:
        sec = float(val)
        if math.isfinite(sec) and sec >= 0.0:
            return sec
        return None
    except ValueError:
        pass

    # 2. HTTP-date representation (e.g. "Wed, 21 Oct 2026 07:28:00 GMT")
    try:
        parsed_dt = parsedate_to_datetime(val)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

    if parsed_dt is not None:
        now_dt = now_dt_fn() if now_dt_fn is not None else datetime.now(timezone.utc)
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
        delta = (parsed_dt - now_dt).total_seconds()
        return max(0.0, delta) if math.isfinite(delta) else None

    return None


def _get_retry_after_header(headers: Any) -> Optional[str]:
    """Extracts Retry-After header case-insensitively from a response headers mapping."""
    if not headers or not hasattr(headers, "get"):
        return None
    direct = headers.get("Retry-After")
    if direct is not None:
        return direct if isinstance(direct, str) else str(direct)
    if hasattr(headers, "items"):
        for k, v in headers.items():
            if isinstance(k, str) and k.lower() == "retry-after" and v is not None:
                return v if isinstance(v, str) else str(v)
    return None


def _parse_trade_count(raw_trades: Any) -> Optional[int]:
    """
    Validates and converts a candle trade count into a non-negative integer.
    - Returns None when `raw_trades` is None (unobserved).
    - Accepts non-negative integers (e.g. 1500, 0), non-negative integral finite
      floats (e.g. 1500.0), and non-negative integral numeric strings (e.g. "1500", "1500.0").
    - Rejects booleans, negative values, fractional values (e.g. 1.5, "1.5"),
      non-finite values (NaN, inf, -inf), and non-numeric types by raising ValueError.
    """
    if raw_trades is None:
        return None
    if isinstance(raw_trades, bool):
        raise ValueError("Boolean value not permitted for trade count")
    if isinstance(raw_trades, int):
        if raw_trades < 0:
            raise ValueError(f"Negative trade count not permitted: {raw_trades}")
        return int(raw_trades)
    if isinstance(raw_trades, float):
        if not math.isfinite(raw_trades) or raw_trades < 0.0 or not raw_trades.is_integer():
            raise ValueError(f"Non-integral, negative, or non-finite float trade count: {raw_trades}")
        return int(raw_trades)
    if isinstance(raw_trades, str):
        s = raw_trades.strip()
        if not s or s.startswith("-"):
            raise ValueError(f"Invalid or negative string trade count: {raw_trades!r}")
        try:
            dec = Decimal(s)
        except InvalidOperation as exc:
            raise ValueError(f"Non-numeric string trade count: {raw_trades!r}") from exc
        if not dec.is_finite() or dec < 0 or dec != int(dec):
            raise ValueError(f"Non-integral, negative, or non-finite string trade count: {raw_trades!r}")
        return int(dec)
    raise ValueError(f"Unsupported trade count type: {type(raw_trades).__name__}")


def _validate_candle_payload(
    payload: Any,
    min_row_len: int = COINBASE_MIN_ROW_LEN,
    feed_name: str = "Feed",
) -> bool:
    """
    Validates that a decoded JSON payload is a non-empty list of candle sequences,
    where every row has at least `min_row_len` elements, all required OHLCV/timestamp
    fields parse as finite floats, and Binance trade counts parse as non-negative
    integral values (rejecting fractional, negative, non-finite, boolean, and string noise).
    """
    if not isinstance(payload, list):
        print(f"[FEED] {feed_name} returned non-list JSON payload (type: {type(payload).__name__}). Failing over.")
        return False
    if not payload:
        print(f"[FEED] {feed_name} returned empty list. Failing over.")
        return False

    float_indices: Tuple[int, ...] = (
        (0, 1, 2, 3, 4, 5, 6, 9)
        if min_row_len >= BINANCE_MIN_ROW_LEN
        else (0, 1, 2, 3, 4, 5)
    )

    for idx, row in enumerate(payload):
        if not isinstance(row, (list, tuple)):
            print(f"[FEED] {feed_name} row #{idx} has unexpected record type ({type(row).__name__}). Failing over.")
            return False
        if len(row) < min_row_len:
            print(
                f"[FEED] {feed_name} row #{idx} has insufficient length "
                f"({len(row)} < {min_row_len}). Failing over."
            )
            return False

        for col_idx in float_indices:
            if col_idx >= len(row):
                print(f"[FEED] {feed_name} row #{idx} missing required column index {col_idx}. Failing over.")
                return False
            raw_val = row[col_idx]
            if isinstance(raw_val, bool):
                print(f"[FEED] {feed_name} row #{idx} column {col_idx} contains boolean value. Failing over.")
                return False
            try:
                num_val = float(raw_val)
            except (TypeError, ValueError):
                print(f"[FEED] {feed_name} row #{idx} column {col_idx} is non-numeric. Failing over.")
                return False
            if not math.isfinite(num_val):
                print(f"[FEED] {feed_name} row #{idx} column {col_idx} is non-finite ({raw_val}). Failing over.")
                return False

        if min_row_len >= BINANCE_MIN_ROW_LEN:
            try:
                _parse_trade_count(row[8])
            except ValueError as exc:
                print(f"[FEED] {feed_name} row #{idx} trade count column 8 is invalid ({exc}). Failing over.")
                return False

    return True


def _fetch_feed_with_retry(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    feed_name: str = "Binance Vision",
    deadline_ts: Optional[float] = None,
    max_attempts: int = EXCHANGE_RETRY_MAX_ATTEMPTS,
    connect_timeout: float = EXCHANGE_CONNECT_TIMEOUT,
    read_timeout: float = EXCHANGE_READ_TIMEOUT,
    base_backoff: float = EXCHANGE_RETRY_BASE_BACKOFF,
    backoff_factor: float = EXCHANGE_RETRY_BACKOFF_FACTOR,
    max_backoff: float = EXCHANGE_RETRY_MAX_BACKOFF,
    max_retry_after: float = EXCHANGE_MAX_RETRY_AFTER_CAP,
    total_deadline: float = EXCHANGE_RETRY_TOTAL_DEADLINE,
    min_row_len: Optional[int] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    jitter_fn: Callable[[float, float], float] = random.uniform,
    clock_fn: Callable[[], float] = time.perf_counter,
    now_dt_fn: Optional[Callable[[], datetime]] = None,
) -> Optional[List[Any]]:
    """
    Executes an HTTP GET request with bounded retries, exponential backoff, and jitter.

    Error Classification & Retry Policy:
    - Transient Network: requests.exceptions.Timeout (ConnectTimeout, ReadTimeout),
      requests.exceptions.ConnectionError, and ChunkedEncodingError are retried with
      exponential backoff + jitter.
    - Transient HTTP Status: 500, 502, 503, 504 are retried with exponential backoff + jitter.
    - Transient Rate Limits: 429 and Binance 418 inspect Retry-After. If Retry-After <= max_retry_after,
      the scheduled delay is clamped to min(max_retry_after, retry_after_val + jitter) so it never
      exceeds max_retry_after. If that delay fits within deadline_ts, sleeps and retries. If Retry-After
      exceeds max_retry_after, is missing/malformed, or exceeds deadline_ts, fails over promptly.
    - Malformed JSON: json.JSONDecodeError / ValueError under HTTP 200 is retried with backoff if
      attempts and budget remain.
    - Invalid Payload Structure / Non-Finite Fields: Validated via _validate_candle_payload before
      returning; invalid payloads fail over cleanly without crashing the caller.
    - Permanent HTTP Status: 400, 401, 403, 404, 405 (and other non-429/418 4xx) and non-retryable
      requests.exceptions.RequestException (e.g. InvalidURL, MissingSchema) fail fast without retry.
    - Unexpected Programming Errors: Non-RequestException errors (e.g. TypeError, AttributeError)
      are NOT swallowed and propagate immediately as diagnosable failures.

    Retry-Scheduling Budget & Timeout Semantics:
    - `deadline_ts` (initialized from `clock_fn() + total_deadline` when not supplied) is a
      cooperative retry-scheduling budget for this provider call, NOT a guaranteed hard wall-clock
      timeout.
    - Before each request and before each sleep, remaining budget (`deadline_ts - clock_fn()`) is
      checked. If remaining budget < 0.5s before an attempt, or if a scheduled sleep would exceed
      `deadline_ts`, the attempt or sleep is refused and the helper returns None.
    - Per-attempt socket timeouts are passed as `(min(connect_timeout, remaining_budget),
      min(read_timeout, remaining_budget))`. In Requests/urllib3, connect and read timeouts are
      separate socket-phase limits (and read timeouts apply per socket read operation rather than
      as a single total transfer timer). Consequently, an in-flight request can exceed `deadline_ts`
      before returning control to Python, and this implementation does not establish a strict
      wall-clock overshoot bound on an in-flight socket call.
    - However, once control returns from `requests.get()` (and after `resp.json()`), if
      `clock_fn() > deadline_ts`, the late response is explicitly rejected (`return None`) rather
      than accepted.
    """
    if deadline_ts is None:
        deadline_ts = clock_fn() + total_deadline

    required_row_len = (
        min_row_len
        if min_row_len is not None
        else (BINANCE_MIN_ROW_LEN if "binance" in feed_name.lower() else COINBASE_MIN_ROW_LEN)
    )

    for attempt in range(1, max_attempts + 1):
        now = clock_fn()
        remaining_budget = deadline_ts - now

        if remaining_budget <= 0.0:
            print(f"[FEED] {feed_name} retry budget exhausted before attempt #{attempt} (deadline elapsed).")
            return None

        # Do not initiate an HTTP handshake if remaining scheduling budget is insufficient
        if remaining_budget < 0.5:
            print(f"[FEED] {feed_name} remaining budget ({remaining_budget:.2f}s) insufficient for attempt #{attempt}. Aborting.")
            return None

        attempt_connect_timeout = min(connect_timeout, remaining_budget)
        attempt_read_timeout = min(read_timeout, remaining_budget)
        timeout_tuple = (attempt_connect_timeout, attempt_read_timeout)

        try:
            resp = requests.get(url, headers=headers, timeout=timeout_tuple)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.Timeout) as exc:
            print(f"[FEED] {feed_name} attempt #{attempt}/{max_attempts} timed out: {type(exc).__name__}.")
            if attempt >= max_attempts:
                return None
            backoff = min(max_backoff, base_backoff * (backoff_factor ** (attempt - 1)))
            raw_jitter = jitter_fn(0.1, 0.5)
            jitter = raw_jitter if (math.isfinite(raw_jitter) and raw_jitter >= 0.0) else 0.0
            delay = backoff + jitter
            now_after_call = clock_fn()
            if (now_after_call + delay) > deadline_ts:
                print(f"[FEED] {feed_name} remaining deadline ({deadline_ts - now_after_call:.2f}s) cannot accommodate backoff delay ({delay:.2f}s). Aborting retries.")
                return None
            sleep_fn(delay)
            continue
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
            print(f"[FEED] {feed_name} attempt #{attempt}/{max_attempts} connection error: {exc}.")
            if attempt >= max_attempts:
                return None
            backoff = min(max_backoff, base_backoff * (backoff_factor ** (attempt - 1)))
            raw_jitter = jitter_fn(0.1, 0.5)
            jitter = raw_jitter if (math.isfinite(raw_jitter) and raw_jitter >= 0.0) else 0.0
            delay = backoff + jitter
            now_after_call = clock_fn()
            if (now_after_call + delay) > deadline_ts:
                print(f"[FEED] {feed_name} remaining deadline ({deadline_ts - now_after_call:.2f}s) cannot accommodate backoff delay ({delay:.2f}s). Aborting retries.")
                return None
            sleep_fn(delay)
            continue
        except (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema, requests.exceptions.InvalidHeader) as exc:
            print(f"[FEED] {feed_name} fatal client configuration error: {exc}.")
            return None
        except requests.exceptions.RequestException as exc:
            print(f"[FEED] {feed_name} non-retryable request error: {exc}.")
            return None

        status = resp.status_code
        if not isinstance(status, int):
            raise TypeError(f"{feed_name} response status_code must be int, got {type(status).__name__}")

        # Reject any response that completed after the provider's scheduling deadline elapsed
        now_after_resp = clock_fn()
        if now_after_resp > deadline_ts:
            print(
                f"[FEED] {feed_name} attempt #{attempt}/{max_attempts} (HTTP {status}) completed "
                f"after scheduling deadline elapsed ({now_after_resp - deadline_ts:.2f}s late). Rejecting."
            )
            return None

        # 1. Permanent client errors (400, 401, 403, 404, 405, and other 4xx except 429/418)
        if (400 <= status < 500) and status not in (429, 418):
            print(f"[FEED] {feed_name} returned permanent HTTP {status}. Bailing out without retry.")
            return None

        # 2. Rate limit and auto-ban (429, 418)
        if status in (429, 418):
            retry_after_str = _get_retry_after_header(getattr(resp, "headers", None))
            retry_after_val = parse_retry_after(retry_after_str, now_dt_fn=now_dt_fn)
            if retry_after_val is None:
                print(f"[FEED] {feed_name} rate limit HTTP {status} missing or unparseable Retry-After header. Failing over promptly.")
                return None
            if retry_after_val > max_retry_after:
                print(f"[FEED] {feed_name} rate limit HTTP {status} Retry-After ({retry_after_val:.2f}s) exceeds maximum cap ({max_retry_after:.2f}s). Failing over promptly.")
                return None
            if attempt >= max_attempts:
                return None
            raw_jitter = jitter_fn(0.1, 0.5)
            jitter = raw_jitter if (math.isfinite(raw_jitter) and raw_jitter >= 0.0) else 0.0
            # Clamp scheduled sleep so Retry-After + jitter never exceeds max_retry_after cap
            delay = min(max_retry_after, retry_after_val + jitter)
            now_after_call = clock_fn()
            if (now_after_call + delay) > deadline_ts:
                print(f"[FEED] {feed_name} rate limit Retry-After delay ({delay:.2f}s) exceeds overall deadline ({deadline_ts - now_after_call:.2f}s). Failing over promptly.")
                return None
            print(f"[FEED] {feed_name} rate limit HTTP {status}. Honoring Retry-After delay of {delay:.2f}s before attempt #{attempt + 1}...")
            sleep_fn(delay)
            continue

        # 3. Transient server errors (500, 502, 503, 504)
        if status in (500, 502, 503, 504):
            print(f"[FEED] {feed_name} attempt #{attempt}/{max_attempts} transient server error HTTP {status}.")
            if attempt >= max_attempts:
                return None
            backoff = min(max_backoff, base_backoff * (backoff_factor ** (attempt - 1)))
            raw_jitter = jitter_fn(0.1, 0.5)
            jitter = raw_jitter if (math.isfinite(raw_jitter) and raw_jitter >= 0.0) else 0.0
            delay = backoff + jitter
            now_after_call = clock_fn()
            if (now_after_call + delay) > deadline_ts:
                print(f"[FEED] {feed_name} remaining deadline ({deadline_ts - now_after_call:.2f}s) cannot accommodate backoff delay ({delay:.2f}s). Aborting retries.")
                return None
            sleep_fn(delay)
            continue

        # 4. Successful response (HTTP 200)
        if status == 200:
            try:
                payload = resp.json()
            except (json.JSONDecodeError, ValueError) as json_err:
                print(f"[FEED] {feed_name} attempt #{attempt}/{max_attempts} malformed JSON body: {json_err}.")
                if attempt >= max_attempts:
                    return None
                backoff = min(max_backoff, base_backoff * (backoff_factor ** (attempt - 1)))
                raw_jitter = jitter_fn(0.1, 0.5)
                jitter = raw_jitter if (math.isfinite(raw_jitter) and raw_jitter >= 0.0) else 0.0
                delay = backoff + jitter
                now_after_call = clock_fn()
                if (now_after_call + delay) > deadline_ts:
                    print(f"[FEED] {feed_name} remaining deadline cannot accommodate backoff delay. Aborting.")
                    return None
                sleep_fn(delay)
                continue

            if clock_fn() > deadline_ts:
                print(f"[FEED] {feed_name} attempt #{attempt}/{max_attempts} JSON decode completed after scheduling deadline. Rejecting.")
                return None

            if not _validate_candle_payload(payload, min_row_len=required_row_len, feed_name=feed_name):
                return None

            return payload

        # Any other unhandled HTTP status
        print(f"[FEED] {feed_name} unexpected HTTP {status}. Bailing out.")
        return None

    return None


def fetch_latest_confirmed_candle(
    symbol: str = "BTCUSDT",
    sleep_fn: Optional[Callable[[float], None]] = None,
    jitter_fn: Optional[Callable[[float, float], float]] = None,
    clock_fn: Optional[Callable[[], float]] = None,
    now_dt_fn: Optional[Callable[[], datetime]] = None,
    total_deadline: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fetches the most recently confirmed closed 1-hour candle alongside
    the preceding sequence of historical candles for rolling feature parity.
    Primary: Binance Dedicated Vision. Fallback: Coinbase Institutional.

    Per-Provider Retry-Scheduling Budget Contract:
    - `total_deadline` (defaulting to `EXCHANGE_RETRY_TOTAL_DEADLINE` = 25.0s) is applied
      as a separate cooperative retry-scheduling budget for each provider (`Binance Vision`
      and `Coinbase Institutional`), rather than a combined wall-clock cap across both
      providers. This ensures that slow Binance retries or timeouts never starve Coinbase
      of its fallback attempt budget. Consequently, cumulative elapsed time across both
      providers can exceed `EXCHANGE_RETRY_TOTAL_DEADLINE` when primary failover occurs.
    """
    _sleep_fn = sleep_fn if sleep_fn is not None else time.sleep
    _jitter_fn = jitter_fn if jitter_fn is not None else random.uniform
    _clock_fn = clock_fn if clock_fn is not None else time.perf_counter
    _provider_budget = total_deadline if total_deadline is not None else EXCHANGE_RETRY_TOTAL_DEADLINE

    # 1. Binance Dedicated Vision (Primary Provider)
    binance_deadline_ts = _clock_fn() + _provider_budget
    binance_url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit=102"
    binance_klines = _fetch_feed_with_retry(
        url=binance_url,
        headers=None,
        feed_name="Binance Vision",
        deadline_ts=binance_deadline_ts,
        min_row_len=BINANCE_MIN_ROW_LEN,
        sleep_fn=_sleep_fn,
        jitter_fn=_jitter_fn,
        clock_fn=_clock_fn,
        now_dt_fn=now_dt_fn,
    )

    if binance_klines is not None and _validate_candle_payload(
        binance_klines, min_row_len=BINANCE_MIN_ROW_LEN, feed_name="Binance Vision"
    ):
        now_ms = (now_dt_fn().timestamp() if now_dt_fn else datetime.now(timezone.utc).timestamp()) * 1000.0
        confirmed_klines = [k for k in binance_klines if float(k[6]) <= now_ms]

        if len(confirmed_klines) >= 35:
            selected_klines = confirmed_klines[-100:]
            c = selected_klines[-1]
            ts = datetime.fromtimestamp(float(c[0]) / 1000.0, tz=timezone.utc)
            vol = float(c[5])
            taker_base = float(c[9])
            taker_ratio = float(taker_base / vol) if vol > 0 else 0.50

            candle_records = []
            for k in selected_klines:
                k_vol = float(k[5])
                k_taker = float(k[9])
                k_ratio = float(k_taker / k_vol) if k_vol > 0 else 0.50
                candle_records.append({
                    "timestamp": pd.to_datetime(float(k[0]), unit="ms", utc=True),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": k_vol,
                    "number_of_trades": _parse_trade_count(k[8]),
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
                "number_of_trades": _parse_trade_count(c[8]),
                "candles": candle_records,
            }
        else:
            print(f"[FEED] Binance Vision returned insufficient confirmed closed candles ({len(confirmed_klines)} < 35). Falling back to Coinbase.")

    # 2. Coinbase Institutional Fallback (allocated its own bounded retry-scheduling budget)
    coinbase_deadline_ts = _clock_fn() + _provider_budget
    coinbase_url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
    coinbase_headers = {"User-Agent": "Mudra-Sync/1.0"}
    coinbase_candles = _fetch_feed_with_retry(
        url=coinbase_url,
        headers=coinbase_headers,
        feed_name="Coinbase Institutional",
        deadline_ts=coinbase_deadline_ts,
        min_row_len=COINBASE_MIN_ROW_LEN,
        sleep_fn=_sleep_fn,
        jitter_fn=_jitter_fn,
        clock_fn=_clock_fn,
        now_dt_fn=now_dt_fn,
    )

    if coinbase_candles is not None and _validate_candle_payload(
        coinbase_candles, min_row_len=COINBASE_MIN_ROW_LEN, feed_name="Coinbase Institutional"
    ):
        now_sec = (now_dt_fn().timestamp() if now_dt_fn else datetime.now(timezone.utc).timestamp())
        confirmed_candles = [k for k in coinbase_candles if (float(k[0]) + 3600.0) <= now_sec]

        if len(confirmed_candles) >= 35:
            selected_candles = list(reversed(confirmed_candles[:100]))
            c = selected_candles[-1]
            ts = datetime.fromtimestamp(float(c[0]), tz=timezone.utc)

            candle_records = []
            for k in selected_candles:
                candle_records.append({
                    "timestamp": pd.to_datetime(float(k[0]), unit="s", utc=True),
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
        else:
            print(f"[FEED] Coinbase Institutional returned insufficient confirmed closed candles ({len(confirmed_candles)} < 35).")

    raise MarketFeedExhaustedError("Failed to fetch confirmed 1-hour candle from all market feeds.")


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
    except MarketFeedExhaustedError as exc:
        return False, f"Market feed error: {exc}", {}

    with get_db() as session:
        latest_ts = _normalize_utc_timestamp(candle["timestamp"])
        backfill_missing_candles(session, candle)

        existing = session.execute(
            select(MarketTick).where(MarketTick.timestamp == latest_ts).order_by(MarketTick.id.desc())
        ).scalars().first()

        if existing and not force:
            reconcile_pending_ticks(session, existing)
            return False, f"Candle for {latest_ts.strftime('%H:00 UTC')} already recorded in Neon.", {"tick_id": existing.id}

        is_existing_tick = existing is not None
        already_executed = is_paper_trade_executed_for_tick(session, existing.id) if is_existing_tick else False

        # 1. Compute Signals and XGBoost Return Forecast
        signals_output = calculate_technical_signals.invoke({
            "close": candle["close"],
            "open_price": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "volume": candle["volume"],
            "taker_buy_ratio": candle.get("taker_buy_ratio"),
            "timestamp": latest_ts,
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
        if is_existing_tick:
            current_tick = existing
            if force:
                current_tick.market_regime = regime
                current_tick.predicted_return_1h = pred_return
                current_tick.predicted_price_1h = pred_price
                current_tick.model_confidence = confidence
                current_tick.features = signals_output
                current_tick.model_version = ACTIVE_MODEL_VERSION
                current_tick.feature_schema_version = FEATURE_SCHEMA_VERSION
                session.flush()
        else:
            current_tick = MarketTick(
                timestamp=latest_ts,
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle["volume"],
                model_version=ACTIVE_MODEL_VERSION,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
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

        # 5. Record Council Audit (Auditable trail of forced reanalysis)
        if is_existing_tick and already_executed:
            exec_notes = f"[FORCED_REANALYSIS] {agent_state.get('thesis', '')[:350]} | Paper trade previously executed for tick #{current_tick.id}; financial execution deduplicated."
        else:
            exec_notes = agent_state.get("thesis", "")[:500]

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
            execution_notes=exec_notes,
        )
        session.add(audit)
        session.flush()

        # 6. Execute Forward Paper Order (Strict Financial Idempotency Guard)
        if is_existing_tick and already_executed:
            print(f"[PAPER] Trade already executed for tick #{current_tick.id}. Financial execution deduplicated (zero fees charged).")
        else:
            order_notes = f"Executed {decision} ({verdict}) at BTC ${candle['close']:,.2f}."
            execute_paper_order(
                session=session,
                action=decision,
                current_price=candle["close"],
                kelly_fraction=kelly,
                notes=order_notes,
                tick_id=current_tick.id,
            )

        summary = get_portfolio_summary(session, candle["close"])
        if is_existing_tick and already_executed:
            return True, f"Updated forecast for existing tick #{current_tick.id} at ${candle['close']:,.2f} ({decision} | Trade deduplicated | No fees)", summary
        return True, f"Recorded tick #{current_tick.id} at ${candle['close']:,.2f} ({decision} | Kelly: {kelly*100:.1f}%)", summary


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    print(f"Triggering sync cycle (force={force_flag})...")
    success, msg, summ = sync_hourly_quant_cycle(force=force_flag)
    print(f"Status: {success} | Message: {msg}")
    if summ:
        print(f"Portfolio Summary: {summ}")
