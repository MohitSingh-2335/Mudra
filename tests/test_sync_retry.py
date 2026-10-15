# tests/test_sync_retry.py
"""
Deterministic Unit & Regression Tests for Bounded Exchange Retries.
Verify _fetch_feed_with_retry and fetch_latest_confirmed_candle
retry behavior, error classification, cooperative retry-scheduling budget enforcement,
late-response rejection, payload validation, and confirmed-candle filtering under
mocked HTTP, sleep, clock, and jitter dependencies.
"""

import json
import math
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call
import requests

from config import EXCHANGE_MAX_RETRY_AFTER_CAP, EXCHANGE_RETRY_TOTAL_DEADLINE
from src.engine.sync import (
    MarketFeedExhaustedError,
    parse_retry_after,
    _parse_trade_count,
    _validate_candle_payload,
    _fetch_feed_with_retry,
    fetch_latest_confirmed_candle,
    sync_hourly_quant_cycle,
)


# ---------------------------------------------------------------------------
# Helpers: Mock Response Factories
# ---------------------------------------------------------------------------

def _make_response(status_code, body=None, headers=None):
    """Creates a mock requests.Response with the given status and JSON body."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    if body is not None:
        resp.json.return_value = body
    else:
        resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    return resp


def _make_binance_klines(n=102, now_dt=None):
    """Generates n mock Binance klines, all confirmed closed before now_dt."""
    if now_dt is None:
        now_dt = datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc)
    now_ms = now_dt.timestamp() * 1000.0
    klines = []
    for i in range(n):
        open_time = now_ms - (n + 1 - i) * 3600000
        close_time = open_time + 3600000 - 1  # Confirmed: close_time < now_ms
        klines.append([
            open_time, "60000.0", "60500.0", "59800.0", "60200.0", "100.0",
            close_time, "6020000.0", 1500, "55.0", "3311000.0", "0"
        ])
    return klines


def _make_coinbase_candles(n=100, now_dt=None):
    """Generates n mock Coinbase candles, descending, all confirmed closed."""
    if now_dt is None:
        now_dt = datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc)
    now_sec = now_dt.timestamp()
    candles = []
    for i in range(n):
        candle_sec = now_sec - (i + 1) * 3600
        candles.append([candle_sec, 59800.0, 60500.0, 60000.0, 60250.0, 100.0])
    return candles


class MockClock:
    """Deterministic clock that advances by a fixed increment on each call."""
    def __init__(self, start=1000.0, increment=0.001):
        self._t = start
        self._increment = increment

    def __call__(self):
        t = self._t
        self._t += self._increment
        return t


class SleepRecorder:
    """Records sleep calls without actually sleeping. Optionally advances a clock."""
    def __init__(self, clock=None):
        self.calls = []
        self._clock = clock

    def __call__(self, seconds):
        self.calls.append(seconds)
        if self._clock is not None:
            self._clock._t += seconds


# Fixed jitter for determinism: always returns 0.25
def _fixed_jitter(lo, hi):
    return 0.25


FIXED_NOW = datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc)


def _fixed_now_dt():
    return FIXED_NOW


# ---------------------------------------------------------------------------
# 0. parse_retry_after tests
# ---------------------------------------------------------------------------

class TestParseRetryAfter:
    def test_integer_seconds(self):
        assert parse_retry_after("5") == 5.0

    def test_float_seconds(self):
        assert parse_retry_after("3.5") == 3.5

    def test_zero_seconds(self):
        assert parse_retry_after("0") == 0.0

    def test_negative_returns_none(self):
        assert parse_retry_after("-1") is None

    def test_nan_returns_none(self):
        assert parse_retry_after("nan") is None

    def test_inf_returns_none(self):
        assert parse_retry_after("inf") is None

    def test_empty_string_returns_none(self):
        assert parse_retry_after("") is None

    def test_none_returns_none(self):
        assert parse_retry_after(None) is None

    def test_garbage_returns_none(self):
        assert parse_retry_after("not-a-number-or-date") is None

    def test_http_date_future(self):
        now = datetime(2026, 10, 21, 7, 28, 0, tzinfo=timezone.utc)
        header = "Wed, 21 Oct 2026 07:28:05 GMT"
        result = parse_retry_after(header, now_dt_fn=lambda: now)
        assert result is not None
        assert 4.5 <= result <= 5.5

    def test_http_date_past_returns_zero(self):
        now = datetime(2026, 10, 21, 7, 30, 0, tzinfo=timezone.utc)
        header = "Wed, 21 Oct 2026 07:28:00 GMT"
        result = parse_retry_after(header, now_dt_fn=lambda: now)
        assert result == 0.0


# ---------------------------------------------------------------------------
# 1. Transient connection failure followed by success
# ---------------------------------------------------------------------------

def test_binance_transient_connection_failure_then_success(monkeypatch):
    """Attempt 1 raises ConnectTimeout; Attempt 2 succeeds with HTTP 200."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)
    klines = _make_binance_klines(now_dt=FIXED_NOW)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise requests.exceptions.ConnectTimeout("Connection timed out")
        return _make_response(200, body=klines)

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=102",
        feed_name="Binance Vision",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
    )
    assert result is not None
    assert len(result) == 102
    assert call_count[0] == 2
    assert len(sleeper.calls) == 1
    # Backoff: min(6.0, 1.0 * 2^0) + 0.25 = 1.25
    assert abs(sleeper.calls[0] - 1.25) < 0.01


# ---------------------------------------------------------------------------
# 2. Binance exhausted retries → Coinbase fallback success
# ---------------------------------------------------------------------------

def test_binance_exhausted_retries_coinbase_fallback(monkeypatch):
    """Binance returns HTTP 503 on all 3 attempts. Coinbase succeeds."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)
    cb_candles = _make_coinbase_candles(now_dt=FIXED_NOW)

    call_count = {"binance": 0, "coinbase": 0}
    def mock_get(url, **kwargs):
        if "binance" in url:
            call_count["binance"] += 1
            return _make_response(503)
        call_count["coinbase"] += 1
        return _make_response(200, body=cb_candles)

    monkeypatch.setattr(requests, "get", mock_get)

    result = fetch_latest_confirmed_candle(
        symbol="BTCUSDT",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
        total_deadline=50.0,
    )
    assert result is not None
    assert result["close"] == 60250.0
    assert result["number_of_trades"] is None  # Coinbase unobserved
    assert call_count["binance"] == 3
    assert call_count["coinbase"] == 1


# ---------------------------------------------------------------------------
# 3. Permanent HTTP error does not retry, falls back safely
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 405])
def test_permanent_http_error_no_retry(monkeypatch, status_code):
    """Permanent HTTP errors bail out immediately with exactly 1 call."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        return _make_response(status_code)

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines",
        feed_name="Binance Vision",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
    )
    assert result is None
    assert call_count[0] == 1
    assert len(sleeper.calls) == 0


# ---------------------------------------------------------------------------
# 4. Retry-After within cap is honored
# ---------------------------------------------------------------------------

def test_retry_after_within_cap_honored(monkeypatch):
    """HTTP 429 with Retry-After: 3 → sleep ~3.25s, then succeed on retry."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)
    klines = _make_binance_klines(now_dt=FIXED_NOW)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _make_response(429, headers={"Retry-After": "3"})
        return _make_response(200, body=klines)

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines",
        feed_name="Binance Vision",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
    )
    assert result is not None
    assert call_count[0] == 2
    assert len(sleeper.calls) == 1
    # Retry-After 3.0 + jitter 0.25 = 3.25
    assert abs(sleeper.calls[0] - 3.25) < 0.01


# ---------------------------------------------------------------------------
# 5. Retry-After above cap fails over without excessive sleep
# ---------------------------------------------------------------------------

def test_retry_after_above_cap_fails_fast(monkeypatch):
    """HTTP 429 with Retry-After: 60 exceeds 15s cap → no sleep, returns None."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        return _make_response(429, headers={"Retry-After": "60"})

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines",
        feed_name="Binance Vision",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
    )
    assert result is None
    assert call_count[0] == 1  # Exactly 1 attempt, no retries
    assert len(sleeper.calls) == 0  # Zero sleep


# ---------------------------------------------------------------------------
# 6. Both providers exhausted → RuntimeError
# ---------------------------------------------------------------------------

def test_both_providers_exhausted_raises_runtime_error(monkeypatch):
    """Both Binance and Coinbase exhaust retries → RuntimeError."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    def mock_get(url, **kwargs):
        return _make_response(502)

    monkeypatch.setattr(requests, "get", mock_get)

    with pytest.raises(RuntimeError, match="Failed to fetch confirmed 1-hour candle from all market feeds"):
        fetch_latest_confirmed_candle(
            symbol="BTCUSDT",
            sleep_fn=sleeper,
            jitter_fn=_fixed_jitter,
            clock_fn=clock,
            now_dt_fn=_fixed_now_dt,
            total_deadline=50.0,
        )


# ---------------------------------------------------------------------------
# 7. Quant cycle failure preserves database, audit, and paper-portfolio state
# ---------------------------------------------------------------------------

def test_quant_cycle_preserves_db_on_fetch_failure(monkeypatch):
    """sync_hourly_quant_cycle returns (False, error, {}) when fetch fails.
    No database session is opened, no ticks/audits/orders are created."""
    from src.engine import sync
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select
    from src.database.models import Base, MarketTick, CouncilAudit, PaperPortfolio
    from src.database.connection import init_db as real_init_db

    # Isolated in-memory DB
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    real_init_db(engine)
    Session = sessionmaker(bind=engine)

    monkeypatch.setattr("src.database.connection.engine", engine)
    monkeypatch.setattr("src.database.connection.SessionLocal", Session)
    monkeypatch.setattr("src.database.connection.test_connection", lambda: (True, "SQLite 3.x", 1.0))

    # Make fetch fail with all providers exhausted
    def mock_get(url, **kwargs):
        return _make_response(500)

    monkeypatch.setattr(requests, "get", mock_get)

    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    # Monkeypatch fetch_latest_confirmed_candle to use our mocked deps
    original_fetch = sync.fetch_latest_confirmed_candle
    def patched_fetch(symbol="BTCUSDT"):
        return original_fetch(
            symbol=symbol,
            sleep_fn=sleeper,
            jitter_fn=_fixed_jitter,
            clock_fn=clock,
            now_dt_fn=_fixed_now_dt,
            total_deadline=50.0,
        )
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", patched_fetch)

    synced, message, summary = sync.sync_hourly_quant_cycle(force=False)

    assert synced is False
    assert "Market feed error" in message
    assert summary == {}

    # Verify zero database mutations
    session = Session()
    assert session.execute(select(MarketTick)).scalars().all() == []
    assert session.execute(select(CouncilAudit)).scalars().all() == []
    assert session.execute(select(PaperPortfolio)).scalars().all() == []
    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


# ---------------------------------------------------------------------------
# 8. Deadline exhaustion prevents another request or sleep
# ---------------------------------------------------------------------------

def test_deadline_exhaustion_prevents_further_attempts(monkeypatch):
    """When clock advances past deadline during retries, no further attempts are made."""
    # Clock starts at 1000.0, deadline at +2.0 = 1002.0
    clock = MockClock(start=1000.0, increment=0.001)
    sleeper = SleepRecorder(clock=clock)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        raise requests.exceptions.ConnectTimeout("timeout")

    monkeypatch.setattr(requests, "get", mock_get)

    # Use a very short deadline so retry budget runs out
    deadline_ts = clock() + 2.0  # deadline at ~1002.001

    # Override sleeper to advance clock past deadline on first sleep
    original_sleeper = sleeper
    def aggressive_sleep(seconds):
        original_sleeper(seconds)
        clock._t = deadline_ts + 1.0  # Jump past deadline

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines",
        feed_name="Binance Vision",
        deadline_ts=deadline_ts,
        sleep_fn=aggressive_sleep,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        max_attempts=5,  # Would do 5 but deadline prevents it
    )
    assert result is None
    assert call_count[0] == 1
    assert len(original_sleeper.calls) == 1


# ---------------------------------------------------------------------------
# 9. Malformed JSON follows bounded retry policy
# ---------------------------------------------------------------------------

def test_malformed_json_retried_within_budget(monkeypatch):
    """HTTP 200 with invalid JSON is retried up to max_attempts."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)
    klines = _make_binance_klines(now_dt=FIXED_NOW)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            # Return a response whose .json() raises JSONDecodeError
            return _make_response(200, body=None)  # body=None triggers JSONDecodeError
        return _make_response(200, body=klines)

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines",
        feed_name="Binance Vision",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
    )
    assert result is not None
    assert call_count[0] == 3
    assert len(sleeper.calls) == 2


# ---------------------------------------------------------------------------
# 10. Confirmed-candle filtering and minimum candle-count rule unchanged
# ---------------------------------------------------------------------------

def test_confirmed_candle_filtering_and_minimum_count(monkeypatch):
    """Binance returns 102 klines, but only 30 are confirmed closed.
    Should fall through to Coinbase since 30 < 35 minimum requirement."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    now_dt = FIXED_NOW
    now_ms = now_dt.timestamp() * 1000.0

    # Create 30 confirmed + 72 unconfirmed (forming) klines
    klines = []
    for i in range(30):
        open_time = now_ms - (31 - i) * 3600000
        close_time = open_time + 3600000 - 1  # close_time < now_ms → confirmed
        klines.append([
            open_time, "60000.0", "60500.0", "59800.0", "60200.0", "100.0",
            close_time, "6020000.0", 1500, "55.0", "3311000.0", "0"
        ])
    for i in range(72):
        # These candles have close_time in the future → unconfirmed
        open_time = now_ms + i * 3600000
        close_time = open_time + 3600000 - 1  # close_time > now_ms
        klines.append([
            open_time, "60000.0", "60500.0", "59800.0", "60200.0", "100.0",
            close_time, "6020000.0", 1500, "55.0", "3311000.0", "0"
        ])

    cb_candles = _make_coinbase_candles(100, now_dt=now_dt)

    call_count = {"binance": 0, "coinbase": 0}
    def mock_get(url, **kwargs):
        if "binance" in url:
            call_count["binance"] += 1
            return _make_response(200, body=klines)
        call_count["coinbase"] += 1
        return _make_response(200, body=cb_candles)

    monkeypatch.setattr(requests, "get", mock_get)

    result = fetch_latest_confirmed_candle(
        symbol="BTCUSDT",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
        total_deadline=50.0,
    )
    # Binance had only 30 confirmed < 35 → fell through to Coinbase
    assert result is not None
    assert call_count["binance"] == 1
    assert call_count["coinbase"] == 1
    assert result["number_of_trades"] is None  # Coinbase
    assert result["close"] == 60250.0
    assert len(result["candles"]) == 100


# ---------------------------------------------------------------------------
# 11. Binance 418 (IP auto-ban) with parseable Retry-After within cap
# ---------------------------------------------------------------------------

def test_binance_418_autoban_with_retry_after(monkeypatch):
    """Binance 418 with Retry-After: 5 → honors sleep, then succeeds."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)
    klines = _make_binance_klines(now_dt=FIXED_NOW)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _make_response(418, headers={"Retry-After": "5"})
        return _make_response(200, body=klines)

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines",
        feed_name="Binance Vision",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
    )
    assert result is not None
    assert call_count[0] == 2
    assert len(sleeper.calls) == 1
    # 5.0 + 0.25 jitter = 5.25
    assert abs(sleeper.calls[0] - 5.25) < 0.01


# ---------------------------------------------------------------------------
# 12. 429 with missing Retry-After header fails over promptly
# ---------------------------------------------------------------------------

def test_429_missing_retry_after_fails_fast(monkeypatch):
    """HTTP 429 without Retry-After header → immediate failover, no sleep."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        return _make_response(429, headers={})  # No Retry-After

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines",
        feed_name="Binance Vision",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
    )
    assert result is None
    assert call_count[0] == 1
    assert len(sleeper.calls) == 0


# ---------------------------------------------------------------------------
# 13. InvalidURL / MissingSchema fails fast with diagnostics
# ---------------------------------------------------------------------------

def test_invalid_url_fails_fast(monkeypatch):
    """InvalidURL raises → fail fast, 1 attempt, zero sleep."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        raise requests.exceptions.InvalidURL("Invalid URL")

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="not-a-url",
        feed_name="Test Feed",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
    )
    assert result is None
    assert call_count[0] == 1
    assert len(sleeper.calls) == 0


# ---------------------------------------------------------------------------
# 14. Non-list JSON payload (e.g. error dict) fails over
# ---------------------------------------------------------------------------

def test_non_list_json_payload_fails_over(monkeypatch):
    """HTTP 200 with dict payload → fails over, no retry."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    def mock_get(url, **kwargs):
        return _make_response(200, body={"error": "invalid"})

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://example.com/api",
        feed_name="Test Feed",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
    )
    assert result is None
    assert len(sleeper.calls) == 0


# ---------------------------------------------------------------------------
# 15. Full Binance success on first attempt (happy path, no retries)
# ---------------------------------------------------------------------------

def test_binance_first_attempt_success(monkeypatch):
    """Happy path: Binance returns 200 on first attempt → zero retries, zero sleeps."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)
    klines = _make_binance_klines(now_dt=FIXED_NOW)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        return _make_response(200, body=klines)

    monkeypatch.setattr(requests, "get", mock_get)

    result = fetch_latest_confirmed_candle(
        symbol="BTCUSDT",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
        total_deadline=50.0,
    )
    assert result is not None
    assert result["close"] == 60200.0
    assert result["number_of_trades"] == 1500
    assert len(result["candles"]) == 100
    assert call_count[0] == 1
    assert len(sleeper.calls) == 0


# ---------------------------------------------------------------------------
# 16. Retry-After exceeds overall deadline → fail over
# ---------------------------------------------------------------------------

def test_retry_after_exceeds_deadline_fails_over(monkeypatch):
    """Retry-After is within the per-header cap (15s) but exceeds remaining deadline."""
    # Clock at 1000.0, deadline at 1002.0 (only 2s budget)
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        return _make_response(429, headers={"Retry-After": "10"})

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines",
        feed_name="Binance Vision",
        deadline_ts=1002.0,
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
    )
    assert result is None
    assert call_count[0] == 1
    assert len(sleeper.calls) == 0


# ---------------------------------------------------------------------------
# 17. Exponential backoff values are correct
# ---------------------------------------------------------------------------

def test_exponential_backoff_values(monkeypatch):
    """Verify backoff sequence: 1.0*2^0 + 0.25 = 1.25, 1.0*2^1 + 0.25 = 2.25."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        return _make_response(503)

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines",
        feed_name="Binance Vision",
        max_attempts=3,
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
    )
    assert result is None
    assert call_count[0] == 3
    assert len(sleeper.calls) == 2
    assert abs(sleeper.calls[0] - 1.25) < 0.01  # base * 2^0 + jitter = 1.0 + 0.25
    assert abs(sleeper.calls[1] - 2.25) < 0.01  # base * 2^1 + jitter = 2.0 + 0.25


# ---------------------------------------------------------------------------
# 18. Fix 1: Unexpected programming errors propagate without being swallowed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prog_exc", [TypeError("bad type"), AttributeError("missing attr"), ZeroDivisionError("div0")])
def test_unexpected_programming_errors_propagate(monkeypatch, prog_exc):
    """Unexpected programming exceptions must propagate immediately without retry or None conversion."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    call_count = [0]
    def buggy_get(url, **kwargs):
        call_count[0] += 1
        raise prog_exc

    monkeypatch.setattr(requests, "get", buggy_get)

    with pytest.raises(type(prog_exc)):
        fetch_latest_confirmed_candle(
            symbol="BTCUSDT",
            sleep_fn=sleeper,
            jitter_fn=_fixed_jitter,
            clock_fn=clock,
            now_dt_fn=_fixed_now_dt,
        )

    assert call_count[0] == 1
    assert len(sleeper.calls) == 0


# ---------------------------------------------------------------------------
# 19. Fix 2: Retry-After boundary values equal to cap never sleep past cap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("retry_after_hdr", ["15", "15.0", "14.9"])
def test_retry_after_at_cap_boundary_clamps_jitter_to_cap(monkeypatch, retry_after_hdr):
    """When Retry-After + jitter would exceed EXCHANGE_MAX_RETRY_AFTER_CAP (15.0s),
    the scheduled delay is clamped to 15.0s and never sleeps past the cap."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)
    klines = _make_binance_klines(now_dt=FIXED_NOW)

    call_count = [0]
    def mock_get(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _make_response(429, headers={"Retry-After": retry_after_hdr})
        return _make_response(200, body=klines)

    monkeypatch.setattr(requests, "get", mock_get)

    result = _fetch_feed_with_retry(
        url="https://data-api.binance.vision/api/v3/klines",
        feed_name="Binance Vision",
        sleep_fn=sleeper,
        jitter_fn=lambda lo, hi: 0.45,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
    )
    assert result is not None
    assert call_count[0] == 2
    assert len(sleeper.calls) == 1
    assert sleeper.calls[0] == pytest.approx(EXCHANGE_MAX_RETRY_AFTER_CAP)
    assert sleeper.calls[0] <= EXCHANGE_MAX_RETRY_AFTER_CAP


# ---------------------------------------------------------------------------
# 20. Fix 3: Truncated rows, non-numeric fields, and non-finite values fall back to Coinbase
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_binance_payload",
    [
        [[1, 2, 3]],  # Truncated row (< 10 elements)
        _make_binance_klines(40, now_dt=FIXED_NOW)[:39] + [[1, 2]],  # Later row truncated
        [[1700000000000, "60000.0", "60500.0", "59800.0", "NaN", "100.0", 1700003599999, "0", 100, "50.0"]],
        [[1700000000000, "60000.0", "inf", "59800.0", "60200.0", "100.0", 1700003599999, "0", 100, "50.0"]],
        [[1700000000000, "60000.0", "60500.0", "-inf", "60200.0", "100.0", 1700003599999, "0", 100, "50.0"]],
        [[1700000000000, "not-a-float", "60500.0", "59800.0", "60200.0", "100.0", 1700003599999, "0", 100, "50.0"]],
        [[1700000000000, "60000.0", "60500.0", "59800.0", "60200.0", "100.0", 1700003599999, "0", "bad-int", "50.0"]],
    ],
)
def test_invalid_binance_payload_rows_fallback_cleanly_to_coinbase(monkeypatch, bad_binance_payload):
    """Invalid Binance row lengths, non-numeric fields, or non-finite values must not raise
    IndexError/ValueError; they must fail over cleanly to Coinbase."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)
    cb_candles = _make_coinbase_candles(100, now_dt=FIXED_NOW)

    call_count = {"binance": 0, "coinbase": 0}
    def mock_get(url, **kwargs):
        if "binance" in url:
            call_count["binance"] += 1
            return _make_response(200, body=bad_binance_payload)
        call_count["coinbase"] += 1
        return _make_response(200, body=cb_candles)

    monkeypatch.setattr(requests, "get", mock_get)

    result = fetch_latest_confirmed_candle(
        symbol="BTCUSDT",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
    )
    assert result is not None
    assert call_count["binance"] == 1
    assert call_count["coinbase"] == 1
    assert result["close"] == 60250.0
    assert result["number_of_trades"] is None


# ---------------------------------------------------------------------------
# 21. Fix 4: Per-provider budget prevents slow Binance retries from starving Coinbase
# ---------------------------------------------------------------------------

def test_binance_consuming_full_default_budget_does_not_starve_coinbase(monkeypatch):
    """Under the default 25.0s per-provider EXCHANGE_RETRY_TOTAL_DEADLINE, if Binance
    runs 3 slow read timeouts (3 * 7.0s = 21.0s) plus 2 inter-attempt waits
    (1.25s + 2.25s = 3.50s, totaling 24.50s on Binance), Coinbase still receives its own
    fresh 25.0s retry-scheduling budget and succeeds."""
    clock = MockClock(start=1000.0, increment=0.0)
    sleeper = SleepRecorder(clock=clock)
    cb_candles = _make_coinbase_candles(100, now_dt=FIXED_NOW)

    call_count = {"binance": 0, "coinbase": 0}
    def mock_get(url, **kwargs):
        if "binance" in url:
            call_count["binance"] += 1
            # Simulate each Binance attempt taking 7.0s (read timeout)
            clock._t += 7.0
            raise requests.exceptions.ReadTimeout("Binance read timed out after 7.0s")
        call_count["coinbase"] += 1
        clock._t += 0.2
        return _make_response(200, body=cb_candles)

    monkeypatch.setattr(requests, "get", mock_get)

    # Use default EXCHANGE_RETRY_TOTAL_DEADLINE (25.0s per provider)
    result = fetch_latest_confirmed_candle(
        symbol="BTCUSDT",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
    )
    # 3 Binance attempts with 2 inter-attempt sleeps (1.25s, 2.25s), followed by 1 Coinbase attempt
    assert call_count["binance"] == 3
    assert len(sleeper.calls) == 2
    assert sleeper.calls[0] == pytest.approx(1.25)
    assert sleeper.calls[1] == pytest.approx(2.25)
    assert call_count["coinbase"] == 1
    assert result["close"] == 60250.0
    assert len(result["candles"]) == 100


# ---------------------------------------------------------------------------
# 22. Fix A: Late in-flight primary response exceeding deadline is rejected & falls back
# ---------------------------------------------------------------------------

def test_in_flight_primary_request_exceeding_deadline_is_rejected_and_falls_back(monkeypatch):
    """If an in-flight Binance HTTP 200 request advances the injected clock beyond
    Binance's 25.0s scheduling deadline before returning, the late response is rejected
    (no second Binance attempt or sleep occurs) and Coinbase still gets its own budget."""
    clock = MockClock(start=1000.0, increment=0.0)
    sleeper = SleepRecorder(clock=clock)
    bn_klines = _make_binance_klines(102, now_dt=FIXED_NOW)
    cb_candles = _make_coinbase_candles(100, now_dt=FIXED_NOW)

    call_count = {"binance": 0, "coinbase": 0}
    def mock_get(url, **kwargs):
        if "binance" in url:
            call_count["binance"] += 1
            # Advance clock by 26.0s (> 25.0s provider deadline) during in-flight request
            clock._t += EXCHANGE_RETRY_TOTAL_DEADLINE + 1.0
            return _make_response(200, body=bn_klines)
        call_count["coinbase"] += 1
        clock._t += 0.5
        return _make_response(200, body=cb_candles)

    monkeypatch.setattr(requests, "get", mock_get)

    result = fetch_latest_confirmed_candle(
        symbol="BTCUSDT",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
    )
    # Late Binance 200 response was rejected; no retries/sleeps on Binance; Coinbase succeeded
    assert call_count["binance"] == 1
    assert len(sleeper.calls) == 0
    assert call_count["coinbase"] == 1
    assert result["close"] == 60250.0
    assert result["number_of_trades"] is None


# ---------------------------------------------------------------------------
# 23. Trade count validation: reject fractional, negative, non-finite & boolean;
#     accept valid integer, integral float, and integral string representations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_trade_count",
    [
        1.5,
        0.25,
        "1.5",
        "-1",
        -1,
        -10.0,
        "-0.5",
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        "NaN",
        "inf",
        "-inf",
        "",
        "bad-int",
    ],
)
def test_invalid_or_fractional_trade_counts_rejected_and_fallback_to_coinbase(
    monkeypatch, invalid_trade_count
):
    """Fractional, negative, non-finite, boolean, and malformed trade counts on Binance
    must be rejected by _validate_candle_payload and fall back cleanly to Coinbase."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)
    bad_klines = _make_binance_klines(40, now_dt=FIXED_NOW)
    bad_klines[-1] = list(bad_klines[-1])
    bad_klines[-1][8] = invalid_trade_count
    cb_candles = _make_coinbase_candles(100, now_dt=FIXED_NOW)

    call_count = {"binance": 0, "coinbase": 0}
    def mock_get(url, **kwargs):
        if "binance" in url:
            call_count["binance"] += 1
            return _make_response(200, body=bad_klines)
        call_count["coinbase"] += 1
        return _make_response(200, body=cb_candles)

    monkeypatch.setattr(requests, "get", mock_get)

    result = fetch_latest_confirmed_candle(
        symbol="BTCUSDT",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
    )
    assert call_count["binance"] == 1
    assert call_count["coinbase"] == 1
    assert result["close"] == 60250.0
    assert result["number_of_trades"] is None


@pytest.mark.parametrize(
    "valid_trade_count,expected_int",
    [
        (1500, 1500),
        (0, 0),
        (1500.0, 1500),
        (0.0, 0),
        ("1500", 1500),
        ("1500.0", 1500),
        ("0", 0),
    ],
)
def test_valid_integral_trade_counts_accepted_and_parsed_as_int(
    monkeypatch, valid_trade_count, expected_int
):
    """Valid integer, integral float, and integral numeric string trade counts
    must be accepted and returned as exact Python ints."""
    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)
    klines = _make_binance_klines(40, now_dt=FIXED_NOW)
    klines[-1] = list(klines[-1])
    klines[-1][8] = valid_trade_count

    monkeypatch.setattr(
        requests,
        "get",
        lambda url, **kwargs: _make_response(200, body=klines),
    )

    result = fetch_latest_confirmed_candle(
        symbol="BTCUSDT",
        sleep_fn=sleeper,
        jitter_fn=_fixed_jitter,
        clock_fn=clock,
        now_dt_fn=_fixed_now_dt,
    )
    assert result["number_of_trades"] == expected_int
    assert isinstance(result["number_of_trades"], int)
    assert not isinstance(result["number_of_trades"], bool)
    assert result["candles"][-1]["number_of_trades"] == expected_int
    assert isinstance(result["candles"][-1]["number_of_trades"], int)


# ---------------------------------------------------------------------------
# 24. Exhausted feeds in sync_hourly_quant_cycle return failure without opening DB session
# ---------------------------------------------------------------------------

def test_quant_cycle_feed_exhaustion_returns_failure_without_opening_db_session(monkeypatch):
    """When all feeds fail and fetch_latest_confirmed_candle raises MarketFeedExhaustedError,
    sync_hourly_quant_cycle catches MarketFeedExhaustedError and returns the fail-closed
    (False, 'Market feed error: ...', {}) tuple without opening get_db()."""
    from src.engine import sync

    clock = MockClock(start=1000.0)
    sleeper = SleepRecorder(clock=clock)

    monkeypatch.setattr(sync, "test_connection", lambda: (True, "PostgreSQL 18.6", 1.0))
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: _make_response(503))

    original_fetch = sync.fetch_latest_confirmed_candle
    monkeypatch.setattr(
        sync,
        "fetch_latest_confirmed_candle",
        lambda symbol="BTCUSDT": original_fetch(
            symbol=symbol,
            sleep_fn=sleeper,
            jitter_fn=_fixed_jitter,
            clock_fn=clock,
            now_dt_fn=_fixed_now_dt,
        ),
    )

    db_opened = [False]
    def forbidden_get_db(*args, **kwargs):
        db_opened[0] = True
        raise AssertionError("get_db() must not be opened when candle fetching fails")

    monkeypatch.setattr(sync, "get_db", forbidden_get_db)

    synced, message, summary = sync_hourly_quant_cycle(force=False)
    assert synced is False
    assert message == "Market feed error: Failed to fetch confirmed 1-hour candle from all market feeds."
    assert summary == {}
    assert db_opened[0] is False


# ---------------------------------------------------------------------------
# 25. Unexpected programming error in sync_hourly_quant_cycle is not swallowed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unexpected_exc",
    [
        TypeError("unexpected type error in feed path"),
        AttributeError("unexpected attribute error in feed path"),
        ZeroDivisionError("unexpected division by zero"),
        RuntimeError("unrelated runtime programming bug"),
    ],
)
def test_quant_cycle_unexpected_programming_error_is_not_swallowed(monkeypatch, unexpected_exc):
    """sync_hourly_quant_cycle catches only MarketFeedExhaustedError; unexpected
    programming exceptions (including plain RuntimeError) propagate rather than being
    disguised as routine market feed errors."""
    from src.engine import sync

    monkeypatch.setattr(sync, "test_connection", lambda: (True, "PostgreSQL 18.6", 1.0))
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: (_ for _ in ()).throw(unexpected_exc))

    db_opened = [False]
    def forbidden_get_db(*args, **kwargs):
        db_opened[0] = True
        raise AssertionError("get_db() must not be opened when an unexpected error occurs")

    monkeypatch.setattr(sync, "get_db", forbidden_get_db)

    with pytest.raises(type(unexpected_exc), match=str(unexpected_exc)):
        sync_hourly_quant_cycle(force=False)

    assert db_opened[0] is False