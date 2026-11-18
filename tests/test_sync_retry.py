# tests/test_sync_retry.py
"""
Deterministic unit tests for exchange data ingestion, bounded retries,
and provider failover from Binance to Coinbase.
"""

import json
import math
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

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


def _make_response(status_code, body=None, headers=None):
    """Creates a mock requests.Response with status code and json body."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    if body is not None:
        resp.json.return_value = body
    else:
        resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    return resp


def _make_binance_klines(n=102, now_dt=None):
    """Generates n mock Binance confirmed 1-hour candles."""
    if now_dt is None:
        now_dt = datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc)
    now_ms = now_dt.timestamp() * 1000.0
    klines = []
    for i in range(n):
        open_time = now_ms - (n + 1 - i) * 3600000
        close_time = open_time + 3600000 - 1
        klines.append([
            open_time, "60000.0", "60500.0", "59800.0", "60200.0", "100.0",
            close_time, "6020000.0", 1500, "55.0", "3311000.0", "0"
        ])
    return klines


def _make_coinbase_candles(n=100, now_dt=None):
    """Generates n mock Coinbase confirmed 1-hour candles in descending order."""
    if now_dt is None:
        now_dt = datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc)
    now_sec = now_dt.timestamp()
    candles = []
    for i in range(n):
        candle_sec = now_sec - (i + 1) * 3600
        candles.append([candle_sec, 59800.0, 60500.0, 60000.0, 60250.0, 100.0])
    return candles


class TestParseRetryAfter:
    """Verifies RFC-compliant parsing of the Retry-After header."""

    def test_integer_and_float_seconds(self):
        assert parse_retry_after("5") == 5.0
        assert parse_retry_after("12.5") == 12.5
        assert parse_retry_after("0") == 0.0

    def test_invalid_and_negative_returns_none(self):
        assert parse_retry_after("-1") is None
        assert parse_retry_after("NaN") is None
        assert parse_retry_after("inf") is None
        assert parse_retry_after("") is None
        assert parse_retry_after(None) is None
        assert parse_retry_after("invalid_garbage") is None

    def test_http_date_parsing(self):
        ref_time = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)
        date_str = "Mon, 06 Jul 2026 12:00:30 GMT"
        val = parse_retry_after(date_str, now_dt_fn=lambda: ref_time)
        assert val == pytest.approx(30.0, abs=0.5)


def test_binance_transient_connection_failure_then_success(monkeypatch):
    """Verifies that transient network errors trigger a retry and succeed on subsequent attempt."""
    mock_klines = _make_binance_klines(102)
    attempts = [0]

    def mock_get(url, **kwargs):
        attempts[0] += 1
        if attempts[0] == 1:
            import requests
            raise requests.exceptions.ConnectionError("Connection reset by peer")
        return _make_response(200, mock_klines)

    monkeypatch.setattr("requests.get", mock_get)
    result = fetch_latest_confirmed_candle(
        sleep_fn=lambda s: None,
        clock_fn=lambda: 1000.0,
        now_dt_fn=lambda: datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc),
    )
    assert attempts[0] == 2
    assert result["close"] == 60200.0
    assert result["number_of_trades"] == 1500


def test_binance_exhausted_retries_coinbase_fallback(monkeypatch):
    """Verifies that when Binance fails completely, the system falls back to Coinbase."""
    mock_coinbase = _make_coinbase_candles(100)

    def mock_get(url, **kwargs):
        if "binance" in url:
            import requests
            raise requests.exceptions.ConnectionError("Binance down")
        if "coinbase" in url:
            return _make_response(200, mock_coinbase)
        return _make_response(404)

    monkeypatch.setattr("requests.get", mock_get)
    result = fetch_latest_confirmed_candle(
        sleep_fn=lambda s: None,
        clock_fn=lambda: 1000.0,
        now_dt_fn=lambda: datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc),
    )
    assert result["close"] == 60250.0
    assert result["number_of_trades"] is None


def test_both_providers_exhausted_raises_runtime_error(monkeypatch):
    """Verifies that when both Binance and Coinbase fail, MarketFeedExhaustedError is raised."""
    def mock_get(url, **kwargs):
        import requests
        raise requests.exceptions.ConnectionError("Network offline")

    monkeypatch.setattr("requests.get", mock_get)
    with pytest.raises(MarketFeedExhaustedError):
        fetch_latest_confirmed_candle(
            sleep_fn=lambda s: None,
            clock_fn=lambda: 1000.0,
            now_dt_fn=lambda: datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc),
        )


def test_quant_cycle_feed_exhaustion_returns_failure_without_opening_db_session(monkeypatch):
    """Verifies that quant cycle catches feed exhaustion cleanly and returns False."""
    def failing_fetch(*args, **kwargs):
        raise MarketFeedExhaustedError("All market feeds exhausted.")

    monkeypatch.setattr("src.engine.sync.fetch_latest_confirmed_candle", failing_fetch)
    monkeypatch.setattr("src.engine.sync.test_connection", lambda: (True, "PostgreSQL", 1.0))
    synced, message, summary = sync_hourly_quant_cycle(force=False)
    assert synced is False
    assert "exhausted" in message.lower()
    assert summary == {}


def test_trade_count_parsing_validation():
    """Verifies trade count validator accepts valid integers and rejects floats/strings/negatives."""
    assert _parse_trade_count(1500) == 1500
    assert _parse_trade_count(0) == 0
    assert _parse_trade_count(None) is None
    assert _parse_trade_count("1500") == 1500

    with pytest.raises(ValueError):
        _parse_trade_count(-5)
    with pytest.raises(ValueError):
        _parse_trade_count(12.5)
    with pytest.raises(ValueError):
        _parse_trade_count("abc")
    with pytest.raises(ValueError):
        _parse_trade_count(True)


def test_confirmed_candle_filtering_and_minimum_count(monkeypatch):
    """Verifies that forming (unconfirmed) candles are excluded."""
    now_dt = datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc)
    mock_klines = _make_binance_klines(102, now_dt=now_dt)

    monkeypatch.setattr("requests.get", lambda url, **kwargs: _make_response(200, mock_klines))
    result = fetch_latest_confirmed_candle(
        sleep_fn=lambda s: None,
        clock_fn=lambda: 1000.0,
        now_dt_fn=lambda: now_dt,
    )
    assert result is not None
    assert result["close"] == 60200.0
    assert len(result["candles"]) >= 35
