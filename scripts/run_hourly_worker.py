"""
Mudra 24/7 Hourly Headless Cron Worker.
Executes the full quant lifecycle: market ingestion, API telemetry,
delayed ground truth reconciliation, multi-agent council deliberation,
paper portfolio execution, and outbox synchronization.
"""
# PRE-LOAD ONNXRUNTIME FIRST to prevent Windows OpenMP (0xC0000005) thread collision
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.connection import get_db, init_db, test_connection
from src.database.models import MarketTick, ApiTelemetry
from src.engine.sync import sync_hourly_quant_cycle, fetch_latest_confirmed_candle

OUTBOX_FILE = ROOT_DIR / "data" / "sync_outbox.json"

# Institutional alias for backwards compatibility
fetch_latest_candle = fetch_latest_confirmed_candle

__all__ = ["run_worker", "flush_outbox", "log_telemetry", "fetch_latest_candle"]


def log_telemetry(session, service: str, endpoint: str, latency_ms: float, status_code: int, error_msg: Optional[str] = None) -> None:
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


def flush_outbox(session: Optional[Any] = None) -> int:
    """
    Flushes cached records from data/sync_outbox.json to the database.
    Durable and idempotent:
    - Never deletes the outbox file before the database commit succeeds.
    - Idempotently skips ticks that already exist in market_ticks (matched by UTC timestamp).
    - Preserves outbox file on parsing, insertion, flush, or database commit failure.
    - Safe against crashes between commit and file unlink (replay skips duplicates).
    """
    if not OUTBOX_FILE.exists():
        return 0

    try:
        content = OUTBOX_FILE.read_text(encoding="utf-8")
        if not content.strip():
            OUTBOX_FILE.unlink(missing_ok=True)
            return 0
        data = json.loads(content)
    except Exception as exc:
        print(f"[WARN] Failed to read/parse outbox file: {exc}")
        return 0

    if not isinstance(data, list) or not data:
        if not isinstance(data, list):
            print("[WARN] Outbox file content is not a list. Preserving file.")
        return 0

    def _apply_ticks(s) -> int:
        from sqlalchemy import select
        flushed_count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tick" and "payload" in item:
                payload = item["payload"]
                raw_ts = payload.get("timestamp")
                if not raw_ts:
                    continue
                if isinstance(raw_ts, str):
                    ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                else:
                    ts = raw_ts
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)

                existing_id = s.execute(
                    select(MarketTick.id).where(MarketTick.timestamp == ts)
                ).scalar_one_or_none()

                if existing_id is not None:
                    continue

                tick_payload = dict(payload)
                tick_payload["timestamp"] = ts
                tick = MarketTick(**tick_payload)
                s.add(tick)
                flushed_count += 1
        s.flush()
        return flushed_count

    if session is None:
        try:
            with get_db() as local_session:
                flushed_count = _apply_ticks(local_session)
            OUTBOX_FILE.unlink(missing_ok=True)
            if flushed_count > 0:
                print(f"[OUTBOX] Successfully flushed {flushed_count} cached items to Neon cloud.")
            return flushed_count
        except Exception as exc:
            print(f"[WARN] Failed to flush outbox: {exc}")
            return 0
    else:
        try:
            flushed_count = _apply_ticks(session)
            session.commit()
            OUTBOX_FILE.unlink(missing_ok=True)
            if flushed_count > 0:
                print(f"[OUTBOX] Successfully flushed {flushed_count} cached items to Neon cloud.")
            return flushed_count
        except Exception as exc:
            session.rollback()
            print(f"[WARN] Failed to flush outbox: {exc}")
            return 0


def run_worker(force: bool = False) -> bool:
    """Executes the complete hourly quantitative pipeline."""
    print(f"\n{'='*70}")
    print(f"Mudra Hourly Worker Starting at {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*70}")

    # 1. Initialize Tables in Neon Cloud
    try:
        init_db()
    except Exception as exc:
        print(f"[FATAL] Cannot initialize database schema: {exc}")
        return False
    
    # 2. Test Connection
    ok, ver, lat = test_connection()
    if not ok:
        print(f"[FATAL] Cannot reach Database: {ver}")
        return False
    print(f"[DB] Connected to {ver[:35]}... (Latency: {lat:.1f}ms)")

    # 3. Flush any pending offline queue items
    try:
        flush_outbox()
    except Exception as exc:
        print(f"[WARN] Outbox flush failed: {exc}")

    # 4. Ingest Candle, Reconcile T-1, Council Deliberation, Paper Order
    print(f"[QUANT] Executing unified hourly quant cycle (force={force})...")
    try:
        synced, message, summary = sync_hourly_quant_cycle(force=force)
    except Exception as exc:
        print(f"[FATAL] Hourly quant cycle crashed with unhandled exception: {exc}")
        return False

    print(f"[STATUS] Synced: {synced} | Message: {message}")

    if summary:
        if "total_equity_usd" in summary:
            print(f"[PORTFOLIO] Total Equity: ${summary.get('total_equity_usd', 0.0):,.2f} | Cash: ${summary.get('cash_balance', 0.0):,.2f} | BTC: {summary.get('btc_balance', 0.0):.6f}")
            print(f"[ALPHA] Strategy Return: {summary.get('cumulative_return_pct', 0.0):+.2f}% | Benchmark: ${summary.get('benchmark_btc_equity', 0.0):,.2f} | Alpha: ${summary.get('alpha_usd', 0.0):+,.2f} ({summary.get('alpha_pct', 0.0):+.2f}%)")
        elif "tick_id" in summary:
            print(f"[INFO] Existing Tick ID: #{summary['tick_id']}")
    print(f"{'='*70}\n")

    # Exit Status Contract:
    # - True (exit 0): New tick successfully synced OR candle was already recorded (idempotent no-op).
    # - False (exit 1): Genuine operational failure (database unreachable, market feeds failed, crash).
    if synced:
        return True
    if "already recorded" in message or (isinstance(summary, dict) and "tick_id" in summary):
        return True
    return False


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    success = run_worker(force=force_flag)
    sys.exit(0 if success else 1)
