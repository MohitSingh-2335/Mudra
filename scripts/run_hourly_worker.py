import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import onnxruntime
except Exception:
    pass

import sys
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.database as db_conn
from src.database import ApiTelemetry, MarketTick, get_db, log_telemetry
from src.engine import sync_hourly_quant_cycle, fetch_latest_confirmed_candle

OUTBOX_FILE = ROOT_DIR / "data" / "sync_outbox.json"
fetch_latest_candle = fetch_latest_confirmed_candle
__all__ = ["run_worker", "flush_outbox", "log_telemetry", "fetch_latest_candle"]


def flush_outbox(session: Optional[Any] = None) -> int:
    # flushes cached ticks from local outbox file to database
    if not OUTBOX_FILE.exists():
        return 0
    try:
        content = OUTBOX_FILE.read_text(encoding="utf-8")
        if not content.strip():
            OUTBOX_FILE.unlink(missing_ok=True)
            return 0
        data = json.loads(content)
        if not isinstance(data, list) or not data:
            return 0
    except Exception:
        return 0

    def _apply(s) -> int:
        from sqlalchemy import select
        flushed = 0
        for item in data:
            if isinstance(item, dict) and item.get("type") == "tick" and "payload" in item:
                payload = dict(item["payload"])
                raw_ts = payload.get("timestamp")
                if not raw_ts:
                    continue
                ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")) if isinstance(raw_ts, str) else raw_ts
                ts = ts.replace(tzinfo=timezone.utc) if not ts.tzinfo else ts.astimezone(timezone.utc)
                if s.execute(select(MarketTick.id).where(MarketTick.timestamp == ts)).scalar_one_or_none() is not None:
                    continue
                payload["timestamp"] = ts
                s.add(MarketTick(**payload))
                flushed += 1
        s.flush()
        return flushed

    try:
        if session is None:
            with get_db() as s:
                count = _apply(s)
        else:
            count = _apply(session)
            session.commit()
        OUTBOX_FILE.unlink(missing_ok=True)
        return count
    except Exception:
        if session:
            session.rollback()
        return 0


def run_worker(force: bool = False) -> bool:
    print(f"[WORKER] Starting Mudra hourly worker at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} (force={force})", flush=True)
    
    print("[WORKER] Step 1/4: Testing database connection...", flush=True)
    ok, ver, lat = db_conn.test_connection()
    if not ok:
        print(f"[WORKER ERROR] Database unreachable: {ver}", flush=True)
        return False
    print(f"[WORKER] Database connected ({ver.splitlines()[0] if ver else 'Unknown'}) in {lat:.1f}ms", flush=True)

    print("[WORKER] Step 2/4: Verifying database schema...", flush=True)
    try:
        schema_info = db_conn.verify_db_schema()
        print(f"[WORKER] Database schema verified: {schema_info.get('tables', 0)} tables on {schema_info.get('dialect', 'unknown')}", flush=True)
    except Exception as schema_err:
        print(f"[WORKER WARNING] Schema verification issue: {schema_err}", flush=True)
        print("[WORKER] Attempting automatic table creation via init_db()...", flush=True)
        try:
            db_conn.init_db()
            schema_info = db_conn.verify_db_schema()
            print(f"[WORKER] Schema recovered and verified: {schema_info.get('tables', 0)} tables ready.", flush=True)
        except Exception as init_err:
            print(f"[WORKER ERROR] Database schema initialization failed: {init_err}", flush=True)
            traceback.print_exc()
            return False

    print("[WORKER] Step 3/4: Checking and flushing local outbox...", flush=True)
    try:
        flushed = flush_outbox()
        if flushed > 0:
            print(f"[WORKER] Flushed {flushed} queued records from outbox to database.", flush=True)
        else:
            print("[WORKER] Outbox empty or no pending records.", flush=True)
    except Exception as outbox_err:
        print(f"[WORKER WARNING] Outbox flush failed: {outbox_err}", flush=True)

    print("[WORKER] Step 4/4: Executing quantitative hourly cycle...", flush=True)
    try:
        synced, msg, summary = sync_hourly_quant_cycle(force=force)
        print(f"[WORKER] Quant cycle result: synced={synced}, status='{msg}'", flush=True)
        if summary:
            print(f"[WORKER] Summary telemetry: {summary}", flush=True)
        success = bool(synced or "already recorded" in msg or (isinstance(summary, dict) and "tick_id" in summary))
        if not success:
            print(f"[WORKER ERROR] Quant cycle did not complete successfully: {msg}", flush=True)
        return success
    except Exception as sync_err:
        print(f"[WORKER ERROR] Unhandled exception during quantitative cycle: {sync_err}", flush=True)
        traceback.print_exc()
        return False


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    success = run_worker(force=force_flag)
    print(f"[WORKER] Execution finished with status: {'SUCCESS' if success else 'FAILURE'}", flush=True)
    sys.exit(0 if success else 1)
