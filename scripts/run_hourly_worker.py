import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.database as db_conn
from src.database import ApiTelemetry, MarketTick, get_db
from src.engine import sync_hourly_quant_cycle, fetch_latest_confirmed_candle

OUTBOX_FILE = ROOT_DIR / "data" / "sync_outbox.json"
fetch_latest_candle = fetch_latest_confirmed_candle
__all__ = ["run_worker", "flush_outbox", "log_telemetry", "fetch_latest_candle"]


def log_telemetry(session, service: str, endpoint: str, latency_ms: float, status_code: int, error_msg: Optional[str] = None) -> None:
    # logs external api call latency and status code
    try:
        session.add(ApiTelemetry(
            timestamp=datetime.now(timezone.utc), service_name=service, endpoint=endpoint,
            latency_ms=latency_ms, status_code=status_code, error_message=error_msg,
        ))
        session.flush()
    except Exception:
        pass


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
    # performs connection check, outbox flush, and quantitative sync cycle
    ok, _, _ = db_conn.test_connection()
    if not ok:
        return False
    try:
        db_conn.verify_db_schema()
        flush_outbox()
        synced, msg, summary = sync_hourly_quant_cycle(force=force)
        return bool(synced or "already recorded" in msg or (isinstance(summary, dict) and "tick_id" in summary))
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(0 if run_worker(force="--force" in sys.argv) else 1)
