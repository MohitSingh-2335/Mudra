import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.database as db_conn
from src.engine import sync_hourly_quant_cycle

__all__ = ["run_worker"]


def run_worker(force: bool = False, require_postgres: bool = False) -> bool:
    """Runs one hourly ingestion, multi-model inference, reconciliation, and audit cycle."""
    print(f"[WORKER] Starting Mudra hourly worker at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} (force={force})", flush=True)

    print("[WORKER] Step 1/3: Testing database connection...", flush=True)
    ok, ver, lat = db_conn.test_connection()
    if not ok:
        print(f"[WORKER ERROR] Database unreachable: {ver}", flush=True)
        return False
    # Fail closed in scheduled CI runs instead of writing silently to ephemeral SQLite
    if require_postgres and db_conn.engine.dialect.name != "postgresql":
        print(f"[WORKER ERROR] PostgreSQL required, connected to {db_conn.engine.dialect.name}.", flush=True)
        return False
    print(f"[WORKER] Database connected ({ver.splitlines()[0] if ver else 'Unknown'}) in {lat:.1f}ms", flush=True)

    print("[WORKER] Step 2/3: Verifying database schema...", flush=True)
    try:
        schema_info = db_conn.verify_db_schema()
        print(f"[WORKER] Database schema verified: {schema_info.get('tables', 0)} tables on {schema_info.get('dialect', 'unknown')}", flush=True)
    except Exception as schema_err:
        print(f"[WORKER WARNING] Schema verification issue: {schema_err}", flush=True)
        try:
            db_conn.init_db()
            schema_info = db_conn.verify_db_schema()
            print(f"[WORKER] Schema recovered: {schema_info.get('tables', 0)} tables ready.", flush=True)
        except Exception as init_err:
            print(f"[WORKER ERROR] Database schema initialization failed: {init_err}", flush=True)
            traceback.print_exc()
            return False

    print("[WORKER] Step 3/3: Executing quantitative hourly cycle...", flush=True)
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
    pg_flag = "--require-postgres" in sys.argv
    success = run_worker(force=force_flag, require_postgres=pg_flag)
    print(f"[WORKER] Execution finished with status: {'SUCCESS' if success else 'FAILURE'}", flush=True)
    sys.exit(0 if success else 1)
