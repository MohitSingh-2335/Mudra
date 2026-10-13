"""
Database Connection Manager and Session Factory for Mudra.
Supports Neon Serverless PostgreSQL with SSL pooling and SQLite local fallback.
"""
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict
import tomllib
import logging

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import Base

# Path to local Streamlit secrets
SECRETS_FILE = Path(".streamlit/secrets.toml")


def get_database_url() -> str:
    """
    Resolves the active DATABASE_URL in order of precedence:
    1. OS environment variable (GitHub Actions / Cloud container)
    2. Streamlit st.secrets (if running inside Streamlit runtime)
    3. Direct read of .streamlit/secrets.toml via tomllib (CLI workers)
    4. Local SQLite file fallback
    """
    # 1. Environment variable
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return _normalize_db_url(db_url)

    # 2. Streamlit runtime secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "neondb" in st.secrets and "DATABASE_URL" in st.secrets["neondb"]:
                return _normalize_db_url(st.secrets["neondb"]["DATABASE_URL"])
            if "DATABASE_URL" in st.secrets:
                return _normalize_db_url(st.secrets["DATABASE_URL"])
    except Exception:
        pass

    # 3. Direct TOML parse for headless CLI / cron scripts
    if SECRETS_FILE.exists():
        try:
            data = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            if "neondb" in data and "DATABASE_URL" in data["neondb"]:
                return _normalize_db_url(data["neondb"]["DATABASE_URL"])
            if "DATABASE_URL" in data:
                return _normalize_db_url(data["DATABASE_URL"])
        except Exception:
            pass

    # 4. Local SQLite fallback
    fallback_path = Path("data/mudra_operational.db")
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{fallback_path.as_posix()}"


def _normalize_db_url(url: str) -> str:
    """
    Normalizes PostgreSQL URL for SQLAlchemy + psycopg2 compatibility.
    """
    clean_url = url.strip()
    if clean_url.startswith("postgres://"):
        return clean_url.replace("postgres://", "postgresql+psycopg2://", 1)
    if clean_url.startswith("postgresql://") and not clean_url.startswith("postgresql+"):
        return clean_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return clean_url


def create_db_engine(db_url: Optional[str] = None) -> Engine:
    """Creates a configured SQLAlchemy Engine."""
    url = db_url or get_database_url()
    
    if "sqlite" in url:
        eng = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        # Ensure schema tables exist on SQLite fallback
        try:
            Base.metadata.create_all(bind=eng)
        except Exception:
            pass
        return eng
    
    # PostgreSQL (Neon Serverless)
    return create_engine(
        url,
        pool_pre_ping=True,  # Test connection liveness before checkout
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,    # Recycle connections every 5 min
        connect_args={"connect_timeout": 10},
        echo=False,
    )


# Global Engine and SessionLocal
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager providing a transactional database session with auto-rollback.
    Usage:
        with get_db() as db:
            db.add(item)
    """
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


_TICK_ID_COLUMN_CACHE: Dict[int, bool] = {}


def has_tick_id_column(session: Session) -> bool:
    """
    Reliably verifies whether the underlying `paper_portfolio` table contains
    the `tick_id` column on disk, caching the result per engine instance.
    """
    try:
        bind = session.get_bind()
        bind_id = id(bind)
        if bind_id in _TICK_ID_COLUMN_CACHE:
            return _TICK_ID_COLUMN_CACHE[bind_id]

        from sqlalchemy import inspect
        inspector = inspect(bind)
        if inspector.has_table("paper_portfolio"):
            cols = {c["name"] for c in inspector.get_columns("paper_portfolio")}
            present = "tick_id" in cols
            _TICK_ID_COLUMN_CACHE[bind_id] = present
            return present
    except Exception as exc:
        logger.warning(f"[DB SCHEMA] Failed to inspect paper_portfolio table columns: {exc}")
    return False


def init_db(target_engine: Optional[Engine] = None) -> None:
    """
    Creates all 5 tables in the database if they do not exist and applies non-destructive schema evolutions.
    Fails closed with actionable diagnostics if required migrations fail.
    """
    target = target_engine or engine
    Base.metadata.create_all(bind=target)

    from sqlalchemy import inspect
    with target.connect() as conn:
        inspector = inspect(conn)

        # 1. Non-destructive migration for paper_portfolio.tick_id
        if inspector.has_table("paper_portfolio"):
            cols = {c["name"] for c in inspector.get_columns("paper_portfolio")}
            if "tick_id" not in cols:
                try:
                    if "postgresql" in target.dialect.name:
                        conn.execute(text("ALTER TABLE paper_portfolio ADD COLUMN IF NOT EXISTS tick_id INTEGER REFERENCES market_ticks(id);"))
                    elif "sqlite" in target.dialect.name:
                        conn.execute(text("ALTER TABLE paper_portfolio ADD COLUMN tick_id INTEGER REFERENCES market_ticks(id);"))
                    conn.commit()
                    logger.info("[DB MIGRATION] Added 'tick_id' column to paper_portfolio.")
                except Exception as exc:
                    logger.error(f"[DB MIGRATION CRITICAL] Failed to add 'tick_id' column to paper_portfolio: {exc}")
                    raise RuntimeError(f"Database migration failed: unable to add required column 'tick_id' to paper_portfolio: {exc}") from exc

            # Verify index on paper_portfolio.tick_id
            existing_indexes = {idx["name"] for idx in inspector.get_indexes("paper_portfolio")}
            if "uq_paper_portfolio_tick_id" not in existing_indexes:
                try:
                    conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_portfolio_tick_id ON paper_portfolio(tick_id) WHERE tick_id IS NOT NULL;"))
                    conn.commit()
                except Exception as exc:
                    logger.warning(f"[DB MIGRATION] Could not create partial unique index 'uq_paper_portfolio_tick_id': {exc}")

            # Update cache
            _TICK_ID_COLUMN_CACHE[id(target)] = True

        # 2. Safe duplicate-detection prior to creating unique index on market_ticks(timestamp)
        if inspector.has_table("market_ticks"):
            existing_mt_indexes = {idx["name"] for idx in inspector.get_indexes("market_ticks")}
            if "uq_market_ticks_timestamp" not in existing_mt_indexes:
                # Detect duplicate timestamps without modifying or deleting data
                try:
                    dup_rows = conn.execute(
                        text("SELECT timestamp, COUNT(*) as cnt FROM market_ticks GROUP BY timestamp HAVING COUNT(*) > 1")
                    ).fetchall()
                    if dup_rows:
                        dup_summary = [(str(r[0]), r[1]) for r in dup_rows[:5]]
                        logger.warning(
                            f"[DB MIGRATION] Cannot create unique index 'uq_market_ticks_timestamp': found {len(dup_rows)} duplicate timestamp group(s) "
                            f"(examples: {dup_summary}). Preserving all existing records; uniqueness constraint installation skipped."
                        )
                    else:
                        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_market_ticks_timestamp ON market_ticks(timestamp);"))
                        conn.commit()
                        logger.info("[DB MIGRATION] Verified 0 duplicate timestamps; created unique index 'uq_market_ticks_timestamp'.")
                except Exception as exc:
                    logger.warning(f"[DB MIGRATION] Failed during market_ticks timestamp uniqueness check or index creation: {exc}")


def test_connection() -> Tuple[bool, str, float]:
    """
    Tests database connectivity and measures round-trip latency.
    Dynamically handles PostgreSQL (SELECT version();) and SQLite (SELECT sqlite_version();).
    Returns: (is_success, server_version_or_error, latency_ms)
    """
    start_time = time.perf_counter()
    try:
        with engine.connect() as conn:
            is_sqlite = engine.dialect.name == "sqlite"
            query_str = "SELECT sqlite_version();" if is_sqlite else "SELECT version();"
            result = conn.execute(text(query_str)).fetchone()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            version_str = f"SQLite {result[0]}" if (is_sqlite and result) else (str(result[0]) if result else "Unknown")
            return True, version_str, latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return False, str(exc), latency_ms


if __name__ == '__main__':
    print("[INFO] Testing Mudra database connectivity and latency...")
    success, info, latency = test_connection()
    status = "SUCCESS" if success else "FAILED"
    print(f"[{status}] Connected: {success} | Version: {info} | Latency: {latency:.2f}ms")

