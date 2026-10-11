"""
Database Connection Manager and Session Factory for Mudra.
Supports Neon Serverless PostgreSQL with SSL pooling and SQLite local fallback.
"""
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Tuple, Optional
import tomllib

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
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
    
    # PostgreSQL (Neon Serverless)
    return create_engine(
        url,
        pool_pre_ping=True,  # Test connection liveness before checkout
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,    # Recycle connections every 5 min
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


def init_db(target_engine: Optional[Engine] = None) -> None:
    """Creates all 5 tables in the database if they do not exist."""
    target = target_engine or engine
    Base.metadata.create_all(bind=target)


def test_connection() -> Tuple[bool, str, float]:
    """
    Tests database connectivity and measures round-trip latency.
    Returns: (is_success, server_version_or_error, latency_ms)
    """
    start_time = time.perf_counter()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();")).fetchone()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            version_str = str(result[0]) if result else "Unknown"
            return True, version_str, latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return False, str(exc), latency_ms
