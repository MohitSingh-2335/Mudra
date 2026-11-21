"""Mudra quantitative trading and machine learning framework package."""
from typing import Any


def __getattr__(name: str) -> Any:
    # lazy-load database symbols on demand to avoid import-time coupling
    from . import database
    if hasattr(database, name):
        return getattr(database, name)
    raise AttributeError(f"module 'src' has no attribute '{name}'")


__all__ = [
    "Base",
    "MarketTick",
    "ApiTelemetry",
    "CouncilAudit",
    "Reconciliation",
    "PaperPortfolio",
    "ModelRegistry",
    "ModelPrediction",
    "HourlyPostmortem",
    "SchemaDriftError",
    "verify_db_schema",
    "get_db",
    "init_db",
    "test_connection",
    "engine",
    "SessionLocal",
    "has_tick_id_column",
]
