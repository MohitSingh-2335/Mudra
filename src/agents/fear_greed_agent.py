# src/agents/fear_greed_agent.py
"""
Alternative data extractor for Alternative.me Crypto Fear & Greed Index.
Pulls macro market sentiment scores and broadcasts daily values across hourly candles.
"""

from datetime import date
from typing import Optional, Union
import pandas as pd
import requests

FNG_API_URL = "https://api.alternative.me/fng/"
NEUTRAL_DEFAULT = 50.0  # Midpoint baseline when API is unreachable


def fetch_fear_greed_index(limit: int = 0) -> Optional[pd.DataFrame]:
    """
    Fetches historical Fear & Greed sentiment from Alternative.me REST API.

    Args:
        limit (int): Number of historical days to fetch (0 = full history).

    Returns:
        Optional[pd.DataFrame]: DataFrame with 'date', 'fng_value', 'fng_classification',
                                or None if request fails.
    """
    try:
        resp = requests.get(
            FNG_API_URL,
            params={"limit": limit, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.date
        df["fng_value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.rename(columns={"value_classification": "fng_classification"})
        return df[["date", "fng_value", "fng_classification"]].sort_values("date").reset_index(drop=True)

    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"[WARNING] fear_greed_agent failed ({e}) — falling back to None")
        return None


def get_fear_greed_for_date(
    target_date: Union[str, date],
    fng_df: Optional[pd.DataFrame] = None,
) -> float:
    """
    Single-value lookup for live/manual inference paths.

    Args:
        target_date: Target calendar date (string or date object).
        fng_df (Optional[pd.DataFrame]): Pre-fetched F&G DataFrame.

    Returns:
        float: Fear & Greed index score (0-100) or 50.0 neutral fallback.
    """
    if fng_df is None:
        fng_df = fetch_fear_greed_index(limit=10)

    if fng_df is None or fng_df.empty:
        return NEUTRAL_DEFAULT

    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()

    match = fng_df[fng_df["date"] == target_date]
    if not match.empty:
        return float(match["fng_value"].iloc[0])

    return NEUTRAL_DEFAULT


def merge_fear_greed(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Broadcasts daily Fear & Greed sentiment across all hourly candles on that date.
    Gracefully fails soft to NEUTRAL_DEFAULT (50.0) if external API is unreachable.

    Args:
        df (pd.DataFrame): Candlestick DataFrame with timestamp column.
        timestamp_col (str): Column name containing timestamps.

    Returns:
        pd.DataFrame: Merged DataFrame with 'fng_value' column.
    """
    fng_df = fetch_fear_greed_index()
    df = df.copy()
    df["_date"] = pd.to_datetime(df[timestamp_col]).dt.date

    if fng_df is None or fng_df.empty:
        df["fng_value"] = NEUTRAL_DEFAULT
        return df.drop(columns=["_date"])

    # Prevent duplicate column suffixes (_x, _y) if column already exists
    if "fng_value" in df.columns:
        df = df.drop(columns=["fng_value"])

    df = df.merge(fng_df[["date", "fng_value"]], left_on="_date", right_on="date", how="left")
    df["fng_value"] = df["fng_value"].fillna(NEUTRAL_DEFAULT)
    return df.drop(columns=["_date", "date"])


if __name__ == "__main__":
    fng = fetch_fear_greed_index(limit=5)
    print(fng if fng is not None else "[INFO] Fetch failed, falling back to NEUTRAL_DEFAULT (50.0)")