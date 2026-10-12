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


def merge_fear_greed(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    fng_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Broadcasts daily Fear & Greed sentiment across hourly candles preserving temporal causality.
    At any hourly timestamp T on calendar date D, the latest fully completed sentiment index is from date D-1.
    Forward-fills past observations across gaps, and fills pre-observation historical dates with NEUTRAL_DEFAULT (50.0).
    Never backward-fills future observations into earlier dates.

    Args:
        df (pd.DataFrame): Candlestick DataFrame with timestamp column.
        timestamp_col (str): Column name containing timestamps.
        fng_df (Optional[pd.DataFrame]): Optional pre-fetched Fear & Greed DataFrame with 'date' and 'fng_value'.
                                         If None, fetches historical index from Alternative.me API.

    Returns:
        pd.DataFrame: Merged DataFrame with causal 'fng_value' column.
    """
    df = df.copy()

    # Parse timestamps and determine the latest fully completed UTC calendar date (D - 1 day)
    ts_series = pd.to_datetime(df[timestamp_col])
    if ts_series.dt.tz is not None:
        ts_utc = ts_series.dt.tz_convert("UTC")
    else:
        ts_utc = ts_series.dt.tz_localize("UTC")
    df["_target_date"] = (ts_utc.dt.floor("D") - pd.Timedelta(days=1)).dt.date

    # Prevent duplicate column suffixes (_x, _y) if column already exists
    if "fng_value" in df.columns:
        df = df.drop(columns=["fng_value"])

    if fng_df is None:
        try:
            fng_df = fetch_fear_greed_index()
        except Exception as e:
            print(f"[WARNING] fear_greed_agent fetch exception ({e}) — falling back to neutral default.")
            fng_df = None

    if fng_df is None or fng_df.empty:
        df["fng_value"] = NEUTRAL_DEFAULT
        return df.drop(columns=["_target_date"])

    fng_df = fng_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(fng_df["date"]):
        fng_df["date"] = pd.to_datetime(fng_df["date"]).dt.date
    if "fng_value" in fng_df.columns:
        fng_df["fng_value"] = pd.to_numeric(fng_df["fng_value"], errors="coerce")

    # Sort chronologically, drop duplicates keeping last observation deterministically
    fng_df = fng_df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    df = df.merge(fng_df[["date", "fng_value"]], left_on="_target_date", right_on="date", how="left")

    # Forward-fill past observations forward, but fill dates before the first observation with NEUTRAL_DEFAULT
    df["fng_value"] = df["fng_value"].ffill().fillna(NEUTRAL_DEFAULT)

    return df.drop(columns=["_target_date", "date"], errors="ignore")


if __name__ == "__main__":
    fng = fetch_fear_greed_index(limit=5)
    print(fng if fng is not None else "[INFO] Fetch failed, falling back to NEUTRAL_DEFAULT (50.0)")