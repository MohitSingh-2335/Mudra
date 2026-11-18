from datetime import date
from typing import Optional, Union
import pandas as pd
import requests

FNG_API_URL = "https://api.alternative.me/fng/"
NEUTRAL_DEFAULT = 50.0


# Fetch historical Fear & Greed sentiment from Alternative.me REST API
def fetch_fear_greed_index(limit: int = 0) -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(FNG_API_URL, params={"limit": limit, "format": "json"}, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.date
        df["fng_value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "fng_value", "value_classification"]].rename(columns={"value_classification": "fng_classification"}).sort_values("date").reset_index(drop=True)
    except Exception:
        return None


# Single-value lookup for target calendar date
def get_fear_greed_for_date(target_date: Union[str, date], fng_df: Optional[pd.DataFrame] = None) -> float:
    if fng_df is None:
        fng_df = fetch_fear_greed_index(limit=10)
    if fng_df is None or fng_df.empty:
        return NEUTRAL_DEFAULT
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    match = fng_df[fng_df["date"] == target_date]
    return float(match["fng_value"].iloc[0]) if not match.empty else NEUTRAL_DEFAULT


# Broadcasts daily sentiment across hourly candles preserving temporal causality (D-1 day lag)
def merge_fear_greed(df: pd.DataFrame, timestamp_col: str = "timestamp", fng_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df.copy()
    ts_series = pd.to_datetime(df[timestamp_col])
    ts_utc = ts_series.dt.tz_convert("UTC") if ts_series.dt.tz is not None else ts_series.dt.tz_localize("UTC")
    df["_target_date"] = (ts_utc.dt.floor("D") - pd.Timedelta(days=1)).dt.date

    if "fng_value" in df.columns:
        df = df.drop(columns=["fng_value"])

    if fng_df is None:
        try:
            fng_df = fetch_fear_greed_index()
        except Exception:
            fng_df = None

    if fng_df is None or fng_df.empty:
        df["fng_value"] = NEUTRAL_DEFAULT
        return df.drop(columns=["_target_date"])

    fng_df = fng_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(fng_df["date"]):
        fng_df["date"] = pd.to_datetime(fng_df["date"]).dt.date
    if "fng_value" in fng_df.columns:
        fng_df["fng_value"] = pd.to_numeric(fng_df["fng_value"], errors="coerce")

    fng_df = fng_df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    df = df.merge(fng_df[["date", "fng_value"]], left_on="_target_date", right_on="date", how="left")
    df["fng_value"] = df["fng_value"].ffill().fillna(NEUTRAL_DEFAULT)
    return df.drop(columns=["_target_date", "date"], errors="ignore")
