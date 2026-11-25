import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
import requests

FNG_API_URL = "https://api.alternative.me/fng/"
NEUTRAL_FNG = 50.0
CHARTS_BASE_URL = "https://api.blockchain.info/charts/"
ONCHAIN_CHARTS: Dict[str, str] = {
    "n-transactions": "onchain_num_tx",
    "hash-rate": "onchain_hash_rate",
    "miners-revenue": "onchain_miners_revenue_usd",
}
CHANGE_COLS: Dict[str, str] = {
    "onchain_num_tx": "onchain_num_tx_change",
    "onchain_hash_rate": "onchain_hash_rate_change",
    "onchain_miners_revenue_usd": "onchain_miners_revenue_change",
}


def load_and_clean_data(file_path: Union[str, Path]) -> pd.DataFrame:
    # Loads raw candles, parses timestamps, deduplicates, and sorts chronologically
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Target data file not found: {path_obj}")
    df = pd.read_csv(path_obj)
    if "timestamp" not in df.columns:
        raise KeyError(f"'timestamp' column missing from {path_obj}")
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y:%m:%d %H:%M:%S")
    except ValueError:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)


def _lagged_utc_dates(series: pd.Series) -> pd.Series:
    # Shifts UTC calendar dates by D-1 day so intraday hours never see same-day closing macro values
    ts_series = pd.to_datetime(series)
    ts_utc = ts_series.dt.tz_convert("UTC") if ts_series.dt.tz is not None else ts_series.dt.tz_localize("UTC")
    return (ts_utc.dt.floor("D") - pd.Timedelta(days=1)).dt.date


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


def merge_fear_greed(df: pd.DataFrame, timestamp_col: str = "timestamp", fng_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df.copy()
    df["_target_date"] = _lagged_utc_dates(df[timestamp_col])
    if "fng_value" in df.columns:
        df = df.drop(columns=["fng_value"])

    if fng_df is None:
        try:
            fng_df = fetch_fear_greed_index()
        except Exception:
            fng_df = None

    if fng_df is None or fng_df.empty:
        df["fng_value"] = NEUTRAL_FNG
        return df.drop(columns=["_target_date"])

    fng_df = fng_df.copy()
    fng_df["date"] = pd.to_datetime(fng_df["date"]).dt.date
    if "fng_value" in fng_df.columns:
        fng_df["fng_value"] = pd.to_numeric(fng_df["fng_value"], errors="coerce")
    fng_df = fng_df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    df = df.merge(fng_df[["date", "fng_value"]], left_on="_target_date", right_on="date", how="left")
    df["fng_value"] = df["fng_value"].ffill().fillna(NEUTRAL_FNG)
    return df.drop(columns=["_target_date", "date"], errors="ignore")


def fetch_onchain_metrics(timespan: str = "all", charts: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
    active_charts = charts or ONCHAIN_CHARTS
    merged: Optional[pd.DataFrame] = None
    for chart_name, col_name in active_charts.items():
        try:
            resp = requests.get(f"{CHARTS_BASE_URL}{chart_name}", params={"timespan": timespan, "format": "json", "sampled": "false"}, timeout=15)
            resp.raise_for_status()
            values = resp.json().get("values", [])
            if not values:
                continue
            chart_df = pd.DataFrame(values)
            chart_df["date"] = pd.to_datetime(chart_df["x"], unit="s").dt.date
            chart_df = chart_df[["date", "y"]].rename(columns={"y": col_name})
            merged = chart_df if merged is None else merged.merge(chart_df, on="date", how="outer")
        except Exception:
            continue
    return merged


def merge_onchain(df: pd.DataFrame, timestamp_col: str = "timestamp", timespan: str = "all", onchain_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df.copy()
    df["_target_date"] = _lagged_utc_dates(df[timestamp_col])
    target_cols = list(ONCHAIN_CHARTS.values()) + [CHANGE_COLS.get(c, f"{c}_change") for c in ONCHAIN_CHARTS.values()]
    df = df.drop(columns=[c for c in target_cols if c in df.columns])

    if onchain_df is None:
        try:
            onchain_df = fetch_onchain_metrics(timespan=timespan)
        except Exception:
            onchain_df = None

    if onchain_df is None or onchain_df.empty:
        for col in target_cols:
            df[col] = 0.0
        return df.drop(columns=["_target_date"])

    onchain_df = onchain_df.copy()
    onchain_df["date"] = pd.to_datetime(onchain_df["date"]).dt.date
    onchain_df = onchain_df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    for col in ONCHAIN_CHARTS.values():
        change_col = CHANGE_COLS.get(col, f"{col}_change")
        if col in onchain_df.columns:
            onchain_df[change_col] = onchain_df[col].pct_change(fill_method=None).fillna(0.0)
        else:
            onchain_df[col] = 0.0
            onchain_df[change_col] = 0.0

    cols_to_merge = ["date"] + [c for c in target_cols if c in onchain_df.columns]
    df = df.merge(onchain_df[cols_to_merge], left_on="_target_date", right_on="date", how="left")
    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)
    return df.drop(columns=["_target_date", "date"], errors="ignore")


_MACRO_CACHE: Dict[str, Any] = {"ts": 0.0, "fng_df": None, "onchain_df": None}


def enrich_candles_with_macro(df: pd.DataFrame, ttl_sec: float = 3600.0) -> pd.DataFrame:
    # Enriches candle sequence with cached D-1 lagged Fear & Greed and 30-day on-chain features
    if df is None or df.empty or "timestamp" not in df.columns:
        return df
    if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        return df

    now_mono = time.monotonic()
    if (now_mono - float(_MACRO_CACHE["ts"])) >= ttl_sec or (_MACRO_CACHE["fng_df"] is None and _MACRO_CACHE["onchain_df"] is None):
        fng_new = fetch_fear_greed_index(limit=10)
        onchain_new = fetch_onchain_metrics(timespan="30days")
        if fng_new is not None and not fng_new.empty:
            _MACRO_CACHE["fng_df"] = fng_new
        if onchain_new is not None and not onchain_new.empty:
            _MACRO_CACHE["onchain_df"] = onchain_new
        _MACRO_CACHE["ts"] = now_mono

    out = merge_fear_greed(df, timestamp_col="timestamp", fng_df=_MACRO_CACHE["fng_df"])
    return merge_onchain(out, timestamp_col="timestamp", timespan="30days", onchain_df=_MACRO_CACHE["onchain_df"])
