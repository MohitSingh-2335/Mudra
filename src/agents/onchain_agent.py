# src/agents/onchain_agent.py
"""
Alternative data extractor for Blockchain.com Bitcoin on-chain metrics.
Retrieves network transaction counts, hash rate, and miner revenue,
broadcasting daily values across hourly candles.
"""

from typing import Dict, Optional
import pandas as pd
import requests

CHARTS_BASE_URL = "https://api.blockchain.info/charts/"

# Mapping: Blockchain.com chart identifier -> Mudra DataFrame column name
CHARTS: Dict[str, str] = {
    "n-transactions": "onchain_num_tx",
    "hash-rate": "onchain_hash_rate",
    "miners-revenue": "onchain_miners_revenue_usd",
}

# Mapping: On-chain level column -> Model contract daily change feature name
CHANGE_COLS: Dict[str, str] = {
    "onchain_num_tx": "onchain_num_tx_change",
    "onchain_hash_rate": "onchain_hash_rate_change",
    "onchain_miners_revenue_usd": "onchain_miners_revenue_change",
}


def _fetch_single_chart(chart_name: str, timespan: str = "all") -> Optional[pd.DataFrame]:
    """
    Fetches historical data points for a single Blockchain.com chart.

    Args:
        chart_name (str): Chart endpoint slug (e.g., 'hash-rate').
        timespan (str): Lookback window (e.g., 'all', '30days').

    Returns:
        Optional[pd.DataFrame]: DataFrame with 'date' and chart value column, or None on failure.
    """
    try:
        resp = requests.get(
            f"{CHARTS_BASE_URL}{chart_name}",
            params={"timespan": timespan, "format": "json", "sampled": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        values = resp.json().get("values", [])
        if not values:
            return None

        df = pd.DataFrame(values)
        df["date"] = pd.to_datetime(df["x"], unit="s").dt.date
        return df[["date", "y"]].rename(columns={"y": chart_name})

    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"[WARNING] onchain_agent failed fetching '{chart_name}': {e}")
        return None


def fetch_onchain_metrics(
    timespan: str = "all",
    charts: Optional[Dict[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetches and merges all configured on-chain charts into one daily-indexed DataFrame.

    Args:
        timespan (str): Lookback window.
        charts (Optional[Dict[str, str]]): Dictionary mapping chart slugs to column names.

    Returns:
        Optional[pd.DataFrame]: Merged on-chain DataFrame with 'date' column, or None on failure.
    """
    active_charts = charts or CHARTS
    merged: Optional[pd.DataFrame] = None
    failed_charts = []

    for chart_name, col_name in active_charts.items():
        chart_df = _fetch_single_chart(chart_name, timespan=timespan)
        if chart_df is None:
            failed_charts.append(chart_name)
            continue
        chart_df = chart_df.rename(columns={chart_name: col_name})
        merged = chart_df if merged is None else merged.merge(chart_df, on="date", how="outer")

    if failed_charts:
        print(f"[WARNING] On-chain metrics partially incomplete. Missing charts: {failed_charts}")

    return merged


def merge_onchain(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    timespan: str = "all",
    onchain_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Broadcasts completed daily on-chain metrics across hourly candles preserving temporal causality.
    At any hourly timestamp T on calendar date D, the latest fully completed daily aggregate is from date D-1.
    Computes day-over-day changes on daily source data before broadcasting to prevent zero-repetition artifacts.
    Eliminates backward-filling (bfill) of future metrics into historical candles.

    Args:
        df (pd.DataFrame): Candlestick DataFrame with timestamp column.
        timestamp_col (str): Name of timestamp column.
        timespan (str): Lookback window for API fetch.
        onchain_df (Optional[pd.DataFrame]): Optional pre-fetched on-chain metrics DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame with causal on-chain metrics and day-over-day change features.
    """
    df = df.copy()

    # Parse timestamps and determine the latest fully completed UTC calendar date (D - 1 day)
    ts_series = pd.to_datetime(df[timestamp_col])
    if ts_series.dt.tz is not None:
        ts_utc = ts_series.dt.tz_convert("UTC")
    else:
        ts_utc = ts_series.dt.tz_localize("UTC")
    df["_target_date"] = (ts_utc.dt.floor("D") - pd.Timedelta(days=1)).dt.date

    # Define all target on-chain columns (levels and day-over-day changes)
    target_cols = list(CHARTS.values()) + [CHANGE_COLS.get(c, f"{c}_change") for c in CHARTS.values()]

    # Prevent duplicate column suffixes (_x, _y) on repeated runs
    existing_cols = [col for col in target_cols if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)

    if onchain_df is None:
        try:
            onchain_df = fetch_onchain_metrics(timespan=timespan)
        except Exception as e:
            print(f"[WARNING] onchain_agent fetch exception ({e}) — populating neutral zeros.")
            onchain_df = None

    if onchain_df is None or onchain_df.empty:
        for col in target_cols:
            df[col] = 0.0
        return df.drop(columns=["_target_date"])

    onchain_df = onchain_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(onchain_df["date"]):
        onchain_df["date"] = pd.to_datetime(onchain_df["date"]).dt.date
    onchain_df = onchain_df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Compute day-over-day percentage changes on daily level prior to merge
    for col in CHARTS.values():
        change_col = CHANGE_COLS.get(col, f"{col}_change")
        if col in onchain_df.columns:
            onchain_df[change_col] = onchain_df[col].pct_change(fill_method=None).fillna(0.0)
        else:
            onchain_df[col] = 0.0
            onchain_df[change_col] = 0.0

    cols_to_merge = ["date"] + target_cols
    available_merge_cols = [c for c in cols_to_merge if c in onchain_df.columns]
    df = df.merge(onchain_df[available_merge_cols], left_on="_target_date", right_on="date", how="left")

    for col in target_cols:
        if col in df.columns:
            # Forward-fill past available knowledge forward, but strictly never backward-fill future data
            df[col] = df[col].ffill().fillna(0.0)

    return df.drop(columns=["_target_date", "date"], errors="ignore")


if __name__ == "__main__":
    result = fetch_onchain_metrics(timespan="30days")
    print(result if result is not None else "[INFO] All on-chain charts failed.")