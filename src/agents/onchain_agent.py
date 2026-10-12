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
) -> pd.DataFrame:
    """
    Broadcasts daily on-chain metrics across all hourly rows for each date.
    Gracefully falls back to neutral zeros if Blockchain.com API is unreachable.

    Args:
        df (pd.DataFrame): Candlestick DataFrame with timestamp column.
        timestamp_col (str): Name of timestamp column.
        timespan (str): Lookback window for API fetch.

    Returns:
        pd.DataFrame: Merged DataFrame with on-chain metric columns.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df[timestamp_col]).dt.date

    # Prevent duplicate column suffixes (_x, _y) on repeated runs
    existing_cols = [col for col in CHARTS.values() if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)

    try:
        onchain_df = fetch_onchain_metrics(timespan=timespan)
    except Exception as e:
        print(f"[WARNING] onchain_agent fetch exception ({e}) — populating neutral zeros.")
        onchain_df = None

    if onchain_df is None or onchain_df.empty:
        for col in CHARTS.values():
            df[col] = 0.0
        return df.drop(columns=["_date"])

    df = df.merge(onchain_df, left_on="_date", right_on="date", how="left")
    for col in CHARTS.values():
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0.0)

    return df.drop(columns=["_date", "date"], errors="ignore")


if __name__ == "__main__":
    result = fetch_onchain_metrics(timespan="30days")
    print(result if result is not None else "[INFO] All on-chain charts failed.")