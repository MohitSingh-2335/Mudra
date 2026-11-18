from typing import Dict, Optional
import pandas as pd
import requests

CHARTS_BASE_URL = "https://api.blockchain.info/charts/"

CHARTS: Dict[str, str] = {
    "n-transactions": "onchain_num_tx",
    "hash-rate": "onchain_hash_rate",
    "miners-revenue": "onchain_miners_revenue_usd",
}

CHANGE_COLS: Dict[str, str] = {
    "onchain_num_tx": "onchain_num_tx_change",
    "onchain_hash_rate": "onchain_hash_rate_change",
    "onchain_miners_revenue_usd": "onchain_miners_revenue_change",
}


# Fetch historical data points for a single Blockchain.com chart
def _fetch_single_chart(chart_name: str, timespan: str = "all") -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(f"{CHARTS_BASE_URL}{chart_name}", params={"timespan": timespan, "format": "json", "sampled": "false"}, timeout=15)
        resp.raise_for_status()
        values = resp.json().get("values", [])
        if not values:
            return None
        df = pd.DataFrame(values)
        df["date"] = pd.to_datetime(df["x"], unit="s").dt.date
        return df[["date", "y"]].rename(columns={"y": chart_name})
    except Exception:
        return None


# Fetch and merge on-chain daily charts into a single DataFrame
def fetch_onchain_metrics(timespan: str = "all", charts: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
    active_charts = charts or CHARTS
    merged: Optional[pd.DataFrame] = None
    for chart_name, col_name in active_charts.items():
        chart_df = _fetch_single_chart(chart_name, timespan=timespan)
        if chart_df is None:
            continue
        chart_df = chart_df.rename(columns={chart_name: col_name})
        merged = chart_df if merged is None else merged.merge(chart_df, on="date", how="outer")
    return merged


# Broadcasts completed daily on-chain metrics across hourly candles preserving temporal causality (D-1 day lag)
def merge_onchain(df: pd.DataFrame, timestamp_col: str = "timestamp", timespan: str = "all", onchain_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df.copy()
    ts_series = pd.to_datetime(df[timestamp_col])
    ts_utc = ts_series.dt.tz_convert("UTC") if ts_series.dt.tz is not None else ts_series.dt.tz_localize("UTC")
    df["_target_date"] = (ts_utc.dt.floor("D") - pd.Timedelta(days=1)).dt.date

    target_cols = list(CHARTS.values()) + [CHANGE_COLS.get(c, f"{c}_change") for c in CHARTS.values()]
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
    if not pd.api.types.is_datetime64_any_dtype(onchain_df["date"]):
        onchain_df["date"] = pd.to_datetime(onchain_df["date"]).dt.date
    onchain_df = onchain_df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    for col in CHARTS.values():
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
