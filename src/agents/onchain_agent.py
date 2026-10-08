import requests
import pandas as pd

CHARTS_BASE_URL = "https://api.blockchain.info/charts/"

# chart-name -> output column name
CHARTS = {
    "n-transactions": "onchain_num_tx",
    "hash-rate": "onchain_hash_rate",
    "miners-revenue": "onchain_miners_revenue_usd",
}


def _fetch_single_chart(chart_name, timespan="all"):
    try:
        resp = requests.get(
            f"{CHARTS_BASE_URL}{chart_name}",
            params={"timespan": timespan, "format": "json", "sampled": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        values = resp.json()["values"]
        df = pd.DataFrame(values)
        df["date"] = pd.to_datetime(df["x"], unit="s").dt.date
        return df[["date", "y"]].rename(columns={"y": chart_name})
    except (requests.RequestException, KeyError, ValueError) as e:
        raise RuntimeError(f"⚠️  onchain_agent failed fetching '{chart_name}' ({e})")


def fetch_onchain_metrics(timespan="all", charts=None):
    """
    Fetches and merges all configured on-chain charts into one daily-indexed
    DataFrame. If any required chart fails, the process stops and reports the 
    failed charts.
    """
    charts = charts or CHARTS
    merged = None
    failed_charts = []
    for chart_name, col_name in charts.items():
        chart_df = _fetch_single_chart(chart_name, timespan=timespan)
        if chart_df is None:
            failed_charts.append(chart_name)
            continue
        chart_df = chart_df.rename(columns={chart_name: col_name})
        merged = chart_df if merged is None else merged.merge(chart_df, on="date", how="outer")
    if failed_charts:
        raise RuntimeError(f'On-chain data loading failed. These charts are failed to load: {failed_charts}')
    return merged


def merge_onchain(df, timestamp_col="timestamp", timespan="all"):
    """
    Daily on-chain values broadcast across all hourly rows on that date.
    If the fetch fails entirely, it will stop the process and raise the error.
    """
    onchain_df = fetch_onchain_metrics(timespan=timespan)

    df = df.copy()
    df["_date"] = pd.to_datetime(df[timestamp_col]).dt.date
    df = df.merge(onchain_df, left_on="_date", right_on="date", how="left")
    return df.drop(columns=["_date", "date"])


if __name__ == "__main__":
    result = fetch_onchain_metrics(timespan="30days")
    print(result if result is not None else "all charts failed")