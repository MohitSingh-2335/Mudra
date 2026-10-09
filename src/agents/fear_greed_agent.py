import requests
import pandas as pd

FNG_API_URL = "https://api.alternative.me/fng/"
NEUTRAL_DEFAULT = 50  # midpoint fallback when API is down


def fetch_fear_greed_index(limit=0):
    """limit=0 = full history. Returns None on any failure (caller handles it)."""
    try:
        resp = requests.get(FNG_API_URL, params={"limit": limit, "format": "json"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.date
        df["fng_value"] = pd.to_numeric(df["value"])
        df = df.rename(columns={"value_classification": "fng_classification"})
        return df[["date", "fng_value", "fng_classification"]].sort_values("date").reset_index(drop=True)
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"⚠️  fear_greed_agent failed ({e}) — returning None")
        return None


def get_fear_greed_for_date(target_date, fng_df=None):
    """Single-value lookup for live/manual inference paths."""
    if fng_df is None:
        fng_df = fetch_fear_greed_index()
    if fng_df is None:
        return NEUTRAL_DEFAULT
    match = fng_df[fng_df["date"] == target_date]
    return int(match["fng_value"].iloc[0]) if not match.empty else NEUTRAL_DEFAULT


def merge_fear_greed(df, timestamp_col="timestamp"):
    """Daily F&G value broadcast across all hourly rows on that date. Fails soft to NEUTRAL_DEFAULT."""
    fng_df = fetch_fear_greed_index()
    df = df.copy()
    df["_date"] = pd.to_datetime(df[timestamp_col]).dt.date

    if fng_df is None:
        df["fng_value"] = NEUTRAL_DEFAULT
        return df.drop(columns=["_date"])

    if "fng_value" in df.columns:
        df = df.drop(columns=["fng_value"])

    df = df.merge(fng_df[["date", "fng_value"]], left_on="_date", right_on="date", how="left")
    df["fng_value"] = df["fng_value"].fillna(NEUTRAL_DEFAULT)
    return df.drop(columns=["_date", "date"])


if __name__ == "__main__":
    fng = fetch_fear_greed_index(limit=5)
    print(fng if fng is not None else "fetch failed, would fall back to NEUTRAL_DEFAULT")