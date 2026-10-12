# scripts/ingest/fetch_fresh_data.py

# Pulls fresh BTCUSDT 1-hour historical candles from Binance and saves them
# to the raw data path defined in config.py, including market microstructure
# fields (quote volume, trade counts, and taker buy volume ratio).

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from binance.client import Client

from config import BTCUSDT_1H_CSV

# --- Configuration ---
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR
YEARS_OF_HISTORY = 4
OUTPUT_PATH = BTCUSDT_1H_CSV
TIMESTAMP_FORMAT = "%Y:%m:%d %H:%M:%S"


def load_binance_credentials():
    """
    Attempts to resolve Binance API credentials from environment variables or .streamlit/secrets.toml.
    If credentials are not found, returns (None, None) to allow unauthenticated public market data access.
    """
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if api_key and api_secret:
        return api_key, api_secret

    secrets_path = os.path.join(ROOT_DIR, ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            import tomllib  # Python 3.11+
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)

            binance_section = secrets.get("binance", {})
            api_key = binance_section.get("api_key") or secrets.get("api_key")
            api_secret = binance_section.get("api_secret") or secrets.get("api_secret")

            if api_key and api_secret:
                return api_key, api_secret
        except Exception as e:
            print(f"[WARNING] Could not read {secrets_path}: {e}")

    print("[INFO] No Binance credentials found. Proceeding with unauthenticated public client.")
    return None, None


def fetch_klines(client, symbol, interval, years_back):
    """
    Paginates historical candles from Binance for the specified lookback window.
    """
    start_str = f"{years_back} years ago UTC"
    print(f"[INFO] Fetching {symbol} {interval} candles from {start_str} to now...")
    print("[INFO] Binance paginates in 1,000-candle chunks. This may take 30-60 seconds.")

    klines = client.get_historical_klines(symbol, interval, start_str)

    if not klines:
        print("[ERROR] No data returned from Binance. Check network or symbols.")
        sys.exit(1)

    print(f"[SUCCESS] Fetched {len(klines)} raw candles.")
    return klines


def klines_to_dataframe(klines):
    """
    Converts raw Binance klines to a structured DataFrame with order-flow microstructure metrics.
    """
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms").dt.strftime(TIMESTAMP_FORMAT)
    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base", "taker_buy_quote",
    ]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)

    # Calculate taker buy volume ratio (microstructure aggressive order flow pressure)
    df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, pd.NA)

    df = df.rename(columns={
        "quote_asset_volume": "quote_volume",
        "taker_buy_base": "taker_buy_base_volume",
        "taker_buy_quote": "taker_buy_quote_volume",
    })

    df = df[[
        "timestamp", "open", "high", "low", "close", "volume",
        "quote_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "taker_buy_ratio",
    ]]
    return df


def backup_existing_file(path):
    """
    Creates a timestamped backup copy before overwriting existing raw data.
    """
    if os.path.exists(path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.replace(".csv", f"_backup_{stamp}.csv")
        shutil.copy2(path, backup_path)
        print(f"[BACKUP] Existing file backed up to: {backup_path}")


def main():
    api_key, api_secret = load_binance_credentials()
    if api_key and api_secret:
        client = Client(api_key, api_secret)
    else:
        client = Client()

    klines = fetch_klines(client, SYMBOL, INTERVAL, YEARS_OF_HISTORY)
    df = klines_to_dataframe(klines)

    print(f"[INFO] Date range fetched: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print(f"[INFO] Total rows: {len(df)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    backup_existing_file(OUTPUT_PATH)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Saved fresh data to {OUTPUT_PATH}")
    print("\nNext steps:")
    print("  1. python scripts/prepare_data.py")
    print("  2. python -m src.model_training")


if __name__ == "__main__":
    main()