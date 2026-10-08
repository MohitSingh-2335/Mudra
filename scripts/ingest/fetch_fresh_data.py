# fetch_fresh_data.py
#
# Pulls fresh BTCUSDT 1-hour historical candles directly from Binance and
# saves them in a format compatible with the existing pipeline, PLUS extra
# native Binance fields missing from older raw data files:
#   - quote_volume, number_of_trades
#   - taker_buy_base_volume, taker_buy_quote_volume, taker_buy_ratio
# These are not yet used by src/feature_engineering.py's create_features() —
# that function still only reads open/high/low/close/volume/timestamp. The
# new columns just ride along in the CSV until create_features() is updated
# to build features from them. This script only fetches and saves; it
# doesn't change what the models train on yet.
#
# Credentials: this script does NOT hardcode any API key/secret. It looks
# for them in one of two places, in this order:
#   1. Environment variables BINANCE_API_KEY / BINANCE_API_SECRET
#   2. .streamlit/secrets.toml, keys named api_key / api_secret

import os
import sys
import shutil
from datetime import datetime

import pandas as pd
from binance.client import Client

# --- Config ---
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR
YEARS_OF_HISTORY = 4          # <-- adjust here if you change your mind later
OUTPUT_PATH = "data/raw/BTCUSDT-1H.csv"
TIMESTAMP_FORMAT = "%Y:%m:%d %H:%M:%S"   # must match src/data_preprocessing.py


def load_binance_credentials():
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if api_key and api_secret:
        return api_key, api_secret

    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            import tomllib  # Python 3.11+
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)

            # Keys live under a [binance] section in this project's secrets.toml
            binance_section = secrets.get("binance", {})
            api_key = binance_section.get("api_key") or secrets.get("api_key")
            api_secret = binance_section.get("api_secret") or secrets.get("api_secret")

            if api_key and api_secret:
                return api_key, api_secret
        except Exception as e:
            print(f"⚠️  Could not read {secrets_path}: {e}")

    print(
        "❌ No Binance credentials found. Set BINANCE_API_KEY and "
        "BINANCE_API_SECRET as environment variables, or make sure "
        ".streamlit/secrets.toml has api_key / api_secret (top-level or "
        "under a [binance] section)."
    )
    sys.exit(1)


def fetch_klines(client, symbol, interval, years_back):
    start_str = f"{years_back} years ago UTC"
    print(f"Fetching {symbol} {interval} candles from {start_str} to now...")
    print("(This may take a minute or two — Binance paginates in 1000-candle chunks.)")

    klines = client.get_historical_klines(symbol, interval, start_str)

    if not klines:
        print("❌ No data returned. Check symbol/interval and your API key permissions.")
        sys.exit(1)

    print(f"✅ Fetched {len(klines)} raw candles.")
    return klines


def klines_to_dataframe(klines):
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

    # Extra signal beyond plain OHLCV — Binance gives us this for free, no
    # reason to throw it away:
    #   - number_of_trades: activity/liquidity proxy for that hour
    #   - taker_buy_ratio: fraction of that hour's volume that was aggressive
    #     BUYING (taker buys) vs. resting sell orders getting filled.
    #     >0.5 = buy-side pressure that hour, <0.5 = sell-side pressure.
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
    if os.path.exists(path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.replace(".csv", f"_backup_{stamp}.csv")
        shutil.copy2(path, backup_path)
        print(f"📦 Existing file backed up to: {backup_path}")


def main():
    api_key, api_secret = load_binance_credentials()
    client = Client(api_key, api_secret)

    klines = fetch_klines(client, SYMBOL, INTERVAL, YEARS_OF_HISTORY)
    df = klines_to_dataframe(klines)

    print(f"Date range fetched: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
    print(f"Total rows: {len(df)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    backup_existing_file(OUTPUT_PATH)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved fresh data to {OUTPUT_PATH}")
    print("\nNext steps:")
    print("  1. python scripts/prepare_data.py")
    print("  2. python -m src.model_training")


if __name__ == "__main__":
    main()