import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd
from binance.client import Client

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import BTCUSDT_1H_CSV

SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR
YEARS_OF_HISTORY = 4
OUTPUT_PATH = BTCUSDT_1H_CSV
TIMESTAMP_FORMAT = "%Y:%m:%d %H:%M:%S"


def load_binance_credentials():
    # Load optional Binance credentials or fall back to public client
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if api_key and api_secret:
        return api_key, api_secret
    secrets_path = os.path.join(ROOT_DIR, ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            import tomllib
            with open(secrets_path, "rb") as f:
                sec = tomllib.load(f)
            b_sec = sec.get("binance", {})
            return (b_sec.get("api_key") or sec.get("api_key")), (b_sec.get("api_secret") or sec.get("api_secret"))
        except Exception:
            pass
    return None, None


def klines_to_dataframe(klines):
    # Convert raw Binance kline array to structured DataFrame with taker order flow
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms").dt.strftime(TIMESTAMP_FORMAT)
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, pd.NA)
    df = df.rename(columns={
        "quote_asset_volume": "quote_volume",
        "taker_buy_base": "taker_buy_base_volume",
        "taker_buy_quote": "taker_buy_quote_volume",
    })
    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume", "taker_buy_ratio"]]


def main():
    api_key, api_secret = load_binance_credentials()
    client = Client(api_key, api_secret) if api_key and api_secret else Client()
    klines = client.get_historical_klines(SYMBOL, INTERVAL, f"{YEARS_OF_HISTORY} years ago UTC")
    if not klines:
        raise RuntimeError("No data returned from Binance.")
    df = klines_to_dataframe(klines)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(OUTPUT_PATH, OUTPUT_PATH.replace(".csv", f"_backup_{stamp}.csv"))
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} candles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
