# scripts/prepare_data.py
"""
Data preparation pipeline for Mudra quantitative models.
Orchestrates raw candlestick cleaning, macro Fear & Greed sentiment merging,
Blockchain.com on-chain network metrics extraction, and technical feature
engineering, writing the final feature matrix to data/processed/featured_btc_data.csv.
"""

from pathlib import Path
from typing import Union
import os
import sys
import pandas as pd

# Ensure repository root is on sys.path for direct CLI script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import BTCUSDT_1H_CSV, FEATURED_BTC_DATA_PATH
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features


def preprocess_and_save_featured_data(
    input_path: Union[str, Path] = BTCUSDT_1H_CSV,
    output_path: Union[str, Path] = FEATURED_BTC_DATA_PATH,
) -> pd.DataFrame:
    """
    Loads raw candlestick data, merges macro and on-chain alternative features,
    generates 39 technical features, and persists the dataset to disk.

    Args:
        input_path (Union[str, Path]): Path to raw OHLCV candlestick CSV.
        output_path (Union[str, Path]): Target path for featured dataset CSV.

    Returns:
        pd.DataFrame: Engineered feature DataFrame.

    Raises:
        FileNotFoundError: If input candlestick CSV does not exist.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"[ERROR] Raw candlestick data not found at '{input_file}'. "
            "Please execute 'python scripts/ingest/fetch_fresh_data.py' first."
        )

    print(f"[INFO] Starting data preparation pipeline from: '{input_file}'")

    # Step 1: Ingest and clean raw candlestick timestamps and prices
    df = load_and_clean_data(input_file)
    print(f"[INFO] Loaded {len(df):,} raw hourly candles ({df['timestamp'].min()} -> {df['timestamp'].max()})")

    # Step 2: Merge macro Fear & Greed Index
    print("[INFO] Merging Fear & Greed sentiment metrics...")
    df = merge_fear_greed(df, timestamp_col="timestamp")

    # Step 3: Merge Blockchain.com on-chain network dynamics
    print("[INFO] Merging on-chain network metrics (transaction count, hash rate, miner revenue)...")
    df = merge_onchain(df, timestamp_col="timestamp")

    # Step 4: Engineer technical indicators, order flow, and return targets
    print("[INFO] Computing 39 technical indicators, microstructure, and target labels...")
    featured_df = create_features(df, drop_na=True)

    # Step 5: Serialize to processed storage
    output_file.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(output_file, index=False)

    print(
        f"[SUCCESS] Featured dataset generated: {len(featured_df):,} rows x "
        f"{featured_df.shape[1]} features -> '{output_file}'"
    )
    return featured_df


if __name__ == "__main__":
    preprocess_and_save_featured_data()