from pathlib import Path
from typing import Union
import os
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import BTCUSDT_1H_CSV, FEATURED_BTC_DATA_PATH
from src.data_preprocessing import (
    load_and_clean_data,
    merge_fear_greed,
    merge_onchain,
)
from src.feature_engineering import create_features


def preprocess_and_save_featured_data(
    input_path: Union[str, Path] = BTCUSDT_1H_CSV,
    output_path: Union[str, Path] = FEATURED_BTC_DATA_PATH,
    fetch_fresh: bool = False,
) -> pd.DataFrame:
    # Preprocesses raw candlesticks, merges sentiment and on-chain metrics, and creates features
    input_file, output_file = Path(input_path), Path(output_path)
    if fetch_fresh or not input_file.exists():
        from scripts.ingest.fetch_fresh_data import main as fetch_main
        fetch_main()

    df = load_and_clean_data(input_file)
    df = merge_fear_greed(df, timestamp_col="timestamp")
    df = merge_onchain(df, timestamp_col="timestamp")
    featured_df = create_features(df, drop_na=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(output_file, index=False)
    print(f"Generated {len(featured_df)} rows x {featured_df.shape[1]} features -> {output_file}")
    return featured_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare feature dataset for Mudra quantitative models.")
    parser.add_argument("--fetch-fresh", action="store_true", help="Fetch latest BTC klines first")
    args = parser.parse_args()
    preprocess_and_save_featured_data(fetch_fresh=args.fetch_fresh)
