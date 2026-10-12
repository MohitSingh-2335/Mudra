# src/data_preprocessing.py
"""
Data preprocessing module for Mudra quantitative pipeline.
Standardizes raw candle timestamps, removes duplicates, and enforces chronological integrity.
"""

from pathlib import Path
from typing import Union
import pandas as pd


def load_and_clean_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads raw candlestick dataset from CSV, parses timestamps into standardized
    UTC datetime objects, deduplicates timestamps, and enforces strict chronological order.

    Args:
        file_path (str | Path): Path to the candlestick CSV file.

    Returns:
        pd.DataFrame: Cleaned, deduplicated, and chronologically ordered DataFrame.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"[ERROR] Target data file not found: {path_obj}")

    df = pd.read_csv(path_obj)

    if "timestamp" not in df.columns:
        raise KeyError(f"[ERROR] 'timestamp' column missing from {path_obj}. Found: {list(df.columns)}")

    # Parse timestamps flexibly (handles both '%Y:%m:%d %H:%M:%S' and standard ISO formats)
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y:%m:%d %H:%M:%S")
    except ValueError:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Deduplicate timestamps, sort chronologically, and reset scrambled index
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df