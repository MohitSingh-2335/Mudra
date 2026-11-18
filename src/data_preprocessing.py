from pathlib import Path
from typing import Union
import pandas as pd


# Load raw candle dataset, parse timestamps, remove duplicates, and sort chronologically
def load_and_clean_data(file_path: Union[str, Path]) -> pd.DataFrame:
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"[ERROR] Target data file not found: {path_obj}")

    df = pd.read_csv(path_obj)
    if "timestamp" not in df.columns:
        raise KeyError(f"[ERROR] 'timestamp' column missing from {path_obj}. Found: {list(df.columns)}")

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y:%m:%d %H:%M:%S")
    except ValueError:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df.sort_values("timestamp").reset_index(drop=True)
