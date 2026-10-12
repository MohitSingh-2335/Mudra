# walk_forward_validate.py
#
# Single train/test split + picking the best of 27 configs is a classic
# false-positive setup (multiple comparisons). This re-tests the winning
# config (LogisticRegression @ threshold=0.68) across several INDEPENDENT
# rolling windows to see if the edge replicates out-of-sample, or if it
# was a lucky draw on one particular slice of history.

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Guard project root resolution for standalone CLI execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.backtesting import run_backtest, compute_metrics, load_and_clean_data, create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.agents.sentiment_agent import merge_sentiment
from config import (
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH,
    BTCUSDT_1H_CSV
)

THRESHOLD = 0.68
N_WINDOWS = 5            # number of independent, non-overlapping test windows
TRAIN_FRAC = 0.6         # fraction of each window used for training


def train(df: pd.DataFrame, split_idx: int) -> Tuple[LogisticRegression, StandardScaler]:
    """Trains a logistic regression model with StandardScaler normalization."""
    feature_cols = [c for c in CLASSIFIER_FEATURES if c in df.columns]
    X_train = df[feature_cols].iloc[:split_idx]
    y_train = df['Target_Movement'].iloc[:split_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def main() -> None:
    print("[INFO] Initializing Walk-Forward Validation Engine across independent windows...")
    if os.path.exists(FEATURED_BTC_DATA_PATH):
        print(f"[INFO] Loading pre-computed featured dataset from {FEATURED_BTC_DATA_PATH}...")
        df = pd.read_csv(FEATURED_BTC_DATA_PATH, parse_dates=['timestamp'])
    else:
        print(f"[INFO] Computing features from raw feed {BTCUSDT_1H_CSV}...")
        raw_df = load_and_clean_data(BTCUSDT_1H_CSV)
        df = create_features(merge_sentiment(merge_onchain(merge_fear_greed(raw_df)))).reset_index(drop=True)

    n = len(df)
    window_size = n // N_WINDOWS

    results: List[Dict[str, Any]] = []
    for w in range(N_WINDOWS):
        start = w * window_size
        end = start + window_size if w < N_WINDOWS - 1 else n
        window_df = df.iloc[start:end].reset_index(drop=True)

        split_idx = int(len(window_df) * TRAIN_FRAC)
        test_df = window_df.iloc[split_idx:].reset_index(drop=True)
        if len(test_df) < 20:
            continue

        print(f"[INFO] Evaluating Window {w + 1}/{N_WINDOWS} ({window_df['timestamp'].iloc[0]} -> {window_df['timestamp'].iloc[-1]})...")
        model, scaler = train(window_df, split_idx)
        ec, tl = run_backtest(test_df, model, scaler, threshold=THRESHOLD)
        m = compute_metrics(ec, tl, test_df)
        m['window'] = w + 1
        m['period_start'] = window_df['timestamp'].iloc[0]
        m['period_end'] = window_df['timestamp'].iloc[-1]
        m['buy_hold_return_pct'] = m.get('bh_return_pct', 0.0)
        m['sharpe_ratio_annualized'] = m.get('sharpe_annualized', 0.0)
        results.append(m)

    out = pd.DataFrame(results)[
        ['window', 'period_start', 'period_end', 'num_trades', 'win_rate_pct',
         'total_return_pct', 'buy_hold_return_pct', 'sharpe_ratio_annualized']
    ]
    print("\n" + "=" * 90)
    print("[VALIDATION] WALK-FORWARD INDEPENDENT WINDOW PERFORMANCE SUMMARY")
    print("=" * 90)
    print(out.to_string(index=False))
    print("=" * 90)

    print(f"\n[METRIC] Windows where strategy beat Buy&Hold: "
          f"{(out['total_return_pct'] > out['buy_hold_return_pct']).sum()} / {len(out)}")
    print(f"[METRIC] Total trades across all windows: {out['num_trades'].sum()}")
    print(f"[METRIC] Mean return across windows: {out['total_return_pct'].mean():.2f}%  "
          f"(std: {out['total_return_pct'].std():.2f}%)")

    Path('data/results').mkdir(parents=True, exist_ok=True)
    out.to_csv('data/results/walk_forward_validation.csv', index=False)
    print("[SUCCESS] Walk-forward validation results serialized to data/results/walk_forward_validation.csv")


if __name__ == '__main__':
    main()