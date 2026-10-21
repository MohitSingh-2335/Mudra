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
    BTCUSDT_1H_CSV,
    DEFAULT_WALK_FORWARD_WINDOWS,
    DEFAULT_WALK_FORWARD_TRAIN_FRAC,
)

THRESHOLD = 0.68
N_WINDOWS = DEFAULT_WALK_FORWARD_WINDOWS
TRAIN_FRAC = DEFAULT_WALK_FORWARD_TRAIN_FRAC


def run_walk_forward_validation(
    df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    n_windows: int = DEFAULT_WALK_FORWARD_WINDOWS,
    train_frac: float = DEFAULT_WALK_FORWARD_TRAIN_FRAC,
    threshold: float = 0.50,
    scaler: Any = None,
    sample_weight: Any = None,
    target_col: str = 'Target_Movement',
) -> pd.DataFrame:
    """
    Executes walk-forward evaluation across N chronological non-overlapping windows.
    Universal for any model (LogisticRegression, XGBoost, etc.) and any feature schema.
    """
    n = len(df)
    window_size = n // n_windows

    results: List[Dict[str, Any]] = []
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else n
        window_df = df.iloc[start:end].reset_index(drop=True)

        split_idx = int(len(window_df) * train_frac)
        test_df = window_df.iloc[split_idx:].reset_index(drop=True)
        if len(test_df) < 20:
            continue

        print(f"[INFO] Evaluating Window {w + 1}/{n_windows} ({window_df['timestamp'].iloc[0]} -> {window_df['timestamp'].iloc[-1]})...")
        X_train = window_df[feature_cols].iloc[:split_idx]
        y_train = window_df[target_col].iloc[:split_idx]

        # Apply scaler if provided
        fit_scaler = None
        if scaler is not None:
            from sklearn.base import clone
            fit_scaler = clone(scaler)
            X_train_fit = fit_scaler.fit_transform(X_train)
        else:
            X_train_fit = X_train

        # Fit model (with sample_weight if provided)
        from sklearn.base import clone
        fit_model = clone(model)
        if sample_weight is not None:
            if hasattr(sample_weight, "iloc"):
                w_train = sample_weight.iloc[start : start + split_idx] if len(sample_weight) == len(df) else sample_weight.iloc[:split_idx]
            else:
                w_train = sample_weight[start : start + split_idx] if len(sample_weight) == len(df) else sample_weight[:split_idx]
            fit_model.fit(X_train_fit, y_train, sample_weight=w_train)
        else:
            fit_model.fit(X_train_fit, y_train)

        ec, tl = run_backtest(
            test_df,
            fit_model,
            fit_scaler,
            threshold=threshold,
            feature_cols=feature_cols,
        )
        m = compute_metrics(ec, tl, test_df)
        m['window'] = w + 1
        m['period_start'] = window_df['timestamp'].iloc[0]
        m['period_end'] = window_df['timestamp'].iloc[-1]
        m['buy_hold_return_pct'] = m.get('bh_return_pct', 0.0)
        m['sharpe_ratio_annualized'] = m.get('sharpe_annualized', 0.0)
        results.append(m)

    return pd.DataFrame(results)[
        ['window', 'period_start', 'period_end', 'num_trades', 'win_rate_pct',
         'total_return_pct', 'buy_hold_return_pct', 'sharpe_ratio_annualized']
    ]


def main() -> None:
    print("[INFO] Initializing Walk-Forward Validation Engine across independent windows...")
    if os.path.exists(FEATURED_BTC_DATA_PATH):
        print(f"[INFO] Loading pre-computed featured dataset from {FEATURED_BTC_DATA_PATH}...")
        df = pd.read_csv(FEATURED_BTC_DATA_PATH, parse_dates=['timestamp'])
    else:
        print(f"[INFO] Computing features from raw feed {BTCUSDT_1H_CSV}...")
        raw_df = load_and_clean_data(BTCUSDT_1H_CSV)
        df = create_features(merge_sentiment(merge_onchain(merge_fear_greed(raw_df)))).reset_index(drop=True)

    feature_cols = [c for c in CLASSIFIER_FEATURES if c in df.columns]
    out = run_walk_forward_validation(
        df=df,
        model=LogisticRegression(max_iter=1000),
        feature_cols=feature_cols,
        n_windows=DEFAULT_WALK_FORWARD_WINDOWS,
        train_frac=DEFAULT_WALK_FORWARD_TRAIN_FRAC,
        threshold=THRESHOLD,
        scaler=StandardScaler(),
    )
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