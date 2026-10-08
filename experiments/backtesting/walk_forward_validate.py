# walk_forward_validate.py
#
# Single train/test split + picking the best of 27 configs is a classic
# false-positive setup (multiple comparisons). This re-tests the winning
# config (LogisticRegression @ threshold=0.68) across several INDEPENDENT
# rolling windows to see if the edge replicates out-of-sample, or if it
# was a lucky draw on one particular slice of history.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.backtesting import run_backtest, compute_metrics, load_and_clean_data, create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.agents.sentiment_agent import merge_sentiment
from config import SVC_FEATURES
from src.agents.fear_greed_agent import merge_fear_greed

THRESHOLD = 0.68
N_WINDOWS = 5            # number of independent, non-overlapping test windows
TRAIN_FRAC = 0.6         # fraction of each window used for training


def train(df, split_idx):
    X_train = df[SVC_FEATURES].iloc[:split_idx]
    y_train = df['Target_Movement'].iloc[:split_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def main():
    df = create_features(merge_sentiment(merge_onchain(merge_fear_greed(load_and_clean_data('data/raw/BTCUSDT-1H.csv'))))).reset_index(drop=True)
    n = len(df)
    window_size = n // N_WINDOWS

    results = []
    for w in range(N_WINDOWS):
        start = w * window_size
        end = start + window_size if w < N_WINDOWS - 1 else n
        window_df = df.iloc[start:end].reset_index(drop=True)

        split_idx = int(len(window_df) * TRAIN_FRAC)
        test_df = window_df.iloc[split_idx:].reset_index(drop=True)
        if len(test_df) < 20:
            continue

        model, scaler = train(window_df, split_idx)
        ec, tl = run_backtest(test_df, model, scaler, threshold=THRESHOLD)
        m = compute_metrics(ec, tl, test_df)
        m['window'] = w + 1
        m['period_start'] = window_df['timestamp'].iloc[0]
        m['period_end'] = window_df['timestamp'].iloc[-1]
        results.append(m)

    out = pd.DataFrame(results)[
        ['window', 'period_start', 'period_end', 'num_trades', 'win_rate_pct',
         'total_return_pct', 'buy_hold_return_pct', 'sharpe_ratio_annualized']
    ]
    print(out.to_string(index=False))

    print(f"\nWindows where strategy beat Buy&Hold: "
          f"{(out['total_return_pct'] > out['buy_hold_return_pct']).sum()} / {len(out)}")
    print(f"Total trades across all windows: {out['num_trades'].sum()}")
    print(f"Mean return across windows: {out['total_return_pct'].mean():.2f}%  "
          f"(std: {out['total_return_pct'].std():.2f}%)")

    out.to_csv('data/results/walk_forward_validation.csv', index=False)


if __name__ == '__main__':
    main()