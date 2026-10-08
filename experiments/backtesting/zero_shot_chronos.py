# zero_shot_chronos.py
#
# Phase 2, step 1: cheapest possible test of a foundation model before any
# fine-tuning investment. Pure zero-shot inference from amazon/chronos-2 —
# no training — rolled forward over held-out test data. Same held-out split
# as Phase 1 (chronological, last 20%), same metrics (RMSE vs naive
# persistence baseline, directional accuracy vs the 52.49% LogisticRegression
# benchmark from 05_PHASE1_BACKTESTING_REPORT.md), so results are directly
# comparable.
#
# Install first:
#   pip install chronos-forecasting torch
#
# Run locally on CPU for this first pass (small eval set, no training) or
# set DEVICE="cuda" if you have a GPU / are in Colab.

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline

from src.data_preprocessing import load_and_clean_data

CONTEXT_LEN = 512   # hours of history fed to the model per forecast
N_EVAL = 500        # held-out points to evaluate — keep small for a first CPU pass
DEVICE = "cpu"       # "cuda" if available


def main():
    df = load_and_clean_data('data/raw/BTCUSDT-1H.csv').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    eval_start = max(CONTEXT_LEN, len(test_df) - N_EVAL)
    eval_indices = list(range(eval_start, len(test_df)))

    print("Loading Chronos-2 (zero-shot, no training)...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)

    actuals, preds, prev_closes = [], [], []

    print(f"Running {len(eval_indices)} rolling zero-shot forecasts (context={CONTEXT_LEN}h each)...")
    for n, i in enumerate(eval_indices):
        context = test_df.iloc[i - CONTEXT_LEN:i]
        context_df = pd.DataFrame({
            'id': ['BTCUSDT'] * len(context),
            'timestamp': context['timestamp'].values,
            'target': context['close'].values,
        })
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=1,
            quantile_levels=[0.5],
            id_column='id',
            timestamp_column='timestamp',
            target='target',
        )
        pred_col = '0.5' if '0.5' in pred_df.columns else pred_df.columns[-1]
        preds.append(pred_df[pred_col].iloc[0])
        actuals.append(test_df['close'].iloc[i])
        prev_closes.append(test_df['close'].iloc[i - 1])

        if n % 50 == 0:
            print(f"  {n}/{len(eval_indices)} done...")

    actuals = np.array(actuals)
    preds = np.array(preds)
    prev_closes = np.array(prev_closes)

    # RMSE vs naive persistence baseline, on the same eval subsample
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    baseline_rmse = np.sqrt(np.mean((actuals - prev_closes) ** 2))

    # Directional accuracy vs Phase 1 benchmarks (LogReg=52.49%, naive baseline=49.58%)
    pred_dir = (preds > prev_closes).astype(int)
    actual_dir = (actuals > prev_closes).astype(int)
    dir_acc = (pred_dir == actual_dir).mean() * 100

    print("\n" + "=" * 60)
    print("ZERO-SHOT CHRONOS-2 RESULTS")
    print("=" * 60)
    print(f"Eval points          : {len(eval_indices)}")
    print(f"RMSE (Chronos-2)      : {rmse:.4f}")
    print(f"RMSE (naive baseline) : {baseline_rmse:.4f}")
    print(f"  -> {'BEATS' if rmse < baseline_rmse else 'does NOT beat'} naive baseline")
    print(f"Directional accuracy  : {dir_acc:.2f}%")
    print(f"  -> vs LogReg benchmark 52.49%: {'BEATS' if dir_acc > 52.49 else 'does NOT beat'}")
    print(f"  -> vs naive baseline 49.58%: {'BEATS' if dir_acc > 49.58 else 'does NOT beat'}")

    out = pd.DataFrame({
        'timestamp': test_df['timestamp'].iloc[eval_indices].values,
        'actual': actuals, 'predicted': preds, 'prev_close': prev_closes,
    })
    out.to_csv('data/results/zero_shot_chronos_results.csv', index=False)
    print("\nSaved: data/results/zero_shot_chronos_results.csv")


if __name__ == '__main__':
    main()