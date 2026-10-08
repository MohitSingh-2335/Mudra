# zero_shot_moirai.py
#
# Phase 2, step 1 (3rd and final TSFM candidate per 04_ROADMAP.md).
# Zero-shot only — no fine-tuning. Same held-out split, same eval
# methodology (CONTEXT_LEN, N_EVAL) as zero_shot_chronos.py / zero_shot_timesfm.py.

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from src.data_preprocessing import load_and_clean_data

CONTEXT_LEN = 512
N_EVAL = 500
BSZ = 32


def main():
    df = load_and_clean_data('data/raw/BTCUSDT-1H.csv').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    eval_start = max(CONTEXT_LEN, len(test_df) - N_EVAL)
    eval_indices = list(range(eval_start, len(test_df)))

    print("Loading Moirai 2.0 (zero-shot, no training)...")
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
        prediction_length=1,
        context_length=CONTEXT_LEN,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=BSZ)

    print(f"Building {len(eval_indices)} rolling windows (context={CONTEXT_LEN}h each)...")
    entries = []
    for i in eval_indices:
        context = test_df['close'].iloc[i - CONTEXT_LEN:i].values.astype(np.float32)
        entries.append({
            'target': context,
            'start': pd.Period(test_df['timestamp'].iloc[i - CONTEXT_LEN], freq='H'),
        })
    ds = ListDataset(entries, freq='H')

    print("Running batched zero-shot inference...")
    forecasts = list(predictor.predict(ds))

    actuals, preds, prev_closes = [], [], []
    for n, (i, fcst) in enumerate(zip(eval_indices, forecasts)):
        median = np.median(fcst.samples[:, 0]) if hasattr(fcst, 'samples') else fcst.mean[0]
        preds.append(median)
        actuals.append(test_df['close'].iloc[i])
        prev_closes.append(test_df['close'].iloc[i - 1])

    actuals = np.array(actuals)
    preds = np.array(preds)
    prev_closes = np.array(prev_closes)

    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    baseline_rmse = np.sqrt(np.mean((actuals - prev_closes) ** 2))
    pred_dir = (preds > prev_closes).astype(int)
    actual_dir = (actuals > prev_closes).astype(int)
    dir_acc = (pred_dir == actual_dir).mean() * 100

    print("\n" + "=" * 60)
    print("ZERO-SHOT MOIRAI 2.0 RESULTS")
    print("=" * 60)
    print(f"Eval points          : {len(eval_indices)}")
    print(f"RMSE (Moirai-2)       : {rmse:.4f}")
    print(f"RMSE (naive baseline) : {baseline_rmse:.4f}")
    print(f"  -> {'BEATS' if rmse < baseline_rmse else 'does NOT beat'} naive baseline")
    print(f"Directional accuracy  : {dir_acc:.2f}%")
    print(f"  -> vs LogReg benchmark 52.49%: {'BEATS' if dir_acc > 52.49 else 'does NOT beat'}")
    print(f"  -> vs naive baseline 49.58%: {'BEATS' if dir_acc > 49.58 else 'does NOT beat'}")

    out = pd.DataFrame({
        'timestamp': test_df['timestamp'].iloc[eval_indices].values,
        'actual': actuals, 'predicted': preds, 'prev_close': prev_closes,
    })
    out.to_csv('data/results/zero_shot_moirai_results.csv', index=False)
    print("\nSaved: data/results/zero_shot_moirai_results.csv")


if __name__ == '__main__':
    main()