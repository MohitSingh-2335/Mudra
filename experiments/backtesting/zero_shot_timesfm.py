# zero_shot_timesfm.py — lines 1-45 replaced

import numpy as np
import pandas as pd
import timesfm

from src.data_preprocessing import load_and_clean_data

CONTEXT_LEN = 512
N_EVAL = 500


def main():
    df = load_and_clean_data('data/raw/BTCUSDT-1H.csv').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    eval_start = max(CONTEXT_LEN, len(test_df) - N_EVAL)
    eval_indices = list(range(eval_start, len(test_df)))

    print("Loading TimesFM 2.5 (zero-shot, no training)...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_LEN,
            max_horizon=1,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    actuals, preds, prev_closes = [], [], []

    print(f"Running {len(eval_indices)} rolling zero-shot forecasts (context={CONTEXT_LEN}h each)...")
    for n, i in enumerate(eval_indices):
        context = test_df['close'].iloc[i - CONTEXT_LEN:i].values
        point_forecast, _ = model.forecast(horizon=1, inputs=[context])
        preds.append(point_forecast[0][0])
        actuals.append(test_df['close'].iloc[i])
        prev_closes.append(test_df['close'].iloc[i - 1])

        if n % 50 == 0:
            print(f"  {n}/{len(eval_indices)} done...")

    actuals = np.array(actuals)
    preds = np.array(preds)
    prev_closes = np.array(prev_closes)

    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    baseline_rmse = np.sqrt(np.mean((actuals - prev_closes) ** 2))
    pred_dir = (preds > prev_closes).astype(int)
    actual_dir = (actuals > prev_closes).astype(int)
    dir_acc = (pred_dir == actual_dir).mean() * 100

    print("\n" + "=" * 60)
    print("ZERO-SHOT TIMESFM 2.5 RESULTS")
    print("=" * 60)
    print(f"Eval points          : {len(eval_indices)}")
    print(f"RMSE (TimesFM)        : {rmse:.4f}")
    print(f"RMSE (naive baseline) : {baseline_rmse:.4f}")
    print(f"  -> {'BEATS' if rmse < baseline_rmse else 'does NOT beat'} naive baseline")
    print(f"Directional accuracy  : {dir_acc:.2f}%")
    print(f"  -> vs LogReg benchmark 52.49%: {'BEATS' if dir_acc > 52.49 else 'does NOT beat'}")
    print(f"  -> vs naive baseline 49.58%: {'BEATS' if dir_acc > 49.58 else 'does NOT beat'}")

    out = pd.DataFrame({
        'timestamp': test_df['timestamp'].iloc[eval_indices].values,
        'actual': actuals, 'predicted': preds, 'prev_close': prev_closes,
    })
    out.to_csv('data/results/zero_shot_timesfm_results.csv', index=False)
    print("\nSaved: data/results/zero_shot_timesfm_results.csv")


if __name__ == '__main__':
    main()