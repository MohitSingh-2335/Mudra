# experiments/backtesting/zero_shot_timesfm.py
#
# Zero-shot evaluation of Google Research's TimesFM 2.5
# (200M parameter decoder-only foundation time-series model).
# Evaluates zero-shot rolling forecasts over the identical chronological
# held-out test split (last 20%) used across classical and foundation benchmarks.
#
# Metrics tracked:
#   1. RMSE vs naive persistence baseline (y_hat = y_{t-1})
#   2. Directional accuracy vs LogisticRegression (52.49%) & naive (49.58%)
#
# Install first (optional foundation research dependency):
#   pip install timesfm torch

import sys
from pathlib import Path
from typing import Any, List, Optional

# Guard project root resolution for standalone CLI execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    import timesfm
    TIMESFM_AVAILABLE = True
except ImportError:
    timesfm = None  # type: ignore[assignment]
    TIMESFM_AVAILABLE = False

from src.data_preprocessing import load_and_clean_data
from config import BTCUSDT_1H_CSV

CONTEXT_LEN = 512   # hours of historical lookback per forecast
N_EVAL = 500        # held-out rolling test points for zero-shot evaluation


def get_device() -> str:
    """Detects available hardware acceleration for foundation model evaluation."""
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    print("[INFO] Initializing Google TimesFM 2.5 Zero-Shot Foundation Benchmark...")
    if not TIMESFM_AVAILABLE:
        print("[ERROR] Optional dependency 'timesfm' is not installed in the active environment.")
        print("[INFO] To run TimesFM zero-shot experiments, install:")
        print("       pip install timesfm torch")
        return

    device = get_device()
    print(f"[INFO] Hardware Acceleration Runtime: {device.upper()}")

    if not os.path.exists(BTCUSDT_1H_CSV):
        raise FileNotFoundError(f"[ERROR] Raw candlestick dataset not found at '{BTCUSDT_1H_CSV}'.")

    df = load_and_clean_data(BTCUSDT_1H_CSV).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    eval_start = max(CONTEXT_LEN, len(test_df) - N_EVAL)
    eval_indices = list(range(eval_start, len(test_df)))

    print("[INFO] Loading TimesFM 2.5 weights (google/timesfm-2.5-200m-pytorch, zero-shot)...")
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

    print(f"[INFO] Running {len(eval_indices)} rolling zero-shot forecasts (context={CONTEXT_LEN}h each)...")
    for n, i in enumerate(eval_indices):
        context = test_df['close'].iloc[i - CONTEXT_LEN:i].values
        point_forecast, _ = model.forecast(horizon=1, inputs=[context])
        preds.append(point_forecast[0][0])
        actuals.append(test_df['close'].iloc[i])
        prev_closes.append(test_df['close'].iloc[i - 1])

        if n % 50 == 0:
            print(f"[PROGRESS] {n}/{len(eval_indices)} rolling horizons completed...")

    actuals = np.array(actuals)
    preds = np.array(preds)
    prev_closes = np.array(prev_closes)

    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    baseline_rmse = np.sqrt(np.mean((actuals - prev_closes) ** 2))
    pred_dir = (preds > prev_closes).astype(int)
    actual_dir = (actuals > prev_closes).astype(int)
    dir_acc = (pred_dir == actual_dir).mean() * 100

    print("\n" + "=" * 60)
    print("[BENCHMARK] ZERO-SHOT TIMESFM 2.5 RESULTS")
    print("=" * 60)
    print(f"Eval points          : {len(eval_indices)}")
    print(f"RMSE (TimesFM 2.5)    : {rmse:.4f}")
    print(f"RMSE (naive baseline) : {baseline_rmse:.4f}")
    print(f"  -> {'BEATS' if rmse < baseline_rmse else 'does NOT beat'} naive baseline")
    print(f"Directional accuracy  : {dir_acc:.2f}%")
    print(f"  -> vs LogReg benchmark 52.49%: {'BEATS' if dir_acc > 52.49 else 'does NOT beat'}")
    print(f"  -> vs naive baseline 49.58%: {'BEATS' if dir_acc > 49.58 else 'does NOT beat'}")

    out = pd.DataFrame({
        'timestamp': test_df['timestamp'].iloc[eval_indices].values,
        'actual': actuals, 'predicted': preds, 'prev_close': prev_closes,
    })
    Path('data/results').mkdir(parents=True, exist_ok=True)
    out.to_csv('data/results/zero_shot_timesfm_results.csv', index=False)
    print("[SUCCESS] Zero-shot TimesFM 2.5 results saved to data/results/zero_shot_timesfm_results.csv")


if __name__ == '__main__':
    main()