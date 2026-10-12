# experiments/backtesting/zero_shot_moirai.py
#
# Phase 2, step 3: zero-shot evaluation of Salesforce Research's MOIRAI 2.0
# (Universal Time-Series Foundation Model based on patch-level masked autoencoding).
# Evaluates zero-shot rolling forecasts over the identical chronological
# held-out test split (last 20%) used across all Phase 1 and Phase 2 benchmarks.
#
# Metrics tracked:
#   1. RMSE vs naive persistence baseline (y_hat = y_{t-1})
#   2. Directional accuracy vs LogisticRegression (52.49%) & naive (49.58%)
#
# Install first (optional foundation research dependency):
#   pip install uni2ts gluonts torch

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
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    MOIRAI_AVAILABLE = True
except ImportError:
    ListDataset = None  # type: ignore[assignment]
    Moirai2Forecast = None  # type: ignore[assignment]
    Moirai2Module = None  # type: ignore[assignment]
    MOIRAI_AVAILABLE = False

from src.data_preprocessing import load_and_clean_data
from config import BTCUSDT_1H_CSV

CONTEXT_LEN = 512   # hours of historical lookback per forecast
N_EVAL = 500        # held-out rolling test points for zero-shot evaluation
BSZ = 32            # batch size for parallel inference


def get_device() -> str:
    """Detects available hardware acceleration for foundation model evaluation."""
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    print("[INFO] Initializing Salesforce MOIRAI 2.0 Zero-Shot Foundation Benchmark...")
    if not MOIRAI_AVAILABLE:
        print("[ERROR] Optional dependency 'uni2ts' or 'gluonts' is not installed in the active environment.")
        print("[INFO] To run MOIRAI 2.0 zero-shot experiments, install:")
        print("       pip install uni2ts gluonts torch")
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

    print("[INFO] Loading MOIRAI 2.0 weights (Salesforce/moirai-2.0-R-small, zero-shot)...")
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
        prediction_length=1,
        context_length=CONTEXT_LEN,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=BSZ)

    print(f"[INFO] Building {len(eval_indices)} rolling windows (context={CONTEXT_LEN}h each)...")
    entries = []
    for i in eval_indices:
        context = test_df['close'].iloc[i - CONTEXT_LEN:i].values.astype(np.float32)
        entries.append({
            'target': context,
            'start': pd.Period(test_df['timestamp'].iloc[i - CONTEXT_LEN], freq='h'),
        })
    ds = ListDataset(entries, freq='h')

    print("[INFO] Running batched zero-shot inference...")
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
    print("[BENCHMARK] ZERO-SHOT MOIRAI 2.0 RESULTS")
    print("=" * 60)
    print(f"Eval points          : {len(eval_indices)}")
    print(f"RMSE (MOIRAI 2.0)     : {rmse:.4f}")
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
    out.to_csv('data/results/zero_shot_moirai_results.csv', index=False)
    print("[SUCCESS] Zero-shot MOIRAI 2.0 results saved to data/results/zero_shot_moirai_results.csv")


if __name__ == '__main__':
    main()