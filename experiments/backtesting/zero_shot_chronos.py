# zero_shot_chronos.py
#
# Zero-shot evaluation of Amazon Chronos-2 foundation time-series model.
# Pure zero-shot inference rolled forward over held-out test data (chronological,
# last 20%), tracking RMSE vs naive persistence baseline and directional accuracy
# against classical model benchmarks (docs/reports/classical_model_backtesting.md).
#
# Install first (optional foundation research dependency):
#   pip install chronos-forecasting torch

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
    from chronos import Chronos2Pipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    Chronos2Pipeline = None  # type: ignore[assignment]
    CHRONOS_AVAILABLE = False

from src.data_preprocessing import load_and_clean_data
from config import BTCUSDT_1H_CSV

CONTEXT_LEN = 512   # hours of history fed to the model per forecast
N_EVAL = 500        # held-out points to evaluate — keep small for a first CPU pass


def get_device() -> str:
    """Detects available hardware acceleration for foundation model evaluation."""
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    print("[INFO] Initializing Amazon Chronos-2 Zero-Shot Foundation Benchmark...")
    if not CHRONOS_AVAILABLE:
        print("[ERROR] Optional dependency 'chronos-forecasting' is not installed in the active environment.")
        print("[INFO] To run zero-shot foundation model experiments, install:")
        print("       pip install chronos-forecasting torch")
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

    print("[INFO] Loading Chronos-2 weights (amazon/chronos-2, zero-shot)...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)

    actuals, preds, prev_closes = [], [], []

    print(f"[INFO] Running {len(eval_indices)} rolling zero-shot forecasts (context={CONTEXT_LEN}h each)...")
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
            print(f"[PROGRESS] {n}/{len(eval_indices)} rolling horizons completed...")

    actuals = np.array(actuals)
    preds = np.array(preds)
    prev_closes = np.array(prev_closes)

    # RMSE vs naive persistence baseline, on the same eval subsample
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    baseline_rmse = np.sqrt(np.mean((actuals - prev_closes) ** 2))

    # Directional accuracy vs classical benchmarks (LogReg=52.49%, naive baseline=49.58%)
    pred_dir = (preds > prev_closes).astype(int)
    actual_dir = (actuals > prev_closes).astype(int)
    dir_acc = (pred_dir == actual_dir).mean() * 100

    print("\n" + "=" * 60)
    print("[BENCHMARK] ZERO-SHOT CHRONOS-2 RESULTS")
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
    Path('data/results').mkdir(parents=True, exist_ok=True)
    out.to_csv('data/results/zero_shot_chronos_results.csv', index=False)
    print("[SUCCESS] Zero-shot Chronos-2 results saved to data/results/zero_shot_chronos_results.csv")


if __name__ == '__main__':
    main()