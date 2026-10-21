# significance_test.py
#
# Tests whether the high-threshold backtest results (LogReg@0.68, XGB@0.70)
# are distinguishable from noise, given tiny trade counts.

import sys
from pathlib import Path
from typing import Any, List, Tuple

# Guard project root resolution for standalone CLI execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.backtesting import run_backtest, load_and_clean_data, create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.agents.sentiment_agent import merge_sentiment
from config import (
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH,
    BTCUSDT_1H_CSV,
    DEFAULT_SIGNIFICANCE_ALPHA,
)

CONFIGS = [
    ("LogisticRegression", LogisticRegression(max_iter=1000), 0.68),
    ("XGBoost", XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        device="cpu",
        random_state=42,
        eval_metric='logloss'
    ), 0.70),
]


def binomial_test(win_rate_pct: float, n_trades: int) -> Any:
    """Computes exact one-sided binomial test under H0: p = 0.5 vs H1: p > 0.5."""
    wins = round(win_rate_pct / 100 * n_trades)
    return stats.binomtest(wins, n_trades, p=0.5, alternative='greater')


def bootstrap_return_ci(
    trade_returns_pct: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Computes non-parametric 95% bootstrap confidence interval of mean per-trade return."""
    rng = np.random.default_rng(seed)
    arr = np.array(trade_returns_pct)
    boot_indices = rng.choice(len(arr), size=(n_boot, len(arr)), replace=True)
    boot_means = arr[boot_indices].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)


def evaluate_trade_significance(
    trade_logs: pd.DataFrame,
    alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    verbose: bool = True
) -> Any:
    """
    Evaluates empirical statistical significance of any model's backtest trade log.
    Universal for Model 1, Model 2, or any future trading strategy.
    """
    n = len(trade_logs)
    if n == 0:
        if verbose:
            print("[WARNING] Zero trades executed across evaluation window.")
        return {
            "total_trades": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "binomial_pvalue": 1.0,
            "is_significant": False,
        }

    wins = int((trade_logs['outcome'] == 'WIN').sum())
    win_rate = float(wins / n * 100.0)
    bt = binomial_test(win_rate, n)
    p_val = float(bt.pvalue)
    is_sig = bool(p_val < alpha)

    mean_gross, lo_g, hi_g = bootstrap_return_ci(trade_logs['gross_return_pct'].values)
    mean_net, lo_n, hi_n = bootstrap_return_ci(trade_logs['net_return_pct'].values)

    result = {
        "total_trades": n,
        "wins": wins,
        "win_rate_pct": round(win_rate, 2),
        "binomial_pvalue": round(p_val, 5),
        "is_significant": is_sig,
        "gross_mean_pct": round(mean_gross, 3),
        "gross_ci_95": (round(lo_g, 3), round(hi_g, 3)),
        "net_mean_pct": round(mean_net, 3),
        "net_ci_95": (round(lo_n, 3), round(hi_n, 3)),
    }

    if verbose:
        print(f"[STATISTIC] Total Trades: {n} | Wins: {wins} | Win Rate: {win_rate:.1f}%")
        print(f"[STATISTIC] Binomial Test vs p=0.5 (H1: win rate > 50%): p-value = {p_val:.4f}")
        decision = f"Significant at alpha={alpha}" if is_sig else "NOT statistically significant (indistinguishable from random noise)"
        print(f"  -> {decision}")
        print(f"[STATISTIC] Gross Per-Trade Return: mean={mean_gross:.3f}%, 95% Bootstrap CI=[{lo_g:.3f}%, {hi_g:.3f}%]")
        gross_eval = "CI strictly excludes 0 (empirical gross edge detected)" if lo_g > 0 else "CI spans 0 (insufficient evidence of gross edge)"
        print(f"  -> {gross_eval}")
        print(f"[STATISTIC] Net Per-Trade Return (Post-Friction): mean={mean_net:.3f}%, 95% Bootstrap CI=[{lo_n:.3f}%, {hi_n:.3f}%]")
        net_eval = "Net CI strictly excludes 0 (institutional fee-clearing edge)" if lo_n > 0 else "Net CI spans 0 (edge fully consumed by transaction friction)"
        print(f"  -> {net_eval}")

    return result


def main() -> None:
    print("[INFO] Initializing Empirical Statistical Significance Benchmark...")
    if os.path.exists(FEATURED_BTC_DATA_PATH):
        print(f"[INFO] Loading pre-computed featured dataset from {FEATURED_BTC_DATA_PATH}...")
        df = pd.read_csv(FEATURED_BTC_DATA_PATH, parse_dates=['timestamp'])
    else:
        print(f"[INFO] Computing features from raw feed {BTCUSDT_1H_CSV}...")
        raw_df = load_and_clean_data(BTCUSDT_1H_CSV)
        df = create_features(merge_sentiment(merge_onchain(merge_fear_greed(raw_df)))).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    for name, model, threshold in CONFIGS:
        feature_cols = [c for c in CLASSIFIER_FEATURES if c in df.columns]
        X_train = df[feature_cols].iloc[:split_idx]
        y_train = df['Target_Movement'].iloc[:split_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)

        ec, tl = run_backtest(test_df, model, scaler, threshold=threshold)

        print(f"\n" + "=" * 80)
        print(f"[SIGNIFICANCE] Model: {name} (Conviction Threshold = {threshold})")
        print("=" * 80)
        evaluate_trade_significance(tl, alpha=DEFAULT_SIGNIFICANCE_ALPHA)


if __name__ == '__main__':
    main()
