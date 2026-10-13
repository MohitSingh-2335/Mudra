# scripts/experiments/sweep_threshold.py
"""
Decision Threshold Sweeper for Directional Trading Models.
Evaluates classification probability thresholds t in [0.50, 0.70] for entering long positions.
Applies realistic transaction friction (0.05% taker fee + 0.02% slippage = 0.07% per fill) on chronological holdout
data to determine the optimal risk-adjusted Sharpe ratio and drawdown profile, saving results
to data/results/threshold_sweep.csv.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ensure repository root is on sys.path for direct CLI script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    BTCUSDT_1H_CSV,
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH,
    THRESHOLD_SWEEP,
    INITIAL_CAPITAL_USD,
    TAKER_FEE_RATE,
    SLIPPAGE_RATE,
    TOTAL_FRICTION_PER_SIDE_RATE,
)
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features

INITIAL_CAPITAL = INITIAL_CAPITAL_USD
FEE_RATE = TAKER_FEE_RATE  # 0.05% Binance VIP-0 taker fee per side
SLIPPAGE = SLIPPAGE_RATE   # 0.02% estimated execution slippage per fill
TOTAL_FRICTION = TOTAL_FRICTION_PER_SIDE_RATE  # 0.07% per execution (0.14% round trip)


def train_classifier(
    df: pd.DataFrame,
    split_idx: int,
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Trains a benchmark Logistic Regression model with standard feature scaling
    on the chronological training partition.

    Args:
        df (pd.DataFrame): Candlestick DataFrame with engineered features.
        split_idx (int): Chronological split index boundary.

    Returns:
        Tuple: (trained_model, fitted_scaler)
    """
    X_train = df[CLASSIFIER_FEATURES].iloc[:split_idx]
    y_train = df["Target_Movement"].iloc[:split_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def run_backtest_threshold(
    test_df: pd.DataFrame,
    model: Any,
    scaler: StandardScaler,
    threshold: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates a long/flat strategy using predicted probabilities above a given threshold.
    Applies canonical transaction friction (0.05% taker fee + 0.02% slippage = 0.07% per fill).

    Args:
        test_df (pd.DataFrame): Chronological holdout test DataFrame.
        model (Any): Trained classifier supporting predict_proba.
        scaler (StandardScaler): Fitted feature scaler.
        threshold (float): Long entry probability cutoff.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (equity_curve_df, trade_log_df)
    """
    X_test_scaled = scaler.transform(test_df[CLASSIFIER_FEATURES])
    probs = model.predict_proba(X_test_scaled)[:, 1]
    signals = (probs > threshold).astype(int)

    equity = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_equity = 0.0
    entry_time = None
    equity_curve: List[Dict[str, Any]] = []
    trade_log: List[Dict[str, Any]] = []

    for i in range(len(test_df)):
        price = float(test_df["close"].iloc[i])
        time = test_df["timestamp"].iloc[i]
        signal = signals[i]

        # Enter Long from Cash
        if position == 0 and signal == 1:
            entry_equity = equity * (1.0 - TOTAL_FRICTION)
            entry_price = price
            entry_time = time
            position = 1

        # Exit Long to Cash
        elif position == 1 and signal == 0:
            gross_return = (price / entry_price) - 1.0
            equity = entry_equity * (1.0 + gross_return) * (1.0 - TOTAL_FRICTION)
            net_return = (equity / (entry_equity / (1.0 - TOTAL_FRICTION))) - 1.0
            trade_log.append({
                "entry_time": entry_time,
                "exit_time": time,
                "entry_price": entry_price,
                "exit_price": price,
                "gross_return_pct": gross_return * 100.0,
                "net_return_pct": net_return * 100.0,
                "outcome": "WIN" if equity > entry_equity else "LOSS",
            })
            position = 0

        mtm_equity = entry_equity * (price / entry_price) if position == 1 else equity
        equity_curve.append({"timestamp": time, "equity": mtm_equity, "position": position})

    # Liquidate open position on final candle
    if position == 1:
        last_price = float(test_df["close"].iloc[-1])
        gross_return = (last_price / entry_price) - 1.0
        equity = entry_equity * (1.0 + gross_return) * (1.0 - TOTAL_FRICTION)
        net_return = (equity / (entry_equity / (1.0 - TOTAL_FRICTION))) - 1.0
        trade_log.append({
            "entry_time": entry_time,
            "exit_time": test_df["timestamp"].iloc[-1],
            "entry_price": entry_price,
            "exit_price": last_price,
            "gross_return_pct": gross_return * 100.0,
            "net_return_pct": net_return * 100.0,
            "outcome": "WIN" if equity > entry_equity else "LOSS",
        })
        equity_curve[-1]["equity"] = equity
        equity_curve[-1]["position"] = 0

    return pd.DataFrame(equity_curve), pd.DataFrame(trade_log)


def compute_metrics(
    equity_curve_df: pd.DataFrame,
    trade_log_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Computes summary quantitative metrics from simulated equity curve and trade logs.
    """
    final_equity = equity_curve_df["equity"].iloc[-1]
    total_return_pct = ((final_equity / INITIAL_CAPITAL) - 1.0) * 100.0
    bh_return_pct = ((test_df["close"].iloc[-1] / test_df["close"].iloc[0]) - 1.0) * 100.0

    hourly_returns = equity_curve_df["equity"].pct_change().fillna(0.0)
    sharpe = float((hourly_returns.mean() / hourly_returns.std()) * np.sqrt(8760)) if hourly_returns.std() > 0 else 0.0

    running_max = equity_curve_df["equity"].cummax()
    drawdown = (equity_curve_df["equity"] - running_max) / running_max
    max_drawdown_pct = float(drawdown.min() * 100.0)

    win_rate = float((trade_log_df["outcome"] == "WIN").mean() * 100.0) if len(trade_log_df) > 0 else 0.0

    return {
        "num_trades": len(trade_log_df),
        "win_rate_pct": win_rate,
        "total_return_pct": total_return_pct,
        "buy_hold_return_pct": bh_return_pct,
        "sharpe_ratio_annualized": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
    }


def run_threshold_sweep(
    data_path: Path = Path(BTCUSDT_1H_CSV),
    output_path: Path = Path(THRESHOLD_SWEEP),
    thresholds: List[float] = [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.68, 0.70],
) -> pd.DataFrame:
    """
    Executes the parameter sweep across candidate threshold values and writes output CSV.
    """
    print(f"[INFO] Ingesting and preparing data for threshold sweep from '{data_path}'...")
    df = load_and_clean_data(data_path)
    df = merge_fear_greed(df, timestamp_col="timestamp")
    df = merge_onchain(df, timestamp_col="timestamp")
    df = create_features(df, drop_na=True)
    print(f"[INFO] Engineered dataset prepared: {len(df):,} chronological candles.")

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"[INFO] Fitting baseline classifier on {split_idx:,} training candles...")
    model, scaler = train_classifier(df, split_idx)

    print(f"[INFO] Simulating backtests across {len(thresholds)} threshold levels...")
    results = []
    for t in thresholds:
        ec, tl = run_backtest_threshold(test_df, model, scaler, threshold=t)
        m = compute_metrics(ec, tl, test_df)
        m["threshold"] = t
        results.append(m)

    out = pd.DataFrame(results)[[
        "threshold",
        "num_trades",
        "win_rate_pct",
        "total_return_pct",
        "buy_hold_return_pct",
        "sharpe_ratio_annualized",
        "max_drawdown_pct",
    ]]

    print("\n--- THRESHOLD SWEEP RESULTS LEADERBOARD ---")
    print(out.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Threshold sweep results successfully saved to: '{output_path}'")
    return out


if __name__ == "__main__":
    run_threshold_sweep()