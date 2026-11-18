from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    BTCUSDT_1H_CSV, CLASSIFIER_FEATURES, THRESHOLD_SWEEP,
    INITIAL_CAPITAL_USD, TOTAL_FRICTION_PER_SIDE_RATE,
)
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features

INITIAL_CAPITAL = INITIAL_CAPITAL_USD
TOTAL_FRICTION = TOTAL_FRICTION_PER_SIDE_RATE


def train_classifier(df: pd.DataFrame, split_idx: int) -> Tuple[LogisticRegression, StandardScaler]:
    # trains scaled logistic regression baseline on train split
    X_train = df[CLASSIFIER_FEATURES].iloc[:split_idx]
    y_train = df["Target_Movement"].iloc[:split_idx]
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(scaler.fit_transform(X_train), y_train)
    return model, scaler


def run_backtest_threshold(test_df: pd.DataFrame, model: Any, scaler: StandardScaler, threshold: float = 0.50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # backtests trading strategy across decision probability threshold with transaction friction
    probs = model.predict_proba(scaler.transform(test_df[CLASSIFIER_FEATURES]))[:, 1]
    signals = (probs > threshold).astype(int)

    equity, position, entry_price, entry_equity, entry_time = INITIAL_CAPITAL, 0, 0.0, 0.0, None
    equity_curve, trade_log = [], []

    for i in range(len(test_df)):
        price, time, signal = float(test_df["close"].iloc[i]), test_df["timestamp"].iloc[i], signals[i]
        if position == 0 and signal == 1:
            entry_equity, entry_price, entry_time, position = equity * (1.0 - TOTAL_FRICTION), price, time, 1
        elif position == 1 and signal == 0:
            gross = (price / entry_price) - 1.0
            equity = entry_equity * (1.0 + gross) * (1.0 - TOTAL_FRICTION)
            net = (equity / (entry_equity / (1.0 - TOTAL_FRICTION))) - 1.0
            trade_log.append({
                "entry_time": entry_time, "exit_time": time, "entry_price": entry_price,
                "exit_price": price, "gross_return_pct": gross * 100.0, "net_return_pct": net * 100.0,
                "outcome": "WIN" if equity > entry_equity else "LOSS",
            })
            position = 0
        equity_curve.append({"timestamp": time, "equity": entry_equity * (price / entry_price) if position == 1 else equity, "position": position})

    if position == 1:
        last_price = float(test_df["close"].iloc[-1])
        gross = (last_price / entry_price) - 1.0
        equity = entry_equity * (1.0 + gross) * (1.0 - TOTAL_FRICTION)
        net = (equity / (entry_equity / (1.0 - TOTAL_FRICTION))) - 1.0
        trade_log.append({
            "entry_time": entry_time, "exit_time": test_df["timestamp"].iloc[-1], "entry_price": entry_price,
            "exit_price": last_price, "gross_return_pct": gross * 100.0, "net_return_pct": net * 100.0,
            "outcome": "WIN" if equity > entry_equity else "LOSS",
        })
        equity_curve[-1]["equity"], equity_curve[-1]["position"] = equity, 0

    return pd.DataFrame(equity_curve), pd.DataFrame(trade_log)


def compute_metrics(equity_curve_df: pd.DataFrame, trade_log_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    # calculates cumulative return, Sharpe ratio, and maximum drawdown
    final_eq = equity_curve_df["equity"].iloc[-1]
    returns = equity_curve_df["equity"].pct_change().fillna(0.0)
    running_max = equity_curve_df["equity"].cummax()
    return {
        "num_trades": len(trade_log_df),
        "win_rate_pct": float((trade_log_df["outcome"] == "WIN").mean() * 100.0) if len(trade_log_df) > 0 else 0.0,
        "total_return_pct": ((final_eq / INITIAL_CAPITAL) - 1.0) * 100.0,
        "buy_hold_return_pct": ((test_df["close"].iloc[-1] / test_df["close"].iloc[0]) - 1.0) * 100.0,
        "sharpe_ratio_annualized": float((returns.mean() / returns.std()) * np.sqrt(8760)) if returns.std() > 0 else 0.0,
        "max_drawdown_pct": float(((equity_curve_df["equity"] - running_max) / running_max).min() * 100.0),
    }


def run_threshold_sweep(data_path: Path = Path(BTCUSDT_1H_CSV), output_path: Path = Path(THRESHOLD_SWEEP), thresholds: List[float] = [0.50, 0.55, 0.60, 0.65, 0.70]) -> pd.DataFrame:
    # sweeps classification decision thresholds and saves comparison CSV
    df = create_features(merge_onchain(merge_fear_greed(load_and_clean_data(data_path), "timestamp"), "timestamp"), drop_na=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    model, scaler = train_classifier(df, split_idx)

    results = []
    for t in thresholds:
        ec, tl = run_backtest_threshold(test_df, model, scaler, threshold=t)
        m = compute_metrics(ec, tl, test_df)
        m["threshold"] = t
        results.append(m)

    out = pd.DataFrame(results)[["threshold", "num_trades", "win_rate_pct", "total_return_pct", "buy_hold_return_pct", "sharpe_ratio_annualized", "max_drawdown_pct"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


if __name__ == "__main__":
    run_threshold_sweep()
