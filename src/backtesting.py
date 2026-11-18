import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH,
    INITIAL_CAPITAL_USD,
    REGRESSOR_FEATURES,
    SLIPPAGE_RATE as CONFIG_SLIPPAGE_RATE,
    TAKER_FEE_RATE as CONFIG_TAKER_FEE_RATE,
)
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.model_loader import load_production_models as loader_load_production_models

INITIAL_CAPITAL = INITIAL_CAPITAL_USD
TAKER_FEE_RATE = CONFIG_TAKER_FEE_RATE
SLIPPAGE_RATE = CONFIG_SLIPPAGE_RATE


def load_backtest_data(data_path: Union[str, Path] = FEATURED_BTC_DATA_PATH, split_ratio: float = 0.8) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Processed dataset not found at: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"[ERROR] Loaded dataset from '{data_path}' is empty.")
    return df.iloc[int(len(df) * split_ratio):].reset_index(drop=True)


def load_production_models() -> Tuple[Any, Any]:
    return loader_load_production_models()


def _simulate_from_signals(
    test_df: pd.DataFrame,
    signals: np.ndarray,
    initial_capital: float = INITIAL_CAPITAL,
    taker_fee_rate: float = TAKER_FEE_RATE,
    slippage_rate: float = SLIPPAGE_RATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    equity = initial_capital
    position = 0
    entry_equity, entry_price, entry_time = 0.0, 0.0, None
    equity_curve, trade_log = [], []
    total_friction = taker_fee_rate + slippage_rate

    def _close_position(curr_price, curr_time):
        nonlocal equity, position
        if position == 1:
            gross_ret = (curr_price / entry_price) - 1.0
        else:
            gross_ret = 1.0 - (curr_price / entry_price)
        equity = entry_equity * (1.0 + gross_ret) * (1.0 - total_friction)
        cap = entry_equity / (1.0 - total_friction)
        net_ret = (equity / cap) - 1.0
        trade_log.append({
            "entry_time": entry_time, "exit_time": curr_time, "entry_price": entry_price, "exit_price": curr_price,
            "gross_return_pct": round(gross_ret * 100.0, 3), "net_return_pct": round(net_ret * 100.0, 3),
            "pnl_usd": round(equity - cap, 2), "equity_after": round(equity, 2), "outcome": "WIN" if net_ret > 0 else "LOSS",
        })
        position = 0

    for i in range(len(test_df)):
        price = float(test_df["close"].iloc[i])
        time = test_df["timestamp"].iloc[i]
        sig = int(signals[i])

        if position == 0 and sig in (1, -1):
            entry_equity, entry_price, entry_time, position = equity * (1.0 - total_friction), price, time, sig
        elif (position == 1 and sig == 0) or (position == -1 and sig == 0):
            _close_position(price, time)

        if position == 1:
            mtm = entry_equity * (price / entry_price)
        elif position == -1:
            mtm = entry_equity * (1.0 + (entry_price - price) / entry_price)
        else:
            mtm = equity

        equity_curve.append({"timestamp": time, "close": price, "equity": mtm, "position": position, "signal": sig})

    if position != 0:
        _close_position(float(test_df["close"].iloc[-1]), test_df["timestamp"].iloc[-1])
        equity_curve[-1]["equity"] = equity
        equity_curve[-1]["position"] = 0

    return pd.DataFrame(equity_curve), pd.DataFrame(trade_log)


def run_backtest_simulation(
    test_df: pd.DataFrame,
    clf_model: Any,
    reg_model: Optional[Any] = None,
    strategy_mode: str = "fused",
    return_threshold: float = 0.002,
    taker_fee_rate: Optional[float] = None,
    slippage_rate: Optional[float] = None,
    clf_features: Optional[List[str]] = None,
    reg_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    f_clf = CLASSIFIER_FEATURES if clf_features is None else clf_features
    clf_preds = clf_model.predict(test_df[f_clf])
    if strategy_mode == "fused" and reg_model is not None:
        f_reg = REGRESSOR_FEATURES if reg_features is None else reg_features
        reg_preds = reg_model.predict(test_df[f_reg])
        signals = ((clf_preds == 1) & (reg_preds >= return_threshold)).astype(int)
    else:
        signals = (clf_preds == 1).astype(int)
    return _simulate_from_signals(test_df, signals, taker_fee_rate=taker_fee_rate or TAKER_FEE_RATE, slippage_rate=slippage_rate or SLIPPAGE_RATE)


def run_backtest(
    test_df: pd.DataFrame,
    model: Any,
    scaler: Optional[Any] = None,
    threshold: float = 0.5,
    taker_fee_rate: Optional[float] = None,
    slippage_rate: Optional[float] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = feature_cols or [c for c in CLASSIFIER_FEATURES if c in test_df.columns]
    X = scaler.transform(test_df[cols]) if scaler else test_df[cols]
    signals = (model.predict_proba(X)[:, 1] >= threshold).astype(int) if hasattr(model, "predict_proba") else model.predict(X)
    return _simulate_from_signals(test_df, signals, taker_fee_rate=taker_fee_rate or TAKER_FEE_RATE, slippage_rate=slippage_rate or SLIPPAGE_RATE)


def calculate_quant_metrics(equity_curve_df: pd.DataFrame, trade_log_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    final_eq = equity_curve_df["equity"].iloc[-1]
    total_ret = ((final_eq / INITIAL_CAPITAL) - 1.0) * 100.0
    start_p, end_p = test_df["close"].iloc[0], test_df["close"].iloc[-1]
    bh_ret = ((end_p / start_p) - 1.0) * 100.0
    bh_final = INITIAL_CAPITAL * (end_p / start_p)

    h_ret = equity_curve_df["equity"].pct_change().fillna(0.0)
    std_ret = h_ret.std()
    sharpe = (h_ret.mean() / std_ret) * np.sqrt(8760) if std_ret > 0 else 0.0

    neg_ret = h_ret[h_ret < 0.0]
    downside = neg_ret.std()
    sortino = (h_ret.mean() / downside) * np.sqrt(8760) if downside > 0 and not np.isnan(downside) else 0.0

    run_max = equity_curve_df["equity"].cummax()
    mdd = ((equity_curve_df["equity"] - run_max) / run_max).min() * 100.0

    bh_series = INITIAL_CAPITAL * (equity_curve_df["close"] / start_p)
    bh_mdd = ((bh_series - bh_series.cummax()) / bh_series.cummax()).min() * 100.0

    n_trades = len(trade_log_df)
    win_rate = (trade_log_df["outcome"] == "WIN").mean() * 100.0 if n_trades > 0 else 0.0
    gains = trade_log_df[trade_log_df["pnl_usd"] > 0]["pnl_usd"].sum() if n_trades > 0 else 0.0
    losses = abs(trade_log_df[trade_log_df["pnl_usd"] < 0]["pnl_usd"].sum()) if n_trades > 0 else 0.0
    profit_factor = (gains / losses) if losses > 0 else (gains if gains > 0 else 1.0)

    return {
        "initial_capital": INITIAL_CAPITAL, "final_equity": round(final_eq, 2), "total_return_pct": round(total_ret, 2),
        "bh_final_equity": round(bh_final, 2), "bh_return_pct": round(bh_ret, 2), "sharpe_annualized": round(sharpe, 3),
        "sortino_annualized": round(sortino, 3), "max_drawdown_pct": round(mdd, 2), "bh_max_drawdown_pct": round(bh_mdd, 2),
        "calmar_ratio": round((total_ret / abs(mdd)) if mdd != 0 else 0.0, 3), "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2), "num_trades": n_trades,
        "avg_trade_return_pct": round(trade_log_df["net_return_pct"].mean(), 3) if n_trades > 0 else 0.0,
        "exposure_pct": round((equity_curve_df["position"] != 0).mean() * 100.0, 2),
    }


compute_metrics = calculate_quant_metrics


def plot_backtest_comparison(equity_curve_df: pd.DataFrame, output_path: Union[str, Path] = "artifacts/backtest_performance.png") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    start_p = equity_curve_df["close"].iloc[0]
    bh_eq = INITIAL_CAPITAL * (equity_curve_df["close"] / start_p)
    s_max = equity_curve_df["equity"].cummax()
    s_dd = ((equity_curve_df["equity"] - s_max) / s_max) * 100.0
    bh_dd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()) * 100.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    ax1.plot(equity_curve_df["timestamp"], equity_curve_df["equity"], label="Mudra Strategy", color="#00C805", linewidth=2)
    ax1.plot(equity_curve_df["timestamp"], bh_eq, label="Buy & Hold BTC", color="#F7931A", linewidth=1.5, alpha=0.85)
    ax1.set_ylabel("Portfolio Equity ($ USD)", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.fill_between(equity_curve_df["timestamp"], s_dd, 0, color="#00C805", alpha=0.3)
    ax2.fill_between(equity_curve_df["timestamp"], bh_dd, 0, color="#F7931A", alpha=0.2)
    ax2.set_ylabel("Drawdown (%)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    test_df = load_backtest_data()
    reg_model, clf_model = load_production_models()
    eq_sel, trades_sel = run_backtest_simulation(test_df, clf_model, reg_model=reg_model, strategy_mode="fused", return_threshold=0.0005)
    metrics = calculate_quant_metrics(eq_sel, trades_sel, test_df)
    print(f"[BACKTEST] Total Return: {metrics['total_return_pct']}% | Sharpe: {metrics['sharpe_annualized']} | Trades: {metrics['num_trades']}")


if __name__ == "__main__":
    main()
