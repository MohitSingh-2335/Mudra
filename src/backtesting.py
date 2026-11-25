import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import (
    BTCUSDT_1H_CSV,
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH,
    INITIAL_CAPITAL_USD,
    REGRESSOR_FEATURES,
    SLIPPAGE_RATE,
    TAKER_FEE_RATE,
    THRESHOLD_SWEEP,
)
from src.data_preprocessing import load_and_clean_data, merge_fear_greed, merge_onchain
from src.feature_engineering import create_features
from src.model_loader import load_production_models


def load_backtest_data(data_path: Union[str, Path] = FEATURED_BTC_DATA_PATH, split_ratio: float = 0.8) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Loaded dataset from '{data_path}' is empty.")
    return df.iloc[int(len(df) * split_ratio):].reset_index(drop=True)


def _simulate_from_signals(
    test_df: pd.DataFrame,
    signals: np.ndarray,
    initial_capital: float = INITIAL_CAPITAL_USD,
    taker_fee_rate: float = TAKER_FEE_RATE,
    slippage_rate: float = SLIPPAGE_RATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Simulates sequential spot entry/exit with per-side taker fee + slippage
    equity, position = initial_capital, 0
    entry_equity, entry_price, entry_time = 0.0, 0.0, None
    equity_curve, trade_log = [], []
    total_friction = taker_fee_rate + slippage_rate

    def _close_position(curr_price: float, curr_time: Any) -> None:
        nonlocal equity, position
        gross_ret = (curr_price / entry_price) - 1.0 if position == 1 else 1.0 - (curr_price / entry_price)
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
        price, ts, sig = float(test_df["close"].iloc[i]), test_df["timestamp"].iloc[i], int(signals[i])
        if position == 0 and sig in (1, -1):
            entry_equity, entry_price, entry_time, position = equity * (1.0 - total_friction), price, ts, sig
        elif position != 0 and sig == 0:
            _close_position(price, ts)

        mtm = (entry_equity * (price / entry_price)) if position == 1 else ((entry_equity * (1.0 + (entry_price - price) / entry_price)) if position == -1 else equity)
        equity_curve.append({"timestamp": ts, "close": price, "equity": mtm, "position": position, "signal": sig})

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
    return _simulate_from_signals(
        test_df,
        signals,
        taker_fee_rate=TAKER_FEE_RATE if taker_fee_rate is None else taker_fee_rate,
        slippage_rate=SLIPPAGE_RATE if slippage_rate is None else slippage_rate,
    )


def calculate_quant_metrics(equity_curve_df: pd.DataFrame, trade_log_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    final_eq = equity_curve_df["equity"].iloc[-1]
    total_ret = ((final_eq / INITIAL_CAPITAL_USD) - 1.0) * 100.0
    start_p, end_p = test_df["close"].iloc[0], test_df["close"].iloc[-1]
    bh_ret = ((end_p / start_p) - 1.0) * 100.0

    h_ret = equity_curve_df["equity"].pct_change().fillna(0.0)
    std_ret = h_ret.std()
    sharpe = (h_ret.mean() / std_ret) * np.sqrt(8760) if std_ret > 0 else 0.0

    neg_ret = h_ret[h_ret < 0.0]
    downside = neg_ret.std()
    sortino = (h_ret.mean() / downside) * np.sqrt(8760) if downside > 0 and not np.isnan(downside) else 0.0

    run_max = equity_curve_df["equity"].cummax()
    mdd = ((equity_curve_df["equity"] - run_max) / run_max).min() * 100.0
    bh_series = INITIAL_CAPITAL_USD * (equity_curve_df["close"] / start_p)
    bh_mdd = ((bh_series - bh_series.cummax()) / bh_series.cummax()).min() * 100.0

    n_trades = len(trade_log_df)
    win_rate = (trade_log_df["outcome"] == "WIN").mean() * 100.0 if n_trades > 0 else 0.0
    gains = trade_log_df[trade_log_df["pnl_usd"] > 0]["pnl_usd"].sum() if n_trades > 0 else 0.0
    losses = abs(trade_log_df[trade_log_df["pnl_usd"] < 0]["pnl_usd"].sum()) if n_trades > 0 else 0.0
    profit_factor = (gains / losses) if losses > 0 else (gains if gains > 0 else 1.0)

    return {
        "initial_capital": INITIAL_CAPITAL_USD, "final_equity": round(final_eq, 2), "total_return_pct": round(total_ret, 2),
        "bh_final_equity": round(INITIAL_CAPITAL_USD * (end_p / start_p), 2), "bh_return_pct": round(bh_ret, 2),
        "sharpe_annualized": round(sharpe, 3), "sortino_annualized": round(sortino, 3),
        "max_drawdown_pct": round(mdd, 2), "bh_max_drawdown_pct": round(bh_mdd, 2),
        "calmar_ratio": round((total_ret / abs(mdd)) if mdd != 0 else 0.0, 3), "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2), "num_trades": n_trades,
        "avg_trade_return_pct": round(trade_log_df["net_return_pct"].mean(), 3) if n_trades > 0 else 0.0,
        "exposure_pct": round((equity_curve_df["position"] != 0).mean() * 100.0, 2),
    }


def train_classifier(df: pd.DataFrame, split_idx: int) -> Tuple[LogisticRegression, StandardScaler]:
    X_train = df[CLASSIFIER_FEATURES].iloc[:split_idx]
    y_train = df["Target_Movement"].iloc[:split_idx]
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(scaler.fit_transform(X_train), y_train)
    return model, scaler


def run_backtest_threshold(test_df: pd.DataFrame, model: Any, scaler: StandardScaler, threshold: float = 0.50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    probs = model.predict_proba(scaler.transform(test_df[CLASSIFIER_FEATURES]))[:, 1]
    signals = (probs > threshold).astype(int)
    return _simulate_from_signals(test_df, signals)


def run_threshold_sweep(
    data_path: Path = Path(BTCUSDT_1H_CSV),
    output_path: Path = Path(THRESHOLD_SWEEP),
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    sweep_vals = [0.50, 0.55, 0.60, 0.65, 0.70] if thresholds is None else thresholds
    df = create_features(merge_onchain(merge_fear_greed(load_and_clean_data(data_path), "timestamp"), "timestamp"), drop_na=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    model, scaler = train_classifier(df, split_idx)

    results = []
    for t in sweep_vals:
        ec, tl = run_backtest_threshold(test_df, model, scaler, threshold=t)
        m = calculate_quant_metrics(ec, tl, test_df)
        results.append({
            "threshold": t, "num_trades": m["num_trades"], "win_rate_pct": m["win_rate_pct"],
            "total_return_pct": m["total_return_pct"], "buy_hold_return_pct": m["bh_return_pct"],
            "sharpe_ratio_annualized": m["sharpe_annualized"], "max_drawdown_pct": m["max_drawdown_pct"],
        })

    out = pd.DataFrame(results)[["threshold", "num_trades", "win_rate_pct", "total_return_pct", "buy_hold_return_pct", "sharpe_ratio_annualized", "max_drawdown_pct"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    test_df = load_backtest_data()
    reg_model, clf_model = load_production_models()
    eq_sel, trades_sel = run_backtest_simulation(test_df, clf_model, reg_model=reg_model, strategy_mode="fused", return_threshold=0.0005)
    metrics = calculate_quant_metrics(eq_sel, trades_sel, test_df)
    print(f"Total Return: {metrics['total_return_pct']}% | Sharpe: {metrics['sharpe_annualized']} | Trades: {metrics['num_trades']}")


if __name__ == "__main__":
    main()
