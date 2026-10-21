# src/backtesting.py - Mudra Quantitative Strategy Backtesting Engine

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to sys.path so config can be imported from any working directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure stdout handles UTF-8 cleanly on Windows PowerShell
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless raster backend for cross-platform stability
import matplotlib.pyplot as plt

# Re-exports for backwards compatibility across empirical research experiments
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.model_loader import load_production_models as loader_load_production_models

from config import (
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    FEATURED_BTC_DATA_PATH,
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES,
    INITIAL_CAPITAL_USD,
    TAKER_FEE_RATE as CONFIG_TAKER_FEE_RATE,
    SLIPPAGE_RATE as CONFIG_SLIPPAGE_RATE,
)

# Realistic Trading Parameters (Canonical Defaults from config.py)
INITIAL_CAPITAL = INITIAL_CAPITAL_USD  # Starting portfolio cash in USDT
TAKER_FEE_RATE = CONFIG_TAKER_FEE_RATE  # 0.05% Binance VIP-0 taker fee per side (decimal rate: 0.0005)
SLIPPAGE_RATE = CONFIG_SLIPPAGE_RATE   # 0.02% estimated execution slippage per fill (decimal rate: 0.0002)


def load_backtest_data(
    data_path: Union[str, Path] = FEATURED_BTC_DATA_PATH,
    split_ratio: float = 0.8
) -> pd.DataFrame:
    """Loads feature-engineered data and extracts the chronological unseen test split."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Processed dataset not found at: {data_path}")

    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    if df.empty:
        raise ValueError(f"[ERROR] Loaded dataset from '{data_path}' is empty.")

    df = df.sort_values('timestamp').reset_index(drop=True)

    split_idx = int(len(df) * split_ratio)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return test_df


def load_production_models() -> Tuple[Any, Any]:
    """Loads verified serialized Optuna-tuned models and forces CPU execution."""
    return loader_load_production_models()


def _simulate_from_signals(
    test_df: pd.DataFrame,
    signals: np.ndarray,
    initial_capital: float = INITIAL_CAPITAL,
    taker_fee_rate: float = TAKER_FEE_RATE,
    slippage_rate: float = SLIPPAGE_RATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates mark-to-market hourly portfolio execution given binary position signals.
    """
    equity = initial_capital
    position = 0  # 0: 100% Cash (USDT), 1: 100% Long (BTC)
    entry_equity = 0.0
    entry_price = 0.0
    entry_time = None

    equity_curve = []
    trade_log = []

    total_friction = taker_fee_rate + slippage_rate

    for i in range(len(test_df)):
        price = float(test_df['close'].iloc[i])
        time = test_df['timestamp'].iloc[i]
        signal = int(signals[i])

        # 1. Buy Signal (Enter Long from Cash)
        if position == 0 and signal == 1:
            entry_equity = equity * (1.0 - total_friction)
            entry_price = price
            entry_time = time
            position = 1

        # 2. Sell Signal (Exit Long to Cash)
        elif position == 1 and signal == 0:
            gross_return = (price / entry_price) - 1.0
            equity = entry_equity * (1.0 + gross_return) * (1.0 - total_friction)
            net_return = (equity / (entry_equity / (1.0 - total_friction))) - 1.0
            profit_loss = equity - (entry_equity / (1.0 - total_friction))

            trade_log.append({
                'entry_time': entry_time,
                'exit_time': time,
                'entry_price': entry_price,
                'exit_price': price,
                'gross_return_pct': round(gross_return * 100.0, 3),
                'net_return_pct': round(net_return * 100.0, 3),
                'pnl_usd': round(profit_loss, 2),
                'equity_after': round(equity, 2),
                'outcome': 'WIN' if net_return > 0 else 'LOSS'
            })
            position = 0

        # Mark-to-market equity for the current hour
        mtm_equity = entry_equity * (price / entry_price) if position == 1 else equity
        equity_curve.append({
            'timestamp': time,
            'close': price,
            'equity': mtm_equity,
            'position': position,
            'signal': signal
        })

    # Liquidate open position at the final candle
    if position == 1:
        last_price = float(test_df['close'].iloc[-1])
        gross_return = (last_price / entry_price) - 1.0
        equity = entry_equity * (1.0 + gross_return) * (1.0 - total_friction)
        net_return = (equity / (entry_equity / (1.0 - total_friction))) - 1.0
        profit_loss = equity - (entry_equity / (1.0 - total_friction))

        trade_log.append({
            'entry_time': entry_time,
            'exit_time': test_df['timestamp'].iloc[-1],
            'entry_price': entry_price,
            'exit_price': last_price,
            'gross_return_pct': round(gross_return * 100.0, 3),
            'net_return_pct': round(net_return * 100.0, 3),
            'pnl_usd': round(profit_loss, 2),
            'equity_after': round(equity, 2),
            'outcome': 'WIN' if net_return > 0 else 'LOSS'
        })
        equity_curve[-1]['equity'] = equity
        equity_curve[-1]['position'] = 0

    equity_curve_df = pd.DataFrame(equity_curve)
    trade_log_df = pd.DataFrame(trade_log)
    return equity_curve_df, trade_log_df


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
    """
    Simulates hourly trade execution across the holdout test period.
    
    Strategies:
      - 'classifier': Long when classifier predicts 1 (Upward), Flat otherwise.
      - 'fused': Long when classifier predicts 1 AND regressor return >= return_threshold.
    """
    feature_clf = CLASSIFIER_FEATURES if clf_features is None else clf_features
    X_clf = test_df[feature_clf]
    clf_preds = clf_model.predict(X_clf)

    if strategy_mode == "fused" and reg_model is not None:
        feature_reg = REGRESSOR_FEATURES if reg_features is None else reg_features
        X_reg = test_df[feature_reg]
        reg_preds = reg_model.predict(X_reg)
        signals = ((clf_preds == 1) & (reg_preds >= return_threshold)).astype(int)
    else:
        signals = (clf_preds == 1).astype(int)

    fee = TAKER_FEE_RATE if taker_fee_rate is None else taker_fee_rate
    slip = SLIPPAGE_RATE if slippage_rate is None else slippage_rate
    return _simulate_from_signals(test_df, signals, taker_fee_rate=fee, slippage_rate=slip)


def run_backtest(
    test_df: pd.DataFrame,
    model: Any,
    scaler: Optional[Any] = None,
    threshold: float = 0.5,
    taker_fee_rate: Optional[float] = None,
    slippage_rate: Optional[float] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backwards-compatible interface supporting empirical research scripts
    with optional feature scalers and probability thresholds.
    """
    if feature_cols is None:
        feature_cols = [c for c in CLASSIFIER_FEATURES if c in test_df.columns]
    X = test_df[feature_cols]

    if scaler is not None:
        X_proc = scaler.transform(X)
    else:
        X_proc = X

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_proc)[:, 1]
        signals = (probs >= threshold).astype(int)
    else:
        signals = model.predict(X_proc)

    fee = TAKER_FEE_RATE if taker_fee_rate is None else taker_fee_rate
    slip = SLIPPAGE_RATE if slippage_rate is None else slippage_rate
    return _simulate_from_signals(test_df, signals, taker_fee_rate=fee, slippage_rate=slip)


def calculate_quant_metrics(
    equity_curve_df: pd.DataFrame,
    trade_log_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """Computes comprehensive quantitative risk and return metrics."""
    final_equity = equity_curve_df['equity'].iloc[-1]
    total_return_pct = ((final_equity / INITIAL_CAPITAL) - 1.0) * 100.0

    # Benchmark: Buy & Hold BTC
    start_btc_price = test_df['close'].iloc[0]
    end_btc_price = test_df['close'].iloc[-1]
    bh_return_pct = ((end_btc_price / start_btc_price) - 1.0) * 100.0
    bh_final_equity = INITIAL_CAPITAL * (end_btc_price / start_btc_price)

    # Hourly returns for volatility and risk ratios
    hourly_returns = equity_curve_df['equity'].pct_change().fillna(0.0)
    std_return = hourly_returns.std()

    # Annualized Sharpe (8760 trading hours per 365-day crypto year)
    sharpe = (hourly_returns.mean() / std_return) * np.sqrt(8760) if std_return > 0 else 0.0

    # Annualized Sortino (Penalizes only downside risk)
    negative_returns = hourly_returns[hourly_returns < 0.0]
    downside_std = negative_returns.std()
    sortino = (hourly_returns.mean() / downside_std) * np.sqrt(8760) if downside_std > 0 and not np.isnan(downside_std) else 0.0

    # Maximum Drawdown (MDD)
    running_max = equity_curve_df['equity'].cummax()
    drawdown = (equity_curve_df['equity'] - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100.0

    # Benchmark Drawdown
    bh_equity_series = INITIAL_CAPITAL * (equity_curve_df['close'] / start_btc_price)
    bh_running_max = bh_equity_series.cummax()
    bh_drawdown = (bh_equity_series - bh_running_max) / bh_running_max
    bh_max_drawdown_pct = bh_drawdown.min() * 100.0

    # Calmar Ratio
    calmar = (total_return_pct / abs(max_drawdown_pct)) if max_drawdown_pct != 0 else 0.0

    # Trade-Level Statistics
    num_trades = len(trade_log_df)
    if num_trades > 0:
        win_rate = (trade_log_df['outcome'] == 'WIN').mean() * 100.0
        gains = trade_log_df[trade_log_df['pnl_usd'] > 0]['pnl_usd'].sum()
        losses = abs(trade_log_df[trade_log_df['pnl_usd'] < 0]['pnl_usd'].sum())
        profit_factor = (gains / losses) if losses > 0 else (gains if gains > 0 else 1.0)
        avg_trade_return = trade_log_df['net_return_pct'].mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_return = 0.0

    exposure_pct = (equity_curve_df['position'] == 1).mean() * 100.0

    return {
        'initial_capital': INITIAL_CAPITAL,
        'final_equity': round(final_equity, 2),
        'total_return_pct': round(total_return_pct, 2),
        'bh_final_equity': round(bh_final_equity, 2),
        'bh_return_pct': round(bh_return_pct, 2),
        'sharpe_annualized': round(sharpe, 3),
        'sortino_annualized': round(sortino, 3),
        'max_drawdown_pct': round(max_drawdown_pct, 2),
        'bh_max_drawdown_pct': round(bh_max_drawdown_pct, 2),
        'calmar_ratio': round(calmar, 3),
        'win_rate_pct': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'num_trades': num_trades,
        'avg_trade_return_pct': round(avg_trade_return, 3),
        'exposure_pct': round(exposure_pct, 2)
    }


# Alias for backwards compatibility with research experiments
compute_metrics = calculate_quant_metrics


def plot_backtest_comparison(
    equity_curve_df: pd.DataFrame,
    output_path: Union[str, Path] = 'artifacts/backtest_performance.png'
) -> None:
    """Generates a professional 2-panel quantitative performance visualization."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    start_price = equity_curve_df['close'].iloc[0]
    bh_equity = INITIAL_CAPITAL * (equity_curve_df['close'] / start_price)

    strategy_max = equity_curve_df['equity'].cummax()
    strategy_dd = ((equity_curve_df['equity'] - strategy_max) / strategy_max) * 100.0

    bh_max = bh_equity.cummax()
    bh_dd = ((bh_equity - bh_max) / bh_max) * 100.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    plt.subplots_adjust(hspace=0.08)

    # Panel 1: Equity Curve Comparison
    ax1.plot(equity_curve_df['timestamp'], equity_curve_df['equity'], label='Mudra Fused Strategy', color='#00C805', linewidth=2)
    ax1.plot(equity_curve_df['timestamp'], bh_equity, label='Buy & Hold BTC Benchmark', color='#F7931A', linewidth=1.5, alpha=0.85)
    ax1.set_ylabel('Portfolio Equity ($ USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Mudra Strategy vs. Buy & Hold BTC — Out-of-Sample Backtest', fontsize=14, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11, frameon=True)

    # Panel 2: Underwater Drawdown Chart
    ax2.fill_between(equity_curve_df['timestamp'], strategy_dd, 0, color='#00C805', alpha=0.3, label='Mudra Drawdown')
    ax2.plot(equity_curve_df['timestamp'], strategy_dd, color='#00C805', linewidth=1)
    ax2.fill_between(equity_curve_df['timestamp'], bh_dd, 0, color='#F7931A', alpha=0.2, label='BTC Drawdown')
    ax2.plot(equity_curve_df['timestamp'], bh_dd, color='#F7931A', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='lower left', fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Performance visualization saved to: {output_path}")


def main() -> None:
    print("[INFO] Initializing Mudra Quantitative Backtesting Engine...")
    test_df = load_backtest_data()
    reg_model, clf_model = load_production_models()

    print(f"[INFO] Evaluation Period : {test_df['timestamp'].iloc[0]} -> {test_df['timestamp'].iloc[-1]} ({len(test_df):,} hours)")
    print(f"[INFO] Initial Capital   : ${INITIAL_CAPITAL:,.2f}")
    print(f"[INFO] Exchange Friction : {TAKER_FEE_RATE*100:.2f}% Taker Fee + {SLIPPAGE_RATE*100:.2f}% Slippage per execution\n")

    # 1. Run Pure Classifier Strategy
    print("[INFO] Simulating Strategy 1: Pure Classifier (Unconstrained)...")
    eq_clf, trades_clf = run_backtest_simulation(test_df, clf_model, strategy_mode="classifier")
    m_clf = calculate_quant_metrics(eq_clf, trades_clf, test_df)

    # 2. Run Selective Fused Alpha Strategy (0.05% threshold)
    print("[INFO] Simulating Strategy 2: Mudra Fused Alpha (Selective 0.05% Filter)...")
    eq_sel, trades_sel = run_backtest_simulation(test_df, clf_model, reg_model=reg_model, strategy_mode="fused", return_threshold=0.0005)
    m_sel = calculate_quant_metrics(eq_sel, trades_sel, test_df)

    # 3. Run Strict Capital Preservation Strategy (0.055% threshold)
    print("[INFO] Simulating Strategy 3: Mudra Capital Preservation (Safe Haven Cash)...")
    eq_cash, trades_cash = run_backtest_simulation(test_df, clf_model, reg_model=reg_model, strategy_mode="fused", return_threshold=0.00055)
    m_cash = calculate_quant_metrics(eq_cash, trades_cash, test_df)

    # Print Institutional Performance Comparison
    print("\n" + "=" * 96)
    print("[BENCHMARK] OUT-OF-SAMPLE QUANTITATIVE PERFORMANCE BENCHMARK (7,000 HOURS)")
    print("=" * 96)
    print(f"{'Metric':<25} | {'Buy & Hold BTC':<16} | {'Pure Classifier':<16} | {'Mudra Fused (0.05%)':<18} | {'Mudra Cash (0.055%)':<18}")
    print("-" * 96)
    print(f"{'Total Return (%)':<25} | {m_sel['bh_return_pct']:>15}% | {m_clf['total_return_pct']:>15}% | {m_sel['total_return_pct']:>17}% | {m_cash['total_return_pct']:>17}%")
    print(f"{'Final Equity ($)':<25} | ${m_sel['bh_final_equity']:>14,.2f} | ${m_clf['final_equity']:>14,.2f} | ${m_sel['final_equity']:>16,.2f} | ${m_cash['final_equity']:>16,.2f}")
    print(f"{'Max Drawdown (MDD)':<25} | {m_sel['bh_max_drawdown_pct']:>15}% | {m_clf['max_drawdown_pct']:>15}% | {m_sel['max_drawdown_pct']:>17}% | {m_cash['max_drawdown_pct']:>17}%")
    print(f"{'Alpha vs BTC (pp)':<25} | {'0.00%':>16} | {m_clf['total_return_pct'] - m_sel['bh_return_pct']:>15.2f}% | {m_sel['total_return_pct'] - m_sel['bh_return_pct']:>17.2f}% | {m_cash['total_return_pct'] - m_sel['bh_return_pct']:>17.2f}%")
    print(f"{'Total Closed Trades':<25} | {'1':>16} | {m_clf['num_trades']:>16} | {m_sel['num_trades']:>18} | {m_cash['num_trades']:>18}")
    print(f"{'Market Exposure (%)':<25} | {'100.0%':>16} | {m_clf['exposure_pct']:>15}% | {m_sel['exposure_pct']:>17}% | {m_cash['exposure_pct']:>18}%")
    print("=" * 96)

    # Save artifacts
    Path('data/results').mkdir(parents=True, exist_ok=True)
    trades_sel.to_csv('data/results/backtest_trade_log.csv', index=False)
    eq_sel.to_csv('data/results/backtest_equity_curve.csv', index=False)
    plot_backtest_comparison(eq_cash, output_path='artifacts/backtest_performance.png')
    print("[SUCCESS] Trade logs and equity curves saved to data/results/")


if __name__ == '__main__':
    main()
