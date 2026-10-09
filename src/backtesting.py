# src/backtesting.py
#
# Event-driven backtest: trains a classifier on the training split, then
# walks forward through the held-out test period simulating a simple
# long/flat strategy on its direction signal. Includes trading fees, since
# a backtest without them is misleading, not just simplified.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.agents.onchain_agent import merge_onchain
from src.agents.sentiment_agent import merge_sentiment
from config import SVC_FEATURES
from src.agents.fear_greed_agent import merge_fear_greed
from config import BTCUSDT_1H_CSV

INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.001  # 0.1% per side (Binance taker fee) — 0.2% round trip


def train_classifier(df, split_idx):
    X_train = df[SVC_FEATURES].iloc[:split_idx]
    y_train = df['Target_Movement'].iloc[:split_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def run_backtest(test_df, model, scaler, threshold=0.5):
    X_test_scaled = scaler.transform(test_df[SVC_FEATURES])
    probs = model.predict_proba(X_test_scaled)[:, 1]
    signals = (probs > threshold).astype(int)

    equity = INITIAL_CAPITAL
    position = 0
    entry_price = entry_equity = entry_time = None
    equity_curve, trade_log = [], []

    for i in range(len(test_df)):
        price = test_df['close'].iloc[i]
        time = test_df['timestamp'].iloc[i]
        signal = signals[i]

        if position == 0 and signal == 1:  # enter long
            entry_equity = equity * (1 - FEE_RATE)
            entry_price = price
            entry_time = time
            position = 1

        elif position == 1 and signal == 0:  # exit long
            gross_return = (price / entry_price) - 1
            equity = entry_equity * (1 + gross_return) * (1 - FEE_RATE)
            trade_log.append({
                'entry_time': entry_time, 'exit_time': time,
                'entry_price': entry_price, 'exit_price': price,
                'gross_return_pct': gross_return * 100,
                'equity_after': equity,
                'outcome': 'WIN' if equity > entry_equity else 'LOSS',
            })
            position = 0

        mtm_equity = entry_equity * (price / entry_price) if position == 1 else equity
        equity_curve.append({'timestamp': time, 'equity': mtm_equity, 'position': position})

    if position == 1:  # force-close any open position at the end
        last_price = test_df['close'].iloc[-1]
        gross_return = (last_price / entry_price) - 1
        equity = entry_equity * (1 + gross_return) * (1 - FEE_RATE)
        trade_log.append({
            'entry_time': entry_time, 'exit_time': test_df['timestamp'].iloc[-1],
            'entry_price': entry_price, 'exit_price': last_price,
            'gross_return_pct': gross_return * 100,
            'equity_after': equity,
            'outcome': 'WIN' if equity > entry_equity else 'LOSS',
            'note': 'force-closed at end of test period',
        })
        equity_curve[-1]['equity'] = equity
        equity_curve[-1]['position'] = 0

    return pd.DataFrame(equity_curve), pd.DataFrame(trade_log)


def compute_metrics(equity_curve_df, trade_log_df, test_df):
    final_equity = equity_curve_df['equity'].iloc[-1]
    total_return_pct = (final_equity / INITIAL_CAPITAL - 1) * 100
    bh_return_pct = (test_df['close'].iloc[-1] / test_df['close'].iloc[0] - 1) * 100

    hourly_returns = equity_curve_df['equity'].pct_change().fillna(0)
    sharpe = (hourly_returns.mean() / hourly_returns.std()) * np.sqrt(24 * 365) if hourly_returns.std() > 0 else 0.0

    running_max = equity_curve_df['equity'].cummax()
    drawdown = (equity_curve_df['equity'] - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100

    win_rate = (trade_log_df['outcome'] == 'WIN').mean() * 100 if len(trade_log_df) > 0 else 0.0

    return {
        'total_return_pct': total_return_pct, 'buy_hold_return_pct': bh_return_pct,
        'sharpe_ratio_annualized': sharpe, 'max_drawdown_pct': max_drawdown_pct,
        'win_rate_pct': win_rate, 'num_trades': len(trade_log_df), 'final_equity': final_equity,
    }


def main(threshold=0.5):
    print("Loading data and training classifier...")
    df = create_features(merge_sentiment(merge_onchain(merge_fear_greed(load_and_clean_data(BTCUSDT_1H_CSV))))).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    model, scaler = train_classifier(df, split_idx)

    print(f"Running backtest on held-out test data (threshold={threshold})...")
    equity_curve_df, trade_log_df = run_backtest(test_df, model, scaler, threshold=threshold)
    m = compute_metrics(equity_curve_df, trade_log_df, test_df)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Test period        : {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}")
    print(f"Initial capital    : ${INITIAL_CAPITAL:,.2f}")
    print(f"Final equity       : ${m['final_equity']:,.2f}")
    print(f"Strategy return    : {m['total_return_pct']:.2f}%")
    print(f"Buy & Hold return  : {m['buy_hold_return_pct']:.2f}%")
    print(f"Sharpe (annualized): {m['sharpe_ratio_annualized']:.3f}")
    print(f"Max drawdown       : {m['max_drawdown_pct']:.2f}%")
    print(f"Win rate           : {m['win_rate_pct']:.2f}%  ({m['num_trades']} trades)")
    print(f"Fee rate           : {FEE_RATE*100:.2f}% per side")
    print("\n✅ Beat Buy & Hold" if m['total_return_pct'] > m['buy_hold_return_pct'] else "\n⚠️  Did NOT beat Buy & Hold")

    trade_log_df.to_csv('data/results/backtest_trade_log.csv', index=False)
    equity_curve_df.to_csv('data/results/backtest_equity_curve.csv', index=False)
    print("Saved: data/results/backtest_trade_log.csv, data/results/backtest_equity_curve.csv")


if __name__ == '__main__':
    import sys
    thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    main(threshold=thresh)