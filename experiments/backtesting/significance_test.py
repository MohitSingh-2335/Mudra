# significance_test.py
#
# Tests whether the high-threshold backtest results (LogReg@0.68, XGB@0.70)
# are distinguishable from noise, given tiny trade counts.

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.backtesting import run_backtest, load_and_clean_data, create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.agents.sentiment_agent import merge_sentiment
from config import SVC_FEATURES
from src.agents.fear_greed_agent import merge_fear_greed

CONFIGS = [
    ("LogisticRegression", LogisticRegression(max_iter=1000), 0.68),
    ("XGBoost", XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                               random_state=42, eval_metric='logloss'), 0.70),
]


def binomial_test(win_rate_pct, n_trades):
    wins = round(win_rate_pct / 100 * n_trades)
    return stats.binomtest(wins, n_trades, p=0.5, alternative='greater')


def bootstrap_return_ci(trade_returns_pct, n_boot=10000):
    arr = np.array(trade_returns_pct)
    boot_means = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return np.mean(arr), lo, hi


def main():
    df = create_features(merge_sentiment(merge_onchain(merge_fear_greed(load_and_clean_data('data/raw/BTCUSDT-1H.csv'))))).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    for name, model, threshold in CONFIGS:
        X_train = df[SVC_FEATURES].iloc[:split_idx]
        y_train = df['Target_Movement'].iloc[:split_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)

        ec, tl = run_backtest(test_df, model, scaler, threshold=threshold)

        print(f"\n=== {name} @ threshold={threshold} ===")
        n = len(tl)
        if n == 0:
            print("No trades — skipping.")
            continue

        wins = (tl['outcome'] == 'WIN').sum()
        win_rate = wins / n * 100
        bt = binomial_test(win_rate, n)
        print(f"Trades: {n}, Wins: {wins}, Win rate: {win_rate:.1f}%")
        print(f"Binomial test vs p=0.5 (H1: win rate > 50%): p-value = {bt.pvalue:.4f}")
        print("  -> " + ("Significant at 5%" if bt.pvalue < 0.05 else "NOT significant — indistinguishable from a fair coin"))

        mean_ret, lo, hi = bootstrap_return_ci(tl['gross_return_pct'].values)
        print(f"Per-trade return: mean={mean_ret:.3f}%, 95% bootstrap CI=[{lo:.3f}%, {hi:.3f}%]")
        print("  -> " + ("CI excludes 0 — some evidence" if lo > 0 or hi < 0 else "CI includes 0 — not distinguishable from zero-edge"))


if __name__ == '__main__':
    main()