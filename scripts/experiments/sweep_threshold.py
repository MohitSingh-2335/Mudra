# scripts/sweep_threshold.py
import pandas as pd
from src.backtesting import (
    train_classifier, run_backtest, compute_metrics,
    load_and_clean_data, create_features
)
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.agents.sentiment_agent import merge_sentiment
from config import SVC_FEATURES

from src.agents.fear_greed_agent import merge_fear_greed
df = create_features(merge_sentiment(merge_onchain(merge_fear_greed(load_and_clean_data('data/raw/BTCUSDT-1H.csv'))))).reset_index(drop=True)
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].reset_index(drop=True)
model, scaler = train_classifier(df, split_idx)

results = []
for t in [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.68, 0.70]:
    ec, tl = run_backtest(test_df, model, scaler, threshold=t)
    m = compute_metrics(ec, tl, test_df)
    m['threshold'] = t
    results.append(m)

out = pd.DataFrame(results)[['threshold', 'num_trades', 'win_rate_pct',
                              'total_return_pct', 'buy_hold_return_pct',
                              'sharpe_ratio_annualized', 'max_drawdown_pct']]
print(out.to_string(index=False))
out.to_csv('data/results/threshold_sweep.csv', index=False)