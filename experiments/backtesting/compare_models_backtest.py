# compare_models_backtest.py

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Guard project root resolution for standalone CLI execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.backtesting import run_backtest, compute_metrics, load_and_clean_data, create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.agents.sentiment_agent import merge_sentiment
from config import (
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH,
    BTCUSDT_1H_CSV
)

THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.68, 0.70]
MAX_SVM_ROWS = 15000

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVC": SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        device="cpu",
        random_state=42,
        eval_metric='logloss'
    ),
}


def train(
    df: pd.DataFrame,
    split_idx: int,
    model: Any,
    name: str
) -> Tuple[Any, StandardScaler]:
    """Trains a candidate baseline classifier with StandardScaler normalization."""
    feature_cols = [c for c in CLASSIFIER_FEATURES if c in df.columns]
    X_train = df[feature_cols].iloc[:split_idx]
    y_train = df['Target_Movement'].iloc[:split_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if name == "SVC" and len(X_train_scaled) > MAX_SVM_ROWS:
        X_train_scaled = X_train_scaled[-MAX_SVM_ROWS:]
        y_train = y_train.iloc[-MAX_SVM_ROWS:]
    model.fit(X_train_scaled, y_train)
    return model, scaler


def main() -> None:
    print("[INFO] Initializing Cross-Model Empirical Backtest Benchmark...")
    if os.path.exists(FEATURED_BTC_DATA_PATH):
        print(f"[INFO] Loading pre-computed featured dataset from {FEATURED_BTC_DATA_PATH}...")
        df = pd.read_csv(FEATURED_BTC_DATA_PATH, parse_dates=['timestamp'])
    else:
        print(f"[INFO] Computing features from raw feed {BTCUSDT_1H_CSV}...")
        raw_df = load_and_clean_data(BTCUSDT_1H_CSV)
        df = create_features(merge_sentiment(merge_onchain(merge_fear_greed(raw_df)))).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    all_results: List[Dict[str, Any]] = []
    for name, model in MODELS.items():
        print(f"[INFO] Training model baseline: {name}...")
        trained_model, scaler = train(df, split_idx, model, name)
        for t in THRESHOLDS:
            ec, tl = run_backtest(test_df, trained_model, scaler, threshold=t)
            m = compute_metrics(ec, tl, test_df)
            m['model'] = name
            m['threshold'] = t
            # Reconcile key aliases for reporting
            m['buy_hold_return_pct'] = m.get('bh_return_pct', 0.0)
            m['sharpe_ratio_annualized'] = m.get('sharpe_annualized', 0.0)
            all_results.append(m)

    out = pd.DataFrame(all_results)[
        ['model', 'threshold', 'num_trades', 'win_rate_pct',
         'total_return_pct', 'buy_hold_return_pct',
         'sharpe_ratio_annualized', 'max_drawdown_pct']
    ]
    print("\n" + "=" * 90)
    print("[BENCHMARK] CANDIDATE MODEL BACKTEST COMPARISON LEADERBOARD")
    print("=" * 90)
    print(out.to_string(index=False))
    print("=" * 90)

    Path('data/results').mkdir(parents=True, exist_ok=True)
    out.to_csv('data/results/model_comparison_backtest.csv', index=False)
    print("[SUCCESS] Backtest results serialized to data/results/model_comparison_backtest.csv")

    best = out.loc[out['total_return_pct'].idxmax()]
    print(f"\n[TOP PERFORMER] {best['model']} @ threshold={best['threshold']} "
          f"-> return {best['total_return_pct']:.2f}% (n={best['num_trades']} trades)")


if __name__ == '__main__':
    main()