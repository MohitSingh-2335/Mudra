# compare_models_backtest.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.backtesting import run_backtest, compute_metrics, load_and_clean_data, create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.agents.sentiment_agent import merge_sentiment
from config import SVC_FEATURES
from src.agents.fear_greed_agent import merge_fear_greed

THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.68, 0.70]
MAX_SVM_ROWS = 15000

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVC": SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                              random_state=42, eval_metric='logloss'),
}


def train(df, split_idx, model, name):
    X_train = df[SVC_FEATURES].iloc[:split_idx]
    y_train = df['Target_Movement'].iloc[:split_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if name == "SVC" and len(X_train_scaled) > MAX_SVM_ROWS:
        X_train_scaled = X_train_scaled[-MAX_SVM_ROWS:]
        y_train = y_train.iloc[-MAX_SVM_ROWS:]
    model.fit(X_train_scaled, y_train)
    return model, scaler


def main():
    df = create_features(merge_sentiment(merge_onchain(merge_fear_greed(load_and_clean_data('data/raw/BTCUSDT-1H.csv'))))).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    all_results = []
    for name, model in MODELS.items():
        print(f"\nTraining {name}...")
        trained_model, scaler = train(df, split_idx, model, name)
        for t in THRESHOLDS:
            ec, tl = run_backtest(test_df, trained_model, scaler, threshold=t)
            m = compute_metrics(ec, tl, test_df)
            m['model'] = name
            m['threshold'] = t
            all_results.append(m)

    out = pd.DataFrame(all_results)[
        ['model', 'threshold', 'num_trades', 'win_rate_pct',
         'total_return_pct', 'buy_hold_return_pct',
         'sharpe_ratio_annualized', 'max_drawdown_pct']
    ]
    print("\n" + out.to_string(index=False))
    out.to_csv('data/results/model_comparison_backtest.csv', index=False)

    best = out.loc[out['total_return_pct'].idxmax()]
    print(f"\nBest: {best['model']} @ threshold={best['threshold']} "
          f"-> return {best['total_return_pct']:.2f}% (n={best['num_trades']} trades)")


if __name__ == '__main__':
    main()