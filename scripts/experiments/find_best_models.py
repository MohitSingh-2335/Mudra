# find_best_models.py
#
# Compares several model families for BOTH prediction tasks, all trained on
# the exact same chronological split and the exact same (fixed) feature sets
# from config.py — so this isolates "which algorithm works best" rather than
# "which had better features." No hyperparameter tuning here on purpose —
# that comes next, only for whichever model wins this comparison.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from config import XGB_FEATURES, SVC_FEATURES


def chronological_split(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    return (
        X.iloc[:split_idx], X.iloc[split_idx:],
        y.iloc[:split_idx], y.iloc[split_idx:],
        split_idx,
    )


def compare_regressors(df):
    print("\n" + "=" * 60)
    print("REGRESSION MODELS — predicting next hour's % return")
    print("(reported as price-equivalent RMSE, for comparability)")
    print("=" * 60)

    X = df[XGB_FEATURES]
    y = df['Target_Return']  # % change target, not absolute price
    X_train, X_test, y_train, y_test, split_idx = chronological_split(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost (current)": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "SVR (rbf)": SVR(kernel='rbf', cache_size=1000),
    }

    MAX_SVM_ROWS = 15000

    current_close_test = df['close'].iloc[split_idx:]
    actual_price_test = df['Target_Close'].iloc[split_idx:]

    naive_pred_price = current_close_test
    baseline_rmse = np.sqrt(mean_squared_error(actual_price_test, naive_pred_price))

    results = {"Naive Baseline (persistence)": baseline_rmse}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if "SVR" in name and len(X_train_scaled) > MAX_SVM_ROWS:
            print(f"  (using most recent {MAX_SVM_ROWS} rows only, to keep training time reasonable)")
            X_fit, y_fit = X_train_scaled[-MAX_SVM_ROWS:], y_train.iloc[-MAX_SVM_ROWS:]
        else:
            X_fit, y_fit = X_train_scaled, y_train
        model.fit(X_fit, y_fit)
        pred_return = model.predict(X_test_scaled)
        pred_price = current_close_test * (1 + pred_return)
        rmse = np.sqrt(mean_squared_error(actual_price_test, pred_price))
        results[name] = rmse
        beats = "✅ beats baseline" if rmse < baseline_rmse else "⚠️  worse than baseline"
        print(f"  RMSE (price-equivalent): {rmse:.4f}  ({beats})")

    print("\n--- Regression leaderboard (lower RMSE = better) ---")
    for name, rmse in sorted(results.items(), key=lambda x: x[1]):
        tag = "  <-- baseline" if "Baseline" in name else ""
        print(f"  {rmse:10.4f}   {name}{tag}")

    return results


def compare_classifiers(df):
    print("\n" + "=" * 60)
    print("CLASSIFICATION MODELS — predicting next hour's direction")
    print("=" * 60)

    X = df[SVC_FEATURES]
    y = df['Target_Movement']
    X_train, X_test, y_train, y_test, split_idx = chronological_split(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost Classifier": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss'),
        "SVC (current, rbf)": SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000),
    }

    # Same reasoning as the regression SVR above — cap SVC's training rows
    # to keep this comparison run in a reasonable amount of time.
    MAX_SVM_ROWS = 15000

    # Naive baseline: always predict the majority class from TRAINING data
    majority_class = y_train.mode()[0]
    baseline_preds = pd.Series(majority_class, index=y_test.index)
    baseline_acc = accuracy_score(y_test, baseline_preds)

    results = {f"Naive Baseline (always '{majority_class}')": baseline_acc}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if "SVC" in name and len(X_train_scaled) > MAX_SVM_ROWS:
            print(f"  (using most recent {MAX_SVM_ROWS} rows only, to keep training time reasonable)")
            X_fit, y_fit = X_train_scaled[-MAX_SVM_ROWS:], y_train.iloc[-MAX_SVM_ROWS:]
        else:
            X_fit, y_fit = X_train_scaled, y_train
        model.fit(X_fit, y_fit)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        beats = "✅ beats baseline" if acc > baseline_acc else "⚠️  does not clearly beat baseline"
        print(f"  Accuracy: {acc:.4f}  ({beats})")

    print("\n--- Classification leaderboard (higher accuracy = better) ---")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        tag = "  <-- baseline" if "Baseline" in name else ""
        print(f"  {acc:10.4f}   {name}{tag}")

    return results


def main():
    print("Loading and preparing data...")
    df = load_and_clean_data('data/raw/BTCUSDT-1H.csv')
    df = create_features(df)
    print(f"Data ready: {len(df)} rows after feature engineering.")

    reg_results = compare_regressors(df)
    clf_results = compare_classifiers(df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_reg = min(reg_results.items(), key=lambda x: x[1])
    best_clf = max(clf_results.items(), key=lambda x: x[1])
    print(f"Best regressor : {best_reg[0]}  (RMSE {best_reg[1]:.4f})")
    print(f"Best classifier: {best_clf[0]}  (Accuracy {best_clf[1]:.4f})")
    print("\nNext step: hyperparameter-tune whichever model(s) won, rather than")
    print("the current XGBoost/SVC defaults, if a different model came out ahead.")


if __name__ == '__main__':
    main()