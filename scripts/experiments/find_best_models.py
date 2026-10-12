# scripts/experiments/find_best_models.py
"""
Algorithmic Horse Race Benchmark for Mudra quantitative models.
Compares multiple model families (Linear Regression, Random Forest, XGBoost, Support Vector Machines)
on strictly identical chronological out-of-sample holdout partitions and identical feature sets.
Evaluates both continuous return prediction (Target_Return) and directional regime movement (Target_Movement)
against naive persistence baselines.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

# Ensure repository root is on sys.path for direct CLI script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    BTCUSDT_1H_CSV,
    CLASSIFIER_FEATURES,
    REGRESSOR_FEATURES,
)
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    """
    Splits feature matrix and target chronologically without shuffling to
    strictly preserve temporal causality and prevent look-ahead bias.

    Args:
        X (pd.DataFrame): Input feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of data reserved for out-of-sample holdout.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, split_idx)
    """
    split_idx = int(len(X) * (1 - test_size))
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
        split_idx,
    )


def compare_regressors(df: pd.DataFrame) -> Dict[str, float]:
    """
    Benchmarks multiple regression architectures forecasting Target_Return (% change).
    Converts return predictions back to price-equivalent RMSE for direct interpretability.

    Args:
        df (pd.DataFrame): Candlestick DataFrame with engineered features.

    Returns:
        Dict[str, float]: Model names mapped to price-equivalent test RMSE.
    """
    print("\n" + "=" * 60)
    print("[INFO] REGRESSION MODELS BENCHMARK (Predicting next hour's % return)")
    print("[INFO] Reported as price-equivalent RMSE for institutional comparability")
    print("=" * 60)

    X = df[REGRESSOR_FEATURES]
    y = df["Target_Return"]
    X_train, X_test, y_train, y_test, split_idx = chronological_split(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost (current)": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "SVR (rbf)": SVR(kernel="rbf", cache_size=1000),
    }

    MAX_SVM_ROWS = 15000
    current_close_test = df["close"].iloc[split_idx:]
    actual_price_test = df["Target_Close"].iloc[split_idx:]

    naive_pred_price = current_close_test
    baseline_rmse = float(np.sqrt(mean_squared_error(actual_price_test, naive_pred_price)))
    results: Dict[str, float] = {"Naive Baseline (persistence)": baseline_rmse}

    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        if "SVR" in name and len(X_train_scaled) > MAX_SVM_ROWS:
            print(f"  [NOTE] SVR capped to most recent {MAX_SVM_ROWS:,} rows for compute bounds.")
            X_fit, y_fit = X_train_scaled[-MAX_SVM_ROWS:], y_train.iloc[-MAX_SVM_ROWS:]
        else:
            X_fit, y_fit = X_train_scaled, y_train

        model.fit(X_fit, y_fit)
        pred_return = model.predict(X_test_scaled)
        pred_price = current_close_test * (1 + pred_return)
        rmse = float(np.sqrt(mean_squared_error(actual_price_test, pred_price)))
        results[name] = rmse

        beats = "[BEATS BASELINE]" if rmse < baseline_rmse else "[LAGS BASELINE]"
        print(f"  Test RMSE (Price-equivalent): ${rmse:.4f}  {beats}")

    print("\n--- Regression Leaderboard (Lower RMSE = Better) ---")
    for name, rmse in sorted(results.items(), key=lambda x: x[1]):
        tag = "  <-- BASELINE" if "Baseline" in name else ""
        print(f"  ${rmse:10.4f}   {name}{tag}")

    return results


def compare_classifiers(df: pd.DataFrame) -> Dict[str, float]:
    """
    Benchmarks multiple classification architectures forecasting Target_Movement (binary direction).

    Args:
        df (pd.DataFrame): Candlestick DataFrame with engineered features.

    Returns:
        Dict[str, float]: Model names mapped to holdout test accuracy.
    """
    print("\n" + "=" * 60)
    print("[INFO] CLASSIFICATION MODELS BENCHMARK (Predicting next hour's direction)")
    print("=" * 60)

    X = df[CLASSIFIER_FEATURES]
    y = df["Target_Movement"]
    X_train, X_test, y_train, y_test, _ = chronological_split(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric="logloss"
        ),
        "SVC (current, rbf)": SVC(kernel="rbf", probability=True, random_state=42, cache_size=1000),
    }

    MAX_SVM_ROWS = 15000
    majority_class = int(y_train.mode()[0])
    baseline_preds = pd.Series(majority_class, index=y_test.index)
    baseline_acc = float(accuracy_score(y_test, baseline_preds))
    results: Dict[str, float] = {f"Naive Baseline (always '{majority_class}')": baseline_acc}

    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        if "SVC" in name and len(X_train_scaled) > MAX_SVM_ROWS:
            print(f"  [NOTE] SVC capped to most recent {MAX_SVM_ROWS:,} rows for compute bounds.")
            X_fit, y_fit = X_train_scaled[-MAX_SVM_ROWS:], y_train.iloc[-MAX_SVM_ROWS:]
        else:
            X_fit, y_fit = X_train_scaled, y_train

        model.fit(X_fit, y_fit)
        preds = model.predict(X_test_scaled)
        acc = float(accuracy_score(y_test, preds))
        results[name] = acc

        beats = "[BEATS BASELINE]" if acc > baseline_acc else "[LAGS BASELINE]"
        print(f"  Holdout Accuracy: {acc * 100:.2f}%  {beats}")

    print("\n--- Classification Leaderboard (Higher Accuracy = Better) ---")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        tag = "  <-- BASELINE" if "Baseline" in name else ""
        print(f"  {acc * 100:9.2f}%   {name}{tag}")

    return results


def main() -> Dict[str, Any]:
    """
    Executes full multi-model benchmark across regression and classification tasks.
    """
    print("[INFO] Loading raw data and engineering features for horse race benchmark...")
    df = load_and_clean_data(BTCUSDT_1H_CSV)
    df = merge_fear_greed(df, timestamp_col="timestamp")
    df = merge_onchain(df, timestamp_col="timestamp")
    df = create_features(df, drop_na=True)
    print(f"[INFO] Dataset ready: {len(df):,} chronological candles after feature engineering.")

    reg_results = compare_regressors(df)
    clf_results = compare_classifiers(df)

    print("\n" + "=" * 60)
    print("[INFO] BENCHMARK SUMMARY & MODEL SELECTION VERDICT")
    print("=" * 60)
    best_reg = min(reg_results.items(), key=lambda x: x[1])
    best_clf = max(clf_results.items(), key=lambda x: x[1])
    print(f"[VERDICT] Best Regressor : {best_reg[0]}  (RMSE: ${best_reg[1]:.4f})")
    print(f"[VERDICT] Best Classifier: {best_clf[0]}  (Accuracy: {best_clf[1] * 100:.2f}%)")

    return {
        "regression": reg_results,
        "classification": clf_results,
        "best_regressor": best_reg,
        "best_classifier": best_clf,
    }


if __name__ == "__main__":
    main()