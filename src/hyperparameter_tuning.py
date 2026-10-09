# src/hyperparameter_tuning.py

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.xgboost
import joblib
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from config import (
    REGRESSOR_FEATURES, 
    CLASSIFIER_FEATURES, 
    REGRESSOR_MODEL_PATH, 
    CLASSIFIER_MODEL_PATH,
    BTCUSDT_1H_CSV,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME
)

def objective_regressor(trial, X_train, y_train, close_train, target_close_train):
    """
    Optuna objective function for XGBoost Regressor using TimeSeriesSplit cross-validation.
    Minimizes mean price-equivalent RMSE across 5 validation folds.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42
    }

    tscv = TimeSeriesSplit(n_splits=5)
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)

        # Predict returns and convert to dollar price
        pred_returns = model.predict(X_val)
        val_close = close_train.iloc[val_idx]
        val_target_close = target_close_train.iloc[val_idx]
        pred_prices = val_close * (1 + pred_returns)

        fold_rmse = np.sqrt(mean_squared_error(val_target_close, pred_prices))
        fold_rmses.append(fold_rmse)

    return np.mean(fold_rmses)

def objective_classifier(trial, X_train, y_train):
    """
    Optuna objective function for XGBoost Classifier using TimeSeriesSplit cross-validation.
    Maximizes mean accuracy across 5 validation folds.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42
    }

    tscv = TimeSeriesSplit(n_splits=5)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)

    return np.mean(fold_accuracies)


def run_tuning(data_path=BTCUSDT_1H_CSV, n_trials=50):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("📊 Loading and preparing data for Optuna tuning...")
    df = load_and_clean_data(data_path)
    df = merge_fear_greed(df, timestamp_col='timestamp')
    df = merge_onchain(df, timestamp_col='timestamp')
    df = create_features(df)
    print("Features ready.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # =========================================================================
    # 1. TUNE REGRESSOR
    # =========================================================================
    print("\n" + "="*60)
    print("🚀 [1/2] Tuning XGBoost Regressor (Minimizing Price RMSE)")
    print("="*60)

    X1 = df[REGRESSOR_FEATURES]
    y1 = df['Target_Return']
    close = df['close']
    target_close = df['Target_Close']

    split_idx_1 = int(len(X1) * 0.8)
    X1_train, X1_test = X1.iloc[:split_idx_1], X1.iloc[split_idx_1:]
    y1_train, y1_test = y1.iloc[:split_idx_1], y1.iloc[split_idx_1:]
    close_train, close_test = close.iloc[:split_idx_1], close.iloc[split_idx_1:]
    target_close_train, target_close_test = target_close.iloc[:split_idx_1], target_close.iloc[split_idx_1:]

    study_reg = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_reg.optimize(
        lambda trial: objective_regressor(trial, X1_train, y1_train, close_train, target_close_train),
        n_trials=n_trials
    )

    print(f"✅ Regressor Tuning Finished! Best CV RMSE: ${study_reg.best_value:.4f}")

    # Train winning model on full training set and evaluate on test holdout
    best_reg_params = study_reg.best_params.copy()
    best_reg_params.update({"tree_method": "hist", "device": "cuda", "random_state": 42})
    best_xgbr = XGBRegressor(**best_reg_params)
    best_xgbr.fit(X1_train, y1_train)

    y1_pred_ret = best_xgbr.predict(X1_test)
    pred_prices = close_test * (1 + y1_pred_ret)
    test_rmse = np.sqrt(mean_squared_error(target_close_test, pred_prices))
    baseline_rmse = np.sqrt(mean_squared_error(target_close_test, close_test))

    print(f"📊 Tuned Regressor Holdout RMSE: ${test_rmse:.4f} (Baseline: ${baseline_rmse:.4f})")

    # Log tuned model run to MLflow
    with mlflow.start_run(run_name="optuna_tuned_xgboost_regressor"):
        mlflow.set_tag("model_type", "XGBRegressor")
        mlflow.set_tag("tuned_by", "Optuna_TPE_TimeSeriesSplit")
        mlflow.log_params(best_reg_params)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metrics({
            "best_cv_rmse": study_reg.best_value,
            "test_rmse_price": test_rmse,
            "naive_baseline_rmse": baseline_rmse,
            "rmse_diff": baseline_rmse - test_rmse
        })
        mlflow.xgboost.log_model(best_xgbr, artifact_path="model")

    joblib.dump(best_xgbr, REGRESSOR_MODEL_PATH)
    print(f"💾 Tuned regressor saved to: {REGRESSOR_MODEL_PATH}")

    # =========================================================================
    # 2. TUNE CLASSIFIER
    # =========================================================================
    print("\n" + "="*60)
    print("🚀 [2/2] Tuning XGBoost Classifier (Maximizing Accuracy)")
    print("="*60)

    X2 = df[CLASSIFIER_FEATURES]
    y2 = df['Target_Movement']

    split_idx_2 = int(len(X2) * 0.8)
    X2_train, X2_test = X2.iloc[:split_idx_2], X2.iloc[split_idx_2:]
    y2_train, y2_test = y2.iloc[:split_idx_2], y2.iloc[split_idx_2:]

    study_clf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_clf.optimize(
        lambda trial: objective_classifier(trial, X2_train, y2_train),
        n_trials=n_trials
    )

    print(f"✅ Classifier Tuning Finished! Best CV Accuracy: {study_clf.best_value:.4f}")

    # Train winning model on full training set and evaluate on test holdout
    best_clf_params = study_clf.best_params.copy()
    best_clf_params.update({"tree_method": "hist", "device": "cuda", "random_state": 42})
    best_xgbc = XGBClassifier(**best_clf_params)
    best_xgbc.fit(X2_train, y2_train)

    y2_pred = best_xgbc.predict(X2_test)
    test_acc = accuracy_score(y2_test, y2_pred)
    majority_class = y2_train.mode()[0]
    baseline_acc = accuracy_score(y2_test, pd.Series(majority_class, index=y2_test.index))

    print(f"📊 Tuned Classifier Holdout Accuracy: {test_acc:.4f} (Baseline: {baseline_acc:.4f})")

    # Log tuned model run to MLflow
    with mlflow.start_run(run_name="optuna_tuned_xgboost_classifier"):
        mlflow.set_tag("model_type", "XGBClassifier")
        mlflow.set_tag("tuned_by", "Optuna_TPE_TimeSeriesSplit")
        mlflow.log_params(best_clf_params)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metrics({
            "best_cv_accuracy": study_clf.best_value,
            "test_accuracy": test_acc,
            "naive_baseline_accuracy": baseline_acc,
            "accuracy_gain_pct": (test_acc - baseline_acc) * 100
        })
        mlflow.xgboost.log_model(best_xgbc, artifact_path="model")

    joblib.dump(best_xgbc, CLASSIFIER_MODEL_PATH)
    print(f"💾 Tuned classifier saved to: {CLASSIFIER_MODEL_PATH}")

    print("\n🎉 All Optuna tuning complete! Check your MLflow UI for visual run comparisons.")


if __name__ == '__main__':
    run_tuning(n_trials=50)
