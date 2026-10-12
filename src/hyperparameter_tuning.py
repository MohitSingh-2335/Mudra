# src/hyperparameter_tuning.py
"""
Bayesian Hyperparameter Optimization engine for Mudra models.
Utilizes Optuna's Tree-structured Parzen Estimator (TPE) algorithm with
5-fold TimeSeriesSplit cross-validation to tune hyperparameters:
1. XGBRegressor: Minimizes out-of-sample price-equivalent RMSE.
2. XGBClassifier: Maximizes out-of-sample directional accuracy.

Logs all optimization trials, winning parameters, and model artifacts to MLflow.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from config import (
    BTCUSDT_1H_CSV,
    CLASSIFIER_FEATURES,
    CLASSIFIER_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGRESSOR_FEATURES,
    REGRESSOR_MODEL_PATH,
)
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features


def get_best_xgboost_device() -> str:
    """
    Detects hardware acceleration and returns 'cuda' if NVIDIA GPU
    is available, or falls back to 'cpu' for portable execution.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def objective_regressor(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    close_train: pd.Series,
    target_close_train: pd.Series,
    device: str = "cpu",
) -> float:
    """
    Optuna objective function for XGBoost Regressor using TimeSeriesSplit cross-validation.
    Minimizes mean price-equivalent RMSE across 5 chronological validation folds.
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
        "device": device,
        "random_state": 42,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    fold_rmses = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)

        # Predict returns and convert to price terms: P_{t+1} = P_t * (1 + return)
        pred_returns = model.predict(X_val)
        val_close = close_train.iloc[val_idx]
        val_target_close = target_close_train.iloc[val_idx]
        pred_prices = val_close * (1 + pred_returns)

        fold_rmse = float(np.sqrt(mean_squared_error(val_target_close, pred_prices)))
        fold_rmses.append(fold_rmse)

    return float(np.mean(fold_rmses))


def objective_classifier(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    device: str = "cpu",
) -> float:
    """
    Optuna objective function for XGBoost Classifier using TimeSeriesSplit cross-validation.
    Maximizes mean accuracy across 5 chronological validation folds.
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
        "device": device,
        "random_state": 42,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    fold_accuracies = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        acc = float(accuracy_score(y_val, y_pred))
        fold_accuracies.append(acc)

    return float(np.mean(fold_accuracies))


def run_tuning(
    data_path: Union[str, Path] = BTCUSDT_1H_CSV,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """
    Runs Optuna hyperparameter optimization for both regressor and classifier models,
    logs experiments and artifacts to MLflow, and saves best models to disk.

    Args:
        data_path (Union[str, Path]): Candlestick data file path.
        n_trials (int): Number of Bayesian optimization trials per model.

    Returns:
        Dict[str, Any]: Dictionary containing best parameters, models, and test metrics.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(f"[INFO] Ingesting and preparing data for Optuna tuning: '{data_path}'")
    df = load_and_clean_data(data_path)

    required_targets = {"Target_Return", "Target_Movement", "Target_Close"}
    if not required_targets.issubset(df.columns):
        print("[INFO] Merging Fear & Greed sentiment metrics...")
        df = merge_fear_greed(df, timestamp_col="timestamp")
        print("[INFO] Merging on-chain network metrics...")
        df = merge_onchain(df, timestamp_col="timestamp")
        print("[INFO] Generating 39 technical features and target labels...")
        df = create_features(df, drop_na=True)

    device = get_best_xgboost_device()
    print(f"[INFO] Hardware accelerated device detected: '{device}'")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results: Dict[str, Any] = {}

    # =========================================================================
    # 1. TUNE REGRESSOR (Continuous Return -> Price RMSE)
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"[INFO] [1/2] Tuning XGBoost Regressor ({n_trials} trials, TimeSeriesSplit CV)...")
    print("=" * 60)

    X1 = df[REGRESSOR_FEATURES]
    y1 = df["Target_Return"]
    close = df["close"]
    target_close = df["Target_Close"]

    split_idx_1 = int(len(X1) * 0.8)
    X1_train, X1_test = X1.iloc[:split_idx_1], X1.iloc[split_idx_1:]
    y1_train, y1_test = y1.iloc[:split_idx_1], y1.iloc[split_idx_1:]
    close_train, close_test = close.iloc[:split_idx_1], close.iloc[split_idx_1:]
    target_close_train, target_close_test = target_close.iloc[:split_idx_1], target_close.iloc[split_idx_1:]

    study_reg = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_reg.optimize(
        lambda trial: objective_regressor(trial, X1_train, y1_train, close_train, target_close_train, device=device),
        n_trials=n_trials,
    )

    print(f"[SUCCESS] Regressor tuning complete. Best CV Price RMSE: ${study_reg.best_value:.4f}")

    best_reg_params = study_reg.best_params.copy()
    best_reg_params.update({"tree_method": "hist", "device": device, "random_state": 42})
    best_xgbr = XGBRegressor(**best_reg_params)
    best_xgbr.fit(X1_train, y1_train)

    y1_pred_ret = best_xgbr.predict(X1_test)
    pred_prices = close_test * (1 + y1_pred_ret)
    test_rmse = float(np.sqrt(mean_squared_error(target_close_test, pred_prices)))
    baseline_rmse = float(np.sqrt(mean_squared_error(target_close_test, close_test)))
    rmse_diff = baseline_rmse - test_rmse

    print(f"[METRIC] Tuned Regressor Holdout RMSE: ${test_rmse:.4f} (Baseline: ${baseline_rmse:.4f})")

    with mlflow.start_run(run_name="optuna_tuned_xgboost_regressor"):
        mlflow.set_tag("model_type", "XGBRegressor")
        mlflow.set_tag("tuned_by", "Optuna_TPE_TimeSeriesSplit")
        mlflow.set_tag("hardware_device", device)
        mlflow.log_params(best_reg_params)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metrics({
            "best_cv_rmse": float(study_reg.best_value),
            "test_rmse_price": test_rmse,
            "naive_baseline_rmse": baseline_rmse,
            "rmse_diff": rmse_diff,
        })
        mlflow.xgboost.log_model(best_xgbr, artifact_path="model")

    reg_save_path = Path(REGRESSOR_MODEL_PATH)
    reg_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_xgbr, reg_save_path)
    print(f"[SUCCESS] Tuned regressor saved to: '{reg_save_path}'")

    results["regressor"] = {
        "best_params": best_reg_params,
        "best_cv_rmse": study_reg.best_value,
        "test_rmse_price": test_rmse,
        "naive_baseline_rmse": baseline_rmse,
        "model": best_xgbr,
    }

    # =========================================================================
    # 2. TUNE CLASSIFIER (Directional Movement -> Accuracy)
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"[INFO] [2/2] Tuning XGBoost Classifier ({n_trials} trials, TimeSeriesSplit CV)...")
    print("=" * 60)

    X2 = df[CLASSIFIER_FEATURES]
    y2 = df["Target_Movement"]

    split_idx_2 = int(len(X2) * 0.8)
    X2_train, X2_test = X2.iloc[:split_idx_2], X2.iloc[split_idx_2:]
    y2_train, y2_test = y2.iloc[:split_idx_2], y2.iloc[split_idx_2:]

    study_clf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_clf.optimize(
        lambda trial: objective_classifier(trial, X2_train, y2_train, device=device),
        n_trials=n_trials,
    )

    print(f"[SUCCESS] Classifier tuning complete. Best CV Accuracy: {study_clf.best_value * 100:.2f}%")

    best_clf_params = study_clf.best_params.copy()
    best_clf_params.update({"tree_method": "hist", "device": device, "random_state": 42})
    best_xgbc = XGBClassifier(**best_clf_params)
    best_xgbc.fit(X2_train, y2_train)

    y2_pred = best_xgbc.predict(X2_test)
    test_acc = float(accuracy_score(y2_test, y2_pred))
    majority_class = int(y2_train.mode()[0])
    baseline_acc = float(accuracy_score(y2_test, pd.Series(majority_class, index=y2_test.index)))
    acc_gain = (test_acc - baseline_acc) * 100

    print(f"[METRIC] Tuned Classifier Holdout Accuracy: {test_acc * 100:.2f}% (Baseline: {baseline_acc * 100:.2f}%)")

    with mlflow.start_run(run_name="optuna_tuned_xgboost_classifier"):
        mlflow.set_tag("model_type", "XGBClassifier")
        mlflow.set_tag("tuned_by", "Optuna_TPE_TimeSeriesSplit")
        mlflow.set_tag("hardware_device", device)
        mlflow.log_params(best_clf_params)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metrics({
            "best_cv_accuracy": float(study_clf.best_value),
            "test_accuracy": test_acc,
            "naive_baseline_accuracy": baseline_acc,
            "accuracy_gain_pct": acc_gain,
        })
        mlflow.xgboost.log_model(best_xgbc, artifact_path="model")

    clf_save_path = Path(CLASSIFIER_MODEL_PATH)
    clf_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_xgbc, clf_save_path)
    print(f"[SUCCESS] Tuned classifier saved to: '{clf_save_path}'")

    results["classifier"] = {
        "best_params": best_clf_params,
        "best_cv_accuracy": study_clf.best_value,
        "test_accuracy": test_acc,
        "naive_baseline_accuracy": baseline_acc,
        "model": best_xgbc,
    }

    print("\n[SUCCESS] Hyperparameter tuning complete. Metrics and artifacts recorded in MLflow.")
    return results


if __name__ == "__main__":
    run_tuning(n_trials=50)
