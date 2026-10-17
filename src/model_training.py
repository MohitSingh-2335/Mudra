# src/model_training.py
"""
Model training engine for Mudra quantitative platform.
Trains baseline dual-target machine learning models:
1. XGBRegressor: Forecasts continuous next-hour percentage return (Target_Return).
2. XGBClassifier: Forecasts binary market directional regime (Target_Movement).

Enforces strict chronological 80/20 train/test splitting to prevent future-data
leakage through rolling/lag feature windows, benchmarks performance against naive
persistence baselines, and logs parameters, metrics, and serialized artifacts to MLflow.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from config import (
    BTCUSDT_1H_CSV,
    CLASSIFIER_FEATURES,
    CLASSIFIER_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    REGRESSOR_FEATURES,
    REGRESSOR_MODEL_PATH,
)
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features


def get_best_xgboost_device() -> str:
    """
    Detects hardware capabilities and returns 'cuda' if NVIDIA GPU
    acceleration is present, or falls back to 'cpu' for portable cross-platform execution.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def train_and_save_models(
    data_path: Union[str, Path] = BTCUSDT_1H_CSV,
    models_dir: Union[str, Path] = MODELS_DIR,
) -> Dict[str, Any]:
    """
    Ingests candlestick data, merges alternative indicators, computes quantitative
    features, trains baseline XGBoost regressor and classifier models, evaluates them
    against naive persistence baselines on held-out chronological data, and logs to MLflow.

    Args:
        data_path (Union[str, Path]): Path to input candlestick CSV.
        models_dir (Union[str, Path]): Directory to store serialized model artifacts.

    Returns:
        Dict[str, Any]: Dictionary containing trained models and test metric evaluations.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(f"[INFO] Starting model training pipeline using: '{data_path}'")
    df = load_and_clean_data(data_path)
    print(f"[INFO] Loaded {len(df):,} candlestick rows.")

    # Check if features are already engineered; otherwise generate them
    required_targets = {"Target_Return", "Target_Movement", "Target_Close"}
    if not required_targets.issubset(df.columns):
        print("[INFO] Merging Fear & Greed sentiment metrics...")
        df = merge_fear_greed(df, timestamp_col="timestamp")
        print("[INFO] Merging on-chain network metrics...")
        df = merge_onchain(df, timestamp_col="timestamp")
        print("[INFO] Generating 39 technical features and target labels...")
        df = create_features(df, drop_na=True)

    device = get_best_xgboost_device()
    print(f"[INFO] Training hardware accelerated device: '{device}'")

    results: Dict[str, Any] = {}

    # =========================================================================
    # 1. Continuous Return Prediction (XGBRegressor -> Target_Return)
    # =========================================================================
    print("\n--- Training Continuous Return Model (XGBRegressor) ---")
    X1 = df[REGRESSOR_FEATURES]
    y1 = df["Target_Return"]

    # Chronological split (last 20% strictly reserved for out-of-sample holdout)
    split_idx_1 = int(len(X1) * 0.8)
    X1_train, X1_test = X1.iloc[:split_idx_1], X1.iloc[split_idx_1:]
    y1_train, y1_test = y1.iloc[:split_idx_1], y1.iloc[split_idx_1:]

    with mlflow.start_run(run_name="xgboost_regressor_baseline"):
        mlflow.set_tag("model_type", "XGBRegressor")
        mlflow.set_tag("target", "Target_Return")
        mlflow.set_tag("hardware_device", device)

        reg_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 5,
            "tree_method": "hist",
            "device": device,
            "random_state": 42,
        }
        mlflow.log_params(reg_params)
        mlflow.log_param("num_features", len(REGRESSOR_FEATURES))
        mlflow.log_param("train_samples", len(X1_train))
        mlflow.log_param("test_samples", len(X1_test))

        xgbr = XGBRegressor(**reg_params)
        xgbr.fit(X1_train, y1_train)

        reg_save_path = Path(REGRESSOR_MODEL_PATH)
        reg_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(xgbr, reg_save_path)
        print(f"[SUCCESS] XGBoost Regressor model saved to: '{reg_save_path}'")

        # Evaluate on chronological out-of-sample test set
        y1_pred_return = xgbr.predict(X1_test)
        current_close_test = df["close"].iloc[split_idx_1:]
        actual_price_test = df["Target_Close"].iloc[split_idx_1:]
        predicted_price_test = current_close_test * (1 + y1_pred_return)

        reg_rmse = float(np.sqrt(mean_squared_error(actual_price_test, predicted_price_test)))
        naive_baseline_rmse = float(np.sqrt(mean_squared_error(actual_price_test, current_close_test)))
        rmse_diff = naive_baseline_rmse - reg_rmse

        mlflow.log_metrics({
            "test_rmse_price": reg_rmse,
            "naive_baseline_rmse": naive_baseline_rmse,
            "rmse_diff": rmse_diff,
        })
        mlflow.xgboost.log_model(xgbr, artifact_path="model")

        print(f"[METRIC] XGBoost Regressor Test RMSE (Price-equivalent): {reg_rmse:.4f}")
        print(f"[BASELINE] Naive Persistence Test RMSE: {naive_baseline_rmse:.4f}")
        if reg_rmse < naive_baseline_rmse:
            print(f"[EVALUATION] Model beats naive baseline by {rmse_diff:.4f} USD.")
        else:
            print(f"[WARNING] Model lags naive baseline by {abs(rmse_diff):.4f} USD.")

        results["regressor"] = {
            "model": xgbr,
            "test_rmse_price": reg_rmse,
            "naive_baseline_rmse": naive_baseline_rmse,
            "rmse_diff": rmse_diff,
        }

    # =========================================================================
    # 2. Directional Movement Prediction (XGBClassifier -> Target_Movement)
    # =========================================================================
    print("\n--- Training Directional Regime Model (XGBClassifier) ---")
    X2 = df[CLASSIFIER_FEATURES]
    y2 = df["Target_Movement"]

    split_idx_2 = int(len(X2) * 0.8)
    X2_train, X2_test = X2.iloc[:split_idx_2], X2.iloc[split_idx_2:]
    y2_train, y2_test = y2.iloc[:split_idx_2], y2.iloc[split_idx_2:]

    with mlflow.start_run(run_name="xgboost_classifier_baseline"):
        mlflow.set_tag("model_type", "XGBClassifier")
        mlflow.set_tag("target", "Target_Movement")
        mlflow.set_tag("hardware_device", device)

        clf_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "device": device,
            "random_state": 42,
        }
        mlflow.log_params(clf_params)
        mlflow.log_param("num_features", len(CLASSIFIER_FEATURES))
        mlflow.log_param("train_samples", len(X2_train))
        mlflow.log_param("test_samples", len(X2_test))

        xgbc = XGBClassifier(**clf_params)
        xgbc.fit(X2_train, y2_train)

        clf_save_path = Path(CLASSIFIER_MODEL_PATH)
        clf_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(xgbc, clf_save_path)
        print(f"[SUCCESS] XGBoost Classifier model saved to: '{clf_save_path}'")

        # Evaluate on chronological out-of-sample test set
        y2_pred = xgbc.predict(X2_test)
        clf_acc = float(accuracy_score(y2_test, y2_pred))

        majority_class = int(y2_train.mode()[0])
        baseline_preds = pd.Series(majority_class, index=y2_test.index)
        baseline_acc = float(accuracy_score(y2_test, baseline_preds))
        acc_gain = (clf_acc - baseline_acc) * 100

        mlflow.log_metrics({
            "test_accuracy": clf_acc,
            "naive_baseline_accuracy": baseline_acc,
            "accuracy_gain_pct": acc_gain,
        })
        mlflow.xgboost.log_model(xgbc, artifact_path="model")

        print(f"[METRIC] XGBoost Classifier Test Accuracy: {clf_acc * 100:.2f}%")
        print(f"[BASELINE] Majority Class ('{majority_class}') Baseline Accuracy: {baseline_acc * 100:.2f}%")
        if clf_acc > baseline_acc:
            print(f"[EVALUATION] Model beats majority baseline by +{acc_gain:.2f} percentage points.")
        else:
            print(f"[WARNING] Model lags majority baseline by {acc_gain:.2f} percentage points.")

        results["classifier"] = {
            "model": xgbc,
            "test_accuracy": clf_acc,
            "naive_baseline_accuracy": baseline_acc,
            "accuracy_gain_pct": acc_gain,
        }

    print("\n[INFO] Model training and evaluation successfully completed.")
    return results


if __name__ == "__main__":
    train_and_save_models(data_path=BTCUSDT_1H_CSV)