# src/model_training.py
from pathlib import Path
from typing import Any, Dict, Union
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from config import (
    BTCUSDT_1H_CSV, CLASSIFIER_FEATURES, CLASSIFIER_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODELS_DIR,
    REGRESSOR_FEATURES, REGRESSOR_MODEL_PATH,
)
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features


def get_best_xgboost_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def train_and_save_models(data_path: Union[str, Path] = BTCUSDT_1H_CSV, models_dir: Union[str, Path] = MODELS_DIR) -> Dict[str, Any]:
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_and_clean_data(data_path)
    if not {"Target_Return", "Target_Movement", "Target_Close"}.issubset(df.columns):
        df = create_features(merge_onchain(merge_fear_greed(df, "timestamp"), "timestamp"), drop_na=True)

    device = get_best_xgboost_device()
    results: Dict[str, Any] = {}

    # 1. Regressor
    X1, y1 = df[REGRESSOR_FEATURES], df["Target_Return"]
    split_1 = int(len(X1) * 0.8)
    X1_tr, X1_te, y1_tr, y1_te = X1.iloc[:split_1], X1.iloc[split_1:], y1.iloc[:split_1], y1.iloc[split_1:]

    with mlflow.start_run(run_name="xgboost_regressor_baseline"):
        reg_params = {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "tree_method": "hist", "device": device, "random_state": 42}
        mlflow.log_params(reg_params)
        xgbr = XGBRegressor(**reg_params).fit(X1_tr, y1_tr)
        joblib.dump(xgbr, REGRESSOR_MODEL_PATH)

        y1_pred = xgbr.predict(X1_te)
        close_te, act_te = df["close"].iloc[split_1:], df["Target_Close"].iloc[split_1:]
        reg_rmse = float(np.sqrt(mean_squared_error(act_te, close_te * (1 + y1_pred))))
        base_rmse = float(np.sqrt(mean_squared_error(act_te, close_te)))
        mlflow.log_metrics({"test_rmse_price": reg_rmse, "naive_baseline_rmse": base_rmse, "rmse_diff": base_rmse - reg_rmse})
        mlflow.xgboost.log_model(xgbr, artifact_path="model")
        results["regressor"] = {"model": xgbr, "test_rmse_price": reg_rmse, "naive_baseline_rmse": base_rmse}

    # 2. Classifier
    X2, y2 = df[CLASSIFIER_FEATURES], df["Target_Movement"]
    split_2 = int(len(X2) * 0.8)
    X2_tr, X2_te, y2_tr, y2_te = X2.iloc[:split_2], X2.iloc[split_2:], y2.iloc[:split_2], y2.iloc[split_2:]

    with mlflow.start_run(run_name="xgboost_classifier_baseline"):
        clf_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "tree_method": "hist", "device": device, "random_state": 42}
        mlflow.log_params(clf_params)
        xgbc = XGBClassifier(**clf_params).fit(X2_tr, y2_tr)
        joblib.dump(xgbc, CLASSIFIER_MODEL_PATH)

        clf_acc = float(accuracy_score(y2_te, xgbc.predict(X2_te)))
        base_acc = float(accuracy_score(y2_te, pd.Series(int(y2_tr.mode()[0]), index=y2_te.index)))
        mlflow.log_metrics({"test_accuracy": clf_acc, "naive_baseline_accuracy": base_acc, "accuracy_gain_pct": (clf_acc - base_acc) * 100})
        mlflow.xgboost.log_model(xgbc, artifact_path="model")
        results["classifier"] = {"model": xgbc, "test_accuracy": clf_acc, "naive_baseline_accuracy": base_acc}

    return results


if __name__ == "__main__":
    train_and_save_models()
