from pathlib import Path
from typing import Any, Dict, Union
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
    BTCUSDT_1H_CSV, CLASSIFIER_FEATURES, CLASSIFIER_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, REGRESSOR_FEATURES, REGRESSOR_MODEL_PATH,
)
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features


def get_best_xgboost_device() -> str:
    # uses cuda if available, otherwise defaults to cpu
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _suggest_xgb_params(trial: optuna.Trial, device: str = "cpu") -> dict:
    # hyperparameter search space for tree-based estimators
    return {
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


def objective_regressor(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, close: pd.Series, target_close: pd.Series, device: str = "cpu") -> float:
    # cross-validation objective minimizing price forecast RMSE
    params = _suggest_xgb_params(trial, device)
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    for tr, val in tscv.split(X):
        model = XGBRegressor(**params).fit(X.iloc[tr], y.iloc[tr])
        pred_prices = close.iloc[val] * (1.0 + model.predict(X.iloc[val]))
        rmses.append(float(np.sqrt(mean_squared_error(target_close.iloc[val], pred_prices))))
    return float(np.mean(rmses))


def objective_classifier(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, device: str = "cpu") -> float:
    # cross-validation objective maximizing directional accuracy
    params = _suggest_xgb_params(trial, device)
    tscv = TimeSeriesSplit(n_splits=5)
    accs = []
    for tr, val in tscv.split(X):
        model = XGBClassifier(**params).fit(X.iloc[tr], y.iloc[tr])
        accs.append(float(accuracy_score(y.iloc[val], model.predict(X.iloc[val]))))
    return float(np.mean(accs))


def run_tuning(data_path: Union[str, Path] = BTCUSDT_1H_CSV, n_trials: int = 50) -> Dict[str, Any]:
    # executes Optuna bayesian search over regressor and classifier
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    df = load_and_clean_data(data_path)
    if not {"Target_Return", "Target_Movement", "Target_Close"}.issubset(df.columns):
        df = create_features(merge_onchain(merge_fear_greed(df, "timestamp"), "timestamp"), drop_na=True)

    device = get_best_xgboost_device()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results: Dict[str, Any] = {}

    split = int(len(df) * 0.8)
    X1_tr, X1_te = df[REGRESSOR_FEATURES].iloc[:split], df[REGRESSOR_FEATURES].iloc[split:]
    y1_tr, y1_te = df["Target_Return"].iloc[:split], df["Target_Return"].iloc[split:]
    c_tr, c_te = df["close"].iloc[:split], df["close"].iloc[split:]
    tc_tr, tc_te = df["Target_Close"].iloc[:split], df["Target_Close"].iloc[split:]

    study_reg = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_reg.optimize(lambda t: objective_regressor(t, X1_tr, y1_tr, c_tr, tc_tr, device), n_trials=n_trials)
    best_reg_params = {**study_reg.best_params, "tree_method": "hist", "device": device, "random_state": 42}
    best_xgbr = XGBRegressor(**best_reg_params).fit(X1_tr, y1_tr)
    test_rmse = float(np.sqrt(mean_squared_error(tc_te, c_te * (1.0 + best_xgbr.predict(X1_te)))))
    base_rmse = float(np.sqrt(mean_squared_error(tc_te, c_te)))

    Path(REGRESSOR_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_xgbr, REGRESSOR_MODEL_PATH)
    results["regressor"] = {"best_params": best_reg_params, "test_rmse_price": test_rmse, "model": best_xgbr}

    X2_tr, X2_te = df[CLASSIFIER_FEATURES].iloc[:split], df[CLASSIFIER_FEATURES].iloc[split:]
    y2_tr, y2_te = df["Target_Movement"].iloc[:split], df["Target_Movement"].iloc[split:]

    study_clf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_clf.optimize(lambda t: objective_classifier(t, X2_tr, y2_tr, device), n_trials=n_trials)
    best_clf_params = {**study_clf.best_params, "tree_method": "hist", "device": device, "random_state": 42}
    best_xgbc = XGBClassifier(**best_clf_params).fit(X2_tr, y2_tr)
    test_acc = float(accuracy_score(y2_te, best_xgbc.predict(X2_te)))

    from src.model_loader import save_production_models
    save_production_models(best_xgbc, best_xgbr, CLASSIFIER_FEATURES, "xgb_v1.1.0_tuned", {"tuned_test_accuracy": round(test_acc, 4), "tuned_test_rmse": round(test_rmse, 2)})
    results["classifier"] = {"best_params": best_clf_params, "test_accuracy": test_acc, "model": best_xgbc}
    return results


if __name__ == "__main__":
    run_tuning(n_trials=50)
