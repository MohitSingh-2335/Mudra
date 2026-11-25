from pathlib import Path
from typing import Any, Dict, Union
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from config import (
    ACTIVE_MODEL_VERSION,
    BTCUSDT_1H_CSV,
    CLASSIFIER_FEATURES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    REGRESSOR_FEATURES,
)
from src.data_preprocessing import load_and_clean_data, merge_fear_greed, merge_onchain
from src.feature_engineering import create_features
from src.model_loader import save_production_models


def _prepare_training_df(data_path: Union[str, Path]) -> pd.DataFrame:
    df = load_and_clean_data(data_path)
    if not {"Target_Return", "Target_Movement", "Target_Close"}.issubset(df.columns):
        df = create_features(merge_onchain(merge_fear_greed(df, "timestamp"), "timestamp"), drop_na=True)
    return df


def train_and_save_models(data_path: Union[str, Path] = BTCUSDT_1H_CSV, models_dir: Union[str, Path] = MODELS_DIR) -> Dict[str, Any]:
    """Trains baseline XGBoost regressor and classifier chronologically and updates manifest digests."""
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = _prepare_training_df(data_path)
    split = int(len(df) * 0.8)
    X1_tr, X1_te, y1_tr = df[REGRESSOR_FEATURES].iloc[:split], df[REGRESSOR_FEATURES].iloc[split:], df["Target_Return"].iloc[:split]
    X2_tr, X2_te, y2_tr, y2_te = df[CLASSIFIER_FEATURES].iloc[:split], df[CLASSIFIER_FEATURES].iloc[split:], df["Target_Movement"].iloc[:split], df["Target_Movement"].iloc[split:]

    with mlflow.start_run(run_name="xgboost_regressor_baseline"):
        reg_params = {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "tree_method": "hist", "device": "cpu", "random_state": 42}
        mlflow.log_params(reg_params)
        xgbr = XGBRegressor(**reg_params).fit(X1_tr, y1_tr)
        close_te, act_te = df["close"].iloc[split:], df["Target_Close"].iloc[split:]
        reg_rmse = float(np.sqrt(mean_squared_error(act_te, close_te * (1 + xgbr.predict(X1_te)))))
        base_rmse = float(np.sqrt(mean_squared_error(act_te, close_te)))
        mlflow.log_metrics({"test_rmse_price": reg_rmse, "naive_baseline_rmse": base_rmse, "rmse_diff": base_rmse - reg_rmse})
        mlflow.xgboost.log_model(xgbr, artifact_path="model")

    with mlflow.start_run(run_name="xgboost_classifier_baseline"):
        clf_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "tree_method": "hist", "device": "cpu", "random_state": 42}
        mlflow.log_params(clf_params)
        xgbc = XGBClassifier(**clf_params).fit(X2_tr, y2_tr)
        clf_acc = float(accuracy_score(y2_te, xgbc.predict(X2_te)))
        base_acc = float(accuracy_score(y2_te, pd.Series(int(y2_tr.mode()[0]), index=y2_te.index)))
        mlflow.log_metrics({"test_accuracy": clf_acc, "naive_baseline_accuracy": base_acc, "accuracy_gain_pct": (clf_acc - base_acc) * 100})
        mlflow.xgboost.log_model(xgbc, artifact_path="model")

    save_production_models(
        xgbc, xgbr, CLASSIFIER_FEATURES, ACTIVE_MODEL_VERSION,
        holdout_df=df.iloc[split:],
        metrics={"test_rmse_price": round(reg_rmse, 2), "test_accuracy": round(clf_acc, 4)},
        regressor_features=REGRESSOR_FEATURES,
    )
    return {
        "regressor": {"model": xgbr, "test_rmse_price": reg_rmse, "naive_baseline_rmse": base_rmse},
        "classifier": {"model": xgbc, "test_accuracy": clf_acc, "naive_baseline_accuracy": base_acc},
    }


def _suggest_xgb_params(trial: optuna.Trial) -> dict:
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
        "device": "cpu",
        "random_state": 42,
    }


def objective_regressor(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, close: pd.Series, target_close: pd.Series) -> float:
    params = _suggest_xgb_params(trial)
    rmses = []
    for tr, val in TimeSeriesSplit(n_splits=5).split(X):
        model = XGBRegressor(**params).fit(X.iloc[tr], y.iloc[tr])
        rmses.append(float(np.sqrt(mean_squared_error(target_close.iloc[val], close.iloc[val] * (1.0 + model.predict(X.iloc[val]))))))
    return float(np.mean(rmses))


def objective_classifier(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = _suggest_xgb_params(trial)
    accs = []
    for tr, val in TimeSeriesSplit(n_splits=5).split(X):
        model = XGBClassifier(**params).fit(X.iloc[tr], y.iloc[tr])
        accs.append(float(accuracy_score(y.iloc[val], model.predict(X.iloc[val]))))
    return float(np.mean(accs))


def run_tuning(data_path: Union[str, Path] = BTCUSDT_1H_CSV, n_trials: int = 50) -> Dict[str, Any]:
    """Runs Optuna time-series CV hyperparameter search and updates verified artifacts/manifest."""
    df = _prepare_training_df(data_path)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    split = int(len(df) * 0.8)

    X1_tr, X1_te = df[REGRESSOR_FEATURES].iloc[:split], df[REGRESSOR_FEATURES].iloc[split:]
    y1_tr = df["Target_Return"].iloc[:split]
    c_tr, c_te = df["close"].iloc[:split], df["close"].iloc[split:]
    tc_tr, tc_te = df["Target_Close"].iloc[:split], df["Target_Close"].iloc[split:]

    study_reg = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_reg.optimize(lambda t: objective_regressor(t, X1_tr, y1_tr, c_tr, tc_tr), n_trials=n_trials)
    best_reg_params = {**study_reg.best_params, "tree_method": "hist", "device": "cpu", "random_state": 42}
    best_xgbr = XGBRegressor(**best_reg_params).fit(X1_tr, y1_tr)
    test_rmse = float(np.sqrt(mean_squared_error(tc_te, c_te * (1.0 + best_xgbr.predict(X1_te)))))

    X2_tr, X2_te = df[CLASSIFIER_FEATURES].iloc[:split], df[CLASSIFIER_FEATURES].iloc[split:]
    y2_tr, y2_te = df["Target_Movement"].iloc[:split], df["Target_Movement"].iloc[split:]
    study_clf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_clf.optimize(lambda t: objective_classifier(t, X2_tr, y2_tr), n_trials=n_trials)
    best_clf_params = {**study_clf.best_params, "tree_method": "hist", "device": "cpu", "random_state": 42}
    best_xgbc = XGBClassifier(**best_clf_params).fit(X2_tr, y2_tr)
    test_acc = float(accuracy_score(y2_te, best_xgbc.predict(X2_te)))

    save_production_models(
        best_xgbc, best_xgbr, CLASSIFIER_FEATURES, ACTIVE_MODEL_VERSION,
        holdout_df=df.iloc[split:],
        metrics={"tuned_test_accuracy": round(test_acc, 4), "tuned_test_rmse": round(test_rmse, 2)},
        regressor_features=REGRESSOR_FEATURES,
    )
    return {
        "regressor": {"best_params": best_reg_params, "test_rmse_price": test_rmse, "model": best_xgbr},
        "classifier": {"best_params": best_clf_params, "test_accuracy": test_acc, "model": best_xgbc},
    }


if __name__ == "__main__":
    train_and_save_models()
