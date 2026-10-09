# src/model_training.py

import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
import joblib
import numpy as np
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
import mlflow
import mlflow.xgboost
from config import (
    REGRESSOR_FEATURES, 
    CLASSIFIER_FEATURES, 
    REGRESSOR_MODEL_PATH, 
    CLASSIFIER_MODEL_PATH, 
    MODELS_DIR, 
    BTCUSDT_1H_CSV,
    MLFLOW_TRACKING_URI, 
    MLFLOW_EXPERIMENT_NAME
)

import sys
sys.stdout.reconfigure(encoding='utf-8')


def train_and_save_models(data_path, models_dir=MODELS_DIR):
    """
    Loads data, engineers features, trains the best models (XGBoostRegressor and XGBClassifier),
    and saves them.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print("Starting model training process...")
    df = load_and_clean_data(data_path)
    print("Data loaded and cleaned.")
    df = merge_fear_greed(df, timestamp_col='timestamp')
    df = merge_onchain(df, timestamp_col='timestamp')
    df = create_features(df)
    print("Features engineered.")

    # --- Regression Model Training (XGBoostRegressor) ---
    # Predicting Target_Return (% change to next close) instead of
    # Target_Close (absolute price). Every model tried on the absolute-price
    # target lost to a do-nothing baseline (see find_best_models.py results)
    # — this reframing is the fix, not just a tuning tweak.
    X1 = df[REGRESSOR_FEATURES]
    y1 = df['Target_Return']

    # Chronological split (NOT random) — data is time-ordered, so the last 20%
    # of rows becomes the held-out test set. This avoids leaking information
    # from "future" rows into training via rolling/lag features.
    split_idx_1 = int(len(X1) * 0.8)
    X1_train, X1_test = X1.iloc[:split_idx_1], X1.iloc[split_idx_1:]
    y1_train, y1_test = y1.iloc[:split_idx_1], y1.iloc[split_idx_1:]

    print("Training XGBoost Regressor for return prediction...")
    with mlflow.start_run(run_name="xgboost_regressor_baseline"):
        mlflow.set_tag("model_type", "XGBRegressor")
        mlflow.set_tag("target", "Target_Return")
        reg_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 5,
            "tree_method": "hist",
            "device": "cuda",
            "random_state": 42
        }
        mlflow.log_params(reg_params)
        mlflow.log_param("num_features", len(REGRESSOR_FEATURES))
        mlflow.log_param("train_samples", len(X1_train))
        mlflow.log_param("test_samples", len(X1_test))
        xgbr = XGBRegressor(**reg_params)
        xgbr.fit(X1_train, y1_train)
        joblib.dump(xgbr, REGRESSOR_MODEL_PATH)
        print("✅ XGBoost Regressor model saved.")

        # --- Evaluate on the held-out (chronologically later) test set ---
        # Report RMSE in PRICE terms (not raw return terms) by reconstructing the
        # implied price from each predicted return, so this stays comparable to
        # the RMSE numbers you've already seen.
        y1_pred_return = xgbr.predict(X1_test)
        current_close_test = df['close'].iloc[split_idx_1:]
        actual_price_test = df['Target_Close'].iloc[split_idx_1:]
        predicted_price_test = current_close_test * (1 + y1_pred_return)

        rmse = np.sqrt(mean_squared_error(actual_price_test, predicted_price_test))
        print(f"📊 XGBoost Regressor — Test RMSE, price-equivalent (chronological holdout): {rmse:.4f}")

        # --- Naive baseline: "next hour's close = this hour's close" (no model at all) ---
        naive_baseline_pred = current_close_test
        baseline_rmse = np.sqrt(mean_squared_error(actual_price_test, naive_baseline_pred))
        mlflow.log_metrics({
                "test_rmse_price": rmse,
                "naive_baseline_rmse": baseline_rmse,
                "rmse_diff": baseline_rmse - rmse
            })
        mlflow.xgboost.log_model(xgbr, artifact_path="model")
        print(f"📊 Naive Baseline (persistence) — Test RMSE: {baseline_rmse:.4f}")
        if rmse < baseline_rmse:
            print(f"   ✅ Model beats the naive baseline (lower RMSE by {baseline_rmse - rmse:.4f}).")
        else:
            print(f"   ⚠️  Model does NOT beat the naive baseline (higher RMSE by {rmse - baseline_rmse:.4f}).")

    # --- Classification Model Training (XGBoostClassifier) ---
    X2 = df[CLASSIFIER_FEATURES]
    y2 = df['Target_Movement']

    # Chronological split here too — same reasoning as above. Note: no more
    # `stratify=y2`, since stratification only makes sense for a random
    # sample; it doesn't apply to a fixed chronological cut.
    split_idx_2 = int(len(X2) * 0.8)
    X2_train, X2_test = X2.iloc[:split_idx_2], X2.iloc[split_idx_2:]
    y2_train, y2_test = y2.iloc[:split_idx_2], y2.iloc[split_idx_2:]

    print("Training XGBoost Classifier for movement prediction...")
    with mlflow.start_run(run_name="xgboost_classifier_baseline"):
        mlflow.set_tag("model_type", "XGBClassifier")
        mlflow.set_tag("target", "Target_Movement")
        clf_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "device": "cuda",
            "random_state": 42
        }
        mlflow.log_params(clf_params)
        mlflow.log_param("num_features", len(CLASSIFIER_FEATURES))
        mlflow.log_param("train_samples", len(X2_train))
        mlflow.log_param("test_samples", len(X2_test))
        xgbc = XGBClassifier(**clf_params)
        xgbc.fit(X2_train, y2_train)
        joblib.dump(xgbc, CLASSIFIER_MODEL_PATH)
        print("✅ XGBoost Classifier model saved.")

        # --- Evaluate on the held-out (chronologically later) test set ---
        y2_pred = xgbc.predict(X2_test)
        acc = accuracy_score(y2_test, y2_pred)
        print(f"📊 XGBoost — Test Accuracy (chronological holdout): {acc:.4f}")

        # --- Naive baseline ---
        majority_class = y2_train.mode()[0]
        baseline_preds = pd.Series(majority_class, index=y2_test.index)
        baseline_acc = accuracy_score(y2_test, baseline_preds)
        mlflow.log_metrics({
                "test_accuracy": acc,
                "naive_baseline_accuracy": baseline_acc,
                "accuracy_gain_pct": (acc - baseline_acc) * 100
            })
        mlflow.xgboost.log_model(xgbc, artifact_path="model")
        print(f"📊 Naive Baseline (always predict '{majority_class}') — Test Accuracy: {baseline_acc:.4f}")
        if acc > baseline_acc:
            print(f"   ✅ Model beats the naive baseline by {(acc - baseline_acc) * 100:.2f} percentage points.")
        else:
            print(f"   ⚠️  Model does NOT clearly beat the naive baseline (diff: {(acc - baseline_acc) * 100:.2f} points).")

    print("\nTraining complete!")
    print("\n⚠️  These are the first metrics measured on a correct, non-leaky split.")
    print("    Any numbers you saw before this fix should be treated as invalid.")

if __name__ == '__main__':
    train_and_save_models(data_path=BTCUSDT_1H_CSV)