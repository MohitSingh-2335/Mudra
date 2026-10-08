# src/model_training.py

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
import joblib
import os
import numpy as np
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from src.agents.sentiment_agent import merge_sentiment
from config import XGB_FEATURES, SVC_FEATURES, MODELS_DIR

def train_and_save_models(data_path, models_dir=MODELS_DIR):
    """
    Loads data, engineers features, trains the best models (XGBoost and SVC),
    and saves them.
    """
    print("Starting model training process...")
    df = load_and_clean_data(data_path)
    print("Data loaded and cleaned.")
    df = merge_fear_greed(df, timestamp_col='timestamp')
    df = merge_onchain(df, timestamp_col='timestamp')
    df = merge_sentiment(df)
    df = create_features(df)
    print("Features engineered.")

    # --- Regression Model Training (XGBoost) ---
    # Predicting Target_Return (% change to next close) instead of
    # Target_Close (absolute price). Every model tried on the absolute-price
    # target lost to a do-nothing baseline (see find_best_models.py results)
    # — this reframing is the fix, not just a tuning tweak.
    X1 = df[XGB_FEATURES]
    y1 = df['Target_Return']

    # Chronological split (NOT random) — data is time-ordered, so the last 20%
    # of rows becomes the held-out test set. This avoids leaking information
    # from "future" rows into training via rolling/lag features.
    split_idx_1 = int(len(X1) * 0.8)
    X1_train, X1_test = X1.iloc[:split_idx_1], X1.iloc[split_idx_1:]
    y1_train, y1_test = y1.iloc[:split_idx_1], y1.iloc[split_idx_1:]

    print("Training XGBoost Regressor for return prediction...")
    xgb_pipeline = make_pipeline(
        StandardScaler(),
        XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    )
    xgb_pipeline.fit(X1_train, y1_train)
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(xgb_pipeline, os.path.join(models_dir, 'best_xgb_model.pkl'))
    print("✅ XGBoost Regressor model saved.")

    # --- Evaluate on the held-out (chronologically later) test set ---
    # Report RMSE in PRICE terms (not raw return terms) by reconstructing the
    # implied price from each predicted return, so this stays comparable to
    # the RMSE numbers you've already seen.
    y1_pred_return = xgb_pipeline.predict(X1_test)
    current_close_test = df['close'].iloc[split_idx_1:]
    actual_price_test = df['Target_Close'].iloc[split_idx_1:]
    predicted_price_test = current_close_test * (1 + y1_pred_return)

    rmse = np.sqrt(mean_squared_error(actual_price_test, predicted_price_test))
    print(f"📊 XGBoost Regressor — Test RMSE, price-equivalent (chronological holdout): {rmse:.4f}")

    # --- Naive baseline: "next hour's close = this hour's close" (no model at all) ---
    naive_baseline_pred = current_close_test
    baseline_rmse = np.sqrt(mean_squared_error(actual_price_test, naive_baseline_pred))
    print(f"📊 Naive Baseline (persistence) — Test RMSE: {baseline_rmse:.4f}")
    if rmse < baseline_rmse:
        print(f"   ✅ Model beats the naive baseline (lower RMSE by {baseline_rmse - rmse:.4f}).")
    else:
        print(f"   ⚠️  Model does NOT beat the naive baseline (higher RMSE by {rmse - baseline_rmse:.4f}).")

    # --- Classification Model Training (SVC) ---
    X2 = df[SVC_FEATURES]
    y2 = df['Target_Movement']

    # Chronological split here too — same reasoning as above. Note: no more
    # `stratify=y2`, since stratification only makes sense for a random
    # sample; it doesn't apply to a fixed chronological cut.
    split_idx_2 = int(len(X2) * 0.8)
    X2_train, X2_test = X2.iloc[:split_idx_2], X2.iloc[split_idx_2:]
    y2_train, y2_test = y2.iloc[:split_idx_2], y2.iloc[split_idx_2:]

    print("Training SVC for movement prediction...")
    # Cap SVC training data at the most recent 15,000 rows — same convention
    # used in find_best_models.py / compare_models_backtest.py / significance_test.py
    # (see phase_1.md). Full ~28k-row rbf-kernel RandomizedSearchCV takes hours;
    # this keeps training time reasonable without changing the chronological
    # holdout used for evaluation.
    SVC_TRAIN_CAP = 15000
    X2_train_capped = X2_train.iloc[-SVC_TRAIN_CAP:]
    y2_train_capped = y2_train.iloc[-SVC_TRAIN_CAP:]

    # Scale the features for SVC
    scaler = StandardScaler()
    X2_train_scaled = scaler.fit_transform(X2_train_capped)
    X2_test_scaled = scaler.transform(X2_test)  # use train-fit scaler, don't refit on test

    # Define parameter grid for RandomizedSearch based on your notebook
    svc_param_grid = {
        'C': [1, 10, 50, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    
    # Using RandomizedSearchCV to find the best SVC
    random_search = RandomizedSearchCV(
        SVC(probability=True), # probability=True is needed for confidence scores
        param_distributions=svc_param_grid,
        n_iter=10, # As in the notebook
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X2_train_scaled, y2_train_capped)
    
    best_svc = random_search.best_estimator_

    joblib.dump(best_svc, os.path.join(models_dir, 'best_svc_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl')) # Save the scaler used for SVC
    print("✅ Best SVC model and scaler saved.")

    # --- Evaluate on the held-out (chronologically later) test set ---
    y2_pred = best_svc.predict(X2_test_scaled)
    acc = accuracy_score(y2_test, y2_pred)
    print(f"📊 SVC — Test Accuracy (chronological holdout): {acc:.4f}")

    # --- Naive baseline: always predict whichever class was most common in TRAINING data ---
    # (using training distribution, not test, so the baseline itself doesn't peek at test labels)
    majority_class = y2_train_capped.mode()[0]
    baseline_preds = pd.Series(majority_class, index=y2_test.index)
    baseline_acc = accuracy_score(y2_test, baseline_preds)
    print(f"📊 Naive Baseline (always predict '{majority_class}') — Test Accuracy: {baseline_acc:.4f}")
    if acc > baseline_acc:
        print(f"   ✅ Model beats the naive baseline by {(acc - baseline_acc) * 100:.2f} percentage points.")
    else:
        print(f"   ⚠️  Model does NOT clearly beat the naive baseline (diff: {(acc - baseline_acc) * 100:.2f} points).")

    print("\nTraining complete!")
    print("\n⚠️  These are the first metrics measured on a correct, non-leaky split.")
    print("    Any numbers you saw before this fix should be treated as invalid.")

if __name__ == '__main__':
    train_and_save_models(data_path='data/raw/BTCUSDT-1H.csv')