# src/explainability.py

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mlflow

from config import (
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH,
    SHAP_REGRESSOR_SUMMARY_PATH,
    SHAP_CLASSIFIER_SUMMARY_PATH,
    SHAP_IMPORTANCE_CSV_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME
)


def get_tree_explainer(model):
    """Initializes a native TreeExplainer for an XGBoost model."""
    return shap.TreeExplainer(model)


def generate_global_explanations():
    """
    Computes global SHAP values on the holdout test set for both Regressor and Classifier.
    Saves beeswarm summary plots and a feature importance CSV table.
    """
    print("🔍 Generating Global SHAP Explainability Reports...")
    df = pd.read_csv(FEATURED_BTC_DATA_PATH)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]

    # 1. Regressor Global SHAP
    print("   Evaluating Regressor TreeSHAP values...")
    xgbr = joblib.load(REGRESSOR_MODEL_PATH)
    X_reg = test_df[REGRESSOR_FEATURES]
    explainer_reg = shap.TreeExplainer(xgbr)
    shap_values_reg = explainer_reg(X_reg)

    # Save Regressor Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_reg, X_reg, show=False, max_display=15)
    plt.title("SHAP Beeswarm: XGBoost Regressor (Price Return Impact)", fontsize=13)
    plt.tight_layout()
    plt.savefig(SHAP_REGRESSOR_SUMMARY_PATH, dpi=150)
    plt.close()
    print(f"   ✅ Regressor summary plot saved to {SHAP_REGRESSOR_SUMMARY_PATH}")

    # 2. Classifier Global SHAP
    print("   Evaluating Classifier TreeSHAP values...")
    xgbc = joblib.load(CLASSIFIER_MODEL_PATH)
    X_clf = test_df[CLASSIFIER_FEATURES]
    explainer_clf = shap.TreeExplainer(xgbc)
    shap_values_clf = explainer_clf(X_clf)

    # Save Classifier Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_clf, X_clf, show=False, max_display=15)
    plt.title("SHAP Beeswarm: XGBoost Classifier (Directional Impact)", fontsize=13)
    plt.tight_layout()
    plt.savefig(SHAP_CLASSIFIER_SUMMARY_PATH, dpi=150)
    plt.close()
    print(f"   ✅ Classifier summary plot saved to {SHAP_CLASSIFIER_SUMMARY_PATH}")

    # 3. Export Mean Absolute SHAP Importance Table
    mean_abs_reg = np.abs(shap_values_reg.values).mean(axis=0)
    mean_abs_clf = np.abs(shap_values_clf.values).mean(axis=0)

    reg_importance = pd.DataFrame({"feature": REGRESSOR_FEATURES, "mean_abs_shap_regressor": mean_abs_reg})
    clf_importance = pd.DataFrame({"feature": CLASSIFIER_FEATURES, "mean_abs_shap_classifier": mean_abs_clf})

    importance_df = pd.merge(clf_importance, reg_importance, on="feature", how="outer").fillna(0)
    importance_df = importance_df.sort_values(by="mean_abs_shap_classifier", ascending=False)
    importance_df.to_csv(SHAP_IMPORTANCE_CSV_PATH, index=False)
    print(f"   ✅ Feature importance ranking table saved to {SHAP_IMPORTANCE_CSV_PATH}")

    # 4. Log SHAP artifacts to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="shap_global_explainability"):
        mlflow.set_tag("task", "SHAP_feature_attribution")
        mlflow.log_artifact(SHAP_REGRESSOR_SUMMARY_PATH)
        mlflow.log_artifact(SHAP_CLASSIFIER_SUMMARY_PATH)
        mlflow.log_artifact(SHAP_IMPORTANCE_CSV_PATH)
        # Log top 3 drivers as tags
        top_features = importance_df["feature"].head(3).tolist()
        mlflow.set_tag("top_directional_drivers", ", ".join(top_features))
        print("   ✅ SHAP plots and tables logged into MLflow.")

    print("\n🎉 Global SHAP Explainability Engine execution complete!")


def explain_single_prediction(explainer, input_row_df, top_n=5):
    """
    Computes local SHAP explanation for a single prediction row.
    Returns:
      - base_value: expected baseline output
      - prediction_value: model output
      - top_positive: list of (feature_name, shap_value, actual_value)
      - top_negative: list of (feature_name, shap_value, actual_value)
      - fig: matplotlib waterfall plot figure
    """
    explanation = explainer(input_row_df)
    shap_vals = explanation.values[0]
    base_val = explanation.base_values[0]
    feature_names = input_row_df.columns.tolist()
    feature_vals = input_row_df.iloc[0].values

    # Collect drivers
    drivers = []
    for f_name, s_val, a_val in zip(feature_names, shap_vals, feature_vals):
        drivers.append({"feature": f_name, "shap": s_val, "value": a_val})

    drivers_df = pd.DataFrame(drivers)
    top_pos = drivers_df[drivers_df["shap"] > 0].sort_values(by="shap", ascending=False).head(top_n).to_dict("records")
    top_neg = drivers_df[drivers_df["shap"] < 0].sort_values(by="shap", ascending=True).head(top_n).to_dict("records")

    # Generate waterfall figure
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(explanation[0], max_display=8, show=False)
    plt.tight_layout()

    return base_val, top_pos, top_neg, fig


if __name__ == '__main__':
    generate_global_explanations()
