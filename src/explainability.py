# src/explainability.py

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Guard project root resolution for standalone CLI execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False

from config import (
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    SHAP_REGRESSOR_SUMMARY_PATH,
    SHAP_CLASSIFIER_SUMMARY_PATH,
    SHAP_IMPORTANCE_CSV_PATH
)


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """
    Initializes a native TreeExplainer for an XGBoost model.

    TreeSHAP computes exact Shapley values in polynomial time O(TLD^2)
    rather than exponential time O(2^M), satisfying efficiency, symmetry,
    dummy, and additivity axioms.
    """
    return shap.TreeExplainer(model)


def generate_global_explanations() -> None:
    """
    Computes global SHAP values on the holdout test set for both Regressor and Classifier.
    Saves beeswarm summary plots and a feature importance CSV table.
    """
    print("[INFO] Generating Global SHAP Explainability Reports...")

    if not os.path.exists(FEATURED_BTC_DATA_PATH):
        raise FileNotFoundError(
            f"[ERROR] Processed dataset not found at '{FEATURED_BTC_DATA_PATH}'. Run data preparation first."
        )
    if not os.path.exists(REGRESSOR_MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] Regressor model not found at '{REGRESSOR_MODEL_PATH}'. Run model training first."
        )
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] Classifier model not found at '{CLASSIFIER_MODEL_PATH}'. Run model training first."
        )

    Path(SHAP_REGRESSOR_SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(SHAP_IMPORTANCE_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURED_BTC_DATA_PATH)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]

    # 1. Regressor Global SHAP
    print("[INFO] Evaluating Regressor TreeSHAP values...")
    xgbr = joblib.load(REGRESSOR_MODEL_PATH)
    X_reg = test_df[REGRESSOR_FEATURES]
    explainer_reg = shap.TreeExplainer(xgbr)
    shap_values_reg = explainer_reg(X_reg)

    # Save Regressor Beeswarm Summary Plot
    fig_reg = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_reg, X_reg, show=False, max_display=15)
    plt.title("SHAP Beeswarm: XGBoost Regressor (Price Return Impact)", fontsize=13)
    plt.tight_layout()
    plt.savefig(SHAP_REGRESSOR_SUMMARY_PATH, dpi=150)
    plt.close(fig_reg)
    print(f"[SUCCESS] Regressor summary plot saved to {SHAP_REGRESSOR_SUMMARY_PATH}")

    # 2. Classifier Global SHAP
    print("[INFO] Evaluating Classifier TreeSHAP values...")
    xgbc = joblib.load(CLASSIFIER_MODEL_PATH)
    X_clf = test_df[CLASSIFIER_FEATURES]
    explainer_clf = shap.TreeExplainer(xgbc)
    shap_values_clf = explainer_clf(X_clf)

    # Save Classifier Beeswarm Summary Plot
    fig_clf = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_clf, X_clf, show=False, max_display=15)
    plt.title("SHAP Beeswarm: XGBoost Classifier (Directional Impact)", fontsize=13)
    plt.tight_layout()
    plt.savefig(SHAP_CLASSIFIER_SUMMARY_PATH, dpi=150)
    plt.close(fig_clf)
    print(f"[SUCCESS] Classifier summary plot saved to {SHAP_CLASSIFIER_SUMMARY_PATH}")

    # 3. Export Mean Absolute SHAP Importance Table
    mean_abs_reg = np.abs(shap_values_reg.values).mean(axis=0)
    mean_abs_clf = np.abs(shap_values_clf.values).mean(axis=0)

    reg_importance = pd.DataFrame({"feature": REGRESSOR_FEATURES, "mean_abs_shap_regressor": mean_abs_reg})
    clf_importance = pd.DataFrame({"feature": CLASSIFIER_FEATURES, "mean_abs_shap_classifier": mean_abs_clf})

    importance_df = pd.merge(clf_importance, reg_importance, on="feature", how="outer").fillna(0)
    importance_df = importance_df.sort_values(by="mean_abs_shap_classifier", ascending=False)
    importance_df.to_csv(SHAP_IMPORTANCE_CSV_PATH, index=False)
    print(f"[SUCCESS] Feature importance ranking table saved to {SHAP_IMPORTANCE_CSV_PATH}")

    # 4. Log SHAP artifacts to MLflow
    if MLFLOW_AVAILABLE and mlflow is not None:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="shap_global_explainability"):
                mlflow.set_tag("task", "SHAP_feature_attribution")
                mlflow.log_artifact(SHAP_REGRESSOR_SUMMARY_PATH)
                mlflow.log_artifact(SHAP_CLASSIFIER_SUMMARY_PATH)
                mlflow.log_artifact(SHAP_IMPORTANCE_CSV_PATH)
                top_features = importance_df["feature"].head(3).tolist()
                mlflow.set_tag("top_directional_drivers", ", ".join(top_features))
                print("[SUCCESS] SHAP plots and tables logged into MLflow.")
        except Exception as e:
            print(f"[WARNING] MLflow tracking encountered an error: {e}")
    else:
        print("[INFO] MLflow logging skipped (mlflow package not installed).")

    print("[SUCCESS] Global SHAP Explainability Engine execution complete.")


def explain_single_prediction(
    explainer: shap.TreeExplainer,
    input_row_df: pd.DataFrame,
    top_n: int = 5,
    return_fig: bool = True
) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]], Optional[plt.Figure]]:
    """
    Computes local SHAP explanation for a single prediction row.

    Parameters:
        explainer: Pre-fitted shap.TreeExplainer instance.
        input_row_df: Single-row DataFrame containing the required feature vector.
        top_n: Number of leading positive and negative drivers to extract.
        return_fig: Whether to render and return the matplotlib waterfall figure.

    Returns:
        Tuple containing:
          - base_value (float): Model's expected value E[f(X)] across the background population.
          - top_positive (List[Dict]): Top features pushing the output upward (feature, shap, value).
          - top_negative (List[Dict]): Top features pulling the output downward (feature, shap, value).
          - fig (Optional[plt.Figure]): Matplotlib waterfall chart for UI rendering.
    """
    explanation = explainer(input_row_df)
    shap_vals = explanation.values[0]
    base_val = float(explanation.base_values[0])
    feature_names = input_row_df.columns.tolist()
    feature_vals = input_row_df.iloc[0].values

    # Collect drivers
    drivers: List[Dict[str, Any]] = []
    for f_name, s_val, a_val in zip(feature_names, shap_vals, feature_vals):
        drivers.append({"feature": str(f_name), "shap": float(s_val), "value": float(a_val)})

    drivers_df = pd.DataFrame(drivers)
    top_pos = drivers_df[drivers_df["shap"] > 0].sort_values(by="shap", ascending=False).head(top_n).to_dict("records")
    top_neg = drivers_df[drivers_df["shap"] < 0].sort_values(by="shap", ascending=True).head(top_n).to_dict("records")

    # Generate waterfall figure
    fig: Optional[plt.Figure] = None
    if return_fig:
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(explanation[0], max_display=8, show=False)
        plt.tight_layout()

    return base_val, top_pos, top_neg, fig


if __name__ == '__main__':
    generate_global_explanations()
