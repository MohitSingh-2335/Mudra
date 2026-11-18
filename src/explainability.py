# src/explainability.py
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import mlflow
except ImportError:
    mlflow = None

from config import (
    REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256,
    CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256,
    REGRESSOR_FEATURES, CLASSIFIER_FEATURES, FEATURED_BTC_DATA_PATH,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    SHAP_REGRESSOR_SUMMARY_PATH, SHAP_CLASSIFIER_SUMMARY_PATH, SHAP_IMPORTANCE_CSV_PATH,
)
from src.model_loader import load_verified_model


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def generate_global_explanations() -> None:
    if not os.path.exists(FEATURED_BTC_DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found at '{FEATURED_BTC_DATA_PATH}'.")

    Path(SHAP_REGRESSOR_SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURED_BTC_DATA_PATH)
    test_df = df.iloc[int(len(df) * 0.8):]

    # 1. Regressor
    xgbr = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
    X_reg = test_df[REGRESSOR_FEATURES]
    shap_vals_reg = shap.TreeExplainer(xgbr)(X_reg)
    fig_reg = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_reg, X_reg, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(SHAP_REGRESSOR_SUMMARY_PATH, dpi=150)
    plt.close(fig_reg)

    # 2. Classifier
    xgbc = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
    X_clf = test_df[CLASSIFIER_FEATURES]
    shap_vals_clf = shap.TreeExplainer(xgbc)(X_clf)
    fig_clf = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_clf, X_clf, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(SHAP_CLASSIFIER_SUMMARY_PATH, dpi=150)
    plt.close(fig_clf)

    # 3. Export Importance Table
    reg_imp = pd.DataFrame({"feature": REGRESSOR_FEATURES, "mean_abs_shap_regressor": np.abs(shap_vals_reg.values).mean(axis=0)})
    clf_imp = pd.DataFrame({"feature": CLASSIFIER_FEATURES, "mean_abs_shap_classifier": np.abs(shap_vals_clf.values).mean(axis=0)})
    imp_df = pd.merge(clf_imp, reg_imp, on="feature", how="outer").fillna(0).sort_values(by="mean_abs_shap_classifier", ascending=False)
    imp_df.to_csv(SHAP_IMPORTANCE_CSV_PATH, index=False)

    if mlflow is not None:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="shap_global_explainability"):
                mlflow.log_artifact(SHAP_REGRESSOR_SUMMARY_PATH)
                mlflow.log_artifact(SHAP_CLASSIFIER_SUMMARY_PATH)
                mlflow.log_artifact(SHAP_IMPORTANCE_CSV_PATH)
        except Exception:
            pass


def explain_single_prediction(
    explainer: shap.TreeExplainer, input_row_df: pd.DataFrame, top_n: int = 5, return_fig: bool = True
) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]], Optional[plt.Figure]]:
    exp = explainer(input_row_df)
    shap_vals, base_val = exp.values[0], float(exp.base_values[0])
    drivers = pd.DataFrame([{"feature": str(f), "shap": float(s), "value": float(v)} for f, s, v in zip(input_row_df.columns, shap_vals, input_row_df.iloc[0].values)])
    top_pos = drivers[drivers["shap"] > 0].sort_values("shap", ascending=False).head(top_n).to_dict("records")
    top_neg = drivers[drivers["shap"] < 0].sort_values("shap", ascending=True).head(top_n).to_dict("records")

    fig = None
    if return_fig:
        fig, _ = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(exp[0], max_display=8, show=False)
        plt.tight_layout()

    return base_val, top_pos, top_neg, fig


if __name__ == '__main__':
    generate_global_explanations()
