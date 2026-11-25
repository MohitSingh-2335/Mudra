# src/explainability.py
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import mlflow
except ImportError:
    mlflow = None

from config import (
    CLASSIFIER_FEATURES,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_MODEL_SHA256,
    FEATURED_BTC_DATA_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGRESSOR_FEATURES,
    REGRESSOR_MODEL_PATH,
    REGRESSOR_MODEL_SHA256,
    SHAP_CLASSIFIER_SUMMARY_PATH,
    SHAP_IMPORTANCE_CSV_PATH,
    SHAP_REGRESSOR_SUMMARY_PATH,
)
from src.model_loader import load_verified_model, restore_estimator_boosters


def generate_global_explanations() -> None:
    if not os.path.exists(FEATURED_BTC_DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found at '{FEATURED_BTC_DATA_PATH}'.")

    Path(SHAP_REGRESSOR_SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURED_BTC_DATA_PATH)
    test_df = df.iloc[int(len(df) * 0.8):]

    xgbr = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
    X_reg = test_df[REGRESSOR_FEATURES]
    shap_vals_reg = shap.TreeExplainer(xgbr)(X_reg)
    fig_reg = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_reg, X_reg, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(SHAP_REGRESSOR_SUMMARY_PATH, dpi=150)
    plt.close(fig_reg)

    xgbc = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
    X_clf = test_df[CLASSIFIER_FEATURES]
    shap_vals_clf = shap.TreeExplainer(xgbc)(X_clf)
    fig_clf = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_clf, X_clf, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(SHAP_CLASSIFIER_SUMMARY_PATH, dpi=150)
    plt.close(fig_clf)

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
    explainer: shap.TreeExplainer, input_row_df: pd.DataFrame, top_n: int = 5, return_fig: bool = False,
) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]], Optional[plt.Figure]]:
    exp = explainer(input_row_df)
    shap_vals, base_val = exp.values[0], float(exp.base_values[0])
    drivers = pd.DataFrame([
        {"feature": str(f), "shap": float(s), "value": float(v)}
        for f, s, v in zip(input_row_df.columns, shap_vals, input_row_df.iloc[0].values)
    ])
    top_pos = drivers[drivers["shap"] > 0].sort_values("shap", ascending=False).head(top_n).to_dict("records")
    top_neg = drivers[drivers["shap"] < 0].sort_values("shap", ascending=True).head(top_n).to_dict("records")

    fig = None
    if return_fig:
        fig, _ = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(exp[0], max_display=8, show=False)
        plt.tight_layout()
    return base_val, top_pos, top_neg, fig


def get_top_shap_drivers(model: Any, X_row: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
    """Extracts top-k features with strongest absolute SHAP impact for one prediction row."""
    if model is None or X_row is None or X_row.empty:
        return []
    try:
        estimator = model
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            cc = model.calibrated_classifiers_[0]
            estimator = getattr(cc, "estimator", getattr(cc, "base_estimator", getattr(model, "base_estimator", model)))
        elif hasattr(model, "base_estimator"):
            estimator = getattr(model, "base_estimator", model)
        elif hasattr(model, "_final_estimator"):
            estimator = getattr(model, "_final_estimator", model)

        restore_estimator_boosters(estimator)
        cols = getattr(estimator, "feature_names_in_", None)
        X_eval = X_row[list(cols)] if (cols is not None and all(c in X_row.columns for c in cols)) else X_row

        exp = shap.TreeExplainer(estimator)(X_eval)
        vals = exp.values
        if vals.ndim == 3:
            vals = vals[0, :, 1] if vals.shape[2] > 1 else vals[0, :, 0]
        elif vals.ndim == 2:
            vals = vals[0]

        drivers = [
            {"feature": str(feat), "value": round(float(X_eval.iloc[0][feat]), 4), "shap_impact": round(float(sv), 5)}
            for feat, sv in zip(X_eval.columns, vals)
        ]
        drivers.sort(key=lambda d: abs(d["shap_impact"]), reverse=True)
        return drivers[:top_k]
    except Exception:
        return []


if __name__ == "__main__":
    generate_global_explanations()
