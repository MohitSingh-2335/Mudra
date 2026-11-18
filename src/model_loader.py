# src/model_loader.py
import io
import json
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib

from config import (
    REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256,
    CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256,
)

__all__ = [
    "validate_sha256_digest", "compute_file_sha256", "verify_model_artifact",
    "load_verified_model", "load_production_models", "save_production_models",
    "get_registered_models",
]

SHA256_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_sha256_digest(digest: str) -> None:
    if not isinstance(digest, str) or not SHA256_HEX_PATTERN.match(digest.strip()):
        raise ValueError(f"Invalid expected SHA-256 digest format: '{digest}'. Expected 64-character hexadecimal string.")


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Artifact file not found: {p}")
    sha256 = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_model_artifact(file_path: Union[str, Path], expected_sha256: str) -> None:
    validate_sha256_digest(expected_sha256)
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Model artifact not found at configured path: {file_path}")
    actual = compute_file_sha256(p)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(f"Cryptographic integrity verification failed for {file_path}: expected {expected_sha256}, got {actual}")


def load_verified_model(file_path: Union[str, Path], expected_sha256: str) -> Any:
    validate_sha256_digest(expected_sha256)
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Model artifact not found at configured path: {file_path}")
    raw_bytes = p.read_bytes()
    computed = hashlib.sha256(raw_bytes).hexdigest()
    if computed.lower() != expected_sha256.lower():
        raise RuntimeError(f"Cryptographic integrity verification failed for {file_path}: expected {expected_sha256}, got {computed}")
    return joblib.load(io.BytesIO(raw_bytes))


def load_production_models() -> Tuple[Any, Any]:
    reg = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
    clf = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
    if hasattr(reg, "set_params"):
        reg.set_params(device="cpu")
    if hasattr(clf, "set_params"):
        clf.set_params(device="cpu")
    return reg, clf


def save_production_models(
    clf_model: Any, reg_model: Any, features: List[str], version: str = "xgb_v1.1.0",
    holdout_df: Optional[Any] = None, metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from config import REGRESSOR_MODEL_PATH, CLASSIFIER_MODEL_PATH, MODEL_MANIFEST_PATH, MODELS_DIR
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf_model, CLASSIFIER_MODEL_PATH)
    joblib.dump(reg_model, REGRESSOR_MODEL_PATH)
    clf_hash = compute_file_sha256(CLASSIFIER_MODEL_PATH)
    reg_hash = compute_file_sha256(REGRESSOR_MODEL_PATH)

    comp_metrics: Dict[str, Any] = {}
    if holdout_df is not None:
        if "Target_Movement" in holdout_df.columns:
            from sklearn.metrics import accuracy_score, log_loss
            y_dir = holdout_df["Target_Movement"].astype(int).values
            probs = clf_model.predict_proba(holdout_df[features])[:, 1]
            comp_metrics["holdout_accuracy"] = round(float(accuracy_score(y_dir, (probs >= 0.5).astype(int))), 4)
            comp_metrics["holdout_log_loss"] = round(float(log_loss(y_dir, probs)), 4)
        if "Target_Return" in holdout_df.columns:
            from sklearn.metrics import mean_absolute_error
            comp_metrics["holdout_mae"] = round(float(mean_absolute_error(holdout_df["Target_Return"].values, reg_model.predict(holdout_df[features]))), 6)
    if metrics:
        comp_metrics.update(metrics)

    manifest = {
        "model_version": version, "feature_schema_version": f"{version}_{len(features)}stationary",
        "features": list(features), "classifier_features": list(features), "regressor_features": list(features),
        "classifier_sha256": clf_hash, "regressor_sha256": reg_hash, "metrics": comp_metrics,
    }
    with open(MODEL_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def get_registered_models() -> Dict[str, Dict[str, Any]]:
    from config import MODELS_DIR, ACTIVE_MODEL_VERSION, REGRESSOR_MODEL_PATH, CLASSIFIER_MODEL_PATH
    models_dir = Path(MODELS_DIR)
    if not models_dir.exists():
        return {}

    registered = {}
    for m_file in sorted(list(models_dir.glob("*manifest*.json"))):
        try:
            with open(m_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("model_version")
            if not version:
                continue
            clf_path = data.get("classifier_artifact_path") or CLASSIFIER_MODEL_PATH
            reg_path = data.get("regressor_artifact_path") or REGRESSOR_MODEL_PATH
            if not Path(clf_path).is_file() or not Path(reg_path).is_file():
                continue
            clf_m = joblib.load(clf_path)
            reg_m = joblib.load(reg_path)
            if hasattr(clf_m, "set_params"):
                clf_m.set_params(device="cpu")
            if hasattr(reg_m, "set_params"):
                reg_m.set_params(device="cpu")

            registered[version] = {
                "model_version": version,
                "status": data.get("status", "ACTIVE" if version == ACTIVE_MODEL_VERSION else "CANDIDATE"),
                "manifest_path": str(m_file), "classifier": clf_m, "regressor": reg_m,
                "features": data.get("features") or data.get("classifier_features") or [],
                "metrics": data.get("metrics", {}), "manifest_data": data,
            }
        except Exception:
            continue
    return registered
