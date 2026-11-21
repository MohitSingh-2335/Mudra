import io
import json
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from config import (
    REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256,
    CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256,
)

__all__ = [
    "validate_sha256_digest", "compute_file_sha256", "verify_model_artifact",
    "load_verified_model", "load_production_models", "save_production_models",
    "get_registered_models",
]

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_sha256_digest(digest: str) -> None:
    # verify expected hash is a valid 64-character hex string
    if not isinstance(digest, str) or not SHA256_PATTERN.match(digest.strip()):
        raise ValueError(f"Invalid expected SHA-256 digest format: '{digest}'. Expected 64-character hexadecimal string.")


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    # computes sha256 hash in 64kb chunks
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Artifact file not found: {p}")
    sha256 = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_model_artifact(file_path: Union[str, Path], expected_sha256: str) -> None:
    # verifies stored model integrity against expected sha256
    validate_sha256_digest(expected_sha256)
    actual = compute_file_sha256(file_path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(f"Cryptographic integrity verification failed for {file_path}: expected {expected_sha256}, got {actual}")


def load_verified_model(file_path: Union[str, Path], expected_sha256: str) -> Any:
    # verifies hash and deserializes model from verified bytes
    validate_sha256_digest(expected_sha256)
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Model artifact not found at configured path: {file_path}")
    raw_bytes = p.read_bytes()
    computed = hashlib.sha256(raw_bytes).hexdigest()
    if computed.lower() != expected_sha256.lower():
        raise RuntimeError(f"Cryptographic integrity verification failed for {file_path}: expected {expected_sha256}, got {computed}")
    return joblib.load(io.BytesIO(raw_bytes))


def restore_estimator_boosters(m: Any) -> None:
    # restores booster handles for XGBoost (CPU mode) and LightGBM (Windows C++ handles)
    if m.__class__.__name__.startswith("XGB") and hasattr(m, "set_params"):
        try:
            m.set_params(device="cpu")
        except Exception:
            pass
    if lgb is not None:
        try:
            if hasattr(m, "calibrated_classifiers_"):
                for cc in getattr(m, "calibrated_classifiers_", []):
                    est = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
                    if est is not None and hasattr(est, "_Booster") and est._Booster is not None:
                        est._Booster = lgb.Booster(model_str=est._Booster.model_to_string())
            if m.__class__.__name__.startswith("LGBM") and hasattr(m, "_Booster") and m._Booster is not None:
                m._Booster = lgb.Booster(model_str=m._Booster.model_to_string())
        except Exception:
            pass


def load_production_models() -> Tuple[Any, Any]:
    # loads regressor and classifier with cpu device enforcement
    reg = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
    clf = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
    for model in (reg, clf):
        restore_estimator_boosters(model)
    return reg, clf


def save_production_models(
    clf_model: Any, reg_model: Any, features: List[str], version: str = "xgb_v1.1.0",
    holdout_df: Optional[Any] = None, metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # persists models, generates sha256 digests, and writes model manifest
    from config import MODEL_MANIFEST_PATH, MODELS_DIR
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf_model, CLASSIFIER_MODEL_PATH)
    joblib.dump(reg_model, REGRESSOR_MODEL_PATH)

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
        "model_version": version,
        "feature_schema_version": f"{version}_{len(features)}stationary",
        "features": list(features),
        "classifier_features": list(features),
        "regressor_features": list(features),
        "classifier_sha256": compute_file_sha256(CLASSIFIER_MODEL_PATH),
        "regressor_sha256": compute_file_sha256(REGRESSOR_MODEL_PATH),
        "metrics": comp_metrics,
    }
    with open(MODEL_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def get_registered_models() -> Dict[str, Dict[str, Any]]:
    # discovers all registered model versions from manifest files
    from config import MODELS_DIR, ACTIVE_MODEL_VERSION
    models_dir = Path(MODELS_DIR)
    if not models_dir.exists():
        return {}

    registered = {}
    for m_file in sorted(models_dir.glob("*manifest*.json")):
        try:
            with open(m_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("model_version")
            if not version:
                continue
            clf_path = data.get("classifier_artifact_path") or CLASSIFIER_MODEL_PATH
            reg_path = data.get("regressor_artifact_path") or REGRESSOR_MODEL_PATH
            from config import BASE_DIR
            if not Path(clf_path).is_file():
                clf_path = str(BASE_DIR / clf_path)
            if not Path(reg_path).is_file():
                reg_path = str(BASE_DIR / reg_path)
            if not Path(clf_path).is_file() or not Path(reg_path).is_file():
                continue

            # Verify integrity using SHA-256 digest when present in manifest
            clf_sha = data.get("classifier_sha256")
            reg_sha = data.get("regressor_sha256")
            clf_m = load_verified_model(clf_path, clf_sha) if clf_sha else joblib.load(clf_path)
            reg_m = load_verified_model(reg_path, reg_sha) if reg_sha else joblib.load(reg_path)
            for m in (clf_m, reg_m):
                restore_estimator_boosters(m)

            registered[version] = {
                "model_version": version,
                "status": data.get("status", "ACTIVE" if version == ACTIVE_MODEL_VERSION else "CANDIDATE"),
                "manifest_path": str(m_file),
                "classifier": clf_m,
                "regressor": reg_m,
                "features": data.get("features") or data.get("classifier_features") or [],
                "metrics": data.get("metrics", {}),
                "manifest_data": data,
            }
        except Exception:
            continue
    return registered
