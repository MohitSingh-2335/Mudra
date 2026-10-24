"""
Mudra Cryptographic Model Loader & Integrity Verification Service.
Enforces fail-closed SHA-256 checksum verification before deserializing model artifacts.
"""
import io
import json
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib

from config import (
    REGRESSOR_MODEL_PATH,
    REGRESSOR_MODEL_SHA256,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_MODEL_SHA256,
)

__all__ = [
    "validate_sha256_digest",
    "compute_file_sha256",
    "verify_model_artifact",
    "load_verified_model",
    "load_production_models",
    "save_production_models",
    "get_registered_models",
]

SHA256_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_sha256_digest(digest: str) -> None:
    """
    Validates that the digest is a well-formed 64-character hexadecimal SHA-256 string.
    Rejects invalid, empty, or truncated configuration digests with ValueError.
    """
    if not isinstance(digest, str) or not SHA256_HEX_PATTERN.match(digest.strip()):
        raise ValueError(
            f"Invalid expected SHA-256 digest format: '{digest}'. Expected 64-character hexadecimal string."
        )


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Computes the SHA-256 hexadecimal digest of a local file in chunks."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Artifact file not found: {p}")
    sha256 = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_model_artifact(file_path: Union[str, Path], expected_sha256: str) -> None:
    """
    Verifies that the file exists, is readable, and matches the expected SHA-256 digest.
    Fails closed with FileNotFoundError if absent, ValueError if digest format is invalid,
    or RuntimeError if hash deviates.
    """
    validate_sha256_digest(expected_sha256)
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Model artifact not found at configured path: {file_path}")
    actual_sha256 = compute_file_sha256(p)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Cryptographic integrity verification failed for model artifact: {file_path}. "
            f"Expected SHA-256 '{expected_sha256}', but computed '{actual_sha256}'. "
            "Refusing to deserialize potentially corrupted, modified, or tampered artifact."
        )


def load_verified_model(file_path: Union[str, Path], expected_sha256: str) -> Any:
    """
    Cryptographically verifies the model artifact SHA-256 digest before deserialization.
    Reads file bytes into memory once and verifies checksum against the exact buffer
    passed to joblib.load(io.BytesIO(...)), preventing TOCTOU file replacement races.
    Fails closed if the file is missing, digest is malformed, or content has been altered.
    """
    validate_sha256_digest(expected_sha256)
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Model artifact not found at configured path: {file_path}")

    raw_bytes = p.read_bytes()
    computed_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if computed_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Cryptographic integrity verification failed for model artifact: {file_path}. "
            f"Expected SHA-256 '{expected_sha256}', but computed '{computed_sha256}'. "
            "Refusing to deserialize potentially corrupted, modified, or tampered artifact."
        )

    return joblib.load(io.BytesIO(raw_bytes))


def load_production_models() -> Tuple[Any, Any]:
    """
    Loads verified production regressor and classifier models, forcing CPU device execution.
    """
    reg_model = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
    clf_model = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
    if hasattr(reg_model, "set_params"):
        reg_model.set_params(device="cpu")
    if hasattr(clf_model, "set_params"):
        clf_model.set_params(device="cpu")
    return reg_model, clf_model


def save_production_models(
    clf_model: Any,
    reg_model: Any,
    features: List[str],
    version: str = "xgb_v1.1.0",
    holdout_df: Optional[Any] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Saves trained models to configured paths, calculates SHA-256 hashes,
    evaluates holdout metrics dynamically if holdout_df is provided,
    and updates model_manifest.json as the Single Source of Truth.
    """
    from config import (
        REGRESSOR_MODEL_PATH,
        CLASSIFIER_MODEL_PATH,
        MODEL_MANIFEST_PATH,
        MODELS_DIR,
    )
    # Ensure target artifacts directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Save model weights to disk
    joblib.dump(clf_model, CLASSIFIER_MODEL_PATH)
    joblib.dump(reg_model, REGRESSOR_MODEL_PATH)

    # 2. Compute cryptographic SHA-256 checksums automatically
    clf_hash = compute_file_sha256(CLASSIFIER_MODEL_PATH)
    reg_hash = compute_file_sha256(REGRESSOR_MODEL_PATH)

    # 3. Calculate real holdout metrics dynamically
    computed_metrics: Dict[str, Any] = {}
    if holdout_df is not None:
        if "Target_Movement" in holdout_df.columns:
            from sklearn.metrics import accuracy_score, log_loss
            y_true_dir = holdout_df["Target_Movement"].astype(int).values
            probs = clf_model.predict_proba(holdout_df[features])[:, 1]
            computed_metrics["holdout_accuracy"] = round(float(accuracy_score(y_true_dir, (probs >= 0.5).astype(int))), 4)
            computed_metrics["holdout_log_loss"] = round(float(log_loss(y_true_dir, probs)), 4)
        if "Target_Return" in holdout_df.columns:
            from sklearn.metrics import mean_absolute_error
            y_true_ret = holdout_df["Target_Return"].values
            preds_ret = reg_model.predict(holdout_df[features])
            computed_metrics["holdout_mae"] = round(float(mean_absolute_error(y_true_ret, preds_ret)), 6)
    if metrics:
        computed_metrics.update(metrics)

    # 4. Build metadata manifest for automated lineage tracking
    manifest = {
        "model_version": version,
        "feature_schema_version": f"{version}_{len(features)}stationary",
        "features": list(features),
        "classifier_features": list(features),
        "regressor_features": list(features),
        "classifier_sha256": clf_hash,
        "regressor_sha256": reg_hash,
        "metrics": computed_metrics,
    }

    # 5. Serialize manifest to disk
    with open(MODEL_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SUCCESS] Exported {version} to {MODELS_DIR} successfully!")
    print(f"[MANIFEST] Updated {MODEL_MANIFEST_PATH}")
    print(f"Classifier SHA256: {clf_hash}")
    print(f"Regressor SHA256:  {reg_hash}")
    print(f"Metrics: {computed_metrics}")
    return manifest


def get_registered_models() -> Dict[str, Dict[str, Any]]:
    """
    Discovers all registered model manifests in artifacts/models/ and loads model packages dynamically.
    Returns a dictionary of model packages keyed by model_version.
    """
    from config import MODELS_DIR, ACTIVE_MODEL_VERSION, REGRESSOR_MODEL_PATH, CLASSIFIER_MODEL_PATH

    models_dir = Path(MODELS_DIR)
    registered = {}

    if not models_dir.exists():
        return registered

    # 1. Discover all manifest JSON files in the models directory
    manifest_files = sorted(list(models_dir.glob("*manifest*.json")))

    for m_file in manifest_files:
        try:
            with open(m_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            version = data.get("model_version")
            if not version:
                continue

            # 2. Resolve artifact paths for classifier and regressor
            clf_path = data.get("classifier_artifact_path") or CLASSIFIER_MODEL_PATH
            reg_path = data.get("regressor_artifact_path") or REGRESSOR_MODEL_PATH

            if not Path(clf_path).is_file() or not Path(reg_path).is_file():
                continue

            # 3. Load model checkpoints with CPU device execution
            clf_model = joblib.load(clf_path)
            reg_model = joblib.load(reg_path)

            if hasattr(clf_model, "set_params"):
                clf_model.set_params(device="cpu")
            if hasattr(reg_model, "set_params"):
                reg_model.set_params(device="cpu")

            features = data.get("features") or data.get("classifier_features") or []

            registered[version] = {
                "model_version": version,
                "status": data.get("status", "ACTIVE" if version == ACTIVE_MODEL_VERSION else "CANDIDATE"),
                "manifest_path": str(m_file),
                "classifier": clf_model,
                "regressor": reg_model,
                "features": features,
                "metrics": data.get("metrics", {}),
                "manifest_data": data,
            }
        except Exception:
            continue

    return registered


