"""
Mudra Cryptographic Model Loader & Integrity Verification Service.
Enforces fail-closed SHA-256 checksum verification before deserializing model artifacts.
"""
import io
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Tuple, Union
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
