"""
Mudra Controlled Retraining Runner Interface.
Defines data loading, terminal target verification, chronological partitioning,
path safety validation, metric calculations, and reproducibility manifest builder.

STRICT SAFETY GOVERNANCE:
Model fitting (.fit / .fit_transform), hyperparameter tuning, and benchmark execution
are explicitly blocked in this module until authorized.
"""
# PRE-LOAD ONNXRUNTIME FIRST to prevent Windows OpenMP (0xC0000005) thread collision
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, precision_score, recall_score

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    BASE_DIR,
    BTCUSDT_1H_CSV,
    CLASSIFIER_FEATURES,
    FEATURED_BTC_DATA_PATH,
    MODELS_DIR,
    REGRESSOR_FEATURES,
)

# Explicit Governance Constants
REUSED_HOLDOUT_WARNING = "REUSED HISTORICAL BENCHMARK — PRIOR MODEL SELECTION EXPOSURE"
DEFAULT_TARGET_HORIZON_HOURS = 1
RANDOM_SEED = 42

__all__ = [
    "REUSED_HOLDOUT_WARNING",
    "DEFAULT_TARGET_HORIZON_HOURS",
    "RANDOM_SEED",
    "validate_output_directory",
    "verify_terminal_row_target",
    "load_and_filter_candidate_dataset",
    "chronological_split_with_purge",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_feature_schema_hash",
    "compute_file_sha256",
    "build_experiment_manifest",
]


# =========================================================================
# 1. Path Safety & Isolation
# =========================================================================

def validate_output_directory(
    output_dir: Union[str, Path],
    allow_non_empty: bool = False,
) -> Path:
    """
    Validates that the target output directory is safe, fail-closed, and strictly
    isolated from protected production models and dataset directories.

    Args:
        output_dir: Destination path for experimental artifacts.
        allow_non_empty: If False, raises FileExistsError if directory already contains files.

    Returns:
        Path: Resolved, verified destination Path.

    Raises:
        ValueError: If output_dir attempts to overwrite or nest inside protected paths,
                    or uses unsafe path traversal.
        FileExistsError: If output_dir already exists with files and allow_non_empty is False.
    """
    if not output_dir or str(output_dir).strip() == "":
        raise ValueError("Output directory cannot be empty.")

    raw_path = Path(output_dir)

    # Check 1: Symlink check on the supplied path and its ancestors before resolution
    if raw_path.is_symlink() or any(p.is_symlink() for p in raw_path.absolute().parents if p.exists()):
        raise ValueError(
            f"Path safety violation: symlinks are not permitted in output directory path '{output_dir}'."
        )

    out_path = raw_path.resolve()

    # Resolve protected paths
    protected_models_dir = Path(MODELS_DIR).resolve()
    protected_data_featured = Path(FEATURED_BTC_DATA_PATH).resolve()
    protected_data_raw = Path(BTCUSDT_1H_CSV).resolve()
    protected_repo_root = Path(BASE_DIR).resolve()
    protected_data_dir = (protected_repo_root / "data").resolve()

    # Check 2: System drive root or repository root / ancestor check
    if out_path.parent == out_path or len(out_path.parts) <= 1:
        raise ValueError(
            f"Path safety violation: output directory '{out_path}' cannot be system root."
        )
    if out_path == protected_repo_root or out_path in protected_repo_root.parents:
        raise ValueError(
            f"Path safety violation: output directory '{out_path}' cannot equal or be an ancestor of repository root."
        )

    # Check 3: Cannot equal or be inside protected models directory
    if out_path == protected_models_dir or protected_models_dir in out_path.parents:
        raise ValueError(
            f"Path safety violation: output directory '{out_path}' attempts to overwrite "
            f"or reside inside protected production models directory '{protected_models_dir}'."
        )

    # Check 4: Cannot collide with dataset files or equal / reside inside protected data directory
    if out_path == protected_data_featured or out_path == protected_data_raw:
        raise ValueError(
            f"Path safety violation: output directory '{out_path}' collides with protected dataset file."
        )
    if out_path == protected_data_dir or protected_data_dir in out_path.parents:
        raise ValueError(
            f"Path safety violation: output directory '{out_path}' cannot equal or reside inside protected data directory '{protected_data_dir}'."
        )

    # Check 5: Check if directory exists and already contains files
    if out_path.exists() and not allow_non_empty:
        contained_files = list(out_path.iterdir())
        if contained_files:
            raise FileExistsError(
                f"Path safety violation: output directory '{out_path}' already contains {len(contained_files)} files. "
                "Refusing to overwrite existing experimental artifacts without explicit authorization."
            )

    return out_path


# =========================================================================
# 2. Dataset Loading & Terminal Target Verification
# =========================================================================

def verify_terminal_row_target(
    cand_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    target_close_col: str = "Target_Close",
    close_col: str = "close",
    horizon_hours: int = 1,
) -> bool:
    """
    Independently verifies the candidate dataset's terminal row target against raw candlestick data.
    Validates both:
    1. The target timestamp equals terminal_timestamp + horizon_hours.
    2. The raw candle's close price at that exact target timestamp matches Target_Close.

    Args:
        cand_df: Candidate DataFrame containing the terminal observation.
        raw_df: Authoritative raw candlestick DataFrame.
        timestamp_col: Name of timestamp column.
        target_close_col: Name of target close price column.
        close_col: Name of raw close price column.
        horizon_hours: Expected prediction horizon in hours.

    Returns:
        bool: True if independently verified.

    Raises:
        ValueError: If verification fails, raw data is missing the target candle,
                    or the target price does not match.
    """
    if cand_df.empty:
        raise ValueError("Cannot verify terminal row on an empty candidate DataFrame.")
    if raw_df.empty:
        raise ValueError("Cannot verify terminal row against an empty raw candlestick DataFrame.")

    last_row = cand_df.iloc[-1]
    terminal_ts = pd.to_datetime(last_row[timestamp_col])
    expected_target_ts = terminal_ts + pd.Timedelta(hours=horizon_hours)
    cand_target_close = float(last_row[target_close_col])

    # Convert raw timestamps to datetime for exact lookup (supports both '%Y:%m:%d %H:%M:%S' and ISO formats)
    try:
        raw_timestamps = pd.to_datetime(raw_df[timestamp_col], format="%Y:%m:%d %H:%M:%S")
    except (ValueError, TypeError):
        raw_timestamps = pd.to_datetime(raw_df[timestamp_col])
    match_mask = raw_timestamps == expected_target_ts
    raw_matches = raw_df[match_mask]

    if raw_matches.empty:
        raise ValueError(
            f"Terminal row target verification failed: raw candlestick observation at expected "
            f"target timestamp '{expected_target_ts}' does not exist in authoritative raw data."
        )

    raw_candle_close = float(raw_matches[close_col].iloc[0])

    if not np.isclose(raw_candle_close, cand_target_close, rtol=1e-5, atol=1e-4):
        raise ValueError(
            f"Terminal row target price mismatch: authoritative raw close ({raw_candle_close}) "
            f"does not match candidate Target_Close ({cand_target_close}) at target timestamp '{expected_target_ts}'."
        )

    return True


def load_and_filter_candidate_dataset(
    candidate_path: Union[str, Path],
    raw_candlestick_path: Optional[Union[str, Path]] = None,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads the candidate dataset read-only, verifies target columns and feature schemas,
    filters irregular-horizon rows (such as the 2-hour exchange halt), and independently
    verifies the terminal row's 1-hour target against raw candle data.

    Args:
        candidate_path: Path to candidate CSV (opened read-only).
        raw_candlestick_path: Path to raw candlestick CSV for terminal target verification.
                              If None, defaults to config.BTCUSDT_1H_CSV if it exists.
        timestamp_col: Name of the timestamp column.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Filtered clean DataFrame and audit summary dict.

    Raises:
        FileNotFoundError: If candidate file does not exist.
        ValueError: If schemas are invalid, timestamps are non-monotonic/duplicated,
                    or terminal target verification fails.
    """
    cand_file = Path(candidate_path)
    if not cand_file.exists():
        raise FileNotFoundError(f"Candidate dataset not found at '{cand_file}'.")

    # Read strictly read-only
    df = pd.read_csv(cand_file)
    raw_count = len(df)

    if raw_count < 10:
        raise ValueError(f"Candidate dataset too small for evaluation: {raw_count} rows.")

    # Validate required columns
    required_targets = ["Target_Close", "Target_Movement", "Target_Return"]
    for col in required_targets:
        if col not in df.columns:
            raise ValueError(f"Candidate dataset missing required target column: '{col}'.")

    if timestamp_col not in df.columns:
        raise ValueError(f"Candidate dataset missing timestamp column: '{timestamp_col}'.")

    # Validate required feature schemas from config
    missing_clf = [f for f in CLASSIFIER_FEATURES if f not in df.columns]
    missing_reg = [f for f in REGRESSOR_FEATURES if f not in df.columns]
    if missing_clf or missing_reg:
        raise ValueError(
            f"Feature schema mismatch. Missing classifier features: {missing_clf}, "
            f"missing regressor features: {missing_reg}."
        )

    # Targets must never appear in input feature schemas
    for col in required_targets:
        if col in CLASSIFIER_FEATURES or col in REGRESSOR_FEATURES:
            raise ValueError(f"Critical target leakage: target column '{col}' is configured as an input feature!")

    # Parse and validate timestamps
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    if df[timestamp_col].duplicated().any():
        dups = df[df[timestamp_col].duplicated()][timestamp_col].tolist()
        raise ValueError(f"Candidate dataset contains duplicate timestamps: {dups[:5]}.")

    if not df[timestamp_col].is_monotonic_increasing:
        raise ValueError("Candidate dataset timestamps are not strictly monotonic increasing.")

    # Compute consecutive row interval
    df["_dt_next"] = df[timestamp_col].shift(-1) - df[timestamp_col]

    # Identify irregular rows (where next row exists and delta != 1 hour)
    irregular_mask = (df["_dt_next"].notna()) & (df["_dt_next"] != pd.Timedelta(hours=1))
    excluded_timestamps = [ts.isoformat() for ts in df.loc[irregular_mask, timestamp_col]]
    excluded_count = int(irregular_mask.sum())

    # Terminal Row Verification
    # The terminal row (index -1) has _dt_next == NaT because shift(-1) reaches the end of the dataframe.
    # It must be independently verified against raw candlestick history.
    raw_path_obj = Path(raw_candlestick_path) if raw_candlestick_path else Path(BTCUSDT_1H_CSV)
    if not raw_path_obj.exists():
        raise ValueError(
            f"Cannot independently verify terminal row target: raw candlestick data not found at '{raw_path_obj}'. "
            "Failing safely per evaluation protocol."
        )

    raw_df = pd.read_csv(raw_path_obj)
    # Perform strict verification
    verify_terminal_row_target(
        cand_df=df,
        raw_df=raw_df,
        timestamp_col=timestamp_col,
        target_close_col="Target_Close",
        close_col="close",
        horizon_hours=1,
    )

    # Verify target alignment on cand_df prior to filtering
    # Every row where _dt_next == 1h must have Target_Close matching subsequent close,
    # and Target_Movement and Target_Return matching exact return formulas
    regular_mask = (df["_dt_next"] == pd.Timedelta(hours=1))
    reg_indices = df[regular_mask].index

    reg_next_close = df.loc[reg_indices + 1, "close"].values
    reg_curr_close = df.loc[reg_indices, "close"].values
    reg_target_close = df.loc[reg_indices, "Target_Close"].values
    reg_target_move = df.loc[reg_indices, "Target_Movement"].values
    reg_target_return = df.loc[reg_indices, "Target_Return"].values

    if not np.isclose(reg_next_close, reg_target_close, rtol=1e-5, atol=1e-4).all():
        raise ValueError("Target alignment check failed: Target_Close does not match subsequent 1h candle Close.")

    expected_move = (reg_next_close > reg_curr_close).astype(int)
    if not (reg_target_move == expected_move).all():
        raise ValueError("Target alignment check failed: Target_Movement does not match directional movement.")

    expected_return = (reg_next_close - reg_curr_close) / reg_curr_close
    if not np.isclose(reg_target_return, expected_return, rtol=1e-5, atol=1e-4).all():
        raise ValueError("Target alignment check failed: Target_Return does not match fractional return formula.")

    # Filter: retain rows where irregular_mask is False (this keeps all verified 1h rows AND the verified terminal row)
    clean_df = df[~irregular_mask].copy().reset_index(drop=True)
    clean_df.drop(columns=["_dt_next"], inplace=True)
    retained_count = len(clean_df)

    # Assert that all retained rows have valid numerical features and targets (zero NaNs or Infs)
    if clean_df.isna().any().any():
        raise ValueError("Data integrity violation: NaNs encountered in retained dataset after filtering.")
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    if np.isinf(clean_df[numeric_cols]).any().any():
        raise ValueError("Data integrity violation: Infinities encountered in retained dataset after filtering.")

    summary = {
        "raw_count": raw_count,
        "retained_count": retained_count,
        "excluded_count": excluded_count,
        "excluded_timestamps": excluded_timestamps,
        "terminal_row_verified": True,
        "terminal_timestamp": clean_df[timestamp_col].iloc[-1].isoformat(),
        "terminal_target_close": float(clean_df["Target_Close"].iloc[-1]),
    }

    return clean_df, summary


# =========================================================================
# 3. Chronological Split & Holdout Governance
# =========================================================================

def chronological_split_with_purge(
    df: pd.DataFrame,
    split_ratio: float = 0.8,
    apply_purge: bool = True,
    min_samples_per_split: int = 50,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame, Dict[str, Any]]:
    """
    Partitions the dataset chronologically into training and holdout benchmark sets,
    applying a transparent 1-row boundary purge to decouple training loss from the benchmark start.

    Args:
        df: Filtered clean DataFrame with verified 1-hour target horizons.
        split_ratio: Fraction of data allocated to training (default: 0.8).
        apply_purge: If True, prunes the final training observation whose target equals the
                     first test row's close price.
        min_samples_per_split: Minimum number of samples required in each partition.
        timestamp_col: Name of the timestamp column.

    Returns:
        Tuple: (train_df, purge_row, test_df, split_summary)
    """
    if not (0.1 < split_ratio < 0.95):
        raise ValueError(f"Invalid split_ratio: {split_ratio}. Must be between 0.1 and 0.95.")

    n_total = len(df)
    if n_total < 2 * min_samples_per_split:
        raise ValueError(
            f"Dataset length {n_total} insufficient for splitting with min_samples_per_split={min_samples_per_split}."
        )

    split_idx = int(n_total * split_ratio)
    if split_idx < min_samples_per_split or (n_total - split_idx) < min_samples_per_split:
        raise ValueError("Computed split index results in partition below min_samples_per_split threshold.")

    if apply_purge:
        # Purge the boundary row at split_idx - 1
        train_df = df.iloc[: split_idx - 1].copy().reset_index(drop=True)
        purge_row = df.iloc[split_idx - 1].copy()
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

        # Document and assert boundary target alignment
        # Purge row target is test_df row 0 close price
        purge_target = float(purge_row["Target_Close"])
        test_first_close = float(test_df["close"].iloc[0])
        boundary_target_matched = bool(np.isclose(purge_target, test_first_close, rtol=1e-5, atol=1e-4))
    else:
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        purge_row = None
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        boundary_target_matched = False

    # Assert strict chronological ordering: train strictly precedes test
    train_max_ts = pd.to_datetime(train_df[timestamp_col].iloc[-1])
    test_min_ts = pd.to_datetime(test_df[timestamp_col].iloc[0])
    if train_max_ts >= test_min_ts:
        raise ValueError(f"Chronological split violation: train_max_ts ({train_max_ts}) >= test_min_ts ({test_min_ts}).")

    train_move_balance = train_df["Target_Movement"].value_counts(normalize=True).to_dict()
    test_move_balance = test_df["Target_Movement"].value_counts(normalize=True).to_dict()

    summary = {
        "split_ratio": split_ratio,
        "apply_purge": apply_purge,
        "boundary_target_matched": boundary_target_matched,
        "holdout_governance_warning": REUSED_HOLDOUT_WARNING,
        "train_count": len(train_df),
        "train_start_ts": train_df[timestamp_col].iloc[0].isoformat(),
        "train_end_ts": train_df[timestamp_col].iloc[-1].isoformat(),
        "train_class_balance": train_move_balance,
        "purge_count": 1 if purge_row is not None else 0,
        "purge_timestamp": purge_row[timestamp_col].isoformat() if purge_row is not None else None,
        "test_count": len(test_df),
        "test_start_ts": test_df[timestamp_col].iloc[0].isoformat(),
        "test_end_ts": test_df[timestamp_col].iloc[-1].isoformat(),
        "test_class_balance": test_move_balance,
    }

    return train_df, purge_row, test_df, summary


# =========================================================================
# 4. Pure Metric Calculation Interfaces
# =========================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes pure classification metrics for directional market regimes.
    Validates input ranges, probability bounds, and handles single-class edge cases.

    Args:
        y_true: True binary direction labels (0 or 1).
        y_pred: Predicted binary direction labels (0 or 1).
        y_proba: Optional predicted probabilities of positive class (shape (N,) or (N, 2)).

    Returns:
        Dict[str, float]: Calculated metrics.

    Raises:
        ValueError: On mismatched lengths, invalid labels, or invalid probabilities.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Array length mismatch: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}.")

    # Validate binary labels
    unique_true = set(np.unique(y_true))
    if not unique_true.issubset({0, 1}):
        raise ValueError(f"y_true must contain binary labels in {{0, 1}}. Found: {unique_true}")
    unique_pred = set(np.unique(y_pred))
    if not unique_pred.issubset({0, 1}):
        raise ValueError(f"y_pred must contain binary labels in {{0, 1}}. Found: {unique_pred}")

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="binary", zero_division=0))

    # Majority class baseline
    class_counts = np.bincount(y_true, minlength=2)
    majority_baseline = float(np.max(class_counts) / len(y_true))

    metrics: Dict[str, float] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "precision": prec,
        "recall": rec,
        "majority_class_baseline": majority_baseline,
        "uniform_random_reference": 0.50,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if len(y_proba) != len(y_true):
            raise ValueError(f"Probability length mismatch: len(y_proba)={len(y_proba)} != len(y_true)={len(y_true)}.")

        # Handle 1D or 2D probability array
        if y_proba.ndim == 2:
            if y_proba.shape[1] != 2:
                raise ValueError(f"y_proba 2D array must have exactly 2 columns. Found shape: {y_proba.shape}")
            probs_pos = y_proba[:, 1]
        elif y_proba.ndim == 1:
            probs_pos = y_proba
        else:
            raise ValueError(f"y_proba must be 1D or 2D. Found dimension: {y_proba.ndim}")

        if (probs_pos < 0.0).any() or (probs_pos > 1.0).any():
            raise ValueError("y_proba contains values outside valid range [0.0, 1.0].")

        # Handle single-class edge case in log_loss
        if len(unique_true) < 2:
            metrics["log_loss"] = float("nan")
        else:
            metrics["log_loss"] = float(log_loss(y_true, probs_pos, labels=[0, 1]))

    return metrics


def compute_regression_metrics(
    y_true_return: np.ndarray,
    y_pred_return: np.ndarray,
    current_close: np.ndarray,
    actual_next_close: np.ndarray,
) -> Dict[str, float]:
    """
    Computes pure continuous return and price-equivalent regression metrics.
    Explicitly defines directional hit rate behavior for flat predictions.

    Directional hit convention:
    Matches when np.sign(y_pred_return) == np.sign(y_true_return).
    A prediction of exactly 0.0 matches a true return of 0.0, but does NOT match
    a non-zero positive or negative movement.

    Args:
        y_true_return: Ground-truth continuous percentage returns.
        y_pred_return: Predicted continuous percentage returns.
        current_close: Base close price at feature time t.
        actual_next_close: Ground-truth close price at target time t+1.

    Returns:
        Dict[str, float]: Calculated metrics.

    Raises:
        ValueError: On mismatched lengths, empty arrays, or non-positive close prices.
    """
    y_true_return = np.asarray(y_true_return, dtype=float)
    y_pred_return = np.asarray(y_pred_return, dtype=float)
    current_close = np.asarray(current_close, dtype=float)
    actual_next_close = np.asarray(actual_next_close, dtype=float)

    n = len(y_true_return)
    if n == 0:
        raise ValueError("Cannot compute regression metrics on empty arrays.")
    if not (n == len(y_pred_return) == len(current_close) == len(actual_next_close)):
        raise ValueError(
            f"Length mismatch: y_true={n}, y_pred={len(y_pred_return)}, "
            f"current_close={len(current_close)}, actual_next_close={len(actual_next_close)}."
        )

    if (current_close <= 0.0).any() or (actual_next_close <= 0.0).any():
        raise ValueError("Close prices must be strictly positive.")

    # Return errors
    return_mae = float(mean_absolute_error(y_true_return, y_pred_return))
    return_rmse = float(np.sqrt(mean_squared_error(y_true_return, y_pred_return)))

    # Price-equivalent forecasts
    predicted_next_close = current_close * (1.0 + y_pred_return)
    price_rmse = float(np.sqrt(mean_squared_error(actual_next_close, predicted_next_close)))

    # Naive persistence baseline: forecast zero return (predicted price = current_close)
    naive_persistence_rmse = float(np.sqrt(mean_squared_error(actual_next_close, current_close)))
    rmse_outperformance = float(naive_persistence_rmse - price_rmse)

    # Directional hit rate
    sign_true = np.sign(y_true_return)
    sign_pred = np.sign(y_pred_return)
    directional_hits = (sign_true == sign_pred).astype(float)
    directional_hit_rate = float(np.mean(directional_hits))

    return {
        "return_mae": return_mae,
        "return_rmse": return_rmse,
        "price_rmse": price_rmse,
        "naive_persistence_rmse": naive_persistence_rmse,
        "rmse_outperformance_usd": rmse_outperformance,
        "directional_hit_rate": directional_hit_rate,
    }


# =========================================================================
# 5. Reproducibility Manifest Builder
# =========================================================================

def compute_feature_schema_hash(features: List[str]) -> str:
    """Computes a deterministic SHA-256 hash for an ordered list of feature names."""
    canonical_repr = json.dumps(list(features), separators=(",", ":"))
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes the SHA-256 hex digest of a file in 64KB chunks."""
    path_obj = Path(filepath)
    hasher = hashlib.sha256()
    with open(path_obj, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_experiment_manifest(
    experiment_id: str,
    candidate_path: Union[str, Path],
    filter_summary: Dict[str, Any],
    split_summary: Dict[str, Any],
    output_dir: Union[str, Path],
    git_commit: Optional[str] = None,
    git_branch: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constructs a comprehensive reproducibility manifest as a pure dictionary.
    Does NOT write to disk.

    Args:
        experiment_id: Unique experiment run identifier.
        candidate_path: Path to candidate dataset.
        filter_summary: Summary dict from load_and_filter_candidate_dataset.
        split_summary: Summary dict from chronological_split_with_purge.
        output_dir: Target output directory path.
        git_commit: Optional Git commit hash.
        git_branch: Optional Git branch name.
        extra_metadata: Optional supplementary execution parameters.

    Returns:
        Dict[str, Any]: Structured manifest dictionary.
    """
    manifest = {
        "manifest_version": "1.0.0",
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit or "uncommitted",
        "git_branch": git_branch or "unknown",
        "dataset": {
            "candidate_path": str(Path(candidate_path).resolve()),
            "raw_count": filter_summary.get("raw_count"),
            "retained_count": filter_summary.get("retained_count"),
            "excluded_count": filter_summary.get("excluded_count"),
            "excluded_timestamps": filter_summary.get("excluded_timestamps"),
            "terminal_row_verified": filter_summary.get("terminal_row_verified"),
            "target_definitions": {
                "Target_Close": "Next confirmed candlestick close price: Close_{t+1}",
                "Target_Movement": "Binary directional indicator: 1 if Close_{t+1} > Close_t else 0",
                "Target_Return": "Continuous fractional return: (Close_{t+1} - Close_t) / Close_t",
                "horizon_hours": DEFAULT_TARGET_HORIZON_HOURS,
            },
        },
        "partitions": {
            "split_ratio": split_summary.get("split_ratio"),
            "apply_purge": split_summary.get("apply_purge"),
            "boundary_target_matched": split_summary.get("boundary_target_matched"),
            "holdout_governance_warning": REUSED_HOLDOUT_WARNING,
            "train": {
                "count": split_summary.get("train_count"),
                "start_timestamp": split_summary.get("train_start_ts"),
                "end_timestamp": split_summary.get("train_end_ts"),
                "class_balance": split_summary.get("train_class_balance"),
            },
            "purge": {
                "count": split_summary.get("purge_count"),
                "timestamp": split_summary.get("purge_timestamp"),
            },
            "benchmark_test": {
                "count": split_summary.get("test_count"),
                "start_timestamp": split_summary.get("test_start_ts"),
                "end_timestamp": split_summary.get("test_end_ts"),
                "class_balance": split_summary.get("test_class_balance"),
            },
        },
        "feature_contracts": {
            "classifier_features_count": len(CLASSIFIER_FEATURES),
            "classifier_features": CLASSIFIER_FEATURES,
            "classifier_schema_hash": compute_feature_schema_hash(CLASSIFIER_FEATURES),
            "regressor_features_count": len(REGRESSOR_FEATURES),
            "regressor_features": REGRESSOR_FEATURES,
            "regressor_schema_hash": compute_feature_schema_hash(REGRESSOR_FEATURES),
        },
        "reproducibility": {
            "random_seed": RANDOM_SEED,
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        },
        "external_data_limitations": {
            "fear_and_greed_latency": (
                "Alternative.me updates once daily between 00:00 and 00:05 UTC. "
                "Intraday predictions between 00:00 and 00:05 UTC may experience up to 5 minutes of upstream publication latency."
            ),
            "onchain_metrics_latency": (
                "Blockchain.com daily chart aggregates have an observed backend publication latency of 1 to 3 hours post-midnight UTC. "
                "Historical joins stamp day D-1 metrics across day D; live inference prior to 03:00 UTC forward-fills the latest confirmed day."
            ),
            "holdout_snooping_warning": REUSED_HOLDOUT_WARNING,
        },
        "output_directory": str(Path(output_dir).resolve()),
    }

    if extra_metadata:
        manifest["extra_metadata"] = extra_metadata

    return manifest


# =========================================================================
# 6. CLI Entrypoint (Validation & Dry-Run Only)
# =========================================================================

def main() -> None:
    """CLI Entrypoint for validation and pre-flight checks."""
    parser = argparse.ArgumentParser(
        description="Mudra Controlled Retraining Runner (Validation Interface)"
    )
    parser.add_argument(
        "--candidate-data",
        type=str,
        default=str(BASE_DIR / "data" / "processed" / "featured_btc_data_candidate.csv"),
        help="Path to candidate dataset CSV.",
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        default=str(BTCUSDT_1H_CSV),
        help="Path to raw candlestick CSV for terminal target verification.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE_DIR / "artifacts" / "experiments" / "p1_controlled_retrain"),
        help="Target output directory.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Chronological train split ratio.",
    )
    parser.add_argument(
        "--dry-run-validation",
        action="store_true",
        help="Perform data loading, filtering, splitting, and path validation without training models.",
    )
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help="Model training execution flag (BLOCKED).",
    )

    args = parser.parse_args()

    if args.execute_training:
        print("[FATAL ERROR] Model training execution is strictly BLOCKED in TASK-P1-09.", file=sys.stderr)
        print("This runner interface is authorized for validation and unit testing only.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Mudra Controlled Retraining Runner — Pre-Flight Validation")
    print(f"{'='*70}")

    validated_out = validate_output_directory(args.output_dir, allow_non_empty=True)
    print(f"[OK] Target output directory validated: {validated_out}")

    clean_df, filter_summary = load_and_filter_candidate_dataset(
        candidate_path=args.candidate_data,
        raw_candlestick_path=args.raw_data,
    )
    print(f"[OK] Candidate dataset loaded: {filter_summary['raw_count']} raw -> {filter_summary['retained_count']} retained.")
    print(f"[OK] Irregular rows excluded: {filter_summary['excluded_count']} ({filter_summary['excluded_timestamps']})")
    print(f"[OK] Terminal row verified against raw data: {filter_summary['terminal_row_verified']}")

    train_df, purge_row, test_df, split_summary = chronological_split_with_purge(
        df=clean_df,
        split_ratio=args.split_ratio,
        apply_purge=True,
    )
    print(f"[OK] Chronological 80/20 split computed with 1-sample purge:")
    print(f"     Train partition: {split_summary['train_count']} rows ({split_summary['train_start_ts']} -> {split_summary['train_end_ts']})")
    print(f"     Purged row:      {split_summary['purge_timestamp']}")
    print(f"     Test benchmark:  {split_summary['test_count']} rows ({split_summary['test_start_ts']} -> {split_summary['test_end_ts']})")
    print(f"     Governance:      {split_summary['holdout_governance_warning']}")
    print("\n[VALIDATION COMPLETE — READY FOR RETRAINING AUTHORIZATION]")


if __name__ == "__main__":
    main()
