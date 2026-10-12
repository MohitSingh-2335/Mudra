"""
Unit tests for Mudra Controlled Retraining Runner Interface.
Verifies data loading, terminal target verification, chronological splitting,
path safety, metric calculations, and manifest building without fitting any models.
"""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from config import CLASSIFIER_FEATURES, REGRESSOR_FEATURES
from scripts.experiments.train_controlled_candidate import (
    REUSED_HOLDOUT_WARNING,
    DEFAULT_TARGET_HORIZON_HOURS,
    RANDOM_SEED,
    validate_output_directory,
    verify_terminal_row_target,
    load_and_filter_candidate_dataset,
    chronological_split_with_purge,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_feature_schema_hash,
    compute_file_sha256,
    build_experiment_manifest,
)


# =========================================================================
# Fixtures
# =========================================================================

def _generate_synthetic_df(
    n_rows: int = 100,
    start_ts: str = "2023-01-01 00:00:00",
    has_gap: bool = False,
    gap_idx: int = 20,
) -> pd.DataFrame:
    """Helper to generate deterministic synthetic candlestick DataFrame with valid schema."""
    timestamps = []
    curr = pd.to_datetime(start_ts)
    for i in range(n_rows):
        if has_gap and i == gap_idx:
            curr += pd.Timedelta(hours=2) # 2-hour jump
        elif i > 0:
            curr += pd.Timedelta(hours=1)
        timestamps.append(curr)

    np.random.seed(42)
    closes = 20000.0 + np.cumsum(np.random.randn(n_rows) * 50.0)
    opens = closes - np.random.randn(n_rows) * 10.0
    highs = np.maximum(opens, closes) + 15.0
    lows = np.minimum(opens, closes) - 15.0
    volumes = 1000.0 + np.abs(np.random.randn(n_rows) * 100.0)

    data = {
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }

    # Add targets
    # Shift(-1) for internal rows
    target_closes = np.roll(closes, -1)
    target_closes[-1] = closes[-1] + 25.0 # Terminal row target
    data["Target_Close"] = target_closes
    data["Target_Movement"] = (target_closes > closes).astype(int)
    data["Target_Return"] = (target_closes - closes) / closes

    # Add required features
    all_features = sorted(list(set(CLASSIFIER_FEATURES + REGRESSOR_FEATURES)))
    for f in all_features:
        if f not in data:
            data[f] = 0.5

    return pd.DataFrame(data)


# =========================================================================
# 1. Path Safety Tests
# =========================================================================

def test_validate_output_directory_rejects_empty():
    with pytest.raises(ValueError, match="Output directory cannot be empty"):
        validate_output_directory("")


def test_validate_output_directory_rejects_models_dir(tmp_path):
    from config import MODELS_DIR
    with pytest.raises(ValueError, match="protected production models directory"):
        validate_output_directory(MODELS_DIR)

    # Subdirectory of models_dir
    nested = Path(MODELS_DIR) / "sub_experiment"
    with pytest.raises(ValueError, match="protected production models directory"):
        validate_output_directory(nested)


def test_validate_output_directory_rejects_dataset_paths():
    from config import FEATURED_BTC_DATA_PATH, BTCUSDT_1H_CSV
    with pytest.raises(ValueError, match="collides with protected dataset file"):
        validate_output_directory(FEATURED_BTC_DATA_PATH)
    with pytest.raises(ValueError, match="collides with protected dataset file"):
        validate_output_directory(BTCUSDT_1H_CSV)
    with pytest.raises(ValueError, match="cannot equal or reside inside protected data directory"):
        validate_output_directory(Path(FEATURED_BTC_DATA_PATH).parent)
    with pytest.raises(ValueError, match="cannot equal or reside inside protected data directory"):
        validate_output_directory(Path(BTCUSDT_1H_CSV).parent)


def test_validate_output_directory_rejects_system_and_repo_root():
    from config import BASE_DIR
    import platform
    # System root
    sys_root = "C:\\" if platform.system() == "Windows" else "/"
    with pytest.raises(ValueError, match="cannot be system root"):
        validate_output_directory(sys_root)

    # Repository root
    with pytest.raises(ValueError, match="cannot equal or be an ancestor of repository root"):
        validate_output_directory(BASE_DIR)

    # Repository parent (ancestor)
    with pytest.raises(ValueError, match="cannot equal or be an ancestor of repository root"):
        validate_output_directory(Path(BASE_DIR).parent)


def test_validate_output_directory_rejects_symlinks(tmp_path):
    target_dir = tmp_path / "symlink_target"
    target_dir.mkdir()
    link_dir = tmp_path / "symlink_test"
    try:
        link_dir.symlink_to(target_dir, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation not supported in current environment/permissions.")

    with pytest.raises(ValueError, match="symlinks are not permitted"):
        validate_output_directory(link_dir)


def test_validate_output_directory_rejects_existing_files(tmp_path):
    existing_dir = tmp_path / "exp_existing"
    existing_dir.mkdir()
    (existing_dir / "old_model.pkl").write_text("dummy")

    with pytest.raises(FileExistsError, match="already contains 1 files"):
        validate_output_directory(existing_dir, allow_non_empty=False)

    # When allow_non_empty is True, succeeds
    res = validate_output_directory(existing_dir, allow_non_empty=True)
    assert res == existing_dir.resolve()


def test_validate_output_directory_accepts_clean_path(tmp_path):
    clean_dir = tmp_path / "clean_experiment_dir"
    res = validate_output_directory(clean_dir)
    assert res == clean_dir.resolve()


# =========================================================================
# 2. Terminal Target Verification Tests
# =========================================================================

def test_verify_terminal_row_target_success():
    cand_df = pd.DataFrame({
        "timestamp": ["2026-07-06 08:00:00"],
        "close": [62904.01],
        "Target_Close": [62898.00],
    })
    raw_df = pd.DataFrame({
        "timestamp": ["2026-07-06 08:00:00", "2026-07-06 09:00:00"],
        "close": [62904.01, 62898.00],
    })
    assert verify_terminal_row_target(cand_df, raw_df) is True


def test_verify_terminal_row_target_missing_raw_candle():
    cand_df = pd.DataFrame({
        "timestamp": ["2026-07-06 08:00:00"],
        "close": [62904.01],
        "Target_Close": [62898.00],
    })
    # Raw data does not have 09:00:00 candle
    raw_df = pd.DataFrame({
        "timestamp": ["2026-07-06 08:00:00"],
        "close": [62904.01],
    })
    with pytest.raises(ValueError, match="does not exist in authoritative raw data"):
        verify_terminal_row_target(cand_df, raw_df)


def test_verify_terminal_row_target_price_mismatch():
    cand_df = pd.DataFrame({
        "timestamp": ["2026-07-06 08:00:00"],
        "close": [62904.01],
        "Target_Close": [62898.00],
    })
    # Raw candle at 09:00:00 exists but close price is different
    raw_df = pd.DataFrame({
        "timestamp": ["2026-07-06 08:00:00", "2026-07-06 09:00:00"],
        "close": [62904.01, 62500.00],
    })
    with pytest.raises(ValueError, match="Terminal row target price mismatch"):
        verify_terminal_row_target(cand_df, raw_df)


# =========================================================================
# 3. Dataset Loading & Filtering Tests
# =========================================================================

def test_load_and_filter_excludes_gap_and_retains_verified_terminal(tmp_path):
    cand_df = _generate_synthetic_df(n_rows=50, has_gap=True, gap_idx=15)
    cand_csv = tmp_path / "cand.csv"
    cand_df.to_csv(cand_csv, index=False)

    # Build matching raw_df with candle at terminal + 1h
    terminal_ts = cand_df["timestamp"].iloc[-1]
    raw_terminal_ts = terminal_ts + pd.Timedelta(hours=1)
    raw_df = cand_df[["timestamp", "close"]].copy()
    raw_df = pd.concat([
        raw_df,
        pd.DataFrame([{"timestamp": raw_terminal_ts, "close": cand_df["Target_Close"].iloc[-1]}])
    ], ignore_index=True)
    raw_csv = tmp_path / "raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    clean_df, summary = load_and_filter_candidate_dataset(
        candidate_path=cand_csv,
        raw_candlestick_path=raw_csv,
    )

    # Row 15 was the 2-hour gap, should be excluded
    assert summary["raw_count"] == 50
    assert summary["excluded_count"] == 1
    assert summary["retained_count"] == 49
    assert summary["terminal_row_verified"] is True
    assert len(clean_df) == 49

    # Verify the excluded timestamp is absent from clean_df
    excluded_ts_str = summary["excluded_timestamps"][0]
    assert pd.to_datetime(excluded_ts_str) not in clean_df["timestamp"].values
    assert clean_df["timestamp"].iloc[-1] == cand_df["timestamp"].iloc[-1]
    assert np.isclose(clean_df["Target_Close"].iloc[-1], cand_df["Target_Close"].iloc[-1])
    assert not clean_df.isna().any().any()


def test_load_and_filter_fails_when_terminal_unverified(tmp_path):
    cand_df = _generate_synthetic_df(n_rows=30)
    cand_csv = tmp_path / "cand.csv"
    cand_df.to_csv(cand_csv, index=False)

    # Empty raw file
    raw_csv = tmp_path / "empty_raw.csv"
    pd.DataFrame({"timestamp": [], "close": []}).to_csv(raw_csv, index=False)

    with pytest.raises(ValueError, match="Cannot verify terminal row against an empty raw candlestick DataFrame"):
        load_and_filter_candidate_dataset(candidate_path=cand_csv, raw_candlestick_path=raw_csv)


def test_load_and_filter_rejects_duplicate_timestamps(tmp_path):
    cand_df = _generate_synthetic_df(n_rows=25)
    # Duplicate row 5
    cand_df.loc[6, "timestamp"] = cand_df.loc[5, "timestamp"]
    cand_csv = tmp_path / "cand_dup.csv"
    cand_df.to_csv(cand_csv, index=False)

    with pytest.raises(ValueError, match="contains duplicate timestamps"):
        load_and_filter_candidate_dataset(cand_csv, raw_candlestick_path=tmp_path / "none.csv")


def test_load_and_filter_rejects_non_monotonic_timestamps(tmp_path):
    cand_df = _generate_synthetic_df(n_rows=25)
    # Swap timestamps
    ts5 = cand_df.loc[5, "timestamp"]
    cand_df.loc[5, "timestamp"] = cand_df.loc[6, "timestamp"]
    cand_df.loc[6, "timestamp"] = ts5
    cand_csv = tmp_path / "cand_non_mono.csv"
    cand_df.to_csv(cand_csv, index=False)

    with pytest.raises(ValueError, match="timestamps are not strictly monotonic"):
        load_and_filter_candidate_dataset(cand_csv, raw_candlestick_path=tmp_path / "none.csv")


def test_load_and_filter_rejects_missing_targets(tmp_path):
    cand_df = _generate_synthetic_df(n_rows=25)
    cand_df.drop(columns=["Target_Return"], inplace=True)
    cand_csv = tmp_path / "cand_missing_col.csv"
    cand_df.to_csv(cand_csv, index=False)

    with pytest.raises(ValueError, match="missing required target column: 'Target_Return'"):
        load_and_filter_candidate_dataset(cand_csv, raw_candlestick_path=tmp_path / "none.csv")


# =========================================================================
# 4. Chronological Splitting Tests
# =========================================================================

def test_chronological_split_with_purge_geometry():
    df = _generate_synthetic_df(n_rows=100)
    train_df, purge_row, test_df, summary = chronological_split_with_purge(
        df=df,
        split_ratio=0.8,
        apply_purge=True,
        min_samples_per_split=10,
    )

    # 100 rows, split_idx = 80
    # train has 0..78 (79 rows), purge is row 79, test has 80..99 (20 rows)
    assert len(train_df) == 79
    assert purge_row is not None
    assert len(test_df) == 20
    assert summary["train_count"] == 79
    assert summary["purge_count"] == 1
    assert summary["test_count"] == 20
    assert summary["holdout_governance_warning"] == REUSED_HOLDOUT_WARNING

    # Assert boundary target alignment: purge_row Target_Close == test_df Close row 0
    assert np.isclose(purge_row["Target_Close"], test_df["close"].iloc[0])
    assert summary["boundary_target_matched"] is True

    # Assert temporal ordering
    assert train_df["timestamp"].iloc[-1] < purge_row["timestamp"]
    assert purge_row["timestamp"] < test_df["timestamp"].iloc[0]


def test_chronological_split_without_purge():
    df = _generate_synthetic_df(n_rows=100)
    train_df, purge_row, test_df, summary = chronological_split_with_purge(
        df=df,
        split_ratio=0.8,
        apply_purge=False,
        min_samples_per_split=10,
    )
    assert len(train_df) == 80
    assert purge_row is None
    assert len(test_df) == 20
    assert summary["purge_count"] == 0


def test_chronological_split_rejects_invalid_split_ratio():
    df = _generate_synthetic_df(n_rows=100)
    with pytest.raises(ValueError, match="Invalid split_ratio"):
        chronological_split_with_purge(df, split_ratio=0.05)
    with pytest.raises(ValueError, match="Invalid split_ratio"):
        chronological_split_with_purge(df, split_ratio=0.98)


def test_chronological_split_rejects_insufficient_samples():
    df = _generate_synthetic_df(n_rows=40)
    with pytest.raises(ValueError, match="insufficient for splitting"):
        chronological_split_with_purge(df, min_samples_per_split=25)


# =========================================================================
# 5. Pure Metrics Tests
# =========================================================================

def test_compute_classification_metrics_hand_calculated():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    # TP=1, FP=1, TN=1, FN=1
    # Acc = 2/4 = 0.50
    # Prec = 1/2 = 0.50, Rec = 1/2 = 0.50, F1 = 0.50
    # Balanced Acc = 0.50 * (1/2 + 1/2) = 0.50
    m = compute_classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.50
    assert m["balanced_accuracy"] == 0.50
    assert m["macro_f1"] == 0.50
    assert m["precision"] == 0.50
    assert m["recall"] == 0.50
    assert m["majority_class_baseline"] == 0.50
    assert m["uniform_random_reference"] == 0.50


def test_compute_classification_metrics_log_loss():
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0])
    y_proba = np.array([0.90, 0.10])
    m = compute_classification_metrics(y_true, y_pred, y_proba)
    # log_loss = -0.5 * (ln(0.9) + ln(0.9)) = -ln(0.9) approx 0.10536
    expected_ll = -np.log(0.90)
    assert np.isclose(m["log_loss"], expected_ll, atol=1e-4)


def test_compute_classification_metrics_invalid_inputs():
    with pytest.raises(ValueError, match="Cannot compute metrics on empty arrays"):
        compute_classification_metrics(np.array([]), np.array([]))

    with pytest.raises(ValueError, match="Array length mismatch"):
        compute_classification_metrics(np.array([1, 0]), np.array([1]))

    with pytest.raises(ValueError, match="binary labels"):
        compute_classification_metrics(np.array([1, 2]), np.array([1, 0]))

    with pytest.raises(ValueError, match="outside valid range"):
        compute_classification_metrics(np.array([1, 0]), np.array([1, 0]), np.array([1.5, 0.2]))


def test_compute_classification_metrics_single_class():
    # When y_true contains only a single class (e.g. all 1s), log_loss should be NaN rather than crashing
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.4])

    m = compute_classification_metrics(y_true, y_pred, y_proba)
    assert np.isnan(m["log_loss"])
    assert np.isclose(m["accuracy"], 2.0 / 3.0)
    assert m["majority_class_baseline"] == 1.0


def test_compute_regression_metrics_hand_calculated():
    y_true_ret = np.array([0.01, -0.02])
    y_pred_ret = np.array([0.02, -0.01])
    curr_close = np.array([100.0, 100.0])
    next_close = np.array([101.0, 98.0])

    # Error 1: 0.02 - 0.01 = 0.01. Error 2: -0.01 - (-0.02) = 0.01.
    # MAE = 0.01. RMSE = 0.01.
    # Predicted price 1: 100 * 1.02 = 102. Actual: 101. Diff = 1.0.
    # Predicted price 2: 100 * 0.99 = 99. Actual: 98. Diff = 1.0.
    # Price RMSE = 1.0 USD.
    # Naive pred price 1: 100. Actual: 101. Diff = -1.0.
    # Naive pred price 2: 100. Actual: 98. Diff = 2.0.
    # Naive MSE = (1 + 4)/2 = 2.5. Naive RMSE = sqrt(2.5) approx 1.5811 USD.
    # Outperformance = 1.5811 - 1.0 = 0.5811 USD.
    # Directional hits: sign([0.01, -0.02]) == [1, -1]. sign([0.02, -0.01]) == [1, -1]. 2/2 = 1.0.

    m = compute_regression_metrics(y_true_ret, y_pred_ret, curr_close, next_close)
    assert np.isclose(m["return_mae"], 0.01)
    assert np.isclose(m["return_rmse"], 0.01)
    assert np.isclose(m["price_rmse"], 1.0)
    assert np.isclose(m["naive_persistence_rmse"], np.sqrt(2.5))
    assert np.isclose(m["rmse_outperformance_usd"], np.sqrt(2.5) - 1.0)
    assert m["directional_hit_rate"] == 1.0


def test_compute_regression_metrics_zero_return_convention():
    # If prediction is 0.0 (flat), does it match non-zero movement?
    y_true_ret = np.array([0.02, -0.02, 0.0])
    y_pred_ret = np.array([0.00, -0.02, 0.0])
    curr = np.array([100.0, 100.0, 100.0])
    next_c = np.array([102.0, 98.0, 100.0])

    m = compute_regression_metrics(y_true_ret, y_pred_ret, curr, next_c)
    # Row 0: sign(0.02)=1 vs sign(0.0)=0 -> mismatch
    # Row 1: sign(-0.02)=-1 vs sign(-0.02)=-1 -> hit
    # Row 2: sign(0.0)=0 vs sign(0.0)=0 -> hit
    # Hit rate = 2/3 approx 0.6667
    assert np.isclose(m["directional_hit_rate"], 2.0 / 3.0)


def test_compute_regression_metrics_invalid_inputs():
    with pytest.raises(ValueError, match="Cannot compute regression metrics on empty arrays"):
        compute_regression_metrics(np.array([]), np.array([]), np.array([]), np.array([]))

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_regression_metrics(np.array([0.01]), np.array([0.01, 0.02]), np.array([100.0]), np.array([101.0]))

    with pytest.raises(ValueError, match="Close prices must be strictly positive"):
        compute_regression_metrics(np.array([0.01]), np.array([0.01]), np.array([-100.0]), np.array([101.0]))

    with pytest.raises(ValueError, match="Close prices must be strictly positive"):
        compute_regression_metrics(np.array([0.01]), np.array([0.01]), np.array([100.0]), np.array([0.0]))


# =========================================================================
# 6. Manifest & Reproducibility Tests
# =========================================================================

def test_compute_feature_schema_hash_deterministic():
    feats1 = ["vol_mean_6h", "rsi", "macd"]
    feats2 = ["vol_mean_6h", "rsi", "macd"]
    feats3 = ["rsi", "vol_mean_6h", "macd"] # Different order

    h1 = compute_feature_schema_hash(feats1)
    h2 = compute_feature_schema_hash(feats2)
    h3 = compute_feature_schema_hash(feats3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64


def test_build_experiment_manifest_pure_dict(tmp_path):
    filter_sum = {
        "raw_count": 34991,
        "retained_count": 34990,
        "excluded_count": 1,
        "excluded_timestamps": ["2023-03-24T12:00:00"],
        "terminal_row_verified": True,
    }
    split_sum = {
        "split_ratio": 0.8,
        "apply_purge": True,
        "boundary_target_matched": True,
        "train_count": 27991,
        "train_start_ts": "2022-07-09T09:00:00",
        "train_end_ts": "2025-09-17T17:00:00",
        "train_class_balance": {1: 0.5086, 0: 0.4914},
        "purge_count": 1,
        "purge_timestamp": "2025-09-17T18:00:00",
        "test_count": 6998,
        "test_start_ts": "2025-09-17T19:00:00",
        "test_end_ts": "2026-07-06T08:00:00",
        "test_class_balance": {0: 0.5039, 1: 0.4961},
    }

    manifest = build_experiment_manifest(
        experiment_id="p1_controlled_retrain_v1",
        candidate_path=tmp_path / "cand.csv",
        filter_summary=filter_sum,
        split_summary=split_sum,
        output_dir=tmp_path / "exp_out",
        git_commit="941b35d6",
        git_branch="code-cleanup",
    )

    assert isinstance(manifest, dict)
    assert manifest["experiment_id"] == "p1_controlled_retrain_v1"
    assert manifest["partitions"]["holdout_governance_warning"] == REUSED_HOLDOUT_WARNING
    assert manifest["reproducibility"]["random_seed"] == RANDOM_SEED
    assert manifest["feature_contracts"]["classifier_features_count"] == len(CLASSIFIER_FEATURES)
    assert manifest["feature_contracts"]["regressor_features_count"] == len(REGRESSOR_FEATURES)
    assert "fear_and_greed_latency" in manifest["external_data_limitations"]
    assert "onchain_metrics_latency" in manifest["external_data_limitations"]
