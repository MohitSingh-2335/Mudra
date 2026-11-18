import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from scripts.prepare_data import preprocess_and_save_featured_data
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import extract_features_for_inference
from src.model_loader import load_production_models


# Run end-to-end quant pipeline: data prep, training, audit, and evaluation
def run_pipeline(
    fetch_fresh: bool = False, tune: bool = False, n_trials: int = 50,
    train_baseline: bool = False, run_shap: bool = True, run_threshold: bool = True,
    run_backtest: bool = True,
) -> None:
    print(f"Starting Mudra Pipeline (Model: {config.ACTIVE_MODEL_VERSION}, Schema: {config.FEATURE_SCHEMA_VERSION})")

    # 1. Feature Engineering
    df_featured = preprocess_and_save_featured_data(fetch_fresh=fetch_fresh)
    print(f"Dataset prepared: {len(df_featured):,} rows, {df_featured.shape[1]} cols.")

    # 2. Training or Tuning
    if tune:
        from src.hyperparameter_tuning import run_tuning
        run_tuning(n_trials=n_trials)
    elif train_baseline:
        from src.model_training import train_and_save_models
        train_and_save_models()

    # 3. Model Verification
    reg_model, clf_model = load_production_models()

    # 4. TreeSHAP Explainability
    if run_shap:
        try:
            from src.explainability import generate_global_explanations
            generate_global_explanations()
        except Exception as exc:
            print(f"SHAP skipped: {exc}")

    # 5. Threshold Sweeping
    if run_threshold:
        try:
            from scripts.experiments.sweep_threshold import run_threshold_sweep
            run_threshold_sweep()
        except Exception as exc:
            print(f"Threshold sweep skipped: {exc}")

    # 6. Walk-Forward Backtesting
    if run_backtest:
        try:
            from src.backtesting import main as run_backtest_main
            run_backtest_main()
        except Exception as exc:
            print(f"Backtest skipped: {exc}")

    # 7. Live Feature Parity Check
    raw_df = load_and_clean_data(config.BTCUSDT_1H_CSV).tail(100)
    live_features = extract_features_for_inference(raw_df)
    missing = [f for f in config.CLASSIFIER_FEATURES + config.REGRESSOR_FEATURES if f not in live_features]
    if missing:
        raise ValueError(f"Feature parity error: {missing}")

    test_clf_vec = pd.DataFrame([live_features])[config.CLASSIFIER_FEATURES]
    test_reg_vec = pd.DataFrame([live_features])[config.REGRESSOR_FEATURES]
    prob_up = float(clf_model.predict_proba(test_clf_vec)[0, 1])
    pred_return = float(reg_model.predict(test_reg_vec)[0])
    print(f"Pipeline complete. Smoke test: Up={prob_up*100:.1f}%, Return={pred_return*100:+.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mudra pipeline.")
    parser.add_argument("--fetch-fresh", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--train-baseline", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-threshold-sweep", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        fetch_fresh=args.fetch_fresh, tune=args.tune or args.full, n_trials=args.trials,
        train_baseline=args.train_baseline, run_shap=not args.skip_shap,
        run_threshold=not args.skip_threshold_sweep, run_backtest=not args.skip_backtest,
    )


if __name__ == "__main__":
    main()
