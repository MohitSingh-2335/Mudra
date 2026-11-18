import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
from typing import Optional, Union
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.data_preprocessing import (
    load_and_clean_data,
    merge_fear_greed,
    merge_onchain,
)
from src.feature_engineering import create_features, extract_features_for_inference
from src.model_loader import load_production_models


# loads optional binance credentials from env or streamlit secrets
def load_binance_credentials():
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if api_key and api_secret:
        return api_key, api_secret
    secrets_path = os.path.join(ROOT_DIR, ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            import tomllib
            with open(secrets_path, "rb") as f:
                sec = tomllib.load(f)
            b_sec = sec.get("binance", {})
            return (b_sec.get("api_key") or sec.get("api_key")), (b_sec.get("api_secret") or sec.get("api_secret"))
        except Exception:
            pass
    return None, None


# fetches raw hourly candlesticks from binance public or authenticated api
def fetch_fresh_binance_data(
    symbol: str = "BTCUSDT",
    years_of_history: int = 4,
    output_path: Union[str, Path] = config.BTCUSDT_1H_CSV,
) -> pd.DataFrame:
    from binance.client import Client
    api_key, api_secret = load_binance_credentials()
    client = Client(api_key, api_secret) if api_key and api_secret else Client()
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, f"{years_of_history} years ago UTC")
    if not klines:
        raise RuntimeError("No data returned from Binance.")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms").dt.strftime("%Y:%m:%d %H:%M:%S")
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, pd.NA)
    df = df.rename(columns={
        "quote_asset_volume": "quote_volume",
        "taker_buy_base": "taker_buy_base_volume",
        "taker_buy_quote": "taker_buy_quote_volume",
    })
    cleaned = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume", "taker_buy_ratio"]]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(out_file, out_file.with_name(f"{out_file.stem}_backup_{stamp}.csv"))
    cleaned.to_csv(out_file, index=False)
    print(f"Saved {len(cleaned)} candles to {out_file}")
    return cleaned


# cleans raw data, merges fear greed & onchain sentiment, and computes features
def preprocess_and_save_featured_data(
    input_path: Union[str, Path] = config.BTCUSDT_1H_CSV,
    output_path: Union[str, Path] = config.FEATURED_BTC_DATA_PATH,
    fetch_fresh: bool = False,
) -> pd.DataFrame:
    input_file, output_file = Path(input_path), Path(output_path)
    if fetch_fresh or not input_file.exists():
        fetch_fresh_binance_data(output_path=input_file)

    df = load_and_clean_data(input_file)
    df = merge_fear_greed(df, timestamp_col="timestamp")
    df = merge_onchain(df, timestamp_col="timestamp")
    featured_df = create_features(df, drop_na=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(output_file, index=False)
    print(f"Generated {len(featured_df)} rows x {featured_df.shape[1]} features -> {output_file}")
    return featured_df


# executes end-to-end quant pipeline: data prep, training, audit, and evaluation
def run_pipeline(
    fetch_fresh: bool = False, tune: bool = False, n_trials: int = 50,
    train_baseline: bool = False, run_shap: bool = True, run_threshold: bool = True,
    run_backtest: bool = True, prepare_only: bool = False,
) -> None:
    print(f"Starting Mudra Pipeline (Model: {config.ACTIVE_MODEL_VERSION}, Schema: {config.FEATURE_SCHEMA_VERSION})")

    df_featured = preprocess_and_save_featured_data(fetch_fresh=fetch_fresh)
    print(f"Dataset prepared: {len(df_featured):,} rows, {df_featured.shape[1]} cols.")
    if prepare_only:
        return

    if tune:
        from src.model_training import run_tuning
        run_tuning(n_trials=n_trials)
    elif train_baseline:
        from src.model_training import train_and_save_models
        train_and_save_models()

    reg_model, clf_model = load_production_models()

    if run_shap:
        try:
            from src.explainability import generate_global_explanations
            generate_global_explanations()
        except Exception as exc:
            print(f"SHAP skipped: {exc}")

    if run_threshold:
        try:
            from scripts.experiments.sweep_threshold import run_threshold_sweep
            run_threshold_sweep()
        except Exception as exc:
            print(f"Threshold sweep skipped: {exc}")

    if run_backtest:
        try:
            from src.backtesting import main as run_backtest_main
            run_backtest_main()
        except Exception as exc:
            print(f"Backtest skipped: {exc}")

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
    parser = argparse.ArgumentParser(description="Run Mudra quant pipeline.")
    parser.add_argument("--fetch-fresh", action="store_true", help="Fetch fresh klines from Binance")
    parser.add_argument("--prepare-only", action="store_true", help="Only run feature dataset preparation")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--train-baseline", action="store_true", help="Train baseline models without tuning")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP summary generation")
    parser.add_argument("--skip-threshold-sweep", action="store_true", help="Skip threshold sweep")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtesting simulation")
    parser.add_argument("--full", action="store_true", help="Run full pipeline including tuning")
    args = parser.parse_args()

    run_pipeline(
        fetch_fresh=args.fetch_fresh, tune=args.tune or args.full, n_trials=args.trials,
        train_baseline=args.train_baseline, run_shap=not args.skip_shap,
        run_threshold=not args.skip_threshold_sweep, run_backtest=not args.skip_backtest,
        prepare_only=args.prepare_only,
    )


if __name__ == "__main__":
    main()
