# Mudra Project Analysis

## 1. Project Type

Mudra is a Python-based quantitative cryptocurrency trading research and prediction system.

Its current implementation is a research prototype for BTC/USDT on one-hour market data. It combines historical market-data processing, technical and alternative-data feature engineering, supervised machine-learning models, fee-aware backtesting, validation experiments, and a Streamlit prediction dashboard.

It is not currently a fully autonomous trading bot or a production-grade portfolio-management platform. The repository contains design documents for that larger vision, but the implemented code is primarily for model experimentation and prediction.

## 2. Main Objective

The project investigates whether machine-learning models can predict the next hourly BTC/USDT price movement or return well enough to support a trading strategy after transaction costs.

The principal prediction tasks are:

- Regression: estimate the next-hour price or percentage return using XGBoost.
- Classification: predict whether the next-hour movement is upward or downward using an SVC model.
- Strategy evaluation: convert predictions into long/flat trading decisions and compare them with buy-and-hold performance.
- Interactive inspection: display live predictions, historical simulation results, and manual predictions through Streamlit.

The research process also tests different models, decision thresholds, validation windows, and zero-shot time-series foundation models.

## 3. Implemented Architecture

```text
Binance historical/live data
        |
        v
Data preparation and timestamp handling
        |
        v
Technical + market microstructure + external features
        |
        v
Feature dataset and next-hour targets
        |
        +--> XGBoost regression
        |
        +--> SVC classification
        |
        +--> Model comparison and validation experiments
        |
        v
Fee-aware backtesting and result CSV files
        |
        v
Streamlit prediction dashboard
```

### Data ingestion

- `scripts/ingest/fetch_fresh_data.py` downloads BTCUSDT candlestick data from Binance through the Binance Python client.
- The main historical data artifact is `data/raw/BTCUSDT-1H.csv`.
- The live dashboard fetches recent one-hour candles from Binance when live mode is selected.
- The current workflow is primarily hourly and BTC/USDT-focused, even though configuration files list additional assets.

### Data preparation

- `scripts/prepare_data.py` loads historical market data and combines it with engineered and external features.
- `src/data_preprocessing.py` handles basic timestamp parsing and chronological ordering.
- `src/feature_engineering.py` creates the model inputs and target columns.
- The prepared feature dataset is stored in `data/processed/featured_btc_data.csv`.

### Feature engineering

The feature pipeline includes several groups of inputs:

- OHLCV and volume-derived values.
- Price changes, returns, rolling means, rolling standard deviations, highs, and lows.
- RSI, MACD, Bollinger Bands, and exponential moving averages.
- Lagged close and volume values.
- Time features such as hour, day of week, and cyclical sine/cosine encodings.
- Trade-count and taker-buy-ratio features that describe market activity and order flow.
- Fear & Greed index values and rolling summaries.
- On-chain transaction, hash-rate, and miner-revenue changes.
- News sentiment scores and rolling sentiment summaries when sentiment data is available.

The external-data agents are located in `src/agents/`:

- `fear_greed_agent.py` retrieves Alternative.me Fear & Greed data.
- `onchain_agent.py` retrieves selected Blockchain.com metrics.
- `sentiment_agent.py` retrieves GDELT headlines and attempts transformer-based sentiment scoring.

## 4. Machine-Learning Components

### Training

`src/model_training.py` trains the main regression and classification models and saves serialized artifacts under `artifacts/models/`.

The configured primary models are:

- XGBoost for regression.
- SVC for direction classification.
- A scaler used before classification inference.

The primary evaluation approach is a chronological train/test split rather than a random split, which is more appropriate for time-series data.

### Model comparison

`scripts/experiments/find_best_models.py` compares several traditional approaches, including linear regression, random forest, XGBoost, SVR, logistic regression, random-forest classification, XGBoost classification, and SVC.

### Additional experiments

The `experiments/backtesting/` directory contains experiments for:

- Comparing model-driven trading strategies.
- Sweeping classification thresholds.
- Walk-forward validation across multiple chronological windows.
- Statistical significance testing.
- Zero-shot forecasts using Chronos, TimesFM, and Moirai.

These scripts are research and evaluation tools, not separate production services.

## 5. Backtesting and Outputs

`src/backtesting.py` implements a long/flat strategy with transaction-fee modeling, equity tracking, trade logging, drawdown, Sharpe ratio, win rate, and buy-and-hold comparison.

Important generated artifacts include:

- `data/results/backtest_equity_curve.csv`: account value through the backtest.
- `data/results/backtest_trade_log.csv`: individual simulated trades.
- `data/results/model_comparison_backtest.csv`: model strategy comparison.
- `data/results/threshold_sweep.csv`: results for different prediction thresholds.
- `data/results/walk_forward_validation.csv`: chronological validation results.
- `data/results/zero_shot_chronos_results.csv`: Chronos experiment results.
- `data/results/zero_shot_moirai_results.csv`: Moirai experiment results.
- `data/results/zero_shot_timesfm_results.csv`: TimesFM experiment results.

The reports in `docs/reports/` indicate that the tested approaches have not yet demonstrated a consistently repeatable, fee-surviving trading advantage. Some high-threshold results appear profitable, but they rely on very few trades, and walk-forward results generally do not consistently beat buy-and-hold.

## 6. User Interface

`app.py` is a Streamlit application named BTC Predictor Suite. It provides three modes:

1. Live Prediction: fetches recent Binance BTCUSDT candles, calculates features, and displays a price and direction prediction.
2. Simulation from File: steps through the featured CSV and compares previous predictions with later observed candles.
3. Manual Prediction: accepts manually entered market values and produces one-off model predictions.

The interface is for prediction and investigation. It does not place exchange orders, manage a live portfolio, or continuously retrain models.

## 7. Repository Organization

- `app.py`: Streamlit dashboard and prediction workflows.
- `config.py`: model paths, feature lists, asset names, and application settings.
- `scripts/ingest/fetch_fresh_data.py`: market-data retrieval.
- `scripts/prepare_data.py`: feature-data preparation and external-data merging.
- `scripts/experiments/find_best_models.py`: model benchmarking.
- `src/`: reusable preprocessing, feature engineering, model training, backtesting, and data-agent modules.
- `experiments/backtesting/`: validation, threshold, significance, walk-forward, and zero-shot research scripts.
- `data/`: raw inputs, processed features, and generated experiment outputs.
- `artifacts/models/`: serialized model and scaler artifacts when present.
- `scripts/`: ingestion, preparation, and model-comparison utilities.
- `docs/reports/`: phase-specific research findings.
- `docs/reference/`: requirements and design notes.
- `archive/planning/`: merged architectural and personal planning material.
- `requirements.txt`: pinned Python dependencies for the implemented environment.
- `README.md`: original project introduction and intended feature list.

## 8. Current Strengths

- Uses chronological evaluation and walk-forward analysis, which better reflects time-series deployment than random splitting.
- Includes transaction fees in backtesting rather than evaluating predictions in isolation.
- Combines technical, market-activity, sentiment, Fear & Greed, and on-chain features.
- Preserves research outputs as CSV files for later comparison.
- Offers both command-line research scripts and an interactive Streamlit interface.

## 9. Current Limitations and Important Gaps

The repository should be understood as an experimental research system because several capabilities described in planning documents are not implemented or are incomplete:

- No exchange order execution, paper-trading engine, or portfolio-management layer.
- No WebSocket streaming or continuous 24/7 service.
- No automated prediction logging, retraining loop, model registry, or deployment pipeline.
- No implemented Kelly sizing, VaR controls, stop-loss system, trailing take-profit, or circuit breakers.
- No SHAP explanation workflow in the active application.
- No complete multi-asset or multi-timeframe training pipeline.
- The saved-model feature definitions and the shorter feature lists used by `app.py` are not fully aligned; this can cause inference inconsistency or feature-count errors.
- Optional external data can be missing when an API request or sentiment model fails.
- `scripts/prepare_data.py` contains duplicated on-chain merging, which produces duplicated columns in the featured dataset.
- Some research scripts require dependencies that are not listed in `requirements.txt`.
- Automated tests and continuous integration are not present.
- Data validation, gap detection, duplicate cleanup, and broad timestamp-format support are limited.

## 10. How to Run the Main Workflow

Install the listed dependencies:

```bash
pip install -r requirements.txt
```

Typical research steps are:

```bash
python scripts/ingest/fetch_fresh_data.py
python scripts/prepare_data.py
python -m src.model_training
python scripts/experiments/find_best_models.py
streamlit run app.py
```

The exact command order may vary depending on which existing data and model artifacts are being reused. Binance credentials for live dashboard access are expected through Streamlit secrets.

## 11. Final Classification

Mudra is best classified as an experimental machine-learning cryptocurrency trading research platform with an interactive prediction dashboard.

Its implemented center of gravity is historical BTC/USDT analysis, supervised next-hour prediction, and strategy validation. Its surrounding documents describe a larger future autonomous trading architecture, but that production architecture should be treated as planned scope until the missing execution, risk, monitoring, testing, and deployment components are implemented.
