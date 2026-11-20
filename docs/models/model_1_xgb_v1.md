# Model Card: Model 1 (`xgb_onnx_v1.0.0`)

**Model Identifier:** `xgb_onnx_v1.0.0`  
**Model Family:** Gradient Boosted Decision Trees (XGBoost)  
**Governance Status:** `ACTIVE PRODUCTION`  
**Artifact Manifest:** `artifacts/models/model_manifest.json`  
**Last Updated:** 2026-11-20  

---

## 1. Hypothesis & Delta from Previous Model

- **Targeted Weakness of Baseline:** The naive persistence baseline ($\hat{P}_{t+1} = P_t$) assumes a pure random walk with no predictable momentum. Attempting to predict absolute future price levels using linear regression, SVR, or default tree ensembles proved completely ineffective because raw prices are non-stationary and cause models to simply memorize historical price levels.
- **Core Hypothesis Tested:** Transforming the learning objective into fractional percentage returns ($R_{t+1}$) for magnitude estimation and binary direction ($Y_{t+1} = \mathbb{I}[R_{t+1} > 0]$) for probability estimation will enable XGBoost to learn weak, non-random directional signals from short-term momentum (RSI), cyclical calendar features, and order-flow microstructure (`taker_buy_ratio`), while respecting temporal causality.
- **Key Variables Changed:**
  - Separated regression from classification: one dedicated regressor for expected return magnitude, one dedicated classifier for direction probability.
  - Implemented 10-fold chronological walk-forward cross-validation (50 Optuna trials) to prevent overfitting.
  - Introduced microstructural order flow (`taker_buy_ratio`, `trades_mean_6h`) and lagged macro on-chain indicators.

---

## 2. Data & Feature Schema

- **Dataset Source:** Binance Spot `BTCUSDT` 1-hour confirmed closed candles, merged with Alternative.me Fear & Greed sentiment and Blockchain.com on-chain metrics.
- **Dataset Range:** July 8, 2022 13:00 UTC to July 6, 2026 08:00 UTC (34,990 clean 1-hour candles after dropping 1 exchange-halt candle on 2023-03-24 12:00 UTC with zero volume).
- **Chronological Split Boundary:**
  - **Train Set (80% Chronological):** 27,991 hours (2022-07-08 13:00 UTC to 2025-09-17 17:00 UTC).
  - **Purge Buffer:** 1 hour (2025-09-17 18:00 UTC) strictly excluded to prevent $T+1\text{h}$ target leakage across the partition boundary.
  - **Out-of-Sample Holdout Test Set (20%):** 6,998 hours (2025-09-17 19:00 UTC to 2026-07-06 08:00 UTC).
- **Feature Schema Version:** `v1_39features`
  - **Classifier Feature Count:** 39 features.
  - **Regressor Feature Count:** 26 features.
- **Complete Feature Set:**
  - *Order Flow & Microstructure:* `taker_buy_ratio`, `taker_buy_ratio_mean_6h`, `trades_mean_6h`, `volume`, `volume_lag_1`.
  - *Price Momentum & Volatility:* `Price Change`, `Volatility`, `high_low_ratio`, `close_open_diff`, `vol_1h`, `vol_mean_6h`, `vol_std_6h`, `vol_max_6h`, `vol_min_6h`, `return_mean_6h`, `return_std_6h`, `rsi`, `macd`, `bb_high`, `bb_low`, `ema_10`, `ema_30`, `price_volatility_interaction`.
  - *Raw Price Levels (Classifier only):* `Rolling_Mean_Close`, `Rolling_Std_Close`, `rolling_max_6h`, `rolling_min_6h`, `close_lag_1`.
  - *Cyclical Time Encodings:* `hour`, `dayofweek`, `day`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`.
  - *Sentiment & On-Chain (Lagged D-1):* `fng_value`, `fng_mean_3d`, `onchain_num_tx_change`, `onchain_hash_rate_change`, `onchain_miners_revenue_change`.

---

## 3. Architecture & Artifact Governance

- **Algorithms:** Dual Gradient Boosted Decision Tree ensembles via `xgboost`.
- **Model Checkpoint Artifacts & Digests:**
  - Classifier Model: `artifacts/models/best_classifier_model.pkl`  
    `SHA-256: df139c1018ed5025b1444d6597c3d98b4d7c377a898487268d18e58d253ae684`
  - Regressor Model: `artifacts/models/best_regressor_model.pkl`  
    `SHA-256: b44fba18cf2c7b51e3297e83a946d869392803a267e951a5e85ea052c63eb854`
- **Trained Hyperparameters:**
  - *Classifier:* `objective='binary:logistic'`, `max_depth=5`, `learning_rate=0.0141`, `n_estimators=200`, `subsample=0.9207`, `colsample_bytree=0.7590`, `min_child_weight=8`, `reg_alpha=7.5706`, `reg_lambda=7.335e-06`, `tree_method='hist'`.
  - *Regressor:* `objective='reg:squarederror'`, `max_depth=3`, `learning_rate=0.1963`, `n_estimators=400`, `subsample=0.6452`, `colsample_bytree=0.8049`, `min_child_weight=6`, `reg_alpha=1.1010`, `reg_lambda=1.1514`, `tree_method='hist'`.
- **Inference Runtime:** CPU-pinned (`device='cpu'`) via `joblib` deserialization with cryptographic hash verification in `src/model_loader.py`. Measured runtime latency is under 1.5 milliseconds per row.

---

## 4. Standardized Evaluation Scorecard

All metrics evaluated against the locked 6,998-hour out-of-sample holdout test partition:

| Metric | Model 1 Value | Benchmark / Baseline | Status / Interpretation |
| :--- | :---: | :---: | :--- |
| **Directional Accuracy** | **53.26%** | 50.39% (Naive) / 50.0% (Coin Flip) | Statistically significant edge ($p < 0.01$). Matches manifest `0.5326`. |
| **Log-Loss** | **0.6914** | 0.6931 (Uninformative Baseline) | Meaningful reduction in cross-entropy below random probability. |
| **Brier Score** | **0.2491** | 0.2500 (Uniform Coin Flip) | Lower than 0.25, indicating positive probability calibration. |
| **ROC-AUC** | **0.5394** | 0.5000 (Random) | Positive discriminative capacity between positive and negative moves. |
| **Spearman Rank IC** | **+0.0149** | 0.0000 | Weak positive rank correlation between predicted return and realized return. |
| **Return MAE** | **0.003221** | 0.003198 (Zero Return Baseline) | Slight underperformance against zero-return baseline due to regression variance. |
| **USD Price RMSE (Holdout)** | **$399.47** | **$398.77** (Naive Persistence) | Did not beat persistence in USD RMSE; decision trees cannot extrapolate new price scales. |
| **USD Price RMSE (Cross-Val)**| **$396.22** | $398.63 (Historical CV Baseline) | Average walk-forward RMSE across Optuna 50-trial cross-validation splits. |
| **Directional Accuracy (Cross-Val)**| **53.17%** | 49.58% | Walk-forward cross-validation average across training windows. |

---

## 5. Strengths & What Worked

1. **Robust Directional Edge:** Sustained 53.26% accuracy across 6,998 out-of-sample test hours, demonstrating that short-term Bitcoin directional bias is not purely random.
2. **Order-Flow Dominance:** TreeSHAP interpretability revealed `taker_buy_ratio` as the single most important predictive feature, confirming that aggressive taker orders crossing the bid-ask spread drive immediate price discovery.
3. **Low Latency & High Reliability:** Sub-2ms execution latency on CPU allows seamless integration into live Streamlit dashboards and the 4-agent LangGraph council.

---

## 6. Weaknesses & Honest Post-Mortem Findings

1. **Non-Stationary Feature Leakage:** The classifier schema includes raw price levels (`close_lag_1`, `Rolling_Mean_Close`, `bb_high`, `bb_low`, `ema_10`, `ema_30`). When Bitcoin's price expanded from $25,000 to $95,000+, these features suffered severe covariate shift, forcing tree leaves to split on historical price values that never occurred during training.
2. **Regression Extrapolation Failure:** The regressor failed to beat naive persistence in USD RMSE ($399.47 vs. $398.77) because decision trees predict constants on bounded partitions and cannot extrapolate unbounded macro trends.
3. **Fee Drag Sensitivity:** While 53.26% directional accuracy is statistically real, backtesting confirmed that trading every single hour results in severe capital loss (-92.47%) due to 0.14% round-trip exchange fees.

---

## 7. Deployment Decision & Handoff to Next Model

- **Deployment Status:** `ACTIVE PRODUCTION`  
  Retained as the primary operational signal generator because its 53.26% directional accuracy is the highest verified holdout hit rate in the repository.
- **Handoff Hypothesis Motivating Model 2:**
  Eliminate all raw dollar price levels (`close_lag_1`, `ema_10`, `ema_30`, `bb_high`, `bb_low`) and replace them with stationary percentage distance metrics, while switching the regressor loss function from squared error to Pseudo-Huber loss to reduce sensitivity to volatility outliers.
