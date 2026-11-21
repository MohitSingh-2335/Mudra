# Model Card: Model 2 (`xgb_v2.0.0`)

**Model Identifier:** `xgb_v2.0.0`  
**Model Family:** Gradient Boosted Decision Trees (XGBoost)  
**Governance Status:** `SHADOW CANDIDATE`  
**Artifact Manifest:** `artifacts/models/xgb_v2_manifest.json`  
**Development Branch:** `aiml-workspace` (`aiml_workspace/model_v2/main.ipynb`)  
**Last Updated:** 2026-11-20  

---

## 1. Hypothesis & Delta from Previous Model

- **Targeted Weakness of Model 1:** Model 1's feature set (`v1_39features`) contained raw dollar price levels (`close_lag_1`, `Rolling_Mean_Close`, `bb_high`, `bb_low`, `ema_10`, `ema_30`). When Bitcoin moved to new price all-time highs in 2024–2026, test-set prices fell completely outside the partition bounds learned by the decision trees during training. Furthermore, Model 1's squared error regression loss overweighted large volatile wick outliers.
- **Core Hypothesis Tested:** 
  1. Replacing raw price levels with normalized percentage distances (`candle_body_pct`, `hl_range_pct`, `sma_10_dist_pct`, `ema_10_dist_pct`, `ema_30_dist_pct`, `bb_width_pct`, `bb_pct_b`) will remove price-level memorization and eliminate covariate shift across multi-year price regimes.
  2. Switching the regressor loss from `reg:squarederror` to Pseudo-Huber loss (`reg:pseudohubererror`) will make return predictions robust to extreme tail spikes.
- **Key Variables Changed:**
  - Removed 13 price-level and macro features: `close_lag_1`, `Rolling_Mean_Close`, `Rolling_Std_Close`, `bb_high`, `bb_low`, `ema_10`, `ema_30`, `close_open_diff`, `fng_value`, `fng_mean_3d`, `onchain_num_tx_change`, `onchain_hash_rate_change`, `onchain_miners_revenue_change`.
  - Added 8 normalized percentage features: `candle_body_pct`, `hl_range_pct`, `sma_10_dist_pct`, `ema_10_dist_pct`, `ema_30_dist_pct`, `bb_width_pct`, `bb_pct_b`, `return_1h_lag`.
  - Added raw microstructure trade variables: `taker_buy_base_volume`, `number_of_trades`.
  - Unified regressor and classifier feature schema to identical 36 features.
  - Regressor loss changed to `reg:pseudohubererror`.

---

## 2. Data & Feature Schema

- **Dataset Source:** Binance Spot `BTCUSDT` 1-hour candles.
- **Dataset Range:** July 8, 2022 13:00 UTC to July 6, 2026 08:00 UTC (34,990 clean 1-hour candles after dropping the exchange-halt row).
- **Chronological Split Boundary:**
  - **Train Set (80% Chronological):** 27,991 hours (2022-07-08 13:00 UTC to 2025-09-17 17:00 UTC).
  - **Purge Buffer:** 1 hour (2025-09-17 18:00 UTC) excluded to prevent look-ahead bias.
  - **Out-of-Sample Holdout Test Set (20%):** 6,998 hours (2025-09-17 19:00 UTC to 2026-07-06 08:00 UTC).
- **Feature Schema Version:** `v2_36stationary` (36 features across both regressor and classifier).
- **Complete Feature List (36 Features):**
  - *Normalized Percentage Features (Stationary):* `candle_body_pct`, `hl_range_pct`, `sma_10_dist_pct`, `ema_10_dist_pct`, `ema_30_dist_pct`, `bb_width_pct`, `bb_pct_b`, `return_1h_lag`, `return_mean_6h`, `return_std_6h`.
  - *Cyclical Time Encodings (Stationary):* `hour`, `dayofweek`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`.
  - *Technical Momentum (Bounded):* `rsi`, `high_low_ratio`, `taker_buy_ratio`, `taker_buy_ratio_mean_6h`.
  - *Residual Non-Stationary Dollar Features (Audit Finding):* `Price Change`, `Volatility`, `vol_1h`, `vol_mean_6h`, `vol_std_6h`, `vol_max_6h`, `vol_min_6h`, `macd`, `price_volatility_interaction`.
  - *Residual Drifting Volumes and Counts (Audit Finding):* `volume`, `volume_lag_1`, `number_of_trades`, `taker_buy_base_volume`, `trades_mean_6h`, `onchain_num_tx`, `onchain_hash_rate`.

---

## 3. Architecture & Artifact Governance

- **Algorithms:** Dual Gradient Boosted Decision Tree ensembles via `xgboost`.
- **Model Checkpoint Artifacts & Digests:**
  - Classifier Model: `artifacts/models/xgb_v2_classifier.pkl`  
    `SHA-256: ac1311b668f49b29ef5e5631d1741e533556c9596658faf5bffdda7ffd8fff49`
  - Regressor Model: `artifacts/models/xgb_v2_regressor.pkl`  
    `SHA-256: 902d033d37ff49b577e7fe7d6f64a8b3b5daffc5e54777578845771a8a424296`
- **Trained Hyperparameters:**
  - *Classifier:* `objective='binary:logistic'`, `eval_metric='logloss'`, `max_depth=4`, `learning_rate=0.015`, `min_child_weight=20`, `reg_alpha=2.0`, `reg_lambda=5.0`, `n_estimators=100`, `random_state=42`.
  - *Regressor:* `objective='reg:pseudohubererror'`, `max_depth=3`, `learning_rate=0.015`, `min_child_weight=25`, `reg_lambda=5.0`, `n_estimators=100`, `random_state=42`.
- **Inference Runtime:** Integrated into `src/engine.py` for continuous live shadow execution during hourly quant cycles. Deserialized via `joblib` on CPU.

---

## 4. Standardized Evaluation Scorecard

All metrics evaluated against the locked 6,998-hour out-of-sample holdout test partition:

| Metric | Model 2 Value | Model 1 Baseline | Naive Baseline | Status / Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Directional Accuracy** | **52.07%** | **53.26%** | 50.39% | Lower than Model 1 by 1.19 percentage points. Verified against manifest `0.5207`. |
| **Log-Loss** | **0.6926** | **0.6914** | 0.6931 | Marginally better than random (0.6931), but worse than Model 1. Matches manifest `0.6926`. |
| **Brier Score** | **0.2497** | **0.2491** | 0.2500 | Very close to uniform uncertainty (0.2500). |
| **ROC-AUC** | **0.5342** | **0.5394** | 0.5000 | Modest ranking ability, lower than Model 1. |
| **Spearman Rank IC** | **+0.0135** | **+0.0149** | 0.0000 | Positive correlation between predicted and actual returns, but weaker than Model 1. |
| **Return MAE** | **0.003222** | **0.003221** | **0.003198** | Similar return error to Model 1. Matches manifest `0.003222`. |
| **USD Price RMSE** | **$401.12** | **$399.47** | **$398.77** | Higher USD error than Model 1 due to conservative Pseudo-Huber shrinkage. |

---

## 5. Strengths & What Worked

1. **Successful Removal of Raw Price Levels:** Eliminated raw Bitcoin dollar levels (`close_lag_1`, `Rolling_Mean_Close`, `bb_high`, `bb_low`, `ema_10`, `ema_30`), ensuring that the model does not memorize historical absolute price numbers.
2. **Pseudo-Huber Loss Robustness:** Replaced standard squared error loss with `reg:pseudohubererror`, providing linear penalty scaling on extreme returns and reducing regressor sensitivity to violent exchange flash wicks.
3. **Operational Shadow Deployment:** Successfully integrated into `src/engine.py` and `app.py`, executing live hourly predictions in shadow consensus mode alongside Model 1 without disrupting live paper trading.

---

## 6. Weaknesses & Honest Post-Mortem Findings

1. **Stationarity Incompleteness (15 Leaking Features):** A rigorous codebase audit revealed that while explicit price levels were removed, **15 of Model 2's 36 features remained non-stationary**:
   - *8 Raw USD Differences/Spreads:* `Price Change` ($close - open$), `vol_1h` ($high - low$), `vol_mean_6h`, `vol_std_6h`, `vol_max_6h`, `vol_min_6h`, `macd` (raw USD EMA difference), and `price_volatility_interaction` ($close \times Volatility$). When Bitcoin quadrupled from $25,000 to $95,000+, these dollar spreads expanded 4-fold, causing severe distribution shift in the test set.
   - *7 Drifting Volumes & Network Counts:* `volume`, `volume_lag_1`, `number_of_trades`, `taker_buy_base_volume`, `trades_mean_6h`, `onchain_num_tx`, `onchain_hash_rate`. Raw Bitcoin volume and transaction counts fluctuate with market cycles and network adoption over multi-year periods.
2. **Feature Implementation Bug:** In `src/feature_engineering.py` line 70, `sma_10_dist_pct` is computed as `(close - Rolling_Mean_Close) / Rolling_Mean_Close`. However, line 25 defines `Rolling_Mean_Close = df['close'].rolling(5).mean()`. The feature is mathematically a 5-hour SMA distance despite its name.
3. **Performance Degradation Below Promotion Threshold:** Directional accuracy dropped from 53.26% to 52.07%, failing Promotion Gate 1.

---

## 7. Deployment Decision & Handoff to Next Model

- **Deployment Status:** `SHADOW CANDIDATE`  
  Maintained in shadow execution mode to log live out-of-sample consensus alongside Model 1. Rejected for active promotion because it did not outperform Model 1.
- **Handoff Hypothesis Motivating Model 3 (`lgbm_v3.0.0`):**
  1. Complete the stationarity transformation by converting the remaining 15 raw dollar and volume features into strictly normalized ratio features (e.g. Volume divided by 24-hour rolling volume, and dollar differences expressed as fractional returns).
  2. Transition from XGBoost to LightGBM to improve training speed and leaf-wise split granularity.
  3. Implement post-hoc probability calibration (Platt Sigmoid Calibration) on an isolated calibration set to transform tree probability outputs into well-calibrated trading probabilities.
