# Mudra Model Evolution & Architecture Ledger

**Document Role:** Master Quantitative Model Evolution Tracking Ledger  
**Scope:** Canonical comparison of all historical, candidate, and roadmap forecasting models in Mudra  
**Last Updated:** 2026-11-20  

---

## 1. Master Model Comparison Ledger

The following table summarizes every model evaluated across identical chronological test splits (6,998 holdout hours on 1-hour Binance `BTCUSDT` candles, 2025-09-17 19:00 UTC to 2026-07-06 08:00 UTC):

| Model Version ID | Model Family | Role / Governance | Feature Schema | Direction Accuracy | Log-Loss | Brier Score | ROC-AUC | Spearman IC | Return MAE | USD Price RMSE | One-Line Engineering Verdict |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Persistence** | Rule Baseline | Benchmark ($\hat{P}_{t+1}=P_t$) | Raw Price ($P_t$) | 50.39% | 0.6931 | 0.2500 | 0.5000 | 0.0000 | 0.003198 | **$398.77** | Default benchmark; difficult for decision trees to beat in USD RMSE due to extrapolation limits. |
| **[Model 1 (`xgb_onnx_v1.0.0`)](../models/model_1_xgb_v1.md)** | XGBoost GBDT | **ACTIVE PRODUCTION** | `v1_39features` (39 Clf / 26 Reg) | **53.26%** | **0.6914** | **0.2491** | **0.5394** | **+0.0149** | 0.003221 | $399.47 | Primary active signal generator; statistically significant directional edge but relies on raw price levels. |
| **[Model 2 (`xgb_v2.0.0`)](../models/model_2_xgb_v2.md)** | XGBoost GBDT | **SHADOW CANDIDATE** | `v2_36stationary` (36 features) | 52.07% | 0.6926 | 0.2497 | 0.5342 | +0.0135 | 0.003222 | $401.12 | Removed raw prices, but 15 unnormalized volume/dollar features remained; failed promotion gate. |
| **[Model 3 (`lgbm_v3.0.0`)](../models/MODEL_CARD_TEMPLATE.md)** | LightGBM GBDT | **ROADMAP / PLANNED** | `v3_stationary` (Strict Stationarity) | *Planned* | *Planned* | *Planned* | *Planned* | *Planned* | *Planned* | *Planned* | Blueprint completed; targets 100% volume/price stationarity, LightGBM speed, and Isotonic calibration. |

---

## 2. Chronological Architecture Evolution

### Generation 0: Naive Persistence Benchmark
- **Hypothesis:** Financial market asset prices closely follow a martingale / random walk with drift over short horizons (1 hour). The best point estimate of next hour's close is simply the current close: $\hat{P}_{t+1} = P_t$.
- **Finding:** Achieved an out-of-sample RMSE of $398.77. In regression framing, predicting raw price levels caused classical regressors (Linear Regression, SVR, Random Forest, XGBoost) to memorize past price regimes and underperform this naive baseline.
- **Resolution:** Abandoned absolute price prediction as an ML target. Switched target framing to fractional return ($R_{t+1} = (P_{t+1} - P_t) / P_t$) and binary direction ($Y_{t+1} = \mathbb{I}[R_{t+1} > 0]$).

### Generation 1: Model 1 (`xgb_onnx_v1.0.0`)
- **Hypothesis:** While regression cannot easily beat persistence, a dual-model setup (XGBoost Regressor for expected magnitude + XGBoost Classifier for directional probability) can extract a weak directional edge from technical momentum, cyclical calendar encodings, and order-flow microstructure (`taker_buy_ratio`).
- **Outcome:** Achieved 53.26% holdout directional accuracy (53.17% in 10-fold chronological walk-forward cross-validation) and 0.6914 log-loss. Order-flow microstructure proved to be the single highest-attribution feature via TreeSHAP.
- **Architectural Bottleneck:** The feature schema (`v1_39features`) retained raw price levels (`close_lag_1`, `Rolling_Mean_Close`, `bb_high`, `bb_low`, `ema_10`, `ema_30`). When Bitcoin climbed from $25,000 to $95,000+, decision tree split thresholds learned on past price levels suffered from distribution shift.

### Generation 2: Model 2 (`xgb_v2.0.0`)
- **Hypothesis:** Replacing raw price levels with normalized distance metrics (`candle_body_pct`, `hl_range_pct`, `sma_10_dist_pct`, `ema_10_dist_pct`, `ema_30_dist_pct`, `bb_width_pct`, `bb_pct_b`) and switching the regressor loss from squared error to Pseudo-Huber loss (`reg:pseudohubererror`) will eliminate covariate shift and reduce sensitivity to crypto volatility outliers.
- **Outcome:** The model successfully executed without price levels and stabilized regressor variance. However, holdout directional accuracy dropped to 52.07% (log-loss 0.6926).
- **Post-Mortem Discovery:** Code audit revealed that 15 of Model 2's 36 features remained non-stationary (8 raw dollar differences such as `Price Change`, `vol_1h`, and `macd`; plus 7 raw drifting counts/volumes such as `volume`, `number_of_trades`, and `onchain_hash_rate`). Furthermore, `sma_10_dist_pct` was computed against a 5-hour rolling mean (`rolling(5)`). As a result, Model 2 was designated as a shadow model and did not replace Model 1.

### Generation 3: Model 3 (`lgbm_v3.0.0`) — Planned Specification
- **Hypothesis:** 
  1. Enforcing 100% strict mathematical stationarity across all features (normalizing volume by 24h rolling volume, converting on-chain metrics to percentage growth rates, and converting MACD and price changes to percentage returns) will eliminate residual covariate shift.
  2. Switching from XGBoost to LightGBM will enable faster leaf-wise tree growth, better categorical binning, and lower training latency.
  3. Adding post-hoc probability calibration (Isotonic Regression or Platt scaling) on a dedicated calibration fold will transform uncalibrated GBDT tree outputs into true posterior probabilities, preventing overconfident mispredictions during low-conviction market regimes.
- **Status:** Architectural blueprint documented in `docs/models/MODEL_CARD_TEMPLATE.md`; implementation pending experimental validation in notebook workspace.

---

## 3. Standardized Governance & Promotion Protocol

Before any candidate model can be promoted from `SHADOW CANDIDATE` to `ACTIVE PRODUCTION`, it must satisfy the following 5 objective promotion gates:

```text
PROMOTION GATE 1 (Statistical Superiority):
├── Directional Accuracy > 53.26% on the locked 6,998-hour holdout test set.
└── Binomial test p-value < 0.05 versus the 50.0% random walk benchmark.

PROMOTION GATE 2 (Probability Calibration):
├── Log-Loss < 0.6914 on the locked holdout test set.
└── Brier Score < 0.2491.

PROMOTION GATE 3 (Economic Viability & Fee Hurdle):
├── Return MAE must demonstrate edge over naive baseline (< 0.003198).
└── Simulated trading under canonical 0.14% round-trip friction must not incur catastrophic fee churn.

PROMOTION GATE 4 (Inference Latency & Security):
├── Single-row inference latency < 50 milliseconds on CPU.
└── Verified cryptographic SHA-256 digest committed in model manifest.

PROMOTION GATE 5 (Operational Stability):
└── Minimum 168 continuous hours (1 week) of error-free live shadow execution in the production Neon PostgreSQL pipeline.
```
