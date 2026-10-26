# Mudra — BTC/USDT 1-Hour ML Forecasting & Paper-Trading MLOps Pipeline
### Empirical Research Finding: Negative Result — No Durable Post-Fee Alpha on 1-Hour OHLCV + Daily Sentiment/On-Chain Features

[![CI/CD Pipeline](https://github.com/MohitSingh-2335/Mudra/actions/workflows/ci.yml/badge.svg)](https://github.com/MohitSingh-2335/Mudra/actions/workflows/ci.yml)
[![Hourly Worker](https://github.com/MohitSingh-2335/Mudra/actions/workflows/hourly_worker.yml/badge.svg)](https://github.com/MohitSingh-2335/Mudra/actions/workflows/hourly_worker.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-EB5E28)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)

**Live Streamlit Monitoring Dashboard:** [mudra-quant-terminal.streamlit.app](https://mudra-quant-terminal.streamlit.app/)

**Mudra** is an end-to-end machine learning engineering and paper-trading MLOps project built to answer a concrete quantitative question: *Can classical gradient-boosted trees or zero-shot time-series foundation models predict next-hour Bitcoin (`BTC/USDT`) returns accurately enough to survive spot/futures transaction costs on a strictly leak-free chronological split?*

**Short answer: No.** Across 34,990 hours of cleaned Binance data (`2022-07-08` to `2026-07-06`), every tested regression model failed to beat a naive persistence baseline ($\hat{P}_{t+1} = P_t$, RMSE **$398.77** vs. active XGBoost RMSE **$399.47**). The best directional classifiers achieved **52.07%–53.26%** out-of-sample accuracy with a log-loss of **0.6914–0.6926** (barely below a 50/50 coin flip at **0.6931**). Apparent backtest gains at high probability thresholds did not survive **Bonferroni multiple-testing correction** ($p = 0.0195 > 0.0019$ across 27 swept configurations) and failed **5-window walk-forward replication** (beating Buy & Hold in only **1 of 5 windows**, with a mean return of **-0.52%**). On the 6,998-hour bear-market holdout where Bitcoin fell **-45.41%**, a return-filtered paper-trading strategy lost **-17.66%** solely because it sat in cash **96.43% of the time** (**cash preservation / low market exposure, not positive trading alpha**).

Rather than hiding this negative finding or inflating metrics with lookahead leakage, this repository preserves the complete experimental audit trail alongside the production-style MLOps pipeline built to test it: cryptographic model manifest governance (`SHA-256`), dual-model shadow evaluation (`Model 1` vs. `Model 2`), deterministic risk-gate orchestration (`LangGraph`), a `FastAPI` REST backend, a `PostgreSQL` 9-table audit ledger, and a `T+1h` delayed reconciliation worker.

---

## 1. Problem Statement & Research Scope

Retail ML crypto projects routinely report 60%+ directional accuracy or high Sharpe ratios due to three methodological bugs:
1. **Random `train_test_split(random_state=42)`** on autocorrelated time-series data, leaking future volatility regimes into training.
2. **Predicting raw price levels ($P_{t+1}$)** and reporting high $R^2$ (~0.99) when the model is merely lagging $P_t$.
3. **Zero-fee backtesting or post-hoc threshold cherry-picking** on a single test window without multiple-hypothesis correction.

Mudra was designed to eliminate all three bugs and evaluate next-hour `BTC/USDT` forecasting under realistic constraints:
- **Regression Task:** Predict 1-hour forward percentage return:
  $$\text{Target\_Return}_t = \frac{\text{Close}_{t+1} - \text{Close}_t}{\text{Close}_t}$$
- **Classification Task:** Predict binary next-hour price direction:
  $$\text{Target\_Movement}_t = \mathbb{I}(\text{Close}_{t+1} > \text{Close}_t) \in \{0, 1\}$$
- **Economic Viability Criterion:** A long/flat or long/short signal is only actionable if its expected 1-hour return exceeds round-trip exchange friction and replicates across independent walk-forward windows.

---

## 2. Dataset & Leak-Free Pipeline Design

### 2.1 Data Sources & Granularity
| Source | Resolution | Features Contributed | Failure Handling |
| :--- | :--- | :--- | :--- |
| **Binance (`BTCUSDT-1H.csv` & REST API)** | 1-Hour (`1h`) | OHLCV, quote asset volume, trade count, taker buy base/quote volume | Automatic failover across `api.binance.com` -> `api.binance.us` -> `CryptoCompare` -> local CSV |
| **Alternative.me Fear & Greed API** | Daily (`1d`, broadcast to `1h`) | `fng_value`, `fng_mean_3d` (72-hour rolling mean) | Fails soft to neutral (`50.0`) on timeout |
| **Blockchain.info Charts API** | Daily (`1d`, broadcast to `1h`) | Day-over-day % changes: `onchain_num_tx_change`, `onchain_hash_rate_change`, `onchain_miners_revenue_change` | Fails soft (`0.0` change) if chart endpoints are unreachable |

### 2.2 Anomaly Removal & Chronological Split Contract
From **35,064** raw 1-hour candles (`2022-07-08 13:00 UTC` to `2026-07-06 08:00 UTC`), the pipeline removes initial rolling-window warmup rows (`LOOKBACK_WINDOW = 48` to `72` hours) and explicitly filters out **1 corrupted exchange-halt candle** (`2023-03-24 12:00:00 UTC`, where Binance halted spot trading and recorded `volume = 0` with flat OHLC), yielding **34,990 causally verified 1-hour rows**:

```text
CLEAN DATASET (34,990 Total 1-Hour Candles)
├── Range: 2022-07-08 13:00:00 UTC to 2026-07-06 08:00:00 UTC
└── Exchange-Halt Anomaly Removed: 2023-03-24 12:00:00 UTC (1 row)

1. TRAIN PERIOD (80% Chronological — 27,991 Hours)
   ├── Start: 2022-07-08 13:00:00 UTC
   └── End:   2025-09-17 17:00:00 UTC

2. PURGE BUFFER (1-Hour Causality Guard — 1 Hour)
   └── Excluded: 2025-09-17 18:00:00 UTC
       (Prevents T+1h forward target at 18:00 UTC from crossing into the 19:00 UTC test boundary)

3. OUT-OF-SAMPLE TEST PERIOD (20% Holdout — 6,998 Hours)
   ├── Start: 2025-09-17 19:00:00 UTC
   └── End:   2026-07-06 08:00:00 UTC
```

### 2.3 Feature Schema Evolution: `v1_39features` (Model 1) vs. `v2_36stationary` (Model 2)
A key technical lesson uncovered during our audit was that **Model 1 (`xgb_onnx_v1.0.0`, active production)** and **Model 2 (`xgb_v2.0.0`, shadow candidate)** use different feature schemas:

- **Model 1 (`v1_39features` — Active Production in `artifacts/models/model_manifest.json`):**
  - Uses **26 features** for `XGBRegressor` and **39 features** for `XGBClassifier`.
  - **Known flaw identified in audit:** `v1_39features` still contains raw non-stationary dollar price levels (`close_lag_1`, `Rolling_Mean_Close`, `bb_high`, `bb_low`, `ema_10`, `ema_30`, `rolling_max_6h`, `rolling_min_6h`) and redundant dollar-difference columns (`Price Change` and `close_open_diff` both equal `close - open`). When BTC trades at $115,000 in the test set after training mostly between $16,000 and $90,000, tree splits on raw dollar levels saturate at the training ceiling.
  - **Note on Model Version Naming:** The identifier `xgb_onnx_v1.0.0` is a **legacy naming artifact** retained in `config.py` and `model_manifest.json` for backward compatibility with database foreign keys and governance tests. The models are serialized and served as native `scikit-learn`/`XGBoost` `joblib` pickles (`best_regressor_model.pkl`, `best_classifier_model.pkl`), **not** ONNX Runtime graphs (`onnxruntime` is only pre-imported on Windows to prevent OpenMP thread deadlocks).

- **Model 2 (`v2_36stationary` — Shadow Candidate in `artifacts/models/xgb_v2_manifest.json` & `aiml_workspace/model_v2/main.ipynb`):**
  - Built in `aiml_workspace/model_v2/main.ipynb` to fix Model 1's non-stationarity by stripping all raw price levels (`open`, `high`, `low`, `close`, `close_lag_1`, `Rolling_Mean_Close`, `ema_10`, `ema_30`, `bb_high`, `bb_low`, `rolling_max_6h`, `rolling_min_6h`) and retaining **36 strictly scale-invariant ratios, returns, cyclical time encodings, order-flow ratios, and daily sentiment/on-chain changes** across both regressor and classifier.

---

## 3. Empirical Results (Including Negative Results)

### 3.1 Out-of-Sample Model-Level Benchmark (6,998-Hour Holdout)
All metrics below are verified directly from `artifacts/models/model_manifest.json`, `artifacts/models/xgb_v2_manifest.json`, `scripts/migrate_db.py`, and `aiml_workspace/model_v2/main.ipynb`:

| Model ID & Schema | Role / Status | Regressor Price RMSE (USD) | Regressor Return MAE | Regressor Direction Hit Rate | Classifier Accuracy | Classifier Log-Loss (Coin Flip = 0.6931) | Classifier ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Persistence ($\hat{P}_{t+1} = P_t$)** | Reference Baseline | **$398.77** | `0.003198` | *Baseline* | `50.42%` (Always-Down) | `0.6931` | `0.5000` |
| **`xgb_onnx_v1.0.0` (`v1_39features`)** | **Active Production** | `$399.47` | `0.003210` | `50.50%` | **53.26%** (F1: `0.5297`) | **0.6914** | `~0.531` |
| **`p1_controlled_candidate_v1` (`v1_39features`)** | Rejected Candidate | `$399.37` | `0.003208` | `50.36%` | `53.03%` (F1: `0.5280`) | `0.6918` | `~0.529` |
| **`xgb_v2.0.0` (`v2_36stationary`, Uncalibrated)** | Notebook Baseline (`main.ipynb`) | *Lose to Naive* (`R² = -0.0046`) | **0.003200** | `51.63%` | `52.49%` | `0.6919` | **0.5252** |
| **`xgb_v2.0.0` (`v2_36stationary`, Cost-Aware + Platt)** | **Shadow Candidate (`xgb_v2_manifest.json`)** | *Lose to Naive* (`R² = -0.0086`) | `0.003222` | `51.20%` | `52.07%` | `0.6926` | `0.5236` |

#### Key ML Findings from `aiml_workspace/model_v2/main.ipynb`:
1. **Why Regressors Lose to Naive Persistence:** Hourly BTC returns have a mean near `0.0000` and a standard deviation of `~0.0048` (`0.48%`). Because the signal-to-noise ratio is extremely low, any non-zero return prediction adds variance that slightly exceeds the zero-prediction error of $\hat{P}_{t+1} = P_t$ (yielding out-of-sample return $R^2$ of `-0.0046` for normal XGBoost and `-0.0086` for cost-aware XGBoost).
2. **Why Normal Uncalibrated XGBoost Beat Cost-Aware Calibrated XGBoost in Model 2:** In `main.ipynb`, weighting training samples by `1 + |Target_Return| / 0.0014` (`cost_weights`) over-emphasized extreme historical volatility spikes from 2022–2024 that did not repeat in the same way during 2025–2026. Similarly, fitting Platt scaling (`CalibratedClassifierCV(method='sigmoid')`) on the chronological validation slice introduced calibration drift because the validation slice had a higher bullish base rate than the bear-market holdout (`49.58%` UP).
3. **Why Removing Non-Stationary Features Lowered Static Holdout Accuracy (`53.26%` -> `52.49%`):** Model 1 (`v1_39features`) inadvertently benefited from regime-level price splits during the late-2025/2026 drawdown. Removing raw dollar levels in `v2_36stationary` made the model mathematically stationary and leak-free across price regimes, exposing the true (~52.5%) directional ceiling of 1-hour technical indicators.

---

### 3.2 Research Phase Reports Summary (`docs/reports/`)
Before building the full MLOps stack, four controlled empirical studies were executed and documented in `docs/reports/`:

1. **[Classical ML Backtesting (`docs/reports/classical_model_backtesting.md`)](docs/reports/classical_model_backtesting.md):**
   - Compared `LinearRegression`, `RandomForest`, `SVR`, and `XGBoost` for regression (all lost to naive persistence: RMSE `399.39`–`409.03` vs. `398.63`) and `LogisticRegression` (`52.49%`), `SVC` (`52.26%`), `XGBoost` (`52.15%`), and `RandomForest` (`51.36%`) for classification.
   - Swept 27 configurations (3 classifiers $\times$ 9 probability thresholds `0.50`–`0.70`) at `0.10%` per-side fee. Low thresholds (`0.50`–`0.60`) lost **-14% to -94%** due to fee churn. The best single result (`LogisticRegression @ 0.68`, **+6.18%** return, `n = 9` trades, raw binomial $p = 0.0195$) **failed Bonferroni correction** ($\alpha = 0.05 / 27 = 0.0019$).
2. **[Zero-Shot Foundation Models (`docs/reports/foundation_model_zero_shot_evaluation.md`)](docs/reports/foundation_model_zero_shot_evaluation.md):**
   - Evaluated **Amazon Chronos-2**, **Google TimesFM 2.5 (200M)**, and **Salesforce Moirai 2.0 (R-small)** zero-shot on 500 held-out windows (`CONTEXT_LEN = 512`).
   - All three lost to naive persistence (`310.06` RMSE) and clustered around a coin flip on direction:
     - `Chronos-2`: RMSE `322.65`, Directional Accuracy `49.60%`
     - `TimesFM 2.5`: RMSE `314.65`, Directional Accuracy `48.60%`
     - `Moirai 2.0`: RMSE `316.35`, Directional Accuracy `49.20%`
3. **[Fear & Greed Feature Evaluation (`docs/reports/fear_greed_feature_evaluation.md`)](docs/reports/fear_greed_feature_evaluation.md):**
   - Added daily `fng_value` and `fng_mean_3d`. SVC accuracy moved by `+0.16 pp` (`52.26%` -> `52.42%`, noise), XGBoost regression RMSE worsened (`403.92` -> `421.86`), and 5-window walk-forward mean return remained negative (**-0.32%**).
4. **[On-Chain Network Metrics Evaluation (`docs/reports/onchain_metrics_evaluation.md`)](docs/reports/onchain_metrics_evaluation.md):**
   - Added daily % changes in BTC transaction count, hash rate, and miner revenue. Walk-forward mean return degraded further to **-1.82%** because broadcasting daily macro series across 24 hourly candles adds dimensionality without intraday signal.

#### 5-Window Walk-Forward Validation Summary (`2022-07` to `2026-07`, `0.10%`/side fee)
| Feature Variant | Sweep @ 0.68 Trades ($n$) | Raw Binomial $p$-value | Survives Bonferroni? | Walk-Forward Windows Beating B&H | Mean Walk-Forward Return (Std) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Price/Volume Baseline** | 9 | `0.0195` | **No** ($> 0.0019$) | **1 / 5** | **-0.52%** ($\pm 3.93\%$) |
| **Phase 3A: + Fear & Greed Index** | 8 | `0.0352` | **No** ($> 0.0056$) | **1 / 5** | **-0.32%** ($\pm 1.21\%$) |
| **Phase 3B: + FNG + On-Chain Metrics** | 6 | `0.0156` | **No** ($> 0.0056$) | **1 / 5** | **-1.82%** ($\pm 2.26\%$) |

---

### 3.3 Holdout Strategy Backtest & Fee Sensitivity Analysis
On the 6,998-hour out-of-sample holdout (`2025-09-17 19:00 UTC` to `2026-07-06 08:00 UTC`), Bitcoin fell from **$115,226.93** to **$62,904.01** (**-45.41%** Buy & Hold return on $10,000 initial capital):

| Strategy / Configuration | Per-Side Friction (Round-Trip) | Total Trades | Market Exposure (% Time in Market) | Final Equity ($10k Start) | Total Return (%) | Proper Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bitcoin Buy & Hold Benchmark** | `0.00%` | 1 | `100.00%` | `$5,459.14` | `-45.41%` | Bear-market benchmark |
| **Unfiltered Classifier (`P(up) > 0.50`)** | `0.10%` (`0.20%` RT) | 1,196 | `~48.0%` | `$753.00` | `-92.47%` | Destroyed by fee churn |
| **Return-Filtered (`E[r] > 0.055%`, Code Default)** | `0.07%` (`0.14%` RT) | 105 | **3.57%** (`96.43%` in cash) | `$8,234.12` | `-17.66%` | **Cash preservation (NOT alpha)**; `0.055%` cutoff tuned post-hoc |
| **Legacy Friction Simulation (Historical Reference)** | `0.15%` (`0.30%` RT) | 105 | **3.57%** (`96.43%` in cash) | `$8,831.00` *(historical split)* | `-11.69%` | Historical Phase 1 reference run |

> **Why "-17.66% vs. -45.41%" is Cash Preservation, Not Alpha:**
> Earlier drafts of this README described the `+27.75 percentage point` difference between `-17.66%` and `-45.41%` as "Alpha." That framing was incorrect. Sitting in cash **96.43% of the time** during a `-45.41%` drawdown naturally avoids most of the crash (a 100% cash portfolio returns `0.00%` and beats Buy & Hold by `+45.41 pp`). On the `3.57%` of hours when the model actually entered the market, net return after `0.07%`/side friction was negative (`-17.66%`), and the `0.055%` return filter was selected on the holdout set.

#### Transaction Cost Assumption & Exchange Tier Sensitivity
- **Why `config.py` uses `0.07%` per side (`0.139951%` compound round-trip):** `config.py` sets `TAKER_FEE_RATE = 0.0005` (`0.05%`, which corresponds to Binance USDT-M Futures taker or discounted BNB/VIP tiers, **not** standard Binance Spot VIP-0 taker which is `0.10%`) plus `SLIPPAGE_RATE = 0.0002` (`0.02%`), giving `0.07%` per side and a compound break-even floor of $1 - (1 - 0.0007)^2 = 0.139951\%$.
- **Why `docs/reports/` uses `0.10%` per side (`0.20%` round-trip):** The research reports in `docs/reports/` use standard Binance Spot VIP-0 taker fees (`0.10%` per side, `0.20%` round-trip).
- **Sensitivity across fee tiers:** Because the active regressor's predicted 1-hour return distribution rarely exceeds `0.14%–0.30%`, raising the round-trip hurdle from `0.14%` (futures/VIP rate) to `0.20%` (standard spot rate) or `0.30%` (spot + slippage) causes the deterministic Risk Critic to veto almost 100% of proposed trades, collapsing the strategy to pure cash holding.

---

## 4. System Architecture & MLOps Engineering

Although the underlying 1-hour ML signal has no durable post-fee edge, the repository implements a complete, tested MLOps and paper-trading software architecture to practice production engineering patterns:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HOURLY INGESTION & FEATURE LAYER                     │
│  Binance 1H OHLCV + Alternative.me FNG + Blockchain.info On-Chain Metrics   │
│         └── src/feature_engineering.py (Strict Causal Rolling Windows)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CRYPTOGRAPHIC MODEL REGISTRY & INFERENCE                 │
│  src/model_loader.py: Verifies SHA-256 hashes before joblib deserialization │
│  ├── Active:    xgb_onnx_v1.0.0 (v1_39features) -> Primary Forecast         │
│  └── Candidate: xgb_v2.0.0      (v2_36stationary) -> Shadow Consensus Log   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              ORCHESTRATION & DETERMINISTIC RISK GATE (LangGraph)            │
│  src/agents/trading_graph.py: START -> strategist -> risk_critic -> END     │
│  ├── Strategist Node:  Combines ML forecast + ChromaDB episodic memory      │
│  └── Risk Critic Node: Deterministic 5-rule hard gate (0.139951% hurdle,    │
│                        3.8% volatility cap, conviction >= 4)                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             PERSISTENCE, RECONCILIATION & SERVING INFRASTRUCTURE            │
│  ├── Neon PostgreSQL (9 Tables): Predictions, Paper Ledger, Model Registry  │
│  ├── T+1h Reconciliation Engine: Pairs past forecasts with realized candles │
│  ├── FastAPI Backend (src/api/main.py): REST endpoints + Pydantic schemas   │
│  └── Streamlit UI (app.py): 3-tab live monitoring & governance dashboard    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Honest Design Rationale for Key Architectural Components
1. **Why LangGraph for a 2-Node Linear Pipeline (`src/agents/trading_graph.py`)?**
   - In production, a 2-step sequence (`strategist -> risk_critic`) where the Risk Critic is 5 deterministic `if` statements could be written as a plain Python function. LangGraph (`StateGraph(TradingAgentState)`) was used intentionally as an **orchestration learning exercise** to practice typed state transitions, LLM tool-calling fallbacks (`gemini-2.5-flash` with deterministic offline fallback when `GEMINI_API_KEY` is absent or quota-limited), and separating LLM narrative generation from hard quantitative risk rules.
2. **How the Deterministic Risk Critic Works (`src/agents/risk_critic_node.py`):**
   - An LLM never approves or vetoes trades on its own. Every `BUY`/`SELL` proposal must pass 5 hard Python checks in `risk_critic_node.py`:
     1. Valid action contract (`BUY`, `SELL`, or `HOLD`).
     2. `current_price > 0` and finite.
     3. **Fee Hurdle Gate:** `expected_return_pct >= MIN_HURDLE_PCT` (`0.139951%` round-trip break-even floor).
     4. **Volatility Circuit Breaker:** `volatility_range_pct <= 3.8%`.
     5. **Conviction Floor:** `conviction_score >= 4` (out of 10).
3. **Why a 9-Table PostgreSQL Schema (`src/database/models.py`)?**
   - Designed to model a multi-version MLOps audit trail on **Neon Serverless PostgreSQL** (with automatic SQLite fallback in CI/local runs):
     1. `market_ticks`: Stores canonical 1-hour OHLCV candles, 39 features JSON, and primary model inference outputs.
     2. `api_telemetry`: Observability table tracking external API calls, latency, rate limits, and token cost quotas.
     3. `council_audits`: Immutable audit trail for multi-agent deliberations, strategist actions, risk verdicts, and veto reasons.
     4. `reconciliations`: Ground-truth delayed reconciliation ledger with residual error decomposition and Black Swan outlier isolation.
     5. `paper_portfolio`: Forward $10,000 paper trading ledger tracking virtual cash, BTC balances, execution fees, slippage, and Alpha vs. Buy & Hold.
     6. `model_registry`: Authoritative registry tracking model versions, cryptographic SHA-256 digests, and feature schemas.
     7. `model_lifecycle_events`: Append-only audit trail tracking model state transitions (REGISTERED, EVALUATED, PROMOTED, etc.).
     8. `model_predictions`: Granular prediction ledger supporting point-in-time forecasts and delayed reconciliations across PRIMARY and SHADOW models.
     9. `prediction_investigations`: Qualitative post-mortem ledger preserving news headlines, macro events, and causal failure explanations.
4. **Delayed `T+1h` Reconciliation & the `"ACCURATE"` Tag (`src/engine/reconciliation.py`):**
   - Every hour, `scripts/run_hourly_worker.py` fetches the newly closed candle and reconciles predictions made $\ge 1$ hour earlier.
   - **Important metric caveat:** In `src/engine/reconciliation.py` (line 138), a prediction is labeled `"ACCURATE"` if `direction_correct` is `True` and `percentage_error <= 0.35%` (`MAX_ACCEPTABLE_ERROR_PCT = 0.35`), `"DIRECTION_ONLY"` if direction matched but error $> 0.35\%$, and `"MISSED_DIRECTION"` if direction was wrong. Because typical 1-hour BTC moves are small (`~0.3%–0.5%`), almost any directionally correct forecast falls under `0.35%` error. Always evaluate the model using **Out-of-Sample Directional Accuracy, Log-Loss, ROC-AUC, and RMSE vs. Naive Persistence**, not the UI `"ACCURATE"` badge count.

---

## 5. Known Limitations & Technical Debt

1. **No Exploitable Alpha at 1-Hour Resolution:** Neither engineered technical indicators (`v1_39features`, `v2_36stationary`), daily Fear & Greed/On-Chain merges, nor zero-shot foundation models (`Chronos-2`, `TimesFM 2.5`, `Moirai 2.0`) overcome the 1-hour signal-to-noise ceiling after transaction costs.
2. **Active Model (`xgb_onnx_v1.0.0`) Uses Non-Stationary Price Features:** While `xgb_v2.0.0` (`v2_36stationary`) fixes this by using strictly stationary features, `xgb_onnx_v1.0.0` remains the primary model in `config.py` (`ACTIVE_MODEL_VERSION = "xgb_onnx_v1.0.0"`) because `xgb_v2.0.0` did not beat `xgb_onnx_v1.0.0` on static holdout accuracy (`52.07%` vs. `53.26%`) and both regressors lose to naive persistence.
3. **Pickled Artifacts Committed to Git (`artifacts/models/*.pkl`):** For zero-cost deployment on Streamlit Community Cloud and GitHub Actions without paid S3/artifact storage, model `.pkl` files are committed directly to `artifacts/models/` and protected by fail-closed `SHA-256` verification in `src/model_loader.py`.
4. **External Cron Dispatch for Hourly Worker:** Because GitHub Actions' native `schedule:` cron can be delayed by 15–45 minutes on free-tier public repos, `.github/workflows/hourly_worker.yml` supports `workflow_dispatch` triggered by an external cron ping.

---

## 6. Repository Structure

```text
Mudra/
├── app.py                              # 3-tab Streamlit monitoring & model governance UI
├── config.py                           # Central configuration, SHA-256 hashes & fee invariants
├── Dockerfile                          # Multi-stage Python 3.11 container image
├── docker-compose.yml                  # Local FastAPI + Streamlit + PostgreSQL stack
├── requirements.txt                    # Pinned production dependencies
├── aiml_workspace/
│   └── model_v2/
│       └── main.ipynb                  # Model 2 (v2_36stationary) research, calibration & audit notebook
├── artifacts/
│   ├── models/
│   │   ├── model_manifest.json         # Active Model 1 (xgb_onnx_v1.0.0, v1_39features) manifest
│   │   ├── xgb_v2_manifest.json        # Shadow Candidate Model 2 (xgb_v2.0.0, v2_36stationary) manifest
│   │   ├── best_regressor_model.pkl    # Active XGBRegressor artifact
│   │   └── best_classifier_model.pkl   # Active XGBClassifier artifact
│   ├── shap_classifier_summary.png     # TreeSHAP global feature importance plot (Classifier)
│   └── shap_regressor_summary.png      # TreeSHAP global feature importance plot (Regressor)
├── data/
│   ├── raw/BTCUSDT-1H.csv              # 35,064 raw 1H Binance candles
│   ├── processed/featured_btc_data.csv # 34,990 cleaned & featured rows
│   └── results/                        # Backtest, threshold sweep & walk-forward CSV outputs
├── docs/
│   └── reports/                        # 4 empirical research reports (Classical, TSFM, FNG, On-Chain)
├── scripts/
│   ├── run_pipeline.py                 # End-to-end 7-stage training, SHAP & backtest CLI
│   ├── run_hourly_worker.py            # Live hourly inference, LangGraph & T+1h reconciliation CLI
│   ├── migrate_db.py                   # Idempotent 9-table PostgreSQL schema migration & seeding
│   └── experiments/                    # Controlled candidate training & threshold sweep scripts
├── src/
│   ├── feature_engineering.py          # Causal feature pipeline (supports v1_39features & v2_36stationary)
│   ├── model_loader.py                 # Fail-closed SHA-256 verification & multi-model consensus loader
│   ├── backtesting.py                  # Chronological backtester with compound fee/slippage accounting
│   ├── agents/                         # LangGraph Strategist, deterministic Risk Critic, FNG & On-Chain agents
│   ├── api/                            # FastAPI routes (main.py) & Pydantic v2 schemas (schemas.py)
│   ├── database/                       # SQLAlchemy 2.0 ORM models (9 tables) & connection pool
│   └── engine/                         # Paper trading ledger & delayed T+1h reconciliation engine
└── tests/                              # 13 pytest modules (500 collected test cases)
```

---

## 7. How to Run Locally

### 7.1 Environment Setup
```bash
git clone https://github.com/MohitSingh-2335/Mudra.git
cd Mudra

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (optional—the pipeline runs offline with SQLite and deterministic Risk Critic fallbacks if omitted):
```ini
DATABASE_URL=postgresql://user:password@localhost:5432/mudra_db
GEMINI_API_KEY=your_gemini_api_key_optional
INITIAL_CAPITAL_USD=10000.0
TAKER_FEE_RATE=0.0005
SLIPPAGE_RATE=0.0002
```

### 7.2 Initialize Database Schema (Dry-Run or Execute)
```bash
# Preview the 9-table schema and model registry seed records (safe dry-run)
python scripts/migrate_db.py

# Apply migrations and seed baseline/candidate model specs to PostgreSQL/SQLite
python scripts/migrate_db.py --execute
```

### 7.3 Run the Quantitative ML Pipeline (`scripts/run_pipeline.py`)
```bash
# Fast verification run (verifies SHA-256 hashes, runs backtest & live parity check)
python scripts/run_pipeline.py --skip-shap

# Full 7-stage pipeline (feature engineering, SHAP explainability, threshold sweep & backtest)
python scripts/run_pipeline.py

# Retrain baseline models or run Optuna Bayesian hyperparameter tuning
python scripts/run_pipeline.py --train-baseline
python scripts/run_pipeline.py --tune --trials 50
```

### 7.4 Run the Hourly Inference & Reconciliation Worker
```bash
python scripts/run_hourly_worker.py
```

### 7.5 Launch the Streamlit Dashboard & FastAPI Server
```bash
# Launch the 3-tab Streamlit dashboard (http://localhost:8501)
streamlit run app.py

# Launch the FastAPI REST service (http://localhost:8000/docs)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7.6 Run the Automated Test Suite
The `tests/` directory contains 13 test modules (500 collected unit and integration test cases) covering temporal split causality, SHA-256 artifact governance, execution cost math, LangGraph deterministic vetoes, and database reconciliation:
```bash
pytest -q
```
