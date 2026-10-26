# Phase 3B Research Report — On-Chain Metrics Agent (BTC/USDT, 1H)

**Status:** Complete. **Verdict:** No exploitable edge found. On-chain metrics (as implemented) do not change the signal ceiling established in Phase 1.

---

## 1. Objective

Test whether adding BTC on-chain network metrics (transaction count, hash rate, miners'
revenue — the second Phase 3 external data agent) moves classifier accuracy or
backtested return, using the same models and evaluation process as Phase 1 and Phase 3A —
isolating on-chain data as the only new variable added on top of the Phase 3A (+FNG) feature set.

---

## 2. What was built

| Component | File | Purpose |
|---|---|---|
| On-chain agent | `src/agents/onchain_agent.py` | Fetches daily `n-transactions`, `hash-rate`, `miners-revenue` from Blockchain.info Charts API; fails soft (skips features, no synthetic default) if all charts fail |
| Feature wiring | `src/feature_engineering.py` | Conditional block adds `onchain_num_tx_change`, `onchain_hash_rate_change`, `onchain_miners_revenue_change` (day-over-day % change, not raw level) when present |
| Data prep wiring | `scripts/prepare_data.py` | Merges on-chain data into raw df, after FNG merge, before `create_features()` |
| Backtest/sweep/training wiring | `src/model_training.py`, `src/backtesting.py`, `scripts/experiments/sweep_threshold.py` | Same merge chained after `merge_fear_greed` at every script that independently loads raw data |
| Feature list update | `config.py` | Three `onchain_*_change` columns added to `REGRESSOR_FEATURES` and `CLASSIFIER_FEATURES` |
| SVC training cap fix | `src/model_training.py` | Added 15,000-row cap to SVC training (previously uncapped, taking hours); brought training time down to minutes, matching the cap convention used since Phase 1 |

**Design choice — % change over raw level:** raw on-chain levels (e.g. hash rate) trend
upward over years and are highly autocorrelated/non-stationary. Day-over-day % change was
used instead, per the stationarity principle carried forward across the pipeline.

---

## 3. Methodology

Identical to Phase 3A, with one additional variable:

- **Data:** `data/raw/BTCUSDT-1H.csv`, same chronological split, now merged with FNG (Phase 3A)
  **and** daily on-chain metrics before feature engineering.
- **Models:** Same as Phase 1 and Phase 3A — LogisticRegression (primary), XGBoost, SVC. No new model
  families.
- **Fees (`0.10%`/side), strategy, threshold sweep range (`0.50`–`0.70`):** unchanged.
- **Validation:** Same two-step check — binomial significance test with multiple-comparisons
  correction, then 5-window walk-forward validation.

---

## 4. Results

### 4.1 Static holdout — model training

| Model | Metric | Phase 1 (Baseline) | Phase 3A (+FNG) | Phase 3B (+FNG+onchain) |
|---|---|---|---|---|
| SVC | Accuracy | 52.26% | 52.42% | 52.37% |
| XGBoost Regressor | RMSE (price-equiv.) | 403.92 (naive 398.63) | 421.86 (naive 398.79) | 411.30 (naive 395.52) |

All three variants: SVC flat within noise, XGBoost regression loses to naive persistence
baseline every time.

### 4.2 Threshold sweep — LogisticRegression (`scripts/experiments/sweep_threshold.py`)

Best-looking result, threshold 0.68:

| Variant | n | Win rate | Return |
|---|---|---|---|
| Phase 1 (Baseline) | 9 | 88.9% | +6.18% |
| Phase 3A (+FNG) | 8 | 87.5% | +4.62% |
| Phase 3B (+FNG+onchain) | 6 | 100.0% | +5.17% |

Same shape as prior phases: losses across 0.50–0.65, apparent win only once trade count
collapses to single digits — here even thinner than before (n=6).

### 4.3 Significance test

| Config | n | Win rate | p-value | Bonferroni threshold (9 configs) | Survives? |
|---|---|---|---|---|---|
| LogReg @ 0.68 | 6 | 100.0% | 0.0156 | 0.0056 | [NO] No |
| XGBoost @ 0.70 | 91 | 56.0% | 0.1472 | — | [NO] Not significant |

### 4.4 Walk-forward validation

| Window | Period | Trades | Win rate | Strategy return | Buy & Hold |
|---|---|---|---|---|---|
| 1 | 2022-07 to 2023-04 | 10 | 50.0% | -2.69% | +79.98% |
| 2 | 2023-04 to 2024-02 | 30 | 60.0% | -2.02% | +69.78% |
| 3 | 2024-02 to 2024-11 | 17 | 41.2% | -4.83% | +56.36% |
| 4 | 2024-11 to 2025-09 | 3 | 33.3% | -0.84% | +5.62% |
| 5 | 2025-09 to 2026-07 | 2 | 50.0% | +1.29% | -10.38% |

**1 of 5 windows beat Buy & Hold. Mean return -1.82% (std 2.26%)** — worse than both Phase 1
(-0.52%) and Phase 3A (-0.32%).

### Summary across all three feature-set variants

| Variant | Sweep@0.68 n | p-value @0.68 | Walk-forward wins | Mean WF return |
|---|---|---|---|---|
| Phase 1 (baseline) | 9 | 0.0195 (fails) | 1/5 | -0.52% |
| Phase 3A (+FNG) | 8 | 0.0352 (fails) | 1/5 | -0.32% |
| Phase 3B (+FNG+onchain) | 6 | 0.0156 (fails) | 1/5 | -1.82% |

---

## 5. Root-cause analysis

1. **Same multiple-comparisons signature, third consecutive occurrence.** High-threshold
   "wins" keep reappearing regardless of feature set — strong evidence this is a property of
   sweeping thresholds on noisy, low-trade-count data, not of any specific feature set.
2. **Granularity mismatch, same as FNG.** On-chain metrics update daily; broadcasting one
   daily % change across ~24 hourly rows produces a near-constant feature within any given
   day, unlikely to carry hourly-resolution information even if the underlying metric has
   real signal at daily/weekly scale.
3. **Walk-forward result got worse, not just flat.** Mean return degraded further than Phase 3A. Plausible explanation: three new features add dimensionality without adding signal,
   which can dilute an already-thin edge in models like SVC/LogReg — consistent with, not
   contradictory to, a genuine-noise interpretation.
4. **Consistent with the standing Phase 1–2 conclusion**: three independent attempts to add
   input signal (TSFMs' architecture change in Phase 2, FNG in Phase 3A, on-chain in Phase 3B) have now all failed to move the
   needle. The pattern points toward a structural ceiling in hourly BTC
   direction prediction from public daily/OHLCV categories, not a gap specific to any one omitted
   feature.

---

## 6. What this rules out (and doesn't)

- [NO] **Ruled out:** Daily on-chain metrics (n-transactions, hash-rate, miners-revenue, as
  day-over-day % change), added on top of price/volume + FNG, do not produce a fee-surviving,
  replicable directional edge on BTC/USDT 1H data.
- [NOTE] **Not ruled out:** Higher-frequency order-book/liquidation data, or using LLM/RAG agents strictly as risk-management and post-mortem diagnostic tools rather than expecting 1-hour directional alpha from coarse daily macro series.

---

## 7. Recommendation

Four consecutive empirical studies (Phase 1 Classical ML, Phase 2 Zero-Shot Foundation Models, Phase 3A Fear & Greed, Phase 3B On-Chain Metrics) confirm that 1-hour BTC/USDT directional forecasting from public OHLCV + daily sentiment/on-chain series hits a hard ~52.5%–53.3% accuracy ceiling that does not survive transaction fees across walk-forward windows.

**Process carried forward (unchanged):** any new feature/config (such as Model 2's 36-stationary-feature schema `v2_36stationary`) must clear both out-of-sample naive-persistence benchmarks and multi-window validation before being claimed as alpha.

---

## 8. Artifacts retained in repository

- `src/agents/onchain_agent.py` — on-chain fetch/merge agent
- SVC training cap fix in `src/model_training.py`
- Modified: `src/feature_engineering.py`, `scripts/prepare_data.py`, `config.py`, `src/model_training.py`, `src/backtesting.py`, `scripts/experiments/sweep_threshold.py`