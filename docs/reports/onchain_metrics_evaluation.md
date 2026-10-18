# Historical Study Report — On-Chain Metrics Agent (BTC/USDT, 1H)

**Status:** Complete. **Verdict:** No exploitable edge found. On-chain metrics (as implemented) do not change the signal ceiling established in Historical Study.

---

## 1. Objective

Test whether adding BTC on-chain network metrics (transaction count, hash rate, miners'
revenue — the second Historical Study agent per `research benchmark protocol`) moves classifier accuracy or
backtested return, using the same models and evaluation process as Historical Study and Historical Study —
isolating on-chain data as the only new variable added on top of the Historical Study (+FNG) feature set.

---

## 2. What was built

| Component | File | Purpose |
|---|---|---|
| On-chain agent | `src/agents/onchain_agent.py` | Fetches daily `n-transactions`, `hash-rate`, `miners-revenue` from Blockchain.info Charts API; fails soft (skips features, no synthetic default) if all charts fail |
| Feature wiring | `src/feature_engineering.py` | Conditional block adds `onchain_num_tx_change`, `onchain_hash_rate_change`, `onchain_miners_revenue_change` (day-over-day % change, not raw level) when present |
| Data prep wiring | `scripts/prepare_data.py` | Merges on-chain data into raw df, after FNG merge, before `create_features()` |
| Backtest/sweep/training wiring | `src/model_training.py`, `src/backtesting.py`, `scripts/experiments/sweep_threshold.py`, `experiments/backtesting/compare_models_backtest.py`, `experiments/backtesting/significance_test.py`, `experiments/backtesting/walk_forward_validate.py` | Same merge chained after `merge_fear_greed` at every script that independently loads raw data |
| Feature list update | `config.py` | Three `onchain_*_change` columns added to both `XGB_FEATURES` and `SVC_FEATURES` |
| SVC training cap fix | `src/model_training.py` | Added 15,000-row cap to SVC training (previously uncapped, taking hours); brought training time down to minutes, matching the cap convention used elsewhere in the project since Historical Study |

**Design choice — % change over raw level:** raw on-chain levels (e.g. hash rate) trend
upward over years and are highly autocorrelated/non-stationary. Day-over-day % change was
used instead, per the same reasoning applied to `return_1h_lag` in earlier phases — a stated
principle carried forward, not a new judgment call.

---

## 3. Methodology

Identical to Historical Study, with one additional variable:

- **Data:** `data/raw/BTCUSDT-1H.csv`, same chronological split, now merged with FNG (Historical Study)
  **and** daily on-chain metrics before feature engineering.
- **Models:** Same as Historical Study — LogisticRegression (primary), XGBoost, SVC. No new model
  families.
- **Fees, strategy, threshold sweep range (0.50–0.70):** unchanged.
- **Validation:** Same two-step check — binomial significance test with multiple-comparisons
  correction, then 5-window walk-forward validation.

---

## 4. Results

### 4.1 Static holdout — model training

| Model | Metric | Historical Study | Historical Study (+FNG) | Historical Study (+FNG+onchain) |
|---|---|---|---|---|
| SVC | Accuracy | 52.26% | 52.42% | 52.37% |
| XGBoost Regressor | RMSE (price-equiv.) | 403.92 (naive 398.63) | 421.86 (naive 398.79) | 411.30 (naive 395.52) |

All three variants: SVC flat within noise, XGBoost regression loses to naive persistence
baseline every time.

### 4.2 Threshold sweep — LogisticRegression (`scripts/experiments/sweep_threshold.py`)

Best-looking result, threshold 0.68:

| Variant | n | Win rate | Return |
|---|---|---|---|
| Historical Study | 9 | 88.9% | +6.18% |
| Historical Study (+FNG) | 8 | 87.5% | +4.62% |
| Historical Study (+FNG+onchain) | 6 | 100.0% | +5.17% |

Same shape as prior phases: losses across 0.50–0.65, apparent win only once trade count
collapses to single digits — here even thinner than before (n=6).

### 4.3 Significance test (`experiments/backtesting/significance_test.py`)

| Config | n | Win rate | p-value | Bonferroni threshold (9 configs) | Survives? |
|---|---|---|---|---|---|
| LogReg @ 0.68 | 6 | 100.0% | 0.0156 | 0.0056 | [NO] No |
| XGBoost @ 0.70 | 91 | 56.0% | 0.1472 | — | [NO] Not significant |

### 4.4 Walk-forward validation (`experiments/backtesting/walk_forward_validate.py`)

| Window | Period | Trades | Win rate | Strategy return | Buy & Hold |
|---|---|---|---|---|---|
| 1 | 2022-07 to 2023-04 | 10 | 50.0% | -2.69% | +79.98% |
| 2 | 2023-04 to 2024-02 | 30 | 60.0% | -2.02% | +69.78% |
| 3 | 2024-02 to 2024-11 | 17 | 41.2% | -4.83% | +56.36% |
| 4 | 2024-11 to 2025-09 | 3 | 33.3% | -0.84% | +5.62% |
| 5 | 2025-09 to 2026-07 | 2 | 50.0% | +1.29% | -10.38% |

**1 of 5 windows beat Buy & Hold. Mean return -1.82% (std 2.26%)** — worse than both Historical Study
(-0.52%) and Historical Study (-0.32%).

### Summary across all three feature-set variants

| Variant | Sweep@0.68 n | p-value @0.68 | Walk-forward wins | Mean WF return |
|---|---|---|---|---|
| Historical Study (baseline) | 9 | 0.0195 (fails) | 1/5 | -0.52% |
| Historical Study (+FNG) | 8 | 0.0352 (fails) | 1/5 | -0.32% |
| Historical Study (+FNG+onchain) | 6 | 0.0156 (fails) | 1/5 | -1.82% |

---

## 5. Root-cause analysis

1. **Same multiple-comparisons signature, third consecutive occurrence.** High-threshold
   "wins" keep reappearing regardless of feature set — strong evidence this is a property of
   sweeping thresholds on noisy, low-trade-count data, not of any specific feature set.
2. **Granularity mismatch, same as FNG.** On-chain metrics update daily; broadcasting one
   daily % change across ~24 hourly rows produces a near-constant feature within any given
   day, unlikely to carry hourly-resolution information even if the underlying metric has
   real signal at daily/weekly scale.
3. **Walk-forward result got worse, not just flat.** Mean return degraded further than Historical Study. Plausible explanation: three new features add dimensionality without adding signal,
   which can dilute an already-thin edge in models like SVC/LogReg — consistent with, not
   contradictory to, a genuine-noise interpretation.
4. **Consistent with the standing Historical Study conclusion**: three independent attempts to add
   input signal (TSFMs' architecture change, FNG, on-chain) have now all failed to move the
   needle. The pattern points increasingly toward a structural ceiling in hourly BTC
   direction prediction from these input categories, not a gap specific to any one omitted
   feature.

---

## 6. What this rules out (and doesn't)

- [NO] **Ruled out:** Daily on-chain metrics (n-transactions, hash-rate, miners-revenue, as
  day-over-day % change), added on top of price/volume + FNG, do not produce a fee-surviving,
  replicable directional edge on BTC/USDT 1H data.
- [NOTE] **Not ruled out:** A differently-framed on-chain feature (e.g. multi-day trend instead of
  single-day change, or interaction terms), other on-chain metrics not tested here (exchange
  netflows, active addresses), or LLM sentiment (the remaining untested Historical Study agent) —
  which carries qualitatively different information than either FNG or on-chain activity
  metrics.

---

## 7. Recommendation

Three Historical Study sub-tests in (TSFM architecture in Historical Study, FNG, on-chain), all negative with
the identical failure signature. Two honest paths forward:

- **Test sentiment (Historical Study) as the last remaining Historical Study agent**, since it's qualitatively
  different from both prior attempts (an LLM's read on news text, not a numeric market/network
  indicator) — worth completing the planned agent set before concluding Historical Study broadly.
- **Or pause Historical Study and switch to Historical Study (multi-asset)** if the emerging hypothesis is that
  the ceiling is structural to hourly BTC prediction specifically, rather than fixable by
  adding more BTC-specific external signals one at a time.

Given sentiment is the last untested agent and qualitatively different from the two negative
results so far, finishing it before deciding on Historical Study vs. Historical Study is the more complete way
to close this question out.

**Process carried forward (unchanged):** any new feature/config must clear both the corrected
significance test and 5-window walk-forward validation before being treated as a real result.

---

## 8. Artifacts produced during this evaluation

- `src/agents/onchain_agent.py` — on-chain fetch/merge agent (kept, reusable for future
  framing experiments)
- SVC training cap fix in `src/model_training.py` (kept — general project improvement,
  independent of this study's data-signal question)
- Modified: `src/feature_engineering.py`, `scripts/prepare_data.py`, `config.py`,
  `src/model_training.py`, `src/backtesting.py`, `scripts/experiments/sweep_threshold.py`,
  `experiments/backtesting/compare_models_backtest.py`, `experiments/backtesting/significance_test.py`,
  `experiments/backtesting/walk_forward_validate.py`