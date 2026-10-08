# Phase 4a Report â€” Fear & Greed Index Agent (BTC/USDT, 1H)

**Status:** Complete. **Verdict:** No exploitable edge found. FNG does not change the signal ceiling established in Phase 1/2.

---

## 1. Objective

Test whether adding the Fear & Greed Index (first non-price/volume signal available to either
the classical models or the TSFMs tested in Phase 1/2) moves classifier accuracy or backtested
return, using the exact same models and evaluation process as Phase 1 â€” isolating the feature
as the only new variable.

---

## 2. What was built

| Component | File | Purpose |
|---|---|---|
| Fear & Greed agent | `src/agents/fear_greed_agent.py` | Fetches daily FNG index from alternative.me, fails soft to neutral=50 on API failure |
| Feature wiring | `src/feature_engineering.py` | Conditional block adds `fng_value` + `fng_mean_3d` (72h/3-day rolling mean) when present |
| Data prep wiring | `scripts/prepare_data.py` | Merges FNG into raw df before `create_features()` |
| Backtest/sweep wiring | `src/backtesting.py`, `scripts/experiments/sweep_threshold.py`, `experiments/backtesting/compare_models_backtest.py`, `experiments/backtesting/significance_test.py`, `experiments/backtesting/walk_forward_validate.py` | Same merge applied at every script that independently loads raw data, so FNG is present everywhere Phase 1's baseline was measured |
| Feature list update | `config.py` | `fng_value`, `fng_mean_3d` added to both `XGB_FEATURES` and `SVC_FEATURES` |

---

## 3. Methodology

Identical to Phase 1, with one variable changed:

- **Data:** `data/raw/BTCUSDT-1H.csv`, same chronological split, now merged with daily FNG values
  (broadcast across all hourly rows on that date) before feature engineering.
- **Models:** Same as Phase 1 â€” LogisticRegression (primary, via `src/backtesting.py` and
  `scripts/experiments/sweep_threshold.py`), XGBoost, SVC. No new model families introduced â€” this
  isolates whether the *feature* helps, not whether a different model can exploit it better.
- **Fees, strategy, threshold sweep range (0.50â€“0.70), SVM row cap:** unchanged from Phase 1.
- **Validation:** Same two-stage check as Phase 1 â€” binomial significance test with
  multiple-comparisons correction, then 5-window walk-forward validation â€” applied before
  treating any single-split result as real.

---

## 4. Results

### 4.1 Static holdout â€” model training

| Model | Metric | Phase 1 (no FNG) | Phase 4a (+FNG) | Î” |
|---|---|---|---|---|
| SVC | Accuracy | 52.26% | 52.42% | +0.16pp (noise) |
| XGBoost Regressor | RMSE (price-equiv.) | 403.92 (naive: 398.63) | 421.86 (naive: 398.79) | **worse**, further above naive |

### 4.2 Threshold sweep â€” LogisticRegression (`scripts/experiments/sweep_threshold.py`)

| Threshold | Num trades | Win rate | Return | Buy & Hold |
|---|---|---|---|---|
| 0.50 | 1196 | 50.5% | -92.47% | -45.59% |
| 0.52 | 1086 | 46.7% | -92.43% | -45.59% |
| 0.55 | 696 | 46.0% | -78.49% | -45.59% |
| 0.58 | 358 | 50.3% | -53.09% | -45.59% |
| 0.60 | 183 | 53.6% | -28.13% | -45.59% |
| 0.63 | 52 | 44.2% | -16.68% | -45.59% |
| 0.65 | 25 | 56.0% | +0.78% | -45.59% |
| 0.68 | 8 | 87.5% | +4.62% | -45.59% |
| 0.70 | 2 | 100.0% | +1.82% | -45.59% |

**Same shape as Phase 1**: heavy losses at low/mid thresholds, apparent recovery only once
trade count collapses to single digits. Best-looking config (0.68) is *not* better than
Phase 1's no-FNG result at the same threshold (+4.62% now vs +6.18% then, n=8 vs n=9).

### 4.3 Significance test (`experiments/backtesting/significance_test.py`)

| Config | n | Win rate | p-value (vs 50%) | Bootstrap 95% CI on mean return |
|---|---|---|---|---|
| LogReg @ 0.68 | 8 | 87.5% | 0.0352 | [0.055%, 1.490%] |
| XGBoost @ 0.70 | 83 | 48.2% | 0.6696 (not significant) | [-0.032%, 0.210%] (includes 0) |

9 thresholds swept for LogReg â†’ correction threshold = 0.05/9 â‰ˆ 0.0056. **p=0.0352 does not
survive correction.** Same failure mode as Phase 1, at a smaller scale.

### 4.4 Walk-forward validation (`experiments/backtesting/walk_forward_validate.py`)

| Window | Period | Trades | Win rate | Strategy return | Buy & Hold |
|---|---|---|---|---|---|
| 1 | 2022-07 to 2023-04 | 4 | 50.0% | -1.98% | +75.65% |
| 2 | 2023-04 to 2024-02 | 19 | 63.2% | -0.56% | +76.41% |
| 3 | 2024-02 to 2024-11 | 5 | 100.0% | +0.25% | +78.83% |
| 4 | 2024-11 to 2025-09 | 2 | 50.0% | -0.62% | +6.85% |
| 5 | 2025-09 to 2026-07 | 2 | 50.0% | +1.29% | -10.92% |

**1 of 5 windows beat Buy & Hold. Mean return -0.32% (std 1.21%)** â€” centered on zero, same
as Phase 1's -0.52% (std 3.93%). No replicated edge.

---

## 5. Root-cause analysis

1. **Same multiple-comparisons signature as Phase 1.** High-threshold "wins" reappear at
   nearly identical trade counts and win rates to the no-FNG baseline â€” strong evidence FNG
   didn't change the underlying signal, it just rode along with the same noise pattern.
2. **Granularity mismatch.** FNG updates once daily; the target is hourly direction. Broadcasting
   one daily value across ~24 rows, plus a 3-day rolling mean, produces a feature that's nearly
   constant within any given day â€” unlikely to carry information at hourly resolution regardless
   of whether FNG itself is informative at daily/weekly scale.
3. **XGBoost regression got measurably worse**, not just flat â€” the `fng_mean_3d` 72-row window
   plausibly added noise rather than signal to that specific target.
4. **Consistent with Phase 1/2's standing conclusion**: the ceiling looks like a property of the
   task/data at this granularity, not a gap any single new input has closed so far.

---

## 6. What this rules out (and doesn't)

- âŒ **Ruled out:** Daily-value Fear & Greed Index (raw + 3-day mean), fed into
  LogisticRegression/SVC/XGBoost on top of the existing feature set, does not produce a
  fee-surviving, replicable directional edge on BTC/USDT 1H data.
- âš ï¸ **Not ruled out:** A differently-framed FNG feature (e.g. day-over-day *change* in FNG
  rather than level, or an interaction term with volatility) might carry information this
  framing didn't capture. Also not ruled out: on-chain metrics and LLM sentiment â€” the two
  remaining Phase 4 agents â€” which carry different information than a coarse daily sentiment
  index and haven't been tested.

---

## 7. Recommendation

Treat FNG (as implemented here) as a closed negative result â€” don't re-sweep it further, for
the same multiple-comparisons reason Phase 1/2 flagged. Two reasonable next moves:

- **Continue Phase 4:** test on-chain metrics or LLM sentiment next, one at a time, same
  isolate-one-variable process used here â€” these carry genuinely different information than
  FNG, so this result doesn't predict their outcome.
- **Switch to Phase 3 (multi-asset):** if the working hypothesis shifts from "this asset needs
  a better feature" to "this feature set is close to its ceiling for BTC specifically," it may
  be more informative to check whether the same pipeline performs differently on a
  higher-liquidity or lower-liquidity asset, rather than continuing to add BTC-only signals.

**Process carried forward (still applies):** any new feature/config must clear both the
corrected significance test and 5-window walk-forward validation before being treated as a
real result â€” this phase is the third consecutive confirmation of why that bar exists.

---

## 8. Artifacts produced this phase

- `src/agents/fear_greed_agent.py` â€” FNG fetch/merge agent (kept, reusable for future framing experiments)
- `data/results/threshold_sweep.csv` â€” updated with +FNG numbers (overwrites Phase 1's file â€” Phase 1's
  numbers are preserved in `phase_1.md` Â§4.2 for comparison)
- Modified: `src/feature_engineering.py`, `scripts/prepare_data.py`, `config.py`, `src/backtesting.py`,
  `scripts/experiments/sweep_threshold.py`, `experiments/backtesting/compare_models_backtest.py`,
  `experiments/backtesting/significance_test.py`, `experiments/backtesting/walk_forward_validate.py`