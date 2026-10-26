# Phase 1 Research Report — Backtesting the Classical ML Models (BTC/USDT, 1H)

**Status:** Complete. **Verdict:** No exploitable edge found. Proceeding to Phase 2 (pretrained time-series foundation models).

---

## 1. Objective

Determine whether any classifier (predicting next-hour BTC/USDT direction) combined with a
confidence threshold produces a strategy that beats Buy & Hold **after fees**, on a proper
chronological (non-leaky) train/test split.

---

## 2. What was built

| Component | File | Purpose |
|---|---|---|
| Model family comparison | `scripts/experiments/find_best_models.py` *(archived; see `data/results/model_comparison_backtest.csv`)* | Compare LinearRegression/RandomForest/XGBoost/SVR (regression) and LogisticRegression/RandomForest/XGBoost/SVC (classification) on identical chronological split + fixed feature sets |
| Event-driven backtester | `src/backtesting.py` | Trains a classifier, walks forward through held-out data, simulates long/flat strategy with 0.1%/side fees, computes Sharpe/drawdown/win-rate/P&L |
| Threshold sweep (single model) | `scripts/experiments/sweep_threshold.py` | Sweeps probability thresholds and saves results to `data/results/threshold_sweep.csv` |
| Threshold sweep (all models) | `compare_models_backtest.py` *(archived; output in `data/results/model_comparison_backtest.csv`)* | LogisticRegression, SVC, XGBoost × thresholds 0.50–0.70 |
| Statistical significance test | `significance_test.py` *(archived)* | Binomial test on win rate + bootstrap CI on per-trade returns, for the best-looking configs |
| Walk-forward validation | `walk_forward_validate.py` *(archived; output in `data/results/walk_forward_validation.csv`)* | Re-tests the winning config across 5 independent, non-overlapping time windows |

---

## 3. Methodology

- **Data:** `data/raw/BTCUSDT-1H.csv`, 35,031 rows after initial feature engineering (later refined to 34,990 rows after removing the 2023-03-24 12:00 UTC exchange-halt candle and 1-hour purge buffer).
- **Split:** Chronological only (`df.iloc[:split_idx]` / `df.iloc[split_idx:]`), never random —
  the codebase's original `train_test_split(random_state=42)` was identified as a data-leakage
  bug and was **not** used for any of this work.
- **Target:**
  - Regression target changed from `Target_Close` (absolute price) to `Target_Return`
    (% change to next close) — every regressor tested on absolute price lost to a naive
    persistence baseline; this is a target-framing fix, not a tuning fix.
  - Classification target: `Target_Movement` (1 = up, 0 = down).
- **Fees:** 0.1% per side (0.2% round trip), matching standard Binance Spot VIP-0 taker fees. A backtest without
  fees is not meaningful for this use case.
- **Strategy:** Simple long/flat. Enter long when `P(up) > threshold`, exit when it drops
  back below.
- **SVM training cap:** SVC/SVR capped at the most recent 15,000 rows across all scripts, to
  keep training time reasonable.

---

## 4. Results

### 4.1 Model family leaderboard

**Regression** — all models lost to the naive persistence baseline:

| Model | RMSE (price-equivalent) | Beats baseline? |
|---|---|---|
| Naive Baseline (persistence) | 398.63 | — |
| Linear Regression | 399.39 | [NO] |
| SVR (rbf) | 403.82 | [NO] |
| XGBoost | 403.92 | [NO] |
| Random Forest | 409.03 | [NO] |

**Classification** — all models beat the naive baseline, but only barely:

| Model | Accuracy | Beats baseline (49.58%)? |
|---|---|---|
| Logistic Regression | 52.49% | [YES] |
| SVC (rbf) | 52.26% | [YES] |
| XGBoost | 52.15% | [YES] |
| Random Forest | 51.36% | [YES] |

**Conclusion at this point:** Regression is a dead end on this feature set (no model beats
"do nothing"). Classification has a small, real-looking edge over guessing — worth
backtesting to see if it survives fees.

### 4.2 Threshold sweep — all 3 classifiers × 9 thresholds (`data/results/model_comparison_backtest.csv`)

Full results in `data/results/model_comparison_backtest.csv`. Summary:

- **At threshold 0.50–0.60** (i.e. most of the data), **every model loses badly** —
  total returns of **-14% to -94%**, all worse than Buy & Hold (-46% over this
  particular test period).
- **As threshold rises, apparent returns improve monotonically for all three models** —
  the same pattern across three independently-trained algorithms. This is the signature
  of the threshold filtering down to fewer and fewer trades (n->0), not a discovered edge.
- **SVC's `predict_proba` rarely exceeds 0.60** — 0 trades at thresholds >=0.63, making it
  incomparable to the other two at high thresholds.
- **Best-looking single result:** LogisticRegression @ threshold=0.68 -> **+6.18% return**,
  88.9% win rate, but only **n=9 trades**.
- Second best: XGBoost @ threshold=0.70 -> +2.34% return, n=23 trades, but win rate only 47.8%
  (positive return despite <50% win rate implies wins sized larger than losses — needs
  scrutiny, not just accepted at face value).

### 4.3 Statistical significance test

| Config | n | Win rate | Binomial p-value (vs 50%) | Bootstrap 95% CI on mean return |
|---|---|---|---|---|
| LogReg @ 0.68 | 9 | 88.9% | **0.0195** (raw) | [0.197%, 1.534%] |
| XGBoost @ 0.70 | 23 | 47.8% | 0.6612 (not significant) | [0.038%, 0.607%] |

LogReg's raw p-value looked significant in isolation — **but this ignores that 27 configs
(3 models × 9 thresholds) were swept and this was the single best result selected after the
fact.** Applying a Bonferroni correction for 27 comparisons: significance threshold becomes
`0.05 / 27 ≈ 0.0019`. **LogReg's p=0.0195 does not survive this correction.** This is a
textbook multiple-comparisons false positive, not a validated edge.

### 4.4 Walk-forward validation (`data/results/walk_forward_validation.csv`)

Re-ran LogisticRegression @ threshold=0.68 across **5 independent, non-overlapping windows**
spanning 2022-07 to 2026-07, to test whether the apparent edge replicates out-of-sample:

| Window | Period | Trades | Win rate | Strategy return | Buy & Hold return |
|---|---|---|---|---|---|
| 1 | 2022-07 to 2023-04 | 6 | 50.0% | -2.25% | +64.44% |
| 2 | 2023-04 to 2024-02 | 10 | 90.0% | +5.32% | +68.53% |
| 3 | 2024-02 to 2024-11 | 16 | 50.0% | -4.86% | +68.04% |
| 4 | 2024-11 to 2025-09 | 4 | 50.0% | -2.10% | +6.99% |
| 5 | 2025-09 to 2026-07 | 2 | 50.0% | +1.29% | -10.87% |

**Result: strategy beat Buy & Hold in exactly 1 of 5 windows.** The 90% win rate was
specific to Window 2 (n=10) — every other window sits at a flat 50%, i.e. a coin flip.
Mean return across windows: **-0.52%** (std 3.93%) — centered on zero.

**This confirms the significance test's warning.** The original "winning" result was the
one lucky draw out of many tested configs, not a real, repeatable edge.

---

## 5. Root-cause analysis — why this failed

1. **Weak base signal.** Classifier accuracy tops out at ~52.5% vs a ~49.6% baseline —
   a ~3-point edge. That's real but small, and standard 0.1%/side trading fees (0.2% round
   trip) eat into a signal this thin very quickly.
2. **High-threshold filtering looks like skill but is mostly noise.** As the confidence
   threshold rises, trade count collapses toward single digits. Small-n win rates swing
   wildly (50%->90%->50% across windows) — this is sampling variance, not model improvement.
3. **Multiple comparisons inflate apparent significance.** Sweeping 27 (model × threshold)
   configs and reporting the single best one, without correction, will produce a
   "significant" result close to 1-in-20 of the time by chance alone — which is
   approximately what happened here.
4. **Opportunity cost of sitting in cash.** At high thresholds, the strategy trades so
   rarely (e.g. 2–16 times over ~8 months) that even when it doesn't lose money, it misses
   the underlying asset's much larger trend (Buy & Hold +64–68% in three of the five
   windows during 2022–2024).
5. **Feature set may simply lack forward-looking information.** `SVC_FEATURES` are almost
   entirely derived from price/volume history (technical indicators, rolling stats, lags).
   No external signal (sentiment, on-chain, macro) is present yet — planned for Phase 3 in the research roadmap.

---

## 6. What this rules out (and doesn't)

- [NO] **Ruled out:** LogisticRegression / SVC / XGBoost, on the price/volume technical
  feature set, with a simple long/flat threshold strategy, do not have a fee-surviving,
  replicable directional edge on BTC/USDT 1H data.
- [NO] **Ruled out:** "Just raise the confidence threshold" as a fix — it doesn't create
  edge, it just shrinks the sample until noise looks like a pattern.
- [NOTE] **Not ruled out:** A better feature set (external data — Phase 3 agents), a different
  target/strategy formulation (e.g. position sizing instead of binary long/flat, multi-class
  instead of binary, holding period other than 1h), or a fundamentally different model
  architecture (Phase 2 foundation models) might still find something these three algorithms couldn't. The
  negative result here is specific to *this* feature set and *these* three classical models —
  not proof that no edge exists in the data at all.

---

## 7. Recommendation

Per the research benchmark protocol, this is the negative result that justifies moving to
**Phase 2 — evaluate pretrained time-series foundation models** (Chronos-2, TimesFM 2.5,
or Moirai 2.0) rather than continuing to tune classical classifiers on this feature set.
Zero-shot forecasting gives a fast first signal on whether the foundation-model direction is worth
pursuing before any fine-tuning investment.

**Before starting Phase 2, carry forward as fixed process (don't repeat Phase 1's single-split trap):**
- Any new model/config must be validated on **multiple independent time windows**, not a
  single train/test split, before being treated as a result.
- Any threshold or hyperparameter sweep must apply a **multiple-comparisons correction**
  (or a held-out validation window used only *after* the winning config is picked from a
  separate tuning window) before reporting significance.
- Always compare against Buy & Hold **and** against the naive baseline — both, not either.

---

## 8. Artifacts retained in repository

- `src/backtesting.py` — reusable backtesting engine
- `scripts/experiments/sweep_threshold.py` — threshold sweep runner
- `data/results/model_comparison_backtest.csv` — full sweep results (27 configs)
- `data/results/threshold_sweep.csv` — single-model threshold sweep output
- `data/results/walk_forward_validation.csv` — 5-window replication results
- *(Note: Standalone one-off experiment runners `find_best_models.py`, `compare_models_backtest.py`, `significance_test.py`, and `walk_forward_validate.py` were archived after exporting their CSV tables to `data/results/`.)*