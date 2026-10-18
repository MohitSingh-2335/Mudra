# Historical Study Report — Zero-Shot Pretrained Time-Series Foundation Models (BTC/USDT, 1H)

**Status:** Complete. **Verdict:** No exploitable edge found. Not fine-tuning any candidate. Proceeding to Historical Study or Historical Study.

---

## 1. Objective

Per `research benchmark protocol` Historical Study: test whether a pretrained time-series foundation model (TSFM)
gives a better signal than the classical models ruled out in Historical Study, at the cheapest
possible cost — zero-shot inference, no training — before committing to any fine-tuning
investment or building PatchTST from scratch (Historical Study).

Three candidates were named in the roadmap: **Chronos-2** (Amazon), **TimesFM 2.5**
(Google), **Moirai 2.0** (Salesforce). All three were tested.

---

## 2. What was built

| Component | File | Purpose |
|---|---|---|
| Zero-shot eval — Chronos-2 | `zero_shot_chronos.py` | Rolling 1-step forecast, `amazon/chronos-2`, no training |
| Zero-shot eval — TimesFM 2.5 | `zero_shot_timesfm.py` | Rolling 1-step forecast, `google/timesfm-2.5-200m-pytorch`, no training |
| Zero-shot eval — Moirai 2.0 | `zero_shot_moirai.py` | Batched rolling 1-step forecast, `Salesforce/moirai-2.0-R-small`, no training |

---

## 3. Methodology

- **Data:** Same held-out test split as Historical Study — chronological, last 20% of
  `data/raw/BTCUSDT-1H.csv` after feature engineering.
- **Eval set:** 500 held-out points, each with a 512-hour rolling context window
  (`CONTEXT_LEN=512`, `N_EVAL=500`), fed to each model fresh — no fine-tuning, no
  gradient updates, purely zero-shot inference.
- **Metrics (identical across all three models, for direct comparability):**
  - RMSE (price-level), vs. naive persistence baseline (310.06)
  - Directional accuracy, vs. naive baseline (49.58%) and vs. Historical Study's best classical
    model, LogisticRegression (52.49%)
- **Why zero-shot first:** cheapest possible signal per the roadmap — no Colab time, no
  fine-tuning setup, before deciding whether the foundation-model direction is worth
  pursuing at all.

---

## 4. Results

| Model | RMSE | Naive RMSE (310.06) | Beats naive? | Directional Acc. | vs naive (49.58%) | vs LogReg (52.49%) |
|---|---|---|---|---|---|---|
| Chronos-2 | 322.65 | 310.06 | [NO] | 49.60% | ~flat | [NO] |
| TimesFM 2.5 | 314.65 | 310.06 | [NO] | 48.60% | [NO] (below) | [NO] |
| Moirai 2.0 | 316.35 | 310.06 | [NO] | 49.20% | [NO] (below) | [NO] |

**Every model lost on RMSE. Every model clustered at ~49% directional accuracy —
statistically indistinguishable from a coin flip, none within reach of the 52.49%
classical baseline.**

---

## 5. Root-cause analysis

1. **Consistent failure across three independent architectures.** Chronos-2 (encoder-decoder,
   pretrained on Chronos mixup + real-world data), TimesFM 2.5 (decoder-only, GiftEval +
   Wikipedia/Trends pretraining), and Moirai 2.0 (decoder-only, LOTSA + GIFT-Eval pretraining)
   share no architecture and no pretraining corpus overlap beyond general time-series data —
   yet all three land in the same narrow failure band. This is a strong signal the problem is
   the *task*, not any one model's design.
2. **Domain mismatch.** All three models are pretrained predominantly on smooth,
   structurally-patterned series — electricity load, traffic, web traffic, weather,
   operational/business metrics. BTC/USDT hourly closes behave close to a random walk with
   thin, noisy signal — a domain these models were not built to specialize in, and zero-shot
   inference gives them no opportunity to adapt.
3. **Same ceiling as Historical Study.** Classical models (LogReg/SVC/XGBoost) topped out at ~52.5%
   accuracy on this feature set; TSFMs land at ~49% predicting from raw price alone (no
   engineered features at all). Neither approach is breaking meaningfully past chance —
   consistent with a data/signal ceiling rather than a model-capability ceiling.

---

## 6. What this rules out (and doesn't)

- [NO] **Ruled out:** Zero-shot Chronos-2, TimesFM 2.5, and Moirai 2.0 do not provide a usable
  directional or price-level edge on BTC/USDT 1H data out of the box.
- [NO] **Ruled out (by decision, not by test):** Fine-tuning any of the three. Zero-shot
  failure was not "close but short" — it was a clean loss on both RMSE and directional
  accuracy across all three models, which is a weak prior for fine-tuning rescuing
  performance. Colab time was not spent chasing this.
- [NOTE] **Not ruled out:** A TSFM fed *engineered* features (via covariate/XReg support) rather
  than raw close price alone; a fundamentally different framing (e.g., return-based instead
  of price-based conditioning); or PatchTST built and trained from scratch on this specific
  dataset (Historical Study) — none of these were tested here.

---

## 7. Recommendation

**Historical Study is closed as a negative result.** Per `research benchmark protocol`'s own decision criteria,
this is exactly the outcome that says "don't sink further time into foundation models
before checking other levers." Do not test additional TSFM candidates — three independent
architectures failing identically is a strong enough pattern that a fourth model has low
expected information value, and continuing to search for one that "works" repeats Historical Study's
multiple-comparisons mistake.

**Suggested next step:** Historical Study (multi-asset support) or Historical Study (external data
agents — Fear & Greed, sentiment, on-chain). Historical Study is arguably the more promising lever
left untested — both the classical models and the TSFMs were ceiling-limited using only
price/volume-derived signal; external data is the one input category neither approach has
had access to yet.

**Process carried forward from Historical Study (still applies):**
- Any future model/config must be validated on multiple independent time windows before
  being treated as a result.
- Any sweep across configs must apply a multiple-comparisons correction.
- Always compare against Buy & Hold and the naive baseline, not either alone.

---

## 8. Environment notes

- All three TSFM packages (`chronos-forecasting`, `timesfm`, `uni2ts`/`gluonts`) were
  installed for this evaluation and have been **uninstalled** — no longer needed with Historical Study
  closed.
- `timesfm` required `torch>=2.4`; local env was upgraded from the CUDA 12.1-pinned
  `2.3.1+cu121` build to a compatible version for zero-shot CPU inference.
- Post-uninstall, a `protobuf` version conflict surfaced (`uni2ts`/transformers pulled in
  `protobuf 7.x`, incompatible with `streamlit` and `google-generativeai`). Resolved by
  pinning `protobuf==4.25.9`. A separate `tensorboard` warning at that point is a harmless
  transitive dependency notice — `tensorboard` is not used anywhere in this codebase.

---

## 9. Artifacts produced during this evaluation

- `zero_shot_chronos.py`, `zero_shot_timesfm.py`, `zero_shot_moirai.py` — eval scripts
  (kept for reference; can be re-run if revisiting TSFMs later, e.g. with covariate support)
- `data/results/zero_shot_chronos_results.csv`
- `data/results/zero_shot_timesfm_results.csv`
- `data/results/zero_shot_moirai_results.csv`