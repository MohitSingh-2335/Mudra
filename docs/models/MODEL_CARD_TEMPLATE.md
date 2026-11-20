# Model Card Template (Mudra Standard Specification)

**Model Identifier:** `[model_id_vX.Y.Z]` (e.g. `lgbm_v3.0.0`)  
**Model Family:** `[LightGBM / XGBoost / CatBoost / PyTorch]`  
**Governance Status:** `[PLANNED / CANDIDATE / ACTIVE / ARCHIVED / REJECTED]`  
**Artifact Manifest:** `artifacts/models/[model_manifest_filename].json`  
**Development Branch:** `[git_branch_name]`  
**Last Updated:** `YYYY-MM-DD`  

---

## 1. Hypothesis & Delta from Previous Model

- **Targeted Weakness of Previous Model:**  
  *State the exact empirical or architectural weakness in the prior model that this version was designed to resolve (e.g. residual non-stationary volume features in Model 2, extrapolation limits, or uncalibrated probabilities).*
- **Core Hypothesis Tested:**  
  *Formulate the precise quantitative hypothesis tested in this version. What specific change in target framing, feature representation, algorithm family, or loss function is expected to produce measurable improvement?*
- **Key Variables Changed:**  
  - *Features Added:* List exact feature names and mathematical definitions.
  - *Features Removed:* List exact dropped features and the reason for removal.
  - *Hyperparameters or Loss Function Modifications:* Specify loss function changes (e.g. focal loss, Huber loss) or structural parameter adjustments.

---

## 2. Data & Feature Schema

- **Dataset Source:** `[e.g. Binance Spot BTCUSDT 1-Hour Confirmed Candles]`
- **Dataset Range:** `[Start Date UTC ===> End Date UTC]`, Total Rows: `[N rows]`
- **Chronological Split Boundary:**
  - **Train Set (80% Chronological):** `[Start UTC ===> End UTC]`, `[N rows]`
  - **Purge Buffer (Causality Guard):** `[1 Hour / N Hours UTC]` strictly excluded to prevent label leakage.
  - **Calibration Set (Optional, if post-hoc calibration applied):** `[Start UTC ===> End UTC]`, `[N rows]`
  - **Holdout Test Set (20% Out-of-Sample):** `[Start UTC ===> End UTC]`, `[N rows]`
- **Feature Schema Version:** `[e.g. v3_stationary]`
- **Complete Feature List:**  
  *Enumerate every feature included in the feature vector. Document whether each feature is bounded, stationary, or cyclical.*

---

## 3. Architecture & Artifact Governance

- **Algorithms:** `[e.g. LightGBM Classifier (LGBMClassifier) & LightGBM Regressor (LGBMRegressor)]`
- **Model Checkpoint Artifacts & Digests:**
  - Classifier Model: `artifacts/models/[clf_filename].pkl`  
    `SHA-256: [64-character hex hash]`
  - Regressor Model: `artifacts/models/[reg_filename].pkl`  
    `SHA-256: [64-character hex hash]`
- **Trained Hyperparameters:**
  - *Classifier:* `objective`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `colsample_bytree`, etc.
  - *Regressor:* `objective`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, etc.
- **Inference Runtime & Security:**
  - In-memory inference framework and hardware target (CPU/GPU).
  - Runtime per-row latency SLA (must be $< 50$ ms).
  - Cryptographic verification function reference in `src/model_loader.py`.

---

## 4. Standardized Evaluation Scorecard

All metrics evaluated against the locked out-of-sample holdout test partition (6,998 hours):

| Metric | Candidate Value | Active Model Baseline | Naive Baseline | Status / Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Directional Accuracy** | `[XX.XX%]` | 53.26% | 50.39% | Must beat active model to pass Gate 1. |
| **Log-Loss** | `[0.XXXX]` | 0.6914 | 0.6931 | Lower is better; cross-entropy of predicted probabilities. |
| **Brier Score** | `[0.XXXX]` | 0.2491 | 0.2500 | Lower is better; mean squared error of probability forecasts. |
| **ROC-AUC** | `[0.XXXX]` | 0.5394 | 0.5000 | Higher is better; ranking discrimination. |
| **Spearman Rank IC** | `[+0.XXXX]` | +0.0149 | 0.0000 | Rank correlation between predicted magnitude and realized return. |
| **Return MAE** | `[0.XXXXXX]` | 0.003221 | 0.003198 | Mean absolute error of fractional price return. |
| **USD Price RMSE** | `[$XXX.XX]` | $399.47 | $398.77 | Root mean squared error in Dollar price space. |
| **Walk-Forward Folds (5-Fold)** | `[Mean: XX.XX%]` | 53.17% | 49.58% | Cross-validation stability across independent non-overlapping folds. |

---

## 5. Strengths & What Worked

1. **[Strength 1]:** *Document empirical successes observed during testing or backtesting.*
2. **[Strength 2]:** *Document improvements in latency, stability, calibration, or feature relevance.*
3. **[Strength 3]:** *Document behavior under live shadow testing conditions.*

---

## 6. Weaknesses & Honest Post-Mortem Findings

1. **[Weakness / Failure 1]:** *Document honest negative findings, distribution shifts, or failure modes.*
2. **[Weakness / Failure 2]:** *Document any feature bugs, data issues, or trade friction realities.*
3. **[Weakness / Failure 3]:** *Document how the model performed after 0.14% exchange fee deduction.*

---

## 7. Deployment Decision & Handoff to Next Model

- **Deployment Decision:** `[ACTIVE / SHADOW CANDIDATE / REJECTED / ARCHIVED]`  
  *Justify the decision using the 5 objective promotion gates from `docs/reports/model_evolution_ledger.md`.*
- **Handoff Hypothesis Motivating Next Version:**  
  *State the concrete hypothesis, architectural change, or feature fix that the next model iteration should tackle.*
