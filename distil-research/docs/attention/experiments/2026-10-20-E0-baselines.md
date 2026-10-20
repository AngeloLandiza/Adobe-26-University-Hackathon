# E0: baselines, the hand-set grader and the learned 12 weights

**Date / who:** 2026-10-20, Claude (Opus 5.5) for Angelo
**Commit:** `0a46ac9` plus uncommitted research files (`dev/distill/train_grader.py` gained `--extra-features`; default behaviour unchanged)
**Question:** on held-back ads, how often does today's grader pick the layout the teacher preferred, with the hand-set weights and with weights learned from the teacher?

## Setup
- **Data:** the corpus has 153 ads (train/holdout split by ad, each ad tagged Solstice or off-brand). `training-data/judgments/judgments.jsonl` holds 715 judgements by `claude-code:claude-opus-5-5` under rubric `f80dbb11`. Confidence ≥ 2 keeps 488 of them. Dropping 27 ties leaves **326 train pairs and 135 holdout pairs** (83 Solstice, 52 off-brand). No corruptions.
- **Models:**
  - **hand-set:** the grader as shipped, `DEFAULT_WEIGHTS` over its 12 summary metrics (9 geometry, 3 pixel).
  - **learned:** student D, a Bradley–Terry fit of the same 12 weights. Bradley–Terry means P(A beats B) = sigmoid(w · (metrics_A − metrics_B)). The weights are kept non-negative, no intercept.
  - **logistic 12** (a control for E1): the same 12 metrics in a plain L2-regularised logistic regression. Its weights may be negative, and the penalty strength is picked by 5-fold cross-validation on train, grouping folds by ad.
- **Training:** projected Nesterov gradient descent, 3000 steps, l2 = 0.01, seed 0 (deterministic).
- **Command** (from `distil-research/chameleon/`): `../../chameleon/.venv/bin/python -m dev.distill.train_grader --min-confidence 2 --report <scratch>/report.md`. The run wrote its report to a scratch path so `training-data/judgments/training-report.*` stays the committed one. The numbers are identical to it.

## Results
"Pairwise accuracy" is the share of judged pairs where the model scores the teacher's winner higher. A score tie counts as a miss. Chance is 50%.

| Model | Holdout acc (all, 135) | Solstice holdout (83) | Off-brand holdout (52) | Train acc (326) |
|---|---|---|---|---|
| hand-set (`DEFAULT_WEIGHTS`) | 70.4% | 73.5% | 65.4% | 63.5% |
| **learned 12 weights (student D)** | **77.8%** | 75.9% | 80.8% | 68.4% |
| logistic 12, signed weights (l2 = 0.1) | 77.0% | 74.7% | 80.8% | 72.7% (CV by ad 69.0%) |

The learned weights put most of their mass on faithfulness (0.33), busyness (0.53 of the pixel group), degradation and legibility. Hierarchy and overlap get 0. The full table is in `training-data/judgments/training-report.md`.

## What we learned
- **The bar is 77.8% on holdout,** from 12 learned non-negative weights. It's 75.9% on Solstice and 80.8% on off-brand. Allowing negative weights (logistic 12) doesn't help: 77.0%.
- **Noise floor:** with 135 holdout pairs, one standard error is about 3.6 points. Differences under about 5 points between models are not distinguishable on this holdout.
- **Ceiling:** the teacher gives the same winner when A and B are swapped only 86% of the time, so about 86% is a practical upper bound for any student on these labels. The gap between the bar and that ceiling is about 8 points.
- Train accuracy is *lower* than holdout. That suggests the holdout pairs are easier (more lopsided contests), not that the model under-fits. Compare models on the same pairs only.

## Decision
**Numbers recorded** (E0 has no go/no-go). The baseline for E1–E3 is the learned 12-weight grader: **77.8% all / 75.9% Solstice / 80.8% off-brand** on holdout.
