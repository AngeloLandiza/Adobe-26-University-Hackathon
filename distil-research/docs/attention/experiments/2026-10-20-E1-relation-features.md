# E1: relation features on top of the 12 metrics (still a linear model)

**Date / who:** 2026-10-20, Claude (Opus 5.5) for Angelo
**Commit:** `0a46ac9` plus uncommitted research files: `dev/attention/{relations,features_dataset}.py` and `dev/distill/train_grader.py --extra-features`
**Question:** does giving the linear grader hand-written relation features, which measure how specific pieces relate (logo to CTA, headline to product, reading order kept, and so on), predict the teacher's pairwise judgements on held-back ads better than the 12 summary metrics alone?

## Setup
- **Data:** the same as E0. 715 judgements (rubric `f80dbb11`, teacher `claude-code:claude-opus-5-5`); confidence ≥ 2 leaves **326 train / 135 holdout pairs** (83 Solstice, 52 off-brand). The relation features are in `training-data/attention/relation_features.jsonl`: 882 judged candidates over 149 cells, rebuilt from the corpus exactly as the pool builds them. All were found, and the rebuilt geometry metrics match the pool to 0.0, so the features describe the very layouts the teacher saw.
- **Features:** 29 in all, defined in `dev/attention/relations.py`. Six are `has_<role>` flags, which are always equal within a pair and so carry no signal. The other 23 are relations, each in [0, 1]: logo–CTA gap, headline above the hero, CTA below the headline, badge–hero gap, badge spread, reading-order agreement with the source (Kendall tau), headline/body font ratio, CTA/headline size, largest empty rectangle, minimum edge margin, text-on-image overlap and its gain over the source, nearest-neighbour alignment, vertical rhythm (CV of gaps), hero/logo/headline area shares, legal font vs floor, share of texts below their floor, the uniform-shrink signature, neighbours kept together, headline scale vs the source, and logo-to-corner distance.
- **Model:** a plain L2-regularised logistic regression on feature differences: P(A beats B) = sigmoid(Σ w_k (f_k(A) − f_k(B)) / s_k). Here s_k is the RMS of feature k's difference over the train pairs, so the weights are comparable. There's no intercept, so swapping A and B flips the answer. Weights are signed, so "less is better" features work. It's fitted by Newton's method. The penalty is chosen from {0.001 … 3} by 5-fold cross-validation (CV) on **train only**, with whole ads per fold. Variants: 12 metrics only, relations only, and both together.
- **Command** (from `distil-research/chameleon/`):
  `../../chameleon/.venv/bin/python -m dev.attention.features_dataset --workers 4`, then
  `../../chameleon/.venv/bin/python -m dev.distill.train_grader --min-confidence 2 --extra-features ../training-data/attention/relation_features.jsonl --report <scratch>/report.md`

## Results
| Model | Holdout acc (all, 135) | Solstice holdout (83) | Off-brand holdout (52) | Train acc (326) | Train CV by ad |
|---|---|---|---|---|---|
| E0 hand-set | 70.4% | 73.5% | 65.4% | 63.5% | – |
| **E0 learned 12 (the bar)** | **77.8%** | 75.9% | 80.8% | 68.4% | – |
| logistic 12 (control, l2 0.1) | 77.0% | 74.7% | 80.8% | 72.7% | 69.0% |
| relations only (l2 0.003) | 69.6% | 71.1% | 67.3% | 77.9% | 68.7% |
| **E1: 12 + relations (l2 1.0)** | **77.0%** | 77.1% | 76.9% | 75.8% | 67.2% |

The paired holdout comparison counts the pairs only E1 gets right against the pairs only the other model gets right. The test is an exact McNemar test, which checks whether that split is more lopsided than chance.
- vs learned 12: 7 vs 8 pairs, p = 1.0
- vs logistic 12: 10 vs 10, p = 1.0
- vs hand-set: 18 vs 9, p = 0.12

**Sensitivity check** (same code, different confidence cut). With confidence ≥ 1 (466 / 210 pairs), learned 12 scores 69.5% and E1 66.7%. With confidence = 3 only (136 / 55 pairs), learned 12 scores 74.5% and E1 72.7%. E1 never wins.

**Largest |weight| in E1** (standardised; + means more is better): faithfulness +0.108, **reading_order +0.096**, degradation +0.059, legibility +0.058, **logo_cta_gap +0.056**, **empty_rect_share −0.056**, **below_floor_share −0.053**, **rhythm_cv −0.053**, **hero_area_share +0.052**, busyness +0.051, **uniform_scale +0.051**, spacing +0.049, **cta_below_headline +0.046**, **logo_area_share +0.043**, **nn_axis_share +0.037**. Relation features are in bold.

## What we learned
- **Relation features add nothing measurable.** E1 scores 77.0% on holdout against the bar's 77.8%, and the paired comparison is 7 vs 8 pairs. Cross-validation on train is worse with them (67.2% vs 69.0%): 23 more inputs on 326 pairs mostly add variance. E1 gains on Solstice (+1.2 points) and loses on off-brand (−3.9), and both shifts are inside the noise (one standard error is about 4.5 points on Solstice and 5.8 on off-brand).
- **The informative relations duplicate existing metrics.** Correlation of pair differences over all 461 pairs:
  - `reading_order` vs `faithfulness`: r = 0.96
  - `below_floor_share` vs `degradation`: r = −0.96
  - `empty_rect_share` vs `space_use`: r = −0.72
  - `headline_scale_vs_source` vs `legibility`: r = 0.73

  On their own, `reading_order` and `below_floor_share` are among the best single predictors (66% and 66% of train pairs where they differ; the teacher prefers fewer texts below their floor 84% of the time on holdout). But the metrics already carry that signal.
- **The "designer intuition" relations are near chance on this data.** Single-feature agreement on train, for the pairs where the feature differs:
  - headline above the hero: 51%
  - text-on-image overlap: 48%
  - neighbours kept together: 52%
  - badge–hero gap: 46%
  - logo-to-corner distance: 50%

  `logo_cta_gap` even points the "wrong" way: the teacher prefers the layout with the logo *farther* from the CTA 62% of the time. That is probably because it rewards keeping the source's arrangement (logo in its corner), which `faithfulness` already tracks. A likely reason the intuitive relations don't show up: the engine's candidates within one cell rarely differ on them. Most judged pairs are two variants of the same template, not "badge attached" vs "badge scattered".
- **The limit looks like the labels and the candidates, not the representation.** The ceiling is the teacher's 86% swap-consistency. The holdout is 135 pairs, so ±3.6 points is one standard error. And the candidates are rarely shippable (judgements README). A linear model with more hand features can't exploit what isn't in the pairs.
- **Implication for attention:** E1 did *not* close the gap, so the README's "attention may not be worth it" exit does not trigger. But this run gives no evidence that relational information is what's missing. On these labels, a richer pairwise representation didn't help a linear model at all.

## Decision
**No-go for relation features as a grader upgrade:** E1 does not beat E0 (77.0% vs 77.8% holdout; −0.8 points; 7 vs 8 paired). Nothing is written to the engine.
**For the attention track, conditional go, with the bar unchanged:**
- **E3 must reach ≥ 81% on holdout** (the bar 77.8% + 3 points) on Solstice and off-brand alike.
- **Run E2 (corruption pre-training, no labels needed) first.** It is the only way to show the model relational contrasts the judged pairs don't contain.
- **Grow the judged set before E3,** especially the holdout (aim for ≥ 300 holdout pairs) and pairs from different floor plans (B2), so a 3-point gain is measurable.

Keep `relations.py` as a probe for H4 and as the source of corruption labels.
