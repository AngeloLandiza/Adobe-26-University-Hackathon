# Attention research: learning how ad elements relate to each other

**Status:** started 2026-10-20. Nothing built yet; this doc defines the question, the model and how it plugs into the other distillation work.
**Tracks:** GitHub issue [#1 "Research Spike: Attention mechanism for elements"](https://github.com/AngeloLandiza/Adobe-26-University-Hackathon/issues/1).
**Part of:** the distillation research in [`../superpowers/plans/2026-10-20-distillation.md`](../superpowers/plans/2026-10-20-distillation.md). Data lives in [`../../training-data/`](../../training-data/).

---

## 1. The question

> Does a model that looks at **how the pieces of an ad relate to each other** judge and plan layouts better than one that only looks at whole-layout summary scores?

Today the engine's grader (node 3) scores a layout with 12 summary numbers, such as "how readable is the text overall", "how much accidental overlap" and "how aligned are things". The first distillation student (D) only learns how much each of those 12 numbers should count. That misses things a designer cares about that only show up **between specific pieces**:

- the logo sitting next to the call-to-action button, rather than stranded in the opposite corner;
- a "NEW" badge staying attached to the product it labels;
- the headline sitting directly above the product it describes;
- body text ending up on the busy part of a background photo, while the calm part sits empty;
- a group of three badges staying in one row instead of being scattered.

An **attention mechanism** is built to learn exactly this.

## 2. What "attention" means here, in plain words

- **Token:** one piece of the ad turned into a list of numbers (its role, where it is, how big, what colour, and so on). An ad with 11 pieces becomes 11 tokens.
- **Attention:** each token builds its own updated description by *looking at every other token* and deciding which ones matter to it. The logo token can learn to "pay attention" to the CTA token, and the badge token to the product token. The model learns these weights from examples; nobody hand-writes "logo near CTA is good".
- **Transformer encoder:** a few attention layers stacked. After them, each token's description includes its context ("I'm a logo, and I'm right next to the button").
- **Order doesn't matter:** an ad's pieces have no natural order, so the model treats them as a *set*. Shuffling the pieces must not change the answer.

## 3. What the model should do

### Input: one layout = one set of tokens

One token per **unit** (the engine's grouping of elements that move together, from `engine/pipeline/priority/units.py`), plus one token describing the canvas.

| Token part | From | Notes |
|---|---|---|
| role (headline, logo, cta, hero_image, badge, legal, …) | node 1's `PriorityPlan` | one-hot or learned embedding |
| importance rank | `PriorityPlan` (`plan.weight(unit_id)`) | so "the headline" and "the 5th badge" differ |
| box `[x, y, w, h]` as fractions of the canvas, centre, area share, aspect | the layout being judged | the core of "where things are relative to each other" |
| type mix (text / image / shape), kind (single / button / group) | `unit_features()` in `engine/pipeline/priority/features.py` | |
| text: font size relative to canvas, line count, length, bold, all-caps | features + `TextMeasurer` | |
| colour: text colour and luminance; backdrop luminance and busyness under the box | the render's pixel metrics (`grading/pixels.py`) | lets it learn "text on busy image" |
| z-order, opacity, rotation | features | |
| **canvas token:** target width, height, aspect class, plus the same ad's **source** layout summary | target + source | so it can judge "faithful to the original" |

The source design can also go in as a second set of tokens (the same pieces at their original positions). Then the model can compare original and resized directly. This is a design choice to test (§7).

### Output heads (one shared encoder, several small heads)

| Head | Output | Used by | Priority |
|---|---|---|---|
| **H1 layout quality** | one score per layout; trained so that score(A) > score(B) when the teacher preferred A | the grader (student D) | first |
| **H2 role and importance** | per token: role probabilities and an importance score, computed on the **source** ad | the ranker (student A) | second |
| **H3 floor-plan fit** | per (ad, target, floor plan): how well that floor plan suits this ad | floor plans / recommender (students B2, C) | third |
| **H4 relation probe** (optional) | for a pair of tokens: "should these be close / aligned / stacked?" | explaining decisions; checking relation features | research only |

## 4. How it supports each distillation model

```
                ┌────────────── attention encoder (shared) ──────────────┐
 ad + layout ──►│ tokens → attention layers → per-token + whole-layout   │
                └───────┬───────────────┬───────────────┬───────────────┘
                        │               │               │
                H1 quality         H2 roles/rank    H3 floor-plan fit
                        │               │               │
   student D: grader ◄──┘   student A ◄─┘   students B2/C ◄┘
   (blend with or replace   (context-aware  (pick or suggest floor plans
    the 12-weight formula)   labels offline)  per ad, not one-size-fits-all)
```

- **D, the grader.** H1 gives a quality score that sees relationships. Use it in one of two ways:
  1. **blend:** `final = linear_score + λ × attention_score`, with λ tuned on holdout ads;
  2. **re-rank:** let the linear grader keep its top 6 candidates (it's cheap) and let H1 choose among them.

  Either way, the hard gates G1–G9 still throw out rule-breaking layouts first, so the model can never cause a disqualification.
- **A, the ranker.** Deciding what a piece *is* depends on context: "the biggest text is the headline" is only true relative to the other texts. H2 sees every piece at once, which the rule-based ranker approximates with hand-written comparisons. It ships only if it beats the rules on holdout ads.
- **B2/C, floor plans.** H3 learns which floor plan suits which ad: "ads with a tall product and a long headline do best with `text_first`". The engine then tries the most promising plans first, or the recommender (C) suggests them per ad.
- **The simpler "relation features" step.** Before training attention, we add a few dozen hand-written pairwise scores (logo–CTA distance, headline-above-product, badge-touches-product, text-on-busy-area) to the linear grader. That is the baseline attention must beat. H4 and the attention maps also show *which* relations the model found, which may suggest better hand-written features.

## 5. Data: where the training examples come from

The model has thousands of numbers to learn, against the grader's 12, so it needs more examples than the teacher alone can afford. Two sources:

1. **Free pre-training pairs: "original vs broken".** Every ad in `training-data/corpus/` (153 so far) is a designed layout. Make a worse copy by **corrupting** it:
   - move a piece to a random spot;
   - make two pieces overlap;
   - shrink the headline below the badge;
   - put the body text on the busy part of an image;
   - separate a badge from its product;
   - scatter a row of badges;
   - push the CTA to the far corner.

   Train H1 so that the original scores higher than its corrupted copies. That gives thousands of pairs with no teacher calls, and each corruption is labelled with what it broke, which helps H4. Also do the same on the engine's own candidates. Planned folder: `training-data/corruptions/`, with a README like the others.
2. **Fine-tuning on the teacher's judgements.** `training-data/judgments/judgments.jsonl` holds the pairwise "A beats B, because …" from Claude Opus 5.5 against the official rubric. Fine-tune H1 on these after pre-training. They are the closest thing we have to the official AI judge.
3. **Later, if needed:** the Crello dataset (real designs, kept local and never committed; see the note in the distillation plan), and teacher role labels (`training-data/role-labels/`, for H2).

**Splits:** the same train/holdout split as the corpus. A corrupted or augmented copy of a holdout ad stays holdout. Accuracy is always reported separately for Solstice and off-brand holdout ads.

## 6. Constraints the design must respect

These come from the challenge rules and the engine's design:

- **Offline at scoring.** The scored run is `docker run --network none`, so the model must run with no internet and no API.
- **CPU, small and fast.** The scoring machine is an Apple M5 Pro, with an amd64 image (possibly emulated). Target: under 1 M parameters, a few milliseconds per layout, and a weights file of a few MB.
- **No heavy dependency in the engine.** Train in PyTorch on a laptop (dev only). Ship the weights as a plain `.npz` file, with a forward pass written in numpy, which the engine already depends on. Load it through the same safe data-file helper as `weights.json` (`engine/pipeline/datafile.py`): a missing or broken file means today's behaviour.
- **Weights must be public** if the engine relies on them (RULES "Models and fine-tuning"). A from-scratch model's weights live in this public repo, which satisfies that.
- **Deterministic:** fixed seeds, and the same answer for the same input.
- **No generated imagery.** The model only scores and plans; it never draws anything.
- **Not tuned to the samples.** The brief: "anything tuned to one design or one size will fail the hidden and unseen cases". Hence holdout ads, off-brand ads and corruption-based pre-training.

## 7. Experiments, in order

Each step has a go/no-go based on **holdout pairwise accuracy**: on held-back ads, how often the model prefers the layout the teacher preferred. Log every run in [`experiments/`](experiments/).

| # | Experiment | Compare against | Go/no-go |
|---|---|---|---|
| E0 | **Baselines:** the hand-set grader, and student D (12 learned weights) | — | numbers recorded |
| E1 | **Relation features:** D + ~30 hand-written pairwise features (still a linear model) | E0 | if E1 already closes most of the gap, attention may not be worth it |
| E2 | **Pre-train H1 on original-vs-corrupted pairs only** | E0, E1 | does it beat chance on teacher judgements *without* seeing any? (a sanity check that corruptions teach real design sense) |
| E3 | **Fine-tune H1 on teacher judgements** | E1 | must beat E1 on holdout by a clear margin (≥ 3 points), on Solstice *and* off-brand |
| E4 | **Integrate:** blend vs re-rank inside `pick_best`; run the 13-size matrix and the Docker check | E3 offline numbers | winners look better on the contact sheets; all gates pass; latency acceptable |
| E5 | H2 (roles/rank) and H3 (floor-plan fit) on the shared encoder | the rule-based ranker; B2 results | ship each head only if it beats what it replaces |

Ablations to run along the way:
- with and without source tokens;
- with and without pixel features (colour and busyness);
- 1 vs 2 vs 4 attention layers;
- the effect of each corruption type.

## 8. Planned layout of the work

```
distil-research/
  docs/attention/
    README.md            this file
    experiments/         one markdown log per experiment run (template inside)
  chameleon/dev/attention/     (to create when E2 starts)
    tokens.py            layout → tokens (shared by training and the engine)
    corrupt.py           original → corrupted copies, labelled by what was broken
    model.py             the PyTorch model (dev only)
    train.py             pre-train + fine-tune; writes the .npz weights
    export.py            PyTorch → numpy weights; checks both give the same scores
  chameleon/engine/pipeline/grading/attention.py   (E4) numpy forward pass + safe loader
  training-data/corruptions/   (E2) generated pairs + README
```

## 9. Open questions

1. Unit tokens vs element tokens: units match how the engine moves things, elements are finer. Start with units.
2. How much does the source-layout context help "faithfulness"? Ablate it in E3.
3. Can H1 replace the 12-score formula entirely, or only re-rank its top candidates? Re-ranking is safer; decide in E4.
4. Is the teacher's taste stable enough to fine-tune on? It depends on the pilot's agreement numbers (`training-data/judgments/pilot/`).
5. The model-size limit and approved-model list in the brief ("being set with the team") are for *locally hosted models*. A small from-scratch scorer is probably fine, but confirm with the organisers before shipping it in the real engine.
