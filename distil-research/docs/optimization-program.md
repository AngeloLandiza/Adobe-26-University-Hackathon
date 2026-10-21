# Optimization program: what actually makes the shipped layouts better?

**Goal:** find the approach that most improves the layouts the engine *ships*, judged the same way every time. ML, other AI, or plain algorithms all qualify. Whatever wins, ships (first into the research engine, then to `chameleon/` through a reviewed PR).

**The yardstick:** the A/B harness (`chameleon/dev/distill/ab_engine.py`).
1. Run the whole engine twice on every **holdout** cell: the baseline, and the baseline plus one change behind an environment switch.
2. Render both winners.
3. Have the teacher (Claude Opus 5.5, harsh senior-designer rubric quoting the official criteria) judge every cell where they differ.
4. Report wins and losses overall and by brand, mean ship-readiness (1–5), runtime, and a sign-test p-value.

**The rule:** a change ships only if it wins clearly on holdout: more wins than losses, p < 0.05, no brand worse, and no new gate failures or crashes.

## What we know so far (2026-10-20)

| Approach | Kind | Holdout result | Status |
|---|---|---|---|
| Learned grader weights (D) | ML | final picks 48% vs hand-set on Solstice holdout | ❌ dropped |
| Relation features (E1) | ML | 77.0% vs 77.8% pairwise | ❌ dropped |
| Attention scorer (E2/E3) | ML | not run; needs ≥ 81% on both brands | ⏸ parked (picking isn't the bottleneck) |
| Hand-drawn floor plans (B1) | human design | each won ≥ 1 sample cell | ✅ in research engine |
| Teacher floor plans, clustered (B2) | AI-designed templates | all 14: 24–9; best 5: 20–3 (selected on holdout) | ✅ 5 in research engine |
| Per-ad floor-plan retrieval (C) | AI-designed + nearest neighbour | 29–11, 48 of 85 picks changed | 🔧 being wired into the engine (`CHAMELEON_RETRIEVAL=1`) |
| Content-aware slot sizing (FLEX) | algorithm (flexbox-style solver) | — | 🔧 being built (`CHAMELEON_FLEX=1`) |

**Why the focus moved from ML to candidates:** across 1,955 critiques, the flaws are mechanical:
- cramped next to empty: 77%
- text too small: 68%
- margins: 58%
- logo: 55%

Every candidate shares them, so a better *picker* can't help. Better *candidates* (B2, C) and better *space allocation* (FLEX) can.

## Queue of approaches to test with the harness

1. **FLEX** vs baseline.
2. **Retrieval** vs baseline.
3. **FLEX + retrieval** vs the better of the two.
4. **Fallback policy:** prefer the plain shrink when the best re-layout scores below a threshold. The teacher preferred the shrink 34–3 where the engine picked it.
5. **Local repair pass:** after picking, nudge sizes and positions to fix the top complaints (dead space, edge margins, tiny text) without breaking gates. Plain search, no ML.
6. **B2 round 2:** teacher floor plans for strip and wide shapes, then retrieval for them.
7. **Only if picking becomes the bottleneck again:** retrain the grader or the attention scorer on judgements of the *new*, better candidates.
8. **Stretch (finale):** a small local model distilled from the teacher's per-ad floor plans, if the rules' model-size limit allows it.

Results and reports appear on the local dashboard (`python -m dev.distill.serve_dashboard`, http://127.0.0.1:8765) under "Experiments & reports"; each A/B report lives in `training-data/ab/<A>__vs__<B>/REPORT.md`.

## Tournament leaderboard (final, 2026-10-21)

Each variant vs the baseline research engine. Teacher-judged on sampled changed cells (holdout ≤ 60, challenge ≤ 100; confidence ≥ 2). The challenge set holds 40 new ads (499 cells), never used for tuning or selection. "Worst slice" is the lowest win share over any brand or shape slice with ≥ 5 decided cells.

| Variant | Kind | Holdout (B–A) | Challenge (B–A) | Challenge win share | Worst slice (hold / chl) | Verdict |
|---|---|---|---|---|---|---|
| **bisect: priority bisection (S2)** | algorithm | **38–17 (p = 0.006)** | **57–37 (p = 0.049)** | 61% | 0.57 / 0.33 | 🏆 **ship**: significant on both sets, best on holdout, works for any shape without data |
| **retr: per-ad floor-plan retrieval** | AI-designed plans + nearest neighbour | 31–20 (p = 0.16) | **56–31 (p = 0.010)** | **64%** | 0.50 / 0.40 | 🥈 strongest on challenge; depends on library coverage |
| grid: responsive design grid (S4) | algorithm | 27–24 | 51–33 (p = 0.063) | 61% | 0.20 / 0.36 | loses strips |
| combo: bisect + retr | both | 19–11 vs bisect | 52–38 (p = 0.17) | 58% | 0.44 / 0.48 | ❌ no better than either alone: the grader picks badly among more diverse candidates |
| flex: content-aware slot sizing | algorithm | 24–29 | 32–47 | 41% | — | ❌ rejected |
| repair: mined rules + local repair (S3) | AI-mined rules + search | 16–19 | 18–34 (p = 0.036) | 35% | — | ❌ rejected |
| Learned grader weights (D) / relation features (E1) | ML | picks 48% (D2) | — | — | — | ❌ rejected earlier |
| structure_reflow (S1) | algorithm | — | — | — | — | ⏸ unfinished |

**Conclusions:**
1. **What generalises:** layouts that don't depend on fixed templates. Either an algorithm that sizes regions by content and importance for any aspect (bisection), or reuse of designer-quality per-ad plans (retrieval). Both beat the engine significantly on ads nothing was tuned on.
2. **Proxies lie:** content-aware sizing cut dead space by 38% and still lost. Only judged A/Bs decide.
3. **Picking matters again once candidates are diverse.** Combining strategies didn't add up, because the engine's grader chooses poorly among more diverse candidates. The S3 rulebook penalty predicted the teacher's choice 76% of the time vs the grader's 65%, which makes it the natural next picker (an ML-free one).

## Next (if continued)
1. **Ship bisection** into the research engine by default (`CHAMELEON_STRATEGIES=bisection`). Propose porting it to `chameleon/` through a reviewed PR.
2. **Better picker:** blend the rulebook penalty (`strategies/rules.py`) into `pick_best`. Then re-run combo (bisect + retr) on holdout, and confirm once on a *fresh* challenge set; the current one has now been used for the final comparison.
3. **S1 structure_reflow:** finish and judge (`.superpowers/sdd/2026-10-20-distillation/task-S1-brief.md`; partial code is in the working tree).

## Resume here (in order, all from `distil-research/chameleon/` with `PYTHONPATH=$PWD`)
1. **Finish round 2 judging** (resumes where it stopped). Each judge command takes `--shard K/4`; run shards 0–3 in parallel, then `--merge`, then `report`:
   - `python -m dev.distill.ab_engine judge --a base --b retr --max-cells 60`
   - same for `base_chl retr_chl` (`--max-cells 100`), `base repair` (60) and `base_chl repair_chl` (100).
2. **Finish bisection.**
   - `python -m dev.distill.ab_engine run --name bisect_chl --env CHAMELEON_STRATEGIES=bisection --split challenge --workers 4`, which resumes the last 26 cells.
   - Judge `base bisect` (60) and `base_chl bisect_chl` (100) as above.
3. **Combine.** Choose per-shape routing on **holdout only**: retrieval everywhere, plus the grid for tall / very tall / wide if it adds on top of retrieval. Run the combined variant, then confirm **once** on the challenge set.
4. **Keep the dashboard live** during runs: `python -m dev.distill.serve_dashboard` (local, http://127.0.0.1:8765).
5. **S1 (optional):** re-dispatch with `.superpowers/sdd/2026-10-20-distillation/task-S1-brief.md`; partial code exists.

## Rulebook picker test (2026-10-21)
`CHAMELEON_PICKER=rules` lets `pick_best` choose the gate-passing candidate with the lowest rulebook penalty. Combined with bisection and retrieval, it was checked on 8 cells (the 3 samples plus 4 challenge ads, `python -m dev.distill.compare_variants`) with a harsh-rubric critique.
- **Result:** the combined variant won 0, lost 3 and tied 4 against today's engine.
- **Why:** the rulebook punishes empty space and small text, so it favours layouts where everything got big (badges as large as the CTA, scattered pieces). That breaks hierarchy, which is the main thing the judge punishes.

**Decision:** the picker stays off. Bisection alone with the existing grader remains the shipped candidate. Sample: 8 cells, a critique rather than the 100-cell judged A/B.

## Bisection v2 (critique-driven) (2026-10-21)
Bisection wins most picks but its layouts still score about 2.3 out of 5 with the teacher. The teacher's written critiques of bisection layouts (158 judged holdout and challenge cells) name the same flaws again and again. v2 fixes the ones we could measure, inside the layout itself, behind a switch: `CHAMELEON_BISECT_V2=1`. With the switch off, the output is byte-identical to before; `tests/test_bisection_v2.py` checks this on the 3 samples.

**Step 1: measure before fixing.** `dev/distill/bisect_flaws.py` has a geometry detector for each flaw family, such as "the logo's size is under 6% of the canvas's long side". To test the detectors, the engine re-ran all 158 critiqued cells, and every rebuilt design matched the judged one (same `design_hash`). Keyword rules then split the cells by whether the teacher's paragraph about the bisection layout names the flaw. A detector is useful only if it fires much more often where the teacher complained.

| Flaw family | Critiques naming it | Detector fires there | Other critiques | Detector fires there | Kept? |
|---|---|---|---|---|---|
| Widow (headline's last line is one short word) | 16 | 50% | 142 | 2% | yes |
| Text too small (smallest text under 1.1% of the long side) | 75 | 56% | 83 | 17% | yes (measured, not fixed) |
| Squashed round shape or image (aspect off by >10%) | 5 | 60% | 153 | 0% | yes (few cases) |
| Legal blown up (wraps to 4+ lines, or as big as body text) | 15 | 33% | 143 | 6% | yes |
| Logo a speck (sqrt(area) < 6% of the long side) | 45 | 31% | 113 | 6% | yes |
| Mixed alignment in a stacked column | 20 | 55% | 138 | 33% | yes (weak) |
| Decoration orphaned from what it framed | 28 | 43% | 130 | 21% | yes |
| Content too close to the edge (< 5% of the short side) | 73 | 34% | 85 | 13% | yes |
| CTA competing with the headline (size ratio) | 19 | 16% | 139 | 12% | **no**: no separation |
| Logo too big | 24 | 17% | 134 | 16% | **no** |
| Floating, disconnected piece (gap to nearest neighbour) | 74 | 7% | 84 | 2% | **no** |

Most CTA complaints are about where the CTA sits (jammed, floating) or that it is too small, not oversized: 59 and 41 of 117, against 19. No CTA size measure separated them, so v2 adds no CTA cap. One plain margin finding: cells built with margin 0.04 drew edge complaints in 25 of 36 cases (69%), against 48 of 122 (39%) at margin 0.07.

**Step 2: the fixes** (all in `engine/pipeline/layout/strategies/bisection.py`, active only with the switch on):
- **Safe margin:** margins are 0.06 or 0.09 of the short side (was 0.04 or 0.07), and at least 3% of the width at the left and right ends.
- **Logo floor:** the logo's sqrt(area) is at least 6.5% of the long side. It may grow past 1.5x its source size to reach that.
- **Legal:** legal text asks for a cell wide enough to fit on 3 lines, or on as few lines as it needs across the canvas. A tall, narrow column counts as cramped for it.
- **Widow:** a multi-line text whose last line is one short word gets a balanced, narrower measure at the same size and line count. If that fails, it drops at most 12% in size.
- **Shared alignment:** units stacked in one column are shifted sideways, only inside their own cells, onto one axis (left, centre or right, following their text alignment).
- **Decorations kept whole:** a circle behind text keeps its proportions, so it is never squashed into an oval. A decoration that framed a product or text in the source follows it, at the same offset and scale. If it can't, it is parked only where no copy ends up half on it.

**Step 3: before and after** (detector fire rates on the final winning layout; train was used for development, and holdout and the critiqued cells only for measurement):

| Flaw | 60 train cells: v1 → v2 | 192 holdout cells: v1 → v2 | 158 critiqued cells: v1 → v2 |
|---|---|---|---|
| Edge crowding | 32% → 8% | 23% → 5% | 23% → 3% |
| Logo a speck | 13% → 0% | 9% → 1% | 13% → 1% |
| Decoration orphaned | 17% → 7% | 48% → 30% | 25% → 15% |
| Mixed alignment | 28% → 10% | 30% → 22% | 35% → 20% |
| Widow (headline) | 3% → 2% | 13% → 4% | 7% → 2% |
| Legal blown up | 8% → 8% | 12% → 7% | 9% → 8% |
| Squash | 7% → 2% | 0% → 0% | 2% → 0% |
| Text too small | 13% → 15% | 31% → 32% | 35% → 39% |

- **Gates:** 480 of 480 bisection variants pass the hard gates, both before and after (60 train cells x 8 variants).
- **Speed:** bisection takes 0.28 s per cell instead of 0.17 s. Engine time per cell is about the same (0.60 → 0.66 s on train).
- **Bisection's share of picks:** about the same (holdout 80% → 79%, challenge 74% → 75%).
- **Changed picks:** 165 of 192 on holdout differ from `ab/bisect`, and 394 of 499 on challenge differ from `ab/bisect_chl` (runs `ab/bisect2`, `ab/bisect2_chl`). Nothing has been judged yet.

**Honest visual verdict** (contact sheets of original | bisection | v2, harsh rubric, 16 train cells plus 8 holdout cells viewed only):
- **Better on roughly a third of the changed cells,** where a targeted flaw was visible: circles now sit behind the product instead of floating as fragments or behind the headline; logos are readable on strips; no squashed sticker; widows are gone; quotes wrap better; there is more breathing room at the edges.
- **About the same on most,** because the flaw wasn't there or the change is a few pixels.
- **Worse on a few (2 of 24 viewed):** the larger margins plus the grader's near-tie picks sometimes choose a variant with the product squeezed into a thin side column.
- **It doesn't fix bisection's biggest weaknesses:** headlines that end up small next to a big hero on tall formats, text too small on strips (slightly worse with the larger margins), and badges as big as the CTA.

v2 is a polish pass, not a step change. A teacher-judged A/B (`bisect` vs `bisect2`) is the only way to know whether it wins.

Commands: `python -m dev.distill.bisect_flaws validate | rates --name X --name Y | sheet --name X --name Y --out F.png | rebuild --name X --env ... --split judged|train|holdout|challenge`. Caches are in `training-data/ab/_flaws.nosync/` (git-ignored).

## Experiments from prior work (2026-10-21)
Three ideas from `docs/prior-work.md` (items 1, 2 and 4 of its experiment list), each behind its own switch, all on top of bisection v2. With a switch off, the output is byte-identical to before (`tests/test_prior_experiments.py` checks this on the 3 samples x 13 targets). Nothing here has been teacher-judged yet; these are CPU measurements only. Tools: `python -m dev.distill.prior_experiments rebuild | relval | nearfit | changed | summary`.

**1. Hierarchy lock (`CHAMELEON_HIER_LOCK=1`, in bisection).** Keep the source's text hierarchy. Example: if the CTA label was half the headline's size in the original, it may not become 0.9x the headline in the resize.
- Other text may keep at most its source size ratio to the headline (was 1.3x that), and at most 0.75x the placed headline (was 0.9x). Tighter caps by role: CTA label 0.6x, badge 0.5x, legal 0.45x.
- A CTA or badge pill grows at most 1.25x as much as the headline did (both measured against the source). Past that, the label sits in a compact pill.
- Readability floors still beat every cap. Our critiques complain about a CTA that is too small (41) or badly placed (59) more often than oversized (19), so the CTA is never shrunk under its 11 px floor to obey a cap.
- Where a cap would push a role under its floor (on strips the headline sits at its 14 px floor and the CTA at 11 px), the engine tries to raise the headline instead (to 11 / 0.6 = 18.3 px). It keeps that layout only if no other text loses legibility; otherwise it keeps the plain one. On crowded strips this rarely succeeds, so the lift seldom applies.

**2. Near-aspect source fit (`CHAMELEON_NEAR_FIT=1`, in `pick_best`).** The candidate already exists: the safe fallback is the whole original scaled uniformly and centred, with the background filling the new canvas (solid shapes stretched, cover images cropped). It always competes, but it rarely wins on the grader. So the switch changes the *policy*: when the source and target aspects are within 1.35x of each other (e.g. 1080x1080 → 1080x1350), the fallback wins unless the best re-layout beats its grader total by more than **0.10**.
- **Why 0.10:** in the train judgments, fallback-vs-layout pairs on near-aspect cells went 12–2 for the fallback. The two it lost were the only ones where the other layout led by 0.16. Every margin from 0.05 to 0.15 matches all 14 train verdicts; 0.10 is the middle.
- **How often it triggers:** 22 of 192 holdout cells are near-aspect. The policy changed the pick on 15 of them (fallback picked in 16, against 1 before).

**3. Relation preservation (`CHAMELEON_RELPRESERVE=1`, in `pick_best`; `engine/pipeline/grading/relations.py`).** A score for how many of the source's pairwise relations a candidate keeps, computed on the content units it keeps (backgrounds, decorations and demoted units are left out). Three kinds of relation:
- **Size order:** text against text by font size, image against image by area. Example: "the CTA is smaller than the headline".
- **Reading order:** "the logo comes before the headline". An above-relation may become a left-of relation (a stack turning into a row on a strip keeps the order).
- **Shared edges:** "these two texts share a left edge". If a stacked pair becomes side by side, left/centre/right maps to top/middle/bottom.

Every candidate's total loses w x (1 − score). w comes from `CHAMELEON_RELPRESERVE_W`; the default, 0.15, was chosen on train judgments only.

**Validation before tuning: does the score agree with the teacher?**

| Judged pairs | Grader total alone | Relation score alone | Grader − 0.15 x (1 − relation) |
|---|---|---|---|
| Train: 466 pairs of candidates within one cell (used to choose w) | 59.7% | 62.7% (of 370 non-tied) | **68.0%** |
| Judged A/B cells, base vs bisect (192 rows, holdout + challenge; check only) | 64.6% | **71.1%** (of 180 non-tied) | 67.7% |
| Same, confidence ≥ 2 (178 rows) | 64.6% | **73.1%** | 68.0% |

The relation score beats the grader in both sets. Blending it in adds about 3 to 8 points. This is the first hand-set term that predicts the teacher better than the grader on pairs it wasn't tuned on. On the A/B cells a larger w scored higher still (74% at w = 0.3–0.5), but w was fixed on train.

**Train measurements** (60 train cells, bisection + v2 baseline; detector rates from `bisect_flaws`):

| | v2 baseline | + hier lock | + near fit | + relation |
|---|---|---|---|---|
| Picks changed vs baseline | – | 24 | 3 | 20 |
| Bisection's share of picks | 75% | 70% | 70% | 57% |
| Fallback picked | 5 | 5 | 8 | 6 |
| Seconds per cell | 0.67 | 0.74 | 0.64 | 0.64 |
| Final designs failing gates | 0 | 0 | 0 | 0 |
| Bisection candidates passing gates | 177/177 | 178/178 | 177/177 | 177/177 |
| CTA competing (detector) | 38% | 38% | 38% | 40% |
| Edge crowding | 8% | 8% | 8% | 13% |
| Decoration orphaned | 7% | 8% | 7% | 13% |
| Mixed alignment | 10% | 10% | 10% | 15% |
| Squash | 2% | 2% | 2% | 5% |
| Legal blown up | 8% | 8% | 7% | 5% |

The CTA detector doesn't move under the lock. It fires mostly where the *source* already had a big CTA (CTA/headline ratio 0.5–0.6, or a wide pill), which the lock deliberately keeps, or where a template rather than bisection won. Where bisection won, the CTA's ratio to the headline dropped modestly (for example 0.45 → 0.35, 0.46 → 0.40).

**Holdout runs** (192 cells each, CPU only, compared with `ab/bisect2` by design hash; no gate failures in any run):

| Run | Picks that differ from bisect2 | Notes |
|---|---|---|
| `b2_hier` | **100 of 192** | 57 are the same bisection variant with re-sized text; 7 moved from bisection to a template. 0.75 s per cell (0.70 before). |
| `b2_nearfit` | **15 of 192** | All on near-aspect cells (15 of 22); 13 replace a bisection layout with the fallback. |
| `b2_rel` | **72 of 192** | 34 move from bisection to a template or the fallback. Fallback picked in 15 cells (3 before), 12 of them *not* near-aspect. |

**Honest visual verdict** (contact sheets of original | v2 | experiment, 12 changed train cells each, harsh rubric, viewed only):
- **Hierarchy lock:** mostly subtle. The headline is a bit bigger and the CTA pill and subcopy a bit smaller (text-only half page, a large leaderboard, a wide skyscraper); about 3 of 12 look better, about 8 the same. One is clearly worse: a leaderboard flipped to a row template whose CTA pill now competes with a one-line headline. It does nothing for the crowded strips, where the floors bind and the lift seldom fits.
- **Near fit:** 3 changed cells. In 2 of them the fallback is clearly closer to a designer's version: a faithful, clean, smaller copy instead of a cramped re-layout with the steps out of order. The third is a toss-up: the shrink leaves the headline and legal small on a half page. This is consistent with the teacher's earlier 34–3 for the shrink where it was picked.
- **Relation preservation:** mixed. About 3 of 12 are better: a more faithful reading order, and a reflow that keeps the source arrangement of a cover. About 4 are worse, because the penalty pulls picks from bisection back to templates, which bring their own flaws: a giant squashed oval decoration in two portrait cells, and a headline squeezed into a narrow column. The rest are about the same. The detector rates above show the same thing: edge, orphan, alignment and squash all rise.

**What to judge next (in order):** `bisect2` vs `b2_nearfit` (15 cells, cheap, most likely to win); `bisect2` vs `b2_rel` (72 cells; the validation is the strongest evidence of the three, but the visual check and the detectors are mixed); `bisect2` vs `b2_hier` (100 cells, mostly small text-size changes, so expect many ties).

## Layer preservation (real ads) (2026-10-21)
On the 50 real Crello ads (`challenge2`), bisection v2 scores about 1.85 out of 5 against about 2.5 on our own ads. The teacher's critiques keep naming the same thing: **real ads are layered, and the engine treats every element as a separate piece.** It pulls copy out of the card, band, label or heart it sat in and leaves the empty shape stranded. It shrinks a full-bleed photo or pattern into a small patch. It places text on busy parts of a photo it wasn't on. `CHAMELEON_LAYERS=1` makes the engine keep those layers. It is off by default; with it off, the output is byte-identical (`tests/test_layers.py` checks golden hashes on the 3 samples).

**Words used here.**
- A **container** is a shape or image whose box holds one or more pieces of content, with the content drawn on top of it. Examples: a white card with a headline and a button on it; a red band with one line of copy; a circle with a price inside.
- The **background layer** is the stack of full-bleed elements at the bottom of the ad. For example: a flat colour, with a cover photo over it, with a tint over that.

**What it does** (`engine/pipeline/layout/layer_groups.py`, with small hooks in bisection, `priority/units.py` and `grading/gates.py`):
- **Background:** every element of the bottom full-bleed stack is background, not just the lowest one. Before, a full-canvas photo over a colour field became the "hero image", and the layout shrank it into a cell. Now every strategy re-fits it full-bleed. A `cover` photo crops to the new canvas. A `contain` raster (a pattern, or a photo the converter didn't mark as cover) must keep its proportions by the rules, so it is fitted as large as the canvas allows.
- **Containers move as one unit.** Bisection places the container and everything on it in one cell, then lays the contents out *inside* the container, using the source's padding scaled with the text.
  - A rectangle (or a cover image) stretches to wrap its contents. If it ran off a canvas edge in the source, it is extended to that edge again, so a band stays a band.
  - An image or a round shape keeps its proportions (a heart stays a heart), and the contents are laid out in its box.
  - A frame image drawn *over* a photo counts as a container too.
  - Containers inside containers nest.
- **Too small to read:** a fixed-proportion container that can't hold its text at 1.2x the readability floor on the target canvas is not kept as a group for that target. A frame round the whole ad on a 320x50 strip is an example. Its contents are then laid out as separate units.
- **Gates (only with the switch on):**
  - G9 ("nothing hidden") now lets an element stay under an image that the source already drew over the whole of it. A frame raster round a photo would otherwise count as hiding the photo.
  - A new gate, `G10_layers`, rejects any candidate (from any strategy) in which a container kept for that target no longer holds everything it held in the source.
  - The general overlap and hidden rules are otherwise unchanged.

**Development data: `crello_dev`.** These are 60 more Crello ads, built with challenge2's converter and quality gate by `dev/distill/make_crello_dev.py`. They share no design family with challenge2. Like challenge2, they are gitignored (`training-data/corpus/crello_dev.nosync/`). Load them with `load_corpus("crello_dev")` or `ab_engine run --split crello_dev`. Selection favoured layered ads but kept 18 simple ones:
- 42 are layered: 34 have at least one container, and 29 have a photo or pattern in the background layer.
- There are 44 top-level containers (8 stretching, 36 fixed-proportion, 9 nested).
- In the pool of 214 eligible families, 86% are layered.

All development (thresholds, the pruning rule, the gate) was done on crello_dev and the synthetic train split. Holdout and challenge2 were run once, at the end, only to measure.

**Detectors** (`dev/distill/layer_flaws.py`; measured on the winning layout, against its source):
- **Container emptied:** a container no longer holds any of its content (each piece less than half inside it).
- **Container split:** a container holds some of its content but not all.
- **Background shrunk:** a background-layer element covered 90% or more of the source but covers less than 90% of the target. *Avoidable* counts only elements that could have filled the target.
- **Text newly on a busy area** or **text newly below 3:1 contrast:** measured on a render without the text, the same way the corpus quality gate measures it.

| Detector (share of cells) | crello_dev 743: v2 → v2+layers | train 60: v2 → v2+layers | holdout 192: v2 → v2+layers | challenge2 623: v2 → v2+layers |
|---|---|---|---|---|
| Container emptied | 38% → 19% | 12% → 3% | 14% → 0% | 38% → 16% |
| Container split | 6% → 0% | 3% → 3% | 1% → 0% | 4% → 0% |
| Background shrunk (all / avoidable) | 10% / 2% → 8% / 0% | 0% → 0% | 0% → 0% | 22% / 0% → 22% / 0% |
| Text newly on a busy area | 11% → 6% | 2% → 2% | 2% → 3% | 9% → 8% |
| Text newly below 3:1 contrast | 17% → 10% | 0% → 0% | 0% → 1% | 11% → 9% |
| Fallback picked | 12.9% → 17.4% | 8.3% → 10.0% | 1.6% → 3.6% | 17.7% → 25.4% |

Notes on the table:
- **Background shrunk:** what remains is all `contain` rasters on a canvas of a different shape. The aspect rule forbids filling the canvas with those, so none of it was avoidable.
- **Emptied containers left over:** these are containers that were judged too small to keep as a group on that target.
- **Other detectors:** the bisection flaw detectors (`bisect_flaws.py`) barely move. On holdout, orphaned decorations fall from 30% to 21%. Text too small is 32% → 34% on holdout and 14% → 14% on challenge2. Floating pieces are 6% → 9% on holdout.
- **Gates and speed:**
  - 98.6% (crello_dev) to 100% (holdout, challenge2) of bisection's candidates pass the hard gates.
  - No winning design fails a gate. Before, 4 challenge2 winners (all fallbacks) failed G9; now none does.
  - Engine time is about the same, 0.6 to 0.8 s per cell.

**Changed picks** (`design_hash` by cell): 66 of 192 holdout cells differ from `ab/bisect2` (run `ab/b2_layers`), and 255 of 623 challenge2 cells differ from `ab/bisect2_chl2` (run `ab/b2_layers_chl2`). The detector caches re-ran every cell, and their designs match those runs exactly. Nothing has been judged yet.

**Honest visual verdict** (16 changed crello_dev cells, source | v2 | v2+layers, against `rubric.md`; plus about 30 cells viewed while developing):
- **Clearly better on about half.**
  - Cards, bands and labels keep their copy. Before, the shape was left empty and the copy floated elsewhere.
  - A full-width band stays full-width with its line of text.
  - A circle or blob keeps its caption, and frames stay round their photo.
  - A white card keeps the headline, the list and the price together.
- **About the same on a quarter.**
- **Worse on about a quarter.** Mostly these are cells where the engine now picks the uniformly shrunk original (fallback) or the proportional reflow, because the grader prefers it to the layered re-layout:
  - on a near-square target that is a fair answer;
  - on a strip (320x50) or a half page it is a small, hard-to-read ad.
  - A few layered layouts are also awkward, such as a big fixed-proportion box holding only a tiny caption.

**Risk:** the fallback share rises, by about 8 points on challenge2. The stranding fix is real, but whether it wins overall depends on those fallback cells. The next step is a teacher-judged A/B (`bisect2` vs `b2_layers`, and `bisect2_chl2` vs `b2_layers_chl2`).

Commands (from `distil-research/chameleon/`, `PYTHONPATH=$PWD`, CPU only):
- `python -m dev.distill.make_crello_dev`
- `python -m dev.distill.layer_flaws rebuild --name X --split crello_dev|train|holdout|challenge2 --env ...`
- `python -m dev.distill.layer_flaws rates --name X --name Y`
- `python -m dev.distill.layer_flaws sheet --name X --name Y --out <a .nosync folder>/sheet.png`

The caches and contact sheets are in `training-data/ab/_flaws.nosync/` (gitignored).

## Round 3 leaderboard (2026-10-21): what is now the best engine

All teacher-judged pairwise on changed cells (confidence >= 2). "Real ads" is challenge2: 50 real Crello ads, 623 cells, never used for development (crello_dev is the development set).

| Step | Holdout (192 cells) | Real ads (623 cells) | Ship-readiness on changed cells |
|---|---|---|---|
| base -> bisection | 38-17 (p=0.006) | – | 2.13 -> 2.31 (holdout) |
| bisection -> v2 (`CHAMELEON_BISECT_V2`) | **33-9** (p<0.001) | 26-17 (p=0.22) | 2.31 -> 2.54 / 1.82 -> 1.88 |
| v2 -> + hierarchy lock | **14-4** (p=0.031) | – | 2.27 -> 2.50 |
| v2 -> + near-fit | 11-4 (15 cells, p=0.12) | – | 2.22 -> 3.17 |
| v2 -> + relation penalty | 28-26 | – | dropped |
| v2 -> v3 (= v2 + hierarchy lock + near-fit) | – | **25-9** (p=0.009) | 1.73 -> 1.93 |
| v3 -> v3L (+ `CHAMELEON_LAYERS`) | 31-20 (p=0.16) | **49-21** (p=0.001) | 2.31 -> 2.49 / **1.61 -> 2.03** |

**Best engine: v3L** = `CHAMELEON_STRATEGIES=bisection CHAMELEON_BISECT_V2=1 CHAMELEON_HIER_LOCK=1 CHAMELEON_NEAR_FIT=1 CHAMELEON_LAYERS=1`.

Weak spots left: strips on real ads (layers lose 6-8 there) and wide formats on holdout (6-7); the plain uniform fit now wins 31% of real-ad cells (12.5% holdout), so the re-layout still often loses to doing nothing clever. Absolute quality is still low (about 2-2.5 of 5).

### End-to-end check (2026-10-21): today's engine vs v3L on real ads, both orders
`ab_engine judge --a base_chl2 --b v3L_chl2 --max-cells 100 --swap-share 1.0`: every pair asked in both image orders (a disagreement counts as unsure).

- **v3L wins 73-17 (10 unsure), p < 0.001**, on 100 judged changed cells; 91 of 100 cells got the same verdict in both orders.
- Every aspect slice favours v3L: strip 23-1, wide 19-6, square 10-3, very tall 9-3, tall 8-2, landscape 4-2 (worst-slice share 0.67).
- Mean ship-readiness on changed cells 1.62 -> 2.13; engine time 0.39 -> 0.76 s per cell.

## Polish pass and near-fit guard (2026-10-21)
Two switches on top of v3L, both off by default; with them off the output is byte-identical (`tests/test_polish.py` checks a golden hash of `resize()` on the 3 samples x 14 targets with the v3L switches on, and the bisection golden tests still pass after the shared helpers moved). Nothing here is teacher-judged yet; CPU measurements only.

**1. Polish (`CHAMELEON_POLISH=1`, `engine/pipeline/layout/polish.py`).** v2's detail fixes only ever applied inside bisection's own layouts. The polish pass runs them on every other gate-passing candidate before grading (templates, reflow, extra floor plans, other strategies; not the fallback, not bisection when v2 built it), so the grader picks among polished versions. It repairs only what is broken, in this order:
- **Round pills:** a pill whose shape is a circle or ellipse keeps the source aspect (the condensed pill squashed it).
- **Logo floor:** the logo's sqrt(area) reaches 6.5% of the long side (the v2 floor), if it can grow without hitting anything.
- **Readable badges and CTAs:** the label reaches max(role floor, 1.1% of the long side). 1.1% is the validated "text too small" threshold. It is capped at 1.5x the source size and at the hierarchy lock's role cap of the headline (CTA 0.6, badge 0.5).
- **Safe margin (the v2 margin):** units too close to the same edge move inward together, so a row stays a row. If a unit can't move, a single text re-wraps narrower at the same size. Failing that, anything but the headline shrinks a little (images to 0.8x; text to 0.85x and never under its floor). A unit that sat at the edge in the source counts as a deliberate bleed and is left alone.
- **Decorations:** a decoration that is squashed, orphaned from the unit it framed, half-covered by copy it didn't touch in the source, or sitting on a unit the pass moved is re-placed the v2 way: round backdrop, a round "hug" round what it held, frame, park, or visible corner. The option with the fewest flaws wins. It is kept only if it beats the current placement, and a container must still hold what it held.

Every growth or move must keep the unit's overlap with other content from growing (with a small gap), keep shared edges and centres, and keep ink at least as far from an edge as before. The headline is never shrunk, and each step is kept only if the candidate still passes every hard gate. A candidate that already fails a gate is left alone. A first version "repaired" such candidates back into the race, which shifted strip picks from bisection to row templates with smaller headlines, so that was removed. Shared code moved from `bisection.py` to `polish.py`: `straddles`, the V2Builder decoration logic, `logo_floor`, the v2 margin and frame constants. Cost: about 1-2 ms per candidate, so engine time is unchanged (0.66-0.70 s per cell).

**Candidate level** (`python -m dev.distill.polish_eval`; every polishable gate-passing candidate before → after; the headline size is identical by construction):

| Detector (share of candidates) | train, 60 cells, 1,064 candidates | crello_dev, 120 cells, 1,510 candidates |
|---|---|---|
| Squash | 8.1% → 2.6% | 0% → 0% |
| Logo a speck | 12.2% → 3.9% | 13.2% → 5.6% |
| Decoration orphaned | 18.0% → 8.3% | 28.2% → 6.6% |
| Edge crowding | 43.9% → 32.0% | 48.1% → 41.9% |
| Text too small | 37.3% → 36.7% | 28.7% → 27.9% |
| Mixed alignment | 29.9% → 30.1% | 27.9% → 28.5% |
| Container emptied | 0.3% → 0.5% | 35.4% → 35.4% |
| Changed / failing a gate after | 83% / 0 | 65% / 0 |

**On the winners** (v3L → v3L + polish → + guard; train is 420 cells, crello_dev is 743): bisection still wins about 60-67% of cells and is untouched, so the shifts are small:
- Squash: 2% → 0% on train.
- Logo a speck: train 1% → 0%; crello_dev 9% → 7%.
- Decoration orphaned: train 4% → 3%; crello_dev 7% → 3%.
- Edge crowding: train 13% → 13%; crello_dev 27% → 26% (25% with the guard).
- Mixed alignment: train 22% → 23%; crello_dev 17% → 19%.
- Text too small: train 17% → 18%; crello_dev 19% → 18%.
- Picks changed by polish: 56 of 420 on train and 77 of 743 on crello_dev.
- No winner fails a gate. crello_dev has the same 4 fallback G9 failures as v3L.

**2. Near-fit guard (`CHAMELEON_NEAR_FIT_GUARD=1`, in `pick_best`; only with `CHAMELEON_NEAR_FIT`).** Near-fit gives near-aspect cells to the uniform fit unless a re-layout leads by more than 0.10. The guard keeps the best re-layout instead when both of these hold:
- the fit's headline (and CTA, if there is one) is at most **0.8x** the re-layout's;
- the fit's largest empty rectangle is at least **0.10** of the canvas larger than the re-layout's.

Features come from `near_fit_features`; analysis is in `dev/distill/nearfit_guard.py`.
- **Why these thresholds (train only):** 14 train judgments compare the fallback with a layout on near-aspect cells. The fallback won 12.
  - The 2 layout wins are one cell judged twice. There the fit's headline is 0.38x the layout's, and its empty rectangle is +0.12 larger than the layout's.
  - The 12 fit wins have empty gaps of +0.07 at most. 0.10 sits between the two groups.
  - Any headline threshold from 0.38 to 0.99 separates train. 0.8 is a design choice ("clearly smaller"), not fitted.
  - Result on train: 14/14 agree.
  - The task brief asked for unused canvas measured against the source. Measured that way it separated nothing on train (+0.07 at most everywhere), because the uniform fit's extra bands are thin at these aspect changes. So the guard compares with the re-layout instead.
- **Check against the teacher (holdout, `bisect2__vs__b2_nearfit`, 18 verdicts, not used for tuning):** the guard fires on 5 of them.
  - 3 agree with the teacher: c27 tasting-note medium rectangles, where it preferred the re-layout.
  - 2 go against it: c17 billboard and c19 medium rectangle, where it preferred the fit.
  - Agreement is 14/18, against 13/18 for "always the fit".
- **Flips (v3L picks):**
  - train: 8 of the 42 cells where near-fit picks the fit (51 near-aspect cells);
  - holdout: 3 of 17, including c17 and c19 above, which the teacher preferred as the fit;
  - crello_dev: 8 cells changed;
  - challenge2: fallback picks 193 → 185.

**Runs** (CPU only, compared with `ab/v3L` and `ab/v3L_chl2` by design hash; no gate failures; about 0.71 s per cell):

| Run | Changed picks | Notes |
|---|---|---|
| `v3L_pol` (holdout) | 25 / 192 | 13 are the same candidate, polished |
| `v4p` (holdout, + guard) | 28 / 192 | the extra 3 are guard flips from fallback to bisection |
| `v4p_chl2` (challenge2, + guard) | 55 / 623 | 30 same candidate polished; fallback 193 → 185 |

**Official samples:**
- **Level 2 (1600x400):** v3L's pick does not change. Polish fixes the row template: the badge goes from 9 to 17.6 px, and the circle behind it stays round or keeps its badge. But the grader still prefers bisection's three-zone layout, 0.921 to 0.917. The polished row does win the nearby LinkedIn banner (1584x396).
- **Level 1 (1080x1350):** the guard replaces the uniform fit with a bisection layout. The headline goes from 60 to about 90 px, the CTA is larger, and the canvas is used top to bottom. The logo ends up a little smaller and with a large gap above the headline. This looks better to me, but the holdout check above warns that the teacher sometimes prefers the faithful fit.

**Honest visual verdict** (source | v3L | v3L + polish + guard: 16 changed train and crello_dev cells, plus all changed sample cells, against `rubric.md`):
- **Better on about 6.** Frames back behind the product (an orb behind the bottle on a story). Huge blurry decorations parked small again. A broken bisection layout replaced by the faithful reflow. Readable badges and pills.
- **Same on about 7.** Mostly the logo floor and margin nudges, a few pixels each.
- **Worse on about 3.**
  - Strips where a polished row now narrowly beats bisection with a smaller headline (crello facebook cover, a linkedin banner).
  - One guard flip on a real-ad story (crd_35): the uniform fit was faithful, and bisection blows up the logo and puts the quote on a busy part of the photo.

**Verdict:**
- Polish is a safe detail pass. It cuts squash, specks and orphans by half to two thirds per candidate, and costs nothing in time or gates. But it rarely changes which layout wins, so expect a small effect.
- The guard rests on one train cell. The holdout check is only 14/18 against 13/18. It should be judged before it ships, and it should never ship without polish.
- What to judge: `v3L` vs `v3L_pol` (25 cells) and `v3L_chl2` vs `v4p_chl2` (55 cells).

Commands (from `distil-research/chameleon/`, `PYTHONPATH=$PWD`, CPU only):
- `python -m dev.distill.polish_eval --split train|crello_dev --cells N`
- `python -m dev.distill.nearfit_guard train | check | flips --split S`
- `python -m dev.distill.layer_flaws rebuild|rates|sheet --name p_tr_v3L|p_tr_pol|p_tr_pg|p_dev_v3L|p_dev_pol|p_dev_pg`

The caches and contact sheets are in `training-data/ab/_flaws.nosync/` (`sheets/samples/`, `sheets/polish_*.png`).

## Anchored retargeting (constraint-style resize) (2026-10-21)
On real ads the plain uniform fit (the safe fallback) scores 2.49 out of 5 with the teacher, against 2.04 for our re-layouts: the judge rewards keeping the original composition. But the uniform fit wastes the new canvas: it letterboxes, and the text stays small. `anchored` is the resize that design tools do (Figma constraints, Adobe XD responsive resize and patent US 11,320,975, Bannersnack smart resize): keep the composition, re-fit the background full-bleed, and re-place each group of content by where it was anchored. It is a new strategy, on with `CHAMELEON_STRATEGIES=bisection,anchored` (no registry edit was needed: strategies load by name). With `anchored` not named, the output is byte-identical (`tests/test_strategy_anchored.py` checks golden hashes of the v3L engine on the 3 samples x 13 targets, taken from git HEAD before the strategy existed).

**Words used here.**
- An **anchor** says which part of the canvas a piece is tied to, per axis. Example: a logo 60 px from the top of a 1080 px square is anchored *top*; on a 1080x1350 portrait it stays near the top, its gap scaled. A headline whose two side gaps are equal is anchored *centre*. A piece spanning most of a side *stretches*.
- The **content scale** `s` is one zoom for all content. `s0` is the uniform fit's scale (`min(W/Ws, H/Hs)`); larger `s` makes the content bigger and uses the canvas.

**What it does** (`engine/pipeline/layout/strategies/anchored.py`):
1. **Pieces** are the plan's content units. With `CHAMELEON_LAYERS=1`, a container and everything on it is one piece. The background layer is re-fitted full-bleed as everywhere else.
2. **Anchors** come from each piece's edge gaps in the source: centred if its gaps are within 10% of the side of each other, pinned to the nearer edge if that gap is under 20% of the side, otherwise centred with its offset kept. A piece that touched an edge stays on it.
3. **Blocks:** pieces stacked across the axis that has spare room, close together and sharing an edge or centre line (a text column of headline, subhead and CTA), move as one block. A photo never joins one. A block too long to fit splits at its widest gap.
4. **Scale:** each candidate uses one `s` from `s0` up to the fill scale (`max(W/Ws, H/Hs)`, at most 3 x `s0`). Text keeps its role floor; a lone text re-wraps inside its scaled width instead of shrinking; logos keep bisection v2's floor and cap; a photo may give up to 30% so the rest fits; a CTA pill too wide at its floor is rebuilt round its label. On a much longer side, anchors are taken inside a frame at most 1.6 x the scaled source side, so a tight cluster stays together instead of flying to the far edges. On a strip (short side at most 0.2 of the long side) everything sits on the middle line.
5. **Collisions:** two pieces that would overlap (and didn't in the source) are pushed apart, keeping their source order. A pair stacked one above the other is first kept apart vertically. If that doesn't fit, they move along the long axis, so "above" becomes "left of" on a wide canvas. Example: square to 1600x400 turns logo / product / text column into a left-to-right row. If no order-keeping placement fits, that `s` is dropped. Pieces that share no alignment line then snap onto a neighbour's line within 2% of the long side.
6. **Hierarchy:** text scales with everything, so the source's text ratios hold. An `s` where a floor makes another text outgrow the headline is dropped.
7. **Candidates** (`anchored/mode=.../r=...`, `r = s / s0`): the largest feasible `s`, the fit `s`, one in between, and a *spread* variant whose centred pieces map proportionally across the whole canvas. That is at most 4 distinct designs per cell, each checked against the hard gates. They compete in `pick_best` like any other candidate (the near-fit rule still treats them as re-layouts).

**Train and crello_dev** (60 cells each, v3L stack; `dev/distill/eval_anchored.py`; caches `anc_tr_base`/`anc2_tr`, `anc_dev_base`/`anc2_dev` in `training-data/ab/_flaws.nosync/`):

| | train: v3L → +anchored | crello_dev: v3L → +anchored |
|---|---|---|
| Anchored wins the pick | 20% (12) | 30% (18) |
| Cells with an anchored candidate | 75% | 92% |
| Anchored candidates passing the gates | 100% | 95% (G8 label/pill on nested pills, G9) |
| Fallback picked | 16.7% → 10.0% | 21.7% → 15.0% |
| Edge crowding (detector) | 13% → 12% | 30% → 35% |
| Mixed alignment | 15% → 27% | 13% → 8% |
| Text too small | 17% → 12% | 10% → 7% |
| Text newly below 3:1 contrast | 0% → 0% | 7% → 3% |
| Container emptied / split | unchanged | unchanged |
| Seconds per cell | 0.67 → 0.72 | 0.85 → 1.06 |

The edge-crowding row is after raising the safe margin to 6% of the short side. With 4%, edge crowding rose to 30% (train) and 52% (crello_dev).

**Holdout and challenge2** (CPU only, compared with `ab/v3L` and `ab/v3L_chl2` by design hash; no gate failures):

| Run | Changed picks | Anchored winners | What they replaced | Fallback picked |
|---|---|---|---|---|
| `v3L_anch` (holdout) | 63 / 192 | 63 | bisection 45, fallback 4, templates/reflow 14 | 24 → 20 |
| `v3L_anch_chl2` (challenge2) | 168 / 623 | 172 | bisection 75, fallback 61, templates/reflow 32 | 193 → 128 |

Every changed pick is an anchored winner. On holdout they come mostly from story (15), billboard (9) and facebook cover (7). On challenge2 they are spread over all 13 targets (7 to 19 each). 143 of the 172 are anchored mode and 29 spread. Engine time is about the same: 0.70 → 0.70 s per cell on holdout, 0.76 → 0.79 s on challenge2.

**Official samples** (`python -m dev.run_matrix`, all 13 targets, plus 1600x400): anchored changes 1 of 42 picks, level 3 → story. Its faithful stack replaces a stack template that put the headline above the product.
- **Level 2 → 1600x400:** the best anchored candidate (r = 2.5) loses to bisection by 0.006. It is a clean, faithful row: circle with badge, logo, product, text column. But its headline is about half the size of bisection's, so call it a toss-up.
- **Level 1 → 1080x1350:** anchored at r = 1.25 is the source at 1.25x, filling the width. To my eye it is better than the uniform fit that wins, but it trails bisection by 0.037 and the near-fit rule keeps the fallback.
- **Level 2 → 1080x1350:** the same story (r = 1.25, −0.015).

**Honest visual verdict** (source | v3L | v3L + anchored | best anchored candidate, against `rubric.md`; 12 crello_dev cells, 10 train winners, and the 3 samples × 14 targets; sheets in `training-data/ab/_flaws.nosync/sheets/anc*` and `chameleon/output/anchored_sheets/`):
- **Better on about half of the anchored winners.**
  - Square ↔ portrait and portrait → story: the composition is kept and the canvas is filled.
  - Skyscraper → portrait and leaderboard → square: big, faithful text instead of a letterboxed shrink.
  - Two youtube thumbnails where bisection had blown up a decoration.
  - A card or quote that keeps its shape at a readable size.
- **About the same on most of the rest.** These are near-identical to the fallback or bisection.
- **Worse on a few.**
  - A leaderboard source → portrait or story spreads a thin strip's pieces over a tall canvas, leaving empty bands.
  - A linkedin banner where the title ends up cramped against its frame.
  - Columns whose CTA keeps its own edge anchor, so alignment is mixed (train mixed alignment 15% → 27%).
  - One candidate that did not win (Easter billboard, spread) put text over an illustration.
- **Where it loses on the grader but looks better:** near-aspect and portrait cells (the samples above, a photo card on youtube). The grader's legibility term prefers text blown up to 1.6x its floor, and the near-fit rule keeps the fallback unless a re-layout leads by 0.10.

**Risks:**
- The biggest change is on real ads: 61 challenge2 cells move from the uniform fit, the teacher's favourite there, to an anchored layout. Anchored is the same composition at a larger scale, but whether the teacher prefers it is exactly what has to be judged.
- Mixed alignment rises on train.
- Nothing here is judged yet. What to judge: `v3L` vs `v3L_anch` (63 cells) and `v3L_chl2` vs `v3L_anch_chl2` (168 cells; the 61 fallback → anchored cells are the key slice).

Commands (from `distil-research/chameleon/`, `PYTHONPATH=$PWD`, CPU only, ≤ 3 processes):
- `python -m dev.distill.eval_anchored rebuild --name X --split train|crello_dev|samples --cells 60 --env CHAMELEON_STRATEGIES=bisection,anchored --env ...`
- `python -m dev.distill.eval_anchored stats --name A --name B`
- `python -m dev.distill.eval_anchored shapes --name X`
- `python -m dev.distill.eval_anchored sheet --base A --name B --out F.png [--pick won]`
- `python -m dev.distill.eval_anchored changed --run ../training-data/ab/v3L_anch --ref ../training-data/ab/v3L`
- Detector rates: `python -m dev.distill.layer_flaws rates --name anc_tr_base --name anc2_tr`

### Anchored retargeting judged (2026-10-21): new best engine
`CHAMELEON_STRATEGIES=bisection,anchored` on top of v3L, vs v3L:
- **Holdout: 46-11 (6 unsure), p < 0.001**, 63 changed cells; ship-readiness 2.22 -> 2.80. Wins in every slice with enough cells (wide 17-2, tall 13-5, landscape 6-0, very tall 5-1, square 4-2; strip 1-1).
- **Real ads (challenge2): 58-21 (21 unsure), p < 0.001**, 100 of 168 changed cells; ship-readiness 1.80 -> 2.17. Square 11-1, strip 14-1, very tall 11-2; weakest slice wide 11-10.

**Best engine now: v4 = v3L + anchored** (`CHAMELEON_STRATEGIES=bisection,anchored CHAMELEON_BISECT_V2=1 CHAMELEON_HIER_LOCK=1 CHAMELEON_NEAR_FIT=1 CHAMELEON_LAYERS=1`). Keeping the original composition while filling the new canvas is what the judge rewards most.
