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

## Text robustness pass (CHAMELEON_TEXT_SLACK) (2026-10-21)
**The problem.** The official converter (`renderer/converter/wrap.py`) word-wraps each text greedily at its box width and only breaks at spaces. The engine measures with the same code and often sizes a box to exactly what its wrap needs. Such a box re-wraps if anything at scoring time differs by a pixel: rounding, another skia or cr-renderer build, or Linux font rasterisation on the reference machine. A **fragile** box is one whose official line count changes when the box is 1 px narrower. Example: "Shop the Summer Line" in a 137.41 px label box whose line is 137.40 px wide (core sample, 300x250). One pixel less and "Line" drops onto a second line, under the button.

**The switch** (`engine/pipeline/layout/text_slack.py`, off by default, byte-identical when off). It is one pass on the final design, after the pick and the refiners. It changes no content or font family, and each change is kept only if every hard gate still passes. Slack is `max(1.5 px, 1.5% of the box width)`: the wrap at width - slack must equal the wrap at width.
1. **Pill labels.** A pill label is a text in a container_label unit, or a text inside a small shape of its own group. It stays on its source line count (one, for every label in the samples) with slack. Fixes, in order:
   - widen the label box inside its pill;
   - else widen the pill symmetrically, if it stays inside the canvas safe area and gains no overlap;
   - else shrink the label font in 0.5% steps, never below its role floor;
   - else widen the pill as far as allowed, and report the label (`Result.notes`).
2. **Every text box.** Widen the box the way `text_align` allows: left keeps x, right keeps the right edge, center grows both ways. This moves no ink. The box must stay in the canvas and inside every element that held it, with no new overlap. Fallbacks, in order:
   - widen up to 15% to a different wrap with the same line count;
   - shrink the font, at most 6% and never below the role floor (a headline that was the largest text stays the largest);
   - last resort: grow on the free side, the ink shifting at most 3 px.
3. **Vertical.** A box shorter than its drawn block grows to it.

**Where it runs.** `CHAMELEON_TEXT_SLACK=all` also runs the pass on every candidate before grading. Not chosen: almost every change is a wider box with the ink in place, which the grader's box metrics read as a different layout. On 99 cells, `all` changed 10 picks for no visual reason; both modes leave no 1 px-fragile text.

**Robustness** (`python -m dev.distill.text_robustness run|report|sheet`). The two sets are the 3 samples x 13 targets (182 texts, 91 pill labels) and 60 holdout cells (320 texts, 129 pill labels).

| Engine | Fragile at 1 px: samples / holdout | Short of full slack | Pill labels fragile at 1 px | Pill labels on >1 line or outside the pill |
|---|---|---|---|---|
| today's engine | 34.1% / 16.2% | 45.1% / 24.7% | 54 / 38 | 0 / 0 |
| today + TEXT_SLACK | 0.5% / 0% | 1.1% / 0% | 0 / 0 | 0 / 0 |
| v4 | 31.9% / 13.1% | 49.5% / 29.1% | 52 / 29 | 0 / 0 |
| **v4 + TEXT_SLACK** | **0% / 0%** | **1.1% / 0%** | **0 / 0** | **0 / 0** |

- **Left short of full slack:** 2 sample texts (advanced_01 leaderboard) are boxed in by the CTA, and their fonts are at the floor. Both still survive 1 px.
- **Short boxes:** v4 had 7 boxes on holdout shorter than their text block; the pass leaves none.
- **No pill label wraps in today's output.** The risk is the fragile ones (52 of 91 in the samples), and the pass takes that to 0.
- **Font change on same-pick cells:** 2 holdout texts shrink (mean −0.77%, at most −1.02%); no sample text changes size.
- **Changes in v4 + TEXT_SLACK vs v4:** 180 text boxes widen with their ink in place, and 8 texts are drawn differently. No pill needed to grow.

**A/B runs** (CPU, `ab/v4ts` vs `ab/v4` on holdout, `ab/v4ts_chl2` vs `ab/v4_chl2` on challenge2):

| | Cells | Designs changed | Picks changed | Changed designs that render differently (768 px) | Gate failures | s / cell |
|---|---|---|---|---|---|---|
| holdout | 192 | 143 | 0 | 22 | 0 | 0.725 → 0.789 |
| challenge2 | 623 | 304 | 0 | 42 | 0 | 1.037 → 1.058 |

**Visual check** (`ab/_flaws.nosync/sheets/text_slack_{samples,ab}.png`: source, v4, v4ts, and a zoom on the changed pixels; the 3 samples at 300x250 plus the 12 cells with the most changed pixels). The samples at 300x250 render the same, with wider boxes. The visible changes are:
- a line break moved, with the same line count ("When it / is gone" → "When it is / gone"; "Every mixed case / of Solstice" → "Every mixed case of / Solstice");
- about 1% font shrinks;
- 1 to 2 px ink shifts.

None reads as better or worse. **Verdict: neutral to the eye and much more robust**, which makes it worth turning on. No judging is needed: the picks are unchanged, and the pixel changes are tiny.

**Risks.**
- The pass only guards against a box getting relatively narrower (glyphs rendering wider at scoring). If glyphs render narrower, a line could take the next word; that is less harmful, since the text still fits its box.
- A rewrap moves a line break the source designer set: same line count, different break.
- The shifted last resort moves centred ink off its centre line by up to 1.5 px.

**Port to today's real engine.** The real engine (`chameleon/engine` as on the `stage1-three-node` branch; this research copy with no switches stands for it here) has the same fragility: 34% of sample texts and 16% of holdout texts. A minimal port:
- copy `text_slack.py` into `engine/pipeline/layout/`;
- replace its `polish.safe_margins` import with the two-line margin helper (polish.py doesn't exist there);
- after `pick_best` in `resize_with_report`, call `best = text_slack.apply(source, target_canvas, plan, measurer, best)` (it never raises, and returns `best` itself when nothing changes).

Everything else it uses already exists there: `geometry`, `place.floor_px`/`resize_text`, `normalize.finalize`, `gates.check_gates` and `TextMeasurer`. With that call, today's engine goes to 0.5% / 0% fragile, with 7 / 5 texts shrunk by about 1% and no pick changes.

## Balance pass (bands, groups, leftover space) (2026-10-21)
The teacher's critiques of v4's layouts (`v3L__vs__v3L_anch` and the other v3L/v4 pairs) keep naming three things the grader barely sees: **lopsided whitespace** ("the stack sits too high, with about 30px above the headline and about 75px of empty band below the button"), **bands that stop spanning** ("the dark header bar no longer spans the canvas. It is inset with cream gaps on the left and right") and **related pieces that drift apart** ("'Ships late June' has drifted far below the CTA, cut off from its button"). `CHAMELEON_BALANCE=1` (`engine/pipeline/layout/balance.py`, one hook at the end of `generate_candidates`, after polish) repairs them on every gate-passing candidate before grading, so `pick_best` chooses among the repaired versions. With the switch off the output is byte-identical (`tests/test_balance.py` checks golden hashes of the v4 engine on the 3 samples x 13 targets, taken from git HEAD e15fb0f before the pass existed). Nothing here is teacher-judged yet; CPU measurements only.

**What it does** (the safe fallback and candidates that already fail a gate are left alone; each step is kept only if the candidate still passes every hard gate and no text ends below its role floor or its size before the pass):
- **a. Bands span.** A shape or image (not the background) that spanned the source edge to edge on an axis (95% of it, or touching both edges) spans the target on that axis again. A rectangle or `cover` image stretches (the image crops); a `contain` image may grow at most 1.25x to span, if it fits. A band that touched both edges touches both again; one inset a little keeps its insets in proportion. A band that would newly cover a content piece it didn't hold in the source is left as it was.
- **b. Groups stay together.** Two content pieces that sat close in the source (stacked or side by side, gap at most the smaller one's size along that axis and at most 8% of the side, overlapping across it by half the narrower one), now more than 1.5x their scaled source gap apart (and 3% of the short side more): the less important one moves toward the more important one, to the scaled source gap, along the axis they now stack on. Pairs with a photo are skipped (a photo moving off its caption is a re-layout, not a split group).
- **c. Leftover space.** Per axis, the free-floating content block (every content piece except those spanning the axis, and small pieces, at most 6% of the source, that hugged an edge in the source: the logo top-left stays top-left; the headline's group always moves) sits in its lane (the canvas, narrowed by fixed pieces before or after it). If its margins are lopsided (the larger more than 2x the smaller and more than 6% of the axis apart) and the source's weren't lopsided the same way, the block shifts toward equal margins. Then, if the block leaves more than 25% of the safe area unused on the short axis, it grows uniformly about its centre by up to 1.15x (texts re-measured and re-wrapped if needed; the headline stays the largest text).
- Every move carries the decorations sitting on the piece, never increases a piece's overlap with another (1.5% of the short side apart), keeps ink inside the v2 safe margin (or no closer than it already was), keeps a decoration that held a piece holding it, and never leaves a piece half on a decoration.

**Thresholds** are taken from the critiques' own numbers and the brief, not fitted: 2x and 6% from "30px above, 75px below" on a 768 px render (2.5x, 6% of the side); 1.5x and 25% / 1.15x as proposed; 95% "spanning" with a 1% edge tolerance (the layer code's tolerance). Train and crello_dev were used only to check them: steps reverted by the gates (1 of about 1,700 kept steps), gate failures (none), and the contact sheets below. No sweep was run, because the detectors are proxies (the content-aware sizing lesson).

**Detectors** (`dev/distill/balance_flaws.py`), validated against the teacher before any tuning. All 650 designs critiqued in `v3L__vs__v3L_anch`, `v3L__vs__v4p`, `v3L_chl2__vs__v3L_anch_chl2`, `v3L_chl2__vs__v4p_chl2` and `base_chl2__vs__v3L_chl2` were rebuilt with their variant's switches (all 650 match the judged design hash). Each critique was split into its A and B paragraphs, and keyword rules decide whether a paragraph names the flaw:

| Detector | critiques naming it | fires there | other critiques | fires there | gap |
|---|---|---|---|---|---|
| margin_imbalance (content block lopsided, the source wasn't) | 160 | 38% | 490 | 25% | +13 pts |
| empty_band (largest empty rectangle >= 25% of the canvas and >= source + 5 pts) | 235 | 36% | 415 | 16% | +19 pts |
| band_broken (a source-spanning band, at least 2x longer than wide, no longer spans) | 11 | 55% | 639 | 8% | +47 pts |
| group_split (a close source pair, no photo, now > 1.5x its scaled gap apart) | 233 | 54% | 417 | 37% | +17 pts |
| for reference: edge (bisect_flaws) | 189 | 32% | 461 | 30% | +1 pts |

- All four separate, but margin imbalance and group split only moderately: the teacher names two or three flaws per design, so "not named" doesn't mean "absent".
- **empty_band** first measured the largest empty strip across the whole canvas. It did not separate on holdout (14% vs 13%), because a photo at the side keeps any full-width strip from being empty. The largest empty rectangle does (holdout 23% vs 11%, challenge2 40% vs 18%).
- **band_broken** first counted any element touching both edges of an axis. On strip sources that flags products (90 px tall on a leaderboard), so a band must be at least 2x longer than wide. Every holdout critique that names a broken header bar (5, all c07) fires.

**Before / after on the winners** (v4 → v4 + balance; caches `bal_{tr,dev,holdout,challenge2}_{v4,b1}` in `training-data/ab/_flaws.nosync/`):

| Detector | train (60) | crello_dev (60) | holdout (192) | challenge2 (623) |
|---|---|---|---|---|
| margin_imbalance | 13% → 5% | 22% → 10% | 14% → 10% | 22% → 15% |
| empty_band | 12% → 7% | 17% → 20% | 9% → 9% | 22% → 21% |
| band_broken | 0% → 0% | 12% → 5% | 11% → 8% | 11% → 9% |
| group_split | 52% → 43% | 20% → 8% | 46% → 33% | 31% → 19% |
| edge crowding | 12% → 10% | 35% → 37% | 16% → 16% | 33% → 37% |
| floating | 12% → 8% | 12% → 8% | 10% → 12% | 16% → 12% |
| mixed alignment | 27% → 25% | 8% → 13% | 33% → 32% | 22% → 23% |
| text too small | 12% → 12% | 7% → 7% | 27% → 27% | 11% → 11% |
| fallback picked | 10.0% → 11.7% | 15.0% → 16.7% | 10.4% → 10.4% | 20.5% → 20.5% |
| winner gate failures | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |

- Edge crowding rises on real ads (challenge2 +4 pts). The new hits are photos and badges that bled off the source edge, stretched back to the edge by step a: deliberate bleeds that the edge detector can't tell from crowding.
- Per candidate (train / crello_dev): 55% / 48% of gate-passing candidates change. Kept steps: band 42 / 117, group 386 / 153, shift or grow 541 / 386. Only 1 step was undone by the gates. The pass costs about 15-30 ms per cell.

**Runs** (CPU only, 3 workers; compared by design hash):

| Run | Changed picks | Same candidate, balanced | Different candidate wins | Fallback |
|---|---|---|---|---|
| `v4b` vs `v4` (holdout) | 76 / 192 | 50 | 26 | 20 → 20 |
| `v4b_chl2` vs `v4_chl2` (challenge2) | 251 / 623 | 127 | 124 | 128 → 128 |

- `v4_chl2` (the v4 switches on challenge2) is byte-identical to `v3L_anch_chl2`: 0 of 623 cells differ. Its `winners.jsonl` has 1,241 rows because another run wrote to the same folder at the same time. The duplicates have the same hashes, and `load_winners` keeps one row per cell.
- Changes are spread over all 13 targets.
- Engine time: holdout 0.72 → 0.99 s per cell (measured while another run shared the CPU), challenge2 1.04 → 0.98 s. The pass itself costs 15-30 ms per cell.

**Official samples** (`eval_anchored rebuild --split samples`, 13 targets + 1600x400; sheets in `chameleon/output/balance_sheets/`): 12 of 42 picks change, none for level 2.
- **Level 1.** Story is better: logo, headline and CTA form one group instead of three bands with a dead gap. YouTube is better: the headline is centred as in the source instead of left-flush. LinkedIn and 1600x400 are a toss-up: the headline and CTA are now stacked together, but the group sits right of an empty middle. Facebook cover is worse: a different bisection layout wins, and its CTA wraps to three lines beside the headline.
- **Level 3.** Mostly pixel-level badge and ribbon pulls. Facebook cover is a toss-up: another cluttered bisection variant wins.

**Honest visual verdict** (source | v4 | v4 + balance, against `rubric.md`; 16 train/crello_dev cells, 10 holdout cells, 3 header-bar holdout cells, 5 challenge2 edge cells, all changed sample cells, about 60 cells in all):
- **Better on about 11.**
  - Header bars span again: c07 story, linkedin and youtube, exactly the critiqued cells.
  - "Ships late June" is back under its button, and the eyebrow sits with its headline (c07 story, a crello "EDUCATING KIDS:" post).
  - A crello youtube thumbnail where a faithful anchored layout now beats a blown-up bisection one.
  - A left stripe that spans the full height again, with a slightly larger block.
- **About the same, or a toss-up, on about 45.** Small group pulls and shifts of a few pixels, inside layouts that were already fine or still broken (the Lima & Co pattern, the TRX banner).
- **Worse on 4, all pick switches rather than the repair itself:**
  - level 1 facebook cover;
  - a train leaderboard → billboard (c20) where a bisection row with the logo dots between headline and subline beats the faithful reflow;
  - a challenge2 story (cr2_06) where the proportional reflow, its yellow band now spanning, beats a bisection layout with a much larger headline;
  - a crello linkedin banner that now goes to the uniform fit (a toss-up the teacher may like).
- Hierarchy is never traded for fill: the pass never shrinks text and grows only uniformly. But because about half of the changed picks are a *different* candidate winning, the grader re-ranks, and that can reward a weaker layout whose band now spans.

**Verdict.** The pass fixes exactly the flaws it targets: group splits −12 pts on holdout and challenge2, imbalance −4 to −12, broken bands −2 to −7, at no gate cost. Its in-place repairs look right. The risk is the pick reshuffle (26 / 124 different winners). What to judge: `v4` vs `v4b` (76 holdout cells) and `v4_chl2` vs `v4b_chl2` (251 cells). If the judge splits, a cheaper variant is to balance only the winner (after `pick_best`), which keeps every pick and only changes the 50 / 127 in-place cells.

Commands (from `distil-research/chameleon/`, `PYTHONPATH=$PWD`, CPU only, ≤ 3 processes):
- `python -m dev.distill.balance_flaws rebuild-judged --name bal_judged` then `validate --name bal_judged`
- `python -m dev.distill.eval_anchored rebuild --name bal_tr_b1 --split train --cells 60 --env <v4 switches> --env CHAMELEON_BALANCE=1` (likewise `crello_dev`, `holdout`, `challenge2`, `samples`)
- `python -m dev.distill.balance_flaws rates --name bal_tr_v4 --name bal_tr_b1`
- `python -m dev.distill.balance_flaws sheet --name bal_dev_v4 --name bal_dev_b1 --out ../training-data/ab/_flaws.nosync/sheets/balance/dev.png`

### Balance pass judged (2026-10-21): leaning better, not adopted
`v4` vs `v4b` (holdout): 18-12 (30 unsure), p = 0.36. `v4_chl2` vs `v4b_chl2` (real ads): 26-19 (35 unsure), p = 0.37.
Split by kind of change: same candidate repaired 11-2 (holdout) / 7-8 (real ads); a different candidate wins 7-10 / 19-11. The two sets disagree, so balance stays off (committed, `CHAMELEON_BALANCE=1`). Re-test once the picker is improved: half its changes are re-picks, which depend on the grader choosing well.

### Text robustness adopted (2026-10-21)
`CHAMELEON_TEXT_SLACK=1` changes no picks and takes text that re-wraps at 1 px narrower from 32% / 13% (samples / holdout) to 0%. Adopted without a judged A/B: its visible changes are line-break moves and ~1% font shrinks.

## Learned picker (CHAMELEON_PICKER=learned) (2026-10-21)
The grader picks the teacher's favourite among its top 4 diverse candidates only 31% of the time (chance ~25%). `CHAMELEON_PICKER=learned` lets a small pairwise logistic model, trained on 312 teacher rankings of exactly those shortlists, choose instead (`engine/pipeline/grading/learned_picker.py`, weights in `picker_weights.json`, 2.5 KB, offline).
- **How:** `pick_best` runs as before (gates, pixels for the top K, near-fit), then rebuilds the picker study's shortlist: the grader's pick, then by total with at most 2 per family, plus the fallback. It adds pixel metrics where the grader skipped them, plus the relation score, family, grader total and target log aspect, and takes the model's argmax. It only chooses among gate-passing candidates. Any error, fewer than 2 shortlisted, or a target within 2% of the source aspect (no such cell in training) keeps the grader's pick. Unset: byte-identical (golden v4 hashes, `tests/test_learned_picker.py`).
- **Offline:** grouped 5-fold CV top-1 43% vs grader 31% (teacher score of pick 2.57 vs 2.46, oracle 2.95); holdout 47% vs 33% (2.77 vs 2.73).
- **Parity:** `python -m dev.distill.train_picker parity` on all 312 ranked cells gives the same grader pick and the same shortlist (312/312), features within 5e-7 of the stored ones (the stored totals are rounded to 6 places), and the engine's pick equals the offline argmax (312/312).
- **Cost:** +0.035 s per cell on average (median 0.018 s, max 0.33 s). Engine time per cell goes from 0.69 to 0.75 s on holdout.
- **What changes** (v5 = v4 + text slack, the new baseline; v5p = v5 + learned picker): holdout 94 / 192 cells (49%), challenge2 284 / 623 (46%), samples 18 / 39. Gate failures 0 everywhere. Fallback picks: holdout 10% → 6%, challenge2 21% → 20%, samples 15% → 8%. The model moves picks from bisection to anchored: holdout anchored 63 → 118 and bisection 91 → 32; challenge2 anchored 172 → 268 and bisection 262 → 151, with reflow 10 → 37. It also overrides near-fit in both directions (challenge2: fallback → other 68, other → fallback 67).
- **Look (rubric, my own eye, 30 cells):** mixed, no clear win.
  - Better: the core linkedin banner (a calm anchored layout like the source instead of a giant two-line headline and a body squeezed into a 6-line column), the advanced large leaderboard, a cr2 facebook cover and a cr2 leaderboard.
  - Worse: cells where it trades a big, readable headline for a smaller, more faithful arrangement: core half_page, core story (b2_tall_3 parks the NEW FLAVOR badge just above the CTA), the advanced linkedin and half_page (badges scattered), and two cases where the fallback replaces a bolder re-layout (a cr2 linkedin banner, c11 instagram_post). It also overrides near-fit on advanced → instagram_post (the near-original fallback becomes a cramped bisection).
  - About half the changed cells are toss-ups.
- **Verdict:** off by default; worth a judged A/B (`v5` vs `v5p`, `v5_chl2` vs `v5p_chl2`, capped). The model has no source-vs-target aspect feature, so it cannot see why near-fit chose the fallback. A cheap variant to judge next to it would leave the pick alone when near-fit chose the fallback. Contact sheet: `training-data/ab/_flaws.nosync/sheets/picker/v5_vs_v5p.png`.

Commands (from `distil-research/chameleon/`, `PYTHONPATH=$PWD`, CPU only, ≤ 3 processes):
- `python -m dev.distill.train_picker train [--lam 3] --write` (CV + holdout report, writes the weights)
- `python -m dev.distill.train_picker parity --workers 3`
- `python -m dev.distill.ab_engine run --name v5p --env <v4 switches> --env CHAMELEON_TEXT_SLACK=1 --env CHAMELEON_PICKER=learned --split holdout --workers 3` (likewise `challenge2`, `samples`)

### Learned picker judged (2026-10-21): a coin flip, dropped
`v5` vs `v5p` (holdout): 27-26 for v5 (7 unsure). `v5_chl2` vs `v5p_chl2` (real ads): 36-34 for the picker (10 unsure). It changes about half of all picks and gains nothing. Lesson: the study's "headroom" (teacher's best of 4 minus the grader's pick) is inflated by judging noise (the max of four noisy scores), so a model fitted to it chases noise. Only judged head-to-head A/Bs decide. Stays off (`CHAMELEON_PICKER=learned`).

## Final confirmation (2026-10-21): today's engine vs v5 on the unseen real ads (challenge3)
v5 = `CHAMELEON_STRATEGIES=bisection,anchored CHAMELEON_BISECT_V2=1 CHAMELEON_HIER_LOCK=1 CHAMELEON_NEAR_FIT=1 CHAMELEON_LAYERS=1 CHAMELEON_TEXT_SLACK=1`.
challenge3 (50 Crello ads, families disjoint from challenge2 and crello_dev) was never used for building, tuning or choosing anything before this run.

- **v5 wins 70-17 (13 unsure), p < 0.001**, 100 of 554 changed cells, every pair judged in both image orders (93 of 100 order-consistent).
- Every aspect slice: very tall 11-1, landscape 6-1, wide 14-3, strip 16-4, square 14-4, tall 9-4 (worst-slice share 0.69).
- Ship-readiness on changed cells 1.49 -> 2.13. Hard-gate failures among winners: today 13, v5 0. Engine time 0.67 -> 1.23 s per cell.

## Second judge (2026-10-21)
Claude Sonnet 5.5 re-judged two key comparisons on the same cells (`*_s` symlinked variants):
- v3L vs anchored (holdout): Sonnet 43-12, Opus 46-11; where both decided, agreement 43/51 (84%), opposite verdicts 8.
- today vs v3L (real ads): Sonnet 68-21, Opus 73-17; agreement 80/83 (96%), opposite 3.
The big wins are not one judge's taste.

## LLM as the layout designer (2026-10-21): ceiling test
`dev/distill/llm_layout.py`: the LLM proposes every element's box and font size from the source JSON (optionally + render); the engine fits text with the official wrap, keeps image proportions, checks the gates, and falls back to v5 when a plan fails. 20 holdout cells, both image orders:

| Designer | Judge | vs v5 | Ship-readiness (v5 -> LLM) | Gate fallbacks |
|---|---|---|---|---|
| Opus + image | Opus | 14-1 | 2.34 -> 3.38 | 4/20 |
| Opus + image | Sonnet (blind second judge) | 14-1 | 2.12 -> 3.09 | 4/20 |
| Sonnet + image | Opus | 11-1 | 2.29 -> 3.39 | 6/20 |
| Sonnet, JSON only | Opus | 13-1 | 2.22 -> 3.28 | 4/20 |

An LLM designer is worth about +1 point of 5, more than every algorithmic step combined; it isn't self-preference (the second judge agrees), Sonnet matches Opus at half the cost, and vision isn't needed. Next: distil into a small open text model (LoRA, trained on Colab) that runs offline in the engine. The brief allows a locally hosted model but the approved-model list and size limit are not published yet.

### Student v1 (Qwen3-4B-Instruct-2507 + LoRA, 370 Sonnet layouts) (2026-10-21)
Trained on Colab L4 in 18 min (132 steps, 3 epochs; eval loss 0.36 -> 0.267, no overfitting). On the 30 held-out prompts: 30/30 valid JSON, 23/30 pass the gates (7 fall back to v5).
**Judged vs v5 (both orders): v5 wins 18-3 (2 unsure), p = 0.001; ship-readiness 2.91 (v5) vs 2.04 (student).** The student learned the format and a sensible structure but not the geometry: content crammed into part of the canvas with dead bands, buttons floating away from their copy, pieces clipped or overlapping. Outputs start with a stray `</tool_call>` (chat-template artefact; parsed around, fix in the next run).
With the engine's balance + text-slack passes applied to each student plan (`ingest --repair`, 18/23 plans changed): v5 still wins 14-7 (2 unsure, p = 0.19); ship-readiness 2.78 vs 2.13. Repair narrows the gap but the 4B student remains below v5. Next: Qwen3-8B on the same data (capacity test), then more data if needed.
