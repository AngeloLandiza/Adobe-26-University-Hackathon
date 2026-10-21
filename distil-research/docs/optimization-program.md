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
