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

Results and reports appear on the dashboard (https://chameleon-distill-dashboard.vercel.app) under "Experiments & reports"; each A/B report lives in `training-data/ab/<A>__vs__<B>/REPORT.md`.

## Tournament leaderboard (2026-10-20, paused)

Each variant vs the baseline research engine. Teacher-judged on sampled changed cells (holdout ≤ 60, challenge ≤ 100; confidence ≥ 2). The challenge set holds 40 new ads, never used for tuning. "Worst slice" is the variant's lowest win share over any brand or shape slice with ≥ 5 decided cells; 0.50 is a coin flip.

| Variant | Kind | Holdout (B–A) | Challenge (B–A) | Worst slice (hold / chl) | Verdict |
|---|---|---|---|---|---|
| **retr: per-ad floor-plan retrieval** | AI-designed plans + nearest neighbour | **30–18** (p = 0.11) | **34–14 (p = 0.006)** | **0.53 / 0.69** | 🏆 **best so far; generalises across every slice** (judging ~half done) |
| grid: responsive design grid (S4) | algorithm | 27–24 | 51–33 (p = 0.063) | 0.20 / 0.36 | wins tall / very tall / wide, **loses strips** |
| flex: content-aware slot sizing | algorithm | 24–29 | 32–47 | 0.20 / 0.31 | ❌ rejected (proxy gains didn't convince the judge) |
| repair: mined rules + local repair (S3) | AI-mined rules + search | 15–15 | 11–17 | 0.33 / 0.27 | ❌ no gain so far (judging ~half done) |
| bisect: priority bisection (S2) | algorithm | engine runs done (192 / 473 of 499) | — | — | ⏳ not judged yet |
| structure_reflow (S1) | algorithm | — | — | — | ⏸ agent stopped mid-build; partial files uncommitted in `engine/pipeline/layout/strategies/` |

**Lesson so far:** what generalises best is *per-ad design reuse* (retrieval), not a fixed algorithm or a learned scorer. Proxies (dead space, text size) don't predict the judge; only teacher-judged A/Bs count.

## Resume here (in order, all from `distil-research/chameleon/` with `PYTHONPATH=$PWD`)
1. **Finish round 2 judging** (resumes where it stopped). Each judge command takes `--shard K/4`; run shards 0–3 in parallel, then `--merge`, then `report`:
   - `python -m dev.distill.ab_engine judge --a base --b retr --max-cells 60`
   - same for `base_chl retr_chl` (`--max-cells 100`), `base repair` (60) and `base_chl repair_chl` (100).
2. **Finish bisection.**
   - `python -m dev.distill.ab_engine run --name bisect_chl --env CHAMELEON_STRATEGIES=bisection --split challenge --workers 4`, which resumes the last 26 cells.
   - Judge `base bisect` (60) and `base_chl bisect_chl` (100) as above.
3. **Combine.** Choose per-shape routing on **holdout only**: retrieval everywhere, plus the grid for tall / very tall / wide if it adds on top of retrieval. Run the combined variant, then confirm **once** on the challenge set.
4. **Keep the dashboard live** during runs: `python -m dev.distill.publish_status --every 15`.
5. **S1 (optional):** re-dispatch with `.superpowers/sdd/2026-10-20-distillation/task-S1-brief.md`; partial code exists.
