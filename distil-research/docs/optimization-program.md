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
