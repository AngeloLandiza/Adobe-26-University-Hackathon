# B2: teacher-designed floor plans

## Decision
Ship 5 of the 14 clustered templates into the research engine (`b2_tall_3`, `b2_tall_4`, `b2_square_1`, `b2_square_3`, `b2_very_tall_1`), next to the 3 hand-drawn ones. Unselected, all 14 already improve the engine's final pick on holdout ads 24–9. Confirm the selected 5 on *fresh* holdout ads before trusting the higher 20–3.

## Run (2026-10-20)
- **Propose:** Claude Opus 5.5 drew 3 floor plans for each of 120 train cells (tall, very tall, square targets). It was shown the original ad, its ranked pieces, the slot-filling rules and the engine's current best layout to beat: 360 plans, 16 workers.
- **Build:** the engine filled every plan: 359 of 360 built (4 workers).
- **Judge:** 431 judgements against the cell's current best, harsh rubric. A plan "wins" only if every verdict on it, including the swapped re-ask, picks it at confidence ≥ 2.
  - square: 79 of 120 plans won (66%)
  - very tall: 66 of 122 (54%)
  - tall: 61 of 117 (52%)
- **Cluster:** k-medoids per shape class over the winning plans gave 14 templates (square 5, tall 5, very tall 4).

## Holdout check (85 holdout cells, 479 judgements, 0 failures)
Each template's best layout per holdout cell against the engine's current best (strict win rule):

| template | win rate | engine would pick it / and it wins |
|---|---|---|
| b2_tall_3 | 53% | 6 / 5 |
| b2_square_3 | 52% | 9 / 8 |
| b2_square_1 | 45% | 6 / 4 |
| b2_tall_4 | 43% | 6 / 5 |
| b2_very_tall_1 | 42% | 9 / 5 |
| b2_tall_1 | 27% | 5 / 4 |
| b2_very_tall_4 | 27% | 11 / 4 |
| b2_square_2, b2_square_4 | 21% | 0 / 0, 7 / 4 |
| b2_very_tall_2 | 15% | 0 / 0 |
| b2_very_tall_3, b2_square_5, b2_tall_2, b2_tall_5 | 8%, 7%, 3%, 0% | harmful or unused |

**Final-pick replay:** the engine's grader picks among its current best plus the new templates' layouts; where the pick changes, we look up the teacher's verdict.

| templates added | picks changed (of 85) | new pick wins – loses |
|---|---|---|
| all 14 (unselected, unbiased) | 37 | **24 – 9** (73%) |
| the 5 shipped (selected on this same holdout, so optimistic) | 27 | 20 – 3 (87%) |
| best one per class | 24 | 19 – 1 |

## Takeaways
- Better **candidates** move the needle where better **picking** did not: D2's learned grader won 48% on holdout Solstice cells.
- Each template comes from one real ad, so some don't transfer, for example plans without a logo slot. Clustering plus a holdout gate is what filters those out.
- **Next:**
  1. Fresh-holdout confirmation of the 5 shipped templates (new ads).
  2. A second B2 round for the strip and wide shapes.
  3. Re-run the matrix and contact sheets with the 8 templates.
