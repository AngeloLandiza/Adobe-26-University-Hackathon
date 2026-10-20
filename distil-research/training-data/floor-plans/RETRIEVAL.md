# Retrieval (C) vs clustered templates (B2), holdout

For each holdout cell of a tall / very_tall / square target, the plans of the k nearest training ads (by `ad_vector`) were built by the engine; the best by the default grader was judged against the cell's current winner with the harsh rubric. win = every verdict picks the retrieved build over the cell's current winner with confidence >= 2; best of the recommended plans by the default grader.

## Retrieval

| class | cells | built | judged | wins | win rate | picks changed | new pick wins - loses |
|---|---|---|---|---|---|---|---|
| square | 29 | 29 | 29 | 17 | 59% | 19 | 14 - 2 |
| tall | 30 | 30 | 30 | 12 | 40% | 14 | 8 - 4 |
| very_tall | 26 | 26 | 26 | 16 | 62% | 15 | 7 - 5 |
| all | 85 | 85 | 85 | 45 | 53% | 48 | 29 - 11 |

## Clustered templates on the same cells

- 85 of the 85 cells have clustered-template builds.
- Per (template, cell) pair: 104 of 399 judged won (26%). One plan per cell is not the same unit as retrieval's best of k, so read this as the cost of a generic template, not a like-for-like win rate.
- Final-pick replay (all 14 templates added): 37 picks changed of 85, new pick wins - loses 24 - 9.

## Comparison

| | picks changed | wins - loses | judged win rate |
|---|---|---|---|
| retrieval (k plans, best by grader) | 48 | 29 - 11 | 53% |
| clustered templates (final pick) | 37 | 24 - 9 | 26% (per pair) |

Caveat: the teacher's verdicts exist only for what was judged; cells where the grader would keep the current winner contribute no change to the final pick either way.
