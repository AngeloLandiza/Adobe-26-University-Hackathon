# A/B: v5 (A) vs llm_student8b_rep (B)

**Verdict: v5 is better, but not significant than llm_student8b_rep: 12 to 6 (1 unsure), sign test p=0.238, on 19 judged changed cells.**

- Cells compared: 30
- Identical winners: 11 (37%)
- Changed winners: 19 (63%); judged 19, not yet judged 0

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | v5 wins (A) | llm_student8b_rep wins (B) | tie / unsure |
|---|---|---|---|
| all | 12 | 6 | 1 |
| offbrand | 4 | 1 | 1 |
| solstice | 8 | 5 | 0 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.20** (worst slice: aspect:wide; 3 slices qualify)

Overall B win share: 0.33. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | v5 wins (A) | llm_student8b_rep wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 1 | 1 | 0 | 0.50 |
| square | 1 | 1 | 0 | 0.50 |
| strip | 0 | 2 | 0 | 1.00 |
| tall | 3 | 1 | 1 | 0.25 |
| very_tall | 3 | 0 | 0 | 0.00 |
| wide | 4 | 1 | 0 | 0.20 |

By brand family:

| | v5 wins (A) | llm_student8b_rep wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 4 | 1 | 1 | 0.20 |
| solstice | 8 | 5 | 0 | 0.38 |

Mean ship-readiness score (1-5) on changed cells:

- A (v5): 2.82
- B (llm_student8b_rep): 2.24

Mean engine seconds per cell (all compared cells):

- A (v5): 0.67
- B (llm_student8b_rep): 0.67
