# A/B: v5 (A) vs llm_student8b_v2 (B)

**Verdict: NO CLEAR DIFFERENCE: v5 8 - llm_student8b_v2 8 (4 unsure) on 20 judged changed cells.**

- Cells compared: 30
- Identical winners: 10 (33%)
- Changed winners: 20 (67%); judged 20, not yet judged 0

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | v5 wins (A) | llm_student8b_v2 wins (B) | tie / unsure |
|---|---|---|---|
| all | 8 | 8 | 4 |
| offbrand | 3 | 2 | 1 |
| solstice | 5 | 6 | 3 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.40** (worst slice: family:offbrand; 3 slices qualify)

Overall B win share: 0.50. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | v5 wins (A) | llm_student8b_v2 wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 0 | 1 | 0 | 1.00 |
| square | 1 | 1 | 0 | 0.50 |
| strip | 0 | 2 | 0 | 1.00 |
| tall | 4 | 0 | 1 | 0.00 |
| very_tall | 1 | 1 | 1 | 0.50 |
| wide | 2 | 3 | 2 | 0.60 |

By brand family:

| | v5 wins (A) | llm_student8b_v2 wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 3 | 2 | 1 | 0.40 |
| solstice | 5 | 6 | 3 | 0.55 |

Mean ship-readiness score (1-5) on changed cells:

- A (v5): 2.70
- B (llm_student8b_v2): 2.58

Mean engine seconds per cell (all compared cells):

- A (v5): 0.67
- B (llm_student8b_v2): 0.67
