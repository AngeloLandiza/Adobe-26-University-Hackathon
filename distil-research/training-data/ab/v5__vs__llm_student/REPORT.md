# A/B: v5 (A) vs llm_student (B)

**Verdict: v5 is significantly better than llm_student: 18 to 3 (2 unsure), sign test p=0.001, on 23 judged changed cells.**

- Cells compared: 30
- Identical winners: 7 (23%)
- Changed winners: 23 (77%); judged 23, not yet judged 0

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | v5 wins (A) | llm_student wins (B) | tie / unsure |
|---|---|---|---|
| all | 18 | 3 | 2 |
| offbrand | 7 | 0 | 0 |
| solstice | 11 | 3 | 2 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.00** (worst slice: aspect:tall; 4 slices qualify)

Overall B win share: 0.14. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | v5 wins (A) | llm_student wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 0 | 1 | 1 | 1.00 |
| square | 3 | 0 | 0 | 0.00 |
| strip | 0 | 2 | 0 | 1.00 |
| tall | 5 | 0 | 0 | 0.00 |
| very_tall | 3 | 0 | 0 | 0.00 |
| wide | 7 | 0 | 1 | 0.00 |

By brand family:

| | v5 wins (A) | llm_student wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 7 | 0 | 0 | 0.00 |
| solstice | 11 | 3 | 2 | 0.21 |

Mean ship-readiness score (1-5) on changed cells:

- A (v5): 2.91
- B (llm_student): 2.04

Mean engine seconds per cell (all compared cells):

- A (v5): 0.67
- B (llm_student): 0.67
