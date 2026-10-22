# A/B: v5 (A) vs llm_student_v2_rep (B)

**Verdict: v5 is better, but not significant than llm_student_v2_rep: 14 to 10 (1 unsure), sign test p=0.541, on 25 judged changed cells.**

- Cells compared: 30
- Identical winners: 5 (17%)
- Changed winners: 25 (83%); judged 25, not yet judged 0

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | v5 wins (A) | llm_student_v2_rep wins (B) | tie / unsure |
|---|---|---|---|
| all | 14 | 10 | 1 |
| offbrand | 5 | 3 | 0 |
| solstice | 9 | 7 | 1 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.20** (worst slice: aspect:tall; 4 slices qualify)

Overall B win share: 0.42. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | v5 wins (A) | llm_student_v2_rep wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 0 | 1 | 0 | 1.00 |
| square | 2 | 1 | 1 | 0.33 |
| strip | 2 | 2 | 0 | 0.50 |
| tall | 4 | 1 | 0 | 0.20 |
| very_tall | 2 | 1 | 0 | 0.33 |
| wide | 4 | 4 | 0 | 0.50 |

By brand family:

| | v5 wins (A) | llm_student_v2_rep wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 5 | 3 | 0 | 0.38 |
| solstice | 9 | 7 | 1 | 0.44 |

Mean ship-readiness score (1-5) on changed cells:

- A (v5): 2.58
- B (llm_student_v2_rep): 2.48

Mean engine seconds per cell (all compared cells):

- A (v5): 0.67
- B (llm_student_v2_rep): 0.67
