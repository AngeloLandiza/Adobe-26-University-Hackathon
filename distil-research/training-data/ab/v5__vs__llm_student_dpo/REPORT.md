# A/B: v5 (A) vs llm_student_dpo (B)

**Verdict: v5 is better, but not significant than llm_student_dpo: 15 to 7 (2 unsure), sign test p=0.134, on 24 judged changed cells.**

- Cells compared: 30
- Identical winners: 6 (20%)
- Changed winners: 24 (80%); judged 24, not yet judged 0

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | v5 wins (A) | llm_student_dpo wins (B) | tie / unsure |
|---|---|---|---|
| all | 15 | 7 | 2 |
| offbrand | 5 | 2 | 1 |
| solstice | 10 | 5 | 1 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.29** (worst slice: family:offbrand; 3 slices qualify)

Overall B win share: 0.32. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | v5 wins (A) | llm_student_dpo wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 0 | 2 | 0 | 1.00 |
| square | 3 | 0 | 1 | 0.00 |
| strip | 2 | 2 | 0 | 0.50 |
| tall | 4 | 0 | 1 | 0.00 |
| very_tall | 2 | 1 | 0 | 0.33 |
| wide | 4 | 2 | 0 | 0.33 |

By brand family:

| | v5 wins (A) | llm_student_dpo wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 5 | 2 | 1 | 0.29 |
| solstice | 10 | 5 | 1 | 0.33 |

Mean ship-readiness score (1-5) on changed cells:

- A (v5): 2.67
- B (llm_student_dpo): 2.27

Mean engine seconds per cell (all compared cells):

- A (v5): 0.67
- B (llm_student_dpo): 0.67
