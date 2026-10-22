# A/B: v5 (A) vs llm_student_gguf (B)

**Verdict: v5 is better, but not significant than llm_student_gguf: 16 to 6 (0 unsure), sign test p=0.052, on 22 judged changed cells.**

- Cells compared: 30
- Identical winners: 8 (27%)
- Changed winners: 22 (73%); judged 22, not yet judged 0

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | v5 wins (A) | llm_student_gguf wins (B) | tie / unsure |
|---|---|---|---|
| all | 16 | 6 | 0 |
| offbrand | 4 | 3 | 0 |
| solstice | 12 | 3 | 0 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.14** (worst slice: aspect:wide; 4 slices qualify)

Overall B win share: 0.27. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | v5 wins (A) | llm_student_gguf wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 0 | 2 | 0 | 1.00 |
| square | 3 | 0 | 0 | 0.00 |
| strip | 1 | 1 | 0 | 0.50 |
| tall | 4 | 1 | 0 | 0.20 |
| very_tall | 2 | 1 | 0 | 0.33 |
| wide | 6 | 1 | 0 | 0.14 |

By brand family:

| | v5 wins (A) | llm_student_gguf wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 4 | 3 | 0 | 0.43 |
| solstice | 12 | 3 | 0 | 0.20 |

Mean ship-readiness score (1-5) on changed cells:

- A (v5): 2.82
- B (llm_student_gguf): 2.09

Mean engine seconds per cell (all compared cells):

- A (v5): 0.67
- B (llm_student_gguf): 0.67
