# A/B: v5 (A) vs llm_sonnet (B)

**Verdict: llm_sonnet is significantly better than v5: 11 to 1 (2 unsure), sign test p=0.006, on 14 judged changed cells.**

- Cells compared: 20
- Identical winners: 6 (30%)
- Changed winners: 14 (70%); judged 14, not yet judged 0

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | v5 wins (A) | llm_sonnet wins (B) | tie / unsure |
|---|---|---|---|
| all | 1 | 11 | 2 |
| offbrand | 1 | 3 | 0 |
| solstice | 0 | 8 | 2 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 1.00** (worst slice: family:solstice; 1 slices qualify)

Overall B win share: 0.92. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | v5 wins (A) | llm_sonnet wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 0 | 1 | 0 | 1.00 |
| square | 0 | 1 | 1 | 1.00 |
| strip | 0 | 3 | 0 | 1.00 |
| tall | 1 | 2 | 0 | 0.67 |
| very_tall | 0 | 2 | 0 | 1.00 |
| wide | 0 | 2 | 1 | 1.00 |

By brand family:

| | v5 wins (A) | llm_sonnet wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 1 | 3 | 0 | 0.75 |
| solstice | 0 | 8 | 2 | 1.00 |

Mean ship-readiness score (1-5) on changed cells:

- A (v5): 2.29
- B (llm_sonnet): 3.39

Mean engine seconds per cell (all compared cells):

- A (v5): 0.68
- B (llm_sonnet): 0.68
