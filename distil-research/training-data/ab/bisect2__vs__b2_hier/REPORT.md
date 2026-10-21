# A/B: bisect2 (A) vs b2_hier (B)

**Verdict: b2_hier is significantly better than bisect2: 14 to 4 (22 unsure), sign test p=0.031, on 40 judged changed cells.**

- Cells compared: 192
- Identical winners: 92 (48%)
- Changed winners: 100 (52%); judged 40, not yet judged 60

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | bisect2 wins (A) | b2_hier wins (B) | tie / unsure |
|---|---|---|---|
| all | 4 | 14 | 22 |
| offbrand | 3 | 6 | 10 |
| solstice | 1 | 8 | 12 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.67** (worst slice: family:offbrand; 2 slices qualify)

Overall B win share: 0.78. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | bisect2 wins (A) | b2_hier wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 1 | 1 | 2 | 0.50 |
| square | 1 | 3 | 2 | 0.75 |
| strip | 2 | 2 | 5 | 0.50 |
| tall | 0 | 1 | 2 | 1.00 |
| very_tall | 0 | 3 | 3 | 1.00 |
| wide | 0 | 4 | 8 | 1.00 |

By brand family:

| | bisect2 wins (A) | b2_hier wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 3 | 6 | 10 | 0.67 |
| solstice | 1 | 8 | 12 | 0.89 |

Mean ship-readiness score (1-5) on changed cells:

- A (bisect2): 2.27
- B (b2_hier): 2.50

Mean engine seconds per cell (all compared cells):

- A (bisect2): 0.70
- B (b2_hier): 0.75
