# A/B: bisect2 (A) vs b2_nearfit (B)

**Verdict: b2_nearfit is better, but not significant than bisect2: 11 to 4 (0 unsure), sign test p=0.118, on 15 judged changed cells.**

- Cells compared: 192
- Identical winners: 177 (92%)
- Changed winners: 15 (8%); judged 15, not yet judged 0

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | bisect2 wins (A) | b2_nearfit wins (B) | tie / unsure |
|---|---|---|---|
| all | 4 | 11 | 0 |
| offbrand | 2 | 4 | 0 |
| solstice | 2 | 7 | 0 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.50** (worst slice: aspect:square; 4 slices qualify)

Overall B win share: 0.73. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | bisect2 wins (A) | b2_nearfit wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 0 | 1 | 0 | 1.00 |
| square | 3 | 3 | 0 | 0.50 |
| tall | 1 | 4 | 0 | 0.80 |
| wide | 0 | 3 | 0 | 1.00 |

By brand family:

| | bisect2 wins (A) | b2_nearfit wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 2 | 4 | 0 | 0.67 |
| solstice | 2 | 7 | 0 | 0.78 |

Mean ship-readiness score (1-5) on changed cells:

- A (bisect2): 2.22
- B (b2_nearfit): 3.17

Mean engine seconds per cell (all compared cells):

- A (bisect2): 0.70
- B (b2_nearfit): 0.79
