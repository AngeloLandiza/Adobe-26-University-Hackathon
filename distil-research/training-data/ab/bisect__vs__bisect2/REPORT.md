# A/B: bisect (A) vs bisect2 (B)

**Verdict: bisect2 is significantly better than bisect: 33 to 9 (18 unsure), sign test p=0.000, on 60 judged changed cells.**

- Cells compared: 192
- Identical winners: 27 (14%)
- Changed winners: 165 (86%); judged 60, not yet judged 105

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | bisect wins (A) | bisect2 wins (B) | tie / unsure |
|---|---|---|---|
| all | 9 | 33 | 18 |
| offbrand | 5 | 14 | 6 |
| solstice | 4 | 19 | 12 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.60** (worst slice: aspect:tall; 7 slices qualify)

Overall B win share: 0.79. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | bisect wins (A) | bisect2 wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 0 | 2 | 0 | 1.00 |
| square | 1 | 6 | 2 | 0.86 |
| strip | 4 | 8 | 2 | 0.67 |
| tall | 2 | 3 | 7 | 0.60 |
| very_tall | 2 | 4 | 1 | 0.67 |
| wide | 0 | 10 | 6 | 1.00 |

By brand family:

| | bisect wins (A) | bisect2 wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 5 | 14 | 6 | 0.74 |
| solstice | 4 | 19 | 12 | 0.83 |

Mean ship-readiness score (1-5) on changed cells:

- A (bisect): 2.31
- B (bisect2): 2.54

Mean engine seconds per cell (all compared cells):

- A (bisect): 0.95
- B (bisect2): 0.70
