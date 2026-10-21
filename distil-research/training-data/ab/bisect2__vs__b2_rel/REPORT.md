# A/B: bisect2 (A) vs b2_rel (B)

**Verdict: b2_rel is better, but not significant than bisect2: 28 to 26 (6 unsure), sign test p=0.892, on 60 judged changed cells.**

- Cells compared: 192
- Identical winners: 120 (62%)
- Changed winners: 72 (38%); judged 60, not yet judged 12

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | bisect2 wins (A) | b2_rel wins (B) | tie / unsure |
|---|---|---|---|
| all | 26 | 28 | 6 |
| offbrand | 9 | 13 | 3 |
| solstice | 17 | 15 | 3 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.20** (worst slice: aspect:landscape; 8 slices qualify)

Overall B win share: 0.52. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | bisect2 wins (A) | b2_rel wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 4 | 1 | 2 | 0.20 |
| square | 7 | 2 | 1 | 0.22 |
| strip | 3 | 6 | 1 | 0.67 |
| tall | 5 | 7 | 0 | 0.58 |
| very_tall | 1 | 4 | 1 | 0.80 |
| wide | 6 | 8 | 1 | 0.57 |

By brand family:

| | bisect2 wins (A) | b2_rel wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 9 | 13 | 3 | 0.59 |
| solstice | 17 | 15 | 3 | 0.47 |

Mean ship-readiness score (1-5) on changed cells:

- A (bisect2): 2.29
- B (b2_rel): 2.32

Mean engine seconds per cell (all compared cells):

- A (bisect2): 0.70
- B (b2_rel): 0.79
