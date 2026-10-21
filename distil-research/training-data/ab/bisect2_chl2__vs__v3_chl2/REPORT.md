# A/B: bisect2_chl2 (A) vs v3_chl2 (B)

**Verdict: v3_chl2 is significantly better than bisect2_chl2: 25 to 9 (46 unsure), sign test p=0.009, on 80 judged changed cells.**

- Cells compared: 623
- Identical winners: 382 (61%)
- Changed winners: 241 (39%); judged 80, not yet judged 161

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | bisect2_chl2 wins (A) | v3_chl2 wins (B) | tie / unsure |
|---|---|---|---|
| all | 9 | 25 | 46 |
| crello | 9 | 25 | 46 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.58** (worst slice: aspect:wide; 4 slices qualify)

Overall B win share: 0.74. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | bisect2_chl2 wins (A) | v3_chl2 wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 0 | 1 | 2 | 1.00 |
| square | 1 | 2 | 7 | 0.67 |
| strip | 0 | 9 | 14 | 1.00 |
| tall | 1 | 3 | 5 | 0.75 |
| very_tall | 2 | 3 | 5 | 0.60 |
| wide | 5 | 7 | 13 | 0.58 |

By brand family:

| | bisect2_chl2 wins (A) | v3_chl2 wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| crello | 9 | 25 | 46 | 0.74 |

Mean ship-readiness score (1-5) on changed cells:

- A (bisect2_chl2): 1.73
- B (v3_chl2): 1.93

Mean engine seconds per cell (all compared cells):

- A (bisect2_chl2): 0.66
- B (v3_chl2): 0.73
