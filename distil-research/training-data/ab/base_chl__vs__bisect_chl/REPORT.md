# A/B: base_chl (A) vs bisect_chl (B)

**Verdict: bisect_chl is significantly better than base_chl: 57 to 37 (6 unsure), sign test p=0.049, on 100 judged changed cells.**

- Cells compared: 499
- Identical winners: 128 (26%)
- Changed winners: 371 (74%); judged 100, not yet judged 271

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | base_chl wins (A) | bisect_chl wins (B) | tie / unsure |
|---|---|---|---|
| all | 37 | 57 | 6 |
| challenge | 37 | 57 | 6 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.33** (worst slice: aspect:landscape; 7 slices qualify)

Overall B win share: 0.61. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | base_chl wins (A) | bisect_chl wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 4 | 2 | 1 | 0.33 |
| square | 7 | 7 | 1 | 0.50 |
| strip | 5 | 15 | 2 | 0.75 |
| tall | 6 | 5 | 0 | 0.45 |
| very_tall | 6 | 12 | 1 | 0.67 |
| wide | 9 | 16 | 1 | 0.64 |

By brand family:

| | base_chl wins (A) | bisect_chl wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| challenge | 37 | 57 | 6 | 0.61 |

Mean ship-readiness score (1-5) on changed cells:

- A (base_chl): 2.14
- B (bisect_chl): 2.18

Mean engine seconds per cell (all compared cells):

- A (base_chl): 0.42
- B (bisect_chl): 2.10
