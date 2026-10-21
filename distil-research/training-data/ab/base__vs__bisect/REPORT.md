# A/B: base (A) vs bisect (B)

**Verdict: bisect is significantly better than base: 38 to 17 (5 unsure), sign test p=0.006, on 60 judged changed cells.**

- Cells compared: 192
- Identical winners: 38 (20%)
- Changed winners: 154 (80%); judged 60, not yet judged 94

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | base wins (A) | bisect wins (B) | tie / unsure |
|---|---|---|---|
| all | 17 | 38 | 5 |
| offbrand | 7 | 15 | 2 |
| solstice | 10 | 23 | 3 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.57** (worst slice: aspect:tall; 6 slices qualify)

Overall B win share: 0.69. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | base wins (A) | bisect wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 1 | 3 | 1 | 0.75 |
| square | 3 | 7 | 1 | 0.70 |
| strip | 2 | 11 | 1 | 0.85 |
| tall | 3 | 4 | 0 | 0.57 |
| very_tall | 1 | 2 | 0 | 0.67 |
| wide | 7 | 11 | 2 | 0.61 |

By brand family:

| | base wins (A) | bisect wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 7 | 15 | 2 | 0.68 |
| solstice | 10 | 23 | 3 | 0.70 |

Mean ship-readiness score (1-5) on changed cells:

- A (base): 2.11
- B (bisect): 2.44

Mean engine seconds per cell (all compared cells):

- A (base): 0.45
- B (bisect): 0.95
