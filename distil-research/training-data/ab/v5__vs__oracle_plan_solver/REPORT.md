# A/B: v5 (A) vs oracle_plan_solver (B)

**Verdict: oracle_plan_solver is better, but not significant than v5: 17 to 9 (1 unsure), sign test p=0.169, on 27 judged changed cells.**

- Cells compared: 31
- Identical winners: 4 (13%)
- Changed winners: 27 (87%); judged 27, not yet judged 0

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | v5 wins (A) | oracle_plan_solver wins (B) | tie / unsure |
|---|---|---|---|
| all | 9 | 17 | 1 |
| offbrand | 5 | 8 | 0 |
| solstice | 4 | 9 | 1 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.50** (worst slice: aspect:tall; 5 slices qualify)

Overall B win share: 0.65. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | v5 wins (A) | oracle_plan_solver wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 0 | 3 | 0 | 1.00 |
| square | 2 | 3 | 0 | 0.60 |
| strip | 2 | 1 | 0 | 0.33 |
| tall | 3 | 3 | 1 | 0.50 |
| very_tall | 1 | 2 | 0 | 0.67 |
| wide | 1 | 5 | 0 | 0.83 |

By brand family:

| | v5 wins (A) | oracle_plan_solver wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 5 | 8 | 0 | 0.62 |
| solstice | 4 | 9 | 1 | 0.69 |

Mean ship-readiness score (1-5) on changed cells:

- A (v5): 2.41
- B (oracle_plan_solver): 2.56

Mean engine seconds per cell (all compared cells):

- A (v5): 0.60
- B (oracle_plan_solver): 0.60
