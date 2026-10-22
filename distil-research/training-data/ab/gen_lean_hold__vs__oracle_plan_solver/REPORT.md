# A/B: gen_lean_hold (A) vs oracle_plan_solver (B)

**Verdict: gen_lean_hold is significantly better than oracle_plan_solver: 20 to 7 (2 unsure), sign test p=0.019, on 29 judged changed cells.**

- Cells compared: 31
- Identical winners: 2 (6%)
- Changed winners: 29 (94%); judged 29, not yet judged 0

Teacher verdicts (confidence >= 2; per cell, majority of original and swapped asks):

| | gen_lean_hold wins (A) | oracle_plan_solver wins (B) | tie / unsure |
|---|---|---|---|
| all | 20 | 7 | 2 |
| offbrand | 12 | 2 | 1 |
| solstice | 8 | 5 | 1 |

**Generalisation score (B's worst-slice win share, slices with >= 5 decided cells): 0.00** (worst slice: aspect:strip; 6 slices qualify)

Overall B win share: 0.26. (Win share = B wins / decided cells; 0.50 is a coin flip, so a generalisation score below 0.50 means B loses somewhere.)

By target aspect class:

| | gen_lean_hold wins (A) | oracle_plan_solver wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| landscape | 3 | 0 | 0 | 0.00 |
| square | 4 | 1 | 0 | 0.20 |
| strip | 5 | 0 | 0 | 0.00 |
| tall | 2 | 4 | 1 | 0.67 |
| very_tall | 3 | 0 | 0 | 0.00 |
| wide | 3 | 2 | 1 | 0.40 |

By brand family:

| | gen_lean_hold wins (A) | oracle_plan_solver wins (B) | tie / unsure | B share |
|---|---|---|---|---|
| offbrand | 12 | 2 | 1 | 0.14 |
| solstice | 8 | 5 | 1 | 0.38 |

Mean ship-readiness score (1-5) on changed cells:

- A (gen_lean_hold): 2.90
- B (oracle_plan_solver): 2.28

Mean engine seconds per cell (all compared cells):

- A (gen_lean_hold): 0.60
- B (oracle_plan_solver): 0.60
