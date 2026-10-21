# Picker study: smoke_pk

Split `holdout`, engine env `{'CHAMELEON_STRATEGIES': 'bisection', 'CHAMELEON_BISECT_V2': '1', 'CHAMELEON_HIER_LOCK': '1', 'CHAMELEON_NEAR_FIT': '1', 'CHAMELEON_LAYERS': '1'}`, K = 4 (at most 2 per strategy family, fallback included when it passes the gates).

- Cells prepared: 2 (0 skipped: fewer than 2 gate-passing candidates)
- Cells ranked by the teacher: 2

**Grader's pick = teacher's top choice: 1 of 2 (50%; chance 29%).**

- Teacher's mean score (1-5) of the grader's pick: 3.00
- Teacher's mean score of its own best candidate: 3.00
- **Headroom** (best - pick, mean per cell): 0.00; cells where a better candidate scored higher: 0
- Grader's pick: mean teacher rank 1.50 of K
- Mean Kendall tau, grader order vs teacher ranking: 0.67 (1 = same order, 0 = unrelated)
- Fallback shown in 2 cells, teacher's top in 0
- Rankings whose order disagrees with their own scores: 0

Strategy families (by how often the teacher ranks them first when shown):

| family | shown | teacher's top | top share | grader's pick | mean score | mean rank |
|---|---|---|---|---|---|---|
| bisection | 4 | 2 | 0.50 | 2 | 2.75 | 1.50 |
| row | 1 | 0 | 0.00 | 0 | 2.00 | 3.00 |
| fallback | 2 | 0 | 0.00 | 0 | 1.00 | 3.50 |
