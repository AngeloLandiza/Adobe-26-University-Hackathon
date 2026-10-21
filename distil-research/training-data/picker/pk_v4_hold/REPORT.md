# Picker study: pk_v4_hold

Split `holdout`, engine env `{'CHAMELEON_STRATEGIES': 'bisection,anchored', 'CHAMELEON_BISECT_V2': '1', 'CHAMELEON_HIER_LOCK': '1', 'CHAMELEON_LAYERS': '1', 'CHAMELEON_NEAR_FIT': '1'}`, K = 4 (at most 2 per strategy family, fallback included when it passes the gates).

- Cells prepared: 30 (0 skipped: fewer than 2 gate-passing candidates)
- Cells ranked by the teacher: 30

**Grader's pick = teacher's top choice: 10 of 30 (33%; chance 25%).**

- Teacher's mean score (1-5) of the grader's pick: 2.73
- Teacher's mean score of its own best candidate: 3.10
- **Headroom** (best - pick, mean per cell): 0.37; cells where a better candidate scored higher: 10
- Grader's pick: mean teacher rank 2.00 of K
- Mean Kendall tau, grader order vs teacher ranking: 0.41 (1 = same order, 0 = unrelated)
- Fallback shown in 30 cells, teacher's top in 4
- Rankings whose order disagrees with their own scores: 0

Strategy families (by how often the teacher ranks them first when shown):

| family | shown | teacher's top | top share | grader's pick | mean score | mean rank |
|---|---|---|---|---|---|---|
| anchored | 27 | 11 | 0.41 | 11 | 2.67 | 1.89 |
| stack | 6 | 2 | 0.33 | 1 | 2.67 | 1.83 |
| bisection | 46 | 13 | 0.28 | 12 | 2.50 | 2.33 |
| fallback | 30 | 4 | 0.13 | 2 | 1.67 | 3.33 |
| split_hero_left | 1 | 0 | 0.00 | 0 | 1.00 | 4.00 |
| text_first | 1 | 0 | 0.00 | 0 | 1.00 | 4.00 |
| row | 9 | 0 | 0.00 | 4 | 2.11 | 2.56 |
