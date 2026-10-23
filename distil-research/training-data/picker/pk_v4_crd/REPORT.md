# Picker study: pk_v4_crd

Split `crello_dev`, engine env `{'CHAMELEON_STRATEGIES': 'bisection,anchored', 'CHAMELEON_BISECT_V2': '1', 'CHAMELEON_HIER_LOCK': '1', 'CHAMELEON_NEAR_FIT': '1', 'CHAMELEON_LAYERS': '1'}`, K = 4 (at most 2 per strategy family, fallback included when it passes the gates).

- Cells prepared: 60 (0 skipped: fewer than 2 gate-passing candidates)
- Cells ranked by the teacher: 60

**Grader's pick = teacher's top choice: 21 of 60 (35%; chance 26%).**

- Teacher's mean score (1-5) of the grader's pick: 2.28
- Teacher's mean score of its own best candidate: 2.75
- **Headroom** (best - pick, mean per cell): 0.47; cells where a better candidate scored higher: 25
- Grader's pick: mean teacher rank 2.12 of K
- Mean Kendall tau, grader order vs teacher ranking: 0.28 (1 = same order, 0 = unrelated)
- Fallback shown in 60 cells, teacher's top in 16
- Rankings whose order disagrees with their own scores: 0

Strategy families (by how often the teacher ranks them first when shown):

| family | shown | teacher's top | top share | grader's pick | mean score | mean rank |
|---|---|---|---|---|---|---|
| b2_square_3 | 1 | 1 | 1.00 | 0 | 3.00 | 1.00 |
| split_hero_left | 1 | 1 | 1.00 | 0 | 2.00 | 1.00 |
| anchored | 58 | 23 | 0.40 | 18 | 2.36 | 2.00 |
| fallback | 60 | 16 | 0.27 | 9 | 2.03 | 2.75 |
| stack | 9 | 2 | 0.22 | 3 | 1.67 | 2.44 |
| row | 11 | 2 | 0.18 | 1 | 2.00 | 2.55 |
| bisection | 81 | 14 | 0.17 | 27 | 2.00 | 2.62 |
| reflow | 8 | 1 | 0.12 | 1 | 2.62 | 2.25 |
| columns | 5 | 0 | 0.00 | 1 | 2.00 | 2.80 |
| b2_square_1 | 1 | 0 | 0.00 | 0 | 1.00 | 4.00 |
| b2_very_tall_1 | 1 | 0 | 0.00 | 0 | 2.00 | 3.00 |
