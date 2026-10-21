# Picker study: pk_v4_train

Split `train`, engine env `{'CHAMELEON_STRATEGIES': 'bisection,anchored', 'CHAMELEON_BISECT_V2': '1', 'CHAMELEON_HIER_LOCK': '1', 'CHAMELEON_NEAR_FIT': '1', 'CHAMELEON_LAYERS': '1'}`, K = 4 (at most 2 per strategy family, fallback included when it passes the gates).

- Cells prepared: 100 (1 skipped: fewer than 2 gate-passing candidates)
- Cells ranked by the teacher: 99

**Grader's pick = teacher's top choice: 27 of 99 (27%; chance 25%).**

- Teacher's mean score (1-5) of the grader's pick: 2.59
- Teacher's mean score of its own best candidate: 3.08
- **Headroom** (best - pick, mean per cell): 0.49; cells where a better candidate scored higher: 45
- Grader's pick: mean teacher rank 2.16 of K
- Mean Kendall tau, grader order vs teacher ranking: 0.29 (1 = same order, 0 = unrelated)
- Fallback shown in 99 cells, teacher's top in 12
- Rankings whose order disagrees with their own scores: 0

Strategy families (by how often the teacher ranks them first when shown):

| family | shown | teacher's top | top share | grader's pick | mean score | mean rank |
|---|---|---|---|---|---|---|
| columns | 1 | 1 | 1.00 | 0 | 4.00 | 1.00 |
| text_first | 4 | 2 | 0.50 | 0 | 2.75 | 2.25 |
| story_centered | 2 | 1 | 0.50 | 0 | 3.00 | 1.50 |
| b2_tall_3 | 4 | 2 | 0.50 | 0 | 3.00 | 1.75 |
| anchored | 75 | 33 | 0.44 | 19 | 2.77 | 1.81 |
| split_hero_left | 3 | 1 | 0.33 | 1 | 2.33 | 2.33 |
| b2_square_3 | 9 | 3 | 0.33 | 2 | 2.56 | 2.22 |
| b2_very_tall_1 | 3 | 1 | 0.33 | 0 | 2.00 | 2.33 |
| row | 34 | 10 | 0.29 | 13 | 2.44 | 2.03 |
| reflow | 4 | 1 | 0.25 | 1 | 3.25 | 2.25 |
| bisection | 140 | 31 | 0.22 | 50 | 2.43 | 2.40 |
| fallback | 99 | 12 | 0.12 | 12 | 1.80 | 3.32 |
| stack | 17 | 1 | 0.06 | 1 | 2.12 | 3.12 |
