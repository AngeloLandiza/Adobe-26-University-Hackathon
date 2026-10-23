Calibration on 40 layouts (gold = opus-default-768-K1).
Rule: cheapest candidate whose mean agreement with the two gold runs (Spearman on overall, mean per-criterion Spearman, worst-quartile overlap) is at least the 10th percentile of the gold run-vs-run agreement over 500 bootstrap resamples of the layouts.
Gold retest lower bounds: spearman 0.91, criteria 0.87, worst_q 0.70, mae 0.28

| config | $/layout | s/call | Spearman overall | MAE overall | bias | mean criterion Spearman | worst-quartile overlap | per criterion Spearman | qualifies |
|---|---|---|---|---|---|---|---|---|---|
| opus-default-768-K1 (rep0 vs rep1) | 0.0415 | 5.12 | 0.94 | 0.20 | 0.00 | 0.91 | 0.90 | coll 0.85 read 0.95 on_b 0.91 inte 0.93 adap 0.89 |  |
| sonnet-low-512-K4 | 0.0067 | 4.59 | 0.82 | 0.70 | 0.07 | 0.73 | 0.70 | coll 0.56 read 0.76 on_b 0.84 inte 0.75 adap 0.73 | no |
| sonnet-low-512-K6 | 0.0052 | 5.79 | 0.76 | 0.85 | -0.05 | 0.75 | 0.75 | coll 0.71 read 0.71 on_b 0.78 inte 0.80 adap 0.74 | no |
| sonnet-low-512-K1 | 0.0165 | 2.03 | 0.86 | 0.50 | -0.30 | 0.77 | 0.85 | coll 0.67 read 0.77 on_b 0.88 inte 0.82 adap 0.71 | no |
| sonnet-low-768-K4 | 0.0067 | 4.68 | 0.83 | 0.70 | -0.25 | 0.75 | 0.80 | coll 0.66 read 0.74 on_b 0.82 inte 0.80 adap 0.75 | no |
| sonnet-medium-512-K4 | 0.0060 | 4.61 | 0.83 | 0.62 | 0.00 | 0.76 | 0.70 | coll 0.61 read 0.75 on_b 0.85 inte 0.79 adap 0.77 | no |
| sonnet-low-768-K1 | 0.0184 | 2.16 | 0.84 | 0.68 | -0.47 | 0.75 | 0.75 | coll 0.68 read 0.78 on_b 0.86 inte 0.76 adap 0.69 | no |
| opus-default-512-K4 | 0.0145 | 11.27 | 0.80 | 0.75 | 0.25 | 0.80 | 0.70 | coll 0.78 read 0.85 on_b 0.80 inte 0.81 adap 0.76 | no |
| opus-default-768-K4 | 0.0154 | 9.64 | 0.82 | 0.50 | 0.07 | 0.84 | 0.90 | coll 0.76 read 0.89 on_b 0.88 inte 0.83 adap 0.86 | no |

Chosen default: none qualifies (keep the gold config)