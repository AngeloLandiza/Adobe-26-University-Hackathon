# Grader training report

teachers: claude-code:claude-opus-5-5: 488
filters: rubric f80dbb11, confidence >= 2; dropped {'teacher': 0, 'low_confidence': 227, 'rubric_version': 0}
examples: train 326, holdout 135, ties dropped 27, skipped (candidate missing from pool) 0
| accuracy | train | holdout |
|---|---|---|
| default | 0.635 | 0.704 |
| learned | 0.684 | 0.778 |

| holdout brand | examples | default | learned |
|---|---|---|---|
| solstice | 83 | 0.735 | 0.759 |
| offbrand | 52 | 0.654 | 0.808 |

holdout gain: +0.074 (needs >= 0.03)

| metric | default | learned |
|---|---|---|
| geometry.hierarchy | 0.200 | 0.000 |
| geometry.legibility | 0.200 | 0.178 |
| geometry.overlap | 0.150 | 0.000 |
| geometry.degradation | 0.100 | 0.174 |
| geometry.faithfulness | 0.080 | 0.333 |
| geometry.space_use | 0.080 | 0.089 |
| geometry.alignment | 0.070 | 0.046 |
| geometry.spacing | 0.070 | 0.133 |
| geometry.balance | 0.050 | 0.047 |
| pixels.text_contrast | 0.600 | 0.457 |
| pixels.busyness | 0.250 | 0.534 |
| pixels.placeholder | 0.150 | 0.009 |
| geometry_share | 0.700 | 0.920 |
