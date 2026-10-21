# Real designer ad layouts vs our engine

What real designers do at each of our 13 target sizes, measured on one shard of the public Crello dataset, and where today's engine (`base`) and bisection (`bisect`) differ. Only aggregates are committed; the raw shard stays in `training-data/references/crello.nosync/` (gitignored). Numbers: `training-data/references/crello_layout_stats.json`, `engine_layout_stats_base.json`, `engine_layout_stats_bisect.json`. Scripts: `chameleon/dev/references/`.

## Method

- Data: `cyberagent/crello` revision 5.1.0, `data/test-00000-of-00004.parquet` (493 templates; 411 in ad/social/promo formats after excluding resumes, invitations, cards, certificates, menus, logos, presentations, calendars and similar).
- Pool per target: ad-format templates whose canvas aspect is within 5% of the target. If fewer than 20 match, the pool is the 20 ad-format templates nearest in aspect, marked `nearest-aspect` below. **For the very wide targets (leaderboard, mobile_banner, large_leaderboard, billboard, linkedin_banner) the real pool is NOT the same shape as the target**; treat those rows as a rough guide only.
- Roles are labelled by simple rules (`dev/references/crello_roles.py`): background (full-bleed), headline (largest non-CTA text), CTA (short text inside a small button shape, or starting with an action verb), logo (small graphic near an edge, or short small text near an edge), product (largest remaining image of at least 5% of the canvas), body (other text), decoration (everything else).
- Positions are the element box centre as a fraction of the canvas; area share is the clipped box over canvas area; font size is a share of the canvas short side (headline: largest, others: median); edge margin is the gap between all non-background content (union box) and each canvas edge, as a share of the short side; empty space is the share of a 64x64 grid not covered by any non-background element box (boxes, not glyph pixels, so it under-states emptiness). Reading order: `reads before` = above by more than 0.05, or on the same row and left.
- Engine side: holdout cells of `ab/base` and `ab/bisect` (192 cells each) regenerated with the same code and environment as `ab_engine.py` (heuristic planner, no teacher), using the engine's own plan roles (hero_image->product, subheadline/other_text/legal/badge->body). The engine's roles are its own judgement, not my heuristic, so role-level gaps include labelling differences.

## Labelling accuracy (honest)

Checked by eye on 30 random ad-format templates (3 contact sheets, boxes drawn in role colours over the template preview; `crello.nosync/review/sheet1-3.png`, seed 0). Scored per role, counting a label as right if a human would accept it:

- **headline: about 24/30 (80%)**. Failures: the real headline taken as CTA because it starts with a verb ("Support"), the real headline being a second, lower-font line, and two cases where a date/price or a half of a split headline won on font size.
- **product / hero image: roughly 85% precision** (about 12 of 14 labelled). Errors: a pink placeholder rectangle and a collage tile. Recall is lower where the photo is full-bleed (it is then labelled background, by design).
- **background: reliable** (full-bleed photo or colour; no visible miss).
- **CTA: poor, about 25-30% precision** (roughly 4 of 14 labelled were real buttons). The verb rule fires on eyebrow lines ("Buy ...", "Get ..."), URLs in a box and date circles. Real CTA buttons are also genuinely rare in these templates, so CTA statistics are the least trustworthy: read them as "a short, boxed or verb-led text", not "a button".
- **logo: poor, about 30% precision**. Small dotted decorations and tiny edge text are labelled logo; true logos are often missed when they are photos inside a larger box.
- **body / decoration**: not scored individually; body is "other text", so it inherits headline errors (a second headline line becomes body).

Headline, product and background statistics are usable; CTA and logo statistics are indicative only. Accuracy was judged by one reader on 30 templates, so treat it as +/- 10 points.

## Matches per target

| target | size | pool | strict matches | used | aspect range used | top formats |
|---|---|---|---|---|---|---|
| instagram_post | 1080x1080 | strict | 63 | 63 | - | Instagram 40, Instagram AD 23 |
| instagram_portrait | 1080x1350 | nearest-aspect | 0 | 20 | [0.714, 0.75] | Flayer 13, Poster US 7 |
| story | 1080x1920 | strict | 41 | 41 | - | Instagram Story 41 |
| youtube_thumbnail | 1280x720 | strict | 41 | 41 | - | FB event cover 15, Youtube 12, Title 8 |
| linkedin_banner | 1584x396 | nearest-aspect | 8 | 20 | [3.0, 4.0] | Twitter 14, VK Community Cover 3, Email header 3 |
| facebook_cover | 820x312 | strict | 28 | 28 | - | Facebook cover 27, Twitch Profile Banner 1 |
| leaderboard | 728x90 | nearest-aspect | 7 | 20 | [3.0, 8.089] | Twitter 8, Leaderboard 7, VK Community Cover 3 |
| medium_rectangle | 300x250 | strict | 50 | 50 | - | Facebook 32, Large Rectangle 10, Medium Rectangle 8 |
| wide_skyscraper | 160x600 | nearest-aspect | 9 | 20 | [0.267, 0.562] | Graphic 10, Skyscraper 9, Instagram Story 1 |
| billboard | 970x250 | nearest-aspect | 8 | 20 | [3.0, 4.0] | Twitter 14, VK Community Cover 3, Email header 3 |
| mobile_banner | 320x50 | nearest-aspect | 0 | 20 | [3.0, 8.089] | Twitter 8, Leaderboard 7, VK Community Cover 3 |
| half_page | 300x600 | nearest-aspect | 10 | 20 | [0.5, 0.562] | Graphic 10, Instagram Story 10 |
| large_leaderboard | 970x90 | nearest-aspect | 0 | 20 | [3.0, 8.089] | Twitter 8, Leaderboard 7, VK Community Cover 3 |

## Per-size summaries (real designers)

### instagram_post (1080x1080) - strict pool, n=63

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.50 (0.41-0.51) | 0.47 (0.34-0.61) | 0.061 | 0.09 (0.06-0.13) |
| body | 0.82 | 0.50 (0.35-0.57) | 0.55 (0.39-0.79) | 0.051 | 0.04 (0.03-0.05) |
| cta | 0.48 | 0.50 (0.29-0.50) | 0.48 (0.30-0.66) | 0.075 | 0.05 (0.03-0.08) |
| logo | 0.51 | 0.35 (0.14-0.83) | 0.78 (0.31-0.86) | 0.016 | - |
| product | 0.44 | 0.50 (0.49-0.62) | 0.50 (0.48-0.54) | 0.483 | - |
| decoration | 0.91 | 0.50 (0.33-0.53) | 0.50 (0.47-0.70) | 0.421 | - |
| background | 0.76 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.04, top 0.05, right 0.00, bottom 0.01. Empty space 0.31. Reading order: headline vs cta: above 0.43, left 0.17, reads-before 0.43 (n=30); headline vs product: above 0.57, left 0.57, reads-before 0.64 (n=28); product vs cta: above 0.60, left 0.13, reads-before 0.60 (n=15); logo vs headline: above 0.38, left 0.47, reads-before 0.44 (n=32).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|                                                |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPHHHCCCCCCCCCCCCCHHHPPPPPPPP       |
|        PPPPPPHHHCCCCCCCCCCCCCHHHPPPPPPPP       |
|        PPPPPPHHHCCCCCCCCCCCCCHHHPPPPPPPP       |
|        PPPPPPPPbCCCCCCCCCCCCCbbbPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPLLLLLLPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPLLLLLLPPPPPPPPPPPPPPPPPPPPP       |
|                                                |
|                                                |
```

### instagram_portrait (1080x1350) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.50 (0.43-0.51) | 0.45 (0.31-0.64) | 0.083 | 0.12 (0.08-0.17) |
| body | 0.90 | 0.51 (0.49-0.53) | 0.52 (0.40-0.72) | 0.069 | 0.05 (0.03-0.06) |
| cta | 0.20 | 0.46 (0.37-0.52) | 0.55 (0.33-0.72) | 0.021 | 0.06 (0.05-0.06) |
| logo | 0.65 | 0.51 (0.50-0.77) | 0.16 (0.11-0.60) | 0.018 | - |
| product | 0.35 | 0.50 (0.44-0.50) | 0.50 (0.42-0.66) | 0.384 | - |
| decoration | 0.95 | 0.50 (0.50-0.50) | 0.51 (0.43-0.63) | 0.413 | - |
| background | 0.90 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.01; left 0.05, top 0.07, right 0.03, bottom 0.09. Empty space 0.35. Reading order: headline vs cta: above 0.75, left 0.00, reads-before 0.75 (n=4); headline vs product: above 0.71, left 0.14, reads-before 0.71 (n=7); product vs cta: above 0.00, left 0.50, reads-before 0.00 (n=2); logo vs headline: above 0.54, left 0.23, reads-before 0.61 (n=13).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|                     LLLLLLL                    |
|         PPPPPPPPPPPPLLLLLLLPPPPPPPPPPP         |
|         PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|         PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|         PPPPHHHHHHHHHHHHHHHHHHHHHHPPPP         |
|         PPPPHHHHHHHHHHHHHHHHHHHHHHPPPP         |
|         PPPPHHHHHHHHHHHHHHHHHHHHHHPPPP         |
|         PPPPPbbbbbbbbbbbbbbbbbbbbPPPPP         |
|         PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|         PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|         PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|                                                |
|                                                |
|                                                |
```

### story (1080x1920) - strict pool, n=41

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.98 | 0.50 (0.49-0.51) | 0.41 (0.24-0.52) | 0.083 | 0.16 (0.11-0.22) |
| body | 0.85 | 0.50 (0.39-0.51) | 0.36 (0.26-0.61) | 0.047 | 0.05 (0.04-0.08) |
| cta | 0.58 | 0.50 (0.49-0.56) | 0.79 (0.33-0.88) | 0.030 | 0.04 (0.03-0.05) |
| logo | 0.56 | 0.50 (0.34-0.62) | 0.14 (0.09-0.85) | 0.007 | - |
| product | 0.68 | 0.50 (0.50-0.51) | 0.50 (0.40-0.59) | 0.481 | - |
| decoration | 0.88 | 0.50 (0.50-0.55) | 0.50 (0.30-0.64) | 0.523 | - |
| background | 0.56 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.04, right 0.00, bottom 0.01. Empty space 0.21. Reading order: headline vs cta: above 0.65, left 0.17, reads-before 0.70 (n=23); headline vs product: above 0.59, left 0.22, reads-before 0.67 (n=27); product vs cta: above 0.71, left 0.23, reads-before 0.71 (n=17); logo vs headline: above 0.61, left 0.30, reads-before 0.61 (n=23).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|        PPPPPPPPPPPPPPLLLLPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPHHHHHHHHHHHHHHHHHHHHHHPPPPPP       |
|        PPPPPHHHHHHHHHHHHHHHHHHHHHHPPPPPP       |
|        PPPPPHHHHHHHHHHHHHHHHHHHHHHPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPCCCCCCCCPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPCCCCCCCCPPPPPPPPPPPPP       |
|                    CCCCCCCC                    |
|                                                |
|                                                |
```

### youtube_thumbnail (1280x720) - strict pool, n=41

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.95 | 0.50 (0.30-0.51) | 0.43 (0.36-0.50) | 0.088 | 0.13 (0.09-0.19) |
| body | 0.78 | 0.50 (0.29-0.66) | 0.59 (0.50-0.70) | 0.043 | 0.05 (0.03-0.07) |
| cta | 0.41 | 0.50 (0.29-0.60) | 0.50 (0.31-0.50) | 0.110 | 0.09 (0.05-0.11) |
| logo | 0.61 | 0.35 (0.16-0.85) | 0.24 (0.14-0.63) | 0.018 | - |
| product | 0.39 | 0.51 (0.39-0.62) | 0.50 (0.50-0.52) | 0.574 | - |
| decoration | 0.98 | 0.50 (0.23-0.50) | 0.50 (0.38-0.50) | 0.409 | - |
| background | 0.68 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.00, right 0.06, bottom 0.00. Empty space 0.39. Reading order: headline vs cta: above 0.40, left 0.27, reads-before 0.53 (n=15); headline vs product: above 0.60, left 0.53, reads-before 0.80 (n=15); product vs cta: above 0.12, left 0.50, reads-before 0.38 (n=8); logo vs headline: above 0.52, left 0.39, reads-before 0.52 (n=23).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPLLLLLLPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPLLLLLLPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPHHHHCCCCCCCCCCCCCCCCHHHPPPPPPP      |
|      PPPPPPHHHHCCCCCCCCCCCCCCCCHHHPPPPPPP      |
|      PPPPPPHHHHCCCCCCCCCCCCCCCCHHHPPPPPPP      |
|      PPPPPPPPPPCCCCCCCCCCCCCCCCPPPPPPPPPP      |
|      PPPPPPPPPPCCCCCCCCCCCCCCCCPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|                                                |
|                                                |
```

### linkedin_banner (1584x396) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.95 | 0.51 (0.43-0.69) | 0.50 (0.41-0.61) | 0.094 | 0.15 (0.12-0.20) |
| body | 0.85 | 0.50 (0.43-0.70) | 0.42 (0.28-0.68) | 0.032 | 0.07 (0.06-0.11) |
| cta | 0.20 | 0.74 (0.72-0.76) | 0.63 (0.52-0.78) | 0.055 | 0.09 (0.06-0.14) |
| logo | 0.55 | 0.54 (0.30-0.83) | 0.39 (0.15-0.78) | 0.007 | - |
| product | 0.35 | 0.31 (0.28-0.67) | 0.50 (0.50-0.50) | 0.532 | - |
| decoration | 1.00 | 0.50 (0.32-0.75) | 0.50 (0.47-0.50) | 0.408 | - |
| background | 0.85 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.10, top 0.05, right 0.12, bottom 0.01. Empty space 0.45. Reading order: headline vs cta: above 0.67, left 0.33, reads-before 0.67 (n=3); headline vs product: above 0.71, left 0.43, reads-before 0.71 (n=7); product vs cta: above 1.00, left 0.00, reads-before 1.00 (n=1); logo vs headline: above 0.46, left 0.27, reads-before 0.55 (n=11).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPHHHHHHHHHHHLLLLHHHHHHHHH           |
|PPPPPPPPPPPPPHHHHHHHHHHHHHHHHHHHHHHHH           |
|PPPPPPPPPPPPPHHHHHHHHHHHHHHHHHHHHHHHH           |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|                                                |
|                                                |
```

### facebook_cover (820x312) - strict pool, n=28

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.96 | 0.50 (0.40-0.68) | 0.48 (0.44-0.56) | 0.073 | 0.15 (0.08-0.22) |
| body | 0.71 | 0.51 (0.41-0.74) | 0.64 (0.49-0.74) | 0.029 | 0.06 (0.05-0.11) |
| cta | 0.39 | 0.50 (0.39-0.60) | 0.50 (0.50-0.56) | 0.290 | 0.14 (0.11-0.19) |
| logo | 0.29 | 0.14 (0.10-0.36) | 0.76 (0.56-0.88) | 0.012 | - |
| product | 0.36 | 0.50 (0.32-0.68) | 0.50 (0.50-0.50) | 0.470 | - |
| decoration | 0.89 | 0.50 (0.38-0.69) | 0.50 (0.39-0.50) | 0.258 | - |
| background | 0.71 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.00, right 0.12, bottom 0.00. Empty space 0.49. Reading order: headline vs cta: above 0.30, left 0.20, reads-before 0.50 (n=10); headline vs product: above 0.44, left 0.33, reads-before 0.67 (n=9); product vs cta: above 0.00, left 0.40, reads-before 0.40 (n=5); logo vs headline: above 0.12, left 0.62, reads-before 0.25 (n=8).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|       PPPPCCCCCCCCCCCCCCCCCCCCCCCCCCPPP        |
|       PPPPCCCCCCCCCCCCCCCCCCCCCCCCCCPPP        |
|       PPPPCCCCCCCCCCCCCCCCCCCCCCCCCCPPP        |
|       PPPPCCCCCCCCCCCCCCCCCCCCCCCCCCPPP        |
|       PPPPCCCCCCCCCCCCCCCCCCCCCCCCCCPPP        |
|       PPPPCCCCCCCCCCCCCCCCCCCCCCCCCCPPP        |
|       PPPPCCCCCCCCCCCCCCCCCCCCCCCCCCPPP        |
|    LLLLLPPCCCCCCCCCCCCCCCCCCCCCCCCCCPPP        |
|    LLLLLPPCCCCCCCCCCCCCCCCCCCCCCCCCCPPP        |
|                                                |
|                                                |
|                                                |
```

### leaderboard (728x90) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.50 (0.32-0.77) | 0.49 (0.43-0.53) | 0.093 | 0.30 (0.15-0.35) |
| body | 0.85 | 0.66 (0.43-0.79) | 0.66 (0.39-0.69) | 0.043 | 0.10 (0.07-0.14) |
| cta | 0.30 | 0.81 (0.69-0.85) | 0.50 (0.50-0.50) | 0.142 | 0.18 (0.12-0.22) |
| logo | 0.55 | 0.72 (0.14-0.88) | 0.50 (0.50-0.63) | 0.019 | - |
| product | 0.55 | 0.46 (0.30-0.59) | 0.50 (0.50-0.50) | 0.461 | - |
| decoration | 0.95 | 0.33 (0.23-0.76) | 0.50 (0.50-0.50) | 0.490 | - |
| background | 0.65 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.00, right 0.00, bottom 0.00. Empty space 0.27. Reading order: headline vs cta: above 0.17, left 0.67, reads-before 0.67 (n=6); headline vs product: above 0.36, left 0.46, reads-before 0.64 (n=11); product vs cta: above 0.00, left 1.00, reads-before 0.80 (n=5); logo vs headline: above 0.27, left 0.36, reads-before 0.55 (n=11).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|      PPPPPPPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC|
|      PPPPPPHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC|
|      PPPPPPHHHHHHHHHHHHHHHHHHCLLLLLLLCCCCCCCCCC|
|      PPPPPPHHHHHHHHHHHHHHHHHHCLLLLLLLCCCCCCCCCC|
|      PPPPPPPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC|
|      PPPPPPPPPPPPPPPPPPbbbbbbCCCCCCCCCCCCCCCCCC|
|      PPPPPPPPPPPPPPPPPPbbbbbbbbbbbbbbbb        |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|                                                |
|                                                |
|                                                |
```

### medium_rectangle (300x250) - strict pool, n=50

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.50 (0.49-0.52) | 0.49 (0.28-0.55) | 0.084 | 0.09 (0.06-0.13) |
| body | 0.80 | 0.50 (0.35-0.52) | 0.59 (0.42-0.73) | 0.064 | 0.04 (0.03-0.06) |
| cta | 0.28 | 0.50 (0.32-0.65) | 0.61 (0.48-0.85) | 0.031 | 0.04 (0.03-0.06) |
| logo | 0.40 | 0.50 (0.20-0.75) | 0.42 (0.12-0.84) | 0.007 | - |
| product | 0.32 | 0.60 (0.47-0.70) | 0.50 (0.49-0.50) | 0.528 | - |
| decoration | 0.94 | 0.50 (0.50-0.55) | 0.50 (0.46-0.63) | 0.497 | - |
| background | 0.78 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.01; left 0.07, top 0.06, right 0.04, bottom 0.04. Empty space 0.40. Reading order: headline vs cta: above 0.57, left 0.07, reads-before 0.57 (n=14); headline vs product: above 0.56, left 0.56, reads-before 0.69 (n=16); product vs cta: above 0.86, left 0.43, reads-before 0.86 (n=7); logo vs headline: above 0.55, left 0.40, reads-before 0.60 (n=20).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|           PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP  |
|           PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP  |
|           PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP  |
|           PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP  |
|           PPHHHHHHHHHLLLLHHHHHHHHHPPPPPPPPPPP  |
|           PPHHHHHHHHHHHHHHHHHHHHHHPPPPPPPPPPP  |
|           PPHHHHHHHCCCCCCCCHHHHHHHPPPPPPPPPPP  |
|           PPPbbbbbbCCCCCCCCbbbbbPPPPPPPPPPPPP  |
|           PPPbbbbbbCCCCCCCCbbbbbPPPPPPPPPPPPP  |
|           PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP  |
|           PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP  |
|           PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP  |
|                                                |
|                                                |
```

### wide_skyscraper (160x600) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.51 (0.50-0.52) | 0.50 (0.23-0.68) | 0.059 | 0.18 (0.13-0.28) |
| body | 0.90 | 0.50 (0.50-0.52) | 0.48 (0.25-0.70) | 0.054 | 0.07 (0.05-0.09) |
| cta | 0.40 | 0.50 (0.50-0.50) | 0.22 (0.19-0.37) | 0.069 | 0.06 (0.04-0.12) |
| logo | 0.40 | 0.50 (0.25-0.60) | 0.46 (0.08-0.88) | 0.013 | - |
| product | 0.45 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 0.509 | - |
| decoration | 0.90 | 0.50 (0.50-0.50) | 0.54 (0.50-0.68) | 0.536 | - |
| background | 0.80 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.15, right 0.00, bottom 0.05. Empty space 0.27. Reading order: headline vs cta: above 0.38, left 0.25, reads-before 0.38 (n=8); headline vs product: above 0.44, left 0.33, reads-before 0.44 (n=9); product vs cta: above 0.25, left 0.25, reads-before 0.25 (n=4); logo vs headline: above 0.38, left 0.38, reads-before 0.50 (n=8).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|                  CCCCCCCCCCCCC                 |
|       PPPPPPPPPPPCCCCCCCCCCCCCPPPPPPPPPP       |
|       PPPPPPPPPPPCCCCCCCCCCCCCPPPPPPPPPP       |
|       PPPPPPPPPPPCCCCCCCCCCCCCPPPPPPPPPP       |
|       PPPPPPPPbbbbbbbLLLLLbbbbbbPPPPPPPP       |
|       PPPPPPPPHHHHHHHLLLLLHHHHHHHPPPPPPP       |
|       PPPPPPPPHHHHHHHHHHHHHHHHHHHPPPPPPP       |
|       PPPPPPPPHHHHHHHHHHHHHHHHHHHPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|                                                |
|                                                |
```

### billboard (970x250) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.95 | 0.51 (0.43-0.69) | 0.50 (0.41-0.61) | 0.094 | 0.15 (0.12-0.20) |
| body | 0.85 | 0.50 (0.43-0.70) | 0.42 (0.28-0.68) | 0.032 | 0.07 (0.06-0.11) |
| cta | 0.20 | 0.74 (0.72-0.76) | 0.63 (0.52-0.78) | 0.055 | 0.09 (0.06-0.14) |
| logo | 0.55 | 0.54 (0.30-0.83) | 0.39 (0.15-0.78) | 0.007 | - |
| product | 0.35 | 0.31 (0.28-0.67) | 0.50 (0.50-0.50) | 0.532 | - |
| decoration | 1.00 | 0.50 (0.32-0.75) | 0.50 (0.47-0.50) | 0.408 | - |
| background | 0.85 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.10, top 0.05, right 0.12, bottom 0.01. Empty space 0.45. Reading order: headline vs cta: above 0.67, left 0.33, reads-before 0.67 (n=3); headline vs product: above 0.71, left 0.43, reads-before 0.71 (n=7); product vs cta: above 1.00, left 0.00, reads-before 1.00 (n=1); logo vs headline: above 0.46, left 0.27, reads-before 0.55 (n=11).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPHHHHHHHHHHHLLLLHHHHHHHHH           |
|PPPPPPPPPPPPPHHHHHHHHHHHHHHHHHHHHHHHH           |
|PPPPPPPPPPPPPHHHHHHHHHHHHHHHHHHHHHHHH           |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|                                                |
|                                                |
```

### mobile_banner (320x50) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.50 (0.32-0.77) | 0.49 (0.43-0.53) | 0.093 | 0.30 (0.15-0.35) |
| body | 0.85 | 0.66 (0.43-0.79) | 0.66 (0.39-0.69) | 0.043 | 0.10 (0.07-0.14) |
| cta | 0.30 | 0.81 (0.69-0.85) | 0.50 (0.50-0.50) | 0.142 | 0.18 (0.12-0.22) |
| logo | 0.55 | 0.72 (0.14-0.88) | 0.50 (0.50-0.63) | 0.019 | - |
| product | 0.55 | 0.46 (0.30-0.59) | 0.50 (0.50-0.50) | 0.461 | - |
| decoration | 0.95 | 0.33 (0.23-0.76) | 0.50 (0.50-0.50) | 0.490 | - |
| background | 0.65 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.00, right 0.00, bottom 0.00. Empty space 0.27. Reading order: headline vs cta: above 0.17, left 0.67, reads-before 0.67 (n=6); headline vs product: above 0.36, left 0.46, reads-before 0.64 (n=11); product vs cta: above 0.00, left 1.00, reads-before 0.80 (n=5); logo vs headline: above 0.27, left 0.36, reads-before 0.55 (n=11).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|      PPPPPPPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC|
|      PPPPPPHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC|
|      PPPPPPHHHHHHHHHHHHHHHHHHCLLLLLLLCCCCCCCCCC|
|      PPPPPPHHHHHHHHHHHHHHHHHHCLLLLLLLCCCCCCCCCC|
|      PPPPPPPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC|
|      PPPPPPPPPPPPPPPPPPbbbbbbCCCCCCCCCCCCCCCCCC|
|      PPPPPPPPPPPPPPPPPPbbbbbbbbbbbbbbbb        |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|                                                |
|                                                |
|                                                |
```

### half_page (300x600) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.95 | 0.50 (0.49-0.51) | 0.54 (0.28-0.71) | 0.080 | 0.18 (0.12-0.29) |
| body | 0.80 | 0.50 (0.48-0.56) | 0.36 (0.22-0.59) | 0.070 | 0.06 (0.04-0.09) |
| cta | 0.55 | 0.50 (0.50-0.67) | 0.59 (0.24-0.85) | 0.023 | 0.04 (0.03-0.05) |
| logo | 0.55 | 0.50 (0.29-0.63) | 0.11 (0.08-0.87) | 0.011 | - |
| product | 0.65 | 0.50 (0.50-0.50) | 0.50 (0.41-0.50) | 0.583 | - |
| decoration | 0.85 | 0.50 (0.46-0.54) | 0.50 (0.36-0.57) | 0.507 | - |
| background | 0.65 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.08, right 0.00, bottom 0.08. Empty space 0.20. Reading order: headline vs cta: above 0.50, left 0.40, reads-before 0.50 (n=10); headline vs product: above 0.42, left 0.33, reads-before 0.42 (n=12); product vs cta: above 0.50, left 0.38, reads-before 0.50 (n=8); logo vs headline: above 0.55, left 0.46, reads-before 0.64 (n=11).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                     LLLLL                      |
|      PPPPPPPPPPPPPPPLLLLLPPPPPPPPPPPPPPPPP     |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP     |
|      PPPPPPPPbbbbbbbbbbbbbbbbbbbbPPPPPPPPP     |
|      PPPPPPPPbbbbbbbbbbbbbbbbbbbbPPPPPPPPP     |
|      PPPPPPPPbbbbbbbbbbbbbbbbbbbbPPPPPPPPP     |
|      PPPPPPPHHHHHHHHHHHHHHHHHHHHHHPPPPPPPP     |
|      PPPPPPPHHHHHHHHCCCCCCCHHHHHHHPPPPPPPP     |
|      PPPPPPPHHHHHHHHCCCCCCCHHHHHHHPPPPPPPP     |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP     |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP     |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP     |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP     |
|                                                |
|                                                |
```

### large_leaderboard (970x90) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.50 (0.32-0.77) | 0.49 (0.43-0.53) | 0.093 | 0.30 (0.15-0.35) |
| body | 0.85 | 0.66 (0.43-0.79) | 0.66 (0.39-0.69) | 0.043 | 0.10 (0.07-0.14) |
| cta | 0.30 | 0.81 (0.69-0.85) | 0.50 (0.50-0.50) | 0.142 | 0.18 (0.12-0.22) |
| logo | 0.55 | 0.72 (0.14-0.88) | 0.50 (0.50-0.63) | 0.019 | - |
| product | 0.55 | 0.46 (0.30-0.59) | 0.50 (0.50-0.50) | 0.461 | - |
| decoration | 0.95 | 0.33 (0.23-0.76) | 0.50 (0.50-0.50) | 0.490 | - |
| background | 0.65 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.00, right 0.00, bottom 0.00. Empty space 0.27. Reading order: headline vs cta: above 0.17, left 0.67, reads-before 0.67 (n=6); headline vs product: above 0.36, left 0.46, reads-before 0.64 (n=11); product vs cta: above 0.00, left 1.00, reads-before 0.80 (n=5); logo vs headline: above 0.27, left 0.36, reads-before 0.55 (n=11).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|      PPPPPPPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC|
|      PPPPPPHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC|
|      PPPPPPHHHHHHHHHHHHHHHHHHCLLLLLLLCCCCCCCCCC|
|      PPPPPPHHHHHHHHHHHHHHHHHHCLLLLLLLCCCCCCCCCC|
|      PPPPPPPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC|
|      PPPPPPPPPPPPPPPPPPbbbbbbCCCCCCCCCCCCCCCCCC|
|      PPPPPPPPPPPPPPPPPPbbbbbbbbbbbbbbbb        |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|                                                |
|                                                |
|                                                |
```

## Our engine vs real practice (holdout cells)

Median centre / area / font by role. `real` = Crello, `base` = today's engine, `bisect` = bisection. Only targets with a strict aspect pool are ranked in the deviation lists; a role is compared only if present in at least 30% of both sides' layouts.

### instagram_post (strict; engine n=14)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.47 / 0.48 / 0.38 | 0.061 / 0.081 / 0.165 | 0.094 / 0.069 / 0.083 |
| body | 0.82 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.55 / 0.63 / 0.73 | 0.051 / 0.098 / 0.096 | 0.037 / 0.022 / 0.023 |
| cta | 0.48 / 0.57 / 0.57 | 0.50 / 0.50 / 0.42 | 0.48 / 0.82 / 0.85 | 0.075 / 0.040 / 0.058 | 0.053 / - / - |
| logo | 0.51 / 0.29 / 0.29 | 0.35 / 0.72 / 0.50 | 0.78 / 0.10 / 0.13 | 0.016 / 0.025 / 0.037 | - / - / - |
| product | 0.44 / 0.79 / 0.79 | 0.50 / 0.50 / 0.50 | 0.50 / 0.33 / 0.33 | 0.483 / 0.130 / 0.118 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.31 / 0.55 / 0.34; headline reads before CTA: 0.43 / 1.00 / 1.00.

### instagram_portrait (nearest-aspect; engine n=12)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.45 / 0.52 / 0.42 | 0.083 / 0.073 / 0.146 | 0.123 / 0.083 / 0.086 |
| body | 0.90 / 1.00 / 1.00 | 0.51 / 0.50 / 0.50 | 0.52 / 0.70 / 0.66 | 0.069 / 0.075 / 0.098 | 0.050 / 0.025 / 0.024 |
| cta | 0.20 / 0.42 / 0.42 | 0.46 / 0.50 / 0.50 | 0.55 / 0.84 / 0.84 | 0.021 / 0.040 / 0.018 | 0.057 / - / - |
| logo | 0.65 / 0.42 / 0.42 | 0.51 / 0.73 / 0.50 | 0.16 / 0.08 / 0.13 | 0.018 / 0.033 / 0.033 | - / - / - |
| product | 0.35 / 0.75 / 0.75 | 0.50 / 0.50 / 0.50 | 0.50 / 0.36 / 0.47 | 0.384 / 0.172 / 0.175 | - / - / - |

Min edge margin (median) real/base/bisect: 0.01 / 0.00 / 0.00; empty space: 0.35 / 0.49 / 0.29; headline reads before CTA: 0.75 / 1.00 / 1.00.

### story (strict; engine n=18)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.98 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.41 / 0.37 / 0.33 | 0.083 / 0.057 / 0.061 | 0.165 / 0.068 / 0.076 |
| body | 0.85 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.36 / 0.62 / 0.68 | 0.047 / 0.054 / 0.084 | 0.054 / 0.023 / 0.024 |
| cta | 0.58 / 0.39 / 0.39 | 0.50 / 0.50 / 0.50 | 0.79 / 0.84 / 0.84 | 0.030 / 0.035 / 0.035 | 0.035 / - / - |
| logo | 0.56 / 0.22 / 0.22 | 0.50 / 0.50 / 0.50 | 0.14 / 0.07 / 0.07 | 0.007 / 0.015 / 0.014 | - / - / - |
| product | 0.68 / 0.72 / 0.72 | 0.50 / 0.50 / 0.50 | 0.50 / 0.48 / 0.48 | 0.481 / 0.125 / 0.146 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.21 / 0.60 / 0.57; headline reads before CTA: 0.70 / 1.00 / 1.00.

### youtube_thumbnail (strict; engine n=15)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.95 / 1.00 / 1.00 | 0.50 / 0.67 / 0.50 | 0.43 / 0.34 / 0.38 | 0.088 / 0.087 / 0.183 | 0.127 / 0.102 / 0.126 |
| body | 0.78 / 1.00 / 1.00 | 0.50 / 0.67 / 0.52 | 0.59 / 0.54 / 0.50 | 0.043 / 0.082 / 0.148 | 0.049 / 0.027 / 0.032 |
| cta | 0.41 / 0.40 / 0.40 | 0.50 / 0.70 / 0.77 | 0.50 / 0.77 / 0.52 | 0.110 / 0.046 / 0.048 | 0.087 / - / - |
| logo | 0.61 / 0.47 / 0.47 | 0.35 / 0.50 / 0.22 | 0.24 / 0.11 / 0.16 | 0.018 / 0.024 / 0.038 | - / - / - |
| product | 0.39 / 0.67 / 0.67 | 0.51 / 0.26 / 0.32 | 0.50 / 0.50 / 0.51 | 0.574 / 0.269 / 0.096 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.39 / 0.45 / 0.40; headline reads before CTA: 0.53 / 1.00 / 0.50.

### linkedin_banner (nearest-aspect; engine n=10)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.95 / 1.00 / 1.00 | 0.51 / 0.50 / 0.33 | 0.50 / 0.43 / 0.50 | 0.094 / 0.118 / 0.287 | 0.146 / 0.141 / 0.246 |
| body | 0.85 / 1.00 / 1.00 | 0.50 / 0.50 / 0.86 | 0.42 / 0.63 / 0.39 | 0.032 / 0.043 / 0.138 | 0.069 / 0.040 / 0.053 |
| cta | 0.20 / 0.40 / 0.40 | 0.74 / 0.88 / 0.77 | 0.63 / 0.50 / 0.52 | 0.055 / 0.044 / 0.071 | 0.088 / - / - |
| logo | 0.55 / 0.30 / 0.30 | 0.54 / 0.52 / 0.17 | 0.39 / 0.18 / 0.28 | 0.007 / 0.035 / 0.041 | - / - / - |
| product | 0.35 / 0.80 / 0.80 | 0.31 / 0.15 / 0.49 | 0.50 / 0.50 / 0.50 | 0.532 / 0.118 / 0.092 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.45 / 0.50 / 0.32; headline reads before CTA: 0.67 / 0.75 / 0.50.

### facebook_cover (strict; engine n=21)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.96 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.48 / 0.41 / 0.45 | 0.073 / 0.135 / 0.263 | 0.146 / 0.138 / 0.207 |
| body | 0.71 / 1.00 / 1.00 | 0.51 / 0.50 / 0.52 | 0.64 / 0.66 / 0.62 | 0.029 / 0.095 / 0.156 | 0.059 / 0.040 / 0.053 |
| cta | 0.39 / 0.48 / 0.48 | 0.50 / 0.87 / 0.80 | 0.50 / 0.50 / 0.62 | 0.290 / 0.046 / 0.072 | 0.144 / - / - |
| logo | 0.29 / 0.24 / 0.24 | 0.14 / 0.52 / 0.19 | 0.76 / 0.18 / 0.26 | 0.012 / 0.053 / 0.089 | - / - / - |
| product | 0.36 / 0.76 / 0.76 | 0.50 / 0.17 / 0.60 | 0.50 / 0.50 / 0.50 | 0.470 / 0.074 / 0.141 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.49 / 0.48 / 0.29; headline reads before CTA: 0.50 / 1.00 / 0.70.

### leaderboard (nearest-aspect; engine n=10)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.50 / 0.55 / 0.34 | 0.49 / 0.25 / 0.50 | 0.093 / 0.138 / 0.264 | 0.297 / 0.264 / 0.342 |
| body | 0.85 / 1.00 / 1.00 | 0.66 / 0.55 / 0.77 | 0.66 / 0.55 / 0.50 | 0.043 / 0.228 / 0.291 | 0.102 / 0.100 / 0.117 |
| cta | 0.30 / 0.50 / 0.50 | 0.81 / 0.88 / 0.86 | 0.50 / 0.50 / 0.50 | 0.142 / 0.174 / 0.157 | 0.180 / - / - |
| logo | 0.55 / 0.40 / 0.40 | 0.72 / 0.50 / 0.09 | 0.50 / 0.26 / 0.25 | 0.019 / 0.051 / 0.024 | - / - / - |
| product | 0.55 / 0.90 / 0.90 | 0.46 / 0.28 / 0.64 | 0.50 / 0.50 / 0.50 | 0.461 / 0.105 / 0.091 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.02 / 0.03; empty space: 0.27 / 0.37 / 0.24; headline reads before CTA: 0.67 / 1.00 / 0.40.

### medium_rectangle (strict; engine n=15)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.50 / 0.50 / 0.33 | 0.49 / 0.20 / 0.58 | 0.084 / 0.118 / 0.265 | 0.088 / 0.090 / 0.148 |
| body | 0.80 / 1.00 / 1.00 | 0.50 / 0.50 / 0.47 | 0.59 / 0.43 / 0.39 | 0.064 / 0.172 / 0.240 | 0.038 / 0.038 / 0.046 |
| cta | 0.28 / 0.20 / 0.20 | 0.50 / 0.50 / 0.76 | 0.61 / 0.81 / 0.58 | 0.031 / 0.072 / 0.086 | 0.040 / - / - |
| logo | 0.40 / 0.27 / 0.27 | 0.50 / 0.21 / 0.80 | 0.42 / 0.89 / 0.15 | 0.007 / 0.021 / 0.026 | - / - / - |
| product | 0.32 / 0.73 / 0.73 | 0.60 / 0.26 / 0.79 | 0.50 / 0.41 / 0.30 | 0.528 / 0.140 / 0.104 | - / - / - |

Min edge margin (median) real/base/bisect: 0.01 / 0.00 / 0.00; empty space: 0.40 / 0.41 / 0.22; headline reads before CTA: 0.57 / 1.00 / 1.00.

### wide_skyscraper (nearest-aspect; engine n=16)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.51 / 0.50 / 0.50 | 0.50 / 0.20 / 0.20 | 0.059 / 0.138 / 0.205 | 0.178 / 0.167 / 0.211 |
| body | 0.90 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.48 / 0.35 / 0.69 | 0.054 / 0.141 / 0.183 | 0.068 / 0.065 / 0.077 |
| cta | 0.40 / 0.56 / 0.56 | 0.50 / 0.50 / 0.50 | 0.22 / 0.85 / 0.85 | 0.069 / 0.096 / 0.078 | 0.063 / - / - |
| logo | 0.40 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 | 0.46 / 0.04 / 0.04 | 0.013 / 0.017 / 0.011 | - / - / - |
| product | 0.45 / 0.81 / 0.81 | 0.50 / 0.50 / 0.50 | 0.50 / 0.58 / 0.58 | 0.509 / 0.216 / 0.216 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.27 / 0.35 / 0.23; headline reads before CTA: 0.38 / 1.00 / 0.89.

### billboard (nearest-aspect; engine n=18)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.95 / 1.00 / 1.00 | 0.51 / 0.50 / 0.32 | 0.50 / 0.43 / 0.50 | 0.094 / 0.131 / 0.291 | 0.146 / 0.180 / 0.241 |
| body | 0.85 / 1.00 / 1.00 | 0.50 / 0.50 / 0.58 | 0.42 / 0.66 / 0.50 | 0.032 / 0.124 / 0.164 | 0.069 / 0.050 / 0.055 |
| cta | 0.20 / 0.44 / 0.44 | 0.74 / 0.88 / 0.80 | 0.63 / 0.50 / 0.50 | 0.055 / 0.059 / 0.106 | 0.088 / - / - |
| logo | 0.55 / 0.33 / 0.33 | 0.54 / 0.51 / 0.11 | 0.39 / 0.18 / 0.22 | 0.007 / 0.036 / 0.043 | - / - / - |
| product | 0.35 / 0.72 / 0.72 | 0.31 / 0.15 / 0.52 | 0.50 / 0.50 / 0.50 | 0.532 / 0.114 / 0.148 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.45 / 0.41 / 0.30; headline reads before CTA: 0.67 / 1.00 / 0.50.

### mobile_banner (nearest-aspect; engine n=20)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.50 / 0.55 / 0.32 | 0.49 / 0.36 / 0.50 | 0.093 / 0.198 / 0.299 | 0.297 / 0.250 / 0.340 |
| body | 0.85 / 1.00 / 1.00 | 0.66 / 0.41 / 0.80 | 0.66 / 0.51 / 0.50 | 0.043 / 0.224 / 0.238 | 0.102 / 0.130 / 0.102 |
| cta | 0.30 / 0.40 / 0.40 | 0.81 / 0.88 / 0.83 | 0.50 / 0.50 / 0.50 | 0.142 / 0.134 / 0.149 | 0.180 / - / - |
| logo | 0.55 / 0.40 / 0.40 | 0.72 / 0.08 / 0.08 | 0.50 / 0.22 / 0.22 | 0.019 / 0.040 / 0.037 | - / - / - |
| product | 0.55 / 0.75 / 0.75 | 0.46 / 0.28 / 0.52 | 0.50 / 0.50 / 0.50 | 0.461 / 0.103 / 0.059 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.27 / 0.34 / 0.30; headline reads before CTA: 0.67 / 1.00 / 0.88.

### half_page (nearest-aspect; engine n=10)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.95 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.54 / 0.21 / 0.20 | 0.080 / 0.141 / 0.146 | 0.180 / 0.122 / 0.133 |
| body | 0.80 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.36 / 0.32 / 0.34 | 0.070 / 0.093 / 0.088 | 0.058 / 0.055 / 0.055 |
| cta | 0.55 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 | 0.59 / 0.83 / 0.85 | 0.023 / 0.079 / 0.075 | 0.038 / - / - |
| logo | 0.55 / 0.20 / 0.20 | 0.50 / 0.61 / 0.50 | 0.11 / 0.06 / 0.06 | 0.011 / 0.023 / 0.018 | - / - / - |
| product | 0.65 / 0.70 / 0.70 | 0.50 / 0.50 / 0.50 | 0.50 / 0.58 / 0.58 | 0.583 / 0.212 / 0.317 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.02 / 0.02; empty space: 0.20 / 0.44 / 0.34; headline reads before CTA: 0.50 / 1.00 / 1.00.

### large_leaderboard (nearest-aspect; engine n=13)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.50 / 0.55 / 0.33 | 0.49 / 0.25 / 0.50 | 0.093 / 0.176 / 0.214 | 0.297 / 0.306 / 0.378 |
| body | 0.85 / 1.00 / 1.00 | 0.66 / 0.55 / 0.60 | 0.66 / 0.56 / 0.50 | 0.043 / 0.236 / 0.209 | 0.102 / 0.100 / 0.111 |
| cta | 0.30 / 0.46 / 0.46 | 0.81 / 0.88 / 0.81 | 0.50 / 0.50 / 0.46 | 0.142 / 0.166 / 0.162 | 0.180 / - / - |
| logo | 0.55 / 0.61 / 0.61 | 0.72 / 0.21 / 0.06 | 0.50 / 0.32 / 0.40 | 0.019 / 0.051 / 0.022 | - / - / - |
| product | 0.55 / 0.69 / 0.69 | 0.46 / 0.23 / 0.69 | 0.50 / 0.50 / 0.50 | 0.461 / 0.069 / 0.069 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.27 / 0.37 / 0.32; headline reads before CTA: 0.67 / 1.00 / 0.33.

## Biggest deviations from real layouts

### base: top 12 (scaled gap = |engine - real| / (real IQR/1.35, floored))

| scaled gap | target | role | metric | real | engine |
|---|---|---|---|---|---|
| 3.8 | instagram_post | product | cy | 0.500 | 0.328 |
| 3.8 | medium_rectangle | product | area | 0.528 | 0.140 |
| 2.4 | medium_rectangle | body | area | 0.064 | 0.172 |
| 2.4 | story | product | area | 0.481 | 0.125 |
| 2.4 | facebook_cover | cta | cx | 0.500 | 0.869 |
| 2.2 | facebook_cover | body | area | 0.029 | 0.095 |
| 2.2 | medium_rectangle | product | cy | 0.500 | 0.410 |
| 2.1 | medium_rectangle | product | cx | 0.604 | 0.257 |
| 1.9 | youtube_thumbnail | cta | cy | 0.499 | 0.767 |
| 1.7 | facebook_cover | product | area | 0.470 | 0.074 |
| 1.7 | instagram_post | product | area | 0.483 | 0.130 |
| 1.6 | story | layout | empty space | 0.212 | 0.596 |

### bisect: top 12 (scaled gap = |engine - real| / (real IQR/1.35, floored))

| scaled gap | target | role | metric | real | engine |
|---|---|---|---|---|---|
| 5.1 | medium_rectangle | product | cy | 0.500 | 0.297 |
| 4.3 | medium_rectangle | headline | cx | 0.500 | 0.327 |
| 4.3 | facebook_cover | body | area | 0.029 | 0.156 |
| 4.2 | medium_rectangle | product | area | 0.528 | 0.104 |
| 3.9 | medium_rectangle | body | area | 0.064 | 0.240 |
| 3.8 | instagram_post | product | cy | 0.500 | 0.328 |
| 3.6 | facebook_cover | headline | area | 0.073 | 0.263 |
| 2.6 | facebook_cover | cta | cy | 0.500 | 0.621 |
| 2.6 | medium_rectangle | headline | area | 0.084 | 0.265 |
| 2.3 | youtube_thumbnail | body | area | 0.043 | 0.148 |
| 2.3 | story | product | area | 0.481 | 0.146 |
| 2.1 | youtube_thumbnail | product | area | 0.574 | 0.096 |

