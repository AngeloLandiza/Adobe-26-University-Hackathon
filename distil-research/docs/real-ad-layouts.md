# Real designer ad layouts vs our engine

What real designers do at each of our 13 target sizes, measured on the public Crello dataset (test split, all 4 shards), and where today's engine (`base`) and bisection (`bisect`) differ. Only aggregates are committed; the raw shards stay in `training-data/references/crello.nosync/` (gitignored). Numbers: `training-data/references/crello_layout_stats.json`, `engine_layout_stats_base.json`, `engine_layout_stats_bisect.json`. Scripts: `chameleon/dev/references/`.

## Method

- Data: `cyberagent/crello` revision 5.1.0, test split, 4 shards `data/test-0000N-of-00004.parquet` (1971 templates; 1606 in ad/social/promo formats after excluding resumes, invitations, cards, certificates, menus, logos, presentations, calendars and similar).
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
| instagram_post | 1080x1080 | strict | 231 | 231 | - | Instagram 147, Instagram AD 84 |
| instagram_portrait | 1080x1350 | nearest-aspect | 0 | 20 | [0.75, 0.75] | Poster US 20 |
| story | 1080x1920 | strict | 162 | 162 | - | Instagram Story 162 |
| youtube_thumbnail | 1280x720 | strict | 153 | 153 | - | FB event cover 60, Youtube 41, Title 29 |
| linkedin_banner | 1584x396 | strict | 31 | 31 | - | Twitter 16, VK Community Cover 14, Web Banner 1 |
| facebook_cover | 820x312 | strict | 127 | 127 | - | Facebook cover 124, Twitch Profile Banner 3 |
| leaderboard | 728x90 | strict | 25 | 25 | - | Leaderboard 25 |
| medium_rectangle | 300x250 | strict | 183 | 183 | - | Facebook 119, Large Rectangle 32, Medium Rectangle 32 |
| wide_skyscraper | 160x600 | strict | 29 | 29 | - | Skyscraper 29 |
| billboard | 970x250 | strict | 31 | 31 | - | Twitter 16, VK Community Cover 14, Web Banner 1 |
| mobile_banner | 320x50 | nearest-aspect | 0 | 20 | [5.906, 8.089] | LinkedIn Cover 12, Leaderboard 8 |
| half_page | 300x600 | strict | 43 | 43 | - | Graphic 43 |
| large_leaderboard | 970x90 | nearest-aspect | 0 | 20 | [8.089, 8.089] | Leaderboard 20 |

## Per-size summaries (real designers)

### instagram_post (1080x1080) - strict pool, n=231

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.99 | 0.50 (0.37-0.51) | 0.48 (0.30-0.61) | 0.071 | 0.11 (0.06-0.16) |
| body | 0.86 | 0.50 (0.35-0.58) | 0.55 (0.33-0.74) | 0.052 | 0.04 (0.03-0.06) |
| cta | 0.39 | 0.50 (0.28-0.52) | 0.48 (0.30-0.66) | 0.074 | 0.05 (0.03-0.08) |
| logo | 0.53 | 0.50 (0.14-0.82) | 0.50 (0.14-0.84) | 0.014 | - |
| product | 0.45 | 0.50 (0.48-0.60) | 0.50 (0.50-0.55) | 0.498 | - |
| decoration | 0.96 | 0.50 (0.42-0.53) | 0.50 (0.48-0.60) | 0.517 | - |
| background | 0.80 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.03, top 0.05, right 0.00, bottom 0.03. Empty space 0.31. Reading order: headline vs cta: above 0.47, left 0.29, reads-before 0.50 (n=88); headline vs product: above 0.52, left 0.52, reads-before 0.62 (n=103); product vs cta: above 0.38, left 0.25, reads-before 0.47 (n=40); logo vs headline: above 0.49, left 0.43, reads-before 0.53 (n=121).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPHHHCCCCCCCCCCCCCHHHHPPPPPPP       |
|       PPPPPPPHHHCCCCLLLLLLCCCHHHHPPPPPPP       |
|       PPPPPPPHHHCCCCLLLLLLCCCHHHHPPPPPPP       |
|       PPPPPPPPPbCCCCCCCCCCCCCbbbPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|                                                |
|                                                |
|                                                |
```

### instagram_portrait (1080x1350) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.50 (0.40-0.52) | 0.50 (0.39-0.57) | 0.101 | 0.11 (0.08-0.19) |
| body | 0.85 | 0.50 (0.48-0.54) | 0.67 (0.42-0.86) | 0.085 | 0.04 (0.03-0.06) |
| cta | 0.25 | 0.50 (0.35-0.50) | 0.76 (0.73-0.80) | 0.092 | 0.05 (0.04-0.06) |
| logo | 0.35 | 0.35 (0.26-0.67) | 0.12 (0.09-0.16) | 0.006 | - |
| product | 0.30 | 0.50 (0.50-0.51) | 0.51 (0.50-0.55) | 0.411 | - |
| decoration | 0.95 | 0.50 (0.50-0.51) | 0.50 (0.48-0.72) | 0.561 | - |
| background | 0.85 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.02; left 0.05, top 0.06, right 0.03, bottom 0.05. Empty space 0.37. Reading order: headline vs cta: above 0.80, left 0.40, reads-before 0.80 (n=5); headline vs product: above 0.67, left 0.33, reads-before 0.67 (n=6); product vs cta: above 1.00, left 0.33, reads-before 1.00 (n=3); logo vs headline: above 0.86, left 0.29, reads-before 0.86 (n=7).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|               LLLL                             |
|                                                |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|        PPPPHHHHHHHHHHHHHHHHHHHHHHHHPPP         |
|        PPPPHHHHHHHHHHHHHHHHHHHHHHHHPPP         |
|        PPPPHHHHHHHHHHHHHHHHHHHHHHHHPPP         |
|        PPPPHHHHHHHHHHHHHHHHHHHHHHHHPPP         |
|        PPPPPbbbCCCCCCCCCCCCCCCbbbbPPPP         |
|        PPPPPbbbCCCCCCCCCCCCCCCbbbbPPPP         |
|        PPPPPPPPCCCCCCCCCCCCCCCPPPPPPPP         |
|                CCCCCCCCCCCCCCC                 |
|                CCCCCCCCCCCCCCC                 |
|                                                |
```

### story (1080x1920) - strict pool, n=162

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.99 | 0.50 (0.49-0.51) | 0.39 (0.23-0.58) | 0.081 | 0.16 (0.10-0.22) |
| body | 0.85 | 0.50 (0.49-0.52) | 0.50 (0.28-0.72) | 0.047 | 0.05 (0.04-0.08) |
| cta | 0.60 | 0.50 (0.50-0.51) | 0.81 (0.52-0.88) | 0.019 | 0.04 (0.03-0.06) |
| logo | 0.54 | 0.50 (0.34-0.83) | 0.37 (0.11-0.87) | 0.014 | - |
| product | 0.66 | 0.50 (0.50-0.50) | 0.50 (0.40-0.58) | 0.485 | - |
| decoration | 0.93 | 0.50 (0.50-0.52) | 0.50 (0.36-0.69) | 0.469 | - |
| background | 0.63 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.03, right 0.00, bottom 0.01. Empty space 0.25. Reading order: headline vs cta: above 0.80, left 0.19, reads-before 0.83 (n=95); headline vs product: above 0.57, left 0.18, reads-before 0.61 (n=105); product vs cta: above 0.76, left 0.18, reads-before 0.76 (n=66); logo vs headline: above 0.44, left 0.29, reads-before 0.46 (n=88).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPHHHHHHHHLLLLLLHHHHHHHHPPPPPP       |
|        PPPPPHHHHHHHHLLLLLLHHHHHHHHPPPPPP       |
|        PPPPPHHHHHHHHHHHHHHHHHHHHHHPPPPPP       |
|        PPPPPPPPbbbbbbbbbbbbbbbbbPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|        PPPPPPPPPPPPPCCCCCCCPPPPPPPPPPPPP       |
|                     CCCCCCC                    |
|                                                |
|                                                |
```

### youtube_thumbnail (1280x720) - strict pool, n=153

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.95 | 0.50 (0.32-0.55) | 0.47 (0.39-0.53) | 0.074 | 0.12 (0.09-0.19) |
| body | 0.77 | 0.50 (0.28-0.65) | 0.56 (0.41-0.71) | 0.041 | 0.05 (0.03-0.08) |
| cta | 0.47 | 0.50 (0.29-0.62) | 0.50 (0.35-0.59) | 0.080 | 0.07 (0.05-0.11) |
| logo | 0.50 | 0.50 (0.12-0.85) | 0.39 (0.14-0.80) | 0.012 | - |
| product | 0.44 | 0.50 (0.30-0.67) | 0.50 (0.50-0.50) | 0.501 | - |
| decoration | 0.94 | 0.50 (0.29-0.64) | 0.50 (0.42-0.51) | 0.384 | - |
| background | 0.74 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.03, top 0.00, right 0.05, bottom 0.02. Empty space 0.41. Reading order: headline vs cta: above 0.52, left 0.20, reads-before 0.55 (n=65); headline vs product: above 0.50, left 0.46, reads-before 0.61 (n=66); product vs cta: above 0.41, left 0.47, reads-before 0.53 (n=34); logo vs headline: above 0.52, left 0.42, reads-before 0.55 (n=73).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPCCCCLLLLLCCCCCPPPPPPPPPP       |
|       PPPPPPPHHHCCCCLLLLLCCCCCHHHHPPPPPP       |
|       PPPPPPPHHHCCCCCCCCCCCCCCHHHHPPPPPP       |
|       PPPPPPPHHHCCCCCCCCCCCCCCHHHHPPPPPP       |
|       PPPPPPPPPbCCCCCCCCCCCCCCbPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|                                                |
|                                                |
|                                                |
```

### linkedin_banner (1584x396) - strict pool, n=31

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.97 | 0.50 (0.39-0.67) | 0.46 (0.41-0.50) | 0.095 | 0.18 (0.13-0.25) |
| body | 0.90 | 0.50 (0.41-0.66) | 0.64 (0.40-0.72) | 0.045 | 0.07 (0.06-0.09) |
| cta | 0.23 | 0.68 (0.50-0.75) | 0.83 (0.71-0.86) | 0.041 | 0.06 (0.05-0.10) |
| logo | 0.52 | 0.52 (0.47-0.76) | 0.64 (0.19-0.88) | 0.022 | - |
| product | 0.35 | 0.50 (0.31-0.60) | 0.50 (0.50-0.50) | 0.609 | - |
| decoration | 0.87 | 0.50 (0.36-0.72) | 0.50 (0.48-0.50) | 0.541 | - |
| background | 0.77 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.22, top 0.00, right 0.15, bottom 0.00. Empty space 0.43. Reading order: headline vs cta: above 1.00, left 0.50, reads-before 1.00 (n=6); headline vs product: above 0.64, left 0.27, reads-before 0.64 (n=11); product vs cta: above 1.00, left 1.00, reads-before 1.00 (n=1); logo vs headline: above 0.38, left 0.31, reads-before 0.38 (n=16).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPHHHHHHHHHHHHHHHHHHHHHHHHPPPPPP      |
|     PPPPPPPHHHHHHHHHHHHHHHHHHHHHHHHPPPPPP      |
|     PPPPPPPHHHHHHHHHHHHHHHHHHHHHHHHPPPPPP      |
|     PPPPPPPPPPPbbbbbbLLLLLLLbbbPPPPPPPPPP      |
|     PPPPPPPPPPPbbbbbbLLLLLLLbbbPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|                                                |
|                                                |
```

### facebook_cover (820x312) - strict pool, n=127

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.94 | 0.50 (0.42-0.68) | 0.50 (0.43-0.57) | 0.081 | 0.16 (0.10-0.22) |
| body | 0.69 | 0.51 (0.40-0.72) | 0.56 (0.39-0.68) | 0.036 | 0.07 (0.05-0.12) |
| cta | 0.39 | 0.51 (0.44-0.72) | 0.50 (0.41-0.56) | 0.118 | 0.11 (0.07-0.17) |
| logo | 0.39 | 0.50 (0.16-0.84) | 0.60 (0.17-0.84) | 0.017 | - |
| product | 0.44 | 0.39 (0.29-0.54) | 0.50 (0.50-0.50) | 0.481 | - |
| decoration | 0.91 | 0.50 (0.36-0.70) | 0.50 (0.49-0.52) | 0.372 | - |
| background | 0.73 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.09, top 0.00, right 0.03, bottom 0.00. Empty space 0.47. Reading order: headline vs cta: above 0.43, left 0.26, reads-before 0.50 (n=42); headline vs product: above 0.39, left 0.33, reads-before 0.52 (n=52); product vs cta: above 0.29, left 0.58, reads-before 0.54 (n=24); logo vs headline: above 0.36, left 0.38, reads-before 0.43 (n=47).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|  PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|  PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|  PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|  PPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCPP             |
|  PPPPPPPPPPPHHHCCCCCCCCCCCCCCCCCHH             |
|  PPPPPPPPPPPHHHCCCCCCCCCCCCCCCCCHH             |
|  PPPPPPPPPPPHHHCCCCCCCCCCCCCCCCCHH             |
|  PPPPPPPPPPPPPPCCCCCLLLLLLCCCCCCPP             |
|  PPPPPPPPPPPPPPCCCCCLLLLLLCCCCCCPP             |
|  PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|  PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP             |
|                                                |
|                                                |
|                                                |
```

### leaderboard (728x90) - strict pool, n=25

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.46 (0.25-0.70) | 0.49 (0.43-0.52) | 0.107 | 0.30 (0.27-0.43) |
| body | 0.72 | 0.53 (0.31-0.68) | 0.50 (0.49-0.66) | 0.104 | 0.16 (0.12-0.22) |
| cta | 0.40 | 0.75 (0.55-0.90) | 0.50 (0.49-0.50) | 0.142 | 0.19 (0.16-0.24) |
| logo | 0.64 | 0.17 (0.09-0.83) | 0.50 (0.50-0.64) | 0.025 | - |
| product | 0.76 | 0.48 (0.33-0.69) | 0.50 (0.50-0.50) | 0.456 | - |
| decoration | 0.96 | 0.27 (0.19-0.52) | 0.50 (0.50-0.50) | 0.467 | - |
| background | 0.48 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.00, right 0.00, bottom 0.00. Empty space 0.09. Reading order: headline vs cta: above 0.20, left 0.80, reads-before 0.70 (n=10); headline vs product: above 0.37, left 0.42, reads-before 0.58 (n=19); product vs cta: above 0.00, left 0.78, reads-before 0.67 (n=9); logo vs headline: above 0.19, left 0.62, reads-before 0.50 (n=16).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|       PPPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC   |
|       PPPHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
|    LLLLLLLLHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
|    LLLLLLLLHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
|    LLLLLLLLHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
|       PPPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC   |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP         |
|                                                |
|                                                |
|                                                |
```

### medium_rectangle (300x250) - strict pool, n=183

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.50 (0.39-0.52) | 0.46 (0.29-0.59) | 0.082 | 0.10 (0.07-0.14) |
| body | 0.84 | 0.50 (0.35-0.52) | 0.55 (0.34-0.72) | 0.055 | 0.04 (0.03-0.06) |
| cta | 0.35 | 0.50 (0.34-0.53) | 0.50 (0.28-0.77) | 0.059 | 0.05 (0.04-0.09) |
| logo | 0.50 | 0.50 (0.17-0.85) | 0.62 (0.14-0.86) | 0.010 | - |
| product | 0.36 | 0.50 (0.48-0.67) | 0.50 (0.50-0.51) | 0.549 | - |
| decoration | 0.94 | 0.50 (0.46-0.60) | 0.50 (0.46-0.63) | 0.417 | - |
| background | 0.75 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.03, top 0.05, right 0.02, bottom 0.04. Empty space 0.38. Reading order: headline vs cta: above 0.52, left 0.17, reads-before 0.52 (n=64); headline vs product: above 0.53, left 0.52, reads-before 0.61 (n=66); product vs cta: above 0.54, left 0.29, reads-before 0.54 (n=28); logo vs headline: above 0.46, left 0.39, reads-before 0.46 (n=92).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPHHHHHCCCCCCCCCCCCHHHHHPPPPPPP      |
|      PPPPPPPHHHHHCCCCCCCCCCCCHHHHHPPPPPPP      |
|      PPPPPPPHHHHHCCCCCCCCCCCCHHHHHPPPPPPP      |
|      PPPPPPPPPbbbCCCCLLLLLCCCbbbPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPLLLLLPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|                                                |
|                                                |
```

### wide_skyscraper (160x600) - strict pool, n=29

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.51 (0.50-0.55) | 0.44 (0.27-0.70) | 0.078 | 0.19 (0.13-0.29) |
| body | 0.86 | 0.50 (0.48-0.50) | 0.53 (0.33-0.74) | 0.088 | 0.09 (0.08-0.12) |
| cta | 0.45 | 0.50 (0.50-0.51) | 0.28 (0.21-0.71) | 0.129 | 0.12 (0.09-0.23) |
| logo | 0.52 | 0.50 (0.49-0.54) | 0.55 (0.14-0.91) | 0.011 | - |
| product | 0.34 | 0.50 (0.50-0.50) | 0.52 (0.39-0.64) | 0.550 | - |
| decoration | 0.97 | 0.50 (0.50-0.50) | 0.62 (0.49-0.80) | 0.383 | - |
| background | 0.86 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.15, right 0.00, bottom 0.00. Empty space 0.31. Reading order: headline vs cta: above 0.46, left 0.08, reads-before 0.46 (n=13); headline vs product: above 0.40, left 0.20, reads-before 0.40 (n=10); product vs cta: above 0.40, left 0.00, reads-before 0.40 (n=5); logo vs headline: above 0.33, left 0.33, reads-before 0.40 (n=15).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|      PPPPPPPPPPCCCCCCCCCCCCCCCCCPPPPPPPPP      |
|      PPPPPPPPPPCCCCCCCCCCCCCCCCCPPPPPPPPP      |
|      PPPPPPPPPPCCCCCCCCCCCCCCCCCPPPPPPPPP      |
|      PPPPPPPPPPCCCCCCCCCCCCCCCCCPPPPPPPPP      |
|      PPPPPPPHHHCCCCCCCCCCCCCCCCCHHPPPPPPP      |
|      PPPPPPbHHHCCCCCCCCCCCCCCCCCHHPPPPPPP      |
|      PPPPPPbHHHHHHHHHLLLLLHHHHHHHHPPPPPPP      |
|      PPPPPPbbbbbbbbbbLLLLLbbbbbbbbPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|      PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|                                                |
|                                                |
```

### billboard (970x250) - strict pool, n=31

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.97 | 0.50 (0.39-0.67) | 0.46 (0.41-0.50) | 0.095 | 0.18 (0.13-0.25) |
| body | 0.90 | 0.50 (0.41-0.66) | 0.64 (0.40-0.72) | 0.045 | 0.07 (0.06-0.09) |
| cta | 0.23 | 0.68 (0.50-0.75) | 0.83 (0.71-0.86) | 0.041 | 0.06 (0.05-0.10) |
| logo | 0.52 | 0.52 (0.47-0.76) | 0.64 (0.19-0.88) | 0.022 | - |
| product | 0.35 | 0.50 (0.31-0.60) | 0.50 (0.50-0.50) | 0.609 | - |
| decoration | 0.87 | 0.50 (0.36-0.72) | 0.50 (0.48-0.50) | 0.541 | - |
| background | 0.77 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.22, top 0.00, right 0.15, bottom 0.00. Empty space 0.43. Reading order: headline vs cta: above 1.00, left 0.50, reads-before 1.00 (n=6); headline vs product: above 0.64, left 0.27, reads-before 0.64 (n=11); product vs cta: above 1.00, left 1.00, reads-before 1.00 (n=1); logo vs headline: above 0.38, left 0.31, reads-before 0.38 (n=16).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPHHHHHHHHHHHHHHHHHHHHHHHHPPPPPP      |
|     PPPPPPPHHHHHHHHHHHHHHHHHHHHHHHHPPPPPP      |
|     PPPPPPPHHHHHHHHHHHHHHHHHHHHHHHHPPPPPP      |
|     PPPPPPPPPPPbbbbbbLLLLLLLbbbPPPPPPPPPP      |
|     PPPPPPPPPPPbbbbbbLLLLLLLbbbPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|     PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      |
|                                                |
|                                                |
```

### mobile_banner (320x50) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.38 (0.28-0.67) | 0.50 (0.43-0.54) | 0.125 | 0.34 (0.29-0.42) |
| body | 0.75 | 0.51 (0.45-0.81) | 0.66 (0.50-0.70) | 0.048 | 0.13 (0.12-0.17) |
| cta | 0.30 | 0.75 (0.62-0.85) | 0.50 (0.49-0.50) | 0.142 | 0.19 (0.15-0.22) |
| logo | 0.35 | 0.18 (0.10-0.63) | 0.50 (0.44-0.50) | 0.034 | - |
| product | 0.45 | 0.50 (0.46-0.68) | 0.50 (0.50-0.50) | 0.456 | - |
| decoration | 0.70 | 0.30 (0.20-0.50) | 0.50 (0.50-0.50) | 0.400 | - |
| background | 0.80 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.23, top 0.00, right 0.07, bottom 0.00. Empty space 0.45. Reading order: headline vs cta: above 0.00, left 0.83, reads-before 0.50 (n=6); headline vs product: above 0.22, left 0.56, reads-before 0.56 (n=9); product vs cta: above 0.00, left 1.00, reads-before 0.80 (n=5); logo vs headline: above 0.29, left 0.57, reads-before 0.71 (n=7).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|        PPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC   |
|    LLLLLLLLLHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
|    LLLLLLLLLHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
|    LLLLLLLLLHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
|     HHHHHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
|        PPPPPPPPbbbbbbbbbbbCCCCCCCCCCCCCCCCCC   |
|        PPPPPPPPbbbbbbbbbbbbbbbbbPPPPPPP        |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|                                                |
|                                                |
|                                                |
```

### half_page (300x600) - strict pool, n=43

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 0.95 | 0.50 (0.47-0.58) | 0.48 (0.23-0.74) | 0.052 | 0.13 (0.07-0.19) |
| body | 0.86 | 0.50 (0.46-0.54) | 0.54 (0.35-0.76) | 0.048 | 0.04 (0.03-0.06) |
| cta | 0.42 | 0.50 (0.50-0.61) | 0.44 (0.27-0.76) | 0.070 | 0.06 (0.04-0.10) |
| logo | 0.42 | 0.54 (0.29-0.72) | 0.70 (0.13-0.87) | 0.013 | - |
| product | 0.49 | 0.50 (0.50-0.50) | 0.50 (0.49-0.54) | 0.509 | - |
| decoration | 0.95 | 0.50 (0.38-0.50) | 0.50 (0.49-0.66) | 0.521 | - |
| background | 0.81 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.06, right 0.00, bottom 0.06. Empty space 0.23. Reading order: headline vs cta: above 0.56, left 0.31, reads-before 0.56 (n=16); headline vs product: above 0.53, left 0.37, reads-before 0.63 (n=19); product vs cta: above 0.42, left 0.25, reads-before 0.42 (n=12); logo vs headline: above 0.39, left 0.28, reads-before 0.44 (n=18).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|                                                |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPCCCCCCCCCCCCCPPPPPPPPPP       |
|       PPPPPPPPPHHCCCCCCCCCCCCCHHPPPPPPPP       |
|       PPPPPPPPPHHCCCCCCCCCCCCCHHPPPPPPPP       |
|       PPPPPPPPPHHCCCCCCCCCCCCCHHPPPPPPPP       |
|       PPPPPPPPPbbbbbbbbbbbbbbbbbPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPLLLLLPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPLLLLLPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|       PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP       |
|                                                |
|                                                |
```

### large_leaderboard (970x90) - nearest-aspect pool, n=20

| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |
|---|---|---|---|---|---|
| headline | 1.00 | 0.40 (0.20-0.67) | 0.49 (0.43-0.52) | 0.100 | 0.34 (0.26-0.44) |
| body | 0.70 | 0.48 (0.31-0.68) | 0.50 (0.50-0.66) | 0.104 | 0.17 (0.10-0.22) |
| cta | 0.50 | 0.75 (0.55-0.90) | 0.50 (0.49-0.50) | 0.142 | 0.19 (0.16-0.24) |
| logo | 0.55 | 0.11 (0.08-0.70) | 0.50 (0.50-0.57) | 0.027 | - |
| product | 0.80 | 0.49 (0.39-0.68) | 0.50 (0.50-0.50) | 0.446 | - |
| decoration | 1.00 | 0.23 (0.17-0.50) | 0.50 (0.50-0.50) | 0.400 | - |
| background | 0.45 | 0.50 (0.50-0.50) | 0.50 (0.50-0.50) | 1.000 | - |

Edge margin (share of short side, median): min-any-edge 0.00; left 0.00, top 0.00, right 0.00, bottom 0.00. Empty space 0.08. Reading order: headline vs cta: above 0.20, left 0.80, reads-before 0.70 (n=10); headline vs product: above 0.38, left 0.50, reads-before 0.62 (n=16); product vs cta: above 0.00, left 0.78, reads-before 0.67 (n=9); logo vs headline: above 0.27, left 0.55, reads-before 0.64 (n=11).

Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):

```
|                                                |
|                                                |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|        PPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC   |
| LLLLLLLLHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
| LLLLLLLLHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
| LLLLLLLLHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
|       HHHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCC   |
|        PPPPPPPPPPPPPPPPPPPCCCCCCCCCCCCCCCCCC   |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|        PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP        |
|                                                |
|                                                |
|                                                |
```

## Our engine vs real practice (holdout cells)

Median centre / area / font by role. `real` = Crello, `base` = today's engine, `bisect` = bisection. Only targets with a strict aspect pool are ranked in the deviation lists; a role is compared only if present in at least 30% of both sides' layouts.

### instagram_post (strict; engine n=14)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.99 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.48 / 0.48 / 0.38 | 0.071 / 0.081 / 0.165 | 0.108 / 0.069 / 0.083 |
| body | 0.86 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.55 / 0.63 / 0.73 | 0.052 / 0.098 / 0.096 | 0.035 / 0.022 / 0.023 |
| cta | 0.39 / 0.57 / 0.57 | 0.50 / 0.50 / 0.42 | 0.48 / 0.82 / 0.85 | 0.074 / 0.040 / 0.058 | 0.054 / - / - |
| logo | 0.53 / 0.29 / 0.29 | 0.50 / 0.72 / 0.50 | 0.50 / 0.10 / 0.13 | 0.014 / 0.025 / 0.037 | - / - / - |
| product | 0.45 / 0.79 / 0.79 | 0.50 / 0.50 / 0.50 | 0.50 / 0.33 / 0.33 | 0.498 / 0.130 / 0.118 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.31 / 0.55 / 0.34; headline reads before CTA: 0.50 / 1.00 / 1.00.

### instagram_portrait (nearest-aspect; engine n=12)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.50 / 0.52 / 0.42 | 0.101 / 0.073 / 0.146 | 0.108 / 0.083 / 0.086 |
| body | 0.85 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.67 / 0.70 / 0.66 | 0.085 / 0.075 / 0.098 | 0.039 / 0.025 / 0.024 |
| cta | 0.25 / 0.42 / 0.42 | 0.50 / 0.50 / 0.50 | 0.76 / 0.84 / 0.84 | 0.092 / 0.040 / 0.018 | 0.046 / - / - |
| logo | 0.35 / 0.42 / 0.42 | 0.35 / 0.73 / 0.50 | 0.12 / 0.08 / 0.13 | 0.006 / 0.033 / 0.033 | - / - / - |
| product | 0.30 / 0.75 / 0.75 | 0.50 / 0.50 / 0.50 | 0.51 / 0.36 / 0.47 | 0.411 / 0.172 / 0.175 | - / - / - |

Min edge margin (median) real/base/bisect: 0.02 / 0.00 / 0.00; empty space: 0.37 / 0.49 / 0.29; headline reads before CTA: 0.80 / 1.00 / 1.00.

### story (strict; engine n=18)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.99 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.39 / 0.37 / 0.33 | 0.081 / 0.057 / 0.061 | 0.156 / 0.068 / 0.076 |
| body | 0.85 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.50 / 0.62 / 0.68 | 0.047 / 0.054 / 0.084 | 0.048 / 0.023 / 0.024 |
| cta | 0.60 / 0.39 / 0.39 | 0.50 / 0.50 / 0.50 | 0.81 / 0.84 / 0.84 | 0.019 / 0.035 / 0.035 | 0.036 / - / - |
| logo | 0.54 / 0.22 / 0.22 | 0.50 / 0.50 / 0.50 | 0.37 / 0.07 / 0.07 | 0.014 / 0.015 / 0.014 | - / - / - |
| product | 0.66 / 0.72 / 0.72 | 0.50 / 0.50 / 0.50 | 0.50 / 0.48 / 0.48 | 0.485 / 0.125 / 0.146 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.25 / 0.60 / 0.57; headline reads before CTA: 0.83 / 1.00 / 1.00.

### youtube_thumbnail (strict; engine n=15)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.95 / 1.00 / 1.00 | 0.50 / 0.67 / 0.50 | 0.47 / 0.34 / 0.38 | 0.074 / 0.087 / 0.183 | 0.123 / 0.102 / 0.126 |
| body | 0.77 / 1.00 / 1.00 | 0.50 / 0.67 / 0.52 | 0.56 / 0.54 / 0.50 | 0.041 / 0.082 / 0.148 | 0.050 / 0.027 / 0.032 |
| cta | 0.47 / 0.40 / 0.40 | 0.50 / 0.70 / 0.77 | 0.50 / 0.77 / 0.52 | 0.080 / 0.046 / 0.048 | 0.070 / - / - |
| logo | 0.50 / 0.47 / 0.47 | 0.50 / 0.50 / 0.22 | 0.39 / 0.11 / 0.16 | 0.012 / 0.024 / 0.038 | - / - / - |
| product | 0.44 / 0.67 / 0.67 | 0.50 / 0.26 / 0.32 | 0.50 / 0.50 / 0.51 | 0.501 / 0.269 / 0.096 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.41 / 0.45 / 0.40; headline reads before CTA: 0.55 / 1.00 / 0.50.

### linkedin_banner (strict; engine n=10)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.97 / 1.00 / 1.00 | 0.50 / 0.50 / 0.33 | 0.46 / 0.43 / 0.50 | 0.095 / 0.118 / 0.287 | 0.184 / 0.141 / 0.246 |
| body | 0.90 / 1.00 / 1.00 | 0.50 / 0.50 / 0.86 | 0.64 / 0.63 / 0.39 | 0.045 / 0.043 / 0.138 | 0.070 / 0.040 / 0.053 |
| cta | 0.23 / 0.40 / 0.40 | 0.68 / 0.88 / 0.77 | 0.83 / 0.50 / 0.52 | 0.041 / 0.044 / 0.071 | 0.058 / - / - |
| logo | 0.52 / 0.30 / 0.30 | 0.52 / 0.52 / 0.17 | 0.64 / 0.18 / 0.28 | 0.022 / 0.035 / 0.041 | - / - / - |
| product | 0.35 / 0.80 / 0.80 | 0.50 / 0.15 / 0.49 | 0.50 / 0.50 / 0.50 | 0.609 / 0.118 / 0.092 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.43 / 0.50 / 0.32; headline reads before CTA: 1.00 / 0.75 / 0.50.

### facebook_cover (strict; engine n=21)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.94 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.50 / 0.41 / 0.45 | 0.081 / 0.135 / 0.263 | 0.159 / 0.138 / 0.207 |
| body | 0.69 / 1.00 / 1.00 | 0.51 / 0.50 / 0.52 | 0.56 / 0.66 / 0.62 | 0.036 / 0.095 / 0.156 | 0.068 / 0.040 / 0.053 |
| cta | 0.39 / 0.48 / 0.48 | 0.51 / 0.87 / 0.80 | 0.50 / 0.50 / 0.62 | 0.118 / 0.046 / 0.072 | 0.111 / - / - |
| logo | 0.39 / 0.24 / 0.24 | 0.50 / 0.52 / 0.19 | 0.60 / 0.18 / 0.26 | 0.017 / 0.053 / 0.089 | - / - / - |
| product | 0.44 / 0.76 / 0.76 | 0.39 / 0.17 / 0.60 | 0.50 / 0.50 / 0.50 | 0.481 / 0.074 / 0.141 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.47 / 0.48 / 0.29; headline reads before CTA: 0.50 / 1.00 / 0.70.

### leaderboard (strict; engine n=10)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.46 / 0.55 / 0.34 | 0.49 / 0.25 / 0.50 | 0.107 / 0.138 / 0.264 | 0.300 / 0.264 / 0.342 |
| body | 0.72 / 1.00 / 1.00 | 0.53 / 0.55 / 0.77 | 0.50 / 0.55 / 0.50 | 0.104 / 0.228 / 0.291 | 0.159 / 0.100 / 0.117 |
| cta | 0.40 / 0.50 / 0.50 | 0.75 / 0.88 / 0.86 | 0.50 / 0.50 / 0.50 | 0.142 / 0.174 / 0.157 | 0.191 / - / - |
| logo | 0.64 / 0.40 / 0.40 | 0.17 / 0.50 / 0.09 | 0.50 / 0.26 / 0.25 | 0.025 / 0.051 / 0.024 | - / - / - |
| product | 0.76 / 0.90 / 0.90 | 0.48 / 0.28 / 0.64 | 0.50 / 0.50 / 0.50 | 0.456 / 0.105 / 0.091 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.02 / 0.03; empty space: 0.09 / 0.37 / 0.24; headline reads before CTA: 0.70 / 1.00 / 0.40.

### medium_rectangle (strict; engine n=15)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.50 / 0.50 / 0.33 | 0.46 / 0.20 / 0.58 | 0.082 / 0.118 / 0.265 | 0.095 / 0.090 / 0.148 |
| body | 0.84 / 1.00 / 1.00 | 0.50 / 0.50 / 0.47 | 0.55 / 0.43 / 0.39 | 0.055 / 0.172 / 0.240 | 0.038 / 0.038 / 0.046 |
| cta | 0.35 / 0.20 / 0.20 | 0.50 / 0.50 / 0.76 | 0.50 / 0.81 / 0.58 | 0.059 / 0.072 / 0.086 | 0.050 / - / - |
| logo | 0.50 / 0.27 / 0.27 | 0.50 / 0.21 / 0.80 | 0.62 / 0.89 / 0.15 | 0.010 / 0.021 / 0.026 | - / - / - |
| product | 0.36 / 0.73 / 0.73 | 0.50 / 0.26 / 0.79 | 0.50 / 0.41 / 0.30 | 0.549 / 0.140 / 0.104 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.38 / 0.41 / 0.22; headline reads before CTA: 0.52 / 1.00 / 1.00.

### wide_skyscraper (strict; engine n=16)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.51 / 0.50 / 0.50 | 0.44 / 0.20 / 0.20 | 0.078 / 0.138 / 0.205 | 0.190 / 0.167 / 0.211 |
| body | 0.86 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.53 / 0.35 / 0.69 | 0.088 / 0.141 / 0.183 | 0.089 / 0.065 / 0.077 |
| cta | 0.45 / 0.56 / 0.56 | 0.50 / 0.50 / 0.50 | 0.28 / 0.85 / 0.85 | 0.129 / 0.096 / 0.078 | 0.125 / - / - |
| logo | 0.52 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 | 0.55 / 0.04 / 0.04 | 0.011 / 0.017 / 0.011 | - / - / - |
| product | 0.34 / 0.81 / 0.81 | 0.50 / 0.50 / 0.50 | 0.52 / 0.58 / 0.58 | 0.550 / 0.216 / 0.216 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.31 / 0.35 / 0.23; headline reads before CTA: 0.46 / 1.00 / 0.89.

### billboard (strict; engine n=18)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.97 / 1.00 / 1.00 | 0.50 / 0.50 / 0.32 | 0.46 / 0.43 / 0.50 | 0.095 / 0.131 / 0.291 | 0.184 / 0.180 / 0.241 |
| body | 0.90 / 1.00 / 1.00 | 0.50 / 0.50 / 0.58 | 0.64 / 0.66 / 0.50 | 0.045 / 0.124 / 0.164 | 0.070 / 0.050 / 0.055 |
| cta | 0.23 / 0.44 / 0.44 | 0.68 / 0.88 / 0.80 | 0.83 / 0.50 / 0.50 | 0.041 / 0.059 / 0.106 | 0.058 / - / - |
| logo | 0.52 / 0.33 / 0.33 | 0.52 / 0.51 / 0.11 | 0.64 / 0.18 / 0.22 | 0.022 / 0.036 / 0.043 | - / - / - |
| product | 0.35 / 0.72 / 0.72 | 0.50 / 0.15 / 0.52 | 0.50 / 0.50 / 0.50 | 0.609 / 0.114 / 0.148 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.43 / 0.41 / 0.30; headline reads before CTA: 1.00 / 1.00 / 0.50.

### mobile_banner (nearest-aspect; engine n=20)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.38 / 0.55 / 0.32 | 0.50 / 0.36 / 0.50 | 0.125 / 0.198 / 0.299 | 0.342 / 0.250 / 0.340 |
| body | 0.75 / 1.00 / 1.00 | 0.51 / 0.41 / 0.80 | 0.66 / 0.51 / 0.50 | 0.048 / 0.224 / 0.238 | 0.126 / 0.130 / 0.102 |
| cta | 0.30 / 0.40 / 0.40 | 0.75 / 0.88 / 0.83 | 0.50 / 0.50 / 0.50 | 0.142 / 0.134 / 0.149 | 0.191 / - / - |
| logo | 0.35 / 0.40 / 0.40 | 0.18 / 0.08 / 0.08 | 0.50 / 0.22 / 0.22 | 0.034 / 0.040 / 0.037 | - / - / - |
| product | 0.45 / 0.75 / 0.75 | 0.50 / 0.28 / 0.52 | 0.50 / 0.50 / 0.50 | 0.456 / 0.103 / 0.059 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.45 / 0.34 / 0.30; headline reads before CTA: 0.50 / 1.00 / 0.88.

### half_page (strict; engine n=10)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 0.95 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.48 / 0.21 / 0.20 | 0.052 / 0.141 / 0.146 | 0.126 / 0.122 / 0.133 |
| body | 0.86 / 1.00 / 1.00 | 0.50 / 0.50 / 0.50 | 0.54 / 0.32 / 0.34 | 0.048 / 0.093 / 0.088 | 0.045 / 0.055 / 0.055 |
| cta | 0.42 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 | 0.44 / 0.83 / 0.85 | 0.070 / 0.079 / 0.075 | 0.062 / - / - |
| logo | 0.42 / 0.20 / 0.20 | 0.54 / 0.61 / 0.50 | 0.70 / 0.06 / 0.06 | 0.013 / 0.023 / 0.018 | - / - / - |
| product | 0.49 / 0.70 / 0.70 | 0.50 / 0.50 / 0.50 | 0.50 / 0.58 / 0.58 | 0.509 / 0.212 / 0.317 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.02 / 0.02; empty space: 0.23 / 0.44 / 0.34; headline reads before CTA: 0.56 / 1.00 / 1.00.

### large_leaderboard (nearest-aspect; engine n=13)

| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |
|---|---|---|---|---|---|
| headline | 1.00 / 1.00 / 1.00 | 0.40 / 0.55 / 0.33 | 0.49 / 0.25 / 0.50 | 0.100 / 0.176 / 0.214 | 0.336 / 0.306 / 0.378 |
| body | 0.70 / 1.00 / 1.00 | 0.48 / 0.55 / 0.60 | 0.50 / 0.56 / 0.50 | 0.104 / 0.236 / 0.209 | 0.173 / 0.100 / 0.111 |
| cta | 0.50 / 0.46 / 0.46 | 0.75 / 0.88 / 0.81 | 0.50 / 0.50 / 0.46 | 0.142 / 0.166 / 0.162 | 0.191 / - / - |
| logo | 0.55 / 0.61 / 0.61 | 0.11 / 0.21 / 0.06 | 0.50 / 0.32 / 0.40 | 0.027 / 0.051 / 0.022 | - / - / - |
| product | 0.80 / 0.69 / 0.69 | 0.49 / 0.23 / 0.69 | 0.50 / 0.50 / 0.50 | 0.446 / 0.069 / 0.069 | - / - / - |

Min edge margin (median) real/base/bisect: 0.00 / 0.00 / 0.00; empty space: 0.08 / 0.37 / 0.32; headline reads before CTA: 0.70 / 1.00 / 0.33.

## Biggest deviations from real layouts

### base: top 12 (scaled gap = |engine - real| / (real IQR/1.35, floored))

| scaled gap | target | role | metric | real | engine |
|---|---|---|---|---|---|
| 4.3 | instagram_post | product | cy | 0.500 | 0.328 |
| 3.4 | leaderboard | headline | cy | 0.491 | 0.251 |
| 2.7 | leaderboard | body | area | 0.104 | 0.228 |
| 2.6 | billboard | body | area | 0.045 | 0.124 |
| 2.5 | medium_rectangle | product | area | 0.549 | 0.140 |
| 2.4 | half_page | headline | area | 0.052 | 0.141 |
| 2.3 | leaderboard | product | area | 0.456 | 0.105 |
| 2.3 | leaderboard | logo | cy | 0.501 | 0.256 |
| 2.3 | medium_rectangle | product | cy | 0.500 | 0.410 |
| 2.2 | billboard | product | area | 0.609 | 0.114 |
| 2.2 | linkedin_banner | product | area | 0.609 | 0.118 |
| 2.1 | medium_rectangle | body | area | 0.055 | 0.172 |

### bisect: top 12 (scaled gap = |engine - real| / (real IQR/1.35, floored))

| scaled gap | target | role | metric | real | engine |
|---|---|---|---|---|---|
| 5.1 | medium_rectangle | product | cy | 0.500 | 0.297 |
| 4.3 | instagram_post | product | cy | 0.500 | 0.328 |
| 4.1 | leaderboard | body | area | 0.104 | 0.291 |
| 3.8 | billboard | body | area | 0.045 | 0.164 |
| 3.8 | billboard | headline | area | 0.095 | 0.291 |
| 3.7 | linkedin_banner | headline | area | 0.095 | 0.287 |
| 3.3 | medium_rectangle | body | area | 0.055 | 0.240 |
| 3.3 | leaderboard | layout | min edge margin | 0.000 | 0.033 |
| 3.2 | facebook_cover | headline | area | 0.081 | 0.263 |
| 3.0 | facebook_cover | body | area | 0.036 | 0.156 |
| 3.0 | linkedin_banner | body | area | 0.045 | 0.138 |
| 2.9 | youtube_thumbnail | body | area | 0.041 | 0.148 |

