# Layout archetypes real designers use, per ad size

Unsupervised clustering of role-labelled Crello templates (all 4 test shards, 1,606 ad-format templates after filtering), per target size, compared with today's engine (`base`) and bisection (`bisect`) on the 192 holdout cells, then turned into testable floor plans (`training-data/references/archetype_floorplans.json`, 26 plans). Scripts: `chameleon/dev/references/arch_extract.py`, `arch_engine_dump.py`, `arch_cluster.py`, `arch_floorplans.py`, `arch_check.py`. Plots and diagrams: `training-data/references/archetypes.nosync/` (gitignored: `pca_*.png`, `boxes_*.png`, `sheet.png`, `check.json`).

## Method (and its limits)

- Vector per layout: for headline, body (union of body text), hero/product, CTA, logo: `[present, cx, cy, w, h]` as canvas fractions (25 numbers). Headline, body and hero have weight 1; **CTA and logo have weight 0.4 because their role labels are only ~25-30% precise** (see `real-ad-layouts.md`); treat any CTA or logo position below as low-confidence.
- Pools: a size with at least 40 strict-aspect templates uses its own pool (instagram_post 231, story 162, youtube_thumbnail 153, facebook_cover 127, medium_rectangle 183, half_page 43). Smaller sizes use the **aspect-class pool** (nearest 100 by aspect within the class): instagram_portrait (tall class, 100 of 399), linkedin_banner and billboard (shared `wide` pool, 100 of 284), leaderboard / mobile_banner / large_leaderboard (shared `strip` pool, only **37** templates, aspects 5.9-8.1), wide_skyscraper (very_tall class, 72). Strip results are thin.
- k-means with k in 2..6 chosen by silhouette; Gaussian mixture (diagonal) chosen by BIC as a cross-check (ARI between the two is reported). **Silhouette is low everywhere (0.36-0.40) and prefers k=2**, which mostly means "hero image present or not". Where n >= 40 and k=3 is within 0.05 of the best silhouette I used k=3 (a deliberate tie-break toward finer structure). BIC prefers more clusters (4-6) on the large pools. Honest reading: real layouts form two broad families (text-only; big hero plus text) with continuous variation inside them, not sharply separated archetypes. Clusters under 5% were dropped (none triggered).
- Medoid = most typical real layout of the cluster (sum of distances). Spread = mean distance to the cluster centroid.

## Archetypes per size

Descriptions use the cluster's median boxes; "text centre" mostly means the text sits over or beside a centred block. Percentages are share of the pool.

| size (pool) | k, silhouette, BIC-k, ARI | archetypes |
|---|---|---|
| instagram_post (231) | 2, 0.39, 6, 0.41 | **A1 55%** text-only: headline + body centred, no hero. **A2 45%** hero image ~51% of canvas, centred, headline and body in/over it |
| instagram_portrait (class tall, 100) | 3, 0.40, 6, 0.55 | **A1 60%** text-only, body-heavy (32% of canvas). **A2 29%** hero ~49% centred + headline + body. **A3 11%** hero ~51% with a big CTA-like element (low confidence) |
| story (162) | 3, 0.37, 5, 0.60 | **A1 59%** hero ~49% centred, headline, short body, CTA near bottom?. **A2 26%** text-only, headline top. **A3 15%** hero ~38% + large headline |
| youtube_thumbnail (153) | 2, 0.38, 4, 0.64 | **A1 56%** text-only (headline, body, boxed CTA?). **A2 44%** hero ~50% centred plus headline/body |
| facebook_cover (127) | 3, 0.36, 4, 0.73 | **A1 56%** text-only centre. **A2 27%** hero ~50% on the left, text right. **A3 17%** hero ~58% centred, headline, CTA? |
| wide: linkedin_banner, billboard (100) | 3, 0.37, 5, 0.74 | **A1 43%** text-only. **A2 34%** hero ~52% centred with text. **A3 23%** big headline + CTA? at right |
| strip: leaderboard, mobile_banner, large_leaderboard (37) | 2, 0.37, 2, 1.00 | **A1 57%** hero ~48% + headline left, body and CTA? right. **A2 43%** text-only, large headline, small body below |
| medium_rectangle (183) | 3, 0.36, 4, 0.74 | **A1 51%** text-only, body-heavy. **A2 36%** hero ~49% centred. **A3 13%** headline-only |
| wide_skyscraper (class very_tall, 72) | 3, 0.38, 4, 0.93 | **A1 47%** text-only. **A2 43%** hero ~53% + headline + body (27%). **A3 10%** headline low, big CTA? |
| half_page (43) | 3, 0.38, 3, 0.95 | **A1 51%** text-only. **A2 40%** hero ~48% + text. **A3 9%** hero ~60% with a CTA? (4 templates, no headline label; dropped from plans) |

Across sizes the one robust finding: **about 35-65% of real ads carry a hero image, and when they do it covers about 50% of the canvas (median 0.43-0.55 by size), usually centred and overlapped by the text**. Strong sizes for this (hero present 57-66%): story, strip.

## Where today's engine and bisection sit

Distance = Euclidean distance in the weighted vector space to the nearest archetype centroid (real layouts' own median is 0.64 averaged over sizes; 0.58 using headline/body/hero only, which removes the CTA/logo labelling confound since the engine labels every CTA/logo).

| size | real median | base | bisect | frac beyond real 90th pct (base / bisect) | hero share of canvas when present: real (present %) / base / bisect |
|---|---|---|---|---|---|
| instagram_post | 0.68 | 0.89 | 0.90 | 0.36 / 0.36 | 0.50 (45%) / 0.13 / 0.12 |
| instagram_portrait | 0.63 | 0.78 | 0.77 | 0.08 / 0.17 | 0.43 (35%) / 0.17 / 0.18 |
| story | 0.63 | 0.76 | 0.83 | 0.33 / 0.56 | 0.49 (66%) / 0.12 / 0.15 |
| youtube_thumbnail | 0.69 | 0.72 | 0.93 | 0.13 / 0.47 | 0.50 (44%) / 0.27 / 0.10 |
| linkedin_banner | 0.55 | 0.75 | 1.03 | 0.30 / 1.00 | 0.51 (34%) / 0.12 / 0.09 |
| billboard | 0.55 | 0.76 | 0.99 | 0.44 / 0.94 | 0.51 (34%) / 0.11 / 0.15 |
| facebook_cover | 0.59 | 0.78 | 0.99 | 0.29 / 0.81 | 0.48 (44%) / 0.07 / 0.14 |
| leaderboard | 0.73 | 0.88 | 1.09 | 0.20 / 0.90 | 0.46 (57%) / 0.10 / 0.09 |
| mobile_banner | 0.73 | 0.88 | 0.99 | 0.35 / 0.65 | 0.46 (57%) / 0.10 / 0.06 |
| large_leaderboard | 0.73 | 0.77 | 1.07 | 0.23 / 1.00 | 0.46 (57%) / 0.07 / 0.07 |
| medium_rectangle | 0.59 | 0.80 | 0.94 | 0.33 / 1.00 | 0.55 (36%) / 0.14 / 0.10 |
| wide_skyscraper | 0.61 | 0.78 | 0.83 | 0.12 / 0.50 | 0.52 (43%) / 0.22 / 0.22 |
| half_page | 0.62 | 0.73 | 0.75 | 0.20 / 0.30 | 0.51 (49%) / 0.21 / 0.32 |
| **mean** | **0.64** | **0.79** | **0.93** | **0.26 / 0.67** | |

- Base sits modestly outside real practice (median distance 0.79 vs 0.64; 26% of its layouts are farther from any archetype than 90% of real layouts).
- **Bisection is farther from real practice than base** (0.93; 67% beyond the real 90th percentile, 94-100% on the wide/strip sizes and medium_rectangle), even though it scores better with the judge: it improves on the grader's terms while moving away from how designers lay things out. PCA plots (`pca_*.png`) show engine layouts landing between the text-only and hero clouds: a small hero (10-15%) that matches neither family.
- The confirmed key gap: the engine's hero is 7-27% of the canvas versus ~50% in real ads.
- Caveats: a single labeller, CTA/logo noise, engine roles come from its own plan, and small-sample class pools for strip.

## Floor plans and offline check

`archetype_floorplans.json`: 26 plans named `arch_<size>_<k>` for strict-pool sizes and `arch_<class>_<k>` for class pools (`arch_wide_*`, `arch_strip_*`, `arch_very_tall_*`, `arch_instagram_portrait_*`); `classes` = the size's aspect class. Geometry only. All 26 load through `_parse_plan` with no skips. Conversion (medoid boxes to content-box fractions) had to bend the real layout because the format forbids overlapping slots and requires `text` and `minor`:

- headline (+ touching body) becomes `text`; otherwise body becomes `minor`; no body: `minor` is a free band (at most 20% thick).
- text is at least 35% x 14% of the content box.
- real ads put text on top of the hero; that overlay cannot be expressed. When the hero covers at least 25%, a text band (30% of height, or 38% of width on landscape/wide/strip) is reserved and the hero is trimmed to the rest; overlapping CTA/logo are cut or dropped. So the hero plans are "hero ~50-60% beside/above the text", not the real overlay.
- half_page cluster 3 had no headline in its medoid and is skipped.

Offline gate check (no teacher; `arch_check.py`): for 115 holdout cells (up to 20 per aspect class; fewer exist for some), every plan of the cell's class was built at 2 margins and gated.

- Candidates: **1011 of 1013 pass the gates (99.8%)**; cell-plan pairs with at least one passing candidate: 506 of 510 (99.2%); every one of the 115 cells has at least one passing archetype candidate. On each plan's own size: 579 of 579 candidates pass.
- Failures: `arch_instagram_post_2` on a long-copy medium_rectangle cell (G9_hidden, text hidden) and `arch_strip_1` built nothing on 3 mobile_banner cells (320x50, too small for its three columns).

## Visual verdict (`archetypes.nosync/sheet.png`: original | today | bisection | best archetype candidate, 28 cells)

Honest read: the archetype candidates look like plausible, clean designs and they do fix the hero size (story_1 with a large bottle and a headline above it, and the facebook_cover and youtube_thumbnail ones with a big product beside text look like real ads). But they are **not clearly better than today**. The reserved text bands and minimum text slots give small, sometimes tiny text (leaderboard/mobile strips, story_2), long copy is squeezed or clipped (medium_rectangle with long headline), logos/CTAs lost to the overlap rules, and hierarchy is weaker than bisection on text-heavy ads. They look different, closer to real practice on hero size, rather than better. The gate numbers only say they are valid. No judge score was run, so whether they win is untested.

## Recommendation: what to A/B first

1. Hero-first archetypes at the sizes where real ads have a hero most often and the engine's hero is smallest: `arch_story_1` (59% of story), `arch_facebook_cover_2` and `_3`, `arch_youtube_thumbnail_2`, `arch_medium_rectangle_2`, `arch_wide_2`, `arch_strip_1` (leaderboard family, thin evidence).
2. Offer them as extra candidates only for ads that have a hero image and short copy (the long-copy cases clip), alongside existing candidates, so the judge decides.
3. Skip the text-only archetypes (`_1` plans) for now; they resemble what `stack`/`columns` already produce.
4. A better follow-up than these plans: let hero and text overlap (a hero-behind-text slot type); that is how real ads reach 50% hero, and the current format cannot say it.

## Result: added to bisection, the grader almost never picks them (2026-10-21)

Run: `ab_engine run --name bisect_arch --env CHAMELEON_STRATEGIES=bisection --env CHAMELEON_EXTRA_TEMPLATES=../training-data/references/arch_selected_templates.json --split holdout` (the shipped plans plus the 7 hero-first archetypes above).

- **1 of 192 holdout picks changed** vs bisection alone (`arch_facebook_cover_2`). Nothing to judge, so no teacher calls were spent.
- Why (`dev/references/arch_rank_diag.py`, every third holdout cell): archetype candidates are built in 54 of 64 cells and **all 176 pass the gates**, but the best one ranks **15th (median)** among the passing candidates, often last.
- What pushes them down (`dev/references/arch_metric_gap.py`, weighted metric gap, winner minus best archetype): legibility +0.029 (largest), then degradation, spacing, hierarchy (+0.013 each) and faithfulness +0.011. The hero takes the room, so the text gets smaller and moves farther from the source layout. Same flaw as the visual verdict.

**Conclusion:** side-by-side hero plans cost too much text size to win, and the grader is right to reject them on legibility. Real ads reach a ~50% hero by putting the text **over** the image, which the floor-plan format can't express. The next real step is a hero-behind-text slot type (a background hero slot with text slots allowed on top, plus a contrast check). That is an engine change, not a new template.
