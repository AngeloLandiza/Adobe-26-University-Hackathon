# Design rulebook: what the teacher punishes, as checkable rules

*Strategy S3 `rule_repair`, 2026-10-20. Code: `chameleon/engine/pipeline/layout/strategies/rules.py` (rules and penalty), `chameleon/dev/rule_repair/mine.py` (mining and calibration).*

## How it was made

1. **Themes.** All 1,955 critiques (`reason` field) in `judgments/judgments.jsonl`, `judgments/d2/judgments.jsonl`, `floor-plans/judgments.jsonl` and `floor-plans/eval-judgments.jsonl` were tagged with keyword patterns (`mine.py themes`). The patterns are coarse, so treat the shares as rough. For example, "edge" also matches "the edge of the circle".
2. **Measurements.** For every judged cell, the candidates were rebuilt exactly as the pool builds them: heuristic plan, merged assets, `safe_fallback` plus `generate_candidates`. That gave 495 cells and 1,642 judged candidates. Floor-plan ids (`p…`, `b2_…`) are missing, because the standard generator doesn't build them. Each candidate got the raw measurements of `rules.measure` (`mine.py calibrate`).
3. **Thresholds.** For each measurement, candidates the teacher scored 1–2 were compared with those scored 3–4 (there are no 5s). The cut that best separates the two groups (Youden's J) set the threshold, rounded to a readable value.
4. **Does the rule agree with the judge?** "Pairwise agreement" covers the 634 judgements where the two candidates got different scores and both were rebuilt. It is how often the candidate with the lower rule penalty was the one the teacher scored higher, counting only pairs the rule tells apart. 0.5 is chance.
5. **Weights.** I set weights by hand from critique frequency and agreement, then checked them as a whole. The full penalty picks the teacher's better candidate in **74.5% of train pairs and 76.4% of holdout pairs**. The engine's current grader (`geometry_total`) manages **54.9% / 65.0%** on the same pairs. Weights fitted by logistic regression did slightly worse on holdout (75.5%). A coordinate search raised train accuracy to 79% but dropped holdout to 72–75%, so it overfits. The hand weights stay.

Sizes are in **rendered px**. The teacher sees renders whose long side is 768 px, so a font's rendered size is `font_size × min(1, 768 / max(W, H))`. "Short side" means `min(W, H)` of the target canvas. A unit's box is the union of its members' drawn boxes. For text, the box is the ink (the widest line placed per `text_align`, and the measured block height), not the slot it was given.

## The rules, ranked by weight × how often they fire

| # | Rule | Threshold | Weight | Fires on (judged candidates) | Pairwise agreement |
|---|---|---|---|---|---|
| R1 | Every text at its role's legible size | rendered px: headline 28, CTA 14.5, subheadline 13, other text 11, badge 8, legal 6 (other 10); never below the engine floor `place.floor_px` | 3.0 | 66% | 0.57 (n=431) |
| R8 | Reading order matches the source | every content-unit pair keeps its top-to-bottom / left-to-right order | 2.0 | 73% | **0.68** (n=487) |
| R2 | No dead region | largest empty rectangle ≤ 22% of the canvas | 2.0 | 41% | 0.62 (n=329) |
| R10 | No new collisions | no overlap between content units beyond what the source had | 2.0 | 15% | **0.69** (n=159) |
| R3 | Different roles never touch | gap ≥ 1% of the short side between units of different roles | 1.0 | 39% | 0.65 (n=303) |
| R7 | Even spacing | coefficient of variation of nearest-neighbour gaps ≤ 0.9 | 1.0 | 39% | 0.66 (n=349) |
| R13 | Hero product not an icon | largest hero/secondary image ≥ 12% of the canvas | 1.0 | 36% | **0.69** (n=290) |
| R9 | Size hierarchy matches the source | text pairs keep their font-size order | 1.0 | 35% | 0.64 (n=246) |
| R4 | Edge margin | each unit ≥ 5% of the short side from every edge (or as close as the source designer put it) | 1.0 | 38% | 0.53 (n=337) |
| R5 | Logo not a speck | logo's short side ≥ 24.5 rendered px | 1.0 | 9% | **0.71** (n=69) |
| R6 | Doesn't look shrunk | content bounding box ≥ 55% of the canvas | 0.5 | 30% | 0.52 (n=270) |
| R11 | No orphan word | no wrapped text whose last line is one word under 35% of the widest line (unless the source already did it) | 0.5 | 12% | 0.64 (n=92) |
| R12 | Shared alignment lines | each unit shares a left/centre/right or top/middle/bottom line (1% tolerance) with another | 0.5 | 28% | 0.58 (n=202) |

Each rule's penalty is in [0, 1]. R1 and R4 are weighted by unit priority (`plan.weight`), and R1 uses a square-root hinge so a little too small already counts. The total is the weighted sum (`rules.penalty`).

## Each rule: why, evidence, measurement

**R1 Text at a legible size.** It is the most frequent complaint (text too small in ~70% of critiques): *"The 'Taste the Glow' label and the 4-packs line are both tiny and hard to read."* · *"the subline is shrunk to about 7px and is unreadable."* Measured as each text unit's smallest font in rendered px, against the per-role cut. Candidates the teacher scored 1–2 had a mean smallest text of 9.0 px; those scored 3–4 had 10.9 px. The CTA separates best: 13.7 px vs 15.8 px, pairwise 0.63. The teacher also calls a 768 px square headline "undersized" well above the engine's 14 px floor. The cut is 28 rendered px.

**R8 Reading order.** It is the strongest single predictor: *"Both put the 'Taste the Glow' CTA at the very top, above the product and headline, so the reading order is reversed."* Measured with the same pairwise row-band order as `metrics._faithfulness` (centres within 10% of the height read as one row). Low-scored candidates agree with the source on 69% of pairs, high-scored on 82%.

**R2 Dead space.** *"The bottom-left half of the canvas is dead space."* · *"leaves a large empty gap above the button."* Measured as the largest axis-aligned empty rectangle on a 48-cells-per-long-side grid, counting only content units (backgrounds and decorations count as empty). The cut is 22% of the canvas.

**R10 Collisions.** *"the CTA button sits right against the end of 'alive' … the headline and the button collide."* Measured as the extra overlap between content-unit boxes beyond the source's own (text deliberately set on a photo doesn't count).

**R3 Cramped.** *"the badge row is crammed against the CTA."* · *"the headline runs straight into the CTA button."* Measured as the closest gap between units of *different* roles, as a share of the short side. A row of same-role chips may sit tight. A first version counted every pair, and the repair then pulled chip rows apart (see the S3 report).

**R7 Uneven gaps.** *"The three elements are spread out with uneven gaps and big dead areas, so the group falls apart."* Measured as the coefficient of variation of the nearest-neighbour gaps: 0.93 for low-scored candidates, 0.73 for high-scored.

**R13 Product shrunk to an icon.** *"The hero can has shrunk to a tiny icon in the bottom corner."* · *"the product bag is a tiny thumbnail at the far right."* Measured as the area share of the largest hero or secondary image.

**R9 Hierarchy.** *"the italic subhead is set almost as large as the headline, so the hierarchy collapses."* · *"The fine print is as loud as the headline."* Measured as the share of text-unit pairs whose font-size order matches the source.

**R4 Edge margins.** *"its headline runs almost to both canvas edges (about 25px margins)."* · *"jams the CTA about 12px from the top edge."* Measured as each non-full-bleed unit's distance to the nearest edge. Pairwise agreement is close to chance (0.53): within a cell, candidates rarely differ on it. So it has a low weight in the penalty, and the repair enforces it as a hard constraint instead (nothing may move into a 3% edge band).

**R5 Logo too small.** *"The SOLSTICE logo has shrunk to an illegible speck."* Measured as the logo box's short side in rendered px. It rarely fires on judged candidates (9%), but agrees with the judge in 71% of the pairs it decides. The critiques also punish *oversized* logos (*"an oversized centred logo that dominates the middle and outranks the message"*). The repair never lets a logo grow past ¼ of the headline's area or 1.5× its height.

**R6 Looks shrunk.** *"it reads like the original shrunk into a tall frame."* Measured as the content bounding box's share of the canvas. Within-cell agreement is at chance (0.52): a shrunk look shows up better through R1, R2 and R13. Low weight.

**R11 Orphan word.** *"the headline wraps to leave 'sunshine.' alone on the second line."* Measured from the renderer-exact wrap (`TextMeasurer.wrap`).

**R12 Alignment.** *"It also mixes a left-aligned text block with a centred phone and CTA."* Measured as the share of units sharing an alignment line. Weak signal.

## Complaint themes (keyword counts over 1,955 critiques)

| Theme | Share | Encoded as |
|---|---|---|
| edge / margin / corner | 71% | R4 (+ edge band in the repair) |
| text too small | 70% | R1 |
| logo (any mention) | 60% | R5 (+ logo caps in the repair) |
| cramped / jammed / touching | 58% | R3, R10 |
| separated / stranded / floating | 53% | R7; the repair keeps clusters together and stops units drifting away |
| dead / empty space | 49% | R2 |
| looks shrunk | 45% | R6, R13 |
| hierarchy | 36% | R9 |
| reading order | 34% | R8 |
| alignment | 31% | R12 |
| collision / text on a decoration | 30% | R10; the repair forbids new text-across-decoration-edge clashes |
| product image small | 21% | R13 |
| orphan word | 12% | R11 |
| uneven gaps (named explicitly) | 6% | R7 |

**Not encoded (yet):** a decoration detached from its product (*"the can floats off-centre from its backing circle"*); in the repair, decorations follow the unit they sit behind. Also not encoded: illegible text over a busy or contrasting decoration (a pixel rule), flavour names separated from their descriptors (needs semantic pairing), and crops of decorations at the canvas edge.
