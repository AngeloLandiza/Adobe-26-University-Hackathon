# challenge2: 50 real ads converted from Crello (EVALUATION ONLY)

**What it is:** 50 source ads that real designers made (Crello / VistaCreate ad templates), converted into our design format, for the A/B harness (`--split challenge2`). The ads in `challenge/` were all written here (by code and by Claude); these were not, so they test whether a change holds up on layouts nobody here drew.

**Licence rule:** the source is `cyberagent/crello` revision 5.1.0 (CDLA-Permissive-2.0), but the VistaCreate template terms are unverified. **Nothing that contains Crello content is committed.** The designs, rasters, previews, review sheets and per-template scan all live in `corpus/challenge2.nosync/`, which is gitignored (`*.nosync/`). This file and `make_real_challenge.py` hold only method and aggregate counts. A/B renders go to `ab/<variant>/renders.nosync/` and `ab/_sources.nosync/`, which are also gitignored.

**Same rule as `challenge/`:** never used for tuning or selection. Don't fit thresholds, templates or weights to these ads, and don't look at per-ad results to fix a particular ad.

## How to build and use it

From `distil-research/chameleon/` (needs the 4 Crello shards in `training-data/references/crello.nosync/data/`):

```
python -m dev.distill.make_real_challenge            # ~2 min with 4 processes; deterministic (--seed 0)
python -m dev.distill.ab_engine run --name base_chl2   --split challenge2 --workers 4
python -m dev.distill.ab_engine run --name bisect_chl2 --split challenge2 --workers 4 --env CHAMELEON_STRATEGIES=bisection
```

- `load_corpus("challenge2")` / `load_challenge2()` return these ads; `load_corpus()` and `include_challenge=True` never do.
- Their images are `cr2_*.png` in `challenge2.nosync/assets/`. `merged_assets_dir(*CHALLENGE2_FOLDERS)` merges them with the brand kit and the synthetic images. The harness does this only for `--split challenge2`, so no other split's assets folder changes.
- Each record carries `split: "challenge2"`, `origin: "crello"`, `brand: "crello"` (its own family in the A/B report), `features: [source aspect class, "photo" | "text-only"]`, and `crello_id` / `crello_format` (only in the gitignored JSON).
- Winner rows now also record `failed_gates`, the hard gates the returned design fails (normally `[]`).

## Method

**Pass 1, scan:** all 1,606 ad-format templates (the `AD_FORMATS` of `dev/references/crello_stats.py`, canvas ≥ 50 px) go through these steps:

1. **Roles** come from `dev/references/crello_roles.label`, the same heuristic behind the real-ad layout statistics. Element ids follow the quality gate's convention, which the engine never reads:
   - background → `bg`
   - decoration → `deco_N`
   - largest photo → `product`, or `deco_backdrop` if it covers ≥ 50% of the canvas (text placed on it is intended)
   - logo → `logo`
   - CTA text and its button → `cta_label` / `cta_bg`, grouped `cta`
   - largest other text → `headline`; the rest → `body_N`
2. **Elements:**
   - A ColoredBackground becomes a full-canvas rectangle.
   - A flat, opaque SvgElement becomes a `rectangle` shape. A flat elliptical one becomes an `ellipse`. Flatness is tested on a nearest-neighbour downsample, so stripes and texture stay rasters.
   - Every other SVG, image or masked-image element becomes Crello's own per-element raster, saved as PNG. It is cropped to the canvas when it bleeds off the edge and isn't rotated, and resampled to its box's aspect.
   - A full-canvas background photo becomes a `cover` image.
   - Opacity and rotation are kept.
3. **Text:**
   - Copied as is: content, line height, alignment, box and rotation. Letter spacing is kept too.
   - Content is upper-cased when Crello capitalizes it, or when the face is caps-only (Bebas Neue, Julius Sans One, …).
   - Colour is the majority colour; other colours and mixed bold become `style_runs`.
   - **Fonts:** the renderer has only Solstice Serif and Solstice Sans.
     - Serif, slab and display-serif faces → Serif.
     - Script and handwriting faces → Serif italic.
     - Everything else → Sans.
     - Heavy display faces, and text Crello marks bold, are bold.
   - **Fitting:** our faces set wider or narrower than Crello's, so:
     - The size shrinks in 1% steps until the text wraps into no more lines than Crello's and no line overflows the box. Letter spacing shrinks with it. The floor is 70% of the original size, or 55% for condensed faces such as Bebas Neue and Oswald.
     - The box then grows downward to the height the gate needs. Text draws from the top, so the glyphs don't move.
     - A grown box may not run into content it didn't touch in Crello. That includes the next line of copy and the bottom of its own button; otherwise it refits under that cap.
     - Box widths are trimmed to the widest line, on the side(s) the alignment leaves empty. Crello boxes are often much wider than their text, which skewed the contrast reading.
4. **Gate:** each template must pass `make_challenge.quality_issues`, i.e. `make_corpus.check(strict=True)` plus `pixel_check` (text fits; only `bg`/`deco*` may be off-canvas; no accidental overlap; image aspect; contrast ≥ 3:1; no busy backdrop). Each template is kept, or gets one skip reason: the first that applies.

**Pass 2, select:**

- **Quota per source aspect class** (engine classes; 50 in all): square 10, landscape 10, tall 10, wide 9, very_tall 6, strip 5.
- **Within a class**, half are text-only where enough exist, and canvas sizes are spread round-robin.
- **Ordering** is by a seeded hash.
- **One template per design family:** no two picks share a text of 8 or more characters. Crello has the same design resized to several formats.
- The picks are converted again, re-checked (all pass), written, and drawn on `challenge2.nosync/sheet.png` (plus `sheet_part1..5.png`), with Crello's own preview beside our render.

## Counts

| | count |
|---|---|
| ad-format templates scanned | 1,606 |
| passed the gate | 327 (≈ 262 distinct design families) |
| skipped | 1,279 |
| selected | **50** |

**Skip reasons** (first failing reason per template):

| reason | templates | what it means |
|---|---|---|
| `gate_overlap` | 604 | the design overlaps text with text, or text/images with a product or logo (e.g. layered display type, copy on a mid-size photo) |
| `gate_contrast` | 251 | some text below 3:1 against what is behind it (thin type on photos, text spanning two-colour blocks, low-contrast decorative text) |
| `font_substitution` | 221 | with our two faces the text can't keep Crello's line count inside its box at ≥ 70% (55% condensed) of the size, or the grown box would collide |
| `text_off_canvas` | 68 | text that starts outside the canvas (bleeding display type) |
| `gate_busy_backdrop` | 61 | text on a busy photo area (Sobel > 40) |
| `unsupported_glyphs` | 47 | characters our faces lack (Cyrillic, emoji, …) |
| `gate_off_canvas` | 18 | a rotated product, logo or text partly off the canvas |
| `gate_image_aspect` | 9 | a raster that couldn't be resampled to its box aspect within 2% (tiny elements) |

No template had to be skipped for missing rasters, gradient backgrounds or a converter error, and every ad-format template has text.

**Bias to keep in mind:** the gate prefers clean layouts. 85% of the scanned templates have a photo/hero image (≥ 5% of the canvas), against 76% of those that pass. Layered display type and text on busy photos are under-represented.

| source aspect class | scanned | passed | selected (photo / text-only) |
|---|---|---|---|
| square | 414 | 95 | 10 (5 / 5) |
| landscape | 400 | 78 | 10 (5 / 5) |
| tall | 399 | 66 | 10 (5 / 5) |
| wide | 284 | 66 | 9 (5 / 4) |
| very_tall | 72 | 15 | 6 (6 / 0: the one text-only template that passed shares a family with a pick) |
| strip | 37 | 7 | 5 (3 / 2) |
| **all** | **1,606** | **327** | **50 (29 / 21)** |

**Sizes covered (26):** 300x250 ×4; 728x90, 160x600, 600x1200 and 2560x1440 ×3 each; 1080x1080, 1128x191, 1200x628, 1296x1728, 1500x500, 1600x400, 336x280, 360x504, 540x810, 600x200, 735x1102, 851x315 and 940x788 ×2 each; 1024x512, 1190x1683, 1200x600, 1280x720, 1920x1080, 3000x1055, 420x654 and 560x315 ×1 each. These span 23 Crello formats (Twitter, Medium Rectangle, Skyscraper, Leaderboard, YouTube, Tumblr, Pinterest, Facebook cover, LinkedIn cover, email header, posters, …).

**Elements:** 2 to 19 per ad (median 8), 390 in all:

| kind | count |
|---|---|
| raster images | 189 (23 `bg`, 8 `deco_backdrop`, 4 `product`, 14 `logo`, 8 button rasters, the rest decorations) |
| text | 137 (48 headlines, 19 CTA labels, 70 body/other) |
| shapes | 64 (62 rectangles, 2 ellipses) |

32 elements are grouped (CTA button + label), 4 are rotated, and 2 texts carry style runs. Text faces: Sans bold 58, Sans regular 40, Serif italic (scripts) 16, Serif bold 12, Serif regular 11.

## Visual check (Claude, on the side-by-side sheets)

All 50 conversions match their Crello previews in layout, colours, images, backgrounds, decorations, button shapes and text positions. Expected differences:

- **Typeface:** everything is set in Solstice Serif or Sans. Scripts read as serif italic, and condensed display faces as a wider, smaller bold sans.
- **Weight:** some light Crello faces come out bold where Crello flagged them bold.
- **Blend effects:** one skyscraper's photo, which Crello tints with a blend effect, comes out slightly darker.

Bugs found on the sheets and fixed before the final build:

- striped SVG backgrounds were collapsed into flat rectangles (the flatness test smoothed them);
- Cyrillic text rendered as missing-glyph boxes (now skipped);
- caps-only faces showed mixed case;
- letter spacing didn't shrink with the type;
- too-wide text boxes made the contrast check fail;
- the same design family was picked in two sizes.

## First runs (2026-10-21, CPU only, heuristic priority, no judge)

623 cells: 50 ads × 13 targets, minus the targets within 2% of an ad's own aspect.

| variant | cells | errors | fallback winners | winners failing a hard gate | mean s/cell |
|---|---|---|---|---|---|
| `base_chl2` (no env) | 623 | 0 | 169 (27.1%) | 4 (all `G9_hidden`, all fallback; 3 from one ad) | 0.39 |
| `bisect_chl2` (`CHAMELEON_STRATEGIES=bisection`) | 623 | 0 | 106 (17.0%) | 3 (`G9_hidden`, the same ad, fallback) | 0.77 |

- **Picks that differ between the two:** 389 of 623 (62%); 234 are identical. 384 bisection winners come from the bisection strategy.
- **Fallback rate:** the base engine falls back far more often on real ads (27%) than on the drawn challenge ads (`base_chl`, 12.4%) or the holdout (`base`, 6.8%). Real layouts are where the template families run out.
- **Not yet judged.** The bisection run used `bisection.py` as it was on disk during an ongoing edit, without `CHAMELEON_BISECT_V2`.
