# corpus: source ads

**What it is:** the ads the whole loop learns from. Each file is one source design in the challenge's schema (`chameleon_design_schema_v0.9.json`): a canvas plus its elements (text, images, shapes) at the original size. The engine resizes each one to many target sizes, and the teacher judges the results.

**Why it exists:** the challenge ships only 3 sample ads. Learning from 3 ads would just memorise them, and the hidden test set has many designs the engine has never seen. More, varied ads make the data, and anything learned from it, generalise.

**File format** (`designs/<task_id>.json`):

```json
{
  "task_id": "c07_portrait_long_copy_v2",
  "split": "holdout",
  "origin": "augmented",
  "brand": "offbrand",
  "parent": "c07_portrait_long_copy",
  "augmentations": ["palette:harbor(inverted)", "jitter:7-units", "remove:ships"],
  "expects_missing_asset": false,
  "notes": "what this ad tests",
  "features": ["light", "portrait", "..."],
  "source": { "canvas": { "width": 1080, "height": 1350 }, "elements": [ ... ] }
}
```

- `split`: `train` (used for learning) or `holdout` (only for checking results, never trained on). **An augmented variant always has its parent's split**, so a holdout design never leaks into train in disguise.
- `origin`: `sample` (one of the 3 official samples), `claude-drafted` (written by Claude, reviewed against renders), `augmented` (a seeded variant of a `claude-drafted` ad, made by `dev/distill/augment.py`), `offbrand` (a hand-written ad for an invented non-Solstice brand, `dev/distill/make_offbrand.py`) or `imported`.
- `brand`: `solstice` or `offbrand`. Samples and drafted ads are `solstice`; off-brand ads are `offbrand`; an augmented variant becomes `offbrand` when its palette or its logo was swapped (it no longer carries Solstice's colours or mark) and otherwise stays `solstice` (a product, background or copy change alone keeps the brand). The hidden test ads are probably Solstice-branded, so holdout accuracy is reported per brand: Solstice holdout measures "does it help on the real brand", off-brand holdout measures "does it generalise".
- `parent` and `augmentations` (augmented only): the drafted ad it came from and the list of operations applied (2 to 4), e.g. `canvas:1080x1080->1080x1350`, `copy:headline=long`, `image:product->syn_prod_sneaker.png`, `remove:badges`, `add:badge(NEW)`, `type:headline-sans`, `palette:grove`, `jitter:5-units`.

**Images:** ads may use the brand kit's three images (`chameleon/brand_kit/assets/`) or the 25 code-drawn stand-ins in [`assets/`](assets/) (all named `syn_*`, see its README; research data only, never shipped). `dev/distill/assets.py` merges both folders into one temporary assets directory per run; the pool, the review sheets and the tests all render through it.

**How it was made:** see "Generation log" below. Fonts are always Solstice Serif and Solstice Sans (the only families the renderer has); variety comes from canvas size, element count, groups, badges, rotation, long text, legal lines, missing assets, colours, images, weights and spacing.

**Feeds:** `candidate-pool/`, and through it everything else.

**Not in here: the challenge set.** `corpus/challenge/` holds 40 evaluation-only ads (new brands, images, structures and sizes) for measuring generalisation in the A/B harness. `load_corpus()` never returns them; see [`challenge/README.md`](challenge/README.md). A second, real-ad challenge, `challenge2` (50 Crello ad templates converted into our format, split `challenge2`, brand family `crello`), lives in the gitignored `corpus/challenge2.nosync/` because its content is Crello's. `load_corpus("challenge2")` returns it, and only method and aggregate counts are committed: see [`CHALLENGE2.md`](CHALLENGE2.md).

**Checked by:** `chameleon/tests/test_corpus.py`: ≥ 30 drafted, ≥ 80 augmented, ≥ 25 off-brand ads; ≥ 8 Solstice and ≥ 6 off-brand holdout; unique ids; every augmented ad's split equals its parent's; off-brand ads use only `syn_` images; every asset exists in one of the two folders and every font in the kit; generated ads pass the layout checks; every ad plans (with the right missing-asset flags) and renders through the merged assets folder; and the quality gate really rejects text on a busy area, a stretched image and overflowing text.

## Generation log

### 2026-10-20: variety pass (153 ads: + 90 augmented, + 30 off-brand, + 25 synthetic images)

**Why:** the first corpus used 3 images, 2 font families and one palette, so a grader trained on it could learn "what looks good for Solstice". This pass adds variety to train, keeps a Solstice holdout, and adds a separate off-brand holdout to measure generalisation.

**Commands** (from `distil-research/chameleon/`, in this order; each is deterministic, and a rerun rewrites only files whose content changed):

```
python -m dev.distill.synth_images   # 25 PNGs + README in corpus/assets/
python -m dev.distill.make_corpus    # the 33 originals (now with brand), review.nosync/sheet.png
python -m dev.distill.augment        # 90 variants, review.nosync/sheet_augmented.png, review.nosync/augment_dropped.txt
python -m dev.distill.make_offbrand  # 30 off-brand ads, review.nosync/sheet_offbrand.png
```

**Counts (origin x brand x split):**

| origin | brand | train | holdout | total |
|---|---|---|---|---|
| `sample` | solstice | 3 | 0 | 3 |
| `claude-drafted` | solstice | 20 | 10 | 30 |
| `augmented` | solstice | 31 | 19 | 50 |
| `augmented` | offbrand | 29 | 11 | 40 |
| `offbrand` | offbrand | 22 | 8 | 30 |
| **all** | | **105** | **48** | **153** |

By brand: Solstice 83 (54 train, 29 holdout), off-brand 70 (51 train, 19 holdout).

**Augmentation** (`augment.py`): 3 variants for each of the 30 drafted ads (never the official samples), each combining 2 to 4 operations picked by a seeded RNG (49 variants have 3 operations, 21 have 2, 20 have 4). How often each was used: canvas 40, remove 38, image 37, type 36, copy 32, add 31, jitter 28, palette 27.
- `palette`: 8 invented palettes (harbor, grove, ember, ultra, bloom, glacier, citrine, slate; high- and low-contrast), mapped role by role from Solstice's colours (plum to dark, cream to light, orange to accent, and so on), 30% of the time inverted light/dark. The gradient background becomes the palette's matching synthetic background, the white logo a synthetic logo of the right tone, and the brand name in copy is replaced (Solstice to e.g. Harbor; "next solstice" the day is left alone).
- `image`: the bottle becomes a synthetic product of a different aspect (tall bottle, wide box, sneaker, phone, pouch), refitted inside the old box; the gradient becomes another dark synthetic background; the logo becomes a wordmark-blocks or round badge logo, keeping the old visible left edge when left-aligned.
- `copy`: shorter or longer headline / subhead / legal / body copy, written out as data in `augment.COPY`; the box height is set to what TextMeasurer measures and the column below is restacked, or the font shrinks (to 75%) if restacking would collide.
- `remove` / `add`: drop badges, a secondary line, legal, the logo or a decoration (closing the gap); add a badge pill (grouped shape + label, corner sticker or beside the button), a translucent decoration, or a legal line in free space at the bottom.
- `canvas`: re-lay onto a nearby size (e.g. 1080x1080 to 1080x1350 or 1200x1200, 728x90 to 970x90, 300x250 to 336x280), scaling uniformly and spreading the extra space over the rows or columns. Big jumps (story to banner) are not attempted: proportional scaling cannot make them look designed.
- `type`: headline serif/sans swap, weight, size (x0.85 to x1.15), subhead italic, tracking on small caps.
- `jitter`: whole units moved by up to 2% (x) / 1.5% (y) of the canvas; units sharing a left edge share one dx, centred units stay centred, rows (badges, lineups) move together.

**Quality gate:** after every operation the ad must pass `make_corpus.check(strict=True)`: text fits its box by both the engine's `TextMeasurer.fits` and the corpus's own estimate; nothing but `bg`/`deco*` off the canvas; no accidental text overlap and (strict) no image/shape partly overlapping other content; non-cover images in their asset's aspect ratio. The finished ad must also pass `make_corpus.pixel_check`: every text at least 3:1 contrast against the darkest/lightest 10% behind it (or no worse than the parent ad, where the parent was already below 3:1), and no text on a busy area (mean Sobel gradient > 40 behind the box). A failing attempt is dropped and logged, and the slot retried with the next seed (up to 10 tries).

**Dropped attempts (11, all for contrast, all replaced by a later seed):** listed in `review.nosync/augment_dropped.txt`. Mostly inverted palettes where a button label or small accent text fell below 3:1 (e.g. `c20_leaderboard_dark_v2` with slate inverted: button label 2.01; `c06_portrait_lineup_v3` with citrine inverted: delivery line 1.25), plus `c10_story_ritual_steps` variants where the orange step numbers fell below the parent's own 2.84, and `c19_mrec_weekend_sale_v2` (legal 2.97). No slot ran out of attempts.

**Off-brand ads** (`make_offbrand.py`, 30 ads, 22 train + 8 holdout, two holdout per brand): four invented brands, each with its own palette, Quillbrook Roasters (coffee, o01 to o08), Voltstride (sneaker drop, o09 to o16), Inkwell & Lantern (bookstore events, o17 to o23), Pacewell (fitness app, o24 to o30). They use only `syn_` images and the two font families. Coverage: 15 source sizes from 320x50 to 1920x1080; 4 to 19 elements; text-only ads (o07, o14, o16, o30); BUSY backgrounds with every text on an opaque calm panel (o03 terrazzo, o09 confetti, o11 stripes, o19 terrazzo, o25 mosaic; their logos sit on a pill or the panel); rotation (o06, o10, o11, o15, o17, o20); grouped steps / features / price cards (o02, o18, o26); long run-on headline and long legal (o10, o17, o26). Text box heights come from TextMeasurer (`Col` stacks a column), so copy always fits. The same checks run on them; the first draft failed 10 (boxes too short, a sticker over the product, a quote mark over its quote) and was fixed by hand, none was dropped.

**Review:** Claude looked at every render (`review.nosync/sheet_augmented.png`, `review.nosync/sheet_offbrand.png`, plus close-up pages) and fixed what looked wrong; see "Review fixes" in the task report. Things fixed after looking: jitter broke badge rows and lineups (now rows move as one unit and columns keep their alignment); tracking was applied to a 150 px "30% OFF" (now only to small caps up to 40 px); badges were jammed into the corners of 90 px strips (now corner badges only on canvases with a side of at least 400 px); the brand rename turned "next solstice" into "next citrine" (now only brand mentions change); logos sat directly on busy confetti/terrazzo (now on a pill or the calm panel).

**Known limitations (this pass)**
- The synthetic images are flat vector-style drawings, not photographs; real campaign photos are richer. They are training stand-ins only.
- All fonts are still Solstice Serif / Solstice Sans: the renderer has no others, so type variety is only weight, style, size, tracking and family swap.
- Augmented variants share their parent's composition; 90 variants of 30 designs are not 90 independent designs. Treat per-design (not per-variant) counts as the real sample size, and keep the parent-split rule.
- The palette swap changes Solstice copy to the palette's name but keeps the drinks-themed copy, so "Harbor botanical sparkling" ads exist. Product swaps also put e.g. a sneaker box in a drinks ad. Layout learning doesn't care; a semantic judge might.
- The contrast floor is 3:1 (WCAG large text), not 4.5:1, matching the original drafted ads' white-on-orange buttons; a few original ads sit slightly below it and their variants are held to the parent's value.
- Canvas changes are limited to nearby sizes; there are no story-to-banner re-layouts.
- `review/` is gitignored; rerun the commands to see the sheets.

### 2026-10-20: first corpus (33 ads)

**Command** (from `distil-research/chameleon/`): `python -m dev.distill.make_corpus`. This rewrites every `designs/*.json`, prints layout warnings (text that overflows its box, content off the canvas, text boxes that overlap by accident; 0 at the time of writing), and saves a contact sheet plus one PNG per ad to `review/` (gitignored). Nothing in it is random: each drafted ad is written out in `make_corpus.py` as explicit data (copy, colours, coordinates).

**Counts:** 33 ads = 23 train + 10 holdout.

| origin | train | holdout |
|---|---|---|
| `sample` (the 3 official samples, copied verbatim from `chameleon/input/`) | 3 | 0 |
| `claude-drafted` (written by Claude, reviewed against renders) | 20 | 10 |

Each file also has `notes` (what the ad tests) and `features` (tags used in the table below).

**Variety: which ads cover which feature**

| feature | train | holdout |
|---|---|---|
| square 1080×1080 | warmup_01, core_01, c01, c03, c04, c05, c25, c28 | c02, c27 |
| portrait 1080×1350 | advanced_01, c06, c08, c29 | c07, c26 |
| story 1080×1920 | c09, c10, c30 | c11 |
| landscape 1200×628 / 1280×720 | c12, c14 | c13, c15 |
| wide banner 1584×396 | c16 | c17 |
| small 300×250 | c18 | c19 |
| strip 728×90 | c20 | c21 |
| other (160×600, 970×250, 820×312) | c22, c23, c24 | |
| 3 to 5 elements | warmup_01, c03, c30 | c15 |
| 16 to 20 elements | advanced_01, c04, c06, c29 | c26 (20) |
| no groups | warmup_01, c03, c24, c30 | c27 |
| button grouped / ungrouped | grouped in most; ungrouped: c10 | ungrouped: c02 |
| badges (count) | core_01 (1), advanced_01 (3), c04 (5), c09 (2), c14 (1), c23 (2) | c13 (2), c17 (3), c21 (1), c26 (3) |
| rotated element | advanced_01, c05, c14 | c11, c26 |
| long run-on headline | advanced_01 | c07 |
| legal / disclaimer line (**bold** = long, 2+ lines) | **advanced_01**, c04, **c08**, **c10**, c23, c25, c29 | **c07**, c11, c19, **c26** |
| no product image | warmup_01, c03, c10 | c07, c15, c17 |
| no logo image | c03, c10, c23 | c02, c13, c15, c17, c19, c21 |
| text only (no image at all) | c03, c10 | c15, c17 |
| overlapping semi-transparent shapes | core_01, c01, c05, c08, c12, c14, c24, c29 | c02, c13, c17, c19, c26 |
| light background | c10, c23, c29 | c02, c07, c13, c17, c19 |
| mid-tone (orange) background | c05 | c15, c21 |
| solid shape background (not the gradient image) | c03, c05, c08, c10, c12, c20, c23, c29, c30 | c02, c07, c13, c15, c17, c19, c21, c27 |
| missing asset (`expects_missing_asset`) | advanced_01, c25 | c26 |
| style runs (two-colour headline) | core_01, c01, c03, c28 | |
| italic text | | c26, c27 |
| line shape | | c07 |

**Conventions worth knowing**
- The logo (`solstice_logo_lockup.png`) is a white wordmark. On light backgrounds the ads either sit it on a dark band or pill (c07, c29) or set "SOLSTICE" as letter-spaced text instead (c02, c10, c13, c17). Those text wordmarks are ordinary text elements, not logos.
- Image boxes keep the asset's aspect ratio: the logo is 3:1, the bottle is 1:1, and the gradient is always `fit: cover`.
- The orange bottle disappears on an orange background, so orange ads put a cream disc behind it (c05, c14, c21).
- Button labels sit at `y = button_centre - 0.68 * font_size`, which centres capitals optically in this renderer.
- Shapes named `deco*` and backgrounds may bleed off the canvas, as Level 2's circles do; nothing else does.

**Known limitations**
- Only 3 images exist, so every product ad shows the same bottle and the same gradient. Variety comes from layout, copy, colour and size, not imagery.
- The ads are synthetic: written by Claude in one session for a fictional brand, not taken from real campaigns. Their layouts lean toward clean, centred or two-column templates and may be tidier than real ads.
- "Reviewed" means Claude checked the renders (the contact sheet plus close-ups) and fixed what looked wrong. Angelo has not yet reviewed every render. Re-run the command and check `review.nosync/sheet.png` before relying on the corpus.
- The renderer ignores `stroke_color` on rectangles and ellipses, so no ad uses outlined shapes.
