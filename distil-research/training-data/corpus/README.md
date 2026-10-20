# corpus: source ads

**What it is:** the ads the whole loop learns from. Each file is one source design in the challenge's schema (`chameleon_design_schema_v0.9.json`): a canvas plus its elements (text, images, shapes) at the original size. The engine resizes each one to many target sizes, and the teacher judges the results.

**Why it exists:** the challenge ships only 3 sample ads. Learning from 3 ads would just memorise them, and the hidden test set has many designs the engine has never seen. More, varied ads make the data, and anything learned from it, generalise.

**File format** (`designs/<task_id>.json`):

```json
{
  "task_id": "c07_story_sale",
  "split": "train",
  "origin": "sample",
  "expects_missing_asset": false,
  "notes": "what this ad tests",
  "source": { "canvas": { "width": 1080, "height": 1920 }, "elements": [ ... ] }
}
```

- `split`: `train` (used for learning) or `holdout` (only for checking results, never trained on).
- `origin`: `sample` (one of the 3 official samples), `claude-drafted` (written by Claude, reviewed by a person) or `imported`.

**How it was made:** see "Generation log" below. Every ad uses only the brand kit's three images (`solstice_bg_gradient.png`, `solstice_bottle.png`, `solstice_logo_lockup.png`) and its two font families, Solstice Serif and Solstice Sans. Variety comes from the source canvas size, element count, groups, badges, rotation, long text, legal lines, missing assets and colours.

**Feeds:** `candidate-pool/`, and through it everything else.

**Checked by:** `chameleon/tests/test_corpus.py`. It checks there are ≥ 30 ads and ≥ 8 holdout, unique ids, that every asset and font exists, and that every ad plans and renders.

## Generation log

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
- "Reviewed" means Claude checked the renders (the contact sheet plus close-ups) and fixed what looked wrong. Angelo has not yet reviewed every render. Re-run the command and check `review/sheet.png` before relying on the corpus.
- The renderer ignores `stroke_color` on rectangles and ellipses, so no ad uses outlined shapes.
