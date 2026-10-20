# corpus/challenge: the generalisation test set (EVALUATION ONLY)

**What it is:** 40 source ads that nothing in the research loop has been trained, tuned or selected on. Same file format as `corpus/designs/` (see `../README.md`), with:

- `split: "challenge"`, `origin: "challenge"`, `brand: "challenge"` (the brand *family* used in A/B reports),
- `challenge_brand`: which invented brand (`lumen`, `obsidian`, `fizzbang`, `monograph`, `meridian`),
- `palette`: `light`, `dark`, `saturated` or `monochrome`.

**Why it exists:** every approach so far (grader weights, relation features, B2 floor plans, retrieval, FLEX) was built or chosen with ads from the same generator, the same 5 brands (Solstice + 4 off-brand) and mostly the same source sizes. Holdout ads are unseen *ads*, but not unseen *kinds* of ads. The hidden test set will contain designs unlike ours. This set asks: does a change still help on ads that look nothing like what it was tuned on? Combined with the worst-slice "generalisation score" in the A/B report, it lets the tournament of approaches prefer the one that is good everywhere, not just good on average.

**THE RULE: never used for tuning or selection.** Do not train on these ads, do not pick thresholds, templates, weights, floor plans or retrieval neighbours with them, do not look at per-ad results to fix a specific ad. They are only for the final A/B comparison between candidate approaches. Once an approach has been changed because of a challenge result, the challenge set no longer measures generalisation for that approach; if that happens, write a new challenge set (new brands, new structures) rather than reuse this one.

`load_corpus()` never returns these ads (they live in their own folder). Ask for them explicitly: `load_corpus("challenge")`, `load_corpus(include_challenge=True)` or `load_challenge()` (`chameleon/dev/distill/corpus.py`).

## How it differs from the corpus

| | corpus (`designs/`) | challenge |
|---|---|---|
| brands | Solstice + Quillbrook, Voltstride, Inkwell & Lantern, Pacewell | Lumen & Fig (skincare, light), Obsidian Audio (headphones, dark), Fizzbang (soda + arcade, saturated), Monograph (gallery + watches, monochrome), Meridian Air (travel, saturated cobalt/coral); 8 ads each |
| images | brand kit + 25 `syn_*` (blurred-blob gradients, busy chips, flat products) | 23 new `chl_*` only: layered photo-like scenes with film grain (dunes, night skyline, concrete, leaves, a 4:1 coastal panorama), halftone and Bauhaus patterns, 9 new products (headphones, earbuds case, serum, cream jar, watch, soda can, controller, suitcase, sunglasses), 7 new logos (one with its own lozenge, square monograms, white/colour pairs) |
| source sizes | mostly the standard targets | 27 sizes, 22 of them never used in the corpus: 600x900, 1456x180, 2400x600, 300x1050, 2000x1000, 1350x1080, 500x1000, 900x1600, 1080x566, 1200x400, 468x60, 240x400, 250x250, 400x400, 600x600, 640x640, 800x800, 1000x1000, 1000x1500, 1200x1600, 800x1200, 1600x900 (plus 970x250, 160x600, 1200x300, 1500x500, 1200x1500, which a few augmented/off-brand ads use) |
| element counts | 3 to 20 | 1 to 26 |

Structures (feature tags in each file):

| structure (feature) | ads |
|---|---|
| two products (`two-products`) | chl02, chl06, chl12, chl20, chl35 |
| quote-led (`quote-led`) | chl05, chl14, chl27 |
| price-led (`price-led`) | chl06, chl18, chl28, chl33, chl36, chl37, chl40 |
| event with a date/time block (`event-date`) | chl16, chl19, chl26, chl32 |
| list-led (`list-led`) | chl03, chl21, chl31, chl32, chl37, chl40 |
| logo-only centrepiece (`logo-centrepiece`) | chl08, chl24 |
| one element (`one-element`) | chl25 (a single headline on the bare canvas) |
| 20+ elements (`20-plus-elements`) | chl13 (22), chl32 (26), chl40 (25) |
| very long legal (`long-legal`) | chl07, chl30, chl38 |
| rotated stickers (`rotated-sticker`) | chl04, chl17 (two), chl39 |
| full-bleed image with a text panel (`full-bleed-panel`) | chl01, chl10, chl23, chl29, chl34 (chl33, chl39 put a panel on a photo too) |

## How it was made

From `distil-research/chameleon/` (both deterministic; a rerun rewrites only changed files):

```
python -m dev.distill.challenge_images   # 23 chl_*.png into corpus/assets/ (seed 4242, PIL/numpy only)
python -m dev.distill.make_challenge     # 40 ads into corpus/challenge/, review.nosync/sheet_challenge.png
```

Every ad is explicit data in `make_challenge.py` (copy, colours, positions), built with `make_offbrand`'s helpers, so text boxes are sized by the engine's TextMeasurer. Each must pass the corpus quality gate (`make_corpus.check(strict=True)` + `pixel_check`: text fits, nothing but background/decoration off canvas, no accidental overlaps, images in their own aspect, text contrast >= 3:1, no text on a busy area); a failing ad is dropped and reported (none are dropped now). Claude looked at the full sheet and close-ups of every ad and fixed what looked unfinished (e.g. contrast of step numbers in chl03, a sticker touching the headline in chl17, empty bands above legal blocks in chl07/chl30/chl38, a logo jammed into a corner in chl05, an orphan word in chl23).

Fonts are still only Solstice Serif / Solstice Sans (the renderer has no others); the copy, brands, quotes and people are invented.

## How to use it (A/B)

```
python -m dev.distill.ab_engine run --split challenge --name base_chl --env CHAMELEON_FLEX=0
python -m dev.distill.ab_engine run --split challenge --name flex_chl --env CHAMELEON_FLEX=1
python -m dev.distill.ab_engine judge --a base_chl --b flex_chl
python -m dev.distill.ab_engine report --a base_chl --b flex_chl
```

Cells = each challenge ad x every target in `dev/targets.json` except those within 2% of the ad's own aspect (about 12 per ad). The source renders the judge shows as "original" are written to `training-data/ab/_sources.nosync/<task>/source.png`. Use `_chl` variant names so challenge winners never mix with holdout winners.

**Checked by:** `chameleon/tests/test_challenge.py`.

**Known limitations**
- Synthetic: written by Claude in one session; tidier than real campaigns, and the photo-like images are drawn, not photographed.
- 40 ads, 8 per brand: a slice of one brand or one aspect class is small, which is why the generalisation score only counts slices with at least 5 decided cells.
- Same author as the corpus, so some habits (centred columns, pill buttons) carry over. A truly independent set would come from another designer.
