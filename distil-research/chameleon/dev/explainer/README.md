# Cut to Fit explainer (artifact source)

Builds the "Cut to Fit" page: 7 ads (3 official samples + 4 held-back/off-brand corpus ads, no Crello content) x 8 sizes x 6 pipeline stages, each the research engine's real output plus its official-renderer image.

From `distil-research/chameleon/` with `PYTHONPATH=$PWD`, writing into a work folder `$O` that holds `examples.json` and `sizes.json`:
1. `python dev/explainer/gen_stage.py original $O`, then one run per stage with its env: `base` (none), `bisect` (`CHAMELEON_STRATEGIES=bisection`), `v2` (+`CHAMELEON_BISECT_V2=1`), `v3` (+`CHAMELEON_HIER_LOCK=1 CHAMELEON_NEAR_FIT=1`), `v3L` (+`CHAMELEON_LAYERS=1`).
2. `python dev/explainer/bundle.py $O <folder with brand_kit + synthetic PNGs> ../../chameleon/brand_kit/fonts`
3. Replace `__FONTS_CSS__` (one @font-face per bundled font) and `__DATA__` (data.json) in `template.html`.
