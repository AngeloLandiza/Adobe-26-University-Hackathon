# corpus/assets: synthetic stand-in images (research only)

**What it is:** 25 PNG images drawn by code (`chameleon/dev/distill/synth_images.py`) for the research corpus.
They exist so the corpus isn't limited to the brand kit's three images (one bottle, one gradient, one logo):
backgrounds of other colours and tones, *busy* textures (the rubric says text must never sit on a busy
image, so the corpus needs examples), product cut-outs with transparency in several proportions, and logo
marks in several aspect ratios.

**How they were made:** pure PIL/numpy drawing with a fixed seed (`SEED = 2026`): linear and radial
gradients, Gaussian-blurred ellipses, random chips and stripes, simple polygons for the products, and
wordmarks set in the brand kit's own Solstice fonts. No generative AI model and no downloads.
`python -m dev.distill.synth_images` (from `distil-research/chameleon/`) redraws them all identically.

**Where they are used:** only by corpus ads (`origin` `augmented` or `offbrand`) via `asset_id: "syn_..."`.
Every file is prefixed `syn_` so it can never collide with a brand-kit asset name;
`dev/distill/assets.py` merges this folder with `chameleon/brand_kit/assets/` for each run.

**Not shipped:** these are research training data. The engine never draws, edits or outputs them (the
challenge disqualifies regenerating image content; these are *inputs* standing in for a real brand's
photos, and they never leave `distil-research/`). The brand names on them (Quillbrook, Voltstride,
Inkwell & Lantern, Pacewell) are invented.

| file | size | what it is |
|---|---|---|
| `syn_bg_harbor_night.png` | 1200x1500 RGB | dark: navy-to-teal vertical gradient with soft out-of-focus light blobs |
| `syn_bg_forest_mist.png` | 1500x1200 RGB | dark: deep-green radial gradient with pale soft blobs |
| `syn_bg_roast_ember.png` | 1200x1200 RGB | dark: espresso-brown diagonal gradient with a warm ember glow |
| `syn_bg_violet_glow.png` | 1200x1600 RGB | dark: deep-violet radial glow with magenta and blue blobs |
| `syn_bg_peach_haze.png` | 1500x1000 RGB | light: peach-to-pink haze with soft white and apricot blobs |
| `syn_bg_glacier.png` | 1200x1500 RGB | light: icy white-to-pale-blue vertical gradient with soft blobs |
| `syn_bg_sunlit_cream.png` | 1600x900 RGB | light: warm cream-to-butter horizontal gradient (wide 16:9) |
| `syn_bg_fog.png` | 1200x1200 RGB | light: cool grey fog gradient with soft blobs |
| `syn_busy_confetti.png` | 1200x1200 RGB | BUSY: dense multicolour confetti chips on dark violet |
| `syn_busy_stripes.png` | 1200x1200 RGB | BUSY: high-contrast diagonal stripes in five colours |
| `syn_busy_terrazzo.png` | 1500x1000 RGB | BUSY: terrazzo stone chips on a pale ground (3:2) |
| `syn_busy_mosaic.png` | 1000x1000 RGB | BUSY: random colour mosaic tiles with pixel noise |
| `syn_prod_tall_bottle.png` | 400x1000 RGBA | product cut-out, tall 2:5: dark cold-brew bottle with a cream label |
| `syn_prod_wide_box.png` | 1000x600 RGBA | product cut-out, wide 5:3: orange sneaker box in three-quarter view |
| `syn_prod_round_tin.png` | 800x800 RGBA | product cut-out, round 1:1: coffee tin seen from above |
| `syn_prod_square_book.png` | 800x800 RGBA | product cut-out, square 1:1: art-book cover with abstract shapes |
| `syn_prod_sneaker.png` | 1200x600 RGBA | product cut-out, wide 2:1: violet-and-lime sneaker in side view |
| `syn_prod_phone.png` | 500x1000 RGBA | product cut-out, tall 1:2: phone showing a fitness-app ring and bars |
| `syn_prod_coffee_bag.png` | 700x900 RGBA | product cut-out, portrait 7:9: stand-up coffee pouch |
| `syn_logo_blocks_light.png` | 1000x250 RGBA | logo, wide 4:1, light (for dark backgrounds): ring mark plus abstract wordmark-like blocks, no text |
| `syn_logo_blocks_dark.png` | 1000x250 RGBA | logo, wide 4:1, dark (for light backgrounds): ring mark plus abstract wordmark-like blocks, no text |
| `syn_logo_quillbrook_badge.png` | 600x600 RGBA | logo, round 1:1: coffee-roaster badge for the invented brand Quillbrook |
| `syn_logo_voltstride_light.png` | 1000x200 RGBA | logo, wide 5:1, light: bolt mark + italic wordmark for the invented sneaker brand Voltstride |
| `syn_logo_inkwell_stacked.png` | 900x600 RGBA | logo, stacked 3:2, dark ink: lantern mark over a two-line serif wordmark for the invented bookstore Inkwell & Lantern |
| `syn_logo_pacewell.png` | 900x400 RGBA | logo, wide 9:4, dark: app-icon tile with a pulse line + wordmark for the invented fitness app Pacewell |
