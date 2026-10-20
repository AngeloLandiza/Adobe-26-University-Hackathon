"""Write the ad corpus: training-data/corpus/designs/*.json plus a review
contact sheet (training-data/corpus/review/sheet.png, gitignored).

    cd distil-research/chameleon && python -m dev.distill.make_corpus

The 3 official samples are copied from input/*.json. Every other ad is
hand-authored below as explicit data (copy, colours, coordinates), the way a
designer would build it, for the fictional brand Solstice (sparkling
botanical drinks). Only the brand kit's three images and two font families
are used; two ads reference a deliberately missing image to test the
missing-asset path. Nothing here is random.

Running it also prints layout warnings (text that overflows its box, content
off the canvas, text boxes that overlap by accident), so a broken ad is
caught before anyone looks at the sheet.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from dev.distill.paths import CORPUS_DIR, DATA_ROOT

CHAMELEON = Path(__file__).resolve().parents[2]
ASSETS_DIR = CHAMELEON / "brand_kit" / "assets"
FONTS_DIR = CHAMELEON / "brand_kit" / "fonts"
INPUT_DIR = CHAMELEON / "input"
REVIEW_DIR = DATA_ROOT / "corpus" / "review"

GRADIENT, BOTTLE, LOGO = "solstice_bg_gradient.png", "solstice_bottle.png", "solstice_logo_lockup.png"
LOGO_ASPECT = 3.0  # solstice_logo_lockup.png is 900x300; the bottle is 800x800
LOGO_INSET = 268 / 900  # transparent margin left of the visible wordmark

# Solstice palette (the samples' colours plus the bottle's sage and a cream).
PLUM = "rgba(43,27,46,1)"
ORANGE = "rgba(232,106,58,1)"
GOLD = "rgba(240,178,66,1)"
RED = "rgba(220,60,50,1)"
SAGE = "rgba(122,154,110,1)"
CREAM = "rgba(250,242,230,1)"
PEACH = "rgba(253,232,214,1)"
MIST = "rgba(232,238,224,1)"
WHITE = "rgba(255,255,255,1)"


def rgba(color: str, alpha: float) -> str:
    """`color` with its alpha replaced, e.g. rgba(PLUM, 0.8)."""
    return color.rsplit(",", 1)[0] + f",{alpha})"


# ---------------------------------------------------------------------------
# Element builders (field order and conventions copied from the samples)
# ---------------------------------------------------------------------------

def _extras(el: dict, group, rotation, opacity) -> dict:
    if rotation:
        el["rotation"] = rotation
    if opacity is not None:
        el["opacity"] = opacity
    if group:
        el["group_id"] = group
    return el


def text(id, content, x, y, w, h, size, *, serif=False, bold=False, italic=False, color=WHITE,
         align="left", z=4, line_height=None, letter_spacing=None, runs=None, group=None, rotation=0,
         opacity=None) -> dict:
    el = {"id": id, "type": "text", "x": x, "y": y, "width": w, "height": h, "z_index": z,
          "content": content, "font_family": "Solstice Serif" if serif else "Solstice Sans",
          "font_size": size}
    if bold:
        el["font_weight"] = "bold"
    if italic:
        el["font_style"] = "italic"
    el["color"] = color
    el["text_align"] = align
    if line_height is not None:
        el["line_height"] = line_height
    if letter_spacing is not None:
        el["letter_spacing"] = letter_spacing
    if runs:
        el["style_runs"] = runs
    return _extras(el, group, rotation, opacity)


def run(content: str, part: str, color: str) -> dict:
    """A style run recolouring the first occurrence of `part` in `content`."""
    start = content.index(part)
    return {"start": start, "end": start + len(part), "color": color}


def shape(id, x, y, w, h, fill, *, kind="rectangle", radius=None, z=1, group=None, rotation=0,
          opacity=None, stroke=None, stroke_width=None) -> dict:
    el = {"id": id, "type": "shape", "x": x, "y": y, "width": w, "height": h, "z_index": z,
          "shape_type": kind, "fill_color": fill}
    if radius:
        el["corner_radius"] = radius
    if stroke:
        el["stroke_color"] = stroke
        el["stroke_width"] = stroke_width
    return _extras(el, group, rotation, opacity)


def image(id, asset, x, y, w, h, *, fit="contain", z=3, group=None, rotation=0, opacity=None) -> dict:
    el = {"id": id, "type": "image", "x": x, "y": y, "width": w, "height": h, "z_index": z,
          "asset_id": asset, "fit": fit}
    return _extras(el, group, rotation, opacity)


def bg_image(w, h) -> dict:
    return image("bg", GRADIENT, 0, 0, w, h, fit="cover", z=0)


def bg_fill(w, h, color) -> dict:
    return shape("bg", 0, 0, w, h, color, z=0)


def logo(x, y, w, *, z=4, group=None) -> dict:
    return image("logo", LOGO, x, y, w, round(w / LOGO_ASPECT), z=z, group=group)


def logo_left(visible_x, y, w, **kw) -> dict:
    """Logo whose visible wordmark (not its transparent box) starts at visible_x."""
    return logo(round(visible_x - LOGO_INSET * w), y, w, **kw)


def bottle(x, y, size, *, id="product", z=3, rotation=0) -> dict:
    return image(id, BOTTLE, x, y, size, size, z=z, rotation=rotation)


def pill(prefix, x, y, w, h, label, size, *, fill=ORANGE, color=WHITE, grouped=True, z=3,
         radius=None, serif=False, letter_spacing=None, rotation=0) -> list[dict]:
    """A button or badge: a rounded shape plus its centred label. The label box
    is placed so the capitals sit in the optical middle of the shape."""
    group = prefix if grouped else None
    label_y = round(y + h / 2 - 0.68 * size)
    return [
        shape(f"{prefix}_bg", x, y, w, h, fill, radius=h / 2 if radius is None else radius, z=z,
              group=group, rotation=rotation),
        text(f"{prefix}_label", label, x, label_y, w, round(size * 1.4), size, serif=serif, bold=True,
             color=color, align="center", z=z + 1, group=group, rotation=rotation,
             letter_spacing=letter_spacing),
    ]


def ellipse(id, x, y, w, h, fill, opacity, *, z=1) -> dict:
    return shape(id, x, y, w, h, fill, kind="ellipse", z=z, opacity=opacity)


def ad(task_id, split, w, h, elements, notes, features) -> dict:
    flat = []
    for el in elements:
        flat.extend(el if isinstance(el, list) else [el])
    return {"task_id": task_id, "split": split, "origin": "claude-drafted",
            "expects_missing_asset": False, "notes": notes, "features": features,
            "source": {"canvas": {"width": w, "height": h}, "elements": flat}}


# ---------------------------------------------------------------------------
# The drafted ads
# ---------------------------------------------------------------------------

LEGAL_CO = "Solstice Botanical Sparkling Co., 2026."


def drafted() -> list[dict]:
    ads = []
    A = ads.append

    # c01 -- square, dark, left-aligned copy with the bottle on the right
    hl = "Golden hour, bottled."
    A(ad("c01_square_golden_hour", "train", 1080, 1080, [
        bg_image(1080, 1080),
        ellipse("deco_glow", 550, 260, 460, 460, ORANGE, 0.22),
        ellipse("deco_ring", 760, 800, 260, 260, WHITE, 0.08),
        bottle(530, 240, 500),
        logo_left(90, 80, 240),
        text("headline", hl, 90, 350, 435, 180, 66, serif=True, bold=True, runs=[run(hl, "bottled.", GOLD)]),
        text("subhead", "Blood orange, wild honey and a sprig of rosemary, lifted with fine bubbles.",
             90, 570, 420, 130, 26, color=rgba(WHITE, 0.85)),
        pill("cta", 90, 740, 330, 86, "Taste the Glow", 24),
        text("note", "Now in 4-packs and singles", 90, 850, 420, 34, 20, color=rgba(WHITE, 0.7)),
    ], "Square, dark gradient, left-aligned copy, two-colour headline, bottle on a soft glow.",
        ["square", "dark", "gradient-bg", "groups", "style-runs", "deco-overlap"]))

    # c02 -- square, light cream, text wordmark instead of the (white) logo, ungrouped button
    A(ad("c02_square_garden_fizz", "holdout", 1080, 1080, [
        bg_fill(1080, 1080, CREAM),
        ellipse("deco_leaf", 620, -120, 600, 600, SAGE, 0.25),
        ellipse("deco_sun", -140, 760, 460, 460, GOLD, 0.3),
        text("wordmark", "SOLSTICE", 340, 70, 400, 44, 30, bold=True, color=PLUM, align="center",
             letter_spacing=8),
        bottle(290, 140, 500),
        text("headline", "Garden-fresh fizz", 90, 650, 900, 90, 68, serif=True, bold=True, color=PLUM,
             align="center"),
        text("subhead", "Cucumber, mint and elderflower, sparkling and lightly sweet.",
             140, 760, 800, 80, 26, color=rgba(PLUM, 0.8), align="center"),
        pill("cta", 370, 880, 340, 84, "Find a Store", 24, fill=PLUM, color=CREAM, grouped=False),
    ], "Light solid background, no logo image (text wordmark), button shape + label NOT grouped.",
        ["square", "light", "solid-bg", "no-logo", "ungrouped-button", "deco-overlap", "bleed"]))

    # c03 -- square, text only, 3 elements
    hl = "Sparkle slowly. Sip at sunset."
    A(ad("c03_square_text_only", "train", 1080, 1080, [
        bg_fill(1080, 1080, PLUM),
        text("headline", hl, 90, 360, 900, 250, 96, serif=True, bold=True, color=CREAM, align="center",
             runs=[run(hl, "Sip at sunset.", GOLD)]),
        text("tagline", "SOLSTICE BOTANICAL SPARKLING", 140, 660, 800, 36, 22, bold=True, color=ORANGE,
             align="center", letter_spacing=4),
    ], "Minimal text-only ad: 3 elements, no image, no logo, no groups.",
        ["square", "dark", "solid-bg", "text-only", "no-image", "no-logo", "3-elements"]))

    # c04 -- square, five flavour badges
    badges = [("WILD HONEY", 190), ("BLOOD ORANGE", 215), ("ROSEMARY", 170),
              ("ELDERFLOWER", 205), ("PINK PEPPERCORN", 240)]
    row1, row2 = badges[:3], badges[3:]
    els = [bg_image(1080, 1080), logo(400, 50, 280), bottle(330, 160, 420),
           text("headline", "Five botanicals. One glow.", 90, 600, 900, 84, 60, serif=True, bold=True,
                align="center")]
    for row, y in ((row1, 712), (row2, 774)):
        x = (1080 - (sum(w for _, w in row) + 20 * (len(row) - 1))) / 2
        for label, w in row:
            n = badges.index((label, w)) + 1
            els.append(pill(f"badge{n}", round(x), y, w, 48, label, 16, fill=rgba(WHITE, 0.16)))
            x += w + 20
    els += [pill("cta", 340, 862, 400, 84, "Build Your Mix", 24),
            text("legal", f"Flavours vary by region. {LEGAL_CO}", 90, 1000, 900, 26, 14,
                 color=rgba(WHITE, 0.55), align="center")]
    A(ad("c04_square_five_badges", "train", 1080, 1080, els,
         "Five grouped translucent badges in two rows, plus button and legal line (17 elements).",
         ["square", "dark", "gradient-bg", "groups", "badges-5", "legal"]))

    # c05 -- square, bright orange, rotated sticker overlapping the product
    A(ad("c05_square_orange_sticker", "train", 1080, 1080, [
        bg_fill(1080, 1080, ORANGE),
        ellipse("deco_halo", 560, -140, 640, 640, WHITE, 0.12),
        logo(380, 50, 320),
        ellipse("deco_disc", 310, 190, 460, 460, CREAM, 0.95, z=2),
        bottle(300, 180, 480),
        # A round sticker whose label wraps to two lines inside the circle.
        shape("sticker_bg", 730, 200, 210, 210, GOLD, kind="ellipse", z=5, group="sticker", rotation=14),
        text("sticker_label", "NEW FLAVOR", 760, 262, 150, 80, 26, bold=True, color=PLUM, align="center",
             z=6, line_height=1.15, group="sticker", rotation=14),
        text("headline", "Blood Orange Sunset", 90, 690, 900, 90, 68, serif=True, bold=True,
             align="center"),
        text("subhead", "Bright citrus, a little bitter, entirely golden.", 140, 795, 800, 44, 28,
             color=rgba(WHITE, 0.92), align="center"),
        pill("cta", 360, 880, 360, 84, "Try It Today", 24, fill=PLUM),
    ], "Mid-tone solid orange background, rotated circular sticker (grouped) overlapping the product box.",
        ["square", "mid-tone", "solid-bg", "rotated", "groups", "deco-overlap", "bleed"]))

    # c06 -- portrait, three-bottle flavour lineup
    els = [bg_image(1080, 1350), logo(380, 70, 320),
           text("headline", "Three ways to chase the light", 90, 240, 900, 84, 60, serif=True, bold=True,
                align="center")]
    lineup = [("Blood Orange", "Bright & zesty"), ("Cucumber Mint", "Cool & crisp"),
              ("Honey Rosemary", "Warm & herbal")]
    for i, (name, desc) in enumerate(lineup):
        x = 30 + 340 * i
        els += [bottle(x, 400, 340, id=f"product_{i + 1}"),
                text(f"flavor_{i + 1}", name, x, 760, 340, 44, 32, serif=True, bold=True, align="center"),
                text(f"flavor_{i + 1}_desc", desc, x, 808, 340, 30, 20, color=rgba(WHITE, 0.75),
                     align="center")]
    els += [text("subhead", "Mix a pack of all three, delivered cold to your door.", 140, 890, 800, 80, 28,
                 color=rgba(WHITE, 0.85), align="center"),
            pill("cta", 330, 1010, 420, 90, "Build Your Pack", 26),
            text("delivery", "Free chilled delivery on every mixed pack", 190, 1140, 700, 34, 22, bold=True,
                 color=GOLD, align="center")]
    A(ad("c06_portrait_lineup", "train", 1080, 1350, els,
         "Portrait flavour lineup: the bottle three times in a row, each with a name and descriptor (16 elements).",
         ["portrait", "dark", "gradient-bg", "groups", "repeated-image", "many-texts"]))

    # c07 -- portrait, light, run-on headline + long legal, logo on a dark band
    A(ad("c07_portrait_long_copy", "holdout", 1080, 1350, [
        bg_fill(1080, 1350, CREAM),
        shape("band", 0, 0, 1080, 180, PLUM, z=1),
        logo(390, 40, 300, z=2),
        text("kicker", "LIMITED SUMMER RELEASE", 190, 250, 700, 36, 24, bold=True, color=ORANGE,
             align="center", letter_spacing=4),
        text("headline", "Our Most Ambitious Small-Batch Botanical Sparkling Blend Yet, Slow-Steeped With "
             "Wild Honey, Blood Orange Zest and Smoked Rosemary for Long Golden Evenings",
             90, 310, 900, 390, 52, serif=True, bold=True, color=PLUM, align="center"),
        shape("divider", 440, 730, 200, 4, ORANGE, kind="line", stroke=ORANGE, stroke_width=4, z=1),
        text("body", "Each bottle is filled by hand in our seaside kitchen and released in a single run "
             "each summer. When it is gone, it is gone until next solstice.",
             140, 770, 800, 130, 26, color=rgba(PLUM, 0.8), align="center"),
        pill("cta", 330, 950, 420, 90, "Reserve a Case", 26),
        text("ships", "Ships late June", 340, 1060, 400, 34, 22, color=rgba(PLUM, 0.7), align="center"),
        text("legal", "Please enjoy responsibly. Contains no alcohol. Pre-orders ship within the mainland "
             "only, quantities are limited, and release dates may shift with the harvest. "
             f"{LEGAL_CO} See solstice.example/terms for full details.",
             90, 1230, 900, 60, 14, color=rgba(PLUM, 0.6), align="center"),
    ], "Light portrait with a run-on 5-line headline, a line divider, and a long multi-line legal block; "
       "logo sits on a dark band (white logo would vanish on cream).",
        ["portrait", "light", "solid-bg", "long-headline", "legal", "no-product", "line-shape", "groups"]))

    # c08 -- portrait, dark plum, overlapping translucent suns, big discount
    A(ad("c08_portrait_sunset_sale", "train", 1080, 1350, [
        bg_fill(1080, 1350, PLUM),
        ellipse("deco_sun_1", 200, 160, 620, 620, ORANGE, 0.35),
        ellipse("deco_sun_2", 420, 330, 520, 520, GOLD, 0.25),
        ellipse("deco_moon", -140, 1000, 420, 420, WHITE, 0.08),
        logo(390, 40, 300),
        text("kicker", "SUNSET SALE", 290, 160, 500, 44, 30, bold=True, color=GOLD, align="center",
             letter_spacing=8),
        bottle(340, 230, 420),
        text("discount", "30% OFF", 90, 670, 900, 200, 150, serif=True, bold=True, align="center"),
        text("subhead", "Every mixed case, this weekend only.", 140, 875, 800, 50, 32,
             color=rgba(WHITE, 0.88), align="center"),
        pill("cta", 330, 980, 420, 90, "Shop the Sale", 26, fill=GOLD, color=PLUM),
        text("legal", "Offer ends Sunday at midnight. Discount applies to mixed cases only and cannot be "
             f"combined with other offers. {LEGAL_CO}", 90, 1240, 900, 50, 14,
             color=rgba(WHITE, 0.55), align="center"),
    ], "Dark solid background with overlapping semi-transparent circles behind the product, huge discount text.",
        ["portrait", "dark", "solid-bg", "deco-overlap", "bleed", "groups", "legal"]))

    # c09 -- story, product launch with two badges
    A(ad("c09_story_launch", "train", 1080, 1920, [
        bg_image(1080, 1920),
        ellipse("deco_halo", 140, 400, 800, 800, WHITE, 0.08),
        logo(360, 110, 360),
        bottle(190, 380, 700),
        text("headline", "Meet Honey Rosemary", 90, 1110, 900, 220, 88, serif=True, bold=True,
             align="center"),
        text("subhead", "Warm, herbal and softly sparkling. Our newest botanical, made for long summer evenings.",
             120, 1340, 840, 130, 34, color=rgba(WHITE, 0.85), align="center"),
        pill("badge1", 318, 1500, 160, 56, "NEW", 20),
        pill("badge2", 502, 1500, 260, 56, "LIMITED RUN", 20, fill=GOLD, color=PLUM),
        pill("cta", 290, 1610, 500, 110, "Shop the Launch", 34),
        text("swipe", "Swipe up to shop", 340, 1790, 400, 36, 24, color=rgba(WHITE, 0.7), align="center"),
    ], "Story (9:16) launch ad: big product, two badges, large button, swipe line.",
        ["story", "dark", "gradient-bg", "badges-2", "groups"]))

    # c10 -- story, light, text-heavy ritual steps, ungrouped button, legal
    els = [bg_fill(1080, 1920, CREAM),
           ellipse("deco_sun", 560, -240, 760, 760, ORANGE, 0.15),
           text("wordmark", "SOLSTICE", 90, 130, 400, 50, 34, bold=True, color=PLUM, letter_spacing=10),
           text("headline", "The golden hour ritual", 90, 290, 760, 260, 104, serif=True, bold=True,
                color=PLUM),
           text("intro", "Three steps to the perfect pour, wherever you watch the sun go down.",
                90, 590, 860, 130, 34, color=rgba(PLUM, 0.8))]
    steps = ["Chill the bottle until it is properly ice cold.",
             "Pour slowly over a tall glass of clear ice.",
             "Finish with a sprig of rosemary and a twist of peel."]
    for i, step in enumerate(steps):
        y = 800 + 190 * i
        els += [text(f"step{i + 1}_num", f"0{i + 1}", 90, y, 140, 96, 72, serif=True, bold=True,
                     color=ORANGE, group=f"step{i + 1}"),
                text(f"step{i + 1}_text", step, 250, y + 26, 740, 90, 34, color=PLUM, group=f"step{i + 1}")]
    els += [pill("cta", 90, 1420, 480, 104, "Shop Ritual Kits", 32, fill=PLUM, color=CREAM, grouped=False),
            text("legal", "Please enjoy responsibly. Ritual kits include one bottle each of three flavours "
                 f"and a reusable glass. Available while stocks last. {LEGAL_CO}",
                 90, 1760, 900, 80, 18, color=rgba(PLUM, 0.6))]
    A(ad("c10_story_ritual_steps", "train", 1080, 1920, els,
         "Light text-heavy story with numbered steps (each number+text grouped), no product image, "
         "ungrouped button, legal.",
         ["story", "light", "solid-bg", "no-image", "no-logo", "text-groups", "ungrouped-button", "legal"]))

    # c11 -- story, rotated ribbon
    A(ad("c11_story_longest_day", "holdout", 1080, 1920, [
        bg_image(1080, 1920),
        ellipse("deco_sun", -220, 1300, 720, 720, ORANGE, 0.2),
        ellipse("deco_moon", 700, 220, 500, 500, WHITE, 0.08),
        logo(380, 100, 320),
        pill("ribbon", 140, 300, 800, 100, "LONGEST DAY SALE", 44, fill=RED, radius=0, letter_spacing=4,
             rotation=-6),
        bottle(240, 450, 600),
        text("headline", "Make the longest day last", 190, 1080, 700, 220, 80, serif=True, bold=True,
             align="center"),
        text("subhead", "Every bottle 25% off from sunrise to sunset on June 21.", 100, 1320, 880, 50, 32,
             color=rgba(WHITE, 0.85), align="center"),
        pill("cta", 290, 1460, 500, 110, "Claim the Deal", 34),
        text("legal", f"One day only. Discount applied at checkout. {LEGAL_CO}", 90, 1800, 900, 30, 18,
             color=rgba(WHITE, 0.55), align="center"),
    ], "Story with a rotated full-width ribbon (shape + label, grouped), product, legal.",
        ["story", "dark", "gradient-bg", "rotated", "groups", "legal", "bleed"]))

    # c12 -- landscape 1200x628, split panel
    A(ad("c12_landscape_split", "train", 1200, 628, [
        bg_fill(1200, 628, PLUM),
        image("photo", GRADIENT, 600, 0, 600, 628, fit="cover", z=1),
        ellipse("deco_glow", 690, 84, 460, 460, ORANGE, 0.3, z=2),
        bottle(660, 54, 520),
        logo_left(70, 60, 210),
        text("headline", "Sparkling botanicals for slow summer nights", 70, 160, 480, 200, 50, serif=True,
             bold=True),
        text("subhead", "Small-batch, lightly sweet, made with real fruit and herbs.", 70, 380, 460, 64, 22,
             color=rgba(WHITE, 0.8)),
        pill("cta", 70, 480, 260, 72, "Shop Now", 24),
    ], "Landscape split layout: copy on a plum panel, product on a half-width gradient image.",
        ["landscape", "dark", "solid-bg", "partial-image-bg", "groups", "deco-overlap"]))

    # c13 -- landscape 1200x628, light, two badges, no logo
    A(ad("c13_landscape_cucumber_mint", "holdout", 1200, 628, [
        bg_fill(1200, 628, MIST),
        ellipse("deco_leaf", 60, 84, 460, 460, SAGE, 0.3),
        bottle(40, 64, 500),
        text("wordmark", "SOLSTICE", 580, 70, 300, 32, 22, bold=True, color=PLUM, letter_spacing=6),
        text("headline", "Cucumber Mint is back", 580, 120, 470, 150, 58, serif=True, bold=True, color=PLUM),
        text("subhead", "Cool, crisp and barely sweet. Back for the summer in cans and bottles.",
             580, 290, 540, 72, 22, color=rgba(PLUM, 0.8)),
        pill("badge1", 580, 390, 230, 44, "BACK BY DEMAND", 15, fill=SAGE),
        pill("badge2", 826, 390, 190, 44, "SUMMER ONLY", 15, fill=GOLD, color=PLUM),
        pill("cta", 580, 470, 320, 76, "Find It Near You", 24, fill=PLUM, color=CREAM),
    ], "Light landscape, product left, copy right, two badges, text wordmark instead of the logo image.",
        ["landscape", "light", "solid-bg", "no-logo", "badges-2", "groups", "deco-overlap"]))

    # c14 -- 1280x720 thumbnail, rotated badge
    A(ad("c14_thumbnail_taste_test", "train", 1280, 720, [
        bg_image(1280, 720),
        ellipse("deco_disc", 760, 140, 460, 460, CREAM, 0.92),
        bottle(720, 100, 540),
        logo_left(80, 60, 210),
        text("headline", "Taste test: three new botanicals", 80, 170, 620, 320, 84, serif=True, bold=True),
        text("subhead", "We tried every flavour so you can pick your favourite.", 80, 510, 580, 84, 28,
             color=rgba(WHITE, 0.85)),
        pill("badge", 1010, 44, 210, 64, "NEW VIDEO", 24, fill=GOLD, color=PLUM, radius=10, rotation=10),
    ], "1280x720 thumbnail: huge 3-line headline, product on a solid disc (intentional layering), "
       "rotated badge, no button.",
        ["landscape", "dark", "gradient-bg", "rotated", "badges-1", "deco-overlap", "no-cta"]))

    # c15 -- 1280x720, text only on orange
    A(ad("c15_landscape_text_only", "holdout", 1280, 720, [
        bg_fill(1280, 720, ORANGE),
        text("headline", "Bubbles, botanicals and a little sunshine.", 120, 150, 1040, 250, 96, serif=True,
             bold=True, align="center"),
        text("subhead", "Solstice botanical sparkling drinks. Now in four flavours.", 190, 420, 900, 50, 30,
             color=PLUM, align="center"),
        pill("cta", 465, 530, 350, 90, "See the Flavours", 28, fill=PLUM, color=CREAM),
    ], "Text-only landscape on a solid orange: no image, no logo, one grouped button (5 elements).",
        ["landscape", "mid-tone", "solid-bg", "text-only", "no-image", "no-logo", "groups"]))

    # c16 -- wide banner 1584x396
    A(ad("c16_banner_bottled_glow", "train", 1584, 396, [
        bg_image(1584, 396),
        ellipse("deco_halo", 960, -160, 560, 560, WHITE, 0.08),
        logo(40, 148, 300),
        text("headline", "Bottled golden hour", 400, 116, 580, 80, 58, serif=True, bold=True),
        text("subhead", "Botanical sparkling, crafted in small batches.", 400, 212, 580, 34, 22,
             color=rgba(WHITE, 0.85)),
        bottle(990, 18, 360),
        pill("cta", 1360, 158, 190, 80, "Shop Now", 24),
    ], "Wide 4:1 banner: logo left, copy centre, product and button right.",
        ["banner", "dark", "gradient-bg", "groups", "bleed"]))

    # c17 -- wide banner, light, three badges, no image
    A(ad("c17_banner_summer_flavours", "holdout", 1584, 396, [
        bg_fill(1584, 396, CREAM),
        ellipse("deco_sun", -120, -140, 440, 440, GOLD, 0.3),
        ellipse("deco_dusk", 1180, 200, 360, 360, ORANGE, 0.18),
        text("wordmark", "SOLSTICE", 90, 56, 300, 40, 26, bold=True, color=PLUM, letter_spacing=8),
        text("headline", "Summer flavours are here", 90, 110, 880, 84, 64, serif=True, bold=True, color=PLUM),
        text("subhead", "Four botanical sparkling drinks, one long season.", 90, 210, 880, 44, 26,
             color=rgba(PLUM, 0.8)),
        pill("badge1", 90, 290, 210, 46, "BLOOD ORANGE", 16),
        pill("badge2", 316, 290, 230, 46, "CUCUMBER MINT", 16, fill=SAGE),
        pill("badge3", 562, 290, 240, 46, "HONEY ROSEMARY", 16, fill=GOLD, color=PLUM),
        pill("cta", 1220, 158, 280, 80, "Shop Flavours", 24, fill=PLUM, color=CREAM),
    ], "Light wide banner, no image at all, three badges, overlapping translucent circles.",
        ["banner", "light", "solid-bg", "no-image", "no-logo", "badges-3", "groups", "deco-overlap", "bleed"]))

    # c18 -- 300x250, dark
    A(ad("c18_mrec_chase_light", "train", 300, 250, [
        bg_image(300, 250),
        logo(100, 10, 100),
        bottle(0, 60, 160),
        text("headline", "Chase the light", 168, 66, 120, 62, 24, serif=True, bold=True),
        text("subhead", "Sparkling botanicals", 168, 134, 120, 38, 13, color=rgba(WHITE, 0.85)),
        pill("cta", 168, 186, 112, 36, "Shop Now", 13),
    ], "Small 300x250 medium rectangle: tiny type, product left, copy right.",
        ["small", "dark", "gradient-bg", "groups"]))

    # c19 -- 300x250, light sale
    A(ad("c19_mrec_weekend_sale", "holdout", 300, 250, [
        bg_fill(300, 250, PEACH),
        ellipse("deco_sun", 160, -30, 190, 190, ORANGE, 0.22),
        bottle(158, 60, 142),
        text("kicker", "WEEKEND ONLY", 16, 20, 140, 18, 11, bold=True, color=ORANGE, letter_spacing=2),
        text("discount", "20% off", 16, 42, 140, 50, 36, serif=True, bold=True, color=PLUM),
        text("subhead", "Every mixed case of Solstice sparkling.", 16, 100, 134, 54, 13, color=rgba(PLUM, 0.8)),
        pill("cta", 16, 170, 120, 36, "Shop Sale", 13, fill=PLUM, color=CREAM),
        text("legal", "Ends Sunday. Online only. Please enjoy responsibly.", 16, 224, 268, 16, 10,
             color=rgba(PLUM, 0.6)),
    ], "Light 300x250 sale ad with a small legal line and an overlapping translucent circle.",
        ["small", "light", "solid-bg", "no-logo", "groups", "legal", "deco-overlap", "bleed"]))

    # c20 -- 728x90 leaderboard, dark
    A(ad("c20_leaderboard_dark", "train", 728, 90, [
        bg_fill(728, 90, PLUM),
        logo(6, 15, 180),
        text("headline", "Chase the light", 196, 12, 360, 40, 28, serif=True, bold=True),
        text("subhead", "Botanical sparkling drinks, small-batch.", 196, 54, 360, 22, 14,
             color=rgba(WHITE, 0.8)),
        pill("cta", 584, 22, 130, 46, "Shop Now", 16),
    ], "728x90 strip: logo, two lines of copy, button.",
        ["strip", "dark", "solid-bg", "groups"]))

    # c21 -- 728x90 leaderboard, orange, one badge
    A(ad("c21_leaderboard_new_flavor", "holdout", 728, 90, [
        bg_fill(728, 90, ORANGE),
        ellipse("deco_disc", 6, 6, 78, 78, CREAM, 0.95, z=2),
        bottle(0, 0, 90),
        text("headline", "Blood Orange Sunset", 96, 12, 330, 40, 28, serif=True, bold=True),
        text("subhead", "New botanical sparkling, out now.", 96, 54, 330, 22, 14, color=PLUM),
        pill("badge", 446, 30, 100, 30, "NEW", 14, fill=GOLD, color=PLUM),
        pill("cta", 584, 22, 130, 46, "Try It", 16, fill=PLUM),
    ], "728x90 strip on orange: product, copy, one badge, button; no logo.",
        ["strip", "mid-tone", "solid-bg", "no-logo", "badges-1", "groups"]))

    # c22 -- 160x600 skyscraper
    A(ad("c22_skyscraper", "train", 160, 600, [
        bg_image(160, 600),
        ellipse("deco_halo", -60, 380, 220, 220, WHITE, 0.08),
        logo(20, 24, 120),
        bottle(0, 90, 160),
        text("headline", "Chase the light", 10, 270, 140, 66, 26, serif=True, bold=True, align="center"),
        text("subhead", "Sparkling botanicals crafted for golden hour.", 14, 350, 132, 80, 14,
             color=rgba(WHITE, 0.85), align="center"),
        pill("cta", 15, 510, 130, 44, "Shop Now", 15),
    ], "Tall 160x600 skyscraper: stacked logo, product, copy, button.",
        ["skyscraper", "dark", "gradient-bg", "groups", "bleed"]))

    # c23 -- 970x250 billboard, light, two badges, legal
    A(ad("c23_billboard_weekend_lineup", "train", 970, 250, [
        bg_fill(970, 250, PEACH),
        ellipse("deco_sun", 10, -30, 280, 280, ORANGE, 0.22),
        bottle(30, 10, 240),
        text("headline", "Long-weekend lineup", 300, 30, 430, 52, 38, serif=True, bold=True, color=PLUM),
        text("subhead", "Four botanical sparkling drinks, chilled and delivered by Friday.", 300, 88, 420, 48,
             17, color=rgba(PLUM, 0.8)),
        pill("badge1", 300, 150, 160, 32, "FREE DELIVERY", 12, fill=SAGE),
        pill("badge2", 472, 150, 120, 32, "4 FLAVOURS", 12, fill=GOLD, color=PLUM),
        pill("cta", 760, 90, 180, 60, "Order Now", 20),
        text("legal", f"Delivery to selected areas only. {LEGAL_CO} Please enjoy responsibly.",
             300, 214, 640, 18, 11, color=rgba(PLUM, 0.6)),
    ], "Light 970x250 billboard: product, copy, two badges, button, legal line.",
        ["billboard", "light", "solid-bg", "no-logo", "badges-2", "groups", "legal", "bleed"]))

    # c24 -- 820x312 cover, no groups, no button
    A(ad("c24_cover_all_summer", "train", 820, 312, [
        bg_image(820, 312),
        ellipse("deco_sun", 470, -70, 320, 320, ORANGE, 0.22),
        ellipse("deco_moon", 620, 150, 260, 260, WHITE, 0.08),
        bottle(520, 6, 300),
        logo(0, 34, 180),
        text("headline", "Chase the light all summer", 50, 104, 440, 116, 46, serif=True, bold=True),
        text("subhead", "Botanical sparkling drinks from the Solstice kitchen.", 50, 230, 420, 52, 18,
             color=rgba(WHITE, 0.85)),
    ], "820x312 cover: no groups and no button; overlapping translucent circles.",
        ["cover", "dark", "gradient-bg", "no-groups", "no-cta", "deco-overlap", "bleed"]))

    # c25 -- square, deliberately missing sticker image
    A(ad("c25_square_missing_sticker", "train", 1080, 1080, [
        bg_image(1080, 1080),
        logo(380, 60, 320),
        bottle(290, 190, 500),
        image("sticker", "solstice_summer_sticker_2026.png", 840, 200, 180, 180, z=5),
        text("headline", "Your summer, sparkling", 90, 710, 900, 90, 68, serif=True, bold=True, align="center"),
        pill("cta", 330, 840, 420, 90, "Shop the Summer Line", 26),
        text("legal", f"Please enjoy responsibly. {LEGAL_CO}", 90, 1000, 900, 26, 14,
             color=rgba(WHITE, 0.55), align="center"),
    ], "Square ad whose corner sticker image is intentionally MISSING from the brand kit.",
        ["square", "dark", "gradient-bg", "missing-asset", "groups", "legal"]))
    ads[-1]["expects_missing_asset"] = True

    # c26 -- portrait event poster, 20 elements, rotated product, missing partner mark
    A(ad("c26_portrait_sunset_sessions", "holdout", 1080, 1350, [
        bg_image(1080, 1350),
        ellipse("deco_sun", -160, -160, 620, 620, ORANGE, 0.22),
        logo(380, 50, 320),
        text("kicker", "SOLSTICE PRESENTS", 240, 190, 600, 36, 24, bold=True, color=GOLD, align="center",
             letter_spacing=6),
        text("headline", "Sunset Sessions", 90, 236, 900, 120, 92, serif=True, bold=True, align="center"),
        text("subhead", "Live music, rooftop pours and the longest golden hour of the year",
             140, 366, 800, 96, 34, serif=True, italic=True, color=rgba(WHITE, 0.9), align="center"),
        bottle(680, 510, 360, rotation=-10),
        text("date", "Saturday 21 June", 90, 520, 560, 50, 38, bold=True),
        text("details", "6:00 pm until sundown, Harbour Rooftop, Pier 9", 90, 580, 540, 80, 26,
             color=rgba(WHITE, 0.85)),
        pill("badge1", 90, 700, 190, 48, "FREE ENTRY", 16),
        pill("badge2", 296, 700, 160, 48, "ALL AGES", 16, fill=SAGE),
        pill("badge3", 472, 700, 170, 48, "TASTINGS", 16, fill=GOLD, color=PLUM),
        pill("cta", 330, 960, 420, 90, "RSVP Free", 28),
        text("rsvp", "Reserve a spot at solstice.example/sessions", 190, 1070, 700, 36, 22,
             color=rgba(WHITE, 0.8), align="center"),
        image("partner", "solstice_partner_mark.png", 860, 1180, 160, 70, z=2),
        text("legal", "Free event, registration required. Capacity is limited and entry is first come, "
             f"first served. {LEGAL_CO}", 90, 1250, 740, 50, 14, color=rgba(WHITE, 0.6)),
    ], "Busy event poster (20 elements): rotated product, italic serif subhead, three badges, "
       "a MISSING partner-mark image, legal.",
        ["portrait", "dark", "gradient-bg", "20-elements", "rotated", "badges-3", "missing-asset", "legal",
         "groups", "deco-overlap", "bleed", "italic"]))
    ads[-1]["expects_missing_asset"] = True

    # c27 -- square, italic quote on plum
    A(ad("c27_square_tasting_note", "holdout", 1080, 1080, [
        bg_fill(1080, 1080, PLUM),
        ellipse("deco_sun", 600, 600, 640, 640, ORANGE, 0.15),
        text("quote_mark", "\u201c", 80, 100, 200, 210, 160, serif=True, bold=True, color=GOLD),
        text("quote", "Like a sunset in a glass: bright, a little bitter, and gone too soon.",
             90, 320, 880, 160, 58, serif=True, italic=True),
        text("attribution", "THE SOLSTICE TASTING TEAM", 90, 520, 500, 34, 22, bold=True, color=GOLD,
             letter_spacing=2),
        bottle(620, 640, 420),
        logo_left(90, 930, 240),
    ], "Quote-style ad: oversized quote glyph, italic serif quote, product bottom-right, no groups.",
        ["square", "dark", "solid-bg", "italic", "no-groups", "no-cta", "bleed"]))

    # c28 -- square, translucent panel behind copy, two-colour headline
    hl = "Sip the Solstice"
    A(ad("c28_square_panel", "train", 1080, 1080, [
        bg_image(1080, 1080),
        shape("panel", 90, 630, 900, 360, rgba(PLUM, 0.7), radius=32, z=1),
        logo(400, 40, 280),
        bottle(315, 150, 460),
        text("headline", hl, 130, 670, 820, 90, 68, serif=True, bold=True, align="center",
             runs=[run(hl, "Solstice", ORANGE)]),
        text("subhead", "Four botanical sparkling drinks, each one a different hour of the longest day.",
             160, 770, 760, 84, 26, color=rgba(WHITE, 0.85), align="center"),
        pill("cta", 360, 880, 360, 80, "Explore Flavours", 24),
    ], "Translucent rounded panel behind the copy block, two-colour headline.",
        ["square", "dark", "gradient-bg", "panel", "style-runs", "groups"]))

    # c29 -- portrait, light, grouped feature list, logo on a pill
    els = [bg_fill(1080, 1350, CREAM),
           ellipse("deco_leaf", 540, 300, 520, 520, SAGE, 0.25),
           shape("logo_pill", 90, 70, 270, 90, PLUM, radius=45, z=2, group="logo_lockup"),
           image("logo", LOGO, 105, 75, 240, 80, z=3, group="logo_lockup"),
           text("headline", "Why it tastes like summer", 90, 210, 900, 90, 66, serif=True, bold=True,
                color=PLUM),
           bottle(540, 300, 520)]
    feats = ["Pressed blood orange and citrus peel", "Wild honey, never cane sugar",
             "Garden herbs, steeped overnight", "Fine bubbles that last to the last sip"]
    for i, feat in enumerate(feats):
        y = 360 + 120 * i
        els += [shape(f"feat{i + 1}_dot", 90, y + 12, 24, 24, ORANGE, kind="ellipse", z=2,
                      group=f"feat{i + 1}"),
                text(f"feat{i + 1}_text", feat, 136, y, 400, 84, 28, color=PLUM, group=f"feat{i + 1}")]
    els += [pill("cta", 90, 900, 400, 90, "Taste the Difference", 26),
            text("body", "Find Solstice in the chilled aisle of your favourite local grocer.",
                 90, 1060, 900, 80, 28, color=rgba(PLUM, 0.8)),
            text("legal", f"{LEGAL_CO} Please enjoy responsibly.", 90, 1270, 900, 26, 14,
                 color=rgba(PLUM, 0.55))]
    A(ad("c29_portrait_feature_list", "train", 1080, 1350, els,
         "Light portrait with a four-item feature list (dot + text grouped) and a logo grouped on a dark pill.",
         ["portrait", "light", "solid-bg", "groups", "list", "legal", "deco-overlap"]))

    # c30 -- story, minimal
    A(ad("c30_story_minimal_glow", "train", 1080, 1920, [
        bg_fill(1080, 1920, PLUM),
        logo(390, 140, 300),
        bottle(90, 340, 900),
        text("headline", "Glow.", 90, 1260, 900, 224, 170, serif=True, bold=True, color=CREAM, align="center"),
        text("cta_text", "SHOP SOLSTICE", 290, 1580, 500, 44, 30, bold=True, color=GOLD, align="center",
             letter_spacing=6),
    ], "Minimal story: 5 elements, huge product, one-word headline, text-only call to action.",
        ["story", "dark", "solid-bg", "5-elements", "no-groups"]))

    return ads


# ---------------------------------------------------------------------------
# Samples, checks, output
# ---------------------------------------------------------------------------

def samples() -> list[dict]:
    out = []
    for path in sorted(INPUT_DIR.glob("*.json")):
        task = json.loads(path.read_text())
        out.append({"task_id": task["task_id"], "split": "train", "origin": "sample",
                    "expects_missing_asset": _missing_assets(task["source"]) != [],
                    "notes": task.get("notes", ""), "features": ["sample", task.get("difficulty", "")],
                    "source": task["source"]})
    return out


def _missing_assets(source: dict) -> list[str]:
    return [el["asset_id"] for el in source["elements"]
            if el["type"] == "image" and not (ASSETS_DIR / el["asset_id"]).is_file()]


def _box(el: dict) -> tuple[float, float, float, float]:
    from engine.pipeline.geometry import rotated_aabb
    return tuple(rotated_aabb(el))


def _intersect(a, b) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _contains(outer, inner) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and inner[0] + inner[2] <= outer[0] + outer[2] and inner[1] + inner[3] <= outer[1] + outer[3])


# Overlaps a designer made on purpose (a sticker or badge laid over the
# product box), so the checker doesn't flag them.
INTENDED_OVERLAPS = {
    "c05_square_orange_sticker": {("product", "sticker_label")},
    "c14_thumbnail_taste_test": {("product", "badge_label")},
}


def check(record: dict) -> list[str]:
    """Layout warnings for one ad: overflowing text, off-canvas content,
    accidental overlaps. Decorative shapes (id deco*) and backgrounds may bleed."""
    from renderer.converter.convert import _get_renderer
    from renderer.converter.wrap import char_widths, wrap_text

    fm = _get_renderer(str(FONTS_DIR)).font_manager
    src = record["source"]
    cw, ch = src["canvas"]["width"], src["canvas"]["height"]
    warnings = []
    ids = [el["id"] for el in src["elements"]]
    if len(ids) != len(set(ids)):
        warnings.append("duplicate element ids")
    if record["expects_missing_asset"] != bool(_missing_assets(src)):
        warnings.append(f"expects_missing_asset is wrong: {_missing_assets(src)}")
    for el in src["elements"]:
        if el["type"] == "text":
            weight = el.get("font_weight", "regular")
            italic = el.get("font_style") == "italic"
            style = {(False, False): "regular", (True, False): "bold", (False, True): "italic",
                     (True, True): "bolditalic"}[(weight == "bold", italic)]
            ls = el.get("letter_spacing", 0)
            widths = [w + ls for w in char_widths(fm, el["font_family"], weight, style, el["font_size"],
                                                 el["content"])]
            lines = wrap_text(el["content"], widths, el["width"]).split("\n")
            pos, widest = 0, 0.0
            for line in lines:
                widest = max(widest, sum(widths[pos:pos + len(line)]))
                pos += len(line) + 1
            need = (len(lines) - 1) * el.get("line_height", 1.2) * el["font_size"] + 1.3 * el["font_size"]
            if need > el["height"] + 0.5:
                warnings.append(f"{el['id']}: {len(lines)} lines need {need:.0f}px, box is {el['height']}")
            if widest > el["width"] + 0.5:
                warnings.append(f"{el['id']}: a line is {widest:.0f}px wide, box is {el['width']}")
        if el["id"] == "bg" or el["id"].startswith("deco"):
            continue
        x, y, w, h = _box(el)
        if x < -0.5 or y < -0.5 or x + w > cw + 0.5 or y + h > ch + 0.5:
            warnings.append(f"{el['id']}: off canvas ({x:.0f},{y:.0f},{w:.0f},{h:.0f})")
    content = [el for el in src["elements"] if el["id"] != "bg" and not el["id"].startswith("deco")]
    for i, a in enumerate(content):
        for b in content[i + 1:]:
            if a.get("group_id") and a.get("group_id") == b.get("group_id"):
                continue
            if "text" not in (a["type"], b["type"]):
                continue
            if (a["id"], b["id"]) in INTENDED_OVERLAPS.get(record["task_id"], set()):
                continue
            shape_el, text_el = (a, b) if a["type"] == "shape" else (b, a)
            if shape_el["type"] == "shape" and _contains(_box(shape_el), _box(text_el)):
                continue  # a container: button, badge or panel behind its text
            if _intersect(_box(a), _box(b)):
                warnings.append(f"overlap: {a['id']} x {b['id']}")
    return warnings


def contact_sheet(records: list[dict], path: Path, cell: int = 420) -> None:
    from engine.pipeline.render import render_to_image

    cols = 6
    rows = -(-len(records) // cols)
    label_h = 34
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), (236, 236, 236))
    draw = ImageDraw.Draw(sheet)
    for i, rec in enumerate(records):
        img = render_to_image(rec["source"], assets_dir=ASSETS_DIR, max_size=cell - 16)
        img.save(path.parent / f"{rec['task_id']}.png")
        cx, cy = (i % cols) * cell, (i // cols) * (cell + label_h)
        sheet.paste(img, (cx + (cell - img.width) // 2, cy + label_h + (cell - img.height) // 2))
        c = rec["source"]["canvas"]
        label = f"{rec['task_id']}  [{rec['split']}]  {c['width']}x{c['height']}  n={len(rec['source']['elements'])}"
        draw.text((cx + 8, cy + 10), label, fill=(20, 20, 20))
    sheet.save(path)


def main() -> None:
    records = samples() + drafted()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    keep = {f"{r['task_id']}.json" for r in records}
    for stale in CORPUS_DIR.glob("*.json"):
        if stale.name not in keep and json.loads(stale.read_text()).get("origin") in ("sample", "claude-drafted"):
            stale.unlink()
    problems = 0
    for rec in records:
        for w in check(rec) if rec["origin"] != "sample" else []:
            print(f"[warn] {rec['task_id']}: {w}")
            problems += 1
        (CORPUS_DIR / f"{rec['task_id']}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    contact_sheet(records, REVIEW_DIR / "sheet.png")
    splits = {s: sum(r["split"] == s for r in records) for s in ("train", "holdout")}
    print(f"wrote {len(records)} ads ({splits['train']} train, {splits['holdout']} holdout) to {CORPUS_DIR}")
    print(f"{problems} layout warnings; review sheet: {REVIEW_DIR / 'sheet.png'}")


if __name__ == "__main__":
    main()
