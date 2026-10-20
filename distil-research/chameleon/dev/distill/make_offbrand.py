"""Write the off-brand ads: 30 hand-authored designs for four invented brands
that are not Solstice, so the corpus teaches "what looks good", not "what
looks good for Solstice".

    cd distil-research/chameleon && python -m dev.distill.make_offbrand

Like make_corpus.py, every ad is explicit data (copy, colours, positions),
written the way a designer would build it. Only the synthetic stand-in images
(training-data/corpus/assets/syn_*.png) and the renderer's two font families
are used. Text box heights are not typed in by hand: `Col` stacks elements
down a column and sizes each text box with the engine's TextMeasurer, so the
copy always fits its box exactly as the engine measures it.

Brands (all invented): Quillbrook Roasters (coffee), Voltstride (a sneaker
drop), Inkwell & Lantern (a bookstore's events), Pacewell (a fitness app).
22 ads are train, 8 holdout (two per brand). Each ad must pass
make_corpus.check(strict) and make_corpus.pixel_check; one that fails is
reported and not written.
"""

from dev.distill import make_corpus as mc
from dev.distill.augment import _hex_rgba, _needed_height, _text_width
from dev.distill.assets import merged_assets_dir

REVIEW_SHEET = mc.REVIEW_DIR / "sheet_offbrand.png"


def c(hex_color: str, alpha: float = 1.0) -> str:
    return _hex_rgba(hex_color, alpha)


# Palettes (dark, light, accents) per brand.
Q = dict(dark="#3B2A20", cream="#F6EDE0", latte="#EADBC8", copper="#A5531F", caramel="#E0A458", sage="#7D8F69")
V = dict(dark="#1B0F2E", purple="#7B2FF7", lime="#C6F432", pink="#FF4FA3", white="#F5F0FF", plum="#2A1B4A")
I = dict(ink="#2A1B3D", paper="#F4F1EA", rust="#B5472F", gold="#E9B44C", teal="#264653", white="#FFFFFF")
P = dict(dark="#222222", cream="#FFF8D6", orange="#FF7A00", yellow="#FFD23F", green="#2E9E5B", white="#FFFFFF")

QLOGO, VLOGO, ILOGO, PLOGO = ("syn_logo_quillbrook_badge.png", "syn_logo_voltstride_light.png",
                              "syn_logo_inkwell_stacked.png", "syn_logo_pacewell.png")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def img(id, asset, x, y, *, w=None, h=None, fit="contain", z=3, rotation=0, group=None) -> dict:
    """An image whose box has the asset's own aspect (give w or h)."""
    aspect = mc._image_aspect(asset)
    if fit == "cover":
        return mc.image(id, asset, x, y, w, h, fit="cover", z=z, rotation=rotation, group=group)
    if w is None:
        w = round(h * aspect, 1)
    h = round(w / aspect, 2)
    return mc.image(id, asset, x, y, w, h, z=z, rotation=rotation, group=group)


def t(id, content, x, y, w, size, **kw) -> dict:
    """A text element whose box height is what the engine measures for it."""
    el = mc.text(id, content, x, y, w, 10, size, **kw)
    el["height"] = _needed_height(el, size)
    return el


def bg(w, h, asset_or_color) -> dict:
    if asset_or_color.endswith(".png"):
        return mc.image("bg", asset_or_color, 0, 0, w, h, fit="cover", z=0)
    return mc.shape("bg", 0, 0, w, h, asset_or_color, z=0)


def label_pill(prefix, x, y, label, size, *, fill, color, h=None, pad=None, **kw) -> list[dict]:
    """A pill whose width follows its label."""
    h = h or round(size * 2.4)
    pad = pad if pad is not None else round(size * 1.3)
    ls = kw.get("letter_spacing") or 0
    w = round(_text_width(label, size, serif=kw.get("serif", False), letter_spacing=ls) + 2 * pad)
    return mc.pill(prefix, round(x), round(y), w, h, label, size, fill=fill, color=color, **kw)


class Col:
    """Stack elements down a column of width w at x, starting at y."""

    def __init__(self, x, y, w, align="left"):
        self.x, self.y, self.w, self.align = x, y, w, align
        self.els: list[dict] = []

    def _x_for(self, width):
        return {"left": self.x, "center": self.x + (self.w - width) / 2, "right": self.x + self.w - width}[self.align]

    def text(self, id, content, size, gap=0, **kw) -> dict:
        el = mc.text(id, content, self.x, round(self.y), self.w, 10, size, align=self.align, **kw)
        el["height"] = _needed_height(el, size)
        self.els.append(el)
        self.y += el["height"] + gap
        return el

    def pill(self, prefix, label, size, gap=0, w=None, h=None, **kw) -> list[dict]:
        h = h or round(size * 3.2)
        if w is None:
            w = round(_text_width(label, size) + size * 3.4)
        els = mc.pill(prefix, round(self._x_for(w)), round(self.y), w, h, label, size, **kw)
        self.els += els
        self.y += h + gap
        return els

    def badges(self, specs, size, gap=0, gap_x=None, h=None) -> list[dict]:
        """A row of badge pills: specs = [(prefix, label, fill, color), ...]."""
        h = h or round(size * 2.4)
        gap_x = gap_x if gap_x is not None else round(size * 0.9)
        pad = round(size * 1.2)
        widths = [round(_text_width(lbl, size, letter_spacing=1) + 2 * pad) for _, lbl, _, _ in specs]
        x = self._x_for(sum(widths) + gap_x * (len(specs) - 1))
        out = []
        for (prefix, lbl, fill, color), w in zip(specs, widths):
            out += mc.pill(prefix, round(x), round(self.y), w, h, lbl, size, fill=fill, color=color, letter_spacing=1)
            x += w + gap_x
        self.els += out
        self.y += h + gap
        return out

    def image(self, id, asset, gap=0, w=None, h=None, **kw) -> dict:
        probe = img(id, asset, 0, 0, w=w, h=h, **kw)
        el = img(id, asset, round(self._x_for(probe["width"])), round(self.y), w=probe["width"], **kw)
        self.els.append(el)
        self.y += el["height"] + gap
        return el

    def space(self, g):
        self.y += g

    def center_in(self, y0, y1) -> "Col":
        """Move the whole stack so it is vertically centred in [y0, y1]."""
        top = min(e["y"] for e in self.els)
        bottom = max(e["y"] + e["height"] for e in self.els)
        dy = round((y0 + y1 - top - bottom) / 2)
        for e in self.els:
            e["y"] += dy
        return self


def ad(task_id, split, w, h, elements, notes, features) -> dict:
    rec = mc.ad(task_id, split, w, h, elements, notes, features)
    rec["origin"], rec["brand"] = "offbrand", "offbrand"
    return rec


# ---------------------------------------------------------------------------
# The ads
# ---------------------------------------------------------------------------

def quillbrook() -> list[dict]:
    ads = []
    A = ads.append
    legal = "Quillbrook Roasters, 2026. Roasted to order; ships in 2 to 4 working days."

    # o01 -- square, dark roast background, product right, copy left
    col = Col(90, 0, 470)
    col.text("headline", "House Blend, slow roasted.", 70, 30, serif=True, bold=True, color=c(Q["cream"]),
             runs=None)
    col.text("subhead", "Chocolate, toasted hazelnut and a little dried cherry. Roasted on Mondays, at your door "
             "by Thursday.", 26, 46, color=c(Q["cream"], 0.85))
    col.pill("cta", "Shop the Blend", 24, 24, fill=c(Q["caramel"]), color=c(Q["dark"]))
    col.text("note", "Whole bean or ground to order", 20, color=c(Q["cream"], 0.7))
    col.center_in(250, 950)
    A(ad("o01_q_square_house_blend", "train", 1080, 1080, [
        bg(1080, 1080, "syn_bg_roast_ember.png"),
        mc.ellipse("deco_glow", 560, 270, 500, 500, c(Q["caramel"]), 0.18),
        img("logo", QLOGO, 90, 70, h=130),
        img("product", "syn_prod_round_tin.png", 590, 320, w=420),
        *col.els,
        *label_pill("badge", 820, 110, "NEW HARVEST", 18, fill=c(Q["sage"]), color=c(Q["cream"]), letter_spacing=2),
    ], "Coffee roaster, dark photo-like background, copy left, round tin product right, badge top right.",
        ["square", "dark", "image-bg", "groups", "badges-1", "deco-overlap", "offbrand", "synthetic-image"]))

    # o02 -- portrait, light, numbered brew steps (grouped), bag product, legal
    els = [bg(1080, 1350, c(Q["cream"])),
           mc.ellipse("deco_sun", 640, -160, 600, 600, c(Q["caramel"]), 0.25),
           img("logo", QLOGO, 90, 70, h=120),
           img("product", "syn_prod_coffee_bag.png", 700, 120, w=300)]
    col = Col(90, 250, 560)
    col.text("kicker", "POUR-OVER, PROPERLY", 22, 16, bold=True, color=c(Q["copper"]), letter_spacing=4)
    col.text("headline", "Three steps to a better cup", 72, 40, serif=True, bold=True, color=c(Q["dark"]))
    els += col.els
    steps = ["Rinse the filter and warm the cup with hot water.",
             "Bloom 20 g of coffee with 40 g of water for 30 seconds.",
             "Pour the rest in slow circles until you reach 320 g."]
    y = 580
    for i, step in enumerate(steps, 1):
        num = mc.text(f"step{i}_num", f"0{i}", 90, y, 110, 90, 64, serif=True, bold=True, color=c(Q["copper"]),
                      group=f"step{i}")
        txt = mc.text(f"step{i}_text", step, 220, y + 14, 760, 10, 30, color=c(Q["dark"]), group=f"step{i}")
        txt["height"] = _needed_height(txt, 30)
        els += [num, txt]
        y += 140
    els += mc.pill("cta", 90, 1040, 380, 90, "Get the Brew Kit", 26, fill=c(Q["dark"]), color=c(Q["cream"]))
    leg = mc.text("legal", legal, 90, 1270, 900, 10, 16, color=c(Q["dark"], 0.65))
    leg["height"] = _needed_height(leg, 16)
    els.append(leg)
    A(ad("o02_q_portrait_pour_over", "holdout", 1080, 1350, els,
         "Light portrait how-to: kicker, two-line headline, three grouped numbered steps, bag product, legal.",
         ["portrait", "light", "solid-bg", "text-groups", "legal", "offbrand", "synthetic-image"]))

    # o03 -- story, BUSY terrazzo background; all text on a calm opaque panel
    col = Col(150, 0, 780, "center")
    col.text("headline", "Cold brew, bottled this morning", 76, 28, serif=True, bold=True, color=c(Q["dark"]))
    col.text("subhead", "Steeped for 18 hours, never heated, never bitter.", 32, 44, color=c(Q["dark"], 0.85))
    col.pill("cta", "Order a Case", 30, 30, fill=c(Q["copper"]), color=c(Q["cream"]))
    col.text("legal", "Keep refrigerated. Best within 10 days of opening.", 18, color=c(Q["dark"], 0.7))
    col.center_in(1150, 1770)
    A(ad("o03_q_story_cold_brew", "train", 1080, 1920, [
        bg(1080, 1920, "syn_busy_terrazzo.png"),
        mc.shape("panel", 90, 1100, 900, 720, c(Q["cream"]), radius=40, z=1),
        img("logo", QLOGO, 440, 90, h=200),
        img("product", "syn_prod_tall_bottle.png", 0, 340, h=700) | {"x": 405},
        *col.els,
    ], "Story on a BUSY terrazzo background: every text sits on an opaque cream panel; product and logo on "
       "the busy area.", ["story", "busy-bg", "panel", "groups", "legal", "offbrand", "synthetic-image"]))

    # o04 -- leaderboard strip
    A(ad("o04_q_leaderboard", "train", 728, 90, [
        bg(728, 90, c(Q["dark"])),
        img("logo", QLOGO, 12, 9, h=72),
        mc.text("headline", "Fresh roast, every Monday", 100, 14, 420, 38, 28, serif=True, bold=True,
                color=c(Q["cream"])),
        mc.text("subhead", "Free delivery on two bags or more.", 100, 54, 420, 22, 15, color=c(Q["cream"], 0.8)),
        *mc.pill("cta", 572, 22, 140, 46, "Shop Coffee", 16, fill=c(Q["caramel"]), color=c(Q["dark"])),
    ], "728x90 strip: round badge logo, two lines of copy, button.",
        ["strip", "dark", "solid-bg", "groups", "offbrand", "synthetic-image"]))

    # o05 -- 300x250, light latte, legal
    A(ad("o05_q_mrec_subscribe", "train", 300, 250, [
        bg(300, 250, c(Q["latte"])),
        img("product", "syn_prod_round_tin.png", 170, 50, w=120),
        mc.text("kicker", "SUBSCRIBE & SAVE", 16, 18, 160, 18, 11, bold=True, color=c(Q["copper"]),
                letter_spacing=1.5),
        mc.text("headline", "15% off every bag", 16, 40, 150, 62, 24, serif=True, bold=True, color=c(Q["dark"])),
        *mc.pill("cta", 16, 120, 126, 36, "Start Saving", 13, fill=c(Q["dark"]), color=c(Q["cream"])),
        mc.text("legal", "Cancel any time. Quillbrook Roasters.", 16, 222, 268, 16, 10, color=c(Q["dark"], 0.7)),
    ], "Light 300x250 subscription ad, tiny type, product right, legal line.",
        ["small", "light", "solid-bg", "groups", "legal", "offbrand", "synthetic-image"]))

    # o06 -- landscape split: dark copy panel left, photo-like image right, rotated badge
    col = Col(70, 0, 480)
    col.text("headline", "Your coffee, on repeat", 54, 22, serif=True, bold=True, color=c(Q["cream"]))
    col.text("subhead", "Pick a roast, pick a rhythm. Skip or pause whenever you like.", 22, 30,
             color=c(Q["cream"], 0.85))
    col.pill("cta", "Build a Subscription", 22, fill=c(Q["caramel"]), color=c(Q["dark"]))
    col.center_in(150, 560)
    A(ad("o06_q_landscape_subscription", "holdout", 1200, 628, [
        bg(1200, 628, c(Q["dark"])),
        mc.image("photo", "syn_bg_roast_ember.png", 620, 0, 580, 628, fit="cover", z=1),
        img("logo", QLOGO, 70, 40, h=90),
        img("product", "syn_prod_coffee_bag.png", 760, 90, h=450),
        *col.els,
        *label_pill("badge", 300, 64, "SAVE 15%", 20, fill=c(Q["sage"]), color=c(Q["cream"]), rotation=-6,
                    z=6),
    ], "Landscape split: copy on a dark panel, product on a half-width photo-like image, rotated badge.",
        ["landscape", "dark", "solid-bg", "partial-image-bg", "rotated", "badges-1", "groups", "offbrand",
         "synthetic-image"]))

    # o07 -- text-only quote, 4 elements
    col = Col(110, 0, 860)
    col.text("quote_mark", "\u201c", 160, 0, serif=True, bold=True, color=c(Q["copper"]))
    col.text("quote", "The best cup I have had outside of my grandmother's kitchen.", 60, 40, serif=True,
             italic=True, color=c(Q["dark"]))
    col.text("attribution", "A QUILLBROOK REGULAR, SINCE 2019", 22, bold=True, color=c(Q["copper"]),
             letter_spacing=3)
    col.center_in(0, 1080)
    A(ad("o07_q_square_quote", "train", 1080, 1080, [bg(1080, 1080, c(Q["latte"])), *col.els],
         "Text-only quote ad: 4 elements, oversized quote mark, italic serif.",
         ["square", "light", "solid-bg", "text-only", "no-image", "italic", "no-groups", "offbrand"]))

    # o08 -- wide banner, light haze, two badges
    A(ad("o08_q_banner_tasting", "train", 1584, 396, [
        bg(1584, 396, "syn_bg_peach_haze.png"),
        img("logo", QLOGO, 60, 98, h=200),
        mc.text("headline", "Free Saturday cupping", 320, 80, 820, 80, 56, serif=True, bold=True,
                color=c(Q["dark"])),
        mc.text("subhead", "Taste six single origins with our roasters, 10 am at the roastery.", 320, 176, 820,
                40, 26, color=c(Q["dark"], 0.85)),
        *label_pill("badge1", 320, 252, "FREE", 16, fill=c(Q["copper"]), color=c(Q["cream"]), letter_spacing=1),
        *label_pill("badge2", 418, 252, "LIMITED SPOTS", 16, fill=c(Q["dark"]), color=c(Q["cream"]),
                    letter_spacing=1),
        *mc.pill("cta", 1240, 158, 280, 80, "Save a Seat", 24, fill=c(Q["dark"]), color=c(Q["cream"])),
    ], "Wide banner on a light photo-like haze: round logo left, copy, two badges, button right.",
        ["banner", "light", "image-bg", "badges-2", "groups", "offbrand", "synthetic-image"]))
    return ads


def voltstride() -> list[dict]:
    ads = []
    A = ads.append

    # o09 -- square, BUSY confetti; copy on an opaque panel
    col = Col(150, 0, 780, "center")
    col.text("kicker", "DROP 07", 26, 10, bold=True, color=c(V["lime"]), letter_spacing=8)
    col.text("headline", "The Volt Runner", 84, 20, bold=True, italic=True, color=c(V["white"]))
    col.text("subhead", "Lightest sole we have ever made. 400 pairs, then gone.", 28, 34, color=c(V["white"], 0.85))
    col.pill("cta", "Enter the Draw", 26, fill=c(V["lime"]), color=c(V["dark"]))
    col.center_in(560, 1010)
    A(ad("o09_v_square_drop", "train", 1080, 1080, [
        bg(1080, 1080, "syn_busy_confetti.png"),
        mc.shape("panel", 90, 540, 900, 490, c(V["dark"]), radius=36, z=1),
        mc.shape("logo_pill", 300, 50, 480, 120, c(V["dark"]), radius=60, z=2, group="logo_lockup"),
        img("logo", VLOGO, 340, 70, w=400, group="logo_lockup"),
        img("product", "syn_prod_sneaker.png", 190, 190, w=700),
        *col.els,
    ], "Sneaker drop on a BUSY confetti background; all copy on an opaque dark panel, logo on a dark pill.",
        ["square", "busy-bg", "panel", "groups", "offbrand", "synthetic-image"]))

    # o10 -- story countdown: rotated product, 3 badges, long headline, long legal
    col = Col(90, 1050, 900, "center")
    col.text("headline", "The Volt Runner 2 lands Friday at 9 am, and this time we made enough for everyone "
             "who waited", 64, 36, bold=True, italic=True, color=c(V["white"]))
    col.badges([("badge1", "CARBON PLATE", c(V["purple"]), c(V["white"])),
                ("badge2", "142 G", c(V["lime"]), c(V["dark"])),
                ("badge3", "RECYCLED KNIT", c(V["pink"]), c(V["dark"]))], 20, 50)
    col.pill("cta", "Set a Reminder", 34, fill=c(V["lime"]), color=c(V["dark"]))
    leg = mc.text("legal", "One pair per person. Draw entries close Thursday at midnight; winners are picked at "
                  "random and emailed by 9 am Friday. Voltstride, 2026. Full terms at voltstride.example/draw.",
                  90, 0, 900, 10, 18, color=c(V["white"], 0.65), align="center")
    leg["height"] = _needed_height(leg, 18)
    leg["y"] = 1860 - leg["height"]
    A(ad("o10_v_story_countdown", "train", 1080, 1920, [
        bg(1080, 1920, "syn_bg_violet_glow.png"),
        mc.ellipse("deco_ring", 140, 260, 800, 800, c(V["purple"]), 0.25),
        img("logo", VLOGO, 290, 110, w=500),
        img("product", "syn_prod_sneaker.png", 110, 420, w=860, rotation=-12),
        *col.els, leg,
    ], "Story with a rotated product, a run-on 3-line headline, three badges in a row and a long legal block.",
        ["story", "dark", "image-bg", "rotated", "long-headline", "badges-3", "legal", "groups", "offbrand",
         "synthetic-image"]))

    # o11 -- portrait, BUSY stripes, lime panel, rotated round sticker
    col = Col(150, 0, 780, "center")
    col.text("headline", "Box fresh, half price", 80, 24, bold=True, italic=True, color=c(V["dark"]))
    col.text("subhead", "Last season's colourways, every size still in stock.", 30, 40, color=c(V["dark"]))
    col.pill("cta", "Shop the Archive", 28, fill=c(V["dark"]), color=c(V["lime"]))
    col.center_in(820, 1250)
    A(ad("o11_v_portrait_archive", "holdout", 1080, 1350, [
        bg(1080, 1350, "syn_busy_stripes.png"),
        mc.shape("panel", 90, 780, 900, 500, c(V["lime"]), radius=0, z=1),
        img("product", "syn_prod_wide_box.png", 100, 300, w=620),
        mc.shape("sticker_bg", 780, 70, 220, 220, c(V["pink"]), kind="ellipse", z=5, group="sticker", rotation=12),
        mc.text("sticker_label", "48H ONLY", 805, 145, 170, 70, 30, bold=True, color=c(V["dark"]), align="center",
                z=6, group="sticker", rotation=12),
        *col.els,
    ], "Portrait on BUSY stripes: copy on a square-cornered lime panel, wide box product, rotated round sticker.",
        ["portrait", "busy-bg", "panel", "rotated", "groups", "offbrand", "synthetic-image"]))

    # o12 -- skyscraper
    col = Col(10, 300, 140, "center")
    col.text("headline", "Run lighter", 24, 12, bold=True, italic=True, color=c(V["white"]))
    col.text("subhead", "Drop 07 is live now.", 14, color=c(V["white"], 0.85))
    col.center_in(270, 490)
    A(ad("o12_v_skyscraper", "train", 160, 600, [
        bg(160, 600, c(V["dark"])),
        img("logo", VLOGO, 15, 30, w=130),
        img("product", "syn_prod_sneaker.png", 10, 160, w=140),
        *col.els,
        *mc.pill("cta", 15, 510, 130, 44, "Shop Now", 15, fill=c(V["lime"]), color=c(V["dark"])),
    ], "160x600 skyscraper: wide logo, wide product, short copy, button.",
        ["skyscraper", "dark", "solid-bg", "groups", "offbrand", "synthetic-image"]))

    # o13 -- billboard
    A(ad("o13_v_billboard", "holdout", 970, 250, [
        bg(970, 250, "syn_bg_violet_glow.png"),
        img("product", "syn_prod_sneaker.png", 20, 45, w=320),
        mc.text("headline", "The Volt Runner", 370, 40, 380, 50, 36, bold=True, italic=True,
                color=c(V["white"])),
        mc.text("subhead", "400 pairs. Draw closes Thursday.", 370, 98, 380, 28, 19, color=c(V["white"], 0.85)),
        *label_pill("badge", 370, 150, "LIMITED", 13, fill=c(V["pink"]), color=c(V["dark"]), letter_spacing=1),
        *mc.pill("cta", 770, 95, 170, 60, "Enter Now", 20, fill=c(V["lime"]), color=c(V["dark"])),
    ], "970x250 billboard on a photo-like violet glow: product left, copy, badge, button.",
        ["billboard", "dark", "image-bg", "badges-1", "groups", "no-logo", "offbrand", "synthetic-image"]))

    # o14 -- 320x50 mobile banner, 4 elements
    A(ad("o14_v_mobile_banner", "train", 320, 50, [
        bg(320, 50, c(V["purple"])),
        mc.text("headline", "Drop 07 is live", 10, 12, 190, 26, 18, bold=True, italic=True, color=c(V["white"])),
        *mc.pill("cta", 214, 9, 96, 32, "Shop", 13, fill=c(V["lime"]), color=c(V["dark"])),
    ], "320x50 mobile banner: 4 elements, text and a button only.",
        ["strip", "mid-tone", "solid-bg", "text-only", "no-image", "groups", "offbrand"]))

    # o15 -- thumbnail, huge headline, rotated badge
    A(ad("o15_v_thumbnail", "holdout", 1280, 720, [
        bg(1280, 720, "syn_bg_violet_glow.png"),
        mc.ellipse("deco_disc", 690, 120, 520, 520, c(V["lime"]), 0.9, z=1),
        img("product", "syn_prod_sneaker.png", 650, 230, w=600),
        img("logo", VLOGO, 80, 60, w=300),
        mc.text("headline", "We tested it on 100 marathons", 80, 160, 560, 360, 84, bold=True, italic=True,
                color=c(V["white"])),
        mc.text("subhead", "Full review inside.", 80, 560, 520, 44, 30, color=c(V["white"], 0.85)),
        *label_pill("badge", 1030, 50, "NEW VIDEO", 22, fill=c(V["pink"]), color=c(V["dark"]), rotation=8, z=6),
    ], "1280x720 thumbnail: huge 3-line headline, product on a solid disc, rotated badge, no button.",
        ["landscape", "dark", "image-bg", "rotated", "badges-1", "no-cta", "deco-overlap", "groups", "offbrand",
         "synthetic-image"]))

    # o16 -- text only on lime
    col = Col(120, 0, 960, "center")
    col.text("headline", "Faster feels like this.", 92, 30, bold=True, italic=True, color=c(V["dark"]))
    col.text("subhead", "Voltstride Drop 07, live now.", 30, 40, color=c(V["dark"]))
    col.pill("cta", "Shop the Drop", 26, fill=c(V["dark"]), color=c(V["lime"]))
    col.center_in(0, 628)
    A(ad("o16_v_landscape_text_only", "train", 1200, 628, [bg(1200, 628, c(V["lime"])), *col.els],
         "Text-only landscape on a bright solid: no image, no logo, one button (5 elements).",
         ["landscape", "light", "solid-bg", "text-only", "no-image", "no-logo", "groups", "offbrand"]))
    return ads


def inkwell() -> list[dict]:
    ads = []
    A = ads.append

    # o17 -- portrait author evening: rotated book, 3 badges, legal (16+ elements)
    col = Col(90, 300, 560)
    col.text("kicker", "AUTHOR EVENING", 22, 14, bold=True, color=c(I["rust"]), letter_spacing=5)
    col.text("headline", "Mara Quill on The Quiet Hours", 64, 24, serif=True, bold=True, color=c(I["ink"]))
    col.text("subhead", "A reading, a conversation and a signing with the author of this year's surprise "
             "bestseller", 28, 30, serif=True, italic=True, color=c(I["ink"], 0.85))
    col.text("date", "Thursday 12 March, 7 pm", 32, 8, bold=True, color=c(I["ink"]))
    col.text("details", "Inkwell & Lantern, upstairs room. Doors at 6:30.", 24, 30, color=c(I["ink"], 0.8))
    col.badges([("badge1", "FREE", c(I["teal"]), c(I["white"])),
                ("badge2", "SIGNED COPIES", c(I["gold"]), c(I["ink"])),
                ("badge3", "Q&A", c(I["rust"]), c(I["white"]))], 16, 50)
    col.pill("cta", "Reserve a Seat", 26, fill=c(I["ink"]), color=c(I["paper"]))
    leg = mc.text("legal", "Seats are free but limited; reservations are held until 6:50 pm. Signed copies "
                  "while stocks last.", 90, 0, 900, 10, 16, color=c(I["ink"], 0.65))
    leg["height"] = _needed_height(leg, 16)
    leg["y"] = 1300 - leg["height"]
    A(ad("o17_i_portrait_author_night", "train", 1080, 1350, [
        bg(1080, 1350, c(I["paper"])),
        mc.ellipse("deco_sun", 620, 380, 560, 560, c(I["gold"]), 0.25),
        img("logo", ILOGO, 90, 50, h=200),
        img("product", "syn_prod_square_book.png", 690, 520, w=330, rotation=8),
        *col.els, leg,
    ], "Light portrait event poster: rotated book, italic serif subhead, date + details, three badges, button, "
       "legal (18 elements).", ["portrait", "light", "solid-bg", "rotated", "badges-3", "legal", "italic",
                                "groups", "18-elements", "offbrand", "synthetic-image"]))

    # o18 -- square book club, feature list grouped
    els = [bg(1080, 1080, "syn_bg_glacier.png"),
           img("product", "syn_prod_square_book.png", 600, 300, w=400),
           img("logo", ILOGO, 90, 60, h=150),
           t("headline", "Join the book club", 90, 250, 480, 62, serif=True, bold=True, color=c(I["ink"]))]
    feats = ["One hand-picked book a month", "A members-only discussion night", "10% off everything in store"]
    for i, f in enumerate(feats, 1):
        y = 440 + 100 * (i - 1)
        feat = t(f"feat{i}_text", f, 140, y, 420, 28, color=c(I["ink"]), group=f"feat{i}")
        els += [mc.shape(f"feat{i}_dot", 90, y + 10, 24, 24, c(I["rust"]), kind="ellipse", z=2, group=f"feat{i}"), feat]
    els += mc.pill("cta", 90, 790, 360, 88, "Become a Member", 26, fill=c(I["teal"]), color=c(I["white"]))
    els.append(mc.text("note", "From 9 a month. Cancel any time.", 90, 900, 500, 34, 22, color=c(I["ink"], 0.75)))
    A(ad("o18_i_square_book_club", "train", 1080, 1080, els,
         "Square on a light photo-like background: three grouped feature bullets, book product right.",
         ["square", "light", "image-bg", "list", "groups", "offbrand", "synthetic-image"]))

    # o19 -- story, BUSY terrazzo, paper panel holding logo + copy
    col = Col(150, 0, 780, "center")
    col.text("kicker", "24 HOURS, 24 READERS", 24, 14, bold=True, color=c(I["rust"]), letter_spacing=4)
    col.text("headline", "The Reading Marathon", 84, 24, serif=True, bold=True, color=c(I["ink"]))
    col.text("subhead", "Drop in any hour, stay as long as you like. Tea and toast all night.", 30, 40,
             color=c(I["ink"], 0.85))
    col.pill("cta", "See the Schedule", 30, 26, fill=c(I["ink"]), color=c(I["paper"]))
    col.text("legal", "Free entry. All proceeds from the café go to the local library fund.", 18,
             color=c(I["ink"], 0.7))
    col.center_in(1150, 1800)
    A(ad("o19_i_story_reading_marathon", "holdout", 1080, 1920, [
        bg(1080, 1920, "syn_busy_terrazzo.png"),
        mc.shape("panel", 90, 880, 900, 960, c(I["paper"]), radius=24, z=1),
        img("logo", ILOGO, 0, 920, h=200) | {"x": 390},
        img("product", "syn_prod_square_book.png", 290, 200, w=500),
        *col.els,
    ], "Story on a BUSY terrazzo background: logo, copy and legal on an opaque paper panel; the book on the "
       "busy area.", ["story", "busy-bg", "panel", "groups", "legal", "offbrand", "synthetic-image"]))

    # o20 -- landscape sale on teal
    A(ad("o20_i_landscape_two_for_one", "train", 1200, 628, [
        bg(1200, 628, "syn_bg_harbor_night.png"),
        mc.ellipse("deco_moon", 700, 40, 560, 560, c(I["white"]), 0.08),
        img("product", "syn_prod_square_book.png", 760, 120, w=380, rotation=-6),
        t("discount", "2 for 1", 80, 60, 600, 150, serif=True, bold=True, color=c(I["gold"])),
        t("subhead", "On every paperback in the shop, all weekend.", 80, 280, 560, 34, color=c(I["white"])),
        *mc.pill("cta", 80, 410, 300, 80, "Browse Paperbacks", 22, fill=c(I["gold"]), color=c(I["ink"])),
        mc.text("legal", "Cheapest of the two is free. In store only, Saturday and Sunday.", 80, 560, 640, 26, 16,
                color=c(I["white"], 0.7)),
    ], "Landscape sale on a dark photo-like background: giant serif discount, rotated book, translucent decoration, legal.",
        ["landscape", "dark", "image-bg", "rotated", "legal", "no-logo", "groups", "deco-overlap", "offbrand",
         "synthetic-image"]))

    # o21 -- wide banner, stacked logo left
    A(ad("o21_i_banner_events", "train", 1584, 396, [
        bg(1584, 396, c(I["paper"])),
        img("logo", ILOGO, 50, 78, h=240),
        mc.text("headline", "This month at the shop", 470, 96, 700, 80, 60, serif=True, bold=True, color=c(I["ink"])),
        mc.text("subhead", "Six readings, two workshops and a poetry night.", 470, 190, 700, 40, 28,
                color=c(I["ink"], 0.85)),
        *mc.pill("cta", 1250, 158, 280, 80, "See All Events", 24, fill=c(I["rust"]), color=c(I["white"])),
    ], "Wide banner: stacked 3:2 logo left, copy centre, button right.",
        ["banner", "light", "solid-bg", "groups", "offbrand", "synthetic-image"]))

    # o22 -- half page 300x600
    col = Col(20, 300, 260, "center")
    col.text("headline", "Staff picks for spring", 28, 10, serif=True, bold=True, color=c(I["paper"]))
    col.text("subhead", "Twelve books our booksellers cannot stop recommending.", 15, 20, color=c(I["paper"], 0.85))
    col.pill("cta", "See the List", 16, 16, fill=c(I["gold"]), color=c(I["ink"]))
    A(ad("o22_i_half_page_staff_picks", "train", 300, 600, [
        bg(300, 600, c(I["ink"])),
        img("product", "syn_prod_square_book.png", 60, 60, w=180),
        *col.els,
        mc.text("legal", "Inkwell & Lantern Books.", 20, 560, 260, 18, 12, color=c(I["paper"], 0.65), align="center"),
    ], "300x600 half page: book, centred serif copy, button, small legal.",
        ["skyscraper", "dark", "solid-bg", "groups", "legal", "no-logo", "offbrand", "synthetic-image"]))

    # o23 -- cover 820x312, no groups, no button
    A(ad("o23_i_cover_poetry", "train", 820, 312, [
        bg(820, 312, "syn_bg_fog.png"),
        mc.ellipse("deco_ring", 520, -60, 340, 340, c(I["gold"]), 0.3),
        img("product", "syn_prod_square_book.png", 560, 46, w=220),
        mc.text("headline", "Poetry night, every last Friday", 50, 70, 460, 116, 44, serif=True, bold=True,
                color=c(I["ink"])),
        mc.text("subhead", "Open mic from 8 pm. Bring a poem or just listen.", 50, 200, 440, 52, 19,
                color=c(I["ink"], 0.85)),
    ], "820x312 cover on a light photo-like fog: no groups, no button, translucent decoration.",
        ["cover", "light", "image-bg", "no-groups", "no-cta", "no-logo", "deco-overlap", "bleed", "offbrand",
         "synthetic-image"]))
    return ads


def pacewell() -> list[dict]:
    ads = []
    A = ads.append

    # o24 -- square streak: phone right, two badges
    col = Col(90, 0, 520)
    col.text("headline", "Keep the streak alive", 72, 28, bold=True, color=c(P["dark"]))
    col.text("subhead", "Ten minutes a day is all it takes. Pacewell plans the rest.", 28, 34,
             color=c(P["dark"], 0.85))
    col.badges([("badge1", "AUDIO COACHING", c(P["dark"]), c(P["cream"])),
                ("badge2", "NO GEAR", c(P["green"]), c(P["white"]))], 16, 40)
    col.pill("cta", "Start Free Week", 26, fill=c(P["orange"]), color=c(P["dark"]))
    col.center_in(220, 960)
    A(ad("o24_p_square_streak", "train", 1080, 1080, [
        bg(1080, 1080, "syn_bg_sunlit_cream.png"),
        mc.ellipse("deco_sun", 620, 220, 460, 460, c(P["yellow"]), 0.5),
        img("logo", PLOGO, 90, 70, h=110),
        img("product", "syn_prod_phone.png", 680, 240, h=640),
        *col.els,
    ], "Fitness app, light photo-like background: copy left, two badges, phone product right on a soft disc.",
        ["square", "light", "image-bg", "badges-2", "groups", "deco-overlap", "offbrand", "synthetic-image"]))

    # o25 -- story, BUSY mosaic, dark panel
    col = Col(150, 0, 780, "center")
    col.text("kicker", "30-DAY CHALLENGE", 26, 14, bold=True, color=c(P["yellow"]), letter_spacing=5)
    col.text("headline", "One month. One habit. Zero excuses.", 78, 30, bold=True, color=c(P["cream"]))
    col.pill("cta", "Join 40,000 Others", 30, 26, fill=c(P["orange"]), color=c(P["dark"]))
    col.text("legal", "Free for 7 days, then 6 a month. Cancel in the app any time.", 18, color=c(P["cream"], 0.7))
    col.center_in(1080, 1790)
    A(ad("o25_p_story_challenge", "train", 1080, 1920, [
        bg(1080, 1920, "syn_busy_mosaic.png"),
        mc.shape("panel", 90, 1020, 900, 830, c(P["dark"]), radius=48, z=1),
        img("product", "syn_prod_phone.png", 0, 120, h=820) | {"x": 335},
        *col.els,
    ], "Story on a BUSY mosaic background: copy on an opaque dark panel, phone on the busy area, no logo.",
        ["story", "busy-bg", "panel", "groups", "legal", "no-logo", "offbrand", "synthetic-image"]))

    # o26 -- portrait plans: three grouped price cards (19 elements)
    els = [bg(1080, 1350, "syn_bg_fog.png"),
           img("logo", PLOGO, 90, 60, h=100)]
    col = Col(90, 200, 900, "center")
    col.text("headline", "Pick the plan that fits your week", 64, 20, bold=True, color=c(P["dark"]))
    col.text("subhead", "Every plan includes audio coaching and offline workouts.", 28, color=c(P["dark"], 0.85))
    els += col.els
    plans = [("Monthly", "6", "per month", c(P["white"]), c(P["dark"])),
             ("Yearly", "48", "per year", c(P["dark"]), c(P["cream"])),
             ("Family", "80", "up to 5 people", c(P["white"]), c(P["dark"]))]
    for i, (name, price, per, fill, ink) in enumerate(plans, 1):
        x = 90 + 310 * (i - 1)
        g = f"plan{i}"
        els += [mc.shape(f"plan{i}_card", x, 520, 280, 360, fill, radius=28, z=1, group=g),
                mc.text(f"plan{i}_name", name.upper(), x, 560, 280, 34, 22, bold=True, color=ink, align="center",
                        letter_spacing=3, z=2, group=g),
                t(f"plan{i}_price", price, x, 615, 280, 96, bold=True, color=ink, align="center", z=2, group=g),
                mc.text(f"plan{i}_per", per, x, 770, 280, 34, 22, color=ink, align="center", z=2, group=g)]
    els += mc.pill("cta", 330, 960, 420, 96, "Start Free Week", 28, fill=c(P["orange"]), color=c(P["dark"]))
    leg = mc.text("legal", "Prices in local currency. Free week converts to the plan you pick unless you cancel "
                  "in the app before it ends. Family plan needs one account per person.", 90, 0, 900, 10, 16,
                  color=c(P["dark"], 0.7), align="center")
    leg["height"] = _needed_height(leg, 16)
    leg["y"] = 1300 - leg["height"]
    els.append(leg)
    A(ad("o26_p_portrait_plans", "holdout", 1080, 1350, els,
         "Light portrait pricing ad: three grouped price cards (card + name + price + period), button, long legal "
         "(19 elements).", ["portrait", "light", "image-bg", "groups", "cards", "legal", "19-elements",
                            "offbrand", "synthetic-image"]))

    # o27 -- 300x250, dark
    A(ad("o27_p_mrec_dark", "holdout", 300, 250, [
        bg(300, 250, c(P["dark"])),
        img("product", "syn_prod_phone.png", 196, 26, h=200),
        mc.text("headline", "Move ten minutes today", 16, 30, 170, 100, 26, bold=True, color=c(P["cream"])),
        *mc.pill("cta", 16, 150, 132, 38, "Try It Free", 14, fill=c(P["orange"]), color=c(P["dark"])),
        mc.text("legal", "7 days free, then 6/month.", 16, 214, 170, 16, 10, color=c(P["cream"], 0.7)),
    ], "Dark 300x250: headline, phone right, button, legal.",
        ["small", "dark", "solid-bg", "groups", "legal", "no-logo", "offbrand", "synthetic-image"]))

    # o28 -- 970x90 large leaderboard
    A(ad("o28_p_large_leaderboard", "train", 970, 90, [
        bg(970, 90, c(P["cream"])),
        img("logo", PLOGO, 16, 18, h=54),
        mc.text("headline", "Your coach, in your pocket", 160, 24, 420, 40, 28, bold=True, color=c(P["dark"])),
        *label_pill("badge", 600, 28, "4.8 STARS", 14, fill=c(P["green"]), color=c(P["white"]), letter_spacing=1),
        *mc.pill("cta", 790, 20, 164, 50, "Get the App", 17, fill=c(P["orange"]), color=c(P["dark"])),
    ], "970x90 strip: wide logo, headline, rating badge, button.",
        ["strip", "light", "solid-bg", "badges-1", "groups", "offbrand", "synthetic-image"]))

    # o29 -- 1920x1080 minimal, 5 elements
    col = Col(160, 0, 980)
    col.text("headline", "Sweat smarter.", 150, 30, bold=True, color=c(P["cream"]))
    col.text("subhead", "Workouts that adapt to how you slept.", 44, 60, color=c(P["cream"], 0.85))
    col.pill("cta", "Download Pacewell", 34, fill=c(P["yellow"]), color=c(P["dark"]))
    col.center_in(0, 1080)
    A(ad("o29_p_landscape_minimal", "train", 1920, 1080, [
        bg(1920, 1080, c(P["dark"])),
        img("product", "syn_prod_phone.png", 1300, 140, h=800),
        *col.els,
    ], "Minimal 1920x1080: huge headline, one line, button, phone (6 elements).",
        ["landscape", "dark", "solid-bg", "5-elements", "groups", "no-logo", "offbrand", "synthetic-image"]))

    # o30 -- text only on yellow, 4 elements
    col = Col(90, 0, 900, "center")
    col.text("headline", "10,000 steps", 140, 20, bold=True, color=c(P["dark"]))
    col.text("subhead", "is a fine start. Pacewell knows what comes next.", 40, 40, color=c(P["dark"]))
    col.text("tagline", "PACEWELL, THE PATIENT COACH", 22, bold=True, color=c(P["dark"], 0.8), letter_spacing=5)
    col.center_in(0, 1080)
    A(ad("o30_p_square_text_only", "train", 1080, 1080, [bg(1080, 1080, c(P["yellow"])), *col.els],
         "Text-only square on a bright yellow: 4 elements, no image, no groups.",
         ["square", "light", "solid-bg", "text-only", "no-image", "no-logo", "no-groups", "offbrand"]))
    return ads


def offbrand_ads() -> list[dict]:
    return quillbrook() + voltstride() + inkwell() + pacewell()


def main() -> None:
    records = offbrand_ads()
    keep, problems = [], 0
    with merged_assets_dir() as assets:
        for rec in records:
            issues = mc.check(rec, strict=True) + mc.pixel_check(rec, assets)
            for w in issues:
                print(f"[drop] {rec['task_id']}: {w}")
            problems += len(issues)
            if not issues:
                keep.append(rec)
    mc.sync_designs(keep, ("offbrand",))
    mc.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    mc.contact_sheet(keep, REVIEW_SHEET)
    splits = {s: sum(r["split"] == s for r in keep) for s in ("train", "holdout")}
    print(f"wrote {len(keep)} of {len(records)} off-brand ads ({splits['train']} train, {splits['holdout']} "
          f"holdout); {problems} problems; review sheet: {REVIEW_SHEET}")


if __name__ == "__main__":
    main()
