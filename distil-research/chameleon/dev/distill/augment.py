"""Write augmented variants of the drafted ads into the corpus.

    cd distil-research/chameleon && python -m dev.distill.augment

For each of the 30 drafted ads (origin "claude-drafted"; never the 3 official
samples) this makes VARIANTS_PER_AD variants, each combining 2-4 of the
operations below, chosen with a seeded RNG (seed = parent task id + slot +
attempt, so a rerun writes the same files):

    canvas    re-lay the ad onto a nearby source size (scale + spread the rows
              or columns over the new space)
    remove    drop an optional element (badges, a secondary line, the logo,
              legal, a decoration) and close the gap
    add       add a badge pill (grouped shape + label), a decoration or a legal line
    copy      shorter or longer headline/subhead/legal copy (COPY below), refit
              with the engine's TextMeasurer, restack what sits below
    image     swap the bottle / background / logo for a synthetic image of a
              different aspect ratio, refitting the box to the new aspect
    type      vary weight, family, italic, letter spacing or size of a text
    palette   recolour everything to one of PALETTES (optionally inverted
              light <-> dark), with a matching background image and logo
    jitter    nudge whole clusters by up to 2% of the canvas, keeping shared
              left edges / centres aligned

Every variant inherits its parent's split (a variant of a holdout ad stays
holdout) and records `parent` and `augmentations`. `brand` becomes "offbrand"
when the palette or the logo was swapped (the result no longer carries
Solstice's colours or mark); otherwise it stays "solstice".

Quality gate: after every operation the ad must pass make_corpus.check
(strict); the finished variant must also pass make_corpus.pixel_check
(contrast floor, no text on a busy backdrop). A failing attempt is dropped
with its reason logged, and the slot is retried with the next seed.
"""

import copy as _copy
import hashlib
import json
import math
import random
import re

from dev.distill import make_corpus as mc
from dev.distill.assets import merged_assets_dir

VARIANTS_PER_AD = 3
MAX_ATTEMPTS = 10
REVIEW_SHEET = mc.REVIEW_DIR / "sheet_augmented.png"
DROP_LOG = mc.REVIEW_DIR / "augment_dropped.txt"

# ---------------------------------------------------------------------------
# Data: palettes, images, alternative copy
# ---------------------------------------------------------------------------

# Solstice colour -> palette role. Alpha is kept from the original colour.
ROLE_OF = {(43, 27, 46): "dark", (250, 242, 230): "light", (253, 232, 214): "tint", (232, 238, 224): "tint",
           (255, 255, 255): "white", (232, 106, 58): "accent", (240, 178, 66): "accent2",
           (122, 154, 110): "accent3", (220, 60, 50): "alert"}
INVERT_ROLE = {"dark": "white", "light": "dark", "tint": "dark", "white": "dark"}

# Invented brand palettes: dark-on-light and light-on-dark, high and low contrast.
PALETTES = {
    "harbor": dict(name="Harbor", dark="#13233F", light="#F3EEE4", tint="#E3EDF2", white="#FFFFFF",
                   accent="#D9502A", accent2="#F3B23C", accent3="#3F8E7E", alert="#C0392B",
                   bg_dark="syn_bg_harbor_night.png", bg_light="syn_bg_glacier.png", contrast="high"),
    "grove": dict(name="Grove", dark="#1E3A2B", light="#EEF1E2", tint="#E1EAD3", white="#F8FAF0",
                  accent="#B8492F", accent2="#E6C35C", accent3="#4F7F55", alert="#A63D2F",
                  bg_dark="syn_bg_forest_mist.png", bg_light="syn_bg_sunlit_cream.png", contrast="high"),
    "ember": dict(name="Ember", dark="#3A2619", light="#EBDCC6", tint="#F2E6D6", white="#F6EDE0",
                  accent="#A5531F", accent2="#E2A55A", accent3="#6F7F52", alert="#9E2A2B",
                  bg_dark="syn_bg_roast_ember.png", bg_light="syn_bg_peach_haze.png", contrast="low"),
    "ultra": dict(name="Ultra", dark="#170C2B", light="#F4EFFF", tint="#EAE2FA", white="#FFFFFF",
                  accent="#7B2FF7", accent2="#C6F432", accent3="#D6337F", alert="#E5322B",
                  bg_dark="syn_bg_violet_glow.png", bg_light="syn_bg_fog.png", contrast="high"),
    "bloom": dict(name="Bloom", dark="#4A1F3A", light="#FBE9EE", tint="#F8DDE4", white="#FFFFFF",
                  accent="#B23A60", accent2="#F2B8C6", accent3="#5F7F6E", alert="#B3261E",
                  bg_dark="syn_bg_violet_glow.png", bg_light="syn_bg_peach_haze.png", contrast="low"),
    "glacier": dict(name="Glacier", dark="#1D2B3A", light="#EAF3FA", tint="#DCEBF5", white="#FFFFFF",
                    accent="#2B67C6", accent2="#8CC8EC", accent3="#2F8483", alert="#C93C3C",
                    bg_dark="syn_bg_harbor_night.png", bg_light="syn_bg_glacier.png", contrast="high"),
    "citrine": dict(name="Citrine", dark="#262626", light="#FFF7D6", tint="#FFF0B8", white="#FFFFFF",
                    accent="#C85A00", accent2="#FFD23F", accent3="#25804A", alert="#C81E34",
                    bg_dark="syn_bg_roast_ember.png", bg_light="syn_bg_sunlit_cream.png", contrast="high"),
    "slate": dict(name="Slate", dark="#39404D", light="#E5E9F0", tint="#D8DEE9", white="#ECEFF4",
                  accent="#4F6F96", accent2="#EBCB8B", accent3="#8FBCBB", alert="#A84F58",
                  bg_dark="syn_bg_harbor_night.png", bg_light="syn_bg_fog.png", contrast="low"),
}

DARK_BGS = ["syn_bg_harbor_night.png", "syn_bg_forest_mist.png", "syn_bg_roast_ember.png", "syn_bg_violet_glow.png"]
# Product swaps never keep the bottle's 1:1 aspect.
PRODUCTS = ["syn_prod_tall_bottle.png", "syn_prod_wide_box.png", "syn_prod_sneaker.png", "syn_prod_phone.png",
            "syn_prod_coffee_bag.png"]
LIGHT_LOGOS = ["syn_logo_blocks_light.png", "syn_logo_quillbrook_badge.png"]
LOGO_ON_LIGHT = "syn_logo_blocks_dark.png"
LOGO_ON_DARK = "syn_logo_blocks_light.png"

BADGE_LABELS = ["NEW", "LIMITED", "BEST SELLER", "SUMMER ONLY", "ONLINE ONLY", "SMALL BATCH", "FREE DELIVERY"]
LEGAL_LINES = ["Please enjoy responsibly. Solstice Botanical Sparkling Co., 2026.",
               "While stocks last. Prices and availability may vary by region. Solstice Botanical Sparkling Co.",
               "Offer valid online only, for a limited time. See solstice.example/terms for details."]

# Alternative copy per drafted ad: element id -> (short, long). "[...]" marks the
# part that keeps the original's highlight colour (a style run). None = no alternative.
COPY = {
    "c01_square_golden_hour": {
        "headline": ("Pure [gold.]", "Golden hour, slowly steeped and [bottled.]"),
        "subhead": ("Blood orange, wild honey, rosemary.",
                    "Blood orange, wild honey and a sprig of rosemary, steeped overnight and lifted with fine, "
                    "lasting bubbles.")},
    "c02_square_garden_fizz": {
        "headline": ("Fresh fizz", "Garden-fresh fizz, all afternoon"),
        "subhead": ("Cucumber, mint, elderflower.",
                    "Cucumber, mint and elderflower, picked and pressed the same morning, sparkling and lightly "
                    "sweet.")},
    "c03_square_text_only": {
        "headline": ("Sip at [sunset.]", "Sparkle slowly, linger longer. [Sip at sunset.]"),
        "tagline": ("SOLSTICE SPARKLING", None)},
    "c04_square_five_badges": {
        "headline": ("Five botanicals.", "Five botanicals, one glow, no added sugar."),
        "legal": ("Flavours vary by region.", "Flavours vary by region and season. Mixed packs ship in recyclable "
                  "cartons. Solstice Botanical Sparkling Co., 2026.")},
    "c05_square_orange_sticker": {
        "headline": ("Sunset", "Blood Orange Sunset is back"),
        "subhead": ("Bright and golden.", None)},
    "c06_portrait_lineup": {
        "headline": ("Pick your light", "Three ways to chase the light, all summer long"),
        "subhead": ("Mix a pack of all three.", "Mix a pack of all three flavours, delivered cold to your door "
                    "every month, and skip or pause any time.")},
    "c07_portrait_long_copy": {
        "headline": ("Our Boldest Small-Batch Blend Yet", None),
        "body": ("Filled by hand and released once a summer.", None),
        "legal": ("Please enjoy responsibly. Contains no alcohol. Pre-orders ship within the mainland only.", None)},
    "c08_portrait_sunset_sale": {
        "subhead": ("This weekend only.", "Every mixed case, this weekend only, with free chilled delivery."),
        "legal": ("Offer ends Sunday at midnight.", None)},
    "c09_story_launch": {
        "headline": ("Honey Rosemary", "Meet Honey Rosemary, our newest botanical"),
        "subhead": ("Warm, herbal, softly sparkling.", None)},
    "c10_story_ritual_steps": {
        "headline": ("The ritual", "The golden hour ritual, in three steps"),
        "intro": ("Three steps to the perfect pour.", None)},
    "c11_story_longest_day": {
        "headline": ("Make it last", "Make the longest day of the year last longer"),
        "subhead": ("25% off on June 21.", None)},
    "c12_landscape_split": {
        "headline": ("Slow summer nights", None),
        "subhead": ("Small-batch and lightly sweet.", "Small-batch, lightly sweet, made with real fruit, garden "
                    "herbs and nothing artificial.")},
    "c13_landscape_cucumber_mint": {
        "headline": ("It's back", "Cucumber Mint is back for summer"),
        "subhead": ("Cool, crisp and barely sweet.", None)},
    "c14_thumbnail_taste_test": {
        "headline": ("Taste test", "Blind taste test: three new botanicals"),
        "subhead": ("Which one wins?", None)},
    "c15_landscape_text_only": {
        "headline": ("A little sunshine.", "Bubbles, botanicals and a little extra sunshine."),
        "subhead": ("Now in four flavours.", None)},
    "c16_banner_bottled_glow": {
        "headline": ("Golden hour", None),
        "subhead": ("Small-batch botanicals.", None)},
    "c17_banner_summer_flavours": {
        "headline": ("Summer is here", "All four summer flavours are here"),
        "subhead": ("Four drinks, one season.", None)},
    "c18_mrec_chase_light": {
        "headline": ("Glow", None),
        "subhead": ("Botanicals", "Sparkling botanicals, small-batch")},
    "c19_mrec_weekend_sale": {
        "subhead": ("Every mixed case.", None),
        "legal": ("Ends Sunday. Online only.", None)},
    "c20_leaderboard_dark": {
        "headline": ("Glow on", None),
        "subhead": ("Botanical sparkling.", None)},
    "c21_leaderboard_new_flavor": {
        "headline": ("Sunset", None),
        "subhead": ("Out now.", None)},
    "c22_skyscraper": {
        "headline": ("Glow", None),
        "subhead": ("Golden-hour botanicals.", "Sparkling botanicals, crafted in small batches for golden hour.")},
    "c23_billboard_weekend_lineup": {
        "headline": ("Weekend lineup", None),
        "subhead": ("Chilled and delivered by Friday.", None),
        "legal": ("Delivery to selected areas only.", None)},
    "c24_cover_all_summer": {
        "headline": ("Chase the light", None),
        "subhead": ("Botanical sparkling drinks.", None)},
    "c25_square_missing_sticker": {
        "headline": ("Summer, sparkling", "Your whole summer, sparkling"),
        "legal": ("Please enjoy responsibly.", None)},
    "c26_portrait_sunset_sessions": {
        "subhead": ("Live music and rooftop pours", None),
        "details": ("6 pm, Harbour Rooftop", None)},
    "c27_square_tasting_note": {
        "quote": ("Sunset in a glass.", "Like a sunset in a glass: bright, a little bitter, golden, and gone "
                  "too soon.")},
    "c28_square_panel": {
        "headline": ("Sip the [glow]", None),
        "subhead": ("Four botanical sparkling drinks.", None)},
    "c29_portrait_feature_list": {
        "headline": ("Tastes like summer", "Why every bottle tastes like summer"),
        "body": ("Find Solstice in the chilled aisle.", "Find Solstice in the chilled aisle of your favourite local "
                 "grocer, or order a mixed case online for delivery.")},
    "c30_story_minimal_glow": {
        "headline": (None, "Glow on.")},
}

# Nearby source sizes each canvas may be re-laid onto. Big jumps (story ->
# banner) are left out: proportional scaling + restacking can't make them look
# designed, so they would only produce broken sources.
CANVAS_MOVES = {
    (1080, 1080): [(1080, 1350), (1200, 1200), (1080, 1200)],
    (1080, 1350): [(1080, 1080), (1080, 1920), (1200, 1500)],
    (1080, 1920): [(1080, 1350), (1080, 1620)],
    (1200, 628): [(1280, 720), (1200, 675)],
    (1280, 720): [(1200, 628), (1920, 1080)],
    (1584, 396): [(1500, 500), (1400, 400)],
    (300, 250): [(336, 280), (300, 300)],
    (728, 90): [(970, 90), (728, 120)],
    (160, 600): [(300, 600), (120, 600)],
    (970, 250): [(970, 300), (1200, 300)],
    (820, 312): [(851, 315), (1000, 380)],
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_RGBA = re.compile(r"rgba\((\d+),(\d+),(\d+),([\d.]+)\)")


def _parse(color: str) -> tuple[tuple[int, int, int], float]:
    m = _RGBA.fullmatch(color.replace(" ", ""))
    return (int(m[1]), int(m[2]), int(m[3])), float(m[4])


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    alpha = round(alpha, 3)
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha:g})"


def _rec(src: dict, task_id: str = "probe", parent: str | None = None) -> dict:
    """A minimal record so make_corpus.check can run on a source."""
    return {"task_id": task_id, "parent": parent, "expects_missing_asset": bool(mc._missing_assets(src)),
            "source": src}


def _ok(src: dict, parent: str) -> bool:
    return not mc.check(_rec(src, parent=parent), strict=True)


def _content(src: dict) -> list[dict]:
    return [e for e in src["elements"] if not mc._is_layer(e)]


def _bbox(els: list[dict]) -> tuple[float, float, float, float]:
    boxes = [mc._box(e) for e in els]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return x0, y0, max(b[0] + b[2] for b in boxes) - x0, max(b[1] + b[3] for b in boxes) - y0


def _clusters(src: dict, containment: bool = False) -> list[list[dict]]:
    """Elements that must move together: a group, a shape + label pair
    (`x_bg`/`x_label`), and with `containment` anything inside a content shape."""
    content = _content(src)
    parent = {e["id"]: e["id"] for e in content}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    by_group: dict[str, str] = {}
    for e in content:
        g = e.get("group_id")
        if g:
            if g in by_group:
                union(e["id"], by_group[g])
            else:
                by_group[g] = e["id"]
        if e["id"].endswith("_bg") and e["id"][:-3] + "_label" in parent:
            union(e["id"], e["id"][:-3] + "_label")
    if containment:
        for s in content:
            if s["type"] == "shape":
                for e in content:
                    if e is not s and mc._contains(mc._box(s), mc._box(e)):
                        union(e["id"], s["id"])
    out: dict[str, list[dict]] = {}
    for e in content:
        out.setdefault(find(e["id"]), []).append(e)
    return list(out.values())


def _move(els: list[dict], dx: float, dy: float) -> None:
    for e in els:
        e["x"] = round(e["x"] + dx, 1)
        e["y"] = round(e["y"] + dy, 1)


def _needed_height(el: dict, size: float) -> float:
    """Height the box needs at `size`: the larger of the engine's measure and
    make_corpus.check's own estimate, plus a pixel of slack."""
    tm = mc.measurer()
    lines = tm.line_count(el, size, el["width"])
    estimate = (lines - 1) * el.get("line_height", 1.2) * size + 1.3 * size
    return math.ceil(max(tm.block_height(el, size, el["width"]), estimate)) + 1


def _bottom_anchored(cluster: list[dict], height: float) -> bool:
    _, y, _, h = _bbox(cluster)
    return any(e["id"] == "legal" for e in cluster) or y + h > height * 0.95


def _shift_below(src: dict, col: tuple[float, float], below_y: float, dy: float, skip: list[dict]) -> None:
    """Move every cluster that starts at or below `below_y` and overlaps the
    column [x0, x1] by dy (bottom-anchored clusters such as legal stay)."""
    height = src["canvas"]["height"]
    for cl in _clusters(src):
        if any(e in skip for e in cl):
            continue
        x, y, w, _ = _bbox(cl)
        if y >= below_y - 2 and x < col[1] and col[0] < x + w and not _bottom_anchored(cl, height):
            _move(cl, 0, dy)


def _set_runs(el: dict, marked: str) -> None:
    """Set content from text with an optional [highlight] marker."""
    color = el["style_runs"][0]["color"] if el.get("style_runs") else None
    m = re.search(r"\[(.+?)\]", marked)
    plain = marked.replace("[", "").replace("]", "")
    el["content"] = plain
    el.pop("style_runs", None)
    if color and m:
        start = m.start()
        el["style_runs"] = [{"start": start, "end": start + len(m[1]), "color": color}]


def _replace_words(el: dict, pairs: list[tuple[str, str]]) -> None:
    """Replace words in a text's content, carrying its style runs along."""
    old = el["content"]
    new = old
    for a, b in pairs:
        new = new.replace(a, b)
    if new == old:
        return
    runs = []
    for r in el.get("style_runs", []):
        part = old[r["start"]:r["end"]]
        for a, b in pairs:
            part = part.replace(a, b)
        at = new.find(part)
        if at >= 0 and part:
            runs.append(dict(r, start=at, end=at + len(part)))
    el["content"] = new
    if runs:
        el["style_runs"] = runs
    else:
        el.pop("style_runs", None)


def fit_text(src: dict, el: dict, parent: str, min_scale: float = 0.75) -> bool:
    """Make `el` fit after its content or style changed: set its height to what
    the text needs and restack the column below it; if that breaks the layout,
    shrink the font (down to min_scale) and try again. Leaves `src` unchanged
    and returns False when nothing works."""
    base = _copy.deepcopy(src)
    orig_size = el["font_size"]
    old_bottom = el["y"] + el["height"]
    size = orig_size
    while size >= max(9.0, orig_size * min_scale):
        tm = mc.measurer()
        if max(tm.line_widths(el, size, el["width"]), default=0) <= el["width"] + 0.5:
            trial = _copy.deepcopy(src)
            t = next(e for e in trial["elements"] if e["id"] == el["id"])
            t["font_size"] = size
            need = _needed_height(t, size)
            delta = need - t["height"]
            t["height"] = need
            _shift_below(trial, (t["x"], t["x"] + t["width"]), old_bottom, delta, [t])
            if _ok(trial, parent):
                src.clear()
                src.update(trial)
                return True
        size -= 0.5 if size <= 30 else 1.0
    src.clear()
    src.update(base)
    return False


def _visible_frac(asset: str) -> tuple[float, float, float, float]:
    """Visible (non-transparent) part of an image as fractions of its size."""
    from PIL import Image

    from dev.distill.assets import BRAND_ASSETS_DIR, SYNTH_ASSETS_DIR
    path = next(p / asset for p in (BRAND_ASSETS_DIR, SYNTH_ASSETS_DIR) if (p / asset).is_file())
    with Image.open(path) as im:
        bbox = im.getchannel("A").getbbox() if im.mode == "RGBA" else None
        w, h = im.size
    if not bbox:
        return 0.0, 0.0, 1.0, 1.0
    return bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h


def _swap_image(el: dict, asset: str, anchor: str = "center") -> None:
    """Point `el` at `asset` and refit its box to the new image's aspect
    inside the old box (cover images keep their box). With anchor 'left' the
    new image's visible left edge lands where the old one's was."""
    old_asset = el["asset_id"]
    el["asset_id"] = asset
    if el.get("fit") == "cover":
        return
    aspect = mc._image_aspect(asset)
    x, y, w, h = el["x"], el["y"], el["width"], el["height"]
    if anchor == "left":
        old_aspect = mc._image_aspect(old_asset) or w / h
        fx0 = _visible_frac(old_asset)[0]
        shown_w = min(w, h * old_aspect)  # contain: the old image's drawn width
        vis_left = x + (w - shown_w) / 2 + fx0 * shown_w
        nfx0 = _visible_frac(asset)[0]
        nw = min(w, h * aspect, x + w - vis_left)
        nh = nw / aspect
        el.update(x=round(vis_left - nfx0 * nw, 1), y=round(y + (h - nh) / 2, 1), width=round(nw, 1),
                  height=round(nh, 1))
        return
    nw, nh = (w, w / aspect) if w / h < aspect else (h * aspect, h)
    el.update(x=round(x + (w - nw) / 2, 1), y=round(y + (h - nh) / 2, 1), width=round(nw, 1), height=round(nh, 1))
    # Keep the ratio exact after rounding.
    el["height"] = round(el["width"] / aspect, 2)


def _tone(record: dict) -> str:
    feats = record.get("features", [])
    return "light" if "light" in feats else "mid" if "mid-tone" in feats else "dark"


# ---------------------------------------------------------------------------
# Operations. Each takes (src, rng, ctx) and returns a label or None (not applicable / failed).
# ctx: {"parent": task_id, "record": parent record, "brand_changed": bool}
# ---------------------------------------------------------------------------

def op_canvas(src, rng, ctx):
    cw, ch = src["canvas"]["width"], src["canvas"]["height"]
    moves = CANVAS_MOVES.get((cw, ch))
    if not moves:
        return None
    content = _content(src)
    for e in content:  # a partial background (touches two canvas edges) can't be re-laid this simply
        x, y, w, h = mc._box(e)
        edges = (x <= 1) + (y <= 1) + (x + w >= cw - 1) + (y + h >= ch - 1)
        if edges >= 2 and e["type"] != "text":
            return None
    W, H = rng.choice(moves)
    s = min(W / cw, H / ch)
    trial = _copy.deepcopy(src)
    for e in trial["elements"]:
        if e["id"] == "bg":
            e.update(x=0, y=0, width=W, height=H)
            continue
        if mc._is_layer(e):
            e.update(x=round(e["x"] * W / cw, 1), y=round(e["y"] * H / ch, 1), width=round(e["width"] * s, 1),
                     height=round(e["height"] * s, 1))
            continue
        e.update(x=e["x"] * s, y=e["y"] * s, width=e["width"] * s, height=e["height"] * s)
        if e["type"] == "text":
            e["font_size"] = max(9.0, round(e["font_size"] * s * 2) / 2)
            if e.get("letter_spacing"):
                e["letter_spacing"] = round(e["letter_spacing"] * s, 1)
        if e.get("corner_radius"):
            e["corner_radius"] = round(e["corner_radius"] * s, 1)
        if e.get("stroke_width"):
            e["stroke_width"] = round(e["stroke_width"] * s, 1)
    trial["canvas"] = {"width": W, "height": H}
    # Spread the extra space over the rows (taller) or columns (wider).
    clusters = _clusters(trial, containment=True)
    vertical = H / ch > W / cw + 1e-9
    extra = (H - ch * s) if vertical else (W - cw * s)
    axis = 1 if vertical else 0
    spans = sorted(((_bbox(c)[axis], _bbox(c)[axis] + _bbox(c)[axis + 2], c) for c in clusters), key=lambda t: t[0])
    bands: list[list] = []
    for lo, hi, c in spans:  # merge clusters that overlap along the axis into bands
        if bands and lo < bands[-1][1] - 1:
            bands[-1][1] = max(bands[-1][1], hi)
            bands[-1][2].append(c)
        else:
            bands.append([lo, hi, [c]])
    n = len(bands)
    for k, (_, _, cs) in enumerate(bands):
        shift = extra * (k + 1) / (n + 1) if n > 1 else extra / 2
        for c in cs:
            _move(c, 0 if vertical else shift, shift if vertical else 0)
    for e in trial["elements"]:
        for key in ("x", "y", "width", "height"):
            e[key] = round(e[key], 1)
        if e["type"] == "image" and e.get("fit") != "cover" and not mc._is_layer(e):
            aspect = mc._image_aspect(e["asset_id"])
            if aspect:
                e["height"] = round(e["width"] / aspect, 2)
    for e in trial["elements"]:  # scaled type can wrap differently: shrink until it fits again
        if e["type"] == "text":
            tm = mc.measurer()
            size = e["font_size"]
            while size > 9 and not tm.fits(e, size, e["width"], e["height"] - 1):
                size -= 0.5
            if size != e["font_size"]:
                e["font_size"] = size
    if not _ok(trial, ctx["parent"]):
        return None
    src.clear()
    src.update(trial)
    return f"canvas:{cw}x{ch}->{W}x{H}"


OPTIONAL_TEXT = ["subhead", "kicker", "note", "delivery", "ships", "swipe", "tagline", "intro", "rsvp",
                 "attribution", "flavor_1_desc"]


def op_remove(src, rng, ctx):
    els = src["elements"]
    ids = {e["id"] for e in els}
    options = []
    if any(i.startswith("badge") for i in ids):
        options.append("badges")
    options += [i for i in OPTIONAL_TEXT if i in ids]
    for i in ("legal", "logo", "wordmark"):
        if i in ids:
            options.append(i)
    options += [i for i in sorted(ids) if i.startswith("deco")]
    rng.shuffle(options)
    for choice in options:
        trial = _copy.deepcopy(src)
        if choice == "badges":
            gone = [e for e in trial["elements"] if e["id"].startswith("badge")]
        elif choice == "logo":
            gone = [e for e in trial["elements"] if e["id"] == "logo" or e.get("group_id") == "logo_lockup"]
        else:
            gone = [e for e in trial["elements"] if e["id"] == choice]
        if len(trial["elements"]) - len(gone) < 3:
            continue
        x, y, w, h = _bbox(gone)
        trial["elements"] = [e for e in trial["elements"] if e not in gone]
        if not all(e["type"] == "shape" for e in gone) and choice not in ("legal", "logo", "wordmark"):
            below = [c for c in _clusters(trial) if _bbox(c)[1] >= y + h - 2
                     and _bbox(c)[0] < x + w and x < _bbox(c)[0] + _bbox(c)[2]
                     and not _bottom_anchored(c, trial["canvas"]["height"])]
            if below:
                next_top = min(_bbox(c)[1] for c in below)
                for c in below:
                    _move(c, 0, y - next_top)
        if not any(e["type"] == "text" for e in trial["elements"]) or not _ok(trial, ctx["parent"]):
            continue
        if choice == "logo":
            ctx["logo_removed"] = True
        src.clear()
        src.update(trial)
        return f"remove:{choice}"
    return None


def _text_width(content: str, size: float, bold=True, serif=False, letter_spacing=0) -> float:
    probe = mc.text("probe", content, 0, 0, 10_000, 1000, size, bold=bold, serif=serif,
                    letter_spacing=letter_spacing or None)
    return max(mc.measurer().line_widths(probe, size, 10_000))


def _free(src: dict, box: tuple[float, float, float, float], pad: float) -> bool:
    W, H = src["canvas"]["width"], src["canvas"]["height"]
    x, y, w, h = box
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return False
    grown = (x - pad, y - pad, w + 2 * pad, h + 2 * pad)
    return not any(mc._intersect(grown, mc._box(e)) for e in _content(src))


def op_add(src, rng, ctx):
    W, H = src["canvas"]["width"], src["canvas"]["height"]
    if len(src["elements"]) > 18:
        return None
    ids = {e["id"] for e in src["elements"]}
    dark = ctx["tone"] != "light"
    kinds = ["badge", "deco"] + ([] if "legal" in ids or len(src["elements"]) > 19 else ["legal"])
    rng.shuffle(kinds)
    m = round(min(W, H) * 0.04)
    for kind in kinds:
        trial = _copy.deepcopy(src)
        if kind == "badge":
            label = rng.choice(BADGE_LABELS)
            fs = max(10, min(24, round(min(W, H) * 0.022)))
            h = round(fs * 2.3)
            w = round(_text_width(label, fs, letter_spacing=1) + 2.6 * fs)
            # Corner stickers only where there is room for one; strips get a badge beside the button.
            spots = [(W - m - w, m), (m, m), (W - m - w, H - m - h), (m, H - m - h)] if min(W, H) >= 400 else []
            cta = next((e for e in trial["elements"] if e["id"] == "cta_bg"), None)
            if cta:
                gap = max(m / 2, fs)
                spots += [(cta["x"] + cta["width"] + gap, cta["y"] + (cta["height"] - h) / 2),
                          (cta["x"] - gap - w, cta["y"] + (cta["height"] - h) / 2)]
            rng.shuffle(spots)
            for bx, by in spots:
                rot = rng.choice([0, 0, -8, 8])
                fill, color = (mc.GOLD, mc.PLUM) if rng.random() < 0.5 else (mc.ORANGE, mc.WHITE)
                if ctx["tone"] == "mid":
                    fill, color = mc.PLUM, mc.CREAM
                pill = mc.pill("badge_new", round(bx), round(by), w, h, label, fs, fill=fill, color=color, z=6,
                               letter_spacing=1, rotation=rot)
                if not _free(trial, mc._box(pill[0]), m / 3):
                    continue
                t2 = _copy.deepcopy(trial)
                t2["elements"] += pill
                if _ok(t2, ctx["parent"]):
                    src.clear()
                    src.update(t2)
                    return f"add:badge({label})"
        elif kind == "deco":
            d = round(min(W, H) * rng.uniform(0.35, 0.6))
            cx, cy = rng.choice([(0, 0), (W, 0), (0, H), (W, H), (W, H / 2), (0, H / 2)])
            color, alpha = (mc.WHITE, 0.08) if dark else (rng.choice([mc.GOLD, mc.ORANGE, mc.SAGE]), 0.22)
            trial["elements"].insert(1, mc.ellipse("deco_added", round(cx - d / 2 + rng.uniform(-0.1, 0.1) * d),
                                                   round(cy - d / 2 + rng.uniform(-0.1, 0.1) * d), d, d, color, alpha))
            if _ok(trial, ctx["parent"]):
                src.clear()
                src.update(trial)
                return "add:deco"
        else:
            fs = max(9, min(16, round(min(W, H) * 0.013)))
            head = next((e for e in trial["elements"] if e["id"] == "headline"), None)
            align = head.get("text_align", "left") if head else "center"
            x = head["x"] if head and align == "left" else m
            color = mc.rgba(mc.WHITE, 0.6) if dark else mc.rgba(mc.PLUM, 0.65)
            if ctx["tone"] == "mid":
                color = mc.rgba(mc.PLUM, 0.85)
            el = mc.text("legal", rng.choice(LEGAL_LINES), x, 0, W - x - m, 10, fs, color=color, align=align,
                         z=4)
            el["height"] = _needed_height(el, fs)
            el["y"] = H - round(H * 0.03) - el["height"]
            lowest = max(mc._box(e)[1] + mc._box(e)[3] for e in _content(trial))
            if el["y"] < lowest + fs:
                continue
            trial["elements"].append(el)
            if _ok(trial, ctx["parent"]):
                src.clear()
                src.update(trial)
                return "add:legal"
    return None


def op_copy(src, rng, ctx):
    alts = COPY.get(ctx["parent"], {})
    els = {e["id"]: e for e in src["elements"]}
    options = [(i, k) for i, pair in alts.items() if i in els for k, v in zip(("short", "long"), pair) if v]
    rng.shuffle(options)
    done, used = [], set()
    want = rng.choice([1, 1, 2])
    for eid, kind in options:
        if eid in used or len(done) >= want:
            continue
        trial = _copy.deepcopy(src)
        el = next(e for e in trial["elements"] if e["id"] == eid)
        _set_runs(el, alts[eid][0 if kind == "short" else 1])
        if fit_text(trial, el, ctx["parent"]):
            src.clear()
            src.update(trial)
            done.append(f"{eid}={kind}")
            used.add(eid)
    return f"copy:{','.join(done)}" if done else None


def op_image(src, rng, ctx):
    els = src["elements"]
    options = []
    if any(e["type"] == "image" and e["asset_id"] == mc.BOTTLE for e in els):
        options.append("product")
    if any(e["type"] == "image" and e["asset_id"] == mc.GRADIENT for e in els):
        options.append("background")
    if any(e["type"] == "image" and e["asset_id"] == mc.LOGO for e in els):
        options.append("logo")
    if not options:
        return None
    rng.shuffle(options)
    picks = options[:rng.choice([1, 2])]
    trial = _copy.deepcopy(src)
    done = []
    for what in picks:
        if what == "product":
            new = rng.choice(PRODUCTS)
            for e in trial["elements"]:
                if e["type"] == "image" and e["asset_id"] == mc.BOTTLE:
                    _swap_image(e, new)
        elif what == "background":
            new = rng.choice(DARK_BGS)
            for e in trial["elements"]:
                if e["type"] == "image" and e["asset_id"] == mc.GRADIENT:
                    _swap_image(e, new)
        else:
            new = rng.choice(LIGHT_LOGOS)
            for e in trial["elements"]:
                if e["type"] == "image" and e["asset_id"] == mc.LOGO:
                    W = src["canvas"]["width"]
                    centred = abs(e["x"] + e["width"] / 2 - W / 2) < W * 0.02
                    _swap_image(e, new, "center" if centred else "left")
            ctx["brand_changed"] = True
        done.append(f"{what}->{new}")
    if not _ok(trial, ctx["parent"]):
        return None
    src.clear()
    src.update(trial)
    return "image:" + ",".join(done)


def op_type(src, rng, ctx):
    texts = {e["id"]: e for e in src["elements"] if e["type"] == "text"}
    options = []
    if "headline" in texts:
        options += ["headline-family", "headline-weight", "headline-size"]
    if any(i in texts for i in ("subhead", "intro", "body")):
        options.append("subhead-italic")
    caps = [i for i, e in texts.items() if e["content"].isupper() and len(e["content"]) <= 40
            and not i.endswith("_label") and e["font_size"] <= 40]  # tracking suits small caps, not display type
    if caps:
        options.append("caps-spacing")
    rng.shuffle(options)
    for choice in options:
        trial = _copy.deepcopy(src)
        tx = {e["id"]: e for e in trial["elements"] if e["type"] == "text"}
        if choice == "headline-family":
            el = tx["headline"]
            el["font_family"] = "Solstice Sans" if el["font_family"] == "Solstice Serif" else "Solstice Serif"
            if el["font_family"] == "Solstice Serif" and el.get("font_style") == "italic":
                el.pop("font_weight", None)  # the kit has no Serif Bold Italic
            label = f"type:headline-{'sans' if 'Sans' in el['font_family'] else 'serif'}"
        elif choice == "headline-weight":
            el = tx["headline"]
            if el.get("font_weight") == "bold":
                el.pop("font_weight")
                label = "type:headline-regular"
            else:
                if el["font_family"] == "Solstice Serif" and el.get("font_style") == "italic":
                    continue
                el["font_weight"] = "bold"
                label = "type:headline-bold"
        elif choice == "headline-size":
            el = tx["headline"]
            f = rng.choice([0.85, 0.9, 1.1, 1.15])
            el["font_size"] = round(el["font_size"] * f * 2) / 2
            label = f"type:headline-size-x{f}"
        elif choice == "subhead-italic":
            el = tx.get("subhead") or tx.get("intro") or tx.get("body")
            if el.get("font_style") == "italic":
                el.pop("font_style")
                label = f"type:{el['id']}-upright"
            else:
                if el["font_family"] == "Solstice Serif" and el.get("font_weight") == "bold":
                    continue
                el["font_style"] = "italic"
                label = f"type:{el['id']}-italic"
        else:
            el = tx[rng.choice(caps)]
            el["letter_spacing"] = round(el["font_size"] * rng.choice([0.05, 0.12, 0.25]), 1)
            label = f"type:{el['id']}-spacing"
        if fit_text(trial, el, ctx["parent"], min_scale=0.6):
            src.clear()
            src.update(trial)
            return label
    return None


def op_palette(src, rng, ctx):
    name = rng.choice(sorted(PALETTES))
    pal = PALETTES[name]
    invert = ctx["tone"] != "mid" and rng.random() < 0.3
    trial = _copy.deepcopy(src)

    def recolor(color: str) -> str:
        rgb, alpha = _parse(color)
        role = ROLE_OF.get(rgb)
        if role is None:
            return color
        if invert:
            role = INVERT_ROLE.get(role, role)
        return _hex_rgba(pal[role], alpha)

    tone_after = ctx["tone"] if not invert else ("light" if ctx["tone"] == "dark" else "dark")
    for e in trial["elements"]:
        for key in ("color", "fill_color", "stroke_color"):
            if e.get(key):
                e[key] = recolor(e[key])
        for r in e.get("style_runs", []):
            if r.get("color"):
                r["color"] = recolor(r["color"])
        if e["type"] == "image":
            if e["asset_id"] == mc.GRADIENT or (invert and e["asset_id"] in DARK_BGS):
                _swap_image(e, pal["bg_light"] if invert else pal["bg_dark"])
            elif e["asset_id"] in (mc.LOGO, LOGO_ON_DARK, LOGO_ON_LIGHT):
                # The logo sits on the darkest role (a dark ad, or a dark band/pill on a light one).
                on_light = invert
                W = trial["canvas"]["width"]
                centred = abs(e["x"] + e["width"] / 2 - W / 2) < W * 0.02
                _swap_image(e, LOGO_ON_LIGHT if on_light else LOGO_ON_DARK, "center" if centred else "left")
        if e["type"] == "text":
            # Brand mentions only: "next solstice" (the day) stays, "solstice.example" (the site) changes.
            _replace_words(e, [("SOLSTICE", pal["name"].upper()), ("Solstice", pal["name"]),
                               ("solstice.example", pal["name"].lower() + ".example")])
    for e in trial["elements"]:  # brand names changed length: refit
        if e["type"] == "text" and not mc.measurer().fits(e, e["font_size"], e["width"], e["height"]):
            if not fit_text(trial, e, ctx["parent"]):
                return None
    if not _ok(trial, ctx["parent"]):
        return None
    src.clear()
    src.update(trial)
    ctx["brand_changed"] = True
    ctx["tone_after"] = tone_after
    return f"palette:{name}" + ("(inverted)" if invert else "")


def _rows(clusters: list[list[dict]]) -> list[list[dict]]:
    """Merge clusters that share a top edge or a vertical centre (a row of
    badges, a lineup of products) so they move as one unit."""
    units: list[list[dict]] = []
    for c in clusters:
        _, y, _, h = _bbox(c)
        for u in units:
            _, uy, _, uh = _bbox(u)
            if abs(uy - y) <= 3 or abs((uy + uh / 2) - (y + h / 2)) <= 3:
                u.extend(c)
                break
        else:
            units.append(list(c))
    return units


def op_jitter(src, rng, ctx):
    """Nudge whole units by up to 2% (x) / 1.5% (y) of the canvas. Units that
    share a left edge share one dx and centred units keep dx = 0, so columns
    stay aligned; rows (badges, lineups) move together."""
    W, H = src["canvas"]["width"], src["canvas"]["height"]
    units = _rows(_clusters(src, containment=True))
    keys: dict[tuple, float] = {}
    moved = 0
    order = list(range(len(units)))
    rng.shuffle(order)
    for i in order:
        x, _, w, _ = _bbox(units[i])
        centred = abs(x + w / 2 - W / 2) < 4
        dx = 0.0 if centred else keys.setdefault(("l", round(x / 4)), rng.uniform(-0.02, 0.02) * W)
        dy = rng.uniform(-0.015, 0.015) * H
        trial = _copy.deepcopy(src)
        ids = {e["id"] for e in units[i]}
        _move([e for e in trial["elements"] if e["id"] in ids], dx, dy)
        if _ok(trial, ctx["parent"]):
            src.clear()
            src.update(trial)
            moved += 1
    return f"jitter:{moved}-units" if moved >= 2 else None


OPS = {"canvas": op_canvas, "remove": op_remove, "add": op_add, "copy": op_copy, "image": op_image,
       "type": op_type, "palette": op_palette, "jitter": op_jitter}
ORDER = ["canvas", "remove", "add", "copy", "image", "type", "palette", "jitter"]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_variant(parent: dict, slot: int, attempt: int, assets, parent_px: dict, log) -> dict | None:
    rng = random.Random(f"{parent['task_id']}:{slot}:{attempt}")
    want = rng.choice([2, 3, 3, 4])
    pool = list(ORDER)
    rng.shuffle(pool)
    src = _copy.deepcopy(parent["source"])
    ctx = {"parent": parent["task_id"], "tone": _tone(parent), "brand_changed": False}
    # The first `want` ops in canonical order; if one doesn't apply, the next spare op stands in.
    applied = []
    for name in sorted(pool[:want], key=ORDER.index) + sorted(pool[want:], key=ORDER.index):
        if len(applied) >= want:
            break
        label = OPS[name](src, rng, ctx)
        if label:
            applied.append(label)
    if len(applied) < 2:
        log(f"{parent['task_id']} slot {slot} attempt {attempt}: only {len(applied)} operation(s) applied")
        return None
    rec = {"task_id": f"{parent['task_id']}_v{slot + 1}", "split": parent["split"], "origin": "augmented",
           "brand": "offbrand" if ctx["brand_changed"] else "solstice", "parent": parent["task_id"],
           "augmentations": applied, "expects_missing_asset": bool(mc._missing_assets(src)),
           "notes": f"Augmented variant of {parent['task_id']}: " + "; ".join(applied) + ".",
           "features": _features(parent, ctx, src), "source": src}
    problems = mc.check(rec, strict=True) + mc.pixel_check(rec, assets, parent_px)
    if problems:
        log(f"{rec['task_id']} attempt {attempt} DROPPED [{'; '.join(applied)}]: {'; '.join(problems)}")
        return None
    return rec


def _features(parent: dict, ctx: dict, src: dict) -> list[str]:
    feats = [f for f in parent.get("features", []) if f not in ("dark", "light", "mid-tone")]
    tone = ctx.get("tone_after", ctx["tone"])
    feats.insert(0, {"mid": "mid-tone"}.get(tone, tone))
    if any(e["type"] == "image" and e["asset_id"].startswith("syn_") for e in src["elements"]):
        feats.append("synthetic-image")
    feats.append("augmented")
    return feats


def generate(drafted: list[dict], log=print) -> tuple[list[dict], list[str]]:
    out, dropped = [], []

    def note(msg):
        dropped.append(msg)
        log(msg)

    with merged_assets_dir() as assets:
        for parent in drafted:
            parent_px = mc.text_backdrops(parent["source"], assets)
            seen = {hashlib.sha1(json.dumps(parent["source"], sort_keys=True).encode()).hexdigest()}
            for slot in range(VARIANTS_PER_AD):
                for attempt in range(MAX_ATTEMPTS):
                    rec = make_variant(parent, slot, attempt, assets, parent_px, note)
                    if rec is None:
                        continue
                    key = hashlib.sha1(json.dumps(rec["source"], sort_keys=True).encode()).hexdigest()
                    if key in seen:
                        note(f"{rec['task_id']} attempt {attempt}: duplicate of an earlier variant")
                        continue
                    seen.add(key)
                    out.append(rec)
                    break
                else:
                    note(f"{parent['task_id']} slot {slot}: no passing variant in {MAX_ATTEMPTS} attempts")
    return out, dropped


def main() -> None:
    drafted = [r for r in mc.drafted() if r["origin"] == "claude-drafted"]
    records, dropped = generate(drafted)
    mc.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    mc.sync_designs(records, ("augmented",))
    DROP_LOG.write_text("\n".join(dropped) + "\n")
    mc.contact_sheet(records, REVIEW_SHEET)
    counts: dict[tuple, int] = {}
    for r in records:
        counts[(r["brand"], r["split"])] = counts.get((r["brand"], r["split"]), 0) + 1
    print(f"wrote {len(records)} augmented ads; by (brand, split): {dict(sorted(counts.items()))}")
    print(f"{len(dropped)} dropped attempts (log: {DROP_LOG}); review sheet: {REVIEW_SHEET}")


if __name__ == "__main__":
    main()
