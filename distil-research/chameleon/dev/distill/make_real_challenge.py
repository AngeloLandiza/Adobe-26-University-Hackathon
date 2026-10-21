"""Write challenge2: ~50 REAL ads converted from the Crello dataset into our
design format, to measure the engine on designs nobody here drew.

    cd distil-research/chameleon && python -m dev.distill.make_real_challenge [--n 50] [--workers 4] [--seed 0]

Source: cyberagent/crello revision 5.1.0 (4 parquet shards under
training-data/references/crello.nosync/data/), ad formats only (the
AD_FORMATS list of dev/references/crello_stats.py: 1,606 templates).

LICENCE RULE: Crello is CDLA-Permissive-2.0 but the VistaCreate template terms
are unverified, so nothing derived from it with its content is ever committed.
Every output of this script goes under training-data/corpus/challenge2.nosync/
(gitignored): the design JSONs (<task_id>.json), the per-element rasters
(assets/cr2_*.png), the Crello previews (previews/), the review sheets and
scan.jsonl (one row per template: kept or why skipped). Only the aggregate
counts in summary.json are meant to be quoted (CHALLENGE2.md).

Two passes:
1. scan (4 processes, one parquet row group at a time): convert every ad
   template, rasters into a per-process temp folder, and run the quality gate
   (`make_challenge.quality_issues`: make_corpus.check strict + pixel_check).
   Each template is kept or gets ONE skip reason (the first that applies).
2. select ~n kept templates stratified by source aspect class (the engine's
   six classes) and, inside a class, half with a photo/hero image and half
   text-only (real ads are about half each), spreading canvas sizes
   round-robin; convert them again for real (rasters into assets/), re-check,
   write, and draw sheet.png (Crello preview | our render, side by side).

Conversion (Crello element -> ours):
- roles from dev/references/crello_roles.label (the same heuristic the
  real-ad statistics use). Ids follow the gate's convention: background ->
  `bg` (more backgrounds `deco_bg_N`), decoration -> `deco_N`, the largest
  photo -> `product` (or `deco_backdrop` when it covers half the canvas or
  more: copy on top of it is intended, not an accidental overlap), the logo -> `logo`, CTA text + its button container ->
  `cta_label`/`cta_bg` grouped `cta`, the biggest other text -> `headline`,
  other text -> `body_N`. The engine never reads ids, only the gate does.
- ColoredBackground -> rectangle shape; a flat opaque SvgElement -> rectangle,
  a flat elliptical one -> ellipse; every other Svg/Image/SvgMask element ->
  its own Crello raster saved as a PNG (cropped to the canvas when it bleeds
  and isn't rotated). A background photo becomes a full-canvas `cover` image.
- text: content (upper-cased when Crello capitalizes), size, colour (the
  majority colour; other colours/bold as style_runs), alignment, line height,
  letter spacing, rotation, box. Fonts map to the two families the renderer
  has: serif / slab / script faces -> Solstice Serif (scripts italic), the
  rest -> Solstice Sans; heavy display faces are bold. Because our faces are
  wider or narrower than Crello's, the font size shrinks (to 70% at most)
  until the text wraps into no more lines than Crello's own, and the box grows
  downward to the measured height (text draws from the top, so nothing moves).
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from dev.distill import assets as assets_mod
from dev.distill.assets import BRAND_FONTS_DIR, CHALLENGE2_ASSETS_DIR, CHALLENGE2_FOLDERS, merged_assets_dir
from dev.distill.paths import CHALLENGE2_DIR, DATA_ROOT

REFS = Path(__file__).resolve().parents[1] / "references"
SHARDS_DIR = DATA_ROOT / "references" / "crello.nosync" / "data"
PREVIEW_DIR = CHALLENGE2_DIR / "previews"
SHEET = CHALLENGE2_DIR / "sheet.png"
SCAN_FILE = CHALLENGE2_DIR / "scan.jsonl"
SUMMARY_FILE = CHALLENGE2_DIR / "summary.json"
PREFIX = "cr2"
SPLIT, ORIGIN, BRAND = "challenge2", "crello", "crello"

COLS = ["id", "length", "format", "canvas_width", "canvas_height", "type", "left", "top", "width", "height",
        "angle", "opacity", "color", "image", "text", "font", "font_size", "font_bold", "font_italic",
        "text_line", "text_color", "text_align", "capitalize", "line_height", "letter_spacing", "preview"]

MIN_FONT_SCALE = 0.70    # our face may set text at most this much smaller than Crello's
MIN_CONDENSED_SCALE = 0.55  # ... or this much for condensed display faces (ours are far wider)
BACKDROP_SHARE = 0.5     # the largest photo covering this share of the canvas is a backdrop layer, not a product
PHOTO_SHARE = 0.05       # an Image/SvgMask element covering this share of the canvas makes the ad "with image"
FLAT_STD = 6.0           # RGB std (0-255) below which a raster is one flat colour
ELLIPSE_IOU = 0.97
# Per source aspect class: how many ads (sums to 50); half text-only where the class has enough.
CLASS_QUOTA = {"square": 10, "landscape": 10, "tall": 10, "wide": 9, "very_tall": 6, "strip": 5}

SERIF = {"Playfair Display", "Abril Fatface", "Libre Baskerville", "Cormorant Infant", "Arapey", "Kreon", "Podkova",
         "Frank Ruhl Libre", "Droid Serif", "Merriweather", "Crete Round", "Alfa Slab One", "Rokkitt", "Tinos",
         "Josefin Slab", "Noticia Text", "Old Standard Tt", "Radley", "Superclarendon", "Vollkorn", "Yeseva One",
         "Glass Antiqua", "Oranienbaum", "Port Lligat Slab", "Source Serif Pro", "Mate", "Kurale", "Rye",
         "Quattrocento", "Bentham", "Junge", "Kelly Slab", "Alice", "Forum", "Bellefair", "Marcellus Sc",
         "Im Fell Dw Pica Sc", "Gfs Didot", "Eb Garamond", "David Libre", "Italiana", "Patua One", "Gravitas One",
         "Vast Shadow", "Fauna One", "Gabriela", "Rammetto One", "Sansita One", "Arima Madurai", "Copse", "Lora",
         "Cormorant", "Prata", "Bilbo", "Elsie", "Elsie Swash Caps", "Berkshire Swash", "Limelight", "Rakkas",
         "Suez One", "Secular One", "Ruslan Display", "Jacques Francois Shadow", "Playball"}
SCRIPT = {"Playlist Script", "Pacifico", "Yellowtail", "Oleo Script", "Kalam", "Bad Script", "Selima Script",
          "Lemon Tuesday", "Sunday", "Cookie", "Brusher Free Font", "Mrs Sheppards", "Grand Hotel", "Great Vibes",
          "Damion", "Parisienne", "Satisfy", "Sacramento", "Mr Dafoe", "Lauren", "Mr De Haviland", "Yesteryear",
          "Dancing Script", "Playlist", "Kaushan Script", "Mrs Saint Delafield", "Allura", "Brusher", "Rochester",
          "Dawning Of A New Day", "Stalemate", "Pinyon Script", "Clicker Script", "La Belle Aurore", "Homemade Apple",
          "Meddon", "Arizonia", "Cedarville Cursive", "Courgette", "Euphoria Script", "Zeyada", "League Script",
          "Give You Glory", "Kristi", "Italianno", "Lovers Quarrel", "Marck Script", "Monsieur La Doulaise",
          "Petit Formal Script", "Caveat", "Merienda One", "Breathe", "Breathe Press", "Rissatypeface", "Okolaks",
          "Mikodacs", "Znikomitno24", "Znikomit", "Sirin Stencil", "Waiting For The Sunrise", "Loved By The King",
          "Nothing You Could Do", "Covered By Your Grace", "Reenie Beanie", "Shadows Into Two",
          "Sue Ellen Francisco", "Just Me Again Down Here", "Rock Salt", "Permanent Marker", "Handlee",
          "Patrick Hand", "Architects Daughter", "Delius Swash Caps", "Dukomdesign Constantine", "Beacon", "Rise",
          "Sensei", "Pirou", "Arkana", "Gluk Foglihtenno06"}
CONDENSED = {"Bebas Neue", "Oswald", "Anton", "Fjalla One", "Six Caps", "Pathway Gothic One", "Yanone Kaffeesatz",
             "Economica", "Open Sans Condensed", "Tulpen One", "Wire One", "Dorsa", "Medula One", "Chathura",
             "Amatic Sc", "Pompiere", "Poiret One", "Julius Sans One", "Gruppo", "Iceland", "Farsan", "Geo",
             "Rationale", "League Gothic", "Teko", "Archivo Narrow"}
# Faces whose lowercase letters are capitals: Crello stores mixed case but shows capitals.
CAPS_ONLY = {"Bebas Neue", "Julius Sans One", "Six Caps", "Graduate", "Marcellus Sc", "Im Fell Dw Pica Sc", "Bungee",
             "Bungee Shade", "Amatic Sc", "Bangers"}
HEAVY = {"Bebas Neue", "Oswald", "Abril Fatface", "Anton", "Alfa Slab One", "Fjalla One", "Rubik One", "Bungee",
         "Bangers", "Black Ops One", "Russo One", "Racing Sans One", "Rammetto One", "Sansita One", "Gravitas One",
         "Lalezar", "Knewave", "Carter One", "Bungee Shade", "Kumar One", "Seymour One", "Ultra", "Titan One",
         "Righteous", "Passion One", "Yeseva One", "Patua One", "Ruslan Display", "Graduate", "Pathway Gothic One",
         "Six Caps", "Medula One", "Contrail One", "Secular One", "Suez One", "Faster One", "Monofett",
         "Vampiro One", "Rakkas", "Limelight", "Bevan", "Archivo Black", "Baloo Tamma"}


class Skip(Exception):
    """A template we can't represent faithfully (the message is the reason)."""


# ---------------------------------------------------------------- reading Crello

def shards() -> list[Path]:
    return sorted(SHARDS_DIR.glob("test-*-of-*.parquet"))


def features() -> dict:
    import pyarrow.parquet as pq
    return json.loads(pq.ParquetFile(shards()[0]).schema_arrow.metadata[b"huggingface"])["info"]["features"]


def _crello_stats():
    sys.path.insert(0, str(REFS))
    import crello_roles  # noqa: E402  (scripts that import each other by bare name)
    import crello_stats
    return crello_roles, crello_stats


def row_groups() -> list[tuple[str, int]]:
    import pyarrow.parquet as pq
    return [(str(s), g) for s in shards() for g in range(pq.ParquetFile(s).num_row_groups)]


def read_group(shard: str, group: int, ids: set[str] | None = None) -> list[dict]:
    import pyarrow.parquet as pq
    rows = pq.ParquetFile(shard).read_row_group(group, columns=COLS).to_pylist()
    return [r for r in rows if ids is None or r["id"] in ids]


def is_ad(row: dict, F: dict) -> bool:
    _, cs = _crello_stats()
    return (F["format"]["names"][row["format"]] in cs.AD_FORMATS
            and row["canvas_width"] >= 50 and row["canvas_height"] >= 50)


# ---------------------------------------------------------------- small helpers

def _rgba(text: str | None, alpha: float = 1.0) -> str:
    """Crello 'rgba(255, 236, 91, 1)' -> 'rgba(255,236,91,1)', alpha multiplied."""
    m = re.match(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)", text or "")
    if not m:
        return f"rgba(0,0,0,{alpha:g})"
    r, g, b = (int(round(float(m[i]))) for i in (1, 2, 3))
    a = (float(m[4]) if m[4] is not None else 1.0) * alpha
    return f"rgba({r},{g},{b},{round(a, 3):g})"


def _num(x: float) -> float:
    return round(float(x), 1)


def _angle(a: float | None) -> float:
    a = float(a or 0) % 360
    if a > 180:
        a -= 360
    return 0.0 if abs(a) < 0.5 else round(a, 2)


def font_for(name: str, bold_share: float) -> dict:
    if name in SCRIPT:
        return {"font_family": "Solstice Serif", "font_style": "italic",
                **({"font_weight": "bold"} if bold_share >= 0.5 else {})}
    family = "Solstice Serif" if name in SERIF else "Solstice Sans"
    return {"font_family": family, **({"font_weight": "bold"} if bold_share >= 0.5 or name in HEAVY else {})}


def _runs(values: list, majority, key: str, content: str) -> list[dict]:
    """Contiguous [start, end) runs of characters whose value differs from the majority."""
    out, start = [], None
    for i, v in enumerate(values + [majority]):
        differs = i < len(values) and v != majority and content[i] not in "\n"
        if differs and start is None:
            start = i
        elif start is not None and (not differs or v != values[start]):
            out.append({"start": start, "end": i, key: values[start]})
            start = i if differs else None
    return out


# ---------------------------------------------------------------- rasters

def _decode(img: dict | None) -> Image.Image | None:
    if not img or not img.get("bytes"):
        return None
    return Image.open(io.BytesIO(img["bytes"])).convert("RGBA")


def _flat(im: Image.Image, mask=None):
    """(mean rgb, std) of the opaque pixels."""
    import numpy as np
    a = np.asarray(im, dtype=float)
    sel = a[..., 3] > 200 if mask is None else mask
    if not sel.any():
        return None, 1e9
    rgb = a[..., :3][sel]
    return tuple(int(round(v)) for v in rgb.mean(axis=0)), float(rgb.std(axis=0).max())


def classify_svg(im: Image.Image) -> tuple[str, tuple] | None:
    """('rectangle'|'ellipse', rgb) for a flat opaque rectangle or ellipse raster, else None."""
    import numpy as np
    # nearest-neighbour keeps stripes and texture visible (a smoothing resize would average them flat)
    small = im.resize((min(im.width, 160), min(im.height, 160)), Image.NEAREST) if max(im.size) > 160 else im
    a = np.asarray(small, dtype=float)
    alpha = a[..., 3]
    if alpha.min() >= 250:
        rgb, std = _flat(small)
        return ("rectangle", rgb) if std < FLAT_STD else None
    h, w = alpha.shape
    if h < 8 or w < 8:
        return None
    yy, xx = np.mgrid[0:h, 0:w]
    ell = ((xx + 0.5 - w / 2) / (w / 2)) ** 2 + ((yy + 0.5 - h / 2) / (h / 2)) ** 2 <= 1.0
    solid = alpha > 128
    iou = (ell & solid).sum() / max(1, (ell | solid).sum())
    if iou < ELLIPSE_IOU:
        return None
    rgb, std = _flat(small, ell & (alpha > 250))
    return ("ellipse", rgb) if std < FLAT_STD else None


def crop_to_canvas(im: Image.Image, box: tuple, W: float, H: float) -> tuple[Image.Image, tuple]:
    """The part of a raster drawn at `box` (x, y, w, h) that is inside the canvas."""
    x, y, w, h = box
    x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(W, x + w), min(H, y + h)
    if (x0, y0, x1, y1) == (x, y, x + w, y + h):
        return im, box
    sx, sy = im.width / w, im.height / h
    crop = im.crop((round((x0 - x) * sx), round((y0 - y) * sy), round((x1 - x) * sx), round((y1 - y) * sy)))
    return crop, (x0, y0, x1 - x0, y1 - y0)


def fit_raster(im: Image.Image, w: float, h: float) -> Image.Image:
    """Resample so the raster's aspect is the box's (the gate checks within 2%)."""
    scale = max(im.width / w, im.height / h, 48 / max(w, h))
    scale = min(scale, 2048 / max(w, h))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return im if size == im.size else im.resize(size, Image.LANCZOS)


# ---------------------------------------------------------------- conversion

def has_photo(row: dict, F: dict) -> bool:
    T = F["type"]["feature"]["names"]
    A = row["canvas_width"] * row["canvas_height"]
    for i in range(row["length"]):
        if T[row["type"][i]] not in ("ImageElement", "SvgMaskElement") or (row["opacity"][i] or 0) < 0.05:
            continue
        x0, y0 = max(0, row["left"][i]), max(0, row["top"][i])
        x1 = min(row["canvas_width"], row["left"][i] + row["width"][i])
        y1 = min(row["canvas_height"], row["top"][i] + row["height"][i])
        if x1 > x0 and y1 > y0 and (x1 - x0) * (y1 - y0) >= PHOTO_SHARE * A:
            return True
    return False


def convert(row: dict, F: dict, task_id: str, asset_dir: Path, tm) -> dict:
    """One Crello row -> our design record; rasters written into `asset_dir`.
    Raises Skip(reason)."""
    roles_mod, _ = _crello_stats()
    T = F["type"]["feature"]["names"]
    FONTS = F["font"]["feature"]["names"]
    ALIGN = F["text_align"]["feature"]["names"]
    W, H = int(row["canvas_width"]), int(row["canvas_height"])
    lay = roles_mod.label(row, F)
    role = {it["i"]: it["role"] for it in lay["items"]}
    area_share = {it["i"]: it["area"] for it in lay["items"]}
    cta_of = {}  # container index -> cta prefix
    n_cta = 0
    for it in sorted((it for it in lay["items"] if it["role"] == "cta"), key=lambda it: it["i"]):
        n_cta += 1
        prefix = "cta" if n_cta == 1 else f"cta{n_cta}"
        it["prefix"] = prefix
        if it.get("cta_box") is not None:
            cta_of[it["cta_box"]["i"]] = prefix
    cta_text = {it["i"]: it["prefix"] for it in lay["items"] if it["role"] == "cta"}
    if not any(T[row["type"][i]] == "TextElement" and (row["text"][i] or "").strip() for i in role):
        raise Skip("no_text")

    counters: Counter = Counter()
    have_bg = False

    def new_id(base: str) -> str:
        counters[base] += 1
        if base in ("deco", "body", "deco_bg"):
            return f"{base}_{counters[base]}"
        return base if counters[base] == 1 else f"{base}_{counters[base]}"

    elements = []
    for z, i in enumerate(sorted(set(role) | set(cta_of))):
        t = T[row["type"][i]]
        r = role.get(i, "cta_box")
        box = (float(row["left"][i]), float(row["top"][i]), float(row["width"][i]), float(row["height"][i]))
        if box[2] < 1 or box[3] < 1:
            continue
        opacity = float(row["opacity"][i] if row["opacity"][i] is not None else 1.0)
        rotation = _angle(row["angle"][i])
        extra = {}
        if rotation:
            extra["rotation"] = rotation
        if opacity < 0.999:
            extra["opacity"] = round(opacity, 3)

        if t == "TextElement":
            content = (row["text"][i] or "").replace("\r", "")
            if not content.strip():
                continue
            face = FONTS[row["font"][i] or 0]
            if row["capitalize"][i] or face in CAPS_ONLY:
                content = content.upper()
            content = content.rstrip()
            n = len(content)
            colors = [_rgba(c) for c in (row["text_color"][i] or [])][:n]
            colors += [colors[-1] if colors else "rgba(0,0,0,1)"] * (n - len(colors))
            bold = list(row["font_bold"][i] or [])[:n]
            bold += [bold[-1] if bold else False] * (n - len(bold))
            italic = list(row["font_italic"][i] or [])[:n]
            visible = [k for k, ch in enumerate(content) if not ch.isspace()] or list(range(n))
            main_color = Counter(colors[k] for k in visible).most_common(1)[0][0]
            bold_share = sum(bold[k] for k in visible) / len(visible)
            el = {"id": "", "type": "text", "x": _num(box[0]), "y": _num(box[1]), "width": _num(box[2]),
                  "height": _num(box[3]), "z_index": z, "content": content,
                  "font_size": round(float(row["font_size"][i] or 12), 1),
                  **font_for(FONTS[row["font"][i]] if row["font"][i] is not None else "", bold_share),
                  "color": main_color,
                  "text_align": (ALIGN[row["text_align"][i]] or "left") if row["text_align"][i] is not None else "left"}
            if italic and sum(italic[k] for k in visible if k < len(italic)) / len(visible) >= 0.5:
                el["font_style"] = "italic"
            lh = float(row["line_height"][i] or 1.0)
            if abs(lh - 1.2) > 0.01:
                el["line_height"] = round(lh, 3)
            ls = float(row["letter_spacing"][i] or 0)
            if abs(ls) > 0.05:
                el["letter_spacing"] = round(ls, 2)
            runs = _runs(colors, main_color, "color", content)
            if 0 < bold_share < 1:
                runs += [{**r_, "font_weight": "bold" if r_["font_weight"] else "regular"}
                         for r_ in _runs(bold, bold_share >= 0.5, "font_weight", content)]
            if runs:
                el["style_runs"] = sorted(runs, key=lambda r_: r_["start"])
            lines = row["text_line"][i] or []
            crello_lines = (max(lines) + 1) if lines else content.count("\n") + 1
            if missing_glyphs(el, tm):
                raise Skip("unsupported_glyphs")  # e.g. Cyrillic or emoji: our two faces would draw boxes
            el["_lines"] = crello_lines
            el["_min_scale"] = MIN_CONDENSED_SCALE if face in CONDENSED else MIN_FONT_SCALE
            fit_text(el, crello_lines, tm)
            _keep_on_canvas(el, W, H, tm)
            tighten(el, tm)
            if r == "cta":
                el["id"] = f"{cta_text[i]}_label"
                el["group_id"] = cta_text[i]
            elif r == "headline":
                el["id"] = new_id("headline")
            elif r == "logo":
                el["id"] = new_id("logo")
            else:
                el["id"] = new_id("body")
            el.update(extra)
            elements.append(el)
            continue

        # non-text: a background colour, a flat shape, or a raster image
        if r == "background":
            base = "bg" if not have_bg else "deco_bg"
            have_bg = True
        elif r == "product" and area_share.get(i, 0) >= BACKDROP_SHARE:
            base = "deco_backdrop"  # a half-canvas-or-bigger photo the copy sits on: a backdrop layer
        elif r == "product":
            base = "product"
        elif r == "logo":
            base = "logo"
        elif i in cta_of:
            base = f"{cta_of[i]}_bg"
        else:
            base = "deco"
        if t == "ColoredBackground":
            colors = row["color"][i] or []
            if len(colors) != 1:
                raise Skip("gradient_background")
            el = {"id": "", "type": "shape", "x": 0, "y": 0, "width": W, "height": H, "z_index": z,
                  "shape_type": "rectangle", "fill_color": _rgba(colors[0])}
        else:
            im = _decode(row["image"][i])
            if im is None:
                raise Skip("missing_raster")
            kind = classify_svg(im) if t == "SvgElement" else None
            inside = box[0] >= -0.5 and box[1] >= -0.5 and box[0] + box[2] <= W + 0.5 and box[1] + box[3] <= H + 0.5
            if kind and (kind[0] == "rectangle" or inside):
                x, y, w, h = box
                if kind[0] == "rectangle" and not rotation:
                    x, y = max(0.0, x), max(0.0, y)
                    w, h = min(W, box[0] + box[2]) - x, min(H, box[1] + box[3]) - y
                el = {"id": "", "type": "shape", "x": _num(x), "y": _num(y), "width": _num(w), "height": _num(h),
                      "z_index": z, "shape_type": kind[0], "fill_color": "rgba({},{},{},1)".format(*kind[1])}
            else:
                if not rotation:
                    im, box = crop_to_canvas(im, box, W, H)
                if box[2] < 1 or box[3] < 1:
                    continue
                if r == "background" and t == "ImageElement" and not rotation and box[2] * box[3] >= 0.99 * W * H:
                    box, fit = (0, 0, W, H), "cover"
                else:
                    fit = "contain"
                im = fit_raster(im, box[2], box[3])
                el = {"id": "", "type": "image", "x": _num(box[0]), "y": _num(box[1]), "width": _num(box[2]),
                      "height": _num(box[3]), "z_index": z, "asset_id": "", "fit": fit}
                if fit != "cover" and abs((el["width"] / el["height"]) / (im.width / im.height) - 1) > 0.015:
                    im = im.resize((max(1, round(im.height * el["width"] / el["height"])), im.height), Image.LANCZOS)
                el["_raster"] = im
        el["id"] = new_id(base)
        if i in cta_of:
            el["group_id"] = cta_of[i]
        el.update(extra)
        elements.append(el)

    if not any(e["type"] == "text" for e in elements):
        raise Skip("no_text")
    settle_text_growth(elements, H, tm)
    for el in elements:
        for k in [k for k in el if k.startswith("_") and k != "_raster"]:
            del el[k]
    for el in elements:
        if "_raster" in el:
            name = f"{task_id}_{el['id']}.png"
            el.pop("_raster").save(asset_dir / name, optimize=True)
            el["asset_id"] = name
    cls = aspect_class(W, H)
    photo = has_photo(row, F)
    return {"task_id": task_id, "split": SPLIT, "origin": ORIGIN, "brand": BRAND, "expects_missing_asset": False,
            "notes": f"Crello ad template ({F['format']['names'][row['format']]}), converted; {len(elements)} elements.",
            "features": [cls, "photo" if photo else "text-only"],
            "crello_id": row["id"], "crello_format": F["format"]["names"][row["format"]],
            "source": {"canvas": {"width": W, "height": H}, "elements": elements}}


def aspect_class(w, h) -> str:
    from engine.pipeline.layout.aspect import aspect_class as ac
    return ac(w, h)


def missing_glyphs(el: dict, tm) -> bool:
    """True if the face the renderer will use has no glyph for some character."""
    import skia
    family, weight, style = tm._font(el)
    weight = "bold" if style in ("bold", "bolditalic") else "regular"
    font = skia.Font(skia.Typeface.MakeFromData(tm._fonts.lookup(family, weight, style)), 12)
    text = "".join(ch for ch in el["content"] if not ch.isspace())
    return bool(text) and 0 in font.textToGlyphs(text)


def text_signature(row: dict, F: dict) -> list[str]:
    """Short hashes of the template's texts of 8+ characters (lower-cased,
    spaces collapsed): the same design resized to other formats, or a sibling
    of it, shares some, so selection takes one per family. Hashes, not text."""
    T = F["type"]["feature"]["names"]
    texts = {" ".join((row["text"][i] or "").lower().split()) for i in range(row["length"])
             if T[row["type"][i]] == "TextElement"}
    return sorted(hashlib.sha1(t.encode()).hexdigest()[:12] for t in texts if len(t) >= 8)


def _text_warnings(el: dict) -> list[str]:
    """The gate's own text checks (make_corpus.check) on this one element."""
    from dev.distill import make_corpus as mc
    lone = {k: v for k, v in el.items() if not k.startswith("_") and k not in ("rotation", "group_id")}
    rec = {"task_id": "_fit", "expects_missing_asset": False,
           "source": {"canvas": {"width": 10 ** 6, "height": 10 ** 6}, "elements": [dict(lone, x=0, y=0)]}}
    return mc.check(rec)


def fit_text(el: dict, crello_lines: int, tm, cap: float | None = None) -> None:
    """Shrink the size (to MIN_FONT_SCALE) until our face wraps into at most
    Crello's line count, no line is wider than the box, and the box height the
    gate needs fits under `cap` (default: no cap); the box grows downward only
    (text draws from the top, so the glyphs stay where Crello put them).
    Letter spacing scales with the size. Raises Skip('font_substitution')."""
    fs0, w, h0 = el.get("_fs0", el["font_size"]), el["width"], el.get("_h0", el["height"])
    ls0 = el.get("_ls0", float(el.get("letter_spacing", 0)))
    el["_fs0"], el["_h0"], el["_ls0"] = fs0, h0, ls0
    lh = float(el.get("line_height", 1.2))
    for k in range(0, int(round((1 - el.get("_min_scale", MIN_FONT_SCALE)) * 100)) + 1):
        fs = round(fs0 * (1 - k / 100), 1)
        if ls0:  # tracking is in px: it shrinks with the type
            el["letter_spacing"] = round(ls0 * fs / fs0, 2)
        lines = tm.line_count(el, fs, w)
        if lines > max(1, crello_lines) or max(tm.line_widths(el, fs, w), default=0.0) > w - 0.5:
            continue
        need = max(tm.block_height(el, fs, w), (lines - 1) * lh * fs + 1.3 * fs)
        h = max(h0, math.ceil(need + 1))
        limit = cap if cap is not None else h + 2 * fs
        el["font_size"] = fs
        while h <= limit:
            el["height"] = h
            if not _text_warnings(el):
                return
            h += max(1, math.ceil(0.05 * fs))
    raise Skip("font_substitution")


def tighten(el: dict, tm) -> None:
    """Shrink a text box's width to its widest line (plus a small margin) on
    the side(s) its alignment leaves empty, so the glyphs don't move but the
    box no longer spans backdrop the text never touches (Crello boxes are often
    far wider than their text, which skews the gate's contrast reading)."""
    if el.get("rotation"):
        return
    fs, w = el["font_size"], el["width"]
    widest = max(tm.line_widths(el, fs, w), default=0.0)
    new_w = math.ceil(widest + max(2.0, 0.04 * fs) + 2 * abs(float(el.get("letter_spacing", 0))))
    if new_w >= w - 1:
        return
    old = (el["x"], el["width"])
    shift = {"center": (w - new_w) / 2, "right": w - new_w}.get(el["text_align"], 0.0)
    el["x"], el["width"] = round(el["x"] + shift, 1), new_w
    if tm.line_count(el, fs, new_w) != tm.line_count(el, fs, w) or _text_warnings(el):
        el["x"], el["width"] = old


def _box4(el: dict) -> tuple[float, float, float, float]:
    from engine.pipeline.geometry import rotated_aabb
    return tuple(rotated_aabb(el))


def _hit(a, b) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _inside(outer, inner) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and inner[0] + inner[2] <= outer[0] + outer[2] and inner[1] + inner[3] <= outer[1] + outer[3])


def settle_text_growth(elements: list[dict], H: int, tm) -> None:
    """A grown text box must not run into content it didn't touch in Crello
    (the next line of copy, a product, the bottom of its button): cap its
    height at the first such element below it (or its container's bottom, or
    the canvas) and refit, shrinking the font if needed."""
    is_layer = lambda e: e["id"] == "bg" or e["id"].startswith("deco")
    content = [e for e in elements if not is_layer(e)]
    for t in [e for e in content if e["type"] == "text" and not e.get("rotation")]:
        orig = (t["x"], t["y"], t["width"], t["_h0"])
        cap = H - t["y"]
        for e in content:
            if e is t or (t.get("group_id") and t.get("group_id") == e.get("group_id") and e["type"] == "text"):
                continue
            eb = _box4(e) if e["type"] != "text" else (e["x"], e["y"], e["width"], e.get("_h0", e["height"]))
            if _hit(orig, eb):
                if e["type"] == "shape" and _inside(eb, orig):
                    cap = min(cap, eb[1] + eb[3] - t["y"])
                continue  # the designer's own overlap: the gate judges it
            if eb[0] < orig[0] + orig[2] and orig[0] < eb[0] + eb[2] and eb[1] >= orig[1]:
                cap = min(cap, eb[1] - t["y"] - 0.5)
        if t["height"] > cap:
            fit_text(t, t["_lines"], tm, cap=math.floor(cap))


def _keep_on_canvas(el: dict, W: int, H: int, tm) -> None:
    """Trim a text box that pokes past the canvas sides where that doesn't move
    the glyphs (a centred box symmetrically, a left box on its right, a right
    box on its left) and refit; Skip('text_off_canvas') when the text itself
    starts outside. (The bottom is settled by settle_text_growth.)"""
    if el.get("rotation"):
        return
    if el["y"] < -0.5 or el["y"] >= H:
        raise Skip("text_off_canvas")
    x, w = el["x"], el["width"]
    if x >= -0.5 and x + w <= W + 0.5:
        return
    align = el["text_align"]
    if align == "center":
        cx = x + w / 2
        half = min(cx, W - cx, w / 2)
        x, w = cx - half, 2 * half
    elif align == "right":
        if x + w > W + 0.5:
            raise Skip("text_off_canvas")
        x, w = max(0.0, x), x + w - max(0.0, x)
    else:
        if x < -0.5:
            raise Skip("text_off_canvas")
        x, w = max(0.0, x), min(W, x + w) - max(0.0, x)
    if w <= 1:
        raise Skip("text_off_canvas")
    el["x"], el["width"] = math.ceil(x * 10) / 10, math.floor(w * 10) / 10
    try:
        fit_text(el, el["_lines"], tm)
    except Skip:
        raise Skip("text_off_canvas")


def gate_reason(issues: list[str]) -> str:
    """Bucket the first quality-gate issue into a short reason."""
    w = issues[0]
    for key, pat in (("gate_overlap", r"^overlap"), ("gate_contrast", r"contrast"), ("gate_busy_backdrop", r"busy"),
                     ("gate_off_canvas", r"off canvas"), ("gate_text_fit", r"lines need|wide, box|does not fit"),
                     ("gate_image_aspect", r"box aspect"), ("gate_missing_asset", r"expects_missing_asset")):
        if re.search(pat, w):
            return key
    return "gate_other"


# ---------------------------------------------------------------- pass 1: scan

@contextlib.contextmanager
def scratch_assets():
    """A temp assets folder (with the brand fonts beside it) that the gate's
    asset lookups also see; removed afterwards."""
    with tempfile.TemporaryDirectory(prefix="chameleon-cr2-") as tmp:
        out = Path(tmp) / "assets"
        out.mkdir()
        (Path(tmp) / "fonts").symlink_to(BRAND_FONTS_DIR.resolve(), target_is_directory=True)
        saved = assets_mod.LOOKUP_FOLDERS
        assets_mod.LOOKUP_FOLDERS = (*saved, out)
        try:
            yield out
        finally:
            assets_mod.LOOKUP_FOLDERS = saved


def check_one(row: dict, F: dict, task_id: str, folder: Path) -> tuple[dict | None, list[str], str | None]:
    """(record, gate issues, skip reason)."""
    from dev.distill.make_challenge import quality_issues
    from dev.distill.make_corpus import measurer
    try:
        rec = convert(row, F, task_id, folder, measurer())
    except Skip as s:
        return None, [], str(s)
    issues = quality_issues(rec, folder)
    return rec, issues, gate_reason(issues) if issues else None


def scan_group(job: tuple[str, int]) -> list[dict]:
    shard, group = job
    F = features()
    out = []
    with scratch_assets() as folder:
        for row in read_group(shard, group):
            if not is_ad(row, F):
                continue
            W, H = row["canvas_width"], row["canvas_height"]
            meta = {"crello_id": row["id"], "shard": Path(shard).name, "group": group, "W": W, "H": H,
                    "aspect_class": aspect_class(W, H), "photo": has_photo(row, F), "n_el": row["length"],
                    "sig": text_signature(row, F)}
            try:
                rec, issues, reason = check_one(row, F, "cr2scan", folder)
            except Exception as exc:  # a converter bug must not sink the scan
                rec, issues, reason = None, [], f"error:{type(exc).__name__}"
            meta.update(kept=reason is None, reason=reason, issue=issues[0] if issues else None,
                        n_out=len(rec["source"]["elements"]) if rec else None)
            out.append(meta)
            for p in folder.glob("*"):
                p.unlink()
    return out


def scan(workers: int) -> list[dict]:
    jobs = row_groups()
    rows = []
    if workers <= 1:
        for j in jobs:
            rows += scan_group(j)
            print(f"[scan] {Path(j[0]).name}#{j[1]}: {len(rows)} ad templates so far", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, 4)) as ex:
            for j, part in zip(jobs, ex.map(scan_group, jobs)):
                rows += part
                print(f"[scan] {Path(j[0]).name}#{j[1]}: {len(rows)} ad templates so far", flush=True)
    return rows


# ---------------------------------------------------------------- pass 2: select and write

def _rank(crello_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{crello_id}".encode()).hexdigest()


def select(scan_rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified pick of kept templates: CLASS_QUOTA (scaled to n) per aspect
    class, half text-only where available, sizes round-robin, at most one
    template per design family (no shared text of 8+ characters)."""
    kept, seen = [], set()
    for r in sorted((r for r in scan_rows if r["kept"]), key=lambda r: _rank(r["crello_id"], seed)):
        if not seen & set(r["sig"]):  # one template per design family
            seen |= set(r["sig"])
            kept.append(r)
    total = sum(CLASS_QUOTA.values())
    quota = {c: max(1, round(q * n / total)) for c, q in CLASS_QUOTA.items()}
    picked = []
    for cls, q in quota.items():
        pools = {p: [r for r in kept if r["aspect_class"] == cls and r["photo"] == p] for p in (True, False)}
        want_txt = min(q // 2, len(pools[False]))
        want = {False: want_txt, True: min(q - want_txt, len(pools[True]))}
        if want[True] + want[False] < q:  # not enough photo ads: top up with text-only
            want[False] = min(len(pools[False]), q - want[True])
        for p, k in want.items():
            by_size = defaultdict(list)
            for r in sorted(pools[p], key=lambda r: _rank(r["crello_id"], seed)):
                by_size[(r["W"], r["H"])].append(r)
            sizes = sorted(by_size, key=lambda s: _rank(f"{s}", seed))
            chosen = []
            while len(chosen) < k and any(by_size.values()):
                for s in sizes:
                    if by_size[s] and len(chosen) < k:
                        chosen.append(by_size[s].pop(0))
            picked += chosen
    return picked


def task_id_for(k: int, r: dict) -> str:
    return f"{PREFIX}_{k:02d}_{r['W']}x{r['H']}_{'img' if r['photo'] else 'txt'}"


def write_selected(picked: list[dict]) -> list[dict]:
    """Convert the picked templates for real; returns the records written."""
    F = features()
    CHALLENGE2_DIR.mkdir(parents=True, exist_ok=True)
    if CHALLENGE2_ASSETS_DIR.exists():
        shutil.rmtree(CHALLENGE2_ASSETS_DIR)
    CHALLENGE2_ASSETS_DIR.mkdir(parents=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for p in list(CHALLENGE2_DIR.glob(f"{PREFIX}_*.json")) + list(PREVIEW_DIR.glob("*.png")):
        p.unlink()
    order = sorted(picked, key=lambda r: (list(CLASS_QUOTA).index(r["aspect_class"]), not r["photo"], r["W"] / r["H"],
                                          r["crello_id"]))
    ids = {r["crello_id"]: task_id_for(k + 1, r) for k, r in enumerate(order)}
    by_group = defaultdict(set)
    for r in order:
        by_group[(r["shard"], r["group"])].add(r["crello_id"])
    from dev.distill.make_challenge import quality_issues
    from dev.distill.make_corpus import measurer
    records = []
    for (shard, group), want in sorted(by_group.items()):
        for row in read_group(str(SHARDS_DIR / shard), group, want):
            tid = ids[row["id"]]
            rec = convert(row, F, tid, CHALLENGE2_ASSETS_DIR, measurer())
            Image.open(io.BytesIO(row["preview"]["bytes"])).convert("RGB").save(PREVIEW_DIR / f"{tid}.png")
            records.append(rec)
    records.sort(key=lambda r: r["task_id"])
    with merged_assets_dir(*CHALLENGE2_FOLDERS) as assets:
        for rec in records:
            issues = quality_issues(rec, assets)
            assert not issues, (rec["task_id"], issues)
            (CHALLENGE2_DIR / f"{rec['task_id']}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        review_sheet(records, assets)
    return records


def review_sheet(records: list[dict], assets, path: Path = SHEET, cell: int = 300, per_row: int = 4) -> list[Path]:
    """Crello's preview (left) next to our render (right) for every ad; also
    split into sheet_partN.png pages of 12 ads for close review."""
    from engine.pipeline.render import render_to_image

    tiles = []
    for rec in records:
        ours = render_to_image(rec["source"], assets_dir=assets, max_size=cell)
        theirs = Image.open(PREVIEW_DIR / f"{rec['task_id']}.png").convert("RGB")
        theirs.thumbnail((cell, cell))
        tile = Image.new("RGB", (2 * cell + 30, cell + 26), (236, 236, 236))
        tile.paste(theirs, ((cell - theirs.width) // 2 + 5, 22 + (cell - theirs.height) // 2))
        tile.paste(ours, (cell + 25 + (cell - ours.width) // 2, 22 + (cell - ours.height) // 2))
        c = rec["source"]["canvas"]
        ImageDraw.Draw(tile).text((6, 5), f"{rec['task_id']}  {c['width']}x{c['height']}  n={len(rec['source']['elements'])}"
                                  "   [Crello preview | ours]", fill=(20, 20, 20))
        tiles.append(tile)

    def grid(ts, out):
        rows = -(-len(ts) // per_row)
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (per_row * (tw + 10), rows * (th + 10)), (255, 255, 255))
        for k, t in enumerate(ts):
            sheet.paste(t, ((k % per_row) * (tw + 10), (k // per_row) * (th + 10)))
        sheet.save(out)
        return out

    outs = [grid(tiles, path)]
    for p in path.parent.glob("sheet_part*.png"):
        p.unlink()
    for k in range(0, len(tiles), 12):
        outs.append(grid(tiles[k:k + 12], path.parent / f"sheet_part{k // 12 + 1}.png"))
    return outs


def summarise(scan_rows: list[dict], records: list[dict]) -> dict:
    """Aggregate numbers only (no Crello content): what CHALLENGE2.md quotes."""
    reasons = Counter(r["reason"] for r in scan_rows if not r["kept"])
    kept = [r for r in scan_rows if r["kept"]]
    cls_of = lambda rs, key: dict(sorted(Counter(key(r) for r in rs).items()))
    els = [len(r["source"]["elements"]) for r in records]
    types = Counter(e["type"] for r in records for e in r["source"]["elements"])
    return {"source": "cyberagent/crello rev 5.1.0, 4 test shards, ad formats (crello_stats.AD_FORMATS)",
            "ad_templates_scanned": len(scan_rows), "kept_by_gate": len(kept),
            "skipped": sum(reasons.values()), "skip_reasons": dict(reasons.most_common()),
            "scanned_by_class": cls_of(scan_rows, lambda r: r["aspect_class"]),
            "kept_by_class": cls_of(kept, lambda r: r["aspect_class"]),
            "kept_by_class_photo": cls_of(kept, lambda r: f"{r['aspect_class']}/{'photo' if r['photo'] else 'text-only'}"),
            "kept_share_photo": round(sum(r["photo"] for r in kept) / max(1, len(kept)), 3),
            "scanned_share_photo": round(sum(r["photo"] for r in scan_rows) / max(1, len(scan_rows)), 3),
            "selected": len(records),
            "selected_by_class": cls_of(records, lambda r: r["features"][0]),
            "selected_by_photo": cls_of(records, lambda r: r["features"][1]),
            "selected_sizes": dict(sorted(Counter(f"{r['source']['canvas']['width']}x{r['source']['canvas']['height']}"
                                                  for r in records).items())),
            "selected_elements": {"min": min(els, default=0), "median": sorted(els)[len(els) // 2] if els else 0,
                                  "max": max(els, default=0), "by_type": dict(types)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rescan", action="store_true", help="redo pass 1 even if scan.jsonl exists")
    args = ap.parse_args(argv)
    CHALLENGE2_DIR.mkdir(parents=True, exist_ok=True)
    if args.rescan or not SCAN_FILE.exists():
        rows = scan(args.workers)
        SCAN_FILE.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rows = [json.loads(x) for x in SCAN_FILE.read_text().splitlines() if x.strip()]
    picked = select(rows, args.n, args.seed)
    records = write_selected(picked)
    summary = summarise(rows, records)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {len(records)} ads to {CHALLENGE2_DIR}; review sheet {SHEET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
