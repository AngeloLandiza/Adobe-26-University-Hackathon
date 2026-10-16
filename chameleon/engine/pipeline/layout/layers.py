"""Background and decoration layers (docs/pipeline/02-resize-engine.md §2),
plus the helpers that keep elements inside the canvas."""

from engine.pipeline.geometry import fit_aspect, rotated_aabb, union_box

_SIZE_KEYS = ("font_size", "letter_spacing", "corner_radius", "stroke_width")


def scale_about(el: dict, factor: float, cx: float, cy: float) -> dict:
    """`el` scaled by `factor` about the point (cx, cy): position, size and the
    size-derived fields. A font never drops below 1 px (the schema minimum)."""
    new = dict(el, x=cx + (el["x"] - cx) * factor, y=cy + (el["y"] - cy) * factor,
               width=el["width"] * factor, height=el["height"] * factor)
    for key in _SIZE_KEYS:
        if el.get(key):
            new[key] = el[key] * factor
    if "font_size" in el:
        new["font_size"] = max(1.0, new["font_size"])
    return new


def fit_inside(elements: list[dict], width: float, height: float) -> list[dict]:
    """Shrink (about their joint centre) and then shift a set of elements
    together, so their rotated bounds sit inside the canvas. Moving a unit's
    members together keeps a label inside its button."""
    x, y, w, h = union_box([rotated_aabb(e) for e in elements])
    factor = min(1.0, width / w if w else 1.0, height / h if h else 1.0)
    if factor < 1:
        elements = [scale_about(e, factor, x + w / 2, y + h / 2) for e in elements]
        x, y, w, h = union_box([rotated_aabb(e) for e in elements])
    dx = max(0.0, -x) - max(0.0, x + w - width)
    dy = max(0.0, -y) - max(0.0, y + h - height)
    return [dict(e, x=e["x"] + dx, y=e["y"] + dy) for e in elements]


def place_background(el: dict, width: float, height: float) -> dict:
    """Fill the canvas. Only shapes and `cover` images may change shape to do
    it (a cover image crops); any other image keeps its aspect and is fitted
    inside, since the aspect rule applies to it."""
    if el["type"] == "image" and el.get("fit", "contain") != "cover":
        x, y, w, h = fit_aspect(el["width"] / el["height"], (0, 0, width, height))
        out = dict(el, x=x, y=y, width=w, height=h)
    else:
        out = dict(el, x=0, y=0, width=width, height=height)
    return fit_inside([out], width, height)[0]


def place_decoration(el: dict, src_canvas: dict, width: float, height: float) -> dict:
    """Scale uniformly by s = min(W/Ws, H/Hs), keep the gap to the source's
    nearest corner (scaled by s), then pull it fully inside the canvas."""
    src_w, src_h = src_canvas["width"], src_canvas["height"]
    s = min(width / src_w, height / src_h)
    new = scale_about(el, s, 0, 0)
    cx, cy = el["x"] + el["width"] / 2, el["y"] + el["height"] / 2
    new["x"] = el["x"] * s if cx < src_w / 2 else width - (src_w - el["x"]) * s
    new["y"] = el["y"] * s if cy < src_h / 2 else height - (src_h - el["y"]) * s
    return fit_inside([new], width, height)[0]
