"""Place one unit inside a slot box (docs/pipeline/02-resize-engine.md §5).

`place_unit` returns updated copies of the unit's elements: only geometry and
size-derived fields (font_size, letter_spacing, corner_radius, stroke_width)
change. Everything fits the slot including rotation, and text is fitted with
the renderer-exact TextMeasurer.
"""

import math

from engine.pipeline.geometry import Box, align_in, fit_aspect, rotated_aabb, union_box
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Unit

FLOORS = {"headline": 14, "cta": 11, "subheadline": 10, "badge": 9, "legal": 7}
DEFAULT_FLOOR = 9
HARD_FLOOR = 6            # last resort; a unit placed at it is marked degraded
MAX_TEXT_GROWTH = 1.5     # text never grows past 1.5x its source size
STEP = 0.5                # font-size search step, px
CONDENSED_PADDING = 0.8   # condensed pill: label inset per side, in label font sizes


def floor_px(role: str, width: float, height: float) -> float:
    """Readability floor for `role`, scaled up on large canvases."""
    return FLOORS.get(role, DEFAULT_FLOOR) * max(1, min(width, height) / 600)


def resize_text(el: dict, font_size: float) -> dict:
    """`el` at a new font size; letter spacing (px) scales with it."""
    out = dict(el, font_size=font_size)
    if el.get("letter_spacing"):
        out["letter_spacing"] = el["letter_spacing"] * font_size / el["font_size"]
    return out


def _fit_text(el: dict, w: float, h: float, lo: float, hi: float, measurer: TextMeasurer) -> float | None:
    """Largest size in [lo, hi] at which `el` fits w x h, re-checked with
    letter spacing scaled to that size (fit_font_size measures with the source
    spacing, which is too small once the text grows)."""
    if hi < lo:
        return None
    size = measurer.fit_font_size(el, w, h, lo, hi)
    while size is not None:
        if measurer.fits(resize_text(el, size), size, w, h):
            return size
        size = size - STEP if size - STEP >= lo else None
    return None


def _rotated_fit(aspect: float, slot: Box, angle: float, align: str) -> Box:
    """Largest box with w/h == aspect whose bounds after rotating by `angle`
    fit the slot, positioned so those rotated bounds sit where `align` says."""
    rad = math.radians(angle)
    cos, sin = abs(math.cos(rad)), abs(math.sin(rad))
    h = min(slot[2] / (aspect * cos + sin), slot[3] / (aspect * sin + cos))
    w = aspect * h
    rw, rh = w * cos + h * sin, w * sin + h * cos
    rx, ry = align_in(slot, rw, rh, align)
    return (rx + (rw - w) / 2, ry + (rh - h) / 2, w, h)


def _aspect(el: dict) -> float:
    return el["width"] / el["height"] if el["height"] else 1.0


def _place_box(el: dict, slot: Box, align: str) -> dict:
    """An image or shape on its own: keep its proportions (a non-rotated cover image fills the slot)."""
    angle = el.get("rotation") or 0
    if angle:
        x, y, w, h = _rotated_fit(_aspect(el), slot, angle, align)
    elif el["type"] == "image" and el.get("fit") == "cover":
        x, y, w, h = slot
    else:
        x, y, w, h = fit_aspect(_aspect(el), slot, align)
    return dict(el, x=x, y=y, width=w, height=h)


def _place_text(el: dict, slot: Box, measurer: TextMeasurer, floor: float, align: str) -> dict | None:
    angle = el.get("rotation") or 0
    x, y, w, h = _rotated_fit(_aspect(el), slot, angle, align) if angle else slot
    size = _fit_text(el, w, h, floor, el["font_size"] * MAX_TEXT_GROWTH, measurer)
    if size is None:
        return None
    text = resize_text(el, size)
    block = measurer.block_height(text, size, w)
    top = y if align == "top" else y + h - block if align == "bottom" else y + (h - block) / 2
    return dict(text, x=x, y=top, width=w, height=block)


def _place_rigid(members: list[dict], slot: Box, measurer: TextMeasurer, floor: float, align: str) -> dict[str, dict] | None:
    """Scale the whole unit uniformly by k (its rotated bounds fit the slot),
    then re-fit each label inside its scaled box, keeping the label's visual
    centre where the source design had it."""
    ux, uy, uw, uh = union_box([rotated_aabb(m) for m in members])
    k = min(slot[2] / uw, slot[3] / uh)
    if any(m["type"] == "text" for m in members):
        k = min(k, MAX_TEXT_GROWTH)
    ox, oy = align_in(slot, uw * k, uh * k, align)
    out = {}
    for m in members:
        new = dict(m, x=ox + (m["x"] - ux) * k, y=oy + (m["y"] - uy) * k, width=m["width"] * k, height=m["height"] * k)
        for key in ("corner_radius", "stroke_width"):
            if m.get(key):
                new[key] = m[key] * k
        if m["type"] == "text":
            size = _fit_text(m, new["width"], new["height"], floor, m["font_size"] * k, measurer)
            if size is None:
                return None
            centre = new["y"] + measurer.block_height(m, m["font_size"], m["width"]) / 2 * k
            text = resize_text(m, size)
            block = measurer.block_height(text, size, new["width"])
            new.update(font_size=size, y=centre - block / 2, height=block)
            if "letter_spacing" in text:
                new["letter_spacing"] = text["letter_spacing"]
        out[m["id"]] = new
    return out


def _place_condensed(members: list[dict], slot: Box, measurer: TextMeasurer, floor: float, align: str) -> dict[str, dict] | None:
    """Rebuild a pill around a one-line label when rigid scaling would shrink
    the label below its floor (the source label box is often far wider than
    its text). Not used for rotated pills."""
    if any(m.get("rotation") for m in members):
        return None
    shape = next(m for m in members if m["type"] == "shape")
    label = next(m for m in members if m["type"] == "text")
    unit_h = union_box([rotated_aabb(m) for m in members])[3]
    ratio = shape["height"] / label["font_size"]
    hi = label["font_size"] * min(MAX_TEXT_GROWTH, slot[3] / unit_h)
    size = math.floor(hi / STEP) * STEP
    while size >= floor:
        text = resize_text(label, size)
        text_w = max(measurer.line_widths(text, size, float("inf")))
        shape_w, shape_h = text_w + 2 * CONDENSED_PADDING * size, size * ratio
        block = measurer.block_height(text, size, text_w)
        if shape_w <= slot[2] and shape_h <= slot[3] and block <= shape_h + 1:
            sx, sy = align_in(slot, shape_w, shape_h, align)
            scale = shape_h / shape["height"]
            new_shape = dict(shape, x=sx, y=sy, width=shape_w, height=shape_h)
            for key in ("corner_radius", "stroke_width"):
                if shape.get(key):
                    new_shape[key] = shape[key] * scale
            new_label = dict(text, x=sx + CONDENSED_PADDING * size, y=sy + (shape_h - block) / 2, width=text_w, height=block)
            return {m["id"]: new_shape if m is shape else new_label for m in members}
        size -= STEP
    return None


def place_unit(unit: Unit, elements: dict[str, dict], box: Box, measurer: TextMeasurer,
               floor: float, align: str = "center") -> dict[str, dict] | None:
    """Fit `unit` into `box`. Returns {element_id: updated copy}, or None when
    it can't fit at `floor` (the caller may retry at HARD_FLOOR)."""
    members = [elements[i] for i in unit.element_ids]
    if len(members) == 1:
        el = members[0]
        if el["type"] == "text":
            placed = _place_text(el, box, measurer, floor, align)
            return None if placed is None else {el["id"]: placed}
        return {el["id"]: _place_box(el, box, align)}
    rigid = _place_rigid(members, box, measurer, floor, align)
    if rigid is not None or unit.kind != "container_label":
        return rigid
    return _place_condensed(members, box, measurer, floor, align)
