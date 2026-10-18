"""Turn placed geometry into a final design (docs/pipeline/02-resize-engine.md §7)."""

import copy
import math

from engine.pipeline.geometry import inside_canvas, rotated_aabb

EDITABLE = {"x", "y", "width", "height", "font_size", "letter_spacing", "corner_radius", "stroke_width"}

# Rounding to 2 decimals must never make text wrap: box sizes round up, text
# sizes round down, positions and decorative sizes round to nearest.
_ROUND_UP = {"width", "height"}
_ROUND_DOWN = {"font_size", "letter_spacing"}


def _round(key: str, value: float) -> float:
    if key in _ROUND_UP:
        return math.ceil(value * 100 - 1e-6) / 100
    if key in _ROUND_DOWN:
        return math.floor(value * 100 + 1e-6) / 100
    return round(value, 2)


def finalize(source: dict, placed: dict[str, dict], target: dict) -> dict | None:
    """Deep-copy each source element (source order) and overwrite only the
    EDITABLE keys from `placed`. Returns None if an element wasn't placed or
    ends up outside the canvas (rotated bounds, 0.5 px tolerance); squashing
    it to fit would hide a layout bug."""
    width, height = int(target["width"]), int(target["height"])
    elements = []
    for src_el in source["elements"]:
        geometry = placed.get(src_el["id"])
        if geometry is None:
            return None
        el = copy.deepcopy(src_el)
        for key in EDITABLE & geometry.keys():
            el[key] = _round(key, geometry[key])
        if el["type"] == "image" and el.get("fit") != "cover" and geometry.get("width") and geometry.get("height"):
            # Rounding both sides would bend a tiny image's proportions (0.375 x 0.1125
            # -> 0.38 x 0.12 is 5% off): derive the height from the rounded width.
            el["height"] = round(el["width"] * geometry["height"] / geometry["width"], 4)
        if not inside_canvas(rotated_aabb(el), width, height):
            return None
        elements.append(el)
    return {"canvas": {"width": width, "height": height}, "elements": elements}
