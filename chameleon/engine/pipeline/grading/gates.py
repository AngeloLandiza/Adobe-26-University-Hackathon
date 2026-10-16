"""Node 3, stage 1: hard gates (docs/pipeline/03-grading-node.md §1).

A candidate that fails any gate is discarded. The gates mirror the official
structural checks plus the rules RULES.md states, with tighter tolerances.
`check_gates` never raises: geometry gates skip elements that already
failed the schema gate.
"""

import math

from engine.pipeline.geometry import element_box, inside_canvas, rotated_aabb
from engine.pipeline.layout.normalize import EDITABLE
from engine.pipeline.occlusion import contains, is_covered, paint_order
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import PriorityPlan

GATES = ("G1_ids", "G2_schema", "G3_canvas", "G4_in_canvas", "G5_aspect", "G6_frozen",
         "G7_text_fit", "G8_groups", "G9_hidden")
ASPECT_TOL = 0.02   # official limit is ~5%
GROUP_TOL = 0.5     # px

# Required fields per chameleon_design_schema_v0.9.json.
_REQUIRED_COMMON = ("id", "type", "x", "y", "width", "height")
_REQUIRED_BY_TYPE = {
    "text": ("content", "font_family", "font_size", "color"),
    "image": ("asset_id",),
    "shape": ("shape_type", "fill_color"),
}
_NUMERIC = ("x", "y", "width", "height", "rotation", "opacity", "font_size", "line_height",
            "letter_spacing", "corner_radius", "stroke_width", "z_index")
_LAYER_ROLES = ("decoration", "background")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _schema_ok(el) -> bool:
    if not isinstance(el, dict) or el.get("type") not in _REQUIRED_BY_TYPE:
        return False
    if any(key not in el for key in _REQUIRED_COMMON + _REQUIRED_BY_TYPE[el["type"]]):
        return False
    if any(key in el and el[key] is not None and not _is_number(el[key]) for key in _NUMERIC):
        return False
    if el["width"] <= 0 or el["height"] <= 0:
        return False
    return el["type"] != "text" or el["font_size"] >= 1


def _hidden_ids(elements: list[dict], plan: PriorityPlan) -> list[str]:
    """Elements fully inside an opaque, non-layer element painted after them."""
    def layer(el: dict) -> bool:
        try:
            return plan.unit_for_element(el["id"]).role in _LAYER_ROLES
        except KeyError:
            return False

    order = paint_order(elements)
    covers = [el for el in elements if not layer(el)]
    return [el["id"] for el in elements if is_covered(el, covers, order)]


def check_gates(source: dict, design: dict, target: dict, plan: PriorityPlan, measurer: TextMeasurer) -> list[str]:
    """The failing gate codes, in GATES order; empty means the candidate passes."""
    failed: set[str] = set()
    elements = design.get("elements") or [] if isinstance(design, dict) else []
    src = {e["id"]: e for e in source["elements"]}

    ids = [e.get("id") if isinstance(e, dict) else None for e in elements]
    if len(ids) != len(set(ids)) or sorted(map(str, ids)) != sorted(map(str, src)):
        failed.add("G1_ids")

    canvas = design.get("canvas") if isinstance(design, dict) else None
    width, height = target["width"], target["height"]
    if not isinstance(canvas, dict) or canvas.get("width") != width or canvas.get("height") != height:
        failed.add("G3_canvas")

    good = []
    for el in elements:
        if _schema_ok(el):
            good.append(el)
        else:
            failed.add("G2_schema")

    by_id = {}
    for el in good:
        by_id[el["id"]] = el
        if not inside_canvas(rotated_aabb(el), width, height):
            failed.add("G4_in_canvas")
        original = src.get(el["id"])
        if original is None:
            continue
        if el["type"] == "image" and el.get("fit", "contain") != "cover" and original["height"] and original["width"]:
            ratio = (el["width"] / el["height"]) / (original["width"] / original["height"])
            if abs(ratio - 1) > ASPECT_TOL:
                failed.add("G5_aspect")
        frozen = lambda e: {k: v for k, v in e.items() if k not in EDITABLE}
        if frozen(el) != frozen(original):
            failed.add("G6_frozen")
        if el["type"] == "text" and not measurer.fits(el, el["font_size"], el["width"], el["height"]):
            failed.add("G7_text_fit")

    for unit in plan.units:
        if unit.kind != "container_label" or not all(i in by_id for i in unit.element_ids):
            continue
        members = [by_id[i] for i in unit.element_ids]
        shape = next((m for m in members if m["type"] == "shape"), None)
        label = next((m for m in members if m["type"] == "text"), None)
        if shape is not None and label is not None and not contains(element_box(shape), element_box(label), GROUP_TOL):
            failed.add("G8_groups")

    if _hidden_ids(good, plan):
        failed.add("G9_hidden")
    return [g for g in GATES if g in failed]
