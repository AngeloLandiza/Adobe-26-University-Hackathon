"""Group a design's elements into units, the pieces the layout moves as one,
and attach the hard signals the LLM can't override
(docs/pipeline/01-priority-node.md §1-2)."""

from pathlib import Path

from engine.pipeline.geometry import Box, element_box, overlap_area
from engine.pipeline.types import Unit
from renderer.converter.assets import asset_path

INSIDE_FRACTION = 0.9      # a label's box must lie at least this much inside its shape
MIN_LABEL_COVERAGE = 0.25  # an implicit label must cover this much of its shape; panels behind copy don't
BACKGROUND_COVERAGE = 0.9  # share of the canvas a background must cover


def _z(el: dict) -> float:
    return el.get("z_index") or 0


def _area(box: Box) -> float:
    return box[2] * box[3]


def _inside_fraction(inner: dict, outer: dict) -> float:
    area = _area(element_box(inner))
    return overlap_area(element_box(inner), element_box(outer)) / area if area > 0 else 0.0


def _group_kind(members: list[dict]) -> str:
    shapes = [m for m in members if m["type"] == "shape"]
    texts = [m for m in members if m["type"] == "text"]
    if len(members) == 2 and len(shapes) == 1 and len(texts) == 1 and _inside_fraction(texts[0], shapes[0]) >= INSIDE_FRACTION:
        return "container_label"
    return "group"


def _implicit_labels(elements: list[dict]) -> dict[str, str]:
    """shape id -> text id for ungrouped texts sitting on an ungrouped shape.
    A shape takes a label only if exactly one text sits on it and that text
    covers a real share of it, so a backdrop panel behind several lines of copy
    (or one small line) stays a separate unit."""
    order = {el["id"]: i for i, el in enumerate(elements)}
    shapes = [el for el in elements if el["type"] == "shape" and el.get("group_id") is None]
    texts = [el for el in elements if el["type"] == "text" and el.get("group_id") is None]
    on_shape: dict[str, list[dict]] = {}
    for text in texts:
        holders = [s for s in shapes if _z(s) < _z(text) and _inside_fraction(text, s) >= INSIDE_FRACTION]
        if holders:
            holder = min(holders, key=lambda s: (_area(element_box(s)), order[s["id"]]))
            on_shape.setdefault(holder["id"], []).append(text)
    by_id = {el["id"]: el for el in elements}
    return {
        shape_id: found[0]["id"]
        for shape_id, found in on_shape.items()
        if len(found) == 1 and _area(element_box(found[0])) >= MIN_LABEL_COVERAGE * _area(element_box(by_id[shape_id]))
    }


def _is_background(el: dict, elements: list[dict], width: float, height: float) -> bool:
    if el["type"] not in ("image", "shape"):
        return False
    if overlap_area(element_box(el), (0, 0, width, height)) < BACKGROUND_COVERAGE * width * height:
        return False
    box = element_box(el)
    return all(_z(o) >= _z(el) for o in elements if o is not el and overlap_area(box, element_box(o)) > 0)


def build_units(source: dict, assets_dir: str | Path = "brand_kit/assets") -> list[Unit]:
    elements = source["elements"]
    width, height = source["canvas"]["width"], source["canvas"]["height"]
    element_ids = {el["id"] for el in elements}

    groups: dict[str, list[dict]] = {}
    for el in elements:
        if el.get("group_id") is not None:
            groups.setdefault(el["group_id"], []).append(el)
    labels = _implicit_labels(elements)
    shape_of_label = {text_id: shape_id for shape_id, text_id in labels.items()}

    units: list[Unit] = []
    seen: set[str] = set()
    for el in elements:
        if el["id"] in seen:
            continue
        group_id = el.get("group_id")
        if group_id is not None:
            members = groups[group_id]
            ids = [m["id"] for m in members]
            # A group id that names some other element would give two units the same id.
            unit_id = f"{group_id}#group" if group_id in element_ids and group_id not in ids else group_id
            unit = Unit(unit_id, ids, _group_kind(members))
        elif el["id"] in labels:
            unit = Unit(el["id"], [el["id"], labels[el["id"]]], "container_label")
        elif el["id"] in shape_of_label:
            shape_id = shape_of_label[el["id"]]
            unit = Unit(shape_id, [shape_id, el["id"]], "container_label")
        else:
            unit = Unit(el["id"], [el["id"]], "single")
        seen.update(unit.element_ids)
        units.append(unit)

    by_id = {el["id"]: el for el in elements}
    for unit in units:
        members = [by_id[i] for i in unit.element_ids]
        if unit.kind == "single" and _is_background(members[0], elements, width, height):
            unit.flags.append("background")
            unit.role = "background"
        if any(m["type"] == "image" and asset_path(str(assets_dir), m.get("asset_id")) is None for m in members):
            unit.flags.append("missing_asset")
        if any(m.get("rotation") for m in members):
            unit.flags.append("rotated")
    return units
