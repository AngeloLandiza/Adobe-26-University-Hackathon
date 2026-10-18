"""Whether one element hides another, as cr-renderer paints them. Shared by
node 2 (keep decorations visible) and node 3's G9 gate (nothing hidden)."""

from engine.pipeline.geometry import Box, rotated_aabb
from renderer.converter.colorutil import parse_rgba


def contains(outer: Box, inner: Box, tol: float = 0.0) -> bool:
    return (outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1]
            and inner[0] + inner[2] <= outer[0] + outer[2] + tol and inner[1] + inner[3] <= outer[1] + outer[3] + tol)


def opaque(el: dict) -> bool:
    """Can this element hide what's under it? Fully opaque images and fully
    opaque shapes only (images count as opaque even if their PNG has
    transparent areas); text never counts."""
    opacity = 1 if el.get("opacity") is None else el["opacity"]
    if opacity < 1:
        return False
    if el["type"] == "image":
        return True
    return el["type"] == "shape" and parse_rgba(el.get("fill_color"), default=(0, 0, 0, 1.0))[3] >= 1


def paint_order(elements: list[dict]) -> dict[int, int]:
    """id(element) -> paint position. cr-renderer sorts by z_index with a
    stable sort, so at equal z the element listed later is painted on top."""
    return {id(el): rank for rank, el in enumerate(sorted(elements, key=lambda e: e.get("z_index") or 0))}


def is_covered(el: dict, covers: list[dict], order: dict[int, int]) -> bool:
    """True if `el` lies fully inside one of the opaque `covers` painted after it."""
    box = rotated_aabb(el)
    return any(c is not el and order[id(c)] > order[id(el)] and opaque(c) and contains(rotated_aabb(c), box)
               for c in covers)
