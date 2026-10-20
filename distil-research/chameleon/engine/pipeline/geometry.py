"""Box helpers shared by all three nodes. A `Box` is (x, y, w, h) in canvas px."""

import math

Box = tuple[float, float, float, float]


def element_box(el: dict) -> Box:
    """The element's unrotated box."""
    return (el["x"], el["y"], el["width"], el["height"])


def rotated_aabb(el: dict) -> Box:
    """Axis-aligned bounds of the element after rotating it (degrees,
    clockwise) about its centre, which is how the renderer draws it."""
    x, y, w, h = element_box(el)
    angle = el.get("rotation") or 0
    if angle == 0:
        return (x, y, w, h)
    rad = math.radians(angle)
    cos, sin = abs(math.cos(rad)), abs(math.sin(rad))
    rw, rh = w * cos + h * sin, w * sin + h * cos
    cx, cy = x + w / 2, y + h / 2
    return (cx - rw / 2, cy - rh / 2, rw, rh)


def overlap_area(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = min(ax + aw, bx + bw) - max(ax, bx)
    dy = min(ay + ah, by + bh) - max(ay, by)
    return max(0, dx) * max(0, dy)


def union_box(boxes: list[Box]) -> Box:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def inside_canvas(box: Box, width: float, height: float, tol: float = 0.5) -> bool:
    """Same tolerance as engine/main.py's self-check."""
    x, y, w, h = box
    return x >= -tol and y >= -tol and x + w <= width + tol and y + h <= height + tol


def align_in(slot: Box, w: float, h: float, align: str = "center") -> tuple[float, float]:
    """Top-left corner for a w x h box inside `slot`: left/right set the
    horizontal edge, top/bottom the vertical one; otherwise it's centred."""
    sx, sy, sw, sh = slot
    x = sx if align == "left" else sx + sw - w if align == "right" else sx + (sw - w) / 2
    y = sy if align == "top" else sy + sh - h if align == "bottom" else sy + (sh - h) / 2
    return x, y


def fit_aspect(aspect: float, slot: Box, align: str = "center") -> Box:
    """Largest box with width/height == `aspect` inside `slot`, aligned by `align_in`."""
    _, _, sw, sh = slot
    if sw / sh > aspect:
        w, h = sh * aspect, sh
    else:
        w, h = sw, sw / aspect
    return (*align_in(slot, w, h, align), w, h)
