"""The design rulebook (docs/design-rulebook.md) as a penalty function.

Rules were mined from 1,955 teacher critiques and calibrated against the
teacher's 1-4 scores (dev/rule_repair/mine.py). `measure` returns the raw,
checkable measurements; `penalty` turns them into one number (0 = no rule
broken, higher = worse) plus a per-rule breakdown. Both never raise.

Units are the plan's content units (background and decoration excluded).
A unit's box is the union of its members' drawn boxes; a text's box is its
ink (widest line placed per text_align, measured block), not the slot it was
given. The teacher sees renders downscaled so the long side is 768 px, so
sizes are judged in rendered px.
"""

import math

import numpy as np

from engine.pipeline.geometry import Box, overlap_area, rotated_aabb, union_box
from engine.pipeline.layout.place import floor_px
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import PriorityPlan, Unit

RENDER_LONG_SIDE = 768    # the teacher's render size (dev/distill/pool.py RENDER_SIZE)
ROW_BAND = 0.1            # reading order: centres this close (share of height) read as one row
FULL_BLEED = 0.9          # a unit spanning this share of a side is full-bleed: no edge margin asked
EMPTY_GRID = 48           # largest-empty-rectangle grid: cells along the long side

# Thresholds (calibrated, see docs/design-rulebook.md). Sizes in rendered px.
TEXT_PX = {"headline": 28.0, "subheadline": 13.0, "cta": 14.5, "badge": 8.0, "legal": 6.0,
           "other_text": 11.0}
DEFAULT_TEXT_PX = 10.0
EMPTY_MAX = 0.22          # largest empty rectangle above this share of the canvas = dead space
BBOX_MIN = 0.55           # content bounding box below this share of the canvas = looks shrunk
EDGE_MIN = 0.05           # content closer than this share of the short side to an edge
GAP_MIN = 0.01            # different-role neighbours closer than this share of the short side = cramped
GAP_CV_MAX = 0.9          # nearest-neighbour gaps more uneven than this (coefficient of variation)
LOGO_PX = 24.5            # logo's short side below this many rendered px = speck
IMAGE_MIN = 0.12          # the largest product/hero image below this share of the canvas = shrunk to an icon
ORPHAN_SHARE = 0.35       # last line holding one word and under this share of the widest line

# Rule weights (critique frequency x calibration agreement, docs/design-rulebook.md).
WEIGHTS = {
    "R1_text_floor": 3.0,
    "R2_dead_space": 2.0,
    "R3_cramped": 1.0,
    "R4_edge_margin": 1.0,
    "R5_logo": 1.0,
    "R6_looks_shrunk": 0.5,
    "R7_uneven_gaps": 1.0,
    "R8_reading_order": 2.0,
    "R9_hierarchy": 1.0,
    "R10_collision": 2.0,
    "R11_orphan_word": 0.5,
    "R12_alignment": 0.5,
    "R13_image_small": 1.0,
}

# name -> True if a higher value is better (used by the calibration script).
MEASURES = {
    "text_floor_share": True, "text_floor_ratio_min": True, "text_px_min": True,
    "px_headline": True, "px_subheadline": True, "px_cta": True, "px_badge": True, "px_legal": True,
    "px_other_text": True, "headline_rel": True,
    "empty_share": False, "content_bbox_share": True, "content_fill": True,
    "edge_min": True, "edge_text_min": True,
    "logo_short_px": True, "logo_share": True,
    "gap_min": True, "gap_cross_min": True, "gap_max": False, "gap_cv": False,
    "order_agreement": True, "size_order_agreement": True, "new_overlap": False,
    "orphans": False, "alignment": True, "image_share": True,
}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def render_scale(width: float, height: float) -> float:
    return min(1.0, RENDER_LONG_SIDE / max(width, height))


def ink_box(el: dict, measurer: TextMeasurer) -> Box:
    """Where a text's glyphs are drawn (its slot can be much wider)."""
    if el.get("type") != "text" or el.get("rotation"):
        return rotated_aabb(el)
    try:
        width = min(max(measurer.line_widths(el, el["font_size"], el["width"]), default=0.0), el["width"])
        block = min(el["height"], measurer.block_height(el, el["font_size"], el["width"]))
    except Exception:
        return rotated_aabb(el)
    align = el.get("text_align") or "left"
    x = el["x"] + (el["width"] - width) / 2 if align == "center" else \
        el["x"] + el["width"] - width if align == "right" else el["x"]
    return (x, el["y"], width, block)


def content_units(plan: PriorityPlan, els: dict[str, dict]) -> list[Unit]:
    return [u for u in plan.ranked_units() if u.element_ids and all(i in els for i in u.element_ids)]


def unit_boxes(units: list[Unit], els: dict[str, dict], measurer: TextMeasurer) -> dict[str, Box]:
    return {u.unit_id: union_box([ink_box(els[i], measurer) for i in u.element_ids]) for u in units}


def largest_empty_share(boxes: list[Box], width: float, height: float) -> float:
    """Area share of the largest axis-aligned rectangle touching no box, on a
    coarse grid (largest rectangle in a histogram, row by row)."""
    if width <= 0 or height <= 0:
        return 0.0
    cell = min(max(width, height) / EMPTY_GRID, min(width, height) / 8)
    cols, rows = max(1, int(round(width / cell))), max(1, int(round(height / cell)))
    cw, ch = width / cols, height / rows
    busy = np.zeros((rows, cols), dtype=bool)
    for x, y, w, h in boxes:
        c0, c1 = int(max(0, math.floor(x / cw))), int(min(cols, math.ceil((x + w) / cw)))
        r0, r1 = int(max(0, math.floor(y / ch))), int(min(rows, math.ceil((y + h) / ch)))
        if c1 > c0 and r1 > r0:
            busy[r0:r1, c0:c1] = True
    best = 0
    heights = np.zeros(cols, dtype=int)
    for r in range(rows):
        heights = np.where(busy[r], 0, heights + 1)
        hs = heights.tolist() + [0]
        stack: list[int] = []
        for c, h in enumerate(hs):
            while stack and hs[stack[-1]] >= h:
                top = hs[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                best = max(best, top * (c - left))
            stack.append(c)
    return best / (rows * cols)


def _gap(a: Box, b: Box) -> float:
    dx = max(0.0, b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]))
    dy = max(0.0, b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]))
    return math.hypot(dx, dy)


def _precedes(a: Box, b: Box, h: float) -> bool:
    ay, by = a[1] + a[3] / 2, b[1] + b[3] / 2
    if abs(ay - by) > ROW_BAND * h:
        return ay < by
    return a[0] + a[2] / 2 < b[0] + b[2] / 2


def order_pairs(units: list[Unit], boxes: dict[str, Box], height: float) -> dict[tuple[str, str], bool]:
    return {(a.unit_id, b.unit_id): _precedes(boxes[a.unit_id], boxes[b.unit_id], height)
            for i, a in enumerate(units) for b in units[i + 1:]}


def _orphaned(el: dict, measurer: TextMeasurer) -> bool:
    """Wrapped onto 2+ lines with a single short word alone on the last."""
    try:
        wrapped = measurer.wrap(el, el["font_size"], el["width"])
        lines = wrapped.split("\n")
        if len(lines) < 2 or len(lines[-1].split()) != 1:
            return False
        widths = measurer.line_widths(el, el["font_size"], el["width"])
        return widths[-1] < ORPHAN_SHARE * max(widths)
    except Exception:
        return False


def _text_sizes(els: dict[str, dict], unit: Unit) -> list[float]:
    return [els[i]["font_size"] for i in unit.element_ids if els[i].get("type") == "text"]


def _is_full_bleed(box: Box, width: float, height: float) -> bool:
    return box[2] >= FULL_BLEED * width or box[3] >= FULL_BLEED * height


def _edge(box: Box, width: float, height: float) -> float:
    x, y, w, h = box
    return min(x, y, width - x - w, height - y - h)


def measure(source: dict, design: dict, plan: PriorityPlan, measurer: TextMeasurer) -> dict[str, float | None]:
    """Raw rule measurements of `design` (see MEASURES); {} if it can't be read."""
    try:
        return _measure(source, design, plan, measurer)
    except Exception:
        return {}


def _measure(source, design, plan, measurer) -> dict[str, float | None]:
    width, height = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    short = min(width, height)
    scale = render_scale(width, height)
    els = {e["id"]: e for e in design["elements"]}
    src_els = {e["id"]: e for e in source["elements"]}
    units = content_units(plan, els)
    boxes = unit_boxes(units, els, measurer)
    src_boxes = unit_boxes(units, src_els, measurer)
    out: dict[str, float | None] = {k: None for k in MEASURES}

    # Text size.
    text_units = [u for u in units if _text_sizes(els, u)]
    if text_units:
        at_floor = [min(_text_sizes(els, u)) >= floor_px(u.role, width, height) - 1e-6 for u in text_units]
        out["text_floor_share"] = sum(at_floor) / len(at_floor)
        out["text_floor_ratio_min"] = min(min(_text_sizes(els, u)) / floor_px(u.role, width, height)
                                          for u in text_units)
        out["text_px_min"] = min(min(_text_sizes(els, u)) for u in text_units) * scale
        for role in ("headline", "subheadline", "cta", "badge", "legal", "other_text"):
            sizes = [min(_text_sizes(els, u)) for u in text_units if u.role == role]
            if sizes:
                out[f"px_{role}"] = min(sizes) * scale
        heads = [max(_text_sizes(els, u)) for u in text_units if u.role == "headline"]
        if heads:
            out["headline_rel"] = max(heads) / short

    # Space.
    if boxes:
        out["empty_share"] = largest_empty_share(list(boxes.values()), width, height)
        bx = union_box(list(boxes.values()))
        out["content_bbox_share"] = overlap_area(bx, (0, 0, width, height)) / (width * height)
        out["content_fill"] = min(1.0, sum(b[2] * b[3] for b in boxes.values()) / (width * height))
        edges = [_edge(b, width, height) / short for b in boxes.values() if not _is_full_bleed(b, width, height)]
        out["edge_min"] = max(0.0, min(edges)) if edges else None
        tedges = [_edge(boxes[u.unit_id], width, height) / short for u in text_units
                  if not _is_full_bleed(boxes[u.unit_id], width, height)]
        out["edge_text_min"] = max(0.0, min(tedges)) if tedges else None
        images = [b[2] * b[3] / (width * height) for u, b in ((u, boxes[u.unit_id]) for u in units)
                  if u.role in ("hero_image", "secondary_image")]
        out["image_share"] = max(images) if images else None

    logos = [boxes[u.unit_id] for u in units if u.role == "logo"]
    if logos:
        out["logo_short_px"] = min(min(b[2], b[3]) for b in logos) * scale
        out["logo_share"] = min(b[2] * b[3] for b in logos) / (width * height)

    # Neighbours: nearest-neighbour gap per unit (units the source already
    # overlapped, like text on a photo, don't count as cramped).
    if len(units) >= 2:
        nn = []
        for u in units:
            gaps = [_gap(boxes[u.unit_id], boxes[v.unit_id]) for v in units if v is not u
                    and overlap_area(src_boxes[u.unit_id], src_boxes[v.unit_id]) <= 0]
            if gaps:
                nn.append(min(gaps) / short)
        # Cramped: the closest pair of different roles (a row of same-role
        # chips may sit tight; a headline running into its button may not).
        cross = [_gap(boxes[a.unit_id], boxes[b.unit_id]) / short for i, a in enumerate(units) for b in units[i + 1:]
                 if a.role != b.role and overlap_area(src_boxes[a.unit_id], src_boxes[b.unit_id]) <= 0]
        if cross:
            out["gap_cross_min"] = min(cross)
        if nn:
            out["gap_min"] = min(nn)
            out["gap_max"] = max(nn)
            mean = sum(nn) / len(nn)
            out["gap_cv"] = (math.sqrt(sum((g - mean) ** 2 for g in nn) / len(nn)) / mean) if mean > 0 else 0.0

        src_h = float(source["canvas"]["height"])
        now, was = order_pairs(units, boxes, height), order_pairs(units, src_boxes, src_h)
        out["order_agreement"] = sum(now[k] == was[k] for k in now) / len(now)
        size_pairs = []
        for i, a in enumerate(text_units):
            for b in text_units[i + 1:]:
                sa, sb = max(_text_sizes(src_els, a)), max(_text_sizes(src_els, b))
                if sa != sb:
                    oa, ob = max(_text_sizes(els, a)), max(_text_sizes(els, b))
                    size_pairs.append(1.0 if (oa - ob) * (sa - sb) > 0 else 0.5 if oa == ob else 0.0)
        out["size_order_agreement"] = sum(size_pairs) / len(size_pairs) if size_pairs else None
        extra = 0.0
        for i, a in enumerate(units):
            for b in units[i + 1:]:
                ba, bb = boxes[a.unit_id], boxes[b.unit_id]
                small = min(ba[2] * ba[3], bb[2] * bb[3])
                if small <= 0:
                    continue
                sa, sb = src_boxes[a.unit_id], src_boxes[b.unit_id]
                ssmall = min(sa[2] * sa[3], sb[2] * sb[3])
                before = overlap_area(sa, sb) / ssmall if ssmall > 0 else 0.0
                extra += max(0.0, overlap_area(ba, bb) / small - before)
        out["new_overlap"] = extra
        tol = 0.01 * max(width, height)
        lines = {k: (x, x + w / 2, x + w, y, y + h / 2, y + h) for k, (x, y, w, h) in boxes.items()}
        out["alignment"] = sum(any(abs(lines[a][i] - lines[b][i]) <= tol for b in lines if b != a for i in range(6))
                               for a in lines) / len(lines)

    texts = [e for e in design["elements"] if e.get("type") == "text"]
    out["orphans"] = float(sum(_orphaned(e, measurer) and not _orphaned(src_els.get(e["id"], e), measurer)
                               for e in texts))
    return {k: (None if v is None else float(v)) for k, v in out.items()}


def penalty(source: dict, design: dict, plan: PriorityPlan, measurer: TextMeasurer,
            weights: dict[str, float] | None = None) -> tuple[float, dict[str, float]]:
    """(total, {rule: weighted penalty}); never raises (an unreadable design
    scores a large constant)."""
    try:
        parts = _penalty_parts(source, design, plan, measurer)
    except Exception:
        return 1e6, {r: 0.0 for r in WEIGHTS}
    weights = WEIGHTS if weights is None else weights
    weighted = {r: weights.get(r, 0.0) * parts.get(r, 0.0) for r in WEIGHTS}
    return sum(weighted.values()), weighted


def penalty_parts(source: dict, design: dict, plan: PriorityPlan, measurer: TextMeasurer) -> dict[str, float]:
    """Each rule's unweighted penalty in [0, 1]; {} if the design can't be read."""
    try:
        return _penalty_parts(source, design, plan, measurer)
    except Exception:
        return {}


def _penalty_parts(source, design, plan, measurer) -> dict[str, float]:
    width, height = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    short = min(width, height)
    scale = render_scale(width, height)
    els = {e["id"]: e for e in design["elements"]}
    units = content_units(plan, els)
    boxes = unit_boxes(units, els, measurer)
    m = _measure(source, design, plan, measurer)
    weights = {u.unit_id: plan.weight(u.unit_id) for u in units}
    total_w = sum(weights.values()) or 1.0
    parts: dict[str, float] = {}

    # R1: every text at its role's legible size (rendered px, and the engine's floor).
    r1 = 0.0
    for u in units:
        sizes = _text_sizes(els, u)
        if not sizes:
            continue
        need = max(TEXT_PX.get(u.role, DEFAULT_TEXT_PX) / scale, floor_px(u.role, width, height))
        r1 += weights[u.unit_id] * _clamp(1 - min(sizes) / need) ** 0.5
    parts["R1_text_floor"] = r1 / total_w

    parts["R2_dead_space"] = _clamp((m["empty_share"] - EMPTY_MAX) / (1 - EMPTY_MAX)) if m["empty_share"] is not None else 0.0

    # R3: cramped neighbours.
    parts["R3_cramped"] = _clamp(1 - m["gap_cross_min"] / GAP_MIN) if m["gap_cross_min"] is not None else 0.0

    # R4: margin to the canvas edge, per unit, weighted by priority. A unit
    # the source designer set closer to its edge is held only to that.
    src_els = {e["id"]: e for e in source["elements"]}
    src_boxes = unit_boxes(units, src_els, measurer)
    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    r4 = 0.0
    for u in units:
        b = boxes[u.unit_id]
        if _is_full_bleed(b, width, height):
            continue
        need = min(EDGE_MIN, max(0.0, _edge(src_boxes[u.unit_id], sw, sh)) / min(sw, sh)) * short
        if need <= 0:
            continue
        r4 += weights[u.unit_id] * _clamp(1 - max(0.0, _edge(b, width, height)) / need)
    parts["R4_edge_margin"] = r4 / total_w

    parts["R5_logo"] = _clamp(1 - m["logo_short_px"] / LOGO_PX) if m["logo_short_px"] is not None else 0.0
    parts["R6_looks_shrunk"] = _clamp(1 - m["content_bbox_share"] / BBOX_MIN) if m["content_bbox_share"] is not None else 0.0
    parts["R7_uneven_gaps"] = _clamp((m["gap_cv"] - GAP_CV_MAX) / GAP_CV_MAX) if m["gap_cv"] is not None else 0.0
    parts["R8_reading_order"] = 1 - m["order_agreement"] if m["order_agreement"] is not None else 0.0
    parts["R9_hierarchy"] = 1 - m["size_order_agreement"] if m["size_order_agreement"] is not None else 0.0
    parts["R10_collision"] = min(1.0, m["new_overlap"]) if m["new_overlap"] is not None else 0.0
    parts["R11_orphan_word"] = min(1.0, m["orphans"] / 2) if m["orphans"] is not None else 0.0
    parts["R12_alignment"] = 1 - m["alignment"] if m["alignment"] is not None else 0.0
    parts["R13_image_small"] = _clamp(1 - m["image_share"] / IMAGE_MIN) if m["image_share"] is not None else 0.0
    return parts
