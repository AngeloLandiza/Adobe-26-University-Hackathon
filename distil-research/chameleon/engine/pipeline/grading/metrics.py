"""Node 3, stage 2: geometry metrics (docs/pipeline/03-grading-node.md §2).

Every metric is in [0, 1], 1 = best. They're computed over the plan's
content units (background and decoration are left out), each weighted by
its priority, `plan.weight` = rank ** -0.7. The weights are starting
guesses until calibrated against a judge.
"""

import math
from itertools import combinations

from engine.pipeline.geometry import Box, overlap_area, rotated_aabb, union_box
from engine.pipeline.layout.place import floor_px
from engine.pipeline.types import Candidate, PriorityPlan, Unit

WEIGHTS = {"hierarchy": 0.20, "legibility": 0.20, "overlap": 0.15, "degradation": 0.10, "faithfulness": 0.08,
           "space_use": 0.08, "alignment": 0.07, "spacing": 0.07, "balance": 0.05}
COMFORT = 1.6             # legibility reaches 1 at 1.6x the role's floor
BELOW_FLOOR_SHARE = 0.25  # legibility a unit can earn below its floor, in proportion to its size
SPACE_BAND = (0.45, 0.85)  # content bounding box as a share of the canvas
ALIGN_TOL = 0.005          # alignment tolerance, share of the canvas's long side
EDGE_MARGIN = 0.02         # spacing: content should sit at least this share of the short side from the edges
ROW_BAND = 0.1             # faithfulness: centres this close (share of height) read as one row


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _area(box: Box) -> float:
    return box[2] * box[3]


def _text_sizes(els: dict[str, dict], unit: Unit) -> list[float]:
    return [els[i]["font_size"] for i in unit.element_ids if els[i]["type"] == "text"]


def _weighted(pairs) -> float:
    """Weighted mean of (weight, score) pairs; 1 when there's nothing to judge."""
    pairs = list(pairs)
    total = sum(w for w, _ in pairs)
    return sum(w * s for w, s in pairs) / total if total else 1.0


def _hierarchy(units, els, boxes, w) -> float:
    """More important units should look more important. Text is compared with
    text by font size and images with images by area; text against image is
    skipped, since a hero image is normally bigger than the headline above it."""
    pairs = []
    for a, b in combinations(units, 2):  # units are in rank order, so a outranks b
        ta, tb = _text_sizes(els, a), _text_sizes(els, b)
        if ta and tb:
            pa, pb = max(ta), max(tb)
        elif not ta and not tb:
            pa, pb = _area(boxes[a.unit_id]), _area(boxes[b.unit_id])
        else:
            continue
        pairs.append((w[a.unit_id] + w[b.unit_id], 1.0 if pa > pb else 0.5 if pa == pb else 0.0))
    return _weighted(pairs)


def _legibility(units, els, w, width, height) -> float:
    """Each text unit's smallest size: from the role's floor up to COMFORT x
    the floor it scores BELOW_FLOOR_SHARE..1, and below the floor up to
    BELOW_FLOOR_SHARE in proportion to its size. On a canvas too small for
    any layout to meet the floors, 9 px text must still beat 3 px text, or
    the uniformly shrunk fallback wins on faithfulness alone."""
    def score(size: float, floor: float) -> float:
        if size < floor:
            return BELOW_FLOOR_SHARE * size / floor
        return BELOW_FLOOR_SHARE + (1 - BELOW_FLOOR_SHARE) * _clamp((size - floor) / ((COMFORT - 1) * floor))

    pairs = []
    for u in units:
        sizes = _text_sizes(els, u)
        if sizes:
            pairs.append((w[u.unit_id], score(min(sizes), floor_px(u.role, width, height))))
    return _weighted(pairs)


def _overlap(units, boxes, src_boxes, w) -> float:
    """Penalise only overlap beyond what the same two units had in the
    source: text deliberately set on a photo isn't a collision."""
    def frac(b1: Box, b2: Box) -> float:
        smaller = min(_area(b1), _area(b2))
        return overlap_area(b1, b2) / smaller if smaller > 0 else 0.0

    pairs = []
    for a, b in combinations(units, 2):
        extra = frac(boxes[a.unit_id], boxes[b.unit_id]) - frac(src_boxes[a.unit_id], src_boxes[b.unit_id])
        pairs.append((w[a.unit_id] + w[b.unit_id], 1.0 - _clamp(extra)))
    return _weighted(pairs)


def _degradation(units, els, w, flagged, width, height) -> float:
    """Weight share of units that are degraded: flagged by node 2, or with
    any text below its role's floor. The second test judges what's on the
    canvas, so the fallback and the reflow are held to the same standard."""
    def degraded(u: Unit) -> bool:
        sizes = _text_sizes(els, u)
        return u.unit_id in flagged or (bool(sizes) and min(sizes) < floor_px(u.role, width, height) - 1e-6)

    total = sum(w[u.unit_id] for u in units)
    lost = sum(w[u.unit_id] for u in units if degraded(u))
    return 1.0 - lost / total if total else 1.0


def _faithfulness(units, els, src_els, boxes, src_boxes, w, height, src_height) -> float:
    """Reading order (top-to-bottom, then left-to-right within a row) and the
    font-size order among texts, both compared with the source."""
    def precedes(a: Box, b: Box, h: float) -> bool:
        ay, by = a[1] + a[3] / 2, b[1] + b[3] / 2
        if abs(ay - by) > ROW_BAND * h:
            return ay < by
        return a[0] + a[2] / 2 < b[0] + b[2] / 2

    order, sizes = [], []
    for a, b in combinations(units, 2):
        pw = w[a.unit_id] + w[b.unit_id]
        same = precedes(src_boxes[a.unit_id], src_boxes[b.unit_id], src_height) == precedes(
            boxes[a.unit_id], boxes[b.unit_id], height)
        order.append((pw, 1.0 if same else 0.0))
        sa, sb = _text_sizes(src_els, a), _text_sizes(src_els, b)
        if sa and sb and max(sa) != max(sb):
            oa, ob = max(_text_sizes(els, a)), max(_text_sizes(els, b))
            agree = (oa - ob) * (max(sa) - max(sb))
            sizes.append((pw, 1.0 if agree > 0 else 0.5 if agree == 0 else 0.0))
    parts = [_weighted(p) for p in (order, sizes) if p]
    return sum(parts) / len(parts) if parts else 1.0


def _space_use(boxes, width, height) -> float:
    if not boxes:
        return 1.0
    x, y, w, h = union_box(list(boxes.values()))
    coverage = overlap_area((x, y, w, h), (0, 0, width, height)) / (width * height)
    low, high = SPACE_BAND
    if coverage < low:
        return coverage / low
    if coverage > high:
        return (1 - coverage) / (1 - high)
    return 1.0


def _alignment(boxes, width, height) -> float:
    """Share of units sharing a left/centre/right or top/middle/bottom line with another."""
    if len(boxes) < 2:
        return 1.0
    tol = ALIGN_TOL * max(width, height)
    lines = {uid: (x, x + w / 2, x + w, y, y + h / 2, y + h) for uid, (x, y, w, h) in boxes.items()}
    aligned = sum(
        any(abs(lines[u][k] - lines[v][k]) <= tol for v in lines if v != u for k in range(6)) for u in lines)
    return aligned / len(lines)


def _spacing(boxes, width, height) -> float:
    """Half: content keeps a margin from the canvas edges. Half: gaps to each
    unit's nearest neighbour are even (low coefficient of variation)."""
    if not boxes:
        return 1.0
    margin = EDGE_MARGIN * min(width, height)
    edge = sum(x >= margin and y >= margin and x + w <= width - margin and y + h <= height - margin
               for x, y, w, h in boxes.values()) / len(boxes)
    if len(boxes) < 2:
        return 0.5 * edge + 0.5

    def gap(a: Box, b: Box) -> float:
        dx = max(0.0, b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]))
        dy = max(0.0, b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]))
        return max(dx, dy)

    gaps = [min(gap(a, b) for v, b in boxes.items() if v != u) for u, a in boxes.items()]
    mean = sum(gaps) / len(gaps)
    if mean == 0:
        return 0.5 * edge
    spread = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps)) / mean
    return 0.5 * edge + 0.5 * (1 - _clamp(spread))


def _balance(boxes, width, height) -> float:
    total = sum(_area(b) for b in boxes.values())
    if not total:
        return 1.0
    cx = sum(_area(b) * (b[0] + b[2] / 2) for b in boxes.values()) / total
    cy = sum(_area(b) * (b[1] + b[3] / 2) for b in boxes.values()) / total
    distance = math.hypot((cx - width / 2) / width, (cy - height / 2) / height)
    return _clamp(1 - distance / (0.5 * math.sqrt(2)))


def geometry_metrics(source: dict, cand: Candidate, plan: PriorityPlan) -> dict[str, float]:
    width, height = cand.design["canvas"]["width"], cand.design["canvas"]["height"]
    els = {e["id"]: e for e in cand.design["elements"]}
    src_els = {e["id"]: e for e in source["elements"]}
    units = [u for u in plan.ranked_units() if all(i in els for i in u.element_ids)]
    w = {u.unit_id: plan.weight(u.unit_id) for u in units}
    boxes = {u.unit_id: union_box([rotated_aabb(els[i]) for i in u.element_ids]) for u in units}
    src_boxes = {u.unit_id: union_box([rotated_aabb(src_els[i]) for i in u.element_ids]) for u in units}
    return {
        "hierarchy": _hierarchy(units, els, boxes, w),
        "legibility": _legibility(units, els, w, width, height),
        "overlap": _overlap(units, boxes, src_boxes, w),
        "degradation": _degradation(units, els, w, set(cand.degraded_unit_ids), width, height),
        "faithfulness": _faithfulness(units, els, src_els, boxes, src_boxes, w, height, source["canvas"]["height"]),
        "space_use": _space_use(boxes, width, height),
        "alignment": _alignment(boxes, width, height),
        "spacing": _spacing(boxes, width, height),
        "balance": _balance(boxes, width, height),
    }


def geometry_total(m: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """`weights` None means the hand-set WEIGHTS."""
    return sum(weight * m[key] for key, weight in (WEIGHTS if weights is None else weights).items())
