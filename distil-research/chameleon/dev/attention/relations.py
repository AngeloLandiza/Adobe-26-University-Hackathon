"""Hand-written relation features: how specific pieces of a layout relate to
each other (experiment E1 in distil-research/docs/attention/README.md §7).

    relation_features(source, design, plan) -> {name: float}

`source` is the original design, `design` the resized layout being judged,
`plan` node 1's PriorityPlan for the source. Every feature is computed from
unit boxes (each unit's elements' rotated bounds, unioned), over the plan's
content units (background and decoration are left out), and lies in [0, 1].
Distances are shares of the design's canvas diagonal. When a role a feature
needs is missing, the feature takes the neutral default below and the
`has_<role>` flag says so. (Both layouts of a judged pair come from the same
ad and plan, so the flags and defaults cancel in a pairwise model; they keep
the vector the same length for every ad.)

| feature | meaning | default |
|---|---|---|
| has_headline, has_hero, has_cta, has_logo, has_badge, has_legal | 1 if the plan has a unit with that role (hero = hero_image) | - |
| logo_cta_gap | edge-to-edge gap between the logo and the CTA | 0.5 |
| headline_above_hero | 0.5 + 0.5 tanh((hero centre y - headline centre y) / 0.1 H): 1 = headline well above the hero | 0.5 |
| cta_below_headline | same form for the CTA against the headline: 1 = CTA well below | 0.5 |
| badge_hero_gap | smallest gap from any badge to the hero (0 = a badge touches it) | 0.5 |
| badge_spread | mean distance of badge centres from their centroid (0 = one tight cluster) | 0 |
| reading_order | share of unit pairs whose reading order (top to bottom, left to right within a ROW_BAND row) matches the source; (Kendall tau + 1) / 2 | 1 |
| headline_body_ratio | r / (1 + r), r = headline font size / largest body (subheadline, other_text) font size | 0.5 |
| cta_headline_size | a / (1 + a), a = CTA box area / headline box area | 0.5 |
| empty_rect_share | largest empty rectangle on a GRID x GRID grid, share of the canvas | - |
| min_edge_margin | smallest distance from any unit to a canvas edge, / short side, / 0.1, clamped | - |
| text_image_overlap | mean over text units of the share of the box covered by image units | 0 |
| text_image_overlap_gain | the same, minus the source's value per unit, floored at 0 (new text-on-image) | 0 |
| nn_axis_share | share of units sharing an edge or centre line (within ALIGN_TOL of the long side) with their nearest neighbour | 1 |
| rhythm_cv | coefficient of variation of the vertical gaps between consecutive stacked units, / 2, clamped | 0 |
| hero_area_share | hero box area / canvas area | 0 |
| logo_area_share | logo box area / canvas area | 0 |
| headline_area_share | headline box area / canvas area | 0 |
| legal_floor_ratio | smallest legal font size / its floor (floor_px), / 2, clamped | 0.5 |
| below_floor_share | share of text units whose smallest font is below its role's floor | 0 |
| uniform_scale | 1 - RMS residual of fitting x' = s x + o (one s, both axes) to every unit's corners, / (0.05 diagonal), clamped: 1 = a uniform shrink of the source (the fallback's signature) | 1 |
| neighbour_kept | of unit pairs within NEAR_SOURCE of each other in the source, the share still within NEAR_DESIGN | 1 |
| headline_scale_vs_source | q / (1 + q), q = (headline font / short side) here over the same in the source | 0.5 |
| logo_corner_dist | distance from the logo's centre to the nearest canvas corner | 0.5 |
"""

import math
from itertools import combinations

from engine.pipeline.geometry import Box, overlap_area, rotated_aabb, union_box
from engine.pipeline.layout.place import floor_px
from engine.pipeline.types import PriorityPlan, Unit

GRID = 20            # empty_rect_share: cells per axis
ROW_BAND = 0.1       # reading_order: centres this close (share of height) read as one row
ALIGN_TOL = 0.01     # nn_axis_share: share of the long side
NEAR_SOURCE = 0.02   # neighbour_kept: "touching" in the source, share of the diagonal
NEAR_DESIGN = 0.03   # ... and still close in the design
MARGIN_FULL = 0.1    # min_edge_margin reaches 1 at this share of the short side
UNIFORM_TOL = 0.05   # uniform_scale reaches 0 at this RMS residual, share of the diagonal
BODY_ROLES = ("subheadline", "other_text")

ROLE_FLAGS = {"headline": "headline", "hero": "hero_image", "cta": "cta", "logo": "logo", "badge": "badge",
              "legal": "legal"}
FEATURE_NAMES: tuple[str, ...] = (
    *(f"has_{k}" for k in ROLE_FLAGS),
    "logo_cta_gap", "headline_above_hero", "cta_below_headline", "badge_hero_gap", "badge_spread",
    "reading_order", "headline_body_ratio", "cta_headline_size", "empty_rect_share", "min_edge_margin",
    "text_image_overlap", "text_image_overlap_gain", "nn_axis_share", "rhythm_cv", "hero_area_share",
    "logo_area_share", "headline_area_share", "legal_floor_ratio", "below_floor_share", "uniform_scale",
    "neighbour_kept", "headline_scale_vs_source", "logo_corner_dist",
)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _area(b: Box) -> float:
    return b[2] * b[3]


def _centre(b: Box) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def _gap(a: Box, b: Box) -> float:
    """Edge-to-edge distance; 0 when the boxes touch or overlap."""
    dx = max(0.0, b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]))
    dy = max(0.0, b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]))
    return math.hypot(dx, dy)


def _squash(r: float) -> float:
    """r / (1 + r) for r >= 0: 0.5 means equal."""
    return _clamp(r / (1 + r)) if r >= 0 else 0.0


def _soft_above(top: Box, below: Box, height: float) -> float:
    return 0.5 + 0.5 * math.tanh((_centre(below)[1] - _centre(top)[1]) / (0.1 * height))


def _precedes(a: Box, b: Box, h: float) -> bool:
    (ax, ay), (bx, by) = _centre(a), _centre(b)
    if abs(ay - by) > ROW_BAND * h:
        return ay < by
    return ax < bx


def _fonts(els: dict, unit: Unit) -> list[float]:
    return [els[i]["font_size"] for i in unit.element_ids if els[i]["type"] == "text"]


def _has_type(els: dict, unit: Unit, kind: str) -> bool:
    return any(els[i]["type"] == kind for i in unit.element_ids)


def _covered_share(box: Box, others: list[Box]) -> float:
    """Share of `box` under the union of `others`, approximated as the largest single cover (no double counting)."""
    area = _area(box)
    if area <= 0 or not others:
        return 0.0
    return _clamp(max(overlap_area(box, o) for o in others) / area)


def _largest_empty_share(boxes: list[Box], width: float, height: float) -> float:
    cw, ch = width / GRID, height / GRID
    filled = [[False] * GRID for _ in range(GRID)]
    for x, y, w, h in boxes:
        c0, c1 = max(0, int(x // cw)), min(GRID - 1, int(math.ceil((x + w) / cw)) - 1)
        r0, r1 = max(0, int(y // ch)), min(GRID - 1, int(math.ceil((y + h) / ch)) - 1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                filled[r][c] = True
    best, heights = 0, [0] * GRID
    for r in range(GRID):  # largest rectangle in a histogram, row by row
        heights = [0 if filled[r][c] else heights[c] + 1 for c in range(GRID)]
        stack: list[int] = []
        for c in range(GRID + 1):
            hgt = heights[c] if c < GRID else 0
            while stack and heights[stack[-1]] >= hgt:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                best = max(best, heights[top] * (c - left))
            stack.append(c)
    return best / (GRID * GRID)


def _uniform_residual(src: list[Box], dst: list[Box]) -> float:
    """RMS residual of the best x' = s x + ox, y' = s y + oy over every box corner."""
    xs = [v for b in src for v in (b[0], b[0] + b[2])]
    ys = [v for b in src for v in (b[1], b[1] + b[3])]
    xd = [v for b in dst for v in (b[0], b[0] + b[2])]
    yd = [v for b in dst for v in (b[1], b[1] + b[3])]
    n = len(xs)
    mx, my, mxd, myd = sum(xs) / n, sum(ys) / n, sum(xd) / n, sum(yd) / n
    num = sum((a - mx) * (b - mxd) for a, b in zip(xs, xd)) + sum((a - my) * (b - myd) for a, b in zip(ys, yd))
    den = sum((a - mx) ** 2 for a in xs) + sum((a - my) ** 2 for a in ys)
    s = num / den if den > 0 else 1.0
    sse = (sum((b - mxd - s * (a - mx)) ** 2 for a, b in zip(xs, xd))
           + sum((b - myd - s * (a - my)) ** 2 for a, b in zip(ys, yd)))
    return math.sqrt(sse / (2 * n))


def _lines(b: Box) -> tuple[float, ...]:
    x, y, w, h = b
    return (x, x + w / 2, x + w, y, y + h / 2, y + h)


def relation_features(source: dict, design: dict, plan: PriorityPlan) -> dict[str, float]:
    width, height = design["canvas"]["width"], design["canvas"]["height"]
    src_w, src_h = source["canvas"]["width"], source["canvas"]["height"]
    diag, src_diag = math.hypot(width, height), math.hypot(src_w, src_h)
    els = {e["id"]: e for e in design["elements"]}
    src_els = {e["id"]: e for e in source["elements"]}
    units = [u for u in plan.ranked_units() if all(i in els and i in src_els for i in u.element_ids)]
    boxes = {u.unit_id: union_box([rotated_aabb(els[i]) for i in u.element_ids]) for u in units}
    src_boxes = {u.unit_id: union_box([rotated_aabb(src_els[i]) for i in u.element_ids]) for u in units}

    def first(role: str) -> Unit | None:
        return next((u for u in units if u.role == role), None)

    head, hero, cta, logo = first("headline"), first("hero_image"), first("cta"), first("logo")
    badges = [u for u in units if u.role == "badge"]
    legal = [u for u in units if u.role == "legal"]
    f: dict[str, float] = {f"has_{k}": float(any(u.role == role for u in units)) for k, role in ROLE_FLAGS.items()}

    f["logo_cta_gap"] = _clamp(_gap(boxes[logo.unit_id], boxes[cta.unit_id]) / diag) if logo and cta else 0.5
    f["headline_above_hero"] = (_soft_above(boxes[head.unit_id], boxes[hero.unit_id], height)
                                if head and hero else 0.5)
    f["cta_below_headline"] = _soft_above(boxes[head.unit_id], boxes[cta.unit_id], height) if head and cta else 0.5
    f["badge_hero_gap"] = (_clamp(min(_gap(boxes[b.unit_id], boxes[hero.unit_id]) for b in badges) / diag)
                           if badges and hero else 0.5)
    if len(badges) >= 2:
        centres = [_centre(boxes[b.unit_id]) for b in badges]
        cx, cy = sum(c[0] for c in centres) / len(centres), sum(c[1] for c in centres) / len(centres)
        f["badge_spread"] = _clamp(sum(math.hypot(x - cx, y - cy) for x, y in centres) / len(centres) / diag)
    else:
        f["badge_spread"] = 0.0

    pairs = list(combinations(units, 2))
    f["reading_order"] = (sum(_precedes(src_boxes[a.unit_id], src_boxes[b.unit_id], src_h)
                              == _precedes(boxes[a.unit_id], boxes[b.unit_id], height) for a, b in pairs) / len(pairs)
                          if pairs else 1.0)

    body = [max(_fonts(els, u)) for u in units if u.role in BODY_ROLES and _fonts(els, u)]
    head_fonts = _fonts(els, head) if head else []
    f["headline_body_ratio"] = _squash(max(head_fonts) / max(body)) if head_fonts and body and max(body) > 0 else 0.5
    f["cta_headline_size"] = (_squash(_area(boxes[cta.unit_id]) / _area(boxes[head.unit_id]))
                              if cta and head and _area(boxes[head.unit_id]) > 0 else 0.5)

    all_boxes = list(boxes.values())
    f["empty_rect_share"] = _largest_empty_share(all_boxes, width, height) if all_boxes else 1.0
    short = min(width, height)
    margins = [min(x, y, width - x - w, height - y - h) for x, y, w, h in all_boxes]
    f["min_edge_margin"] = _clamp(min(margins) / short / MARGIN_FULL) if margins else 1.0

    text_units = [u for u in units if _has_type(els, u, "text") and not _has_type(els, u, "image")]
    image_units = [u for u in units if _has_type(els, u, "image")]

    def coverage(u: Unit, bx: dict) -> float:
        return _covered_share(bx[u.unit_id], [bx[i.unit_id] for i in image_units if i is not u])

    if text_units:
        f["text_image_overlap"] = sum(coverage(u, boxes) for u in text_units) / len(text_units)
        f["text_image_overlap_gain"] = sum(max(0.0, coverage(u, boxes) - coverage(u, src_boxes))
                                           for u in text_units) / len(text_units)
    else:
        f["text_image_overlap"] = f["text_image_overlap_gain"] = 0.0

    if len(units) >= 2:
        tol = ALIGN_TOL * max(width, height)
        aligned = 0
        for u in units:
            nn = min((v for v in units if v is not u),
                     key=lambda v: (_gap(boxes[u.unit_id], boxes[v.unit_id]),
                                    math.dist(_centre(boxes[u.unit_id]), _centre(boxes[v.unit_id]))))
            aligned += any(abs(p - q) <= tol for p, q in zip(_lines(boxes[u.unit_id]), _lines(boxes[nn.unit_id])))
        f["nn_axis_share"] = aligned / len(units)
    else:
        f["nn_axis_share"] = 1.0

    stack = sorted(all_boxes, key=lambda b: (b[1], b[0]))
    gaps = [b[1] - (a[1] + a[3]) for a, b in zip(stack, stack[1:]) if b[1] >= a[1] + a[3]]
    if len(gaps) >= 2 and sum(gaps) > 0:
        mean = sum(gaps) / len(gaps)
        f["rhythm_cv"] = _clamp(math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps)) / mean / 2)
    else:
        f["rhythm_cv"] = 0.0

    canvas = width * height
    f["hero_area_share"] = _clamp(_area(boxes[hero.unit_id]) / canvas) if hero else 0.0
    f["logo_area_share"] = _clamp(_area(boxes[logo.unit_id]) / canvas) if logo else 0.0
    f["headline_area_share"] = _clamp(_area(boxes[head.unit_id]) / canvas) if head else 0.0

    legal_fonts = [s for u in legal for s in _fonts(els, u)]
    f["legal_floor_ratio"] = (_clamp(min(legal_fonts) / floor_px("legal", width, height) / 2)
                              if legal_fonts else 0.5)
    texted = [u for u in units if _fonts(els, u)]
    f["below_floor_share"] = (sum(min(_fonts(els, u)) < floor_px(u.role, width, height) - 1e-6 for u in texted)
                              / len(texted) if texted else 0.0)

    order = [u.unit_id for u in units]
    f["uniform_scale"] = (1 - _clamp(_uniform_residual([src_boxes[i] for i in order], [boxes[i] for i in order])
                                     / (UNIFORM_TOL * diag)) if units else 1.0)

    near = [(a, b) for a, b in pairs if _gap(src_boxes[a.unit_id], src_boxes[b.unit_id]) <= NEAR_SOURCE * src_diag]
    f["neighbour_kept"] = (sum(_gap(boxes[a.unit_id], boxes[b.unit_id]) <= NEAR_DESIGN * diag for a, b in near)
                           / len(near) if near else 1.0)

    src_head_fonts = _fonts(src_els, head) if head else []
    if head_fonts and src_head_fonts and max(src_head_fonts) > 0:
        q = (max(head_fonts) / short) / (max(src_head_fonts) / min(src_w, src_h))
        f["headline_scale_vs_source"] = _squash(q)
    else:
        f["headline_scale_vs_source"] = 0.5
    if logo:
        cx, cy = _centre(boxes[logo.unit_id])
        f["logo_corner_dist"] = _clamp(min(math.hypot(cx - x, cy - y) for x in (0, width) for y in (0, height)) / diag)
    else:
        f["logo_corner_dist"] = 0.5
    return {k: float(f[k]) for k in FEATURE_NAMES}
