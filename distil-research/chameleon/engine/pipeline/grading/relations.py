"""Relation preservation: how many of the source's pairwise relations a
layout keeps (research switch CHAMELEON_RELPRESERVE=1, used in pick_best).

A "relation" is a fact about two content units of the source design:
- size order: one text is clearly bigger than another (font size, more than
  SIZE_TOL apart), or one image clearly bigger than another (area, more than
  AREA_TOL apart, as in ReLayout's Size Rel);
- reading order: one unit comes before the other: it is wholly above it, or
  wholly left of it on the same band. In the layout, "before" may become
  either above or left of (a square's stack turning into a strip's row keeps
  the reading order); it is broken when the later unit now comes first;
- shared edge: two units share a left, centre or right line (or a top,
  middle or bottom line) within ALIGN_TOL of the long side. In the layout the
  pair must share one of those lines again; when a stacked pair became a
  side-by-side pair, left/centre/right map to top/middle/bottom.

The score is the mean, over the families that have any relation, of the
priority-weighted share kept (1.0 = everything kept). Only content units the
layout keeps are judged: backgrounds, decorations and units the candidate
flagged as degraded (demoted to fine print) are left out. Text is judged by
where its glyphs are (the widest measured line, per alignment), not its box.
"""

from __future__ import annotations

import os
from itertools import combinations

from engine.pipeline.geometry import Box, rotated_aabb, union_box

ENV = "CHAMELEON_RELPRESERVE"
WEIGHT_ENV = "CHAMELEON_RELPRESERVE_W"
DEFAULT_WEIGHT = 0.15   # chosen on train judgments (dev/distill/prior_experiments.py relval)
SIZE_TOL = 0.10         # texts whose sizes are within 10% are "the same size": no order to keep
AREA_TOL = 0.20         # images whose areas are within 20% likewise
ALIGN_TOL = 0.01        # share of the canvas's long side: edges this close are shared
ORDER_TOL = 0.005       # share of the canvas's long side: touching boxes still read in order
LAYER_ROLES = ("background", "decoration")


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def weight() -> float:
    try:
        w = float(os.environ.get(WEIGHT_ENV, DEFAULT_WEIGHT))
    except ValueError:
        return DEFAULT_WEIGHT
    return w if w >= 0 else DEFAULT_WEIGHT


def _ink(el: dict, m) -> Box:
    if el["type"] != "text" or el.get("rotation") or m is None:
        return rotated_aabb(el)
    try:
        widths = m.line_widths(el, el["font_size"], el["width"]) or [0.0]
        block = m.block_height(el, el["font_size"], el["width"])
    except Exception:
        return rotated_aabb(el)
    w = min(max(widths), el["width"])
    align = el.get("text_align") or "left"
    x = el["x"] + (el["width"] - w) / 2 if align == "center" else el["x"] + el["width"] - w if align == "right" else el["x"]
    return (x, el["y"], w, min(el["height"], block) if block > 0 else el["height"])


def _lines(b: Box) -> tuple:
    x, y, w, h = b
    return (x, x + w / 2, x + w, y, y + h / 2, y + h)  # L C R T M B


def _before(a: Box, b: Box, tol: float) -> str | None:
    """'above' / 'left' when a clearly comes before b, 'after' when b clearly comes before a, else None."""
    if a[1] + a[3] <= b[1] + tol:
        return "above"
    if b[1] + b[3] <= a[1] + tol:
        return "after"
    if a[0] + a[2] <= b[0] + tol:
        return "left"
    if b[0] + b[2] <= a[0] + tol:
        return "after"
    return None


def _stacked(a: Box, b: Box) -> bool | None:
    """True: one above the other; False: side by side; None: overlapping."""
    if a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1]:
        return True
    if a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]:
        return False
    return None


def _shared(a: Box, b: Box, tol: float) -> set[int]:
    la, lb = _lines(a), _lines(b)
    return {k for k in range(6) if abs(la[k] - lb[k]) <= tol}


def relation_parts(source: dict, design: dict, plan, measurer=None, skip=()) -> dict[str, tuple[float, float]]:
    """{family: (kept weight, total weight)} for size / order / align."""
    els = {e["id"]: e for e in design["elements"]}
    src = {e["id"]: e for e in source["elements"]}
    skip = set(skip or ())
    units = [u for u in plan.ranked_units() if u.role not in LAYER_ROLES and u.unit_id not in skip
             and u.element_ids and all(i in els and i in src for i in u.element_ids)]
    W, H = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    SW, SH = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    b0 = {u.unit_id: union_box([_ink(src[i], measurer) for i in u.element_ids]) for u in units}
    b1 = {u.unit_id: union_box([_ink(els[i], measurer) for i in u.element_ids]) for u in units}
    w = {u.unit_id: plan.weight(u.unit_id) for u in units}

    def font(e, u):
        sizes = [e[i]["font_size"] for i in u.element_ids if e[i]["type"] == "text"]
        return max(sizes) if sizes else None

    parts = {k: [0.0, 0.0] for k in ("size", "order", "align")}
    for a, b in combinations(units, 2):
        pw = w[a.unit_id] + w[b.unit_id]
        A0, B0, A1, B1 = b0[a.unit_id], b0[b.unit_id], b1[a.unit_id], b1[b.unit_id]
        # size order
        fa0, fb0 = font(src, a), font(src, b)
        if fa0 and fb0:
            s0, s1, tol = fa0 / fb0, font(els, a) / font(els, b), SIZE_TOL
        elif fa0 is None and fb0 is None and A0[2] * A0[3] > 0 and B0[2] * B0[3] > 0 and B1[2] * B1[3] > 0:
            s0, s1, tol = A0[2] * A0[3] / (B0[2] * B0[3]), A1[2] * A1[3] / (B1[2] * B1[3]), AREA_TOL
        else:
            s0 = None
        if s0 is not None and (s0 > 1 + tol or s0 < 1 / (1 + tol)):
            kept = 1.0 if (s1 > 1) == (s0 > 1) and s1 != 1 else 0.5 if s1 == 1 else 0.0
            parts["size"][0] += pw * kept
            parts["size"][1] += pw
        # reading order
        rel0 = _before(A0, B0, ORDER_TOL * max(SW, SH))
        if rel0 is not None:
            first0 = rel0 != "after"
            rel1 = _before(A1, B1, ORDER_TOL * max(W, H))
            if rel1 is None:  # overlapping now: judge by centres
                ca, cb = (A1[1] + A1[3] / 2, A1[0] + A1[2] / 2), (B1[1] + B1[3] / 2, B1[0] + B1[2] / 2)
                kept = 0.5 if (ca < cb) == first0 else 0.0
            else:
                kept = 1.0 if (rel1 != "after") == first0 else 0.0
            parts["order"][0] += pw * kept
            parts["order"][1] += pw
        # shared edges
        shared0 = _shared(A0, B0, ALIGN_TOL * max(SW, SH))
        if shared0:
            st0, st1 = _stacked(A0, B0), _stacked(A1, B1)
            want = shared0
            if st0 is not None and st1 is not None and st0 != st1:
                want = {(k + 3) % 6 for k in shared0}  # L C R <-> T M B
            kept = 1.0 if want & _shared(A1, B1, ALIGN_TOL * max(W, H)) else 0.0
            parts["align"][0] += pw * kept
            parts["align"][1] += pw
    return {k: (v[0], v[1]) for k, v in parts.items()}


def relation_score(source: dict, design: dict, plan, measurer=None, skip=()) -> float:
    """Mean kept share over the families with any relation; 1.0 if none."""
    try:
        parts = relation_parts(source, design, plan, measurer, skip)
    except Exception:
        return 1.0
    shares = [kept / total for kept, total in parts.values() if total > 0]
    return sum(shares) / len(shares) if shares else 1.0
