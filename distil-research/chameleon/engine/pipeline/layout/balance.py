"""Balance pass (CHAMELEON_BALANCE=1; off by default, and with it off nothing
here runs and the engine's output is unchanged).

The teacher's critiques of the anchored layouts (v4) keep naming three
things the grader barely sees:

- **bands stop spanning**: "the dark header bar no longer spans the canvas.
  It is inset with cream gaps on the left and right" (anchored scales a band
  with its content instead of stretching it);
- **related pieces drift apart**: "'Ships late June' has drifted far below
  the CTA, cut off from its button";
- **lopsided whitespace**: "the stack sits too high, with about 30px above
  the headline and about 75px of empty band below the button".

`balance_all` takes every finished, gate-passing candidate (the safe
fallback excluded) and, before grading, applies three steps in order, each
kept only if the candidate still passes every hard gate and no text ends
below its role floor (or below its size before the pass):

a. **Bands span.** A shape or image (not the background) that spanned the
   source edge to edge on an axis (BAND_SPAN of it, or touching both edges)
   spans the target on that axis again: a rectangle or a `cover` image
   stretches (a cover image crops), a `contain` image is scaled up to span
   if that is at most BAND_IMG_MAX x and fits. A band that touched both
   edges touches both again; one inset a little keeps its insets in
   proportion. Its contents (layer_groups containers) stay inside it. A band
   that would newly cover a content piece it didn't hold stays as it was.
b. **Groups stay together.** Two content pieces that sat close in the
   source (stacked or side by side, a gap at most PAIR_GAP x the smaller
   one's size along that axis and PAIR_GAP_SIDE of the side, overlapping
   across it by PAIR_ACROSS: a caption under a CTA, a sub-line under a
   headline) and are now more than SPLIT_RATIO x their scaled source gap
   apart: the follower (the less important one) moves toward its leader
   along the axis they now stack on, to the scaled source gap.
c. **Balance the leftover space.** Per axis, the free-floating content
   block (every content piece but those spanning the axis, and small pieces
   that hug an edge in the source: a logo top-left stays top-left; the
   headline's group always moves) sits between the canvas edges, narrowed
   by any fixed piece before or after it in its lane. If its two margins
   are lopsided (the larger more than IMB_RATIO x the smaller and more than
   IMB_SHARE of the axis apart) and the source's weren't lopsided the same
   way, the block shifts toward equal margins. Then, if the block leaves
   more than FILL_UNUSED of the safe area unused on the short axis, it may
   grow uniformly about its centre, up to MAX_GROW x, with every text
   re-measured (re-wrapped if needed), the headline still the largest
   text, nothing past the safe margin.

Every move or growth carries the decorations sitting on the piece with it,
never increases a piece's overlap with another (GAP x the short side apart),
keeps ink inside the v2 safe margin (or no closer to an edge than it already
was), keeps a decoration that held a piece holding it, and never leaves a
piece half on a decoration it wasn't half on. Never raises: a candidate that
can't be balanced is returned as it was.
"""

from __future__ import annotations

import math
import os

from engine.pipeline.geometry import Box, overlap_area, rotated_aabb, union_box
from engine.pipeline.types import Candidate, PriorityPlan, Unit

ENV = "CHAMELEON_BALANCE"
BAND_SPAN = 0.95       # spans this share of an axis: a band
EDGE_TOL = 0.01        # share of the side: this close to an edge touches it
BAND_IMG_MAX = 1.25    # a contain image may grow at most this much to span again
PAIR_GAP = 1.0         # close pair: gap at most the smaller piece's size along the axis ...
PAIR_GAP_SIDE = 0.08   # ... and this share of the source side ...
PAIR_ACROSS = 0.5      # ... overlapping across the axis by this share of the narrower one
SPLIT_RATIO = 1.5      # a close pair now farther apart than this x the scaled source gap is split ...
SPLIT_SLACK = 0.03     # ... (and by this share of the short side more)
IMB_RATIO = 2.0        # lopsided: one margin more than this x the other ...
IMB_SHARE = 0.06       # ... and the two more than this share of the axis apart
IMB_SOURCE = 0.5       # the source lopsided the same way by this x the imbalance: deliberate
HUG = 0.1              # a small piece this close to an edge in the source (share of the side) stays on it
HUG_AREA = 0.06        # ... small: at most this share of the source canvas
FILL_UNUSED = 0.25     # a block leaving this share of the safe area unused on the short axis may grow ...
MAX_GROW = 1.15        # ... by at most this
GROW_STEPS = (1.15, 1.1, 1.05)
SHIFT_FRACTIONS = (1.0, 0.75, 0.5, 0.25)
GAP = 0.015            # least gap to a piece a move didn't already touch, x the short side (as polish)
ATTACH = 0.3           # a decoration covering this share of a piece (or the piece of it) sits on it
HOLD = 0.5             # a decoration holding this share of a piece holds it
_LAYER_ROLES = ("background", "decoration")
_PHOTO_ROLES = ("hero_image", "secondary_image")
_EPS = 0.5


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def balance_all(source: dict, target: dict, plan: PriorityPlan, measurer, candidates: list[Candidate]) -> list[Candidate]:
    out = []
    layout = _layout_plan(source, plan, target)
    for cand in candidates:
        if cand.is_fallback:
            out.append(cand)
            continue
        try:
            out.append(balance(source, target, plan, measurer, cand, layout))
        except Exception:  # balance never sinks a candidate
            out.append(cand)
    return out


def _layout_plan(source: dict, plan: PriorityPlan, target: dict) -> PriorityPlan:
    """With CHAMELEON_LAYERS=1 a container and everything on it is one piece."""
    if os.environ.get("CHAMELEON_LAYERS") == "1":
        try:
            from engine.pipeline.layout import layer_groups
            return layer_groups.detect(source, plan, target).plan
        except Exception:
            pass
    return plan


def balance(source: dict, target: dict, plan: PriorityPlan, measurer, cand: Candidate,
            layout: PriorityPlan | None = None) -> Candidate:
    """`cand` with bands spanning, close pairs together and its leftover space
    balanced (same id, template, params and degraded units), or `cand` itself."""
    b = _Balancer(source, target, plan, layout or _layout_plan(source, plan, target), measurer, cand)
    if not b.pieces or b.base_failed:
        return cand
    for step in (b.bands, b.groups, b.spread):
        saved = {k: dict(v) for k, v in b.placed.items()}, list(b.fixes)
        if step() and not b.ok():
            b.placed, b.fixes = saved
    if not b.fixes:
        return cand
    design = b.design()
    if design is None or not b.ok(design):
        return cand
    return Candidate(cand.candidate_id, design, cand.template, cand.params, list(cand.degraded_unit_ids),
                     cand.is_fallback)


# ---------------------------------------------------------------- geometry

def _span(b: Box, ax: str) -> tuple[float, float]:
    return (b[0], b[0] + b[2]) if ax == "x" else (b[1], b[1] + b[3])


def _area(b: Box) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def _inflate(b: Box, g: float) -> Box:
    return (b[0] - g, b[1] - g, b[2] + 2 * g, b[3] + 2 * g)


def _gap(a: Box, b: Box) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return math.hypot(dx, dy)


def spans(b: Box, ax: str, side: float) -> bool:
    """Spans BAND_SPAN of the axis, or touches both of its edges."""
    lo, hi = _span(b, ax)
    tol = EDGE_TOL * side
    return (hi - lo) >= BAND_SPAN * side or (lo <= tol and hi >= side - tol)


def _across(a: Box, b: Box, ax: str) -> bool:
    """Overlap across `ax` (on the other axis) by PAIR_ACROSS of the narrower one."""
    c = "x" if ax == "y" else "y"
    (a0, a1), (b0, b1) = _span(a, c), _span(b, c)
    return min(a1, b1) - max(a0, b0) >= PAIR_ACROSS * min(a1 - a0, b1 - b0)


def close_pairs(boxes: dict[str, Box], W: float, H: float) -> list[tuple[str, str, str, float]]:
    """(a, b, axis, gap) for pieces that sit close (see the module docstring, b)."""
    ids = sorted(boxes)
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            A, B = boxes[a], boxes[b]
            if overlap_area(A, B) > 0:
                continue
            for ax, side in (("y", H), ("x", W)):
                if not _across(A, B, ax):
                    continue
                (p0, p1), (q0, q1) = _span(A, ax), _span(B, ax)
                gap = max(q0 - p1, p0 - q1)
                if 0 <= gap <= PAIR_GAP * min(p1 - p0, q1 - q0) and gap <= PAIR_GAP_SIDE * side:
                    out.append((a, b, ax, gap))
    return out


def lopsided(m0: float, m1: float, side: float) -> tuple[bool, float]:
    """(lopsided?, signed imbalance as a share of the side; + means more room at the end)."""
    imb = (m1 - m0) / side if side else 0.0
    return max(m0, m1) > IMB_RATIO * max(min(m0, m1), 1.0) and abs(imb) > IMB_SHARE, imb


# ---------------------------------------------------------------- the pass

class _Balancer:
    def __init__(self, source: dict, target: dict, plan: PriorityPlan, layout: PriorityPlan, measurer, cand: Candidate):
        from engine.pipeline.grading.gates import check_gates
        from engine.pipeline.layout.polish import safe_margins
        self.check_gates = check_gates
        self.source, self.target, self.plan, self.layout, self.m, self.cand = source, target, plan, layout, measurer, cand
        self.W, self.H = float(target["width"]), float(target["height"])
        self.Ws, self.Hs = float(source["canvas"]["width"]), float(source["canvas"]["height"])
        self.short = min(self.W, self.H)
        self.src = {e["id"]: e for e in source["elements"]}
        self.placed = {e["id"]: dict(e) for e in cand.design["elements"]}
        self.orig_font = {i: e["font_size"] for i, e in self.placed.items() if e["type"] == "text"}
        self.role = {i: u.role for u in plan.units for i in u.element_ids}
        self.pieces = [u for u in layout.ranked_units()
                       if u.role not in _LAYER_ROLES and u.element_ids and all(i in self.placed for i in u.element_ids)]
        self.decos = [i for u in layout.units if u.role == "decoration" for i in u.element_ids if i in self.placed]
        self.mx, self.my = safe_margins(self.W, self.H)
        self.gap = GAP * self.short
        self.fixes: list[str] = []
        self.base_failed = set(check_gates(source, cand.design, target, plan, measurer))
        self.src_box = {u.unit_id: self._ubox(u, self.src) for u in self.pieces}

    # -- bookkeeping ---------------------------------------------------------
    def design(self) -> dict | None:
        from engine.pipeline.layout.normalize import finalize
        return finalize(self.source, self.placed, self.target)

    def ok(self, design: dict | None = None) -> bool:
        design = design if design is not None else self.design()
        if design is None:
            return False
        if set(self.check_gates(self.source, design, self.target, self.plan, self.m)) - self.base_failed:
            return False
        from engine.pipeline.layout.place import floor_px
        for e in design["elements"]:
            if e["type"] == "text" and e["font_size"] < min(self.orig_font[e["id"]],
                                                              floor_px(self.role.get(e["id"]), self.W, self.H)) - 0.01:
                return False
        return True

    def ink(self, el: dict) -> Box:
        if el["type"] != "text" or el.get("rotation"):
            return rotated_aabb(el)
        widths = self.m.line_widths(el, el["font_size"], el["width"]) or [0.0]
        w = min(max(widths), el["width"])
        align = el.get("text_align") or "left"
        x = el["x"] + (el["width"] - w) / 2 if align == "center" else \
            el["x"] + el["width"] - w if align == "right" else el["x"]
        return (x, el["y"], w, min(el["height"], self.m.block_height(el, el["font_size"], el["width"])))

    def _ubox(self, unit: Unit, geo: dict | None = None) -> Box:
        geo = geo or self.placed
        return union_box([self.ink(geo[i]) for i in unit.element_ids])

    def boxes(self) -> dict[str, Box]:
        return {u.unit_id: self._ubox(u) for u in self.pieces}

    def attached(self, unit: Unit, boxes: dict[str, Box]) -> list[str]:
        """Decorations sitting on this piece and no other (bands excepted): they move with it."""
        ub = boxes[unit.unit_id]
        out = []
        for did in self.decos:
            db = rotated_aabb(self.placed[did])
            if spans(db, "x", self.W) or spans(db, "y", self.H) or _area(db) >= 0.5 * self.W * self.H:
                continue

            def on(b):
                small = min(_area(db), _area(b))
                return small > 0 and overlap_area(db, b) >= ATTACH * small
            if on(ub) and not any(on(boxes[v.unit_id]) for v in self.pieces if v is not unit):
                out.append(did)
        return out

    # -- checks for a moved / grown set of pieces ----------------------------
    def _valid(self, moved: list[Unit], new: dict[str, Box], old: dict[str, Box], deco_ids: set[str],
               near: set[str] = frozenset()) -> bool:
        """No other piece gains overlap (GAP apart; `near` pieces only must not touch),
        ink stays in the safe margin (or no closer than before), a decoration holding
        a piece still holds it, and nothing becomes half-on a decoration."""
        from engine.pipeline.layout.polish import straddles
        ids = {u.unit_id for u in moved}
        for u in moved:
            nb, ob = new[u.unit_id], old[u.unit_id]
            if not self._safe(nb, ob):
                return False
            for v in self.pieces:
                if v.unit_id in ids:
                    continue
                vb = old[v.unit_id]
                g = 0.0 if v.unit_id in near else self.gap
                if overlap_area(_inflate(nb, g), vb) > overlap_area(_inflate(ob, g), vb) + 1e-6:
                    return False
            for did in self.decos:
                if did in deco_ids:
                    continue
                d = self.placed[did]
                db = rotated_aabb(d)
                if overlap_area(db, ob) >= HOLD * _area(ob) and overlap_area(db, nb) < HOLD * _area(nb):
                    return False
                if straddles(d, db, nb) and not straddles(d, db, ob):
                    return False
        return True

    def _safe(self, new: Box, old: Box) -> bool:
        W, H = self.W, self.H
        for d_new, d_old, m in ((new[0], old[0], self.mx), (W - new[0] - new[2], W - old[0] - old[2], self.mx),
                                (new[1], old[1], self.my), (H - new[1] - new[3], H - old[1] - old[3], self.my)):
            if d_new < min(m, d_old) - _EPS:
                return False
        return True

    def _shift_geo(self, ids: list[str], dx: float, dy: float) -> dict[str, dict]:
        return {i: dict(self.placed[i], x=self.placed[i]["x"] + dx, y=self.placed[i]["y"] + dy) for i in ids}

    # -- a. bands span -------------------------------------------------------
    def bands(self) -> bool:
        bg = {i for u in self.plan.units if u.role == "background" for i in u.element_ids}
        changed = False
        boxes = self.boxes()
        for e0 in self.source["elements"]:
            eid = e0["id"]
            if e0["type"] not in ("shape", "image") or eid in bg or eid not in self.placed or e0.get("rotation"):
                continue
            b0 = rotated_aabb(e0)
            if spans(b0, "x", self.Ws) and spans(b0, "y", self.Hs):
                continue  # a full-bleed layer, not a band
            for ax, sside, side in (("x", self.Ws, self.W), ("y", self.Hs, self.H)):
                cur = self.placed[eid]
                if not spans(b0, ax, sside) or spans(rotated_aabb(cur), ax, side):
                    continue
                new = self._stretched(e0, cur, ax, sside, side)
                if new is None or not self._band_clear(eid, b0, rotated_aabb(cur), rotated_aabb(new), boxes):
                    continue
                self.placed[eid] = new
                self.fixes.append(f"band:{eid}:{ax}")
                changed = True
        return changed

    def _stretched(self, e0: dict, cur: dict, ax: str, sside: float, side: float) -> dict | None:
        lo0, hi0 = _span(rotated_aabb(e0), ax)
        tol = EDGE_TOL * sside
        if lo0 <= tol and hi0 >= sside - tol:
            lo, hi = 0.0, side
        else:
            lo, hi = max(0.0, lo0 / sside * side), min(side, hi0 / sside * side)
        key, pos = ("width", "x") if ax == "x" else ("height", "y")
        if e0["type"] == "shape" and e0.get("shape_type") not in ("ellipse", "circle"):
            return dict(cur, **{pos: lo, key: hi - lo})
        if e0["type"] == "image" and e0.get("fit") == "cover":
            return dict(cur, **{pos: lo, key: hi - lo})
        if e0["type"] != "image" or not cur["width"] or not cur["height"]:
            return None
        k = (hi - lo) / cur[key]
        if k > BAND_IMG_MAX:
            return None
        other, opos = ("height", "y") if ax == "x" else ("width", "x")
        oside = self.H if ax == "x" else self.W
        olen = cur[other] * k
        if olen > oside:
            return None
        omid = cur[opos] + cur[other] / 2
        o = min(max(0.0, omid - olen / 2), oside - olen)
        return dict(cur, **{pos: lo, key: hi - lo, opos: o, other: olen})

    def _band_clear(self, eid: str, src_box: Box, old: Box, new: Box, boxes: dict[str, Box]) -> bool:
        """The stretched band covers no more of a content piece it didn't hold in the source."""
        for u in self.pieces:
            if eid in u.element_ids:
                continue
            sb = self.src_box[u.unit_id]
            if overlap_area(src_box, sb) >= HOLD * _area(sb):
                continue  # it sat on this band in the source
            vb = boxes[u.unit_id]
            if overlap_area(new, vb) > overlap_area(old, vb) + 1e-6:
                return False
        return True

    # -- b. groups stay together ---------------------------------------------
    def groups(self) -> bool:
        by_id = {u.unit_id: u for u in self.pieces}
        pairs = close_pairs(self.src_box, self.Ws, self.Hs)
        rank = lambda u: u.rank if u.rank is not None else 10 ** 6
        pairs.sort(key=lambda p: (min(rank(by_id[p[0]]), rank(by_id[p[1]])), max(rank(by_id[p[0]]), rank(by_id[p[1]]))))
        changed = False
        for _ in range(2):
            for a, b, ax, g in pairs:
                lead, foll = sorted((by_id[a], by_id[b]), key=lambda u: (rank(u), u.unit_id))
                if lead.role in _PHOTO_ROLES or foll.role in _PHOTO_ROLES or foll.role == "headline":
                    continue  # a photo off its caption is a re-layout, not a split group
                if self._pull(lead, foll, g):
                    changed = True
        return changed

    def _scale_of(self, unit: Unit, boxes: dict[str, Box]) -> float:
        fr = [self.placed[i]["font_size"] / self.src[i]["font_size"] for i in unit.element_ids
              if self.src[i]["type"] == "text" and self.src[i].get("font_size")]
        if fr:
            return max(fr)
        a0, a1 = _area(self.src_box[unit.unit_id]), _area(boxes[unit.unit_id])
        return math.sqrt(a1 / a0) if a0 > 0 and a1 > 0 else 1.0

    def _pull(self, lead: Unit, foll: Unit, g_src: float) -> bool:
        boxes = self.boxes()
        lb, fb = boxes[lead.unit_id], boxes[foll.unit_id]
        want = g_src * math.sqrt(self._scale_of(lead, boxes) * self._scale_of(foll, boxes))
        now = _gap(lb, fb)
        if now <= SPLIT_RATIO * want or now <= want + SPLIT_SLACK * self.short:
            return False
        ax = next((a for a in ("y", "x") if _across(lb, fb, a)), None)
        if ax is None:
            return False
        (l0, l1), (f0, f1) = _span(lb, ax), _span(fb, ax)
        cur = f0 - l1 if f0 >= l1 else l0 - f1  # gap along the axis
        if cur <= want:
            return False
        sign = -1.0 if f0 >= l1 else 1.0         # follower after the leader: move back toward it
        decos = self.attached(foll, boxes)
        ids = list(foll.element_ids) + decos
        for frac in SHIFT_FRACTIONS:
            d = sign * (cur - want) * frac
            dx, dy = (d, 0.0) if ax == "x" else (0.0, d)
            geo = self._shift_geo(ids, dx, dy)
            nb = (fb[0] + dx, fb[1] + dy, fb[2], fb[3])
            if not self._valid([foll], {**boxes, foll.unit_id: nb}, boxes, set(decos), near={lead.unit_id}):
                continue
            if not self._inside(geo):
                continue
            self.placed.update(geo)
            self.fixes.append(f"group:{foll.unit_id}->{lead.unit_id}")
            return True
        return False

    def _inside(self, geo: dict[str, dict]) -> bool:
        for g in geo.values():
            x, y, w, h = rotated_aabb(g)
            if x < -_EPS or y < -_EPS or x + w > self.W + _EPS or y + h > self.H + _EPS:
                return False
        return True

    # -- c. balance the leftover space ---------------------------------------
    def spread(self) -> bool:
        changed = False
        for ax in ("y", "x"):
            changed |= self._shift_axis(ax)
        changed |= self._grow()
        return changed

    def _headline_group(self, boxes: dict[str, Box]) -> set[str]:
        """The headline and every piece chained to it by close source pairs."""
        head = next((u.unit_id for u in self.pieces if u.role == "headline"), None)
        if head is None:
            return set()
        links: dict[str, set[str]] = {}
        for a, b, _, _ in close_pairs(self.src_box, self.Ws, self.Hs):
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
        seen, todo = {head}, [head]
        while todo:
            for n in links.get(todo.pop(), ()):
                if n not in seen:
                    seen.add(n)
                    todo.append(n)
        return seen

    def _movable(self, ax: str, boxes: dict[str, Box]) -> tuple[list[Unit], list[Unit]]:
        side, sside = (self.W, self.Ws) if ax == "x" else (self.H, self.Hs)
        head = self._headline_group(boxes)
        move, fixed = [], []
        for u in self.pieces:
            sb, tb = self.src_box[u.unit_id], boxes[u.unit_id]
            if spans(tb, ax, side) or spans(sb, ax, sside):
                fixed.append(u)
                continue
            lo, hi = _span(sb, ax)
            hug = min(lo, sside - hi) <= HUG * sside and _area(sb) <= HUG_AREA * self.Ws * self.Hs
            if hug and u.unit_id not in head:
                fixed.append(u)
            else:
                move.append(u)
        return move, fixed

    def _lane(self, block: Box, fixed: list[Unit], boxes: dict[str, Box], ax: str, side: float,
              block_src: Box | None = None, src: bool = False) -> tuple[float, float]:
        """The free interval for the block on `ax`: the canvas, narrowed by fixed
        pieces in its lane (overlapping it across `ax`) wholly before or after it."""
        lo, hi = 0.0, side
        b0, b1 = _span(block, ax)
        for u in fixed:
            fb = self.src_box[u.unit_id] if src else boxes[u.unit_id]
            if spans(fb, ax, side):
                continue
            c = "x" if ax == "y" else "y"
            (p0, p1), (q0, q1) = _span(block, c), _span(fb, c)
            if min(p1, q1) - max(p0, q0) <= 0:
                continue
            f0, f1 = _span(fb, ax)
            if f1 <= b0 + _EPS:
                lo = max(lo, f1)
            elif f0 >= b1 - _EPS:
                hi = min(hi, f0)
        return lo, hi

    def _shift_axis(self, ax: str) -> bool:
        boxes = self.boxes()
        move, fixed = self._movable(ax, boxes)
        if not move:
            return False
        side, sside = (self.W, self.Ws) if ax == "x" else (self.H, self.Hs)
        block = union_box([boxes[u.unit_id] for u in move])
        sblock = union_box([self.src_box[u.unit_id] for u in move])
        lo, hi = self._lane(block, fixed, boxes, ax, side)
        slo, shi = self._lane(sblock, fixed, boxes, ax, sside, src=True)
        b0, b1 = _span(block, ax)
        s0, s1 = _span(sblock, ax)
        lop, imb = lopsided(b0 - lo, hi - b1, side)
        slop, simb = lopsided(s0 - slo, shi - s1, sside)
        if not lop:
            return False
        if simb and (simb > 0) == (imb > 0) and abs(simb) >= IMB_SOURCE * abs(imb):
            return False  # the source was lopsided this way on purpose
        d_full = ((hi - b1) - (b0 - lo)) / 2
        decos = sorted({d for u in move for d in self.attached(u, boxes)})
        ids = [i for u in move for i in u.element_ids] + decos
        for frac in SHIFT_FRACTIONS:
            d = d_full * frac
            dx, dy = (d, 0.0) if ax == "x" else (0.0, d)
            new = dict(boxes)
            for u in move:
                b = boxes[u.unit_id]
                new[u.unit_id] = (b[0] + dx, b[1] + dy, b[2], b[3])
            if not self._valid(move, new, boxes, set(decos)):
                continue
            geo = self._shift_geo(ids, dx, dy)
            if not self._inside(geo):
                continue
            self.placed.update(geo)
            self.fixes.append(f"shift:{ax}:{d:.1f}")
            return True
        return False

    def _grow(self) -> bool:
        """Uniform growth of the free block about its centre when it leaves much of the
        safe area unused on the short axis."""
        from engine.pipeline.layout.layers import scale_about
        boxes = self.boxes()
        sax = "x" if self.W <= self.H else "y"
        mx, fx = self._movable("x", boxes)
        my, fy = self._movable("y", boxes)
        ids_y = {u.unit_id for u in my}
        move = [u for u in mx if u.unit_id in ids_y]
        if not move:
            return False
        block = union_box([boxes[u.unit_id] for u in move])
        safe = self.W - 2 * self.mx if sax == "x" else self.H - 2 * self.my
        used = block[2] if sax == "x" else block[3]
        if safe <= 0 or used >= (1 - FILL_UNUSED) * safe:
            return False
        cx, cy = block[0] + block[2] / 2, block[1] + block[3] / 2
        decos = sorted({d for u in move for d in self.attached(u, boxes)})
        ids = [i for u in move for i in u.element_ids] + decos
        head = [i for u in self.plan.units if u.role == "headline" for i in u.element_ids
                if self.placed.get(i, {}).get("type") == "text"]
        for k in GROW_STEPS:
            if k > MAX_GROW:
                continue
            geo = {}
            ok = True
            for i in ids:
                g = scale_about(self.placed[i], k, cx, cy)
                if g["type"] == "text":
                    g = self._remeasure(g)
                    if g is None:
                        ok = False
                        break
                geo[i] = g
            if not ok or not self._inside(geo):
                continue
            trial = {**self.placed, **geo}
            new = dict(boxes)
            for u in move:
                new[u.unit_id] = self._ubox(u, trial)
            if not self._valid(move, new, boxes, set(decos)):
                continue
            if head and not self._head_largest(trial, head):
                continue
            self.placed.update(geo)
            self.fixes.append(f"grow:{k}")
            return True
        return False

    def _remeasure(self, g: dict) -> dict | None:
        """A grown text that no longer fits its box re-wraps in its (grown) width; None if a word can't fit."""
        f = g["font_size"]
        if self.m.fits(g, f, g["width"], g["height"]):
            return g
        if self.m.longest_word_width(g, f) > g["width"]:
            return None
        h = self.m.block_height(g, f, g["width"]) + 0.01
        out = dict(g, height=h)
        return out if self.m.fits(out, f, out["width"], out["height"]) else None

    def _head_largest(self, trial: dict, head: list[str]) -> bool:
        """No text that was at most the headline's size before the growth outgrows it."""
        before = max(self.placed[i]["font_size"] for i in head)
        after = max(trial[i]["font_size"] for i in head)
        return not any(e["type"] == "text" and i not in head and self.placed[i]["font_size"] <= before + 0.01
                       and e["font_size"] > after + 0.01 for i, e in trial.items())
