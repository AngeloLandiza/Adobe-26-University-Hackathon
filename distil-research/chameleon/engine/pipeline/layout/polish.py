"""Polish pass (CHAMELEON_POLISH=1; off by default, and with it off nothing
here runs and the engine's output is unchanged), plus the bisection v2
helpers it shares with strategies/bisection.py.

Bisection v2 (CHAMELEON_BISECT_V2) fixes the flaws the teacher names most,
but only inside bisection's own layouts. The templates (row / stack /
columns), the proportional reflow, the extra floor plans and the other
strategies still ship them: a round decoration squashed into an oval
behind a badge, a badge label parked at its 9 px floor on a 1600 px wide
banner, a logo that is a speck, copy jammed against the canvas edge, a
decoration that framed the product left behind in a corner.

`polish_all` takes every finished candidate (the safe fallback excluded, and
bisection's own candidates when v2 already built them) and repairs, in this
order, only what is broken:

1. round pills: a container label whose shape is a circle/ellipse keeps
   the source's aspect (the condensed pill squashes it); the shape becomes
   the smallest ellipse of the source aspect round its label;
2. logo floor: the logo's sqrt(area) reaches LOGO_MIN_LONG of the long side
   (the v2 floor, `logo_floor`);
3. readable badges and pills: a CTA or badge label reaches
   max(role floor, PILL_MIN_LONG x the long side), the size below which the
   validated "text too small" detector fires, capped at 1.5x its source size
   and at the hierarchy lock's role cap of the headline (CTA 0.6, badge 0.5);
4. safe margin: a content unit (that was clear of the edge in the source)
   whose ink is closer to an edge than the v2 margin is shifted inward (the
   units too close to one side together, so a row stays a row); a
   single text that can't shift is re-wrapped narrower at the same size;
   failing both, any unit but the headline shrinks into the safe area (an
   image to 0.8x, text to 0.85x and never under its floor, a logo never
   under the logo floor);
5. decorations: one that is squashed, orphaned from the unit it framed in
   the source, straddled by copy it didn't touch in the source, or that sat
   on a unit this pass moved, is re-placed the v2 way (round backdrop, frame
   that follows its unit, corner parking where no copy is half on it, the v1
   visible corner): the option with the fewest of those flaws, kept only if
   it has fewer than before (or no more, for a stale one) and a container
   still holds every unit it held.

Growth and moves never increase a unit's overlap with any other content
unit (with a small gap), a margin repair never breaks an edge or centre a
unit shared with another, never bring ink closer to an edge than the v2
margin (or than it already was), and the headline is never shrunk. Each
step is kept only if the candidate still passes every hard gate; a
candidate that already fails one is left as it is (and stays rejected), so
the grader chooses among the same candidates as without the pass. Never
raises: a candidate that can't be polished is returned as it was.
"""

from __future__ import annotations

import math
import os

from engine.pipeline.geometry import Box, overlap_area, rotated_aabb, union_box
from engine.pipeline.types import Candidate, PriorityPlan, Unit

ENV = "CHAMELEON_POLISH"

# -- shared with bisection v2 (strategies/bisection.py imports these) -------------
V2_MARGINS = (0.06, 0.09)  # share of the short side ...
V2_SIDE_MARGIN = 0.03      # ... and at least this share of the width (height) at the sides (top/bottom)
LOGO_MIN_LONG = 0.065      # logo sqrt(area) at least this share of the long side ...
LOGO_MIN_CAP = 0.5         # ... capped at this share of the short side (logo height)
FRAME_SCALES = (1.0, 0.9, 0.8, 0.7, 0.6, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5)  # frame sizes tried, x its mapped size
FRAME_SHARE = 0.3          # a decoration covering this share of a content unit (or vice versa) frames it
ROUND_SHAPES = ("ellipse", "circle")

# -- polish only ------------------------------------------------------------------
PILL_MIN_LONG = 0.011      # a CTA/badge label under this share of the long side is too small to read
PILL_ROLE_CAPS = {"cta": 0.6, "badge": 0.5}  # x the headline's size (the hierarchy lock's role caps)
MAX_GROWTH = 1.5           # text never grows past this x its source size (place.MAX_TEXT_GROWTH)
SQUASH_TOL = 0.02          # a round shape whose aspect moved more than this is squashed
GAP = 0.015                # least gap to a neighbour a moved unit didn't already touch, x the short side
EDGE_SOURCE = 0.02         # a unit closer than this x the source's short side to its edge bled on purpose
MIN_IMAGE_SHRINK = 0.8     # an image may shrink to this x to keep the margin ...
MIN_TEXT_SHRINK = 0.85     # ... and a text (never the headline, never under its floor) to this
GROWTH_FRACTIONS = (1.0, 0.75, 0.5, 0.25)  # share of the wanted growth tried, largest first
HUG_PAD = 1.1              # a round decoration hugging what it held: this x the tightest ellipse
ANCHORS = ((0.5, 0.5), (0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5))
_LAYER_ROLES = ("background", "decoration")
_EPS = 0.5
ALIGN_PX = 1.0             # edges/centres this close count as aligned


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def logo_floor(width: float, height: float, uw: float, uh: float, base: float) -> float:
    """(v2) Least logo height for a uw x uh logo: `base`, and a sqrt(area) of
    LOGO_MIN_LONG x the long side (how big it looks), capped at LOGO_MIN_CAP
    of the short side and 0.9 of the width."""
    want = LOGO_MIN_LONG * max(width, height) / math.sqrt(uw / uh)
    return max(base, min(want, LOGO_MIN_CAP * min(width, height), 0.9 * width * uh / uw))


def safe_margins(width: float, height: float, margin: float = V2_MARGINS[0]) -> tuple[float, float]:
    """(left/right, top/bottom) v2 safe margin in px."""
    m = margin * min(width, height)
    return max(m, V2_SIDE_MARGIN * width), max(m, V2_SIDE_MARGIN * height)


def straddles(shape: dict, box: Box, other: Box) -> bool:
    """Is `other` partly on and partly off `shape` drawn at `box`? A round
    shape is judged by its ellipse, anything else by its box."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return False
    if shape.get("shape_type") in ROUND_SHAPES:
        cx, cy, a, b = x + w / 2, y + h / 2, w / 2, h / 2
        corners = [(other[0] + i * other[2], other[1] + j * other[3]) for i in (0, 1) for j in (0, 1)]
        inside = all(((px - cx) / a) ** 2 + ((py - cy) / b) ** 2 <= 1 for px, py in corners)
        nx = (min(max(cx, other[0]), other[0] + other[2]) - cx) / a
        ny = (min(max(cy, other[1]), other[1] + other[3]) - cy) / b
        return not inside and nx * nx + ny * ny < 1
    ox = min(x + w, other[0] + other[2]) - max(x, other[0])
    oy = min(y + h, other[1] + other[3]) - max(y, other[1])
    if ox <= 0 or oy <= 0:
        return False
    return ox * oy < other[2] * other[3] - 1e-6


_V2_BUILDER = None


def v2_builder():
    """The engine's _Builder with layers placed the v2 way (made on first use:
    generate.py imports this module's callers lazily)."""
    global _V2_BUILDER
    if _V2_BUILDER is not None:
        return _V2_BUILDER
    from engine.pipeline.geometry import overlap_area
    from engine.pipeline.layout.generate import BACKDROP_TOL, _Builder
    from engine.pipeline.layout.layers import fit_inside, place_background, place_decoration, scale_about
    from engine.pipeline.occlusion import contains, is_covered, paint_order

    class V2Builder(_Builder):
        def place_layers(self) -> None:
            for unit in self.plan.units:
                if unit.role == "background":
                    self.placed.update((m["id"], place_background(m, self.width, self.height))
                                       for m in self._members(unit))
                elif unit.role == "decoration":
                    for m in self._members(unit):
                        self.placed[m["id"]] = (self._round_backdrop(m) or self._frame(m) or self._park(m)
                                                or self._visible_decoration(m))

        def _is_content(self, eid: str) -> bool:
            return self.plan.unit_for_element(eid).role not in ("background", "decoration")

        def _covers_under(self, shape: dict, box: Box, skip: set) -> bool:
            """Would `shape` at `box` paint over content drawn before it?"""
            order = paint_order(self.source["elements"])
            return any(overlap_area(box, rotated_aabb(g)) > 0 for eid, g in self.placed.items()
                       if eid not in skip and self._is_content(eid) and order[id(self.els[eid])] < order[id(shape)])

        def _round_backdrop(self, shape: dict) -> dict | None:
            """The v1 backdrop, but a round shape keeps its proportions: its box
            grows on the short side round the same centre (None if that leaves
            the canvas or covers content painted before it)."""
            out = self._backdrop(shape)
            if out is None or shape.get("shape_type") not in ROUND_SHAPES or not shape["height"] or not out["height"]:
                return out
            a0 = shape["width"] / shape["height"]
            x, y, w, h = out["x"], out["y"], out["width"], out["height"]
            if abs((w / h) / a0 - 1) <= 0.02:
                return out
            w2, h2 = (w, w / a0) if w / h > a0 else (h * a0, h)
            if w2 > self.width or h2 > self.height:
                return None
            nx = min(max(x + (w - w2) / 2, 0.0), self.width - w2)
            ny = min(max(y + (h - h2) / 2, 0.0), self.height - h2)
            order = paint_order(self.source["elements"])
            held = {t["id"] for t in self.source["elements"] if t["type"] == "text"
                    and order[id(t)] > order[id(shape)] and contains(rotated_aabb(shape), rotated_aabb(t), BACKDROP_TOL)}
            if self._covers_under(shape, (nx, ny, w2, h2), held):
                return None
            return dict(out, x=nx, y=ny, width=w2, height=h2)

        def _untouched(self, deco: dict, skip=None) -> list[Box]:
            """Placed boxes of the content units `deco` didn't touch in the
            source: they must not end up half on it."""
            d0 = rotated_aabb(deco)
            out = []
            for unit in self.plan.units:
                if unit is skip or unit.role in ("background", "decoration") or not unit.element_ids:
                    continue
                if not all(i in self.placed for i in unit.element_ids):
                    continue
                if overlap_area(d0, union_box([rotated_aabb(self.els[i]) for i in unit.element_ids])) <= 0:
                    out.append(union_box([rotated_aabb(self.placed[i]) for i in unit.element_ids]))
            return out

        def _park(self, deco: dict) -> dict | None:
            """The v1 corner parking (each corner, then smaller), but only where
            no copy it didn't touch in the source ends up half on it, nothing
            painted before it is covered, and it isn't hidden. None if nowhere."""
            others = self._untouched(deco)
            content = [self.placed[e["id"]] for e in self.source["elements"]
                       if e["id"] in self.placed and self._is_content(e["id"])]
            for k in (1.0, 0.75, 0.5):
                for corner in (None, "tl", "tr", "bl", "br"):
                    cand = place_decoration(deco, self.source["canvas"], self.width, self.height, corner)
                    if k != 1.0:
                        cx = 0.0 if cand["x"] + cand["width"] / 2 < self.width / 2 else self.width
                        cy = 0.0 if cand["y"] + cand["height"] / 2 < self.height / 2 else self.height
                        cand = fit_inside([scale_about(cand, k, cx, cy)], self.width, self.height)[0]
                    box = rotated_aabb(cand)
                    if any(straddles(deco, box, o) for o in others) or self._covers_under(deco, box, set()):
                        continue
                    ordered = sorted(content + [cand], key=lambda e: self._source_index[e["id"]])
                    if not is_covered(cand, content, paint_order(ordered)):
                        return cand
            return None

        def _frame(self, deco: dict) -> dict | None:
            """A decoration that framed a content unit in the source (covering
            FRAME_SHARE of the smaller of the two) follows it: same offset from
            the unit's centre and the unit's scale. None if there is no such
            unit, or the frame would leave it, cover content painted before it,
            or be hidden."""
            d0 = rotated_aabb(deco)
            best, best_share = None, 0.0
            for unit in self.plan.units:
                if unit.role in ("background", "decoration") or not unit.element_ids:
                    continue
                if not all(i in self.placed for i in unit.element_ids):
                    continue
                c0 = union_box([rotated_aabb(self.els[i]) for i in unit.element_ids])
                small = min(d0[2] * d0[3], c0[2] * c0[3])
                share = overlap_area(d0, c0) / small if small > 0 else 0.0
                if share >= FRAME_SHARE and share > best_share:
                    best, best_share = unit, share
            if best is None:
                return None
            c0 = union_box([rotated_aabb(self.els[i]) for i in best.element_ids])
            c1 = union_box([rotated_aabb(self.placed[i]) for i in best.element_ids])
            if c0[2] * c0[3] <= 0:
                return None
            k = math.sqrt(c1[2] * c1[3] / (c0[2] * c0[3]))
            cx = c1[0] + c1[2] / 2 + (d0[0] + d0[2] / 2 - c0[0] - c0[2] / 2) * k
            cy = c1[1] + c1[3] / 2 + (d0[1] + d0[3] / 2 - c0[1] - c0[3] / 2) * k
            w, h = deco["width"] * k, deco["height"] * k
            new = dict(deco, x=cx - w / 2, y=cy - h / 2, width=w, height=h)
            for key in ("corner_radius", "stroke_width"):
                if deco.get(key):
                    new[key] = deco[key] * k
            content = [self.placed[e["id"]] for e in self.source["elements"]
                       if e["id"] in self.placed and self._is_content(e["id"])]
            others = self._untouched(deco, skip=best)
            # Its own size first; smaller (to 0.6x) if that runs into
            # other copy; larger (to 1.5x) if a frame wholly inside the
            # unit it sits behind would count as hidden (G9).
            for grow in FRAME_SCALES:
                cand = scale_about(new, grow, new["x"] + new["width"] / 2, new["y"] + new["height"] / 2)
                cand = fit_inside([cand], self.width, self.height)[0]
                box = rotated_aabb(cand)
                if overlap_area(box, c1) <= 0 or self._covers_under(deco, box, set(best.element_ids)):
                    continue
                if any(straddles(deco, box, o) for o in others):
                    continue
                ordered = sorted(content + [cand], key=lambda e: self._source_index[e["id"]])
                if not is_covered(cand, content, paint_order(ordered)):
                    return cand
            return None

    _V2_BUILDER = V2Builder
    return _V2_BUILDER


# ================================================================== the polish pass

def _skip(cand: Candidate) -> bool:
    """The safe fallback (uniform shrink: nothing squashed or moved) and
    bisection's candidates when v2 built them with these very fixes."""
    return cand.is_fallback or (cand.template == "bisection" and os.environ.get("CHAMELEON_BISECT_V2") == "1")


def polish_all(source: dict, target: dict, plan: PriorityPlan, measurer, candidates: list[Candidate]) -> list[Candidate]:
    out = []
    for cand in candidates:
        if _skip(cand):
            out.append(cand)
            continue
        try:
            out.append(polish(source, target, plan, measurer, cand))
        except Exception:  # polish never sinks a candidate
            out.append(cand)
    return out


def polish(source: dict, target: dict, plan: PriorityPlan, measurer, cand: Candidate) -> Candidate:
    """`cand` with its broken details repaired (same id, template, params and
    degraded units), or `cand` itself if nothing needed or survived repair."""
    p = _Polisher(source, target, plan, measurer, cand)
    if not p.content or p.base_failed:  # a candidate the gates reject stays rejected: polish adds no candidates
        return cand
    for step in (p.round_pills, p.logo, p.pills, p.margins):
        saved = (dict(p.placed), set(p.changed), list(p.fixes))
        if step() and not p.gates_ok():
            p.placed, p.changed, p.fixes = saved
    p.decorations()
    if not p.fixes:
        return cand
    design = p.design(final=True)
    if design is None or not p.gates_ok(design):
        return cand
    return Candidate(cand.candidate_id, design, cand.template, cand.params, list(cand.degraded_unit_ids),
                     cand.is_fallback)


def _inflate(b: Box, g: float) -> Box:
    return (b[0] - g, b[1] - g, b[2] + 2 * g, b[3] + 2 * g)


class _Polisher:
    def __init__(self, source: dict, target: dict, plan: PriorityPlan, measurer, cand: Candidate):
        from engine.pipeline.grading.gates import check_gates
        self.check_gates = check_gates
        self.source, self.target, self.plan, self.m, self.cand = source, target, plan, measurer, cand
        self.W, self.H = float(target["width"]), float(target["height"])
        self.short, self.long = min(self.W, self.H), max(self.W, self.H)
        self.src = {e["id"]: e for e in source["elements"]}
        self.placed = {e["id"]: dict(e) for e in cand.design["elements"]}
        self.content = [u for u in plan.ranked_units()
                        if u.role not in _LAYER_ROLES and u.element_ids and all(i in self.placed for i in u.element_ids)]
        self.decos = [i for u in plan.units if u.role == "decoration" for i in u.element_ids if i in self.placed]
        self.mx, self.my = safe_margins(self.W, self.H)
        self.gap = GAP * self.short
        self.base_failed = set(check_gates(source, cand.design, target, plan, measurer))
        self.changed: set[str] = set()   # unit ids this pass moved or resized
        self.fixes: list[str] = []
        head = next((u for u in self.content if u.role == "headline"), None)
        self.head_font = max((self.placed[i]["font_size"] for i in head.element_ids
                              if self.placed[i]["type"] == "text"), default=None) if head else None

    # -- bookkeeping ---------------------------------------------------------
    def design(self, final: bool = False) -> dict | None:
        if final:
            from engine.pipeline.layout.normalize import finalize
            return finalize(self.source, self.placed, self.target)
        return {"canvas": dict(self.cand.design["canvas"]),
                "elements": [self.placed[e["id"]] for e in self.source["elements"]]}

    def gates_ok(self, design: dict | None = None) -> bool:
        design = design if design is not None else self.design(final=True)
        if design is None:
            return False
        return set(self.check_gates(self.source, design, self.target, self.plan, self.m)) <= self.base_failed

    def ink(self, el: dict) -> Box:
        if el["type"] != "text" or el.get("rotation"):
            return rotated_aabb(el)
        widths = self.m.line_widths(el, el["font_size"], el["width"]) or [0.0]
        w = min(max(widths), el["width"])
        align = el.get("text_align") or "left"
        x = el["x"] + (el["width"] - w) / 2 if align == "center" else \
            el["x"] + el["width"] - w if align == "right" else el["x"]
        return (x, el["y"], w, min(el["height"], self.m.block_height(el, el["font_size"], el["width"])))

    def ubox(self, unit: Unit, geo: dict | None = None) -> Box:
        geo = geo or self.placed
        return union_box([self.ink(geo[i]) for i in unit.element_ids])

    def _clear(self, unit: Unit, new: Box, old: Box) -> bool:
        """No other content unit gains overlap with `unit` (a gap of GAP x the short side),
        and a decoration (a card, a band) holding the unit still holds it."""
        if not self._held_ok(new, old):
            return False
        for v in self.content:
            if v is unit:
                continue
            vb = self.ubox(v)
            if overlap_area(_inflate(new, self.gap), vb) > overlap_area(_inflate(old, self.gap), vb) + 1e-6:
                return False
        return True

    def _held_ok(self, new: Box, old: Box) -> bool:
        """A decoration holding the old box (at least half of it on it) holds the new one."""
        for did in self.decos:
            db = rotated_aabb(self.placed[did])
            if overlap_area(db, old) >= 0.5 * old[2] * old[3] and overlap_area(db, new) < 0.5 * new[2] * new[3]:
                return False
        return True

    def _safe(self, new: Box, old: Box) -> bool:
        """Ink no closer to an edge than the v2 margin, or than it already was."""
        W, H = self.W, self.H
        for d_new, d_old, m in ((new[0], old[0], self.mx), (W - new[0] - new[2], W - old[0] - old[2], self.mx),
                                (new[1], old[1], self.my), (H - new[1] - new[3], H - old[1] - old[3], self.my)):
            if d_new < min(m, d_old) - _EPS:
                return False
        return True

    def _text_ok(self, geo: dict) -> bool:
        return all(g["type"] != "text" or self.m.fits(g, g["font_size"], g["width"], g["height"]) for g in geo.values())

    def _commit(self, unit: Unit, geo: dict, fix: str) -> None:
        self.placed.update(geo)
        self.changed.add(unit.unit_id)
        self.fixes.append(fix)

    def _shift_in(self, box: Box, old: Box) -> tuple[float, float]:
        """Shift that brings `box` back inside the safe area on sides where `old` was inside it."""
        W, H = self.W, self.H
        dx = dy = 0.0
        lo_x, hi_x = min(self.mx, old[0]), W - min(self.mx, W - old[0] - old[2])
        lo_y, hi_y = min(self.my, old[1]), H - min(self.my, H - old[1] - old[3])
        if box[2] <= hi_x - lo_x:
            dx = max(0.0, lo_x - box[0]) - max(0.0, box[0] + box[2] - hi_x)
        if box[3] <= hi_y - lo_y:
            dy = max(0.0, lo_y - box[1]) - max(0.0, box[1] + box[3] - hi_y)
        return dx, dy

    def _grow(self, unit: Unit, k_want: float, fix: str) -> bool:
        """Scale `unit` by up to k_want about one of ANCHORS of its box (then
        nudged back inside the safe area), the largest share of the growth
        that stays clear of other content; True if it grew."""
        from engine.pipeline.layout.layers import scale_about
        if k_want <= 1.01:
            return False
        members = {i: self.placed[i] for i in unit.element_ids}
        old = self.ubox(unit)
        ex, ey, ew, eh = union_box([rotated_aabb(g) for g in members.values()])
        for frac in GROWTH_FRACTIONS:
            k = 1 + (k_want - 1) * frac
            if k <= 1.01:
                break
            for ax, ay in ANCHORS:
                px, py = ex + ax * ew, ey + ay * eh
                geo = {i: scale_about(g, k, px, py) for i, g in members.items()}
                for i, g in geo.items():
                    if g["type"] == "text":
                        g["height"] = max(g["height"], self.m.block_height(g, g["font_size"], g["width"]))
                box = self.ubox(unit, geo)
                dx, dy = self._shift_in(box, old)
                if dx or dy:
                    geo = {i: dict(g, x=g["x"] + dx, y=g["y"] + dy) for i, g in geo.items()}
                    box = (box[0] + dx, box[1] + dy, box[2], box[3])
                eb = union_box([rotated_aabb(g) for g in geo.values()])
                if eb[0] < -_EPS or eb[1] < -_EPS or eb[0] + eb[2] > self.W + _EPS or eb[1] + eb[3] > self.H + _EPS:
                    continue
                if self._safe(box, old) and self._clear(unit, box, old) and self._text_ok(geo):
                    self._commit(unit, geo, fix)
                    return True
        return False

    # -- 1. round pills --------------------------------------------------------
    def round_pills(self) -> bool:
        done = False
        for unit in self.content:
            if unit.kind != "container_label":
                continue
            shape = next((self.placed[i] for i in unit.element_ids if self.placed[i]["type"] == "shape"), None)
            label = next((self.placed[i] for i in unit.element_ids if self.placed[i]["type"] == "text"), None)
            if shape is None or label is None or shape.get("shape_type") not in ROUND_SHAPES or shape.get("rotation"):
                continue
            s0 = self.src[shape["id"]]
            if not s0["height"] or not shape["height"]:
                continue
            a0 = s0["width"] / s0["height"]
            if abs((shape["width"] / shape["height"]) / a0 - 1) <= SQUASH_TOL:
                continue
            lx, ly, lw, lh = label["x"], label["y"], label["width"], label["height"]
            b = math.sqrt((lw / (2 * a0)) ** 2 + (lh / 2) ** 2)  # smallest ellipse of aspect a0 round the label box
            w, h = 2 * a0 * b, 2 * b
            cx, cy = lx + lw / 2, ly + lh / 2
            new = dict(shape, x=cx - w / 2, y=cy - h / 2, width=w, height=h)
            if shape.get("corner_radius"):
                new["corner_radius"] = shape["corner_radius"] * h / shape["height"]
            geo = {shape["id"]: new, label["id"]: label}
            old, box = self.ubox(unit), self.ubox(unit, geo)
            dx, dy = self._shift_in(box, old)
            geo = {i: dict(g, x=g["x"] + dx, y=g["y"] + dy) for i, g in geo.items()}
            box = (box[0] + dx, box[1] + dy, box[2], box[3])
            if box[0] < 0 or box[1] < 0 or box[0] + box[2] > self.W or box[1] + box[3] > self.H:
                continue
            if self._safe(box, old) and self._clear(unit, box, old):
                self._commit(unit, geo, "round_pill")
                done = True
        return done

    # -- 2. logo floor -----------------------------------------------------------
    def logo(self) -> bool:
        from engine.pipeline.layout.strategies.bisection import logo_min_height
        done = False
        for unit in self.content:
            if unit.role != "logo":
                continue
            _, _, uw, uh = union_box([rotated_aabb(self.placed[i]) for i in unit.element_ids])
            if uw <= 0 or uh <= 0:
                continue
            want = logo_floor(self.W, self.H, uw, uh, logo_min_height(self.W, self.H))
            if uh < want - 0.05:
                done |= self._grow(unit, want / uh, "logo_floor")
        return done

    # -- 3. readable badges and pills ------------------------------------------------
    def pills(self) -> bool:
        from engine.pipeline.layout.place import floor_px
        done = False
        for unit in self.content:
            if unit.role not in PILL_ROLE_CAPS:
                continue
            texts = [i for i in unit.element_ids if self.placed[i]["type"] == "text"]
            if not texts:
                continue
            f = max(self.placed[i]["font_size"] for i in texts)
            f_src = max(self.src[i]["font_size"] for i in texts)
            want = max(floor_px(unit.role, self.W, self.H), PILL_MIN_LONG * self.long)
            cap = f_src * MAX_GROWTH
            if self.head_font:
                cap = min(cap, PILL_ROLE_CAPS[unit.role] * self.head_font)
            want = min(want, cap)
            if f > 0 and f < want - 0.05:
                done |= self._grow(unit, want / f, f"{unit.role}_size")
        return done

    # -- 4. safe margin ----------------------------------------------------------
    def _bled(self, unit: Unit) -> bool:
        """Did the unit sit at (or over) an edge in the source? Then it bleeds on purpose."""
        sw, sh = self.source["canvas"]["width"], self.source["canvas"]["height"]
        x, y, w, h = union_box([rotated_aabb(self.src[i]) for i in unit.element_ids])
        e = EDGE_SOURCE * min(sw, sh)
        return x < e or y < e or x + w > sw - e or y + h > sh - e

    def _aligned(self, a: Box, b: Box) -> set[str]:
        """Which edges/centres two boxes share (within ALIGN_PX)."""
        out = set()
        for name, fa, fb in (("left", a[0], b[0]), ("right", a[0] + a[2], b[0] + b[2]),
                             ("cx", a[0] + a[2] / 2, b[0] + b[2] / 2), ("top", a[1], b[1]),
                             ("bottom", a[1] + a[3], b[1] + b[3]), ("cy", a[1] + a[3] / 2, b[1] + b[3] / 2)):
            if abs(fa - fb) <= ALIGN_PX:
                out.add(name)
        return out

    def _keeps_alignment(self, moved: dict[str, Box], old: dict[str, Box]) -> bool:
        """Every edge/centre a moved unit shared with an unmoved one is still shared."""
        for u in self.content:
            if u.unit_id not in moved:
                continue
            for v in self.content:
                if v.unit_id in moved:
                    continue
                vb = self.ubox(v)
                if not self._aligned(old[u.unit_id], vb) <= self._aligned(moved[u.unit_id], vb):
                    return False
        return True

    def _needs_margin(self, unit: Unit) -> bool:
        b = self.ubox(unit)
        return not (b[0] >= self.mx - _EPS and b[1] >= self.my - _EPS and b[0] + b[2] <= self.W - self.mx + _EPS
                    and b[1] + b[3] <= self.H - self.my + _EPS) and not self._bled(unit)

    def _side_shifts(self) -> bool:
        """Per side, every unit too close to it moves inward together (a row at
        the bottom stays a row), if none runs into anything else."""
        done = False
        W, H = self.W, self.H
        for side in ("left", "right", "top", "bottom"):
            group = []
            for u in self.content:
                if not self._needs_margin(u):
                    continue
                b = self.ubox(u)
                gap = {"left": b[0] - self.mx, "right": W - self.mx - b[0] - b[2],
                       "top": b[1] - self.my, "bottom": H - self.my - b[1] - b[3]}[side]
                if gap < -_EPS:
                    group.append((u, -gap))
            if len(group) < 2:
                continue  # a lone unit is handled by the per-unit pass
            d = max(v for _, v in group)
            dx = d if side == "left" else -d if side == "right" else 0.0
            dy = d if side == "top" else -d if side == "bottom" else 0.0
            ids = {u.unit_id for u, _ in group}
            old = {u.unit_id: self.ubox(u) for u, _ in group}
            new = {k: (b[0] + dx, b[1] + dy, b[2], b[3]) for k, b in old.items()}
            ok = all(self._safe(new[k], old[k]) for k in ids) and self._keeps_alignment(new, old)
            for u, _ in group:
                if not ok or not self._held_ok(new[u.unit_id], old[u.unit_id]):
                    ok = False
                    break
                for v in self.content:
                    if v.unit_id in ids:
                        continue
                    vb = self.ubox(v)
                    if overlap_area(_inflate(new[u.unit_id], self.gap), vb) > \
                            overlap_area(_inflate(old[u.unit_id], self.gap), vb) + 1e-6:
                        ok = False
                        break
            if ok:
                for u, _ in group:
                    self._commit(u, {i: dict(self.placed[i], x=self.placed[i]["x"] + dx, y=self.placed[i]["y"] + dy)
                                     for i in u.element_ids}, "margin")
                done = True
        return done

    def margins(self) -> bool:
        from engine.pipeline.layout.layers import scale_about
        done = self._side_shifts()
        for unit in self.content:
            old = self.ubox(unit)
            W, H = self.W, self.H
            if (old[0] >= self.mx - _EPS and old[1] >= self.my - _EPS and old[0] + old[2] <= W - self.mx + _EPS
                    and old[1] + old[3] <= H - self.my + _EPS) or self._bled(unit):
                continue
            members = {i: self.placed[i] for i in unit.element_ids}
            tries = []
            # a: shift inward
            dx = max(0.0, self.mx - old[0]) - max(0.0, old[0] + old[2] - (W - self.mx)) if old[2] <= W - 2 * self.mx else 0.0
            dy = max(0.0, self.my - old[1]) - max(0.0, old[1] + old[3] - (H - self.my)) if old[3] <= H - 2 * self.my else 0.0
            if dx or dy:
                tries.append({i: dict(g, x=g["x"] + dx, y=g["y"] + dy) for i, g in members.items()})
            # b: a single text re-wrapped narrower at the same size
            if len(members) == 1 and next(iter(members.values()))["type"] == "text":
                el = next(iter(members.values()))
                if not el.get("rotation"):
                    x0, x1 = max(el["x"], self.mx), min(el["x"] + el["width"], W - self.mx)
                    if x1 - x0 > 0 and x1 - x0 < el["width"]:
                        h = self.m.block_height(el, el["font_size"], x1 - x0)
                        cy = el["y"] + el["height"] / 2
                        y0 = min(max(cy - h / 2, self.my), H - self.my - h)
                        new = dict(el, x=x0, y=y0, width=x1 - x0, height=h)
                        if self.m.fits(new, new["font_size"], new["width"], new["height"]):
                            tries.append({el["id"]: new})
            # c: anything but the headline shrinks into the safe area: an image to
            # MIN_IMAGE_SHRINK, text to MIN_TEXT_SHRINK and never under its floor, a logo not under its floor
            if unit.role != "headline":
                x0, y0 = max(old[0], self.mx), max(old[1], self.my)
                x1, y1 = min(old[0] + old[2], W - self.mx), min(old[1] + old[3], H - self.my)
                if x1 > x0 and y1 > y0 and old[2] > 0 and old[3] > 0:
                    k = min(1.0, (x1 - x0) / old[2], (y1 - y0) / old[3])
                    if self._min_scale(unit) <= k < 1:
                        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                        ocx, ocy = old[0] + old[2] / 2, old[1] + old[3] / 2
                        geo = {i: scale_about(g, k, ocx, ocy) for i, g in members.items()}
                        geo = {i: dict(g, x=g["x"] + cx - ocx, y=g["y"] + cy - ocy) for i, g in geo.items()}
                        if self._text_ok(geo):
                            tries.append(geo)
            for geo in tries:
                box = self.ubox(unit, geo)
                if not self._clear(unit, box, old) or not self._keeps_alignment({unit.unit_id: box}, {unit.unit_id: old}):
                    continue
                inside = (box[0] >= self.mx - _EPS and box[1] >= self.my - _EPS
                          and box[0] + box[2] <= W - self.mx + _EPS and box[1] + box[3] <= H - self.my + _EPS)
                if inside or (self._safe(box, old) and self._edge(box) > self._edge(old) + 1):
                    self._commit(unit, geo, "margin")
                    done = True
                    break
        return done

    def _min_scale(self, unit: Unit) -> float:
        """Least scale a margin repair may give `unit` (1.0: none)."""
        from engine.pipeline.layout.place import floor_px
        from engine.pipeline.layout.strategies.bisection import logo_min_height
        members = [self.placed[i] for i in unit.element_ids]
        texts = [g for g in members if g["type"] == "text"]
        if not texts:
            k = MIN_IMAGE_SHRINK
            if unit.role == "logo":
                _, _, uw, uh = union_box([rotated_aabb(g) for g in members])
                if uw <= 0 or uh <= 0:
                    return 1.0
                k = max(k, logo_floor(self.W, self.H, uw, uh, logo_min_height(self.W, self.H)) / uh)
            return k
        floor = floor_px(unit.role, self.W, self.H)
        if unit.role in PILL_ROLE_CAPS:
            floor = max(floor, PILL_MIN_LONG * self.long)
        f = min(g["font_size"] for g in texts)
        return max(MIN_TEXT_SHRINK, floor / f if f > 0 else 1.0)

    def _edge(self, b: Box) -> float:
        """Least distance to an edge, relative to the margin on that axis (in px of the x margin)."""
        r = self.mx / self.my if self.my else 1.0
        return min(b[0], self.W - b[0] - b[2], (b[1]) * r, (self.H - b[1] - b[3]) * r)

    # -- 5. decorations ---------------------------------------------------------------
    def _deco_flaws(self, d0: dict, d1: dict, units: dict) -> int:
        """Squashed round shape + orphaned from what it framed + copy it didn't touch half on it."""
        n = 0
        if d0.get("shape_type") in ROUND_SHAPES and d0["height"] and d1["height"] and d0["width"]:
            if abs((d1["width"] / d1["height"]) / (d0["width"] / d0["height"]) - 1) > SQUASH_TOL:
                n += 1
        b0, b1 = rotated_aabb(d0), rotated_aabb(d1)
        framed = kept = 0
        for unit, (c0, c1) in units.items():
            small = min(b0[2] * b0[3], c0[2] * c0[3])
            touched = overlap_area(b0, c0) > 0
            if small > 0 and overlap_area(b0, c0) >= FRAME_SHARE * small:
                framed += 1
                kept += overlap_area(b1, c1) > 0
            if not touched and straddles(d1, b1, c1):
                n += 1
        if framed and not kept:
            n += 1
        return n

    def _hug(self, d0: dict) -> dict | None:
        """A round decoration that held content in the source (a circle behind
        a badge) as the smallest ellipse of its source aspect round that
        content's ink now, padded by HUG_PAD and kept on the canvas; None if
        it holds nothing, or can't hold it whole."""
        if d0.get("shape_type") not in ROUND_SHAPES or d0.get("rotation") or not d0["height"] or not d0["width"]:
            return None
        b0 = rotated_aabb(d0)
        held = [u for u in self.content
                if overlap_area(b0, (cb := union_box([rotated_aabb(self.src[i]) for i in u.element_ids])))
                >= 0.5 * cb[2] * cb[3] > 0]
        if not held:
            return None
        x, y, w, h = union_box([self.ubox(u) for u in held])
        a0 = d0["width"] / d0["height"]
        rb = HUG_PAD * math.sqrt((w / (2 * a0)) ** 2 + (h / 2) ** 2)
        nw, nh = 2 * a0 * rb, 2 * rb
        if nw > self.W or nh > self.H:
            return None
        nx = min(max(x + w / 2 - nw / 2, 0.0), self.W - nw)
        ny = min(max(y + h / 2 - nh / 2, 0.0), self.H - nh)
        if not (nx <= x and ny <= y and x + w <= nx + nw and y + h <= ny + nh):
            return None
        new = dict(d0, x=nx, y=ny, width=nw, height=nh)
        for key in ("corner_radius", "stroke_width"):
            if d0.get(key):
                new[key] = d0[key] * nh / d0["height"]
        return new

    def _holds(self, deco: dict, unit: Unit) -> bool:
        ub = self.ubox(unit)
        return ub[2] * ub[3] > 0 and overlap_area(rotated_aabb(deco), ub) >= 0.5 * ub[2] * ub[3]

    def _still_holds(self, d0: dict, cur: dict, new: dict) -> bool:
        """A container (it held a unit in the source) still holds every unit it holds now."""
        b0 = rotated_aabb(d0)
        for u in self.content:
            sb = union_box([rotated_aabb(self.src[i]) for i in u.element_ids])
            if sb[2] * sb[3] > 0 and overlap_area(b0, sb) >= 0.5 * sb[2] * sb[3]:
                if self._holds(cur, u) and not self._holds(new, u):
                    return False
        return True

    def decorations(self) -> None:
        if not self.decos:
            return
        src_boxes = {u.unit_id: union_box([rotated_aabb(self.src[i]) for i in u.element_ids]) for u in self.content}
        orig = {e["id"]: e for e in self.cand.design["elements"]}
        b = None
        for did in self.decos:
            units = {u.unit_id: (src_boxes[u.unit_id], self.ubox(u)) for u in self.content}
            cur, d0 = self.placed[did], self.src[did]
            before = self._deco_flaws(d0, cur, units)
            ob = rotated_aabb(orig[did])
            stale = any(overlap_area(ob, union_box([rotated_aabb(orig[i]) for i in u.element_ids])) > 0
                        for u in self.content if u.unit_id in self.changed)
            if not before and not stale:
                continue
            if b is None:
                b = v2_builder()(self.source, self.target, self.plan, self.m)
            b.placed = {i: g for i, g in self.placed.items() if i in {e for u in self.content for e in u.element_ids}}
            best, best_n = None, None
            for make in (b._round_backdrop, self._hug, b._frame, b._park, b._visible_decoration):
                try:
                    new = make(d0)
                except Exception:
                    new = None
                if new is None or not self._still_holds(d0, cur, new):
                    continue
                n = self._deco_flaws(d0, new, units)
                if best_n is None or n < best_n:
                    best, best_n = new, n
            if best is None or not (best_n < before or (stale and best_n <= before)):
                continue
            saved = self.placed[did]
            self.placed[did] = best
            if self.gates_ok():
                self.fixes.append("decoration")
            else:
                self.placed[did] = saved
