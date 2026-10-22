"""Geometry solver pass (CHAMELEON_SOLVER=1; off by default, and with it off
nothing here runs and the engine's output is unchanged).

"Structure from the candidate, geometry from a solver." Every layout the
engine builds (bisection, anchored, templates, reflow) already decides the
*structure*: what sits above or left of what, what is grouped, which edge a
piece hugs. What it gets wrong is the *distribution* of space: lopsided
margins, a dead band under the stack, uneven gaps, pieces almost but not
quite aligned. The content-aware sizer (FLEX) lost because it only removed
dead space; this pass keeps the structure as constraints and redistributes
the space with a small linear program (scipy HiGHS).

1. **Blocks.** The layout plan's content units (with CHAMELEON_LAYERS=1 a
   container and everything on it is one unit); units that overlap on the
   candidate (text set on a photo) merge into one rigid block. A decoration
   sitting on exactly one block moves and scales with it. Text boxes are
   first tightened to their inked lines (plus the text-slack margin; the
   wrap is unchanged), so the solver reasons about what is visible.
   Background layers and other decorations stay where they are.
2. **Topology** (read off the candidate): an XY-cut tree of the blocks
   (rows and columns, nested); every pair of blocks in consecutive children
   of a row/column keeps its order with a gap (hard); shared alignment
   lines (left/centre/right, top/middle/bottom) and a child centred in its
   row/column; small pieces that hugged an edge in the source and still do
   (pinned); blocks that span an axis or bleed off an edge (fixed there).
3. **LP** over x, y and one uniform scale s per block (images and groups
   keep their aspect by construction; the variables are blocks, not
   elements). Hard: inside the canvas, the safe margin (or no closer than
   the candidate was), order with gaps, the headline stays the largest text
   (s of any text block <= the headline block's), no text past 1.5x its
   source size, s >= 1 (nothing shrinks). Soft, L1, tiered weights:
   keep shared lines (W_KEEP) > pinned edges (W_PIN) > equal gaps within a
   row/column (two gap variables: gaps between source-close pairs may be
   smaller than the rest) and equal opposite margins of the free content
   (unless the source was lopsided the same way) (W_GAP, W_MARGIN) > snap
   nearly aligned lines (W_SNAP) > stay near the candidate's own block
   centres (W_FID) > grow toward the safe area (W_FILL, only when the
   content leaves FILL_UNUSED of the safe area unused on an axis).
   The candidate itself is a feasible point, so the LP never makes a
   candidate infeasible; a solve that changes nothing returns it as it was.
4. The solved candidate is kept only if it passes every hard gate the
   original passed (the engine re-checks in pick_best).

`solve_all` is the hook (generate.py, after the other passes). With
CHAMELEON_SOLVER=winner only the picked layout is re-solved instead
(engine/resize.py, after pick_best and the refiners, before text slack), so
the grader's picks stay those of the engine without the pass. Never raises.
`realize` (used by dev/distill/oracle_plan.py) solves a placement with
extra anchor pulls and a larger growth cap.

Sonnet's proportions (CHAMELEON_TARGETS=1 before grading, =winner after the
pick; layout/targets.py): every block whose units have a predicted size
(text by role, CTA pill, logo, hero) gets a target scale (the units'
priority-weighted geometric mean; none when they disagree) and a strong L1
pull (W_TARGET, the W_KEEP tier) toward it. Its scale may then go below 1
(down to T_S_MIN, never under a text's role floor or the v2 logo floor) or
past S_MAX (up to the target, T_S_MAX; text still at most 1.5x its source
size); hard rules unchanged otherwise, and the headline's font stays the
largest. A lone text that shrank is re-wrapped on fewer lines when it has
the room (T_REWRAP). Without predictions nothing changes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from engine.pipeline.geometry import Box, overlap_area, rotated_aabb, union_box
from engine.pipeline.types import Candidate, PriorityPlan

ENV = "CHAMELEON_SOLVER"
GAP_MIN = 0.02        # least gap between ordered blocks, share of the short side (or the candidate's, if smaller)
ALIGN_KEEP = 0.003    # lines this close (share of the long side, at least 1 px) are a shared axis: kept
SNAP = 0.02           # stacked blocks' lines this close (share of the long side) snap together
SPAN = 0.95           # a block covering this share of an axis spans it (a band): fixed on it
EDGE_TOL = 0.01       # share of the side: this close to an edge, a block bleeds off it
HUG = 0.1             # a small block this close to an edge in the source (share of the side) ...
HUG_AREA = 0.06       # ... (at most this share of the source canvas) ...
PIN_GAP = 2.5         # ... and within this x the safe margin of that edge on the candidate is pinned to it
S_MAX = 1.2           # a block grows at most this much ...
FILL_UNUSED = 0.25    # ... and only when the content leaves this share of the safe area unused on an axis
MAX_TEXT_GROWTH = 1.5  # no text past this x its source size (place.MAX_TEXT_GROWTH)
LOGO_MAX_GROWTH = 1.5  # no logo past this x its source size (generate.LOGO_MAX_GROWTH)
ATTACH = 0.3          # a decoration covering this share of a block (or the block of it) sits on it
IMB_SOURCE = 0.5      # the source lopsided the same way by this x the candidate's imbalance: kept lopsided
MIN_CHANGE = 0.5      # px; a solve moving nothing more than this returns the candidate unchanged
# objective tiers (per px of the short side; FILL per unit of scale)
W_KEEP = 1000.0
W_PIN = 300.0
W_GAP = 100.0
W_MARGIN = 100.0
W_SNAP = 30.0
W_ANCHOR = 30.0       # (realize) plan anchors
W_FID = 10.0
W_FILL = 1.0
# Sonnet's proportions (CHAMELEON_TARGETS, layout/targets.py): per-block scale pulled toward the predicted size
W_TARGET = 1000.0     # per px of the block's long side between its size and the target size
T_S_MIN = 0.6         # a block with a target may shrink to this x (never below its role floors) ...
T_S_MAX = 2.5         # ... and grow to this x (text still at most MAX_TEXT_GROWTH x its source size)
LOGO_MIN_PX = 12      # (targets) a logo never shrinks below this height (bisection.LOGO_MIN_PX) ...
T_SHRINK_IMAGES = True   # (targets) False: a block without text (hero, logo) never shrinks (worse on train: hero, logo)
T_REWRAP = True       # (targets) a shrunk lone text re-wraps into its pre-solve width when that saves a line
T_DEAD = 0.05         # a block within this ratio of its target is left alone (no pull)
_LAYER_ROLES = ("background", "decoration")
_EPS = 0.5


def enabled() -> bool:
    """CHAMELEON_SOLVER=1: every gate-passing candidate is re-solved before grading."""
    return os.environ.get(ENV) == "1"


def winner_only() -> bool:
    """CHAMELEON_SOLVER=winner: only the picked layout is re-solved (after pick_best), so no pick changes."""
    return os.environ.get(ENV) == "winner"


# ---------------------------------------------------------------- small helpers

def _span(b: Box, ax: str) -> tuple[float, float]:
    return (b[0], b[0] + b[2]) if ax == "x" else (b[1], b[1] + b[3])


def _area(b: Box) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def _other(ax: str) -> str:
    return "y" if ax == "x" else "x"


def _spans(b: Box, ax: str, side: float) -> bool:
    lo, hi = _span(b, ax)
    tol = EDGE_TOL * side
    return (hi - lo) >= SPAN * side or (lo <= tol and hi >= side - tol)


def _cross_overlap(a: Box, b: Box, ax: str) -> float:
    """Overlap of a and b on the axis across `ax`."""
    (a0, a1), (b0, b1) = _span(a, _other(ax)), _span(b, _other(ax))
    return min(a1, b1) - max(a0, b0)


# ---------------------------------------------------------------- blocks

@dataclass
class Block:
    bid: str
    unit_ids: list[str]
    content: list[str]                  # content element ids
    decos: list[str] = field(default_factory=list)  # attached decoration ids (move and scale with it)
    box: Box = (0.0, 0.0, 0.0, 0.0)     # extent of the content (tightened text boxes)
    full: Box = (0.0, 0.0, 0.0, 0.0)    # extent with the attached decorations
    role: str | None = None
    rank: float = math.inf
    texts: list[str] = field(default_factory=list)
    src_box: Box | None = None
    s_max: float = 1.0
    s_lo: float = 1.0                               # least scale (below 1 only with a target)
    t_ratio: float | None = None                    # (targets) scale that gives the predicted size
    font: float = 0.0                               # largest font in the block
    fixed: tuple[bool, bool] = (False, False)       # (x, y)
    bleed: tuple[bool, bool, bool, bool] = (False, False, False, False)  # l, t, r, b
    pin: tuple[bool, bool, bool, bool] = (False, False, False, False)


@dataclass
class Node:
    kind: str                       # leaf | stack | free
    axis: str | None = None         # stack axis
    children: list["Node"] = field(default_factory=list)
    leaf: int | None = None

    def leaves(self) -> list[int]:
        if self.kind == "leaf":
            return [self.leaf]
        return [i for c in self.children for i in c.leaves()]


@dataclass
class Opts:
    s_max: float = S_MAX
    fill_unused: float = FILL_UNUSED
    anchors: dict = field(default_factory=dict)   # block id -> {"x": "start|centre|end", "y": ...} (realize)
    gap_min_hard: bool = False                    # (realize) gaps >= GAP_MIN even if the start was closer
    fidelity: float = W_FID
    targets: dict | None = None                   # (CHAMELEON_TARGETS) layout/targets.predict output
    w_target: float | None = None                 # (targets) pull weight; None: W_TARGET


def _layout_plan(source: dict, plan: PriorityPlan, target: dict) -> PriorityPlan:
    if os.environ.get("CHAMELEON_LAYERS") == "1":
        try:
            from engine.pipeline.layout import layer_groups
            return layer_groups.detect(source, plan, target).plan
        except Exception:
            pass
    return plan


class _Model:
    """A candidate read as blocks + topology, ready to solve."""

    def __init__(self, source: dict, target: dict, plan: PriorityPlan, layout: PriorityPlan, measurer,
                 design: dict, opts: Opts):
        from engine.pipeline.layout.polish import safe_margins
        self.source, self.target, self.plan, self.layout, self.m, self.opts = source, target, plan, layout, measurer, opts
        self.W, self.H = float(target["width"]), float(target["height"])
        self.Ws, self.Hs = float(source["canvas"]["width"]), float(source["canvas"]["height"])
        self.short, self.long = min(self.W, self.H), max(self.W, self.H)
        self.src = {e["id"]: e for e in source["elements"]}
        self.placed = {e["id"]: dict(e) for e in design["elements"]}
        self.untight = {e["id"]: e for e in design["elements"]}   # before the text boxes are tightened
        self.mx, self.my = safe_margins(self.W, self.H)
        self.tightened: set[str] = set()
        self._tighten()
        self.blocks = self._blocks()
        for b in self.blocks:
            b.font = max((self.placed[i].get("font_size") or 0.0 for i in b.texts), default=0.0)
        if opts.targets:
            self._target_ratios(opts.targets)

    # -- text boxes tightened to their ink -----------------------------------
    def _tighten(self) -> None:
        for u in self.layout.units:
            if u.role in _LAYER_ROLES:
                continue
            for i in u.element_ids:
                el = self.placed.get(i)
                if el is None or el["type"] != "text" or el.get("rotation"):
                    continue
                new = tighten_text(el, self.m)
                if new is not el:
                    self.placed[i] = new
                    self.tightened.add(i)

    def _ebox(self, i: str) -> Box:
        return rotated_aabb(self.placed[i])

    def _blocks(self) -> list[Block]:
        units = [u for u in self.layout.ranked_units()
                 if u.role not in _LAYER_ROLES and u.element_ids and all(i in self.placed for i in u.element_ids)]
        boxes = {u.unit_id: union_box([self._ebox(i) for i in u.element_ids]) for u in units}
        # merge overlapping units (text set on a photo): one rigid block
        parent = {u.unit_id: u.unit_id for u in units}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i, a in enumerate(units):
            for b in units[i + 1:]:
                A, B = boxes[a.unit_id], boxes[b.unit_id]
                if overlap_area((A[0] + _EPS, A[1] + _EPS, A[2] - 2 * _EPS, A[3] - 2 * _EPS), B) > 0:
                    parent[find(a.unit_id)] = find(b.unit_id)
        groups: dict[str, list] = {}
        for u in units:
            groups.setdefault(find(u.unit_id), []).append(u)
        blocks = []
        for members in groups.values():
            members.sort(key=lambda u: (u.rank if u.rank is not None else math.inf, u.unit_id))
            ids = [i for u in members for i in u.element_ids]
            top = members[0]
            b = Block("+".join(u.unit_id for u in members), [u.unit_id for u in members], ids,
                      role=top.role, rank=top.rank if top.rank is not None else math.inf)
            b.box = union_box([boxes[u.unit_id] for u in members])
            b.texts = [i for i in ids if self.placed[i]["type"] == "text"]
            try:
                b.src_box = union_box([rotated_aabb(self.src[i]) for i in ids])
            except KeyError:
                b.src_box = None
            blocks.append(b)
        blocks.sort(key=lambda b: (b.rank, b.bid))
        self._attach(blocks)
        for b in blocks:
            b.full = union_box([b.box] + [self._ebox(d) for d in b.decos])
            self._flags(b)
        return blocks

    def _attach(self, blocks: list[Block]) -> None:
        decos = [i for u in self.layout.units if u.role == "decoration" for i in u.element_ids if i in self.placed]
        for d in decos:
            db = self._ebox(d)
            if _spans(db, "x", self.W) or _spans(db, "y", self.H) or _area(db) >= 0.5 * self.W * self.H:
                continue

            def on(b: Box) -> bool:
                small = min(_area(db), _area(b))
                return small > 0 and overlap_area(db, b) >= ATTACH * small
            hits = [b for b in blocks if on(b.box)]
            if len(hits) == 1:
                hits[0].decos.append(d)

    def _flags(self, b: Block) -> None:
        W, H = self.W, self.H
        sx, sy = _spans(b.box, "x", W), _spans(b.box, "y", H)
        x0, y0, w, h = b.full
        tx, ty = EDGE_TOL * W, EDGE_TOL * H
        b.bleed = (x0 <= tx, y0 <= ty, x0 + w >= W - tx, y0 + h >= H - ty)
        b.fixed = (sx, sy)
        pin = [False] * 4
        sb = b.src_box
        if sb is not None and _area(sb) <= HUG_AREA * self.Ws * self.Hs:
            bx, by, bw, bh = b.box
            gaps_src = (sb[0], sb[1], self.Ws - sb[0] - sb[2], self.Hs - sb[1] - sb[3])
            sides_src = (self.Ws, self.Hs, self.Ws, self.Hs)
            gaps = (bx, by, W - bx - bw, H - by - bh)
            mg = (self.mx, self.my, self.mx, self.my)
            for k in range(4):
                if gaps_src[k] <= HUG * sides_src[k] and gaps[k] <= PIN_GAP * mg[k] and gaps[k] < gaps[(k + 2) % 4]:
                    pin[k] = True
        b.pin = tuple(pin)
        b.s_max = 1.0 if (sx or sy) else self._s_cap(b)

    def _s_cap(self, b: Block) -> float:
        cap = self.opts.s_max
        for i in b.texts:
            s0 = self.src.get(i, {}).get("font_size")
            f = self.placed[i]["font_size"]
            if s0 and f:
                cap = min(cap, max(1.0, MAX_TEXT_GROWTH * s0 / f))
        if b.role == "logo":
            for i in b.content:
                e0, e = self.src.get(i), self.placed[i]
                if e0 and e["width"] > 0 and e["height"] > 0:
                    cap = min(cap, max(1.0, LOGO_MAX_GROWTH * min(e0["width"] / e["width"], e0["height"] / e["height"])))
        return max(1.0, cap)


    # -- Sonnet's proportions (targets) --------------------------------------------
    def _floor_scale(self, b: "Block") -> float:
        """Least scale keeping every text of the block at its role floor (and a logo at its v2 floor)."""
        from engine.pipeline.layout.place import floor_px
        lo = T_S_MIN
        for i in b.texts:
            f = self.placed[i].get("font_size") or 0.0
            try:
                role = self.plan.unit_for_element(i).role or "other_text"
            except KeyError:
                role = "other_text"
            fl = floor_px(role, self.W, self.H)
            lo = max(lo, 1.0 if f <= fl else fl / f)
        if b.role == "logo":
            from engine.pipeline.layout.polish import logo_floor
            x, y, w, h = b.box
            if w > 0 and h > 0:
                fl = logo_floor(self.W, self.H, w, h, min(LOGO_MIN_PX, h))
                lo = max(lo, 1.0 if h <= fl else fl / h)
        return min(1.0, lo)

    def _target_ratios(self, T: dict) -> None:
        from engine.pipeline.layout.targets import group_ratio
        for b in self.blocks:
            if b.fixed[0] or b.fixed[1]:
                continue
            # the priority-plan units inside the block (a layer group holds several)
            ids = set(b.content)
            members = [u for u in self.plan.ranked_units() if u.element_ids and set(u.element_ids) <= ids]
            r = group_ratio(members, self.placed, T, self.W, self.H)
            if r is None or abs(r - 1.0) <= T_DEAD:
                continue
            lo = self._floor_scale(b)
            hi = T_S_MAX
            for i in b.texts:
                s0 = self.src.get(i, {}).get("font_size")
                f = self.placed[i].get("font_size")
                if s0 and f:
                    hi = min(hi, max(1.0, MAX_TEXT_GROWTH * s0 / f))
            if b.role == "logo":
                for i in b.content:
                    e0, e = self.src.get(i), self.placed[i]
                    if e0 and e["width"] > 0 and e["height"] > 0:
                        hi = min(hi, max(1.0, LOGO_MAX_GROWTH * min(e0["width"] / e["width"], e0["height"] / e["height"])))
            if not b.texts and not T_SHRINK_IMAGES:  # pictures may grow toward their target, never shrink
                lo = 1.0
            b.t_ratio = min(max(r, lo), max(hi, 1.0))
            if abs(b.t_ratio - 1.0) <= 1e-3:
                b.t_ratio = None
                continue
            b.s_lo = min(1.0, lo) if b.t_ratio < 1.0 else 1.0
            b.s_max = max(b.s_max, b.t_ratio)


def tighten_text(el: dict, m) -> dict:
    """A text box narrowed to its widest inked line plus the text-slack margin
    (ink in place: left keeps x, right keeps the right edge, centre both) and
    cut to its drawn block; the same wrap, or `el` itself."""
    from engine.pipeline.layout.text_slack import slack_px
    f, w, h = el["font_size"], el["width"], el["height"]
    widths = m.line_widths(el, f, w)
    if not widths:
        return el
    ink = max(widths)
    nw = min(w, ink + slack_px(ink))
    lines = m.line_count(el, f, w)
    out = dict(el)
    if nw < w - _EPS and m.line_count(el, f, nw) == lines:
        align = el.get("text_align") or "left"
        if align == "center":
            out["x"] = el["x"] + (w - nw) / 2
        elif align == "right":
            out["x"] = el["x"] + w - nw
        out["width"] = nw
    block = m.block_height(out, f, out["width"])
    if out["height"] > block + 1.0:
        out["height"] = block + _EPS
    if out == el or not m.fits(out, f, out["width"], out["height"]) or m.line_count(out, f, out["width"]) != lines:
        return el
    return out


# ---------------------------------------------------------------- topology

def xy_tree(idx: list[int], boxes: list[Box], W: float, H: float) -> Node:
    """Recursive XY-cut of the blocks: a row or column wherever a clean cut
    exists (the axis with the widest gap), a `free` node where none does."""
    if len(idx) == 1:
        return Node("leaf", leaf=idx[0])
    best = None
    for ax, side in (("y", H), ("x", W)):
        order = sorted(idx, key=lambda i: (_span(boxes[i], ax)[0], i))
        groups, cur, end, widest = [], [order[0]], _span(boxes[order[0]], ax)[1], 0.0
        for i in order[1:]:
            lo, hi = _span(boxes[i], ax)
            if lo >= end - 1e-6:
                widest = max(widest, lo - end)
                groups.append(cur)
                cur = [i]
            else:
                cur.append(i)
            end = max(end, hi)
        groups.append(cur)
        if len(groups) > 1:
            score = widest / side
            if best is None or score > best[0] + 1e-9:
                best = (score, ax, groups)
    if best is None:
        return Node("free", children=[Node("leaf", leaf=i) for i in sorted(idx)])
    _, ax, groups = best
    return Node("stack", axis=ax, children=[xy_tree(g, boxes, W, H) for g in groups])


# ---------------------------------------------------------------- LP

class _LP:
    def __init__(self):
        self.lb, self.ub, self.c = [], [], []
        self.A_ub, self.b_ub, self.A_eq, self.b_eq = [], [], [], []

    def var(self, lb=None, ub=None, cost=0.0) -> int:
        self.lb.append(lb)
        self.ub.append(ub)
        self.c.append(cost)
        return len(self.c) - 1

    def le(self, e: dict, rhs: float = 0.0) -> None:
        """sum(e) + e[None] <= rhs"""
        self.A_ub.append({k: v for k, v in e.items() if k is not None})
        self.b_ub.append(rhs - e.get(None, 0.0))

    def eq(self, e: dict, rhs: float = 0.0) -> None:
        self.A_eq.append({k: v for k, v in e.items() if k is not None})
        self.b_eq.append(rhs - e.get(None, 0.0))

    def absolute(self, e: dict, weight: float) -> None:
        """weight x |e|"""
        if weight <= 0:
            return
        t = self.var(0.0, None, weight)
        self.le(_add(e, {t: -1.0}))
        self.le(_add(_scale(e, -1.0), {t: -1.0}))

    def solve(self):
        import numpy as np
        from scipy.optimize import linprog
        n = len(self.c)

        def dense(rows):
            if not rows:
                return None
            A = np.zeros((len(rows), n))
            for r, row in enumerate(rows):
                for k, v in row.items():
                    A[r, k] += v
            return A
        res = linprog(np.array(self.c), A_ub=dense(self.A_ub), b_ub=np.array(self.b_ub) if self.b_ub else None,
                      A_eq=dense(self.A_eq), b_eq=np.array(self.b_eq) if self.b_eq else None,
                      bounds=list(zip(self.lb, self.ub)), method="highs")
        return res.x if res.status == 0 else None


def _add(*es: dict) -> dict:
    out: dict = {}
    for e in es:
        for k, v in e.items():
            out[k] = out.get(k, 0.0) + v
    return out


def _scale(e: dict, k: float) -> dict:
    return {key: v * k for key, v in e.items()}


class _Solve:
    """Variables: per block x, y (top-left of its content extent) and s."""

    def __init__(self, model: _Model, start: list[Box] | None = None):
        self.M = model
        self.B = model.blocks
        self.lp = _LP()
        self.S = model.short
        self.pos: list[dict] = []
        for b in self.B:
            x0, y0, w, h = b.box
            vx = self.lp.var(x0 if b.fixed[0] else None, x0 if b.fixed[0] else None)
            vy = self.lp.var(y0 if b.fixed[1] else None, y0 if b.fixed[1] else None)
            vs = self.lp.var(b.s_lo, max(1.0, b.s_max))
            self.pos.append({"x": vx, "y": vy, "s": vs})

    # expressions on the content extent / the full extent
    def start(self, i: int, ax: str) -> dict:
        return {self.pos[i][ax]: 1.0}

    def size(self, i: int, ax: str) -> float:
        b = self.B[i].box
        return b[2] if ax == "x" else b[3]

    def end(self, i: int, ax: str) -> dict:
        return {self.pos[i][ax]: 1.0, self.pos[i]["s"]: self.size(i, ax)}

    def centre(self, i: int, ax: str) -> dict:
        return {self.pos[i][ax]: 1.0, self.pos[i]["s"]: self.size(i, ax) / 2}

    def line(self, i: int, ax: str, k: int) -> dict:
        return (self.start, self.centre, self.end)[k](i, ax)

    def full(self, i: int, ax: str) -> tuple[dict, dict]:
        b = self.B[i]
        (c0, c1), (f0, f1) = _span(b.box, ax), _span(b.full, ax)
        return {self.pos[i][ax]: 1.0, self.pos[i]["s"]: f0 - c0}, {self.pos[i][ax]: 1.0, self.pos[i]["s"]: f1 - c0}

    def w(self, weight: float) -> float:
        return weight / self.S  # per px


def _cur_line(b: Box, ax: str, k: int) -> float:
    lo, hi = _span(b, ax)
    return (lo, (lo + hi) / 2, hi)[k]


def solve_model(model: _Model) -> tuple[list[tuple[float, float, float]], dict] | None:
    """(x, y, s) per block, plus stats; None when the LP has no solution."""
    B = model.blocks
    if not B:
        return None
    W, H, opts = model.W, model.H, model.opts
    P = _Solve(model)
    lp = P.lp
    boxes = [b.box for b in B]
    side = {"x": W, "y": H}
    marg = {"x": model.mx, "y": model.my}
    gmin = GAP_MIN * model.short
    tree = xy_tree(list(range(len(B))), boxes, W, H)
    stats = {"blocks": len(B), "stacks": 0, "keeps": 0, "snaps": 0, "pins": 0}

    # -- hard: in the canvas, the safe margin, bleeds --------------------------
    for i, b in enumerate(B):
        for ax in ("x", "y"):
            f0, f1 = P.full(i, ax)
            (cf0, cf1) = _span(b.full, ax)
            lp.le(_scale(f0, -1.0), -min(0.0, cf0))                  # full start >= 0
            lp.le(f1, max(side[ax], cf1))                            # full end <= side
            c0, c1 = _span(b.box, ax)
            k0, k1 = (0, 2) if ax == "x" else (1, 3)
            if b.bleed[k0] and not b.fixed[ax == "y"]:
                lp.eq(P.start(i, ax), c0)
            else:
                lp.le(_scale(P.start(i, ax), -1.0), -min(marg[ax], c0))
            if b.bleed[k1] and not b.fixed[ax == "y"]:
                lp.eq(P.end(i, ax), c1)
            else:
                lp.le(P.end(i, ax), side[ax] - min(marg[ax], side[ax] - c1))

    # -- hard: hierarchy (no text block outgrows the headline block's scale) -----
    head = next((i for i, b in enumerate(B) if b.role == "headline" and b.texts), None)
    if head is not None:
        fh = B[head].font
        for i, b in enumerate(B):
            if i != head and b.texts:
                if opts.targets and fh > 0 and 0 < b.font <= fh:   # its largest font stays at most the headline's
                    lp.le({P.pos[i]["s"]: b.font, P.pos[head]["s"]: -fh})
                else:
                    lp.le({P.pos[i]["s"]: 1.0, P.pos[head]["s"]: -1.0})

    # -- the tree: order, gaps, centring --------------------------------------
    close = _close_pairs(model)

    def extreme(leaves: list[int], ax: str, lo: bool) -> int:
        key = (lambda i: _span(boxes[i], ax)[0]) if lo else (lambda i: -_span(boxes[i], ax)[1])
        return min(leaves, key=lambda i: (key(i), i))

    def walk(node: Node) -> None:
        if node.kind == "leaf":
            return
        if node.kind == "free":
            ls = node.leaves()
            for a in range(len(ls)):
                for c in range(a + 1, len(ls)):
                    _order_pair(P, model, ls[a], ls[c], gmin)
            return
        ax = node.axis
        stats["stacks"] += 1
        g_in = lp.var(0.0, None)
        g_out = lp.var(0.0, None)
        lp.le({g_in: 1.0, g_out: -1.0})
        kids = node.children
        for k in range(len(kids) - 1):
            A, C = kids[k].leaves(), kids[k + 1].leaves()
            best = None
            for p in A:
                for q in C:
                    g = _span(boxes[q], ax)[0] - _span(boxes[p], ax)[1]
                    gm = gmin if opts.gap_min_hard else min(gmin, max(0.0, g))
                    lp.le(_add(P.end(p, ax), _scale(P.start(q, ax), -1.0)), -gm)
                    across = _cross_overlap(boxes[p], boxes[q], ax) > 0
                    key = (not across, g)
                    if best is None or key < best[0]:
                        best = (key, p, q)
            _, p, q = best
            if _pinned(B[p], ax) or _pinned(B[q], ax) or B[p].fixed[ax == "y"] or B[q].fixed[ax == "y"]:
                continue
            gap = _add(P.start(q, ax), _scale(P.end(p, ax), -1.0))
            gv = g_in if (B[p].bid, B[q].bid) in close or (B[q].bid, B[p].bid) in close else g_out
            lp.absolute(_add(gap, {gv: -1.0}), P.w(W_GAP))
        # a child centred across the stack (in its row / column) stays centred; nearly centred snaps
        cx = _other(ax)
        L = node.leaves()
        lo_all, hi_all = extreme(L, cx, True), extreme(L, cx, False)
        span_all = _span(boxes[hi_all], cx)[1] - _span(boxes[lo_all], cx)[0]
        for kid in kids:
            kl = kid.leaves()
            if len(kl) == len(L):
                continue
            lo, hi = extreme(kl, cx, True), extreme(kl, cx, False)
            if any(B[i].fixed[cx == "y"] for i in kl):
                continue
            c_kid = (_span(boxes[lo], cx)[0] + _span(boxes[hi], cx)[1]) / 2
            c_all = (_span(boxes[lo_all], cx)[0] + _span(boxes[hi_all], cx)[1]) / 2
            if lo == lo_all and hi == hi_all or span_all <= 0:
                continue
            d = abs(c_kid - c_all)
            e = _scale(_add(P.start(lo, cx), P.end(hi, cx), _scale(P.start(lo_all, cx), -1.0),
                            _scale(P.end(hi_all, cx), -1.0)), 0.5)
            if d <= max(1.0, ALIGN_KEEP * model.long):
                lp.absolute(e, P.w(W_KEEP))
                stats["keeps"] += 1
            elif d <= SNAP * model.long:
                lp.absolute(e, P.w(W_SNAP))
                stats["snaps"] += 1
        for kid in kids:
            walk(kid)

    walk(tree)

    # -- shared lines kept, near lines snapped (block pairs) -------------------
    tol = max(1.0, ALIGN_KEEP * model.long)
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            for ax in ("x", "y"):
                if B[i].fixed[ax == "y"] and B[j].fixed[ax == "y"]:
                    continue
                diffs = [abs(_cur_line(B[i].box, ax, k) - _cur_line(B[j].box, ax, k)) for k in range(3)]
                kept = [k for k in range(3) if diffs[k] <= tol]
                for k in kept:
                    lp.absolute(_add(P.line(i, ax, k), _scale(P.line(j, ax, k), -1.0)), P.w(W_KEEP))
                    stats["keeps"] += 1
                if kept:
                    continue
                # snap only blocks stacked across this axis (one above the other, for x lines)
                o = _other(ax)
                (a0, a1), (c0, c1) = _span(B[i].box, o), _span(B[j].box, o)
                stacked = a1 <= c0 + _EPS or c1 <= a0 + _EPS
                k = min(range(3), key=lambda k: diffs[k])
                if stacked and diffs[k] <= SNAP * model.long and _cross_overlap(B[i].box, B[j].box, o) > 0:
                    lp.absolute(_add(P.line(i, ax, k), _scale(P.line(j, ax, k), -1.0)), P.w(W_SNAP))
                    stats["snaps"] += 1

    # -- pinned edges ------------------------------------------------------------
    for i, b in enumerate(B):
        for k, ax, at_start in ((0, "x", True), (1, "y", True), (2, "x", False), (3, "y", False)):
            if b.pin[k] and not b.fixed[ax == "y"]:
                cur = _span(b.box, ax)[0 if at_start else 1]
                lp.absolute(_add(P.start(i, ax) if at_start else P.end(i, ax), {None: -cur}), P.w(W_PIN))
                stats["pins"] += 1

    # -- equal margins of the free content -----------------------------------------
    for ax in ("x", "y"):
        _margins(P, model, ax, boxes)

    # -- plan anchors (realize) -----------------------------------------------------
    for i, b in enumerate(B):
        for ax, where in (opts.anchors.get(b.bid) or {}).items():
            if b.fixed[ax == "y"]:
                continue
            if where == "start":
                lp.absolute(_add(P.start(i, ax), {None: -marg[ax]}), P.w(W_ANCHOR))
            elif where == "end":
                lp.absolute(_add(P.end(i, ax), {None: -(side[ax] - marg[ax])}), P.w(W_ANCHOR))
            elif where == "centre":
                lp.absolute(_add(P.centre(i, ax), {None: -side[ax] / 2}), P.w(W_ANCHOR))

    # -- fidelity: block centres stay near the candidate's ---------------------------
    for i, b in enumerate(B):
        for ax in ("x", "y"):
            if not b.fixed[ax == "y"]:
                c = sum(_span(b.box, ax)) / 2
                lp.absolute(_add(P.centre(i, ax), {None: -c}), P.w(opts.fidelity))

    # -- Sonnet's proportions: each block's scale pulled toward its target size -------
    for i, b in enumerate(B):
        if b.t_ratio is not None:
            L = max(b.box[2], b.box[3])
            lp.absolute({P.pos[i]["s"]: L, None: -b.t_ratio * L}, P.w(W_TARGET if opts.w_target is None else opts.w_target))
            stats["targets"] = stats.get("targets", 0) + 1

    # -- fill: grow toward the safe area when much of it is unused --------------------
    if _fill_allowed(model, boxes):
        for i, b in enumerate(B):
            if b.s_max > 1.0 and b.t_ratio is None:
                wt = (b.rank ** -0.7) if math.isfinite(b.rank) and b.rank > 0 else 0.2
                lp.c[P.pos[i]["s"]] -= W_FILL * wt
    else:
        for i, b in enumerate(B):
            if b.t_ratio is None:
                lp.ub[P.pos[i]["s"]] = 1.0

    x = lp.solve()
    if x is None:
        return None
    out = [(float(x[p["x"]]), float(x[p["y"]]), float(x[p["s"]])) for p in P.pos]
    stats["vars"] = len(lp.c)
    stats["rows"] = len(lp.b_ub) + len(lp.b_eq)
    return out, stats


def _pinned(b: Block, ax: str) -> bool:
    return (b.pin[0] or b.pin[2]) if ax == "x" else (b.pin[1] or b.pin[3])


def _order_pair(P: _Solve, model: _Model, i: int, j: int, gmin: float) -> None:
    """Keep two blocks apart along the axis on which they are farther apart."""
    a, b = model.blocks[i].box, model.blocks[j].box
    best = None
    for ax in ("x", "y"):
        (a0, a1), (b0, b1) = _span(a, ax), _span(b, ax)
        for p, q, g in ((i, j, b0 - a1), (j, i, a0 - b1)):
            if g >= -1e-6 and (best is None or g > best[3]):
                best = (ax, p, q, g)
    if best is None:
        return
    ax, p, q, g = best
    gm = gmin if model.opts.gap_min_hard else min(gmin, max(0.0, g))
    P.lp.le(_add(P.end(p, ax), _scale(P.start(q, ax), -1.0)), -gm)


def _close_pairs(model: _Model) -> set[tuple[str, str]]:
    """Blocks that sat close in the source (balance.close_pairs on their source boxes)."""
    from engine.pipeline.layout.balance import close_pairs
    sb = {b.bid: b.src_box for b in model.blocks if b.src_box is not None}
    try:
        return {(a, b) for a, b, _, _ in close_pairs(sb, model.Ws, model.Hs)}
    except Exception:
        return set()


def _margins(P: _Solve, model: _Model, ax: str, boxes: list[Box]) -> None:
    """Equal start and end margins of the free content on `ax` (blocks not
    spanning it and not pinned on it), measured from the canvas edge or from a
    pinned block before / after it in its lane; the source's own lopsidedness
    (same direction) is kept."""
    B = model.blocks
    side = model.W if ax == "x" else model.H
    sside = model.Ws if ax == "x" else model.Hs
    F = [i for i, b in enumerate(B) if not b.fixed[ax == "y"] and not _pinned(b, ax)]
    if not F:
        return
    lo = min(F, key=lambda i: (_span(boxes[i], ax)[0], i))
    hi = min(F, key=lambda i: (-_span(boxes[i], ax)[1], i))
    U = union_box([boxes[i] for i in F])
    u0, u1 = _span(U, ax)
    ref_lo, ref_hi = None, None
    for i, b in enumerate(B):
        if i in F or b.fixed[ax == "y"]:
            continue
        if _cross_overlap(boxes[i], U, ax) <= 0:
            continue
        s0, s1 = _span(boxes[i], ax)
        if s1 <= u0 + _EPS and (ref_lo is None or s1 > _span(boxes[ref_lo], ax)[1]):
            ref_lo = i
        elif s0 >= u1 - _EPS and (ref_hi is None or s0 < _span(boxes[ref_hi], ax)[0]):
            ref_hi = i
    m_lo = _add(P.start(lo, ax), _scale(P.end(ref_lo, ax), -1.0)) if ref_lo is not None else P.start(lo, ax)
    m_hi = _add(P.start(ref_hi, ax), _scale(P.end(hi, ax), -1.0)) if ref_hi is not None else \
        _add({None: side}, _scale(P.end(hi, ax), -1.0))
    cur_lo = u0 - (_span(boxes[ref_lo], ax)[1] if ref_lo is not None else 0.0)
    cur_hi = (_span(boxes[ref_hi], ax)[0] if ref_hi is not None else side) - u1
    target = 0.0
    if ref_lo is None and ref_hi is None:
        srcs = [B[i].src_box for i in F if B[i].src_box is not None]
        if srcs:
            S = union_box(srcs)
            s0, s1 = _span(S, ax)
            simb = ((sside - s1) - s0) / sside    # + : more room at the end
            cimb = (cur_hi - cur_lo) / side
            if simb and cimb and (simb > 0) == (cimb > 0) and abs(simb) >= IMB_SOURCE * abs(cimb):
                target = -min(abs(simb) * side, abs(cur_hi - cur_lo)) * (1 if simb > 0 else -1)
    # m_lo - m_hi -> target
    P.lp.absolute(_add(m_lo, _scale(m_hi, -1.0), {None: -target}), P.w(W_MARGIN))


def _fill_allowed(model: _Model, boxes: list[Box]) -> bool:
    """The content leaves more than fill_unused of the safe area unused on an axis."""
    U = union_box(boxes)
    for ax, side, m in (("x", model.W, model.mx), ("y", model.H, model.my)):
        safe = side - 2 * m
        lo, hi = _span(U, ax)
        if safe > 0 and (hi - lo) < (1 - model.opts.fill_unused) * safe:
            return True
    return False


# ---------------------------------------------------------------- applying a solution

def _apply(model: _Model, sol: list[tuple[float, float, float]]) -> dict[str, dict] | None:
    """placed geometry with every block moved to (x, y) and scaled by s about its content's top-left."""
    from engine.pipeline.layout.place import resize_text
    m = model.m
    placed = {k: dict(v) for k, v in model.placed.items()}
    for b, (x, y, s) in zip(model.blocks, sol):
        bx, by = b.box[0], b.box[1]
        s = 1.0 if abs(s - 1.0) < 1e-4 else s
        for i in b.content + b.decos:
            e = model.placed[i]
            new = dict(e, x=x + (e["x"] - bx) * s, y=y + (e["y"] - by) * s, width=e["width"] * s, height=e["height"] * s)
            for key in ("corner_radius", "stroke_width"):
                if e.get(key):
                    new[key] = e[key] * s
            if e["type"] == "text" and s != 1.0:
                f0 = e["font_size"]
                t = resize_text(e, f0 * s)
                new.update(font_size=t["font_size"])
                if "letter_spacing" in t:
                    new["letter_spacing"] = t["letter_spacing"]
                if not m.fits(new, new["font_size"], new["width"], new["height"]):
                    lo, hi = (f0, f0 * s) if s > 1.0 else (0.85 * f0 * s, f0 * s)
                    size = m.fit_font_size(dict(e), new["width"], new["height"], lo, hi)
                    size = (f0 if s > 1.0 else None) if size is None else size
                    if size is None:
                        return None
                    t = resize_text(e, size)
                    new.update(font_size=size)
                    if "letter_spacing" in t:
                        new["letter_spacing"] = t["letter_spacing"]
                    if not m.fits(new, size, new["width"], new["height"]):
                        return None
            placed[i] = new
    if model.opts.targets and T_REWRAP:
        _rewrap(model, sol, placed)
    return placed


def _rewrap(model: _Model, sol, placed: dict) -> None:
    """(targets) A lone text block that shrank keeps its line breaks under a uniform scale ("Chase /
    the Light" set smaller); re-wrap it on fewer lines in the narrowest of its inked width, its slot
    width on the candidate and the safe width of the canvas that saves a line, centred on its solved
    box (its alignment edge kept), if it stays inside the safe margins and off every other block."""
    from engine.pipeline.layout.text_slack import slack_px
    m = model.m
    solved = []
    for b, (x, y, s) in zip(model.blocks, sol):
        fx, fy, fw, fh = b.full
        solved.append((x + (fx - b.box[0]) * s, y + (fy - b.box[1]) * s, fw * s, fh * s))
    for k, (b, (x, y, s)) in enumerate(zip(model.blocks, sol)):
        if s > 1.0 - 1e-3 or len(b.content) != 1 or b.decos or b.content[0] not in b.texts:
            continue
        i = b.content[0]
        e0, new = model.placed[i], placed[i]
        if new.get("rotation"):
            continue
        f = new["font_size"]
        n_now = m.line_count(new, f, new["width"])
        for w0 in sorted({e0["width"], model.untight.get(i, e0)["width"], model.W - 2 * model.mx}):
            n_wide = m.line_count(new, f, w0)
            if n_wide >= n_now or w0 <= new["width"]:
                continue
            ink = max(m.line_widths(new, f, w0), default=0.0)
            nw = min(w0, ink + slack_px(ink))
            if m.line_count(new, f, nw) != n_wide:
                nw = w0
            nh = m.block_height(new, f, nw) + _EPS
            align = new.get("text_align") or "left"
            nx = new["x"] if align == "left" else new["x"] + new["width"] - nw if align == "right" else \
                new["x"] + (new["width"] - nw) / 2
            nx = min(max(nx, model.mx), model.W - model.mx - nw)
            ny = new["y"] + (new["height"] - nh) / 2
            box = (nx, ny, nw, nh)
            if nx < model.mx - _EPS or ny < 0 or nx + nw > model.W - model.mx + _EPS or ny + nh > model.H:
                continue
            if any(overlap_area(box, o) > 0.5 for j, o in enumerate(solved) if j != k):
                continue
            out = dict(new, x=nx, y=ny, width=nw, height=nh)
            if m.fits(out, f, nw, nh) and m.line_count(out, f, nw) == n_wide:
                placed[i] = out
                break


def _moved(model: _Model, sol) -> bool:
    for b, (x, y, s) in zip(model.blocks, sol):
        if abs(x - b.box[0]) > MIN_CHANGE or abs(y - b.box[1]) > MIN_CHANGE or abs(s - 1.0) * max(b.box[2], b.box[3]) > MIN_CHANGE:
            return True
    return False


def solve_design(source: dict, target: dict, plan: PriorityPlan, measurer, design: dict,
                 layout: PriorityPlan | None = None, opts: Opts | None = None, stats: dict | None = None) -> dict | None:
    """The design re-solved (finalized), or None when nothing changes or it can't be solved."""
    from engine.pipeline.layout.normalize import finalize
    opts = opts or Opts()
    model = _Model(source, target, plan, layout or _layout_plan(source, plan, target), measurer, design, opts)
    res = solve_model(model)
    if res is None:
        if stats is not None:
            stats["status"] = "infeasible"
        return None
    sol, st = res
    if stats is not None:
        stats.update(st)
    if not _moved(model, sol):
        if stats is not None:
            stats["status"] = "unchanged"
        return None
    placed = _apply(model, sol)
    if placed is None:
        if stats is not None:
            stats["status"] = "text"
        return None
    out = finalize(source, placed, target)
    if stats is not None:
        stats["status"] = "solved" if out is not None else "finalize"
    return out


def target_opts(source: dict, target: dict, plan: PriorityPlan) -> Opts | None:
    """(CHAMELEON_TARGETS) solver options with Sonnet's predicted proportions, or None (nothing predicted)."""
    from engine.pipeline.layout import targets
    pred = targets.predict(source, plan, target)
    return Opts(targets=pred) if pred else None


def solve_candidate(source: dict, target: dict, plan: PriorityPlan, measurer, cand: Candidate,
                    layout: PriorityPlan | None = None, stats: dict | None = None,
                    opts: Opts | None = None) -> Candidate:
    """`cand` re-solved (same id, template, params and degraded units), or
    `cand` itself if it fails a gate, nothing changes, or the result fails a
    gate the original passed."""
    from engine.pipeline.grading.gates import check_gates
    if cand.is_fallback:
        return cand
    if check_gates(source, cand.design, target, plan, measurer):
        if stats is not None:
            stats["status"] = "base_failed"
        return cand
    design = solve_design(source, target, plan, measurer, cand.design, layout, opts=opts, stats=stats)
    if design is None:
        return cand
    if check_gates(source, design, target, plan, measurer):
        if stats is not None:
            stats["status"] = "gates"
        return cand
    return Candidate(cand.candidate_id, design, cand.template, cand.params, list(cand.degraded_unit_ids),
                     cand.is_fallback)


def solve_all(source: dict, target: dict, plan: PriorityPlan, measurer, candidates: list[Candidate],
              log: list | None = None, with_targets: bool = False) -> list[Candidate]:
    """Every gate-passing non-fallback candidate re-solved (never raises);
    `with_targets`: with Sonnet's proportions as pulls (CHAMELEON_TARGETS=1)."""
    try:
        layout = _layout_plan(source, plan, target)
        opts = target_opts(source, target, plan) if with_targets else None
    except Exception:
        return candidates
    out = []
    for cand in candidates:
        stats: dict = {}
        try:
            new = solve_candidate(source, target, plan, measurer, cand, layout, stats, opts)
        except Exception as exc:  # the solver never sinks a candidate
            stats["status"] = f"error:{type(exc).__name__}"
            new = cand
        if log is not None:
            log.append({"candidate_id": cand.candidate_id, **stats})
        out.append(new)
    return out


def realize(source: dict, target: dict, plan: PriorityPlan, measurer, design: dict, anchors: dict,
            s_max: float = 2.0, layout: PriorityPlan | None = None, stats: dict | None = None,
            targets: dict | None = None) -> dict | None:
    """(oracle plan) Solve a naive placement with plan anchors pulling blocks
    to canvas edges / the centre line and a larger growth cap; None if it
    can't be solved. Blocks are named by their unit ids joined with '+'."""
    opts = Opts(s_max=s_max, fill_unused=0.0, anchors=anchors, gap_min_hard=False, fidelity=W_FID / 5,
                targets=targets or None)
    return solve_design(source, target, plan, measurer, design, layout, opts, stats)
