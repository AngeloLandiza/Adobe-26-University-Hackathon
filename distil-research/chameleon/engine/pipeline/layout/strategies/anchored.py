"""Strategy `anchored`: constraint-style retargeting (on with
CHAMELEON_STRATEGIES=...,anchored; with it off nothing here runs).

The idea of Figma constraints, Adobe XD responsive resize (patent US
11,320,975) and pro smart-resize tools (Bannersnack: "anchored elements stay
in place and scale proportionally; grouped items move and scale together"):
keep the original's composition, re-fit the background full-bleed, and
re-place every piece of content by its *anchor* instead of re-laying it out.
The plain uniform fit (the safe fallback) keeps the composition too, but
letterboxes it on any other shape; this uses the new canvas.

1. **Pieces.** The layout plan's content units (with CHAMELEON_LAYERS=1, a
   container and everything on it is one unit, layout/layer_groups.py).
   Each has a tight source box (a text's inked lines, not its slot). A
   text with a small decoration on it (a highlight behind a word) scales
   rigidly instead of re-wrapping, so the decoration stays on its words.
2. **Anchors**, per axis, from the box's edge distances in the source:
   `stretch` when it spans STRETCH of the axis; `centre` when its two edge
   gaps are within CENTRE_TOL of the side of each other; otherwise pinned to
   the nearer edge (`start` / `end`) if that gap is at most EDGE_SHARE of
   the side, else `centre` with its offset from the middle kept. A piece
   within BLEED_TOL of an edge bled off it and stays on it.
3. **Blocks.** Pieces stacked across the nudge axis (see 5), close
   (ADJ_GAP) and sharing an edge or centre line (ALIGN_TOL), none of them a
   photo and none with a same-role neighbour beside it along the nudge
   axis, form a block: a text column (headline, subhead, CTA) or a row of
   badges. A block keeps its stacking and alignment and moves as one; it is
   anchored by its union box. A block too long to fit is split in two at
   its widest gap (and again, down to single pieces).
4. **Scale.** One content scale s for everything, tried from the uniform-fit
   scale s0 = min(W/Ws, H/Hs) up to the fill scale max(W/Ws, H/Hs) (at most
   RATIOS[-1] x s0). Each piece is scaled by s, raised to its texts' role
   floors, logos held between bisection v2's floor and cap, and shrunk only
   if it can't fit the canvas: a photo may give up to 30% (PHOTO_GIVE) so
   the rest fits (the flexible element absorbs the squeeze), and a button
   too wide at its floor is condensed round its label. A lone text re-wraps
   inside its scaled width (or the canvas width) instead of shrinking.
   Edge gaps scale with s (at least SAFE_MARGIN of the short side, unless
   the piece bled off that edge). Anchors are taken inside a frame at most
   SPREAD_MAX x the scaled source side long (placed by the content's own
   anchor), so a tight cluster on a much longer side stays together.
5. **Collisions** are resolved by nudging, mainly along the axis with
   slack (the larger of W/Ws, H/Hs; the long side on a tie). Two pieces
   that would overlap (they didn't in the source) are kept apart in source
   order: a pair stacked across that axis is first pushed apart across it
   (above stays above) if that still fits, else along it; above then
   becomes left-of (on a wide canvas) or left-of becomes above (on a tall
   one), and left-of stays left-of. Positions are the
   anchored ones moved as little as possible (iterative projection); if no
   order-keeping placement fits the canvas, that s is dropped. On a strip
   (short side at most STRIP of the long one) every piece sits on the
   middle line across it, as banner designers set a row.
   Then a piece that shares no edge or centre line with another snaps onto
   the nearest one within SNAP (bisection v2's shared alignment).
6. **Hierarchy.** Text scales with everything, so the source's text ratios
   hold; a scale where a floor makes another text outgrow the headline (it
   didn't in the source) is dropped.
7. Decorations follow the unit they framed (bisection v2's frame rule),
   else keep their own anchor at scale s, else are parked as v2 parks them.

Candidates (ids `anchored/mode=.../r=...`, r = s / s0): the largest
feasible s, the uniform-fit s (anchored edges), one in between, and the
largest s with anchors on the whole canvas and centred pieces spread
proportionally along the nudge axis (Figma's "scale" constraint); at most
MAX_CANDIDATES distinct designs, each passing the hard gates. Never raises.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from engine.pipeline.geometry import Box, overlap_area, rotated_aabb, union_box
from engine.pipeline.layout.place import floor_px, resize_text
from engine.pipeline.types import Candidate, PriorityPlan, Unit

NAME = "anchored"
STRETCH = 0.8          # a piece spanning this share of an axis stretches along it
CENTRE_TOL = 0.1       # edge gaps within this share of the side of each other: centred
EDGE_SHARE = 0.2       # pinned to the nearer edge if its gap is at most this share of the side
BLEED_TOL = 0.01       # a piece this close to an edge (share of the side) bled off it
SAFE_MARGIN = 0.06     # least gap to a canvas edge, share of the short side (bleeding pieces excepted; as bisection v2)
GUTTER = 0.025         # least gap between two pieces nudged apart, share of the short side
STRIP = 0.2            # a canvas whose short side is at most this share of its long side is a strip
OVERLAP_SHARE = 0.1    # pieces that overlapped in the source may overlap up to this much more (share of the smaller)
TOUCH = 0.005          # pieces closer than this across the nudge axis (share of the short side) collide
ADJ_GAP = 0.1          # (blocks) stacked pieces at most this share of the source side apart ...
ALIGN_TOL = 0.03       # ... sharing an edge or centre within this share of the source side
SPREAD_MAX = 1.6       # anchors are taken in a frame at most this x the scaled source side (spread: the canvas)
SNAP = 0.02            # an unaligned piece moves onto another's edge or centre line this close (share of the long side)
ALIGNED = 0.005        # ... unless it already shares one this close (the grader's alignment tolerance)
PHOTO_GIVE = (1.0, 0.85, 0.7)  # a photo may give up this much of its scale so the rest fits (it absorbs the squeeze)
RATIOS = (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0)  # s / s0 tried (and the fill scale, if lower)
MAX_CANDIDATES = 4
SAME_ASPECT_TOL = 0.02  # as generate.py: the proportional reflow is the answer there
NUDGE_ROUNDS = 10
SOLVE_ITERS = 400
_PHOTO_ROLES = ("hero_image", "secondary_image")
_LAYER_ROLES = ("background", "decoration")


def enabled() -> bool:
    raw = os.environ.get("CHAMELEON_STRATEGIES", "")
    return NAME in [n.strip() for n in raw.split(",")]


# ---------------------------------------------------------------- anchors

def anchor(lo: float, size: float, side: float) -> str:
    """start | centre | end | stretch for a piece at [lo, lo + size] on a side of length `side`."""
    if side <= 0:
        return "centre"
    if size >= STRETCH * side:
        return "stretch"
    g0, g1 = lo, side - lo - size
    if abs(g0 - g1) <= CENTRE_TOL * side:
        return "centre"
    if g0 <= g1:
        return "start" if g0 <= EDGE_SHARE * side else "centre"
    return "end" if g1 <= EDGE_SHARE * side else "centre"


def place_on_axis(kind: str, lo: float, size: float, side: float, new_size: float, new_side: float,
                  s: float, margin: float, spread: bool = False, middle: bool = False,
                  frame: tuple[float, float] | None = None) -> tuple[float, float, float]:
    """(position, least position, greatest position) on the target axis for a
    piece of `new_size`. Positions are taken inside `frame` (offset, length;
    default the whole axis): edge gaps scale with s; a centred piece keeps its
    offset from the middle, scaled by s (or by the frame's ratio to the source
    side when `spread`; dropped when `middle`, which centres on the canvas).
    A piece that bled off an edge stays on the canvas's edge; the bounds keep
    `margin` from the canvas's other edges."""
    f0, flen = frame if frame is not None else (0.0, new_side)
    g0, g1 = lo, side - lo - size
    m0 = 0.0 if g0 <= BLEED_TOL * side else margin
    m1 = 0.0 if g1 <= BLEED_TOL * side else margin
    hi = new_side - m1 - new_size
    if kind == "start" or kind == "stretch":
        p = g0 * s if m0 == 0 else f0 + g0 * s
    elif kind == "end":
        p = new_side - new_size - g1 * s if m1 == 0 else f0 + flen - new_size - g1 * s
    elif middle and kind == "centre":
        p = (new_side - new_size) / 2
    else:
        off = (lo + size / 2 - side / 2) * ((flen / side) if spread else s)
        p = f0 + flen / 2 + off - new_size / 2
    return p, m0, hi


# ---------------------------------------------------------------- pieces

@dataclass
class _Piece:
    unit: Unit
    members: list[dict]
    box: Box                       # tight source box
    text: bool                     # a lone, unrotated text: re-wraps
    photo: bool                    # a hero / secondary image: never joins a block
    group: object = None           # layer_groups.Group for a container unit


@dataclass
class _Sized:
    piece: _Piece
    geo: dict                      # element id -> geometry, the piece's box at (0, 0)
    w: float
    h: float
    core: Box | None = None        # what must not collide, in the piece's box (default: all of it)


@dataclass
class _Entity:
    """A block (one or more pieces) being placed."""

    pieces: list[_Piece]
    box: Box                       # source union box
    align: str = "centre"          # (blocks) shared line along the nudge axis
    sized: list[_Sized] = field(default_factory=list)
    offs: list[tuple[float, float]] = field(default_factory=list)  # each piece's (x, y) in the entity
    w: float = 0.0
    h: float = 0.0
    x: float = 0.0
    y: float = 0.0
    core: Box | None = None        # (single pieces) the part that must not collide, in the entity's box
    src_core: Box | None = None    # ... and in the source

    def rect(self) -> Box:
        cx, cy, cw, ch = self.core if self.core is not None else (0.0, 0.0, self.w, self.h)
        return (self.x + cx, self.y + cy, cw, ch)


class _Ctx:
    def __init__(self, source: dict, target: dict, plan: PriorityPlan, layout: PriorityPlan, measurer, layers):
        self.source, self.plan, self.layout, self.m, self.layers = source, plan, layout, measurer, layers
        self.Ws, self.Hs = float(source["canvas"]["width"]), float(source["canvas"]["height"])
        self.W, self.H = float(target["width"]), float(target["height"])
        self.target = target
        self.k: dict[str, float] = {}   # (targets) unit id -> the piece's own scale (Sonnet's predicted size)
        self.short = min(self.W, self.H)
        self.margin = SAFE_MARGIN * self.short
        self.els = {e["id"]: e for e in source["elements"]}
        self.role = {}
        for u in plan.units:
            for i in u.element_ids:
                self.role[i] = u.role
        rx, ry = self.W / self.Ws, self.H / self.Hs
        self.s0, self.s_fill = min(rx, ry), max(rx, ry)
        # nudge along the axis with slack; the long side on a tie
        if abs(rx - ry) > 1e-9:
            self.axis = "x" if rx > ry else "y"
        else:
            self.axis = "x" if self.W > self.H else "y"

    def floor(self, eid: str) -> float:
        return floor_px(self.role.get(eid), self.W, self.H)


def _area(b: Box) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def _ink_box(m, el: dict) -> Box:
    """Where a text's lines are drawn (the widest line placed per text_align, the measured block)."""
    if el.get("rotation"):
        return rotated_aabb(el)
    w = min(max(m.line_widths(el, el["font_size"], el["width"]), default=0.0), el["width"])
    align = el.get("text_align") or "left"
    x = el["x"] + (el["width"] - w) / 2 if align == "center" else \
        el["x"] + el["width"] - w if align == "right" else el["x"]
    return (x, el["y"], max(w, 1.0), max(1.0, min(el["height"], m.block_height(el, el["font_size"], el["width"]))))


def _framed_texts(ctx: _Ctx) -> set[str]:
    """Texts a decoration sits behind or on (a highlight behind a word, a frame
    round a line): re-wrapping them would leave the decoration on the wrong words."""
    decos = [rotated_aabb(ctx.els[i]) for u in ctx.layout.units if u.role == "decoration" for i in u.element_ids]
    out = set()
    for e in ctx.source["elements"]:
        if e["type"] != "text":
            continue
        b = rotated_aabb(e)
        for d in decos:  # mostly on the text, and not much bigger than it
            if _area(d) > 0 and overlap_area(b, d) >= 0.5 * _area(d) and _area(d) <= 1.5 * _area(b):
                out.add(e["id"])
                break
    return out


def _pieces(ctx: _Ctx) -> list[_Piece]:
    framed = _framed_texts(ctx)
    out = []
    for u in ctx.layout.units:
        if u.role in _LAYER_ROLES or not u.element_ids:
            continue
        members = [ctx.els[i] for i in u.element_ids]
        text = len(members) == 1 and members[0]["type"] == "text" and not members[0].get("rotation") \
            and members[0]["id"] not in framed
        box = _ink_box(ctx.m, members[0]) if text else union_box([rotated_aabb(e) for e in members])
        group = ctx.layers.by_uid.get(u.unit_id) if ctx.layers is not None else None
        out.append(_Piece(u, members, box, text, u.role in _PHOTO_ROLES, group))
    return out


def _avail(ctx: _Ctx) -> tuple[float, float]:
    return ctx.W - 2 * ctx.margin, ctx.H - 2 * ctx.margin


def _size_text(ctx: _Ctx, p: _Piece, s: float, width_cap: float | None) -> _Sized | None:
    """A lone text at s x its source size (at least its floor), wrapped in its
    scaled source width (at most the canvas width, or `width_cap` when it
    stretches), shrunk only where a word or the block can't fit."""
    m = ctx.m
    el = p.members[0]
    f0 = el["font_size"]
    if f0 <= 0:
        return None
    floor = ctx.floor(el["id"])
    aw, ah = _avail(ctx)
    f = _font(max(f0 * s, floor))
    slot = el["width"] * f / f0
    slot = min(slot, aw) if width_cap is None else min(max(slot, width_cap), ctx.W)
    text = resize_text(el, f)
    if m.longest_word_width(text, f) > slot:
        g = m.fit_font_size(el, slot, math.inf, floor, f)
        g = None if g is None else _font(g)
        if g is None:
            return None
        f, text = g, resize_text(el, g)
    bh = m.block_height(text, f, slot)
    if bh > ah:  # too tall: wider first, then smaller
        slot = max(slot, aw)
        bh = m.block_height(text, f, slot)
        if bh > ah:
            g = m.fit_font_size(el, slot, ah, floor, f)
            g = None if g is None else _font(g)
            if g is None:
                return None
            f, text = g, resize_text(el, g)
            bh = m.block_height(text, f, slot)
    lines = m.line_widths(text, f, slot)
    tw = min(slot, max(lines, default=slot) + 1.0)
    if abs(m.block_height(text, f, tw) - bh) > 1e-6 or not m.fits(text, f, tw, bh + 0.01):
        tw = slot
    if not m.fits(text, f, tw, bh + 0.01):
        return None
    return _Sized(p, {el["id"]: dict(text, x=0.0, y=0.0, width=tw, height=bh + 0.01)}, tw, bh + 0.01)


def _font(f: float) -> float:
    """A font size as finalize writes it (2 decimals, rounded down), so what is measured is what ships."""
    return max(1.0, math.floor(f * 100 + 1e-6) / 100)


def _logo_cap(ctx: _Ctx, w: float, h: float) -> float:
    from engine.pipeline.layout.strategies.bisection import LOGO_MAX_SHORT, LOGO_MAX_WIDTH
    return min(LOGO_MAX_SHORT * ctx.short / h if h else math.inf, LOGO_MAX_WIDTH * ctx.W / w if w else math.inf)


def _off_canvas(ctx: _Ctx, box: Box) -> bool:
    """Runs off the source canvas (more than BLEED_TOL of a side)."""
    x, y, w, h = box
    tx, ty = BLEED_TOL * ctx.Ws, BLEED_TOL * ctx.Hs
    return x < -tx or y < -ty or x + w > ctx.Ws + tx or y + h > ctx.Hs + ty


def _logo_min(ctx: _Ctx, w: float, h: float) -> float:
    """Scale at which a logo stops being a speck (bisection v2's floor: sqrt(area)
    at least LOGO_MIN_LONG of the long side, its height at most LOGO_MIN_CAP of the short)."""
    from engine.pipeline.layout.strategies.bisection import LOGO_MIN_CAP, LOGO_MIN_LONG
    if w <= 0 or h <= 0:
        return 0.0
    return min(LOGO_MIN_LONG * max(ctx.W, ctx.H) / math.sqrt(w * h), LOGO_MIN_CAP * ctx.short / h)


def _condensed(ctx: _Ctx, p: _Piece, k: float) -> _Sized | None:
    """A button too wide for the canvas at its floor: the pill rebuilt round its
    label (the engine's condensed pill), label at most k x its source size."""
    from engine.pipeline.layout.place import _place_condensed
    label = next((e for e in p.members if e["type"] == "text"), None)
    if label is None or label["font_size"] <= 0:
        return None
    aw, ah = _avail(ctx)
    floor = ctx.floor(label["id"])
    geo = _place_condensed(p.members, (0.0, 0.0, aw, min(ah, p.box[3] * k)), ctx.m, floor, "center",
                           max(floor, label["font_size"] * k))
    if geo is None:
        return None
    ux, uy, uw, uh = union_box([rotated_aabb(g) for g in geo.values()])
    geo = {i: dict(g, x=g["x"] - ux, y=g["y"] - uy) for i, g in geo.items()}
    return _Sized(p, geo, uw, uh)


def _size_rigid(ctx: _Ctx, p: _Piece, s: float, photo_k: float = 1.0) -> _Sized | None:
    """The piece scaled as a whole by k (s, raised so its texts reach their
    floors, lowered only to fit the canvas or a logo's cap; a photo by
    `photo_k` x s). A button that can't fit at its floor is condensed."""
    from engine.pipeline.layout.layers import scale_about
    m = ctx.m
    ux, uy, uw, uh = p.box
    if uw <= 0 or uh <= 0:
        return None
    need = max((ctx.floor(e["id"]) / e["font_size"] for e in p.members if e["type"] == "text" and e["font_size"] > 0),
               default=0.0)
    k = max(s * (photo_k if p.photo else 1.0), need)
    aw, ah = _avail(ctx)
    cap = min(aw / uw, ah / uh)
    if p.unit.role == "logo":
        cap = min(cap, max(_logo_cap(ctx, uw, uh), s))
        k = max(k, min(_logo_min(ctx, uw, uh), cap))
    if k > cap:
        k = cap
        if k < need - 1e-9:
            return _condensed(ctx, p, max(s, need)) if p.unit.kind == "container_label" else None
    geo = {}
    for e in p.members:
        new = scale_about(e, k, ux, uy)
        new["x"] -= ux
        new["y"] -= uy
        if e["type"] == "text":
            new = resize_text(new, _font(new["font_size"])) if new["font_size"] > 1 else new
            f = new["font_size"]
            if not m.fits(new, f, new["width"], new["height"]):
                g = m.fit_font_size(new, new["width"], new["height"], ctx.floor(e["id"]), f)
                g = None if g is None else _font(g)
                if g is None:
                    return None
                new = resize_text(new, g)
        geo[e["id"]] = new
    if p.unit.kind == "container_label":  # a label that stuck out of its pill: the pill grows to hold it (as the fallback does)
        from engine.pipeline.layout.fallback import _wrap_label
        geo = {g["id"]: g for g in _wrap_label(list(geo.values()))}
    gx, gy, gw, gh = union_box([rotated_aabb(g) for g in geo.values()])
    if abs(gx) > 1e-6 or abs(gy) > 1e-6:
        geo = {i: dict(g, x=g["x"] - gx, y=g["y"] - gy) for i, g in geo.items()}
    w, h = max(uw * k, gw), max(uh * k, gh)
    core = None
    if p.group is not None and _off_canvas(ctx, rotated_aabb(ctx.els[p.group.container])):
        inner = [rotated_aabb(g) for i, g in geo.items() if i != p.group.container]
        core = union_box(inner) if inner else None
    return _Sized(p, geo, w, h, core)


def _size(ctx: _Ctx, p: _Piece, s: float, width_cap: float | None = None, photo_k: float = 1.0) -> _Sized | None:
    return _size_text(ctx, p, s, width_cap) if p.text else _size_rigid(ctx, p, s, photo_k)


# ---------------------------------------------------------------- blocks

def _blocks(ctx: _Ctx, pieces: list[_Piece]) -> list[list[_Piece]]:
    """Pieces stacked across the nudge axis, adjacent and aligned, none a photo:
    union-find over such pairs; a component whose pieces sit side by side
    (overlap across the stacking axis) stays apart."""
    L = ctx.axis
    side_c = ctx.Hs if L == "x" else ctx.Ws
    side_l = ctx.Ws if L == "x" else ctx.Hs

    def span(b: Box, axis: str) -> tuple[float, float]:
        return (b[0], b[0] + b[2]) if axis == "x" else (b[1], b[1] + b[3])

    C = "y" if L == "x" else "x"
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def in_row(i: int) -> bool:
        """Has a neighbour of its role beside it along the nudge axis (a row of badges)."""
        a = pieces[i]
        (c0, c1), (l0, l1) = span(a.box, C), span(a.box, L)
        for j, b in enumerate(pieces):
            if j == i or b.unit.role != a.unit.role:
                continue
            (d0, d1), (m0, m1) = span(b.box, C), span(b.box, L)
            if min(c1, d1) - max(c0, d0) >= 0.5 * min(c1 - c0, d1 - d0) and \
                    max(m0 - l1, l0 - m1) <= ADJ_GAP * side_l and min(l1, m1) <= max(l0, m0):
                return True
        return False

    rowed = [in_row(i) for i in range(len(pieces))]
    for i, a in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            b = pieces[j]
            if a.photo or b.photo or rowed[i] or rowed[j]:
                continue
            (a0, a1), (b0, b1) = span(a.box, C), span(b.box, C)
            gap = max(b0 - a1, a0 - b1)
            if gap < -0.25 * min(a1 - a0, b1 - b0) or gap > ADJ_GAP * side_c:
                continue
            (l0, l1), (m0, m1) = span(a.box, L), span(b.box, L)
            if min(l1, m1) - max(l0, m0) < 0.5 * min(l1 - l0, m1 - m0):
                continue
            tol = ALIGN_TOL * side_l
            if _shared_align(l0, l1, m0, m1, tol) is None:
                continue
            parent[find(i)] = find(j)
    comps: dict[int, list[_Piece]] = {}
    for i, p in enumerate(pieces):
        comps.setdefault(find(i), []).append(p)
    out = []
    for comp in comps.values():
        comp.sort(key=lambda p: span(p.box, C)[0])
        stacked = all(span(comp[k].box, C)[1] - span(comp[k + 1].box, C)[0]
                      <= 0.25 * min(span(comp[k].box, C)[1] - span(comp[k].box, C)[0],
                                    span(comp[k + 1].box, C)[1] - span(comp[k + 1].box, C)[0])
                      for k in range(len(comp) - 1))
        if len(comp) > 1 and stacked:
            out.append(comp)
        else:
            out += [[p] for p in comp]
    return out


def _shared_align(l0, l1, m0, m1, tol) -> str | None:
    if abs((l0 + l1) / 2 - (m0 + m1) / 2) <= tol:
        return "centre"
    if abs(l0 - m0) <= tol:
        return "start"
    if abs(l1 - m1) <= tol:
        return "end"
    return None


def _block_align(ctx: _Ctx, pieces: list[_Piece]) -> str:
    L = ctx.axis
    side_l = ctx.Ws if L == "x" else ctx.Hs
    votes = {"centre": 0, "start": 0, "end": 0}
    for a, b in zip(pieces, pieces[1:]):
        sa = (a.box[0], a.box[0] + a.box[2]) if L == "x" else (a.box[1], a.box[1] + a.box[3])
        sb = (b.box[0], b.box[0] + b.box[2]) if L == "x" else (b.box[1], b.box[1] + b.box[3])
        k = _shared_align(*sa, *sb, ALIGN_TOL * side_l)
        if k:
            votes[k] += 1
    return max(("centre", "start", "end"), key=lambda k: votes[k])


def _halves(ctx: _Ctx, block: list[_Piece]) -> list[list[_Piece]]:
    """A block too long to fit, split in two at its widest source gap (ties: nearer the middle)."""
    C = "y" if ctx.axis == "x" else "x"

    def gap(k):
        a, b = block[k - 1].box, block[k].box
        return (b[1] - a[1] - a[3]) if C == "y" else (b[0] - a[0] - a[2])

    k = max(range(1, len(block)), key=lambda k: (round(gap(k), 1), -abs(k - len(block) / 2)))
    return [block[:k], block[k:]]


def _lay_entity(ctx: _Ctx, ent: _Entity, s: float, photo_k: float = 1.0) -> bool:
    """Size the entity's pieces and stack them (blocks: across the nudge axis,
    in source order, source gaps scaled by s, on their shared line)."""
    L = ctx.axis
    ent.sized = []
    for p in ent.pieces:
        cap = None
        if p.text and len(ent.pieces) == 1:
            kx = anchor(p.box[0], p.box[2], ctx.Ws)
            if kx == "stretch":
                g0, g1 = p.box[0], ctx.Ws - p.box[0] - p.box[2]
                cap = ctx.W - max(g0 * s, ctx.margin) - max(g1 * s, ctx.margin)
        sz = _size(ctx, p, ctx.k.get(p.unit.unit_id, s), cap, photo_k)
        if sz is None:
            return False
        ent.sized.append(sz)
    if len(ent.sized) == 1:
        sz = ent.sized[0]
        ent.w, ent.h = sz.w, sz.h
        ent.offs = [(0.0, 0.0)]
        ent.core = sz.core
        if sz.core is not None:  # a container that ran off the canvas (a blob behind a badge): only its contents count
            p = ent.pieces[0]
            ent.src_core = union_box([rotated_aabb(ctx.els[i]) for i in p.unit.element_ids if i != p.group.container])
        return True
    # a stack along C (the axis across the nudge axis)
    along = [(sz.h if L == "x" else sz.w) for sz in ent.sized]
    across = [(sz.w if L == "x" else sz.h) for sz in ent.sized]
    span_l = max(across)
    cursor, offs = 0.0, []
    for k, p in enumerate(ent.pieces):
        if k:
            prev = ent.pieces[k - 1].box
            gap = (p.box[1] - prev[1] - prev[3]) if L == "x" else (p.box[0] - prev[0] - prev[2])
            cursor += gap * s
            cursor = max(cursor, offs[-1][1] + 0.0)  # never above the previous piece's start
        a = across[k]
        lo = 0.0 if ent.align == "start" else span_l - a if ent.align == "end" else (span_l - a) / 2
        offs.append((lo, cursor))
        cursor += along[k]
    total = cursor - min(o[1] for o in offs)
    base = min(o[1] for o in offs)
    offs = [(lo, c - base) for lo, c in offs]
    if L == "x":
        ent.offs = [(lo, c) for lo, c in offs]
        ent.w, ent.h = span_l, total
    else:
        ent.offs = [(c, lo) for lo, c in offs]
        ent.w, ent.h = total, span_l
    return True


# ---------------------------------------------------------------- nudging

def _order(a: _Entity, b: _Entity, L: str) -> tuple[_Entity, _Entity]:
    """Which comes first along the nudge axis: by the source's order along
    it if they were apart along it, else by their order across it."""
    def box(e):
        return e.src_core if e.src_core is not None else e.box

    def lo(e, ax):
        return box(e)[0] if ax == "x" else box(e)[1]

    def hi(e, ax):
        return box(e)[0] + box(e)[2] if ax == "x" else box(e)[1] + box(e)[3]

    C = "y" if L == "x" else "x"
    if hi(a, L) <= lo(b, L) or hi(b, L) <= lo(a, L):
        return (a, b) if lo(a, L) + hi(a, L) <= lo(b, L) + hi(b, L) else (b, a)
    return (a, b) if lo(a, C) + hi(a, C) <= lo(b, C) + hi(b, C) else (b, a)


def _solve(sizes: list[float], desired: list[float], bounds: list[tuple[float, float]],
           cons: set[tuple[int, int]], gap: float) -> list[float] | None:
    """Positions on one axis, from the desired ones moved as little as the
    order constraints (a before b, `gap` apart) need, inside the bounds
    (iterative projection); None if they can't all hold."""
    p = [min(max(d, lo), max(lo, hi)) for d, (lo, hi) in zip(desired, bounds)]
    order = sorted(cons)
    for _ in range(SOLVE_ITERS):
        worst = 0.0
        for a, b in order:
            v = p[a] + sizes[a] + gap - p[b]
            if v <= 1e-6:
                continue
            room_a, room_b = max(0.0, p[a] - bounds[a][0]), max(0.0, bounds[b][1] - p[b])
            da = min(v / 2, room_a)
            db = min(v - da, room_b)
            da = min(v - db, room_a)
            p[a] -= da
            p[b] += db
            worst = max(worst, v)
        if worst <= 1e-3:
            break
    if any(p[a] + sizes[a] + gap - p[b] > 0.5 for a, b in order):
        return None
    return p


def _nudge(ctx: _Ctx, ents: list[_Entity], desired: dict, bounds: dict) -> bool:
    """Move the entities, as little as possible, until no two overlap that
    didn't in the source, keeping source order: a pair stacked across the
    nudge axis in the source is first kept apart across it (as it was), and
    along the nudge axis if that doesn't fit; any other pair along the nudge
    axis. False if no order-keeping placement fits the canvas."""
    L = ctx.axis
    C = "y" if L == "x" else "x"
    gap = GUTTER * ctx.short
    touch = TOUCH * ctx.short
    n = len(ents)

    def sbox(e) -> Box:
        return e.src_core if e.src_core is not None else e.box

    def off(e, ax) -> float:
        return 0.0 if e.core is None else (e.core[0] if ax == "x" else e.core[1])

    def iv(e, ax):
        r = e.rect()
        return (r[0], r[0] + r[2]) if ax == "x" else (r[1], r[1] + r[3])

    def src(e, ax):
        b = sbox(e)
        return (b[0], b[0] + b[2]) if ax == "x" else (b[1], b[1] + b[3])

    def sizes(ax):  # of the colliding part; the solver works in its coordinates
        return [e.rect()[2] if ax == "x" else e.rect()[3] for e in ents]

    des = {ax: [d + off(e, ax) for e, d in zip(ents, desired[ax])] for ax in ("x", "y")}
    bnd = {ax: [(lo + off(e, ax), hi + off(e, ax)) for e, (lo, hi) in zip(ents, bounds[ax])] for ax in ("x", "y")}

    def share(a: Box, b: Box) -> float:
        small = min(_area(a), _area(b))
        return overlap_area(a, b) / small if small > 0 else 0.0

    src_share = {(i, j): share(sbox(ents[i]), sbox(ents[j])) for i in range(n) for j in range(i + 1, n)}
    exempt = {k for k, v in src_share.items() if v > 0}

    def off_canvas(e) -> bool:
        return _off_canvas(ctx, sbox(e))

    def grew(i, j) -> bool:
        """An overlap the source had, now clearly larger (a piece that ran off the
        source canvas is pulled in, so its overlaps may grow)."""
        a, b = ents[i], ents[j]
        if off_canvas(a) or off_canvas(b):
            return False
        return share(a.rect(), b.rect()) > src_share[(i, j)] + OVERLAP_SHARE

    def stacked_c(i, j) -> bool:
        (a0, a1), (b0, b1) = src(ents[i], C), src(ents[j], C)
        return a1 <= b0 + 0.5 or b1 <= a0 + 0.5

    def inverted(i, j) -> bool:
        """Stacked across the nudge axis in the source, the other way round now."""
        (a0, a1), (b0, b1) = src(ents[i], C), src(ents[j], C)
        ci, cj = iv(ents[i], C), iv(ents[j], C)
        if a1 <= b0 + 0.5:
            return ci[0] + ci[1] > cj[0] + cj[1] + 0.5
        if b1 <= a0 + 0.5:
            return cj[0] + cj[1] > ci[0] + ci[1] + 0.5
        return False

    def colliding() -> list[tuple[int, int]]:
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in exempt and not grew(i, j):
                    continue
                ci, cj = iv(ents[i], C), iv(ents[j], C)
                li, lj = iv(ents[i], L), iv(ents[j], L)
                apart_c = ci[0] >= cj[1] + touch or cj[0] >= ci[1] + touch
                apart_l = li[0] >= lj[1] + gap - 0.5 or lj[0] >= li[1] + gap - 0.5
                if apart_l or (apart_c and not inverted(i, j)):
                    continue
                out.append((i, j))
        return out

    def ordered(i, j, ax) -> tuple[int, int]:
        if ax == C:  # stacked across: by the source's order across
            (a0, a1), (b0, b1) = src(ents[i], C), src(ents[j], C)
            return (i, j) if a0 + a1 <= b0 + b1 else (j, i)
        a, _ = _order(ents[i], ents[j], L)
        return (i, j) if a is ents[i] else (j, i)

    cons = {L: set(), C: set()}
    for ax in (L, C):
        for e, v in zip(ents, [min(max(d, lo), max(lo, hi)) for d, (lo, hi) in zip(desired[ax], bounds[ax])]):
            setattr(e, ax, v)
    desired, bounds = des, bnd
    for _ in range(NUDGE_ROUNDS):
        new = colliding()
        if not new:
            return True
        for i, j in new:
            if stacked_c(i, j) and ordered(i, j, C) not in cons[C]:
                trial = cons[C] | {ordered(i, j, C)}
                if _solve(sizes(C), desired[C], bounds[C], trial, gap) is not None:
                    cons[C] = trial
                    continue
            cons[L].add(ordered(i, j, L))
        for ax in (C, L):
            if any((b, a) in cons[ax] for a, b in cons[ax]):
                return False
            pos = _solve(sizes(ax), desired[ax], bounds[ax], cons[ax], gap)
            if pos is None:
                return False
            for e, v in zip(ents, pos):
                setattr(e, ax, v - off(e, ax))
    return False


def _lines(e: _Entity, ax: str) -> tuple[float, float, float]:
    lo, size = (e.x, e.w) if ax == "x" else (e.y, e.h)
    return lo, lo + size / 2, lo + size


def _snap(ctx: _Ctx, ents: list[_Entity], bounds: dict) -> None:
    """Shared lines: an entity that shares no edge or centre line with another
    on an axis moves onto the nearest one within SNAP of the long side, if
    that stays inside its bounds and overlaps nothing more (least important first)."""
    tol = SNAP * max(ctx.W, ctx.H)
    exact = ALIGNED * max(ctx.W, ctx.H)
    rank = {id(e): min((p.unit.rank if p.unit.rank is not None else 10 ** 6) for p in e.pieces) for e in ents}
    index = {id(e): i for i, e in enumerate(ents)}

    def overlaps(e) -> list[float]:
        return [overlap_area(e.rect(), f.rect()) for f in ents if f is not e]

    for ax in ("x", "y"):
        for e in sorted(ents, key=lambda e: -rank[id(e)]):
            mine = _lines(e, ax)
            others = [f for f in ents if f is not e]
            if any(abs(mine[k] - _lines(f, ax)[k]) <= exact for f in others for k in range(3)):
                continue
            moves = sorted((d for f in others for k in range(3)
                            for d in [_lines(f, ax)[k] - mine[k]] if abs(d) <= tol), key=abs)
            lo, hi = bounds[ax][index[id(e)]]
            before = overlaps(e)
            for d in moves:
                old = e.x if ax == "x" else e.y
                new = old + d
                if new < lo - 0.5 or new > max(lo, hi) + 0.5:
                    continue
                setattr(e, ax, new)
                if all(a <= b + 1e-6 for a, b in zip(overlaps(e), before)):
                    break
                setattr(e, ax, old)


# ---------------------------------------------------------------- layout

def _layout(ctx: _Ctx, pieces: list[_Piece], s: float, spread: bool) -> dict[str, dict] | None:
    """Element geometry for every content piece at scale s (photos shrunk by
    PHOTO_GIVE if that is what it takes to fit), or None."""
    has_photo = any(p.photo for p in pieces)
    for photo_k in PHOTO_GIVE if has_photo else PHOTO_GIVE[:1]:
        geo = _layout_at(ctx, pieces, s, spread, photo_k)
        if geo is not None:
            return geo
    return None


def _frames(ctx: _Ctx, pieces: list[_Piece], s: float, spread: bool) -> dict[str, tuple[float, float]]:
    """Where the pieces' anchors are taken, per axis: the canvas, or (unless
    `spread`) a frame at most SPREAD_MAX x the scaled source side long,
    placed by the anchor of the content's union: a tight cluster on a much
    longer side stays together instead of flying to the far edges."""
    ux, uy, uw, uh = union_box([p.box for p in pieces])
    out = {}
    for ax, lo, size, side, new in (("x", ux, uw, ctx.Ws, ctx.W), ("y", uy, uh, ctx.Hs, ctx.H)):
        flen = new if spread else min(new, side * s * SPREAD_MAX)
        kind = anchor(lo, size, side)
        off = 0.0 if kind == "start" else new - flen if kind == "end" else (new - flen) / 2
        out[ax] = (off, flen)
    return out


def _layout_at(ctx: _Ctx, pieces: list[_Piece], s: float, spread: bool, photo_k: float) -> dict[str, dict] | None:
    groups = _blocks(ctx, pieces)
    for _ in range(len(pieces) + 1):
        ents = []
        for g in groups:
            box = union_box([p.box for p in g])
            ents.append(_Entity(g, box, _block_align(ctx, g) if len(g) > 1 else "centre"))
        aw, ah = _avail(ctx)
        split = None
        for k, e in enumerate(ents):
            if not _lay_entity(ctx, e, s, photo_k) or e.w > ctx.W + 0.5 or e.h > ctx.H + 0.5:
                if len(e.pieces) > 1:
                    split = k
                    break
                return None
            if len(e.pieces) > 1 and (e.w > aw + 0.5 or e.h > ah + 0.5):
                split = k
                break
        if split is not None:
            groups = groups[:split] + _halves(ctx, groups[split]) + groups[split + 1:]
            continue
        break
    else:
        return None
    L = ctx.axis
    strip = min(ctx.W, ctx.H) <= STRIP * max(ctx.W, ctx.H)
    frames = _frames(ctx, pieces, s, spread)
    desired, bounds = {"x": [], "y": []}, {"x": [], "y": []}
    for e in ents:
        bx, by, bw, bh = e.box
        kx, ky = anchor(bx, bw, ctx.Ws), anchor(by, bh, ctx.Hs)
        if strip:  # a strip: across it everything sits on the middle line
            kx, ky = ("centre", ky) if L == "y" else (kx, "centre")
        x, x0, x1 = place_on_axis(kx, bx, bw, ctx.Ws, e.w, ctx.W, s, ctx.margin, spread and L == "x", strip,
                                  frames["x"])
        y, y0, y1 = place_on_axis(ky, by, bh, ctx.Hs, e.h, ctx.H, s, ctx.margin, spread and L == "y", strip,
                                  frames["y"])
        if x1 < x0 - 0.5 or y1 < y0 - 0.5:
            return None
        desired["x"].append(x)
        desired["y"].append(y)
        bounds["x"].append((x0, x1))
        bounds["y"].append((y0, y1))
    if not _nudge(ctx, ents, desired, bounds):
        return None
    _snap(ctx, ents, bounds)
    geo: dict[str, dict] = {}
    for e in ents:
        for sz, (ox, oy) in zip(e.sized, e.offs):
            for eid, g in sz.geo.items():
                geo[eid] = dict(g, x=g["x"] + e.x + ox, y=g["y"] + e.y + oy)
    return geo


def _hierarchy_ok(ctx: _Ctx, geo: dict[str, dict]) -> bool:
    """No text outgrows the headline that didn't in the source."""
    heads = [i for i, r in ctx.role.items() if r == "headline" and ctx.els[i]["type"] == "text"]
    if not heads:
        return True
    hs, hn = max(ctx.els[i]["font_size"] for i in heads), max(geo[i]["font_size"] for i in heads if i in geo)
    for eid, g in geo.items():
        if g.get("type") != "text" or eid in heads:
            continue
        if ctx.els[eid]["font_size"] <= hs + 1e-6 and g["font_size"] > hn + 0.01:
            return False
    return True


_BUILDER = None


def _builder_cls():
    """bisection v2's builder (frames follow their unit, parking that never
    straddles copy), with decorations first trying their own anchor at scale s."""
    global _BUILDER
    if _BUILDER is not None:
        return _BUILDER
    from engine.pipeline.layout.layers import fit_inside, scale_about
    from engine.pipeline.layout.strategies.bisection import _v2_builder, straddles
    from engine.pipeline.occlusion import is_covered, paint_order

    base = _v2_builder()

    class AnchoredBuilder(base):
        s = 1.0
        spread = False
        axis = "x"
        frames = {"x": None, "y": None}

        def place_layers(self) -> None:
            for unit in self.plan.units:
                if unit.role == "background":
                    from engine.pipeline.layout.layers import place_background
                    self.placed.update((m["id"], place_background(m, self.width, self.height))
                                       for m in self._members(unit))
                elif unit.role == "decoration":
                    for m in self._members(unit):
                        self.placed[m["id"]] = (self._round_backdrop(m) or self._frame(m) or self._anchored(m)
                                                or self._park(m) or self._visible_decoration(m))

        def _anchored(self, deco: dict) -> dict | None:
            sw, sh = self.source["canvas"]["width"], self.source["canvas"]["height"]
            bx, by, bw, bh = rotated_aabb(deco)
            if bw <= 0 or bh <= 0:
                return None
            k = self.s
            kx, ky = anchor(bx, bw, sw), anchor(by, bh, sh)
            new = scale_about(deco, k, deco["x"] + deco["width"] / 2, deco["y"] + deco["height"] / 2)
            nb = rotated_aabb(new)
            x = place_on_axis(kx, bx, bw, sw, nb[2], self.width, k, 0.0, self.spread and self.axis == "x",
                              frame=self.frames["x"])[0]
            y = place_on_axis(ky, by, bh, sh, nb[3], self.height, k, 0.0, self.spread and self.axis == "y",
                              frame=self.frames["y"])[0]
            new = dict(new, x=new["x"] + x - nb[0], y=new["y"] + y - nb[1])
            cand = fit_inside([new], self.width, self.height)[0]
            box = rotated_aabb(cand)
            if any(straddles(deco, box, o) for o in self._untouched(deco)) or self._covers_under(deco, box, set()):
                return None
            content = [self.placed[e["id"]] for e in self.source["elements"]
                       if e["id"] in self.placed and self._is_content(e["id"])]
            ordered = sorted(content + [cand], key=lambda e: self._source_index[e["id"]])
            return None if is_covered(cand, content, paint_order(ordered)) else cand

    _BUILDER = AnchoredBuilder
    return _BUILDER


def _build(ctx: _Ctx, pieces: list[_Piece], s: float, spread: bool, measurer, mode: str | None = None) -> Candidate | None:
    geo = _layout(ctx, pieces, s, spread)
    if geo is None or not _hierarchy_ok(ctx, geo):
        return None
    cls = _builder_cls()
    b = cls(ctx.source, ctx.target, ctx.layout, measurer)
    b.s, b.spread, b.axis = s, spread, ctx.axis
    b.frames = _frames(ctx, pieces, s, spread)
    for p in pieces:
        unit_geo = {i: geo[i] for i in p.unit.element_ids}
        b._record(p.unit, unit_geo, False)
    r = s / ctx.s0
    params = {"mode": mode or ("spread" if spread else "anchored"), "r": round(r, 3)}
    cid = f"{NAME}/" + "/".join(f"{k}={v}" for k, v in sorted(params.items()))
    return b.finish(cid, NAME, params)


def _ratios(ctx: _Ctx) -> list[float]:
    top = ctx.s_fill / ctx.s0
    out = [r for r in RATIOS if r <= top + 1e-9]
    if top < RATIOS[-1] and (not out or top - out[-1] > 0.03):
        out.append(top)
    return out


def candidates(source: dict, target: dict, plan: PriorityPlan, measurer) -> list[Candidate]:
    """Anchored candidates for this target (see the module docstring), each
    passing the hard gates. [] when the target has the source's aspect or
    anything goes wrong."""
    try:
        return _candidates(source, target, plan, measurer)
    except Exception:
        return []


def _candidates(source: dict, target: dict, plan: PriorityPlan, measurer) -> list[Candidate]:
    from engine.pipeline.grading.gates import check_gates

    if not source or not plan or measurer is None or not plan.ranked_units():
        return []
    src_w, src_h = source["canvas"]["width"], source["canvas"]["height"]
    W, H = target["width"], target["height"]
    if min(src_w, src_h, W, H) <= 0:
        return []
    if abs((W / H) / (src_w / src_h) - 1) <= SAME_ASPECT_TOL:
        return []
    layers, layout = None, plan
    if os.environ.get("CHAMELEON_LAYERS") == "1":  # (layers) a container and its contents move as one piece
        from engine.pipeline.layout import layer_groups
        layers = layer_groups.detect(source, plan, target)
        layout = layers.plan
    ctx = _Ctx(source, target, plan, layout, measurer, layers)
    pieces = _pieces(ctx)
    if not pieces:
        return []
    built: dict[tuple[float, bool], Candidate] = {}
    for r in _ratios(ctx):
        try:
            cand = _build(ctx, pieces, ctx.s0 * r, False, measurer)
        except Exception:
            cand = None
        if cand is not None:
            built[(r, False)] = cand
    feasible = sorted(r for r, _ in built)
    picks: list[tuple[float, bool]] = []
    if feasible:
        top = feasible[-1]
        picks.append((top, False))
        if feasible[0] != top:
            picks.append((feasible[0], False))
        mid = [r for r in feasible if feasible[0] < r < top]
        if mid:
            target_r = math.sqrt(feasible[0] * top)
            picks.append((min(mid, key=lambda r: abs(r - target_r)), False))
        for r in reversed(feasible):  # the spread variant at the largest scale that allows it
            try:
                cand = _build(ctx, pieces, ctx.s0 * r, True, measurer)
            except Exception:
                cand = None
            if cand is not None:
                built[(r, True)] = cand
                picks.append((r, True))
                break
    cap = MAX_CANDIDATES
    from engine.pipeline.layout import targets as _targets
    if _targets.enabled():  # (CHAMELEON_TARGETS=1) pieces at Sonnet's predicted sizes, anchored as usual
        extra = _target_builds(ctx, pieces, measurer, plan)
        built.update(extra)
        picks.extend(extra)
        cap += len(extra)
    out, seen = [], set()
    for key in picks:
        cand = built[key]
        try:
            if check_gates(source, cand.design, target, plan, measurer):
                continue
            digest = json.dumps(cand.design, sort_keys=True)
        except Exception:
            continue
        if digest in seen:
            continue
        seen.add(digest)
        out.append(cand)
        if len(out) >= cap:
            break
    return out


TARGET_PICKS = 2   # (targets) candidates with every piece at its predicted size: the largest feasible scale ...
                   # ... for the rest (positions, gaps, pieces without a target) and the one nearest the pieces' own


def _target_builds(ctx: _Ctx, pieces: list[_Piece], measurer, plan: PriorityPlan) -> dict:
    """(CHAMELEON_TARGETS=1) Each piece with a predicted size (layout/targets.py) is
    sized at that scale instead of the global s; positions, gaps and the
    other pieces follow s as usual. Built at every ratio; kept: the largest
    feasible s and the s nearest the pieces' own scales. Keys (r, "targets")."""
    from engine.pipeline.layout import targets as _targets
    T = _targets.predict(ctx.source, plan, ctx.target)
    if not T:
        return {}
    ks = {}
    for p in pieces:
        ids = set(p.unit.element_ids)
        members = [u for u in plan.ranked_units() if u.element_ids and set(u.element_ids) <= ids]
        k = _targets.group_ratio(members, ctx.els, T, ctx.W, ctx.H)  # target size / source size
        if k is not None:
            ks[p.unit.unit_id] = k
    if not ks:
        return {}
    ctx.k = ks
    built = {}
    try:
        for r in _ratios(ctx):
            try:
                cand = _build(ctx, pieces, ctx.s0 * r, False, measurer, mode="targets")
            except Exception:
                cand = None
            if cand is not None:
                built[(r, "targets")] = cand
    finally:
        ctx.k = {}
    if not built:
        return {}
    rs = sorted(r for r, _ in built)
    own = math.exp(sum(math.log(k) for k in ks.values()) / len(ks)) / ctx.s0
    keep = {rs[-1], min(rs, key=lambda r: abs(math.log(r / own)))}
    return {key: c for key, c in built.items() if key[0] in keep}
