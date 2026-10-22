"""Oracle-plan test: does "plan from Sonnet + geometry from the solver" keep
Sonnet's quality? (docs/research/beating-v5-offline.md, approach #2.)

Sonnet's compact holdout layouts (training-data/llm-distill/qc.nosync/
gen_holdout*.jsonl) are reduced to a DISCRETE plan per content unit (the
layout plan's units: with CHAMELEON_LAYERS=1 a container and its contents
are one unit):

- region: the range of cells of a coarse grid on the target's safe area it
  covers (3x3; 1x5 on wide/strip targets, 5x1 on very tall ones);
- order: reading order (region row, column, then position);
- anchor per axis relative to the canvas: start | centre | end | none;
- scale bucket: text by its largest font / the short side, anything else by
  sqrt(area) / sqrt(canvas), on a geometric grid (x1.3 per step);
- group: units Sonnet set close together and aligned (union-find);
- stack axis of a region holding several units; demoted (text at its floor);
- tree: nested rows / columns of the units (XY-cut of Sonnet's boxes), and
  each text's line count.

`realize` throws Sonnet's coordinates away and rebuilds the layout from the
plan only. The plan's structure is the XY-cut tree of Sonnet's units (nested
rows and columns with their order; discrete, like a region grid but without
cells that are too coarse for a headline spanning two of them) plus a line
count per text. The tree is allocated top-down in the safe area, each unit
at its bucket size (text at its bucket font, never below its role floor, on
its planned number of lines; the engine's place_unit fits it), the
background and decorations are placed the engine's way, and then the
geometry solver (engine/pipeline/layout/solver.py, `realize`) redistributes
the space with the plan's anchors as pulls. This is the geometry a future
small "plan model" would get.

    python -m dev.distill.oracle_plan run --baseline v5 [--name oracle_plan_solver] [--naive-name oracle_plan_naive]
    python -m dev.distill.oracle_plan show --cell <cell>           # the plan of one cell

Writes A/B variants (winners.jsonl + renders.nosync/) over the cells of the
baseline that have a Sonnet layout; a plan that fails a hard gate falls
back to the baseline's winner (like llm_layout ingest). Gitignored data: the
Sonnet layouts are under qc.nosync; holdout ads are our own synthetic ads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dev.distill import ab_engine as ab

QC = ab.DATA_ROOT / "llm-distill" / "qc.nosync"
GEN_FILES = ("gen_holdout.jsonl", "gen_holdout_rejected.jsonl")
V5 = {"CHAMELEON_STRATEGIES": "bisection,anchored", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
      "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1", "CHAMELEON_TEXT_SLACK": "1"}
ANCHOR_TOL = 0.1      # side gaps within this share of the side: centred
EDGE_ANCHOR = 0.15    # nearer gap at most this share of the side: anchored to that edge
COVER = 0.3           # a unit covers a grid cell when it overlaps this share of the cell's extent
FONT_BASE, AREA_BASE, STEP = 0.01, 0.05, 1.3  # bucket grids: base x STEP**k
GROUP_GAP = 0.03      # units this close (share of the short side) ...
GROUP_ALIGN = 0.02    # ... sharing an edge or centre line within this share of the long side are one group
GAP = 0.02            # naive placement: gap between stacked units, share of the short side
BLEED = 0.01          # a unit this close to a canvas edge (share of the side) bleeds off it
_LAYER = ("background", "decoration")


def grid_shape(W: float, H: float) -> tuple[int, int]:
    """(rows, cols) of the plan grid for a target."""
    a = W / H
    if a >= 2.2:
        return 1, 5
    if a <= 0.45:
        return 5, 1
    return 3, 3


@dataclass
class UnitPlan:
    uid: str
    rows: tuple[int, int]
    cols: tuple[int, int]
    order: int
    anchor: dict
    kind: str             # text | box
    bucket: int
    group: int
    demoted: bool = False
    stack: str | None = None   # axis of the region's stack (filled for regions with several units)
    lines: int = 1             # text: the number of lines Sonnet set it on
    bleed: tuple = (False, False, False, False)  # touched the canvas edge (l, t, r, b) in Sonnet's layout


@dataclass
class Plan:
    rows: int
    cols: int
    units: dict[str, UnitPlan] = field(default_factory=dict)
    tree: list | str | None = None  # nested rows/columns: unit id, or [axis ("x" | "y" | "free"), [children]]

    def to_json(self) -> dict:
        return {"grid": [self.rows, self.cols], "tree": self.tree,
                "units": {u: vars(p) for u, p in self.units.items()}}


# ---------------------------------------------------------------- extraction

def _layout(source: dict, plan, target: dict):
    from engine.pipeline.layout import layer_groups
    return layer_groups.detect(source, plan, target).plan


def _content(layout, els: dict) -> list:
    return [u for u in layout.ranked_units() if u.role not in _LAYER and u.element_ids
            and all(i in els for i in u.element_ids)]


def _ink(el: dict, m):
    from engine.pipeline.geometry import rotated_aabb
    if el["type"] != "text" or el.get("rotation"):
        return rotated_aabb(el)
    widths = m.line_widths(el, el["font_size"], el["width"]) or [0.0]
    w = min(max(widths), el["width"])
    align = el.get("text_align") or "left"
    x = el["x"] + (el["width"] - w) / 2 if align == "center" else \
        el["x"] + el["width"] - w if align == "right" else el["x"]
    return (x, el["y"], w, min(el["height"], m.block_height(el, el["font_size"], el["width"])))


def _range(lo: float, hi: float, origin: float, cell: float, n: int) -> tuple[int, int]:
    """Grid cells [a, b] an interval covers (COVER of a cell), at least the one holding its centre."""
    c = min(n - 1, max(0, int(((lo + hi) / 2 - origin) // cell)))
    hit = [k for k in range(n) if min(hi, origin + (k + 1) * cell) - max(lo, origin + k * cell) >= COVER * cell]
    hit = sorted(set(hit) | {c})
    return hit[0], hit[-1]


def _box(el: dict):
    from engine.pipeline.geometry import rotated_aabb
    return rotated_aabb(el)


def _anchor(lo: float, hi: float, side: float) -> str:
    g0, g1 = lo, side - hi
    if abs(g0 - g1) <= ANCHOR_TOL * side:
        return "centre"
    if min(g0, g1) <= EDGE_ANCHOR * side:
        return "start" if g0 < g1 else "end"
    return "none"


def bucket(value: float, base: float) -> int:
    return int(round(math.log(max(value, 1e-6) / base) / math.log(STEP)))


def extract(source: dict, target: dict, design: dict, plan, m) -> Plan:
    """The discrete plan of a layout (see the module docstring)."""
    from engine.pipeline.geometry import union_box
    from engine.pipeline.layout.place import floor_px
    from engine.pipeline.layout.polish import safe_margins
    W, H = float(target["width"]), float(target["height"])
    els = {e["id"]: e for e in design["elements"]}
    layout = _layout(source, plan, target)
    units = _content(layout, els)
    R, C = grid_shape(W, H)
    mx, my = safe_margins(W, H)
    cw, ch = (W - 2 * mx) / C, (H - 2 * my) / R
    short, long_ = min(W, H), max(W, H)
    boxes = {u.unit_id: union_box([_ink(els[i], m) for i in u.element_ids]) for u in units}
    out = Plan(R, C)
    # groups: close and aligned pairs
    parent = {u.unit_id: u.unit_id for u in units}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            A, B = boxes[a.unit_id], boxes[b.unit_id]
            dx = max(0.0, max(A[0], B[0]) - min(A[0] + A[2], B[0] + B[2]))
            dy = max(0.0, max(A[1], B[1]) - min(A[1] + A[3], B[1] + B[3]))
            lines = [(A[0], B[0]), (A[0] + A[2] / 2, B[0] + B[2] / 2), (A[0] + A[2], B[0] + B[2]),
                     (A[1], B[1]), (A[1] + A[3] / 2, B[1] + B[3] / 2), (A[1] + A[3], B[1] + B[3])]
            aligned = any(abs(p - q) <= GROUP_ALIGN * long_ for p, q in lines)
            if max(dx, dy) <= GROUP_GAP * short and aligned:
                parent[find(a.unit_id)] = find(b.unit_id)
    gid = {}
    for u in units:
        gid.setdefault(find(u.unit_id), len(gid))
    for u in units:
        x, y, w, h = boxes[u.unit_id]
        texts = [els[i] for i in u.element_ids if els[i]["type"] == "text"]
        if texts:
            f = max(t["font_size"] for t in texts)
            kind, b = "text", bucket(f / short, FONT_BASE)
            demoted = f <= 1.05 * floor_px(u.role, W, H)
        else:
            kind, b = "box", bucket(math.sqrt(max(w * h, 1.0) / (W * H)), AREA_BASE)
            demoted = False
        out.units[u.unit_id] = UnitPlan(u.unit_id, _range(y, y + h, my, ch, R), _range(x, x + w, mx, cw, C), 0,
                                        {"x": _anchor(x, x + w, W), "y": _anchor(y, y + h, H)}, kind, b,
                                        gid[find(u.unit_id)], demoted)
        tol_x, tol_y = BLEED * W, BLEED * H
        cb = union_box([_ink(els[i], m) if els[i]["type"] == "text" else _box(els[i]) for i in u.element_ids])
        out.units[u.unit_id].bleed = (cb[0] <= tol_x, cb[1] <= tol_y, cb[0] + cb[2] >= W - tol_x, cb[1] + cb[3] >= H - tol_y)
        if len(texts) == 1 and len(u.element_ids) == 1:
            t = texts[0]
            out.units[u.unit_id].lines = max(1, m.line_count(t, t["font_size"], t["width"]))
    order = sorted(units, key=lambda u: (out.units[u.unit_id].rows[0], out.units[u.unit_id].cols[0],
                                         boxes[u.unit_id][1], boxes[u.unit_id][0]))
    for k, u in enumerate(order):
        out.units[u.unit_id].order = k
    # stack axis of units sharing a region range: the axis they are farther apart on in Sonnet's layout
    by_region = defaultdict(list)
    for u in units:
        p = out.units[u.unit_id]
        by_region[(p.rows, p.cols)].append(u.unit_id)
    from engine.pipeline.layout.solver import xy_tree
    ids_all = [u.unit_id for u in units]
    out.tree = _tree_json(xy_tree(list(range(len(ids_all))), [boxes[i] for i in ids_all], W, H), ids_all, boxes)
    for ids in by_region.values():
        if len(ids) < 2:
            continue
        bs = [boxes[i] for i in ids]
        spread_y = max(b[1] + b[3] / 2 for b in bs) - min(b[1] + b[3] / 2 for b in bs)
        spread_x = max(b[0] + b[2] / 2 for b in bs) - min(b[0] + b[2] / 2 for b in bs)
        for i in ids:
            out.units[i].stack = "y" if spread_y >= spread_x else "x"
    return out


def _tree_json(node, ids: list[str], boxes: dict):
    """solver.Node -> nested lists; a `free` node (no clean cut: overlapping units) keeps the axis
    its units are farther apart on."""
    if node.kind == "leaf":
        return ids[node.leaf]
    kids = [_tree_json(c, ids, boxes) for c in node.children]
    if node.kind == "free":
        bs = [boxes[ids[i]] for i in node.leaves()]
        sy = max(b[1] + b[3] / 2 for b in bs) - min(b[1] + b[3] / 2 for b in bs)
        sx = max(b[0] + b[2] / 2 for b in bs) - min(b[0] + b[2] / 2 for b in bs)
        ax = "y" if sy >= sx else "x"
        key = (lambda i: boxes[i][1]) if ax == "y" else (lambda i: boxes[i][0])
        kids.sort(key=lambda k: min(key(i) for i in _tree_ids(k)))
        return ["free-" + ax, kids]
    return [node.axis, kids]


def _tree_ids(t) -> list[str]:
    return [t] if isinstance(t, str) else [i for c in t[1] for i in _tree_ids(c)]


# ---------------------------------------------------------------- realization

def _font(p: UnitPlan, unit, W: float, H: float) -> float:
    from engine.pipeline.layout.place import floor_px
    floor = floor_px(unit.role, W, H)
    if p.demoted:
        return floor
    return max(floor, STEP ** p.bucket * FONT_BASE * min(W, H))


class _Naive:
    """Top-down allocation of the plan's tree in the safe area, units at their bucket sizes."""

    def __init__(self, source: dict, target: dict, plan, P: Plan, m):
        from engine.pipeline.geometry import rotated_aabb, union_box
        from engine.pipeline.layout.generate import _Builder
        from engine.pipeline.layout.polish import safe_margins
        self.P, self.m = P, m
        self.W, self.H = float(target["width"]), float(target["height"])
        self.layout = _layout(source, plan, target)
        self.els = {e["id"]: e for e in source["elements"]}
        self.units = {u.unit_id: u for u in _content(self.layout, self.els) if u.unit_id in P.units}
        self.ub = {k: union_box([rotated_aabb(self.els[i]) for i in u.element_ids]) for k, u in self.units.items()}
        self.mx, self.my = safe_margins(self.W, self.H)
        self.gap = GAP * min(self.W, self.H)
        self.b = _Builder(source, target, self.layout, m)

    # -- wanted extents ------------------------------------------------------
    def _single_text(self, uid):
        u = self.units[uid]
        texts = [self.els[i] for i in u.element_ids if self.els[i]["type"] == "text"]
        return texts[0] if len(u.element_ids) == 1 and len(texts) == 1 else None

    def _scale(self, uid) -> float:
        """Unit scale vs its source size at its bucket size."""
        u, p = self.units[uid], self.P.units[uid]
        if p.kind == "text":
            f = _font(p, u, self.W, self.H)
            return f / max(self.els[i]["font_size"] for i in u.element_ids if self.els[i]["type"] == "text")
        side = STEP ** p.bucket * AREA_BASE * math.sqrt(self.W * self.H)
        _, _, w, h = self.ub[uid]
        return side / math.sqrt(max(w * h, 1.0))

    def _text_width(self, uid) -> float:
        """Narrowest width that sets the text on at most its planned lines, at its bucket font."""
        from engine.pipeline.layout.place import resize_text
        el = self._single_text(uid)
        p = self.P.units[uid]
        f = _font(p, self.units[uid], self.W, self.H)
        t = dict(el, font_size=f)  # source letter spacing, as place_unit's fit_font_size measures it
        full = max(self.m.line_widths(t, f, float("inf")))
        lo, hi = full / max(1, p.lines) * 0.5, full + 1
        for _ in range(18):
            mid = (lo + hi) / 2
            if self.m.line_count(t, f, mid) <= p.lines and max(self.m.line_widths(t, f, mid)) <= mid:
                hi = mid
            else:
                lo = mid
        return hi + 2

    def ext(self, node, ax: str, cross: float, low: bool = False) -> float:
        """Wanted extent along `ax` (`low`: the least it can take, its text at the role floor)."""
        from engine.pipeline.layout.place import floor_px
        if isinstance(node, str):
            el = self._single_text(node)
            u = self.units[node]
            if el is not None:
                from engine.pipeline.layout.place import resize_text
                f = _font(self.P.units[node], u, self.W, self.H)
                w = self._text_width(node)
                if low:
                    f = floor_px(u.role, self.W, self.H)
                    w = max(self.m.line_widths(resize_text(el, f), f, float("inf"))) + 2 if ax == "x" else w
                w = min(cross, w) if ax == "y" else w
                if ax == "x":
                    return w
                return self.m.block_height(resize_text(el, f), f, w) + 1
            k = self._scale(node)
            if low:
                texts = [self.els[i]["font_size"] for i in u.element_ids if self.els[i]["type"] == "text"]
                k = floor_px(u.role, self.W, self.H) / max(texts) if texts else k * 0.3
            _, _, w, h = self.ub[node]
            w, h = w * k, h * k
            if ax == "y":
                return h * min(1.0, cross / w) if w else h
            return w * min(1.0, cross / h) if h else w
        axis = node[0][-1]
        kids = node[1]
        if axis == ax:
            return sum(self.ext(c, ax, cross, low) for c in kids) + self.gap * (len(kids) - 1)
        o = "x" if ax == "y" else "y"
        nat = [self.ext(c, o, math.inf if o == "x" else cross, low) for c in kids]
        room = cross - self.gap * (len(kids) - 1)
        k = min(1.0, room / sum(nat)) if sum(nat) > 0 else 1.0
        return max(self.ext(c, ax, n * k, low) for c, n in zip(kids, nat))

    # -- placement -------------------------------------------------------------
    def place(self, uid: str, box) -> bool:
        from engine.pipeline.geometry import align_in
        from engine.pipeline.layout.place import HARD_FLOOR, floor_px, place_unit
        u, p = self.units[uid], self.P.units[uid]
        align = {"start": "left", "end": "right"}.get(p.anchor["x"], "center")
        floor = floor_px(u.role, self.W, self.H)
        max_font = None
        if p.kind == "text":
            max_font = _font(p, u, self.W, self.H)
            el = self._single_text(uid)
            if el is not None:  # the planned line length, not the whole cell
                w = min(box[2], self._text_width(uid))
                x, _ = align_in(box, w, box[3], align)
                box = (x, box[1], w, box[3])
        else:
            k = self._scale(uid)
            _, _, w, h = self.ub[uid]
            w, h = min(box[2], w * k), min(box[3], h * k)
            x, y = align_in(box, w, h, align)
            box = (x, y, w, h)
            align = "center"
        geo = place_unit(u, self.els, box, self.m, floor, align=align, max_font=max_font)
        if geo is None:
            geo = place_unit(u, self.els, box, self.m, HARD_FLOOR, align=align, max_font=max_font)
        if geo is None:
            return False
        self.b._record(u, geo, False)
        return True

    def _bleeds(self, node, k: int) -> bool:
        if isinstance(node, str):
            return node in self.P.units and bool(self.P.units[node].bleed[k])
        kids = node[1]
        if node[0][-1] == ("y" if k in (1, 3) else "x"):
            return self._bleeds(kids[0] if k < 2 else kids[-1], k)
        return any(self._bleeds(c, k) for c in kids)

    def alloc(self, node, rect) -> bool:
        if isinstance(node, str):
            return node not in self.units or self.place(node, rect)
        ax = node[0][-1]
        kids = node[1]
        x, y, w, h = rect
        length, cross = (h, w) if ax == "y" else (w, h)
        want = [self.ext(c, ax, cross) for c in kids]
        room = length - self.gap * (len(kids) - 1)
        if sum(want) > room:  # squeeze: every child keeps at least its extent at the role floors
            low = [min(wv, self.ext(c, ax, cross, True)) for c, wv in zip(kids, want)]
            spare = sum(want) - sum(low)
            t = max(0.0, (room - sum(low)) / spare) if spare > 0 else 0.0
            ext = [lo + (wv - lo) * t for lo, wv in zip(low, want)]
            if sum(ext) > room:
                ext = [e * room / sum(ext) for e in ext]
        else:
            ext = want
        used = sum(ext) + self.gap * (len(kids) - 1)
        start = y if ax == "y" else x
        k0, k1 = (1, 3) if ax == "y" else (0, 2)
        if self._bleeds(kids[0], k0):        # a band off the start edge holds the pack there
            pos = start
        elif self._bleeds(kids[-1], k1):
            pos = start + length - used
        else:
            pos = start + (length - used) / 2   # the pack centred; the solver spreads it
        for c, e in zip(kids, ext):
            sub = (x, pos, w, e) if ax == "y" else (pos, y, e, h)
            if not self.alloc(c, sub):
                return False
            pos += e + self.gap
        return True

    def _bleed(self) -> None:
        """A stretching container (rectangle / cover image) that touched canvas edges in the plan
        reaches them again, if that newly covers no other unit."""
        from engine.pipeline.geometry import overlap_area, rotated_aabb, union_box
        for uid, u in self.units.items():
            p = self.P.units[uid]
            if u.kind != "layer_group" or not any(p.bleed):
                continue
            c = self.b.placed.get(u.element_ids[0])
            src = self.els[u.element_ids[0]]
            if c is None or not ((src["type"] == "shape" and src.get("shape_type", "rectangle") == "rectangle")
                                 or (src["type"] == "image" and src.get("fit") == "cover")):
                continue
            x0, y0, x1, y1 = c["x"], c["y"], c["x"] + c["width"], c["y"] + c["height"]
            l, t, r, b = p.bleed
            nx0, ny0 = (0.0 if l else x0), (0.0 if t else y0)
            nx1, ny1 = (self.W if r else x1), (self.H if b else y1)
            old, new = (x0, y0, x1 - x0, y1 - y0), (nx0, ny0, nx1 - nx0, ny1 - ny0)
            others = [union_box([rotated_aabb(self.b.placed[i]) for i in v.element_ids])
                      for k, v in self.units.items() if k != uid and all(i in self.b.placed for i in v.element_ids)]
            if any(overlap_area(new, o) > overlap_area(old, o) + 1e-6 for o in others):
                continue
            self.b.placed[c["id"]] = dict(c, x=nx0, y=ny0, width=nx1 - nx0, height=ny1 - ny0)

    def run(self) -> dict | None:
        from engine.pipeline.layout.normalize import finalize
        if not self.units or self.P.tree is None:
            return None
        safe = (self.mx, self.my, self.W - 2 * self.mx, self.H - 2 * self.my)
        if not self.alloc(self.P.tree, safe):
            return None
        self._bleed()
        try:
            self.b.place_layers()
        except Exception:
            return None
        return finalize(self.b.source, self.b.placed, {"width": int(self.W), "height": int(self.H)})


def naive(source: dict, target: dict, plan, P: Plan, m) -> dict | None:
    """The plan placed without Sonnet's coordinates (see the module docstring); None if a unit can't be placed."""
    return _Naive(source, target, plan, P, m).run()


def realize(source: dict, target: dict, plan, P: Plan, m, stats: dict | None = None) -> tuple[dict | None, dict | None]:
    """(naive placement, plan + solver) for a plan; either may be None."""
    from engine.pipeline.layout import solver
    design = naive(source, target, plan, P, m)
    if design is None:
        return None, None
    layout = _layout(source, plan, target)
    anchors = {uid: {ax: a for ax, a in p.anchor.items() if a != "none"} for uid, p in P.units.items()}
    solved = solver.realize(source, target, plan, m, design, anchors, layout=layout, stats=stats)
    return design, solved if solved is not None else design


# ---------------------------------------------------------------- run

def _sonnet_rows() -> dict[str, dict]:
    out = {}
    for f in GEN_FILES:
        p = QC / f
        if p.exists():
            for line in p.open():
                r = json.loads(line)
                if r.get("answer"):
                    out.setdefault(r["cell"], r)
    return out


def run(baseline: str, name: str, naive_name: str | None) -> None:
    os.environ.update(V5)
    os.environ["CHAMELEON_LLM"] = "heuristic"
    os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
    from dev.distill.corpus import load_corpus
    from dev.distill.llm_layout import _compact_to_elements, execute
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.priority import plan as priority_plan
    from engine.pipeline.render import render_to_image
    from engine.resize import _get_measurer

    m = _get_measurer()
    designs = {d["task_id"]: d for d in load_corpus(None)}
    targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
    base = ab.load_winners(baseline)
    sonnet = _sonnet_rows()
    cells = sorted(c for c in base if c in sonnet)
    outs = {name: [], **({naive_name: []} if naive_name else {})}
    plans = {}
    tally = defaultdict(int)
    for c in cells:
        r = base[c]
        source = designs[r["task_id"]]["source"]
        t = targets[r["target"]]
        target = {"width": t["width"], "height": t["height"]}
        plan = priority_plan(source)
        st: dict = {}
        try:
            raw = execute(source, target, plan, m, _compact_to_elements(json.loads(sonnet[c]["answer"])["e"]))
            P = extract(source, target, raw, plan, m)
            plans[c] = P.to_json()
            nv, sv = realize(source, target, plan, P, m, st)
        except Exception as exc:
            nv = sv = None
            st["status"] = f"error:{type(exc).__name__}: {exc}"
        for vname, design in ((name, sv), (naive_name, nv)):
            if vname is None:
                continue
            failed = ["no_design"] if design is None else check_gates(source, design, target, plan, m)
            out_dir = ab.AB_ROOT / vname
            png = Path(r["png"])
            (out_dir / png).parent.mkdir(parents=True, exist_ok=True)
            if failed:
                tally[f"{vname}:fallback"] += 1
                for f in failed:
                    tally[f"{vname}:{f}"] += 1
                (out_dir / png).write_bytes((ab.AB_ROOT / baseline / r["png"]).read_bytes())
                outs[vname].append({**r, "variant": vname, "candidate_id": "fallback-" + baseline,
                                    "failed_plan_gates": failed})
            else:
                tally[f"{vname}:ok"] += 1
                render_to_image(design, assets_dir=ab._assets(), max_size=ab.RENDER_SIZE).save(out_dir / png)
                outs[vname].append({**r, "variant": vname, "candidate_id": "oracle_plan", "is_fallback": False,
                                    "design_hash": ab.design_hash(design), "failed_gates": [],
                                    "solver_status": st.get("status")})
        print(f"[oracle] {c}: solver={st.get('status')}", flush=True)
    for vname, rows in outs.items():
        out_dir = ab.AB_ROOT / vname
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "variant.json").write_text(json.dumps(
            {"name": vname, "env": {**V5, "ORACLE_PLAN": "sonnet gen_holdout", "solver": str(vname == name),
                                    "fallback": baseline}}, indent=2))
        (out_dir / "winners.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (QC / f"plans_{name}.json").write_text(json.dumps(plans, indent=1))
    print(json.dumps({"cells": len(cells), **tally}, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dev.distill.oracle_plan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--baseline", default="v5")
    r.add_argument("--name", default="oracle_plan_solver")
    r.add_argument("--naive-name", default="oracle_plan_naive")
    s = sub.add_parser("show")
    s.add_argument("--plans", default=str(QC / "plans_oracle_plan_solver.json"))
    s.add_argument("--cell", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.baseline, args.name, args.naive_name or None)
    else:
        print(json.dumps(json.loads(Path(args.plans).read_text())[args.cell], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
