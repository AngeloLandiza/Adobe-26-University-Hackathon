"""Text robustness pass (CHAMELEON_TEXT_SLACK; off by default, and with it
off nothing here runs and the engine's output is unchanged).

The official converter word-wraps each text greedily at its box width
(renderer/converter/wrap.py), and the engine sizes many boxes to exactly
their widest line: a quarter of all text boxes re-wrap if the box is 1 px
narrower. Any tiny difference at scoring time (rounding, another skia
build, font rasterisation on the reference machine) then re-wraps them, and
a CTA label that re-wraps spills its second line out of its pill. This pass
gives every text box slack, without changing content or font family:

1. **Pill labels** (a text in a container_label unit, or a text lying in a
   small shape of its own group, `pill_pairs`) render on no more lines than
   in the source (one, for every label in the samples), and keep slack:
   the label box widens inside its pill; past the pill's edge, the pill
   widens symmetrically with it if it stays inside the canvas safe area (or
   no closer to an edge than it was), inside every element that held it,
   and gains no overlap with other content; otherwise the label font steps
   down (0.5% steps) until the label fits with slack, never below its role
   floor (place.floor_px), the label box re-centred vertically on its old
   ink. A label that fits nowhere keeps the option with the fewest lines and
   is reported (`Result.notes`).
2. **Slack for every text box:** its wrap at width - slack_px(width) is the
   wrap at its width (max(1.5 px, 1.5% of the width) of room before any
   line re-wraps). The box widens first (text_align: left keeps x, right
   keeps the right edge, center grows both ways), which moves no ink: it
   must stay in the canvas and in every element that held it and gain no
   overlap with other content. When keeping the wrap gives no slack (the
   next word would join a line first) the box may widen up to REWRAP_MAX
   to a wrap with the same line count. Else the font steps down by at most
   MAX_SHRINK, never below the role floor, the line count unchanged, and a
   headline that was the largest text stays the largest. Last resort, at
   the same size and wrap: a box blocked on its aligned side grows on the
   other side (its ink shifts by at most SHIFT_MAX px).
3. **Vertical:** a box shorter than the text's drawn block (TextMeasurer.
   block_height) grows to it (moving up if it would leave the canvas),
   with no new overlap.

Every change is kept only if the design still passes every hard gate a
candidate passed before (each element's change is checked on its own if
the whole set fails); a candidate the gates already reject is left alone.

Where it runs (engine/resize.py): CHAMELEON_TEXT_SLACK=1 on the final
design (after the pick and the refiners), so the pick is the one without
the switch; CHAMELEON_TEXT_SLACK=all on every candidate before grading too.
Almost every change is a wider box with the ink where it was, which the
grader's box metrics read as a different layout although nothing renders
differently: on 99 cells (3 samples x 13 targets, 60 holdout) "all"
changed 10 picks for no visual reason, while both leave no text that
re-wraps 1 px narrower (dev/distill/text_robustness.py), so 1 is the
final-design pass. Cost: a few wraps per text element.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from engine.pipeline.geometry import Box, overlap_area, rotated_aabb
from engine.pipeline.types import Candidate, PriorityPlan

ENV = "CHAMELEON_TEXT_SLACK"
SLACK_PX = 1.5            # least slack, px ...
SLACK_SHARE = 0.015       # ... or this share of the box width, whichever is larger
MAX_SHRINK = 0.06         # a text (not a pill label) loses at most this share of its font size
SHRINK_STEP = 0.005       # font size steps, x the current size
SHIFT_MAX = 3.0           # px: last resort, a blocked box may grow on its other side, moving its ink this far at most
REWRAP_MAX = 0.15         # a box may widen by this share to find a same-line-count wrap with slack
INSIDE = 0.9              # a label's source box at least this much inside its shape ...
PILL_AREA = 0.15          # ... and the shape at most this share of the canvas (units.MAX_LABEL_SHAPE_AREA)
TOL = 0.5                 # px, containment tolerance (the gates' GROUP_TOL)
_LAYER_ROLES = ("background", "decoration")


def mode() -> str | None:
    """None (off), "final" (CHAMELEON_TEXT_SLACK=1: the final design) or
    "all" (CHAMELEON_TEXT_SLACK=all: every candidate before grading too)."""
    v = os.environ.get(ENV, "")
    return "final" if v == "1" else "all" if v == "all" else None


def slack_px(width: float) -> float:
    return max(SLACK_PX, SLACK_SHARE * width)


def _box(el: dict) -> Box:
    return (el["x"], el["y"], el["width"], el["height"])


def _inside_share(inner: Box, outer: Box) -> float:
    area = inner[2] * inner[3]
    return overlap_area(inner, outer) / area if area > 0 else 0.0


def _contains(outer: Box, inner: Box, tol: float = TOL) -> bool:
    return (outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1]
            and inner[0] + inner[2] <= outer[0] + outer[2] + tol and inner[1] + inner[3] <= outer[1] + outer[3] + tol)


def pill_pairs(source: dict, plan: PriorityPlan) -> dict[str, str]:
    """label text id -> its pill shape id: every container_label unit, plus a
    text in a multi-element unit lying (INSIDE) in a small shape of that unit
    painted under it."""
    els = {e["id"]: e for e in source["elements"]}
    cw, ch = source["canvas"]["width"], source["canvas"]["height"]
    out = {}
    for u in plan.units:
        members = [els[i] for i in u.element_ids if i in els]
        shapes = [m for m in members if m["type"] == "shape"]
        texts = [m for m in members if m["type"] == "text"]
        if not shapes or not texts:
            continue
        if u.kind == "container_label":
            out[texts[0]["id"]] = shapes[0]["id"]
            continue
        for t in texts:
            holders = [s for s in shapes if (s.get("z_index") or 0) <= (t.get("z_index") or 0)
                       and s["width"] * s["height"] <= PILL_AREA * cw * ch
                       and _inside_share(_box(t), _box(s)) >= INSIDE]
            if holders:
                out[t["id"]] = min(holders, key=lambda s: s["width"] * s["height"])["id"]
    return out


def robust(m, el: dict, font_size: float | None = None, width: float | None = None) -> bool:
    """Does the official wrap stay the same with the box slack_px narrower?"""
    f = el["font_size"] if font_size is None else font_size
    w = el["width"] if width is None else width
    return m.wrap(el, f, w - slack_px(w)) == m.wrap(el, f, w)


def _need(m, el: dict, f: float, wrapped: str) -> float:
    """Least max width that reproduces `wrapped` (greedy wrap: every line,
    plus the space it broke at when it holds more than one word, must fit)."""
    widths = m._char_widths(el, f)
    text = str(el.get("content", ""))
    need, start = 0.0, 0
    for i in range(len(wrapped) + 1):
        if i == len(wrapped) or wrapped[i] == "\n":
            w = sum(widths[start:i])
            if i < len(wrapped) and text[i] == " " and " " in text[start:i]:
                w += widths[i]
            need = max(need, w)
            start = i + 1
    return need


def _slack_width(m, el: dict, f: float, wrapped: str) -> float | None:
    """Least width (0.01 px grid) at which `el` at size f wraps as `wrapped`
    with slack, or None."""
    need = _need(m, el, f, wrapped)
    w = math.ceil(max(need + SLACK_PX, need / (1 - SLACK_SHARE)) * 100 + 1) / 100
    for _ in range(4):
        if m.wrap(el, f, w) == wrapped and m.wrap(el, f, w - slack_px(w)) == wrapped:
            return w
        w += 0.5
    return None


@dataclass
class Result:
    design: dict
    changed: list[str] = field(default_factory=list)   # element ids changed
    notes: list[str] = field(default_factory=list)     # e.g. "pill_unresolved:<id>"


class _Pass:
    def __init__(self, source: dict, target: dict, plan: PriorityPlan, m, design: dict):
        from engine.pipeline.layout.place import floor_px
        from engine.pipeline.layout.polish import safe_margins
        self.source, self.target, self.plan, self.m = source, target, plan, m
        self.W, self.H = float(target["width"]), float(target["height"])
        self.src = {e["id"]: e for e in source["elements"]}
        self.placed = {e["id"]: dict(e) for e in design["elements"]}
        self.pills = {t: s for t, s in pill_pairs(source, plan).items() if t in self.placed and s in self.placed}
        self.role = {}
        for u in plan.units:
            for i in u.element_ids:
                self.role[i] = u.role
        self.floor = lambda i: floor_px(self.role.get(i) or "", self.W, self.H)
        self.mx, self.my = safe_margins(self.W, self.H)
        self.notes: list[str] = []
        self.changes: dict[str, dict[str, dict]] = {}   # text id -> the geometry it changed

    # -- constraints -------------------------------------------------------------
    def _layer(self, eid: str) -> bool:
        return self.role.get(eid) in _LAYER_ROLES

    def _holders(self, eid: str, box: Box) -> list[str]:
        return [o for o, g in self.placed.items() if o != eid and g["type"] in ("shape", "image")
                and _contains(rotated_aabb(g), box)]

    def _ok(self, geo: dict[str, dict], exempt: set[str]) -> bool:
        """The changed elements stay in the canvas and in every element that
        held them, and gain no overlap with other content."""
        for eid, new in geo.items():
            old_b, new_b = rotated_aabb(self.placed[eid]), rotated_aabb(new)
            if not _contains((0, 0, self.W, self.H), new_b, 0.0):
                return False
            held = set(self._holders(eid, old_b))
            for o in held - set(geo):
                if not _contains(rotated_aabb(self.placed[o]), new_b):
                    return False
            for o, g in self.placed.items():
                if o in geo or o in held or o in exempt or self._layer(o):
                    continue
                ob = rotated_aabb(g)
                if overlap_area(new_b, ob) > overlap_area(old_b, ob) + 1e-6:
                    return False
        return True

    def _safe_pill(self, old: Box, new: Box) -> bool:
        """A pill's box no closer to an edge than the safe margin, or than it already was."""
        for d_new, d_old, mg in ((new[0], old[0], self.mx), (self.W - new[0] - new[2], self.W - old[0] - old[2], self.mx),
                                 (new[1], old[1], self.my), (self.H - new[1] - new[3], self.H - old[1] - old[3], self.my)):
            if d_new < min(mg, d_old) - 1e-6:
                return False
        return True

    def _headline_floor(self, eid: str) -> float:
        """A headline that is the largest text stays the largest."""
        if self.role.get(eid) != "headline":
            return 0.0
        heads = [g["font_size"] for i, g in self.placed.items() if g["type"] == "text" and self.role.get(i) == "headline"]
        others = [g["font_size"] for i, g in self.placed.items() if g["type"] == "text" and self.role.get(i) != "headline"]
        return max(others) if others and heads and max(heads) >= max(others) else 0.0

    # -- geometry ------------------------------------------------------------------
    def _widened(self, el: dict, w: float, symmetric: bool, anchor: str | None = None) -> dict:
        """`el` at width w: the side text_align keeps fixed stays (both sides
        grow for `symmetric` or centred text), or the side `anchor` names."""
        d = w - el["width"]
        side = anchor or ("center" if symmetric else el.get("text_align") or "left")
        x = el["x"] - d / 2 if side == "center" else el["x"] - d if side == "right" else el["x"]
        return dict(el, x=x, width=w)

    def _sized(self, el: dict, f: float) -> dict:
        from engine.pipeline.layout.place import resize_text
        return resize_text(el, f)

    def _recentre(self, old: dict, new: dict) -> dict:
        """Keep the label's ink centred where it was (cr-renderer draws from the box top)."""
        b0 = self.m.block_height(old, old["font_size"], old["width"])
        b1 = self.m.block_height(new, new["font_size"], new["width"])
        if b1 >= b0 - 1e-6:
            return new
        return dict(new, y=new["y"] + (min(b0, old["height"]) - b1) / 2, height=max(b1, new["height"] - (b0 - b1)))

    def _with_pill(self, label: dict, new: dict) -> dict[str, dict] | None:
        """Geometry for a pill label at `new`: the pill unchanged if it still
        holds the label box, else widened symmetrically just enough."""
        sid = self.pills[label["id"]]
        shape = self.placed[sid]
        sb, nb = _box(shape), _box(new)
        if _contains(sb, nb, 0.0) or shape.get("rotation") or new.get("rotation"):
            return {label["id"]: new} if _contains(sb, nb, 0.0) else None
        grow = max(0.0, sb[0] - nb[0], nb[0] + nb[2] - sb[0] - sb[2])
        if nb[1] < sb[1] - TOL or nb[1] + nb[3] > sb[1] + sb[3] + TOL:
            return None
        ns = dict(shape, x=shape["x"] - grow, width=shape["width"] + 2 * grow)
        if not self._safe_pill(sb, _box(ns)):
            return None
        return {label["id"]: new, sid: ns}

    def _try(self, geo: dict[str, dict] | None, exempt: set[str]) -> dict[str, dict] | None:
        if geo is None:
            return None
        texts = [g for g in geo.values() if g["type"] == "text"]
        if not all(self.m.fits(g, g["font_size"], g["width"], g["height"]) for g in texts):
            return None
        return geo if self._ok(geo, exempt) else None

    # -- the fixes ------------------------------------------------------------------
    def _sizes(self, el: dict, cap: float | None) -> list[float]:
        f0 = el["font_size"]
        lo = max(self.floor(el["id"]), self._headline_floor(el["id"]), 1.0)
        if cap is not None:
            lo = max(lo, f0 * (1 - cap))
        out, k = [f0], 1
        while True:
            f = f0 * (1 - SHRINK_STEP * k)
            if f < lo - 1e-9 or k > 400:
                break
            out.append(math.floor(f * 100) / 100)
            k += 1
        return out

    def fix_text(self, eid: str) -> None:
        el = self.placed[eid]
        if not str(el.get("content", "")).strip() or el["width"] <= 0:
            return
        pill = eid in self.pills
        lines = self.m.line_count(el, el["font_size"], el["width"])
        want = lines
        if pill:
            s0 = self.src[eid]
            want = max(1, self.m.line_count(s0, s0["font_size"], s0["width"]))
            if lines <= want and robust(self.m, el):
                return
        elif robust(self.m, el):
            return
        exempt = {self.pills[eid]} if pill else set()
        for f in self._sizes(el, None if pill else MAX_SHRINK):
            new = self._sized(el, f) if f != el["font_size"] else dict(el)
            geo = self._fit_at(el, new, want, pill, exempt)
            if geo is not None:
                self._commit(eid, geo)
                return
        if not pill and not el.get("rotation"):
            # Last resort, at the same size and wrap: grow on the free side, the ink shifting
            # by the widening (at most SHIFT_MAX px) when the aligned side is blocked.
            w2 = _slack_width(self.m, el, el["font_size"], self.m.wrap(el, el["font_size"], el["width"]))
            if w2 is not None and w2 - el["width"] <= SHIFT_MAX:
                for anchor in ("left", "right", "center"):
                    geo = self._try({eid: self._widened(el, w2, False, anchor)}, exempt)
                    if geo is not None:
                        self._commit(eid, geo)
                        return
        if pill and lines > want:
            self._best_effort_pill(el, exempt)

    def _fit_at(self, el: dict, new: dict, want: int, pill: bool, exempt: set[str]) -> dict[str, dict] | None:
        """Geometry for `el` drawn as `new` (its font already set) with slack:
        as it is, or widened to the least width that keeps the wrap (a pill
        label: that wraps to at most `want` lines) with slack."""
        f, w = new["font_size"], new["width"]
        cur = self.m.wrap(new, f, w)
        n = cur.count("\n") + 1
        lines_ok = n <= want if pill else n == want
        if lines_ok and robust(self.m, new):
            return self._place(el, new, pill, exempt)
        rotated = bool(new.get("rotation"))
        if rotated and (new.get("text_align") or "left") != "center" and not pill:
            return None  # a rotated box turns about its centre: only a centred text keeps its ink when it widens
        wrapped = cur if lines_ok else self._wrap_for(new, want) if pill else None
        if wrapped is None:
            return None
        w2 = _slack_width(self.m, new, f, wrapped)
        if w2 is None and lines_ok:
            w2 = self._rewrap_width(new, n)
        if w2 is None:
            return None
        return self._place(el, self._widened(new, max(w, w2), symmetric=pill or rotated), pill, exempt)

    def _rewrap_width(self, el: dict, lines: int) -> float | None:
        """Widening that keeps the wrap gives no slack (the next word would
        join a line first): the least wider width, up to REWRAP_MAX x, that
        wraps to the same number of lines with slack (the breaks move)."""
        f, w = el["font_size"], el["width"]
        k = 1
        while k * 0.5 <= REWRAP_MAX * w:
            w2 = w + k * 0.5
            n = self.m.line_count(el, f, w2)
            if n < lines:
                return None
            if n == lines and robust(self.m, el, f, w2):
                return w2
            k += 1
        return None

    def _place(self, el: dict, new: dict, pill: bool, exempt: set[str]) -> dict[str, dict] | None:
        if pill:
            return self._try(self._with_pill(el, self._recentre(el, new)), exempt)
        return self._try({el["id"]: new}, exempt)

    def _wrap_for(self, el: dict, lines: int) -> str | None:
        """The wrap of `el` at the least width (up to the canvas) giving at most `lines` lines."""
        f = el["font_size"]
        if self.m.line_count(el, f, self.W) > lines:
            return None
        lo, hi = el["width"], self.W
        for _ in range(30):
            mid = (lo + hi) / 2
            if self.m.line_count(el, f, mid) <= lines:
                hi = mid
            else:
                lo = mid
        return self.m.wrap(el, f, hi)

    def _best_effort_pill(self, el: dict, exempt: set[str]) -> None:
        """No size fits: at the floor, widen label and pill as far as allowed;
        kept only if that drops lines."""
        sizes = self._sizes(el, None)
        new = self._sized(el, sizes[-1]) if sizes[-1] != el["font_size"] else dict(el)
        f = new["font_size"]
        lines0 = self.m.line_count(el, el["font_size"], el["width"])
        best, lo, hi = None, new["width"], self.W
        for _ in range(20):
            mid = (lo + hi) / 2
            geo = self._try(self._with_pill(el, self._recentre(el, self._widened(new, mid, True))), exempt)
            if geo is not None:
                best, lo = geo, mid
            else:
                hi = mid
        if best is not None and self.m.line_count(best[el["id"]], f, best[el["id"]]["width"]) < lines0:
            self._commit(el["id"], best)
        self.notes.append(f"pill_unresolved:{el['id']}")

    def fix_height(self, eid: str) -> None:
        el = self.placed[eid]
        need = self.m.block_height(el, el["font_size"], el["width"])
        if el["height"] >= need - 1e-6 or el.get("rotation"):
            return
        h = math.ceil(need * 100) / 100
        y = min(el["y"], self.H - h)
        if y < 0:
            return
        exempt = {self.pills[eid]} if eid in self.pills else set()
        geo = self._try({eid: dict(el, y=y, height=h)}, exempt)
        if geo is not None:
            self._commit(eid, geo)

    def _commit(self, eid: str, geo: dict[str, dict]) -> None:
        self.changes.setdefault(eid, {}).update(geo)
        self.placed.update(geo)

    def run(self) -> None:
        texts = [e["id"] for e in self.source["elements"] if e["type"] == "text" and e["id"] in self.placed]
        for eid in sorted(texts, key=lambda i: i not in self.pills):  # pill labels first
            self.fix_text(eid)
        for eid in texts:
            self.fix_height(eid)


def fix_design(source: dict, target: dict, plan: PriorityPlan, measurer, design: dict) -> Result:
    """`design` with text slack (Result.design is `design` itself when nothing changed or survived)."""
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.layout.normalize import finalize
    if check_gates(source, design, target, plan, measurer):
        return Result(design)
    p = _Pass(source, target, plan, measurer, design)
    p.run()
    if not p.changes:
        return Result(design, [], p.notes)

    def build(geos: list[dict[str, dict]]) -> dict | None:
        placed = {e["id"]: dict(e) for e in design["elements"]}
        for g in geos:
            placed.update(g)
        out = finalize(source, placed, target)
        return out if out is not None and not check_gates(source, out, target, plan, measurer) else None

    out = build(list(p.changes.values()))
    kept = list(p.changes)
    if out is None:  # keep each element's change only if it passes on its own (in order)
        kept, geos = [], []
        for eid, g in p.changes.items():
            if build(geos + [g]) is not None:
                geos.append(g)
                kept.append(eid)
        out = build(geos) if geos else None
    if out is None:
        return Result(design, [], p.notes)
    return Result(out, kept, p.notes)


def apply(source: dict, target: dict, plan: PriorityPlan, measurer, cand: Candidate) -> Candidate:
    """`cand` with text slack (same id, template, params, degraded units); never raises."""
    try:
        res = fix_design(source, target, plan, measurer, cand.design)
    except Exception:
        return cand
    if res.design is cand.design:
        return cand
    return Candidate(cand.candidate_id, res.design, cand.template, cand.params, list(cand.degraded_unit_ids),
                     cand.is_fallback)


def apply_all(source: dict, target: dict, plan: PriorityPlan, measurer, candidates: list[Candidate]) -> list[Candidate]:
    return [apply(source, target, plan, measurer, c) for c in candidates]
