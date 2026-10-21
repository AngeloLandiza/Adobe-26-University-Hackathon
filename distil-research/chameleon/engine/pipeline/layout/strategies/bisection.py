"""Strategy S2 `bisection`: priority-weighted recursive splitting.

A treemap-like layout that works for any target shape, extreme strips
(728x90, 320x50) and skyscrapers (160x600) included.

1. The plan's content units become *items* in source reading order (top to
   bottom, then left to right); consecutive same-role non-text units, such as
   a run of badges, stay together as one item.
2. The content box is cut recursively into two parts until each part holds
   one item: a clearly elongated piece along its long side, a squarish one
   along whichever axis leaves the tighter side more room above its floors
   (a lone text is judged by the font size it would actually fit at; near-ties
   stack top to bottom). Each part's share of the cut comes from
   `split_extent`: its value (priority weight x role boost x sqrt of the area
   it would like at a comfortable size), clamped to what it needs at its
   readability floor and to what it can actually use (text stops at 1.5x its
   source size or its hierarchy cap, an image at its aspect, a logo at a
   modest cap), so spare room goes to whoever can use it.
3. Each item is placed in its cell with the engine's `place_unit`, most
   important first, keeping the hierarchy: no other text outgrows the
   headline, legal stays at its floor, the logo keeps a visible minimum.
4. Items that ended up below their floor get a bigger value and the layout is
   redone (a few rounds); the best round is kept.

On a crowded canvas the least important items (never the headline; the hero,
logo and CTA last) leave the cuts for a "minor zone" cell where they are
shelf-packed at their floor size, or smaller (6 px, then 4 px) if even that
doesn't leave the main items their floors.

Variants: cut order (`rank`: cut next to the most important item so it gets
its own cell early; `balanced`: cut where the value halves), margin 0.04 /
0.07, direction (`long`: as above; `flow`: the first cut follows the canvas's
reading direction, left to right when wide, top down otherwise). Identical
layouts are returned once. Never raises; every candidate is
built valid by construction (finalize + the engine's layer placement) and
re-checked against the hard gates before it is returned.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from engine.pipeline.geometry import Box, align_in, rotated_aabb, union_box
from engine.pipeline.layout.place import HARD_FLOOR, MAX_TEXT_GROWTH, floor_px, place_unit, resize_text
from engine.pipeline.types import Candidate, PriorityPlan, Unit

NAME = "bisection"
MARGINS = (0.04, 0.07)
ORDERS = ("rank", "balanced")
DIRECTIONS = ("long", "flow")
GUTTER = 0.025            # space between sibling cells, share of the canvas's short side
ROLE_BOOST = {"headline": 1.4, "cta": 1.2, "logo": 1.6}
COMFORT = {"headline": 1.6, "cta": 1.3, "subheadline": 1.2}  # desired text size, x the role's floor
HIERARCHY_CAP = 0.9       # no other text grows past this share of the headline's size ...
RATIO_SLACK = 1.3         # ... nor past 1.3x its source size ratio to the headline
LOGO_MAX_SHORT = 0.28     # a logo is at most this share of the short side tall ...
LOGO_MAX_WIDTH = 0.35     # ... and this share of the canvas wide (or its min height, if larger)
LOGO_MIN_SHARE = 0.08     # logo height at least this share of the short side ...
LOGO_MIN_PX = 12          # ... and at least this many px (capped at half the short side)
HERO_MIN_SHORT = 0.3      # ... and a hero's long side at least this share of the canvas's short side
HERO_MIN = 0.3            # an image may shrink to this share of its largest fit before it's "cramped"
ROOM_CAP = 3.0            # roominess above this counts the same
AXIS_FREE = 3.0           # a piece longer than this ratio is always cut along its long side
STACK_BIAS = 1.15         # near-ties between the two axes go to stacking (a horizontal cut)
REPAIR_ROUNDS = 5
EMPTY_PENALTY = 0.5       # internal score: legibility minus this x the largest empty rectangle share
BOOST = 1.8               # value multiplier for an item that missed its floor
ZONE_FONTS = (None, HARD_FLOOR, 4.0)  # the minor zone's text size: the floors, then smaller on a crowded canvas
CROWD_SLACK = 1.8        # content needs this much more room than its floor-size area to lay out well
SAME_ASPECT_TOL = 0.02    # same as generate.py: the proportional reflow is the answer there
_NO_CLUSTER = {"headline", "subheadline", "other_text", "legal", None}
_SEARCH_STEPS = 14


@dataclass
class Span:
    """What one side of a cut needs along the cut axis: at least `lo`, can use
    at most `hi`, and asks for a share in proportion to `value`."""

    lo: float
    hi: float
    value: float


def split_extent(length: float, a: Span, b: Span) -> float:
    """Extent along the cut axis for side `a` (side `b` gets the rest).

    Minimums first (scaled down together if they can't both be met), then the
    value share, then any room a saturated side can't use goes to the other;
    when both saturate, the surplus is shared in proportion to their maximums
    so the empty space is spread rather than pooled on one side."""
    if a.lo + b.lo >= length:
        total = a.lo + b.lo
        return length * a.lo / total if total > 0 else length / 2
    if a.hi + b.hi <= length:
        total = a.hi + b.hi
        return length * a.hi / total if total > 0 else length / 2
    total = a.value + b.value
    e = length * (a.value / total if total > 0 else 0.5)
    e = min(max(e, a.lo), length - b.lo)
    if e > a.hi:
        e = a.hi
    elif length - e > b.hi:
        e = length - b.hi
    return e


def logo_min_height(width: float, height: float) -> float:
    short = min(width, height)
    return min(max(LOGO_MIN_PX, LOGO_MIN_SHARE * short), 0.5 * short)


@dataclass
class _Item:
    units: list[Unit]
    key: int
    order: float              # source reading position
    rank: int
    base_value: float
    boost: float = 1.0
    cells: list[Box] = field(default_factory=list)
    minor: bool = False       # the zone for demoted units: packed in shelves at their floor size
    zone_font: float | None = None  # on a crowded canvas the zone's text goes below the floors, to this size

    @property
    def role(self) -> str | None:
        return self.units[0].role

    @property
    def value(self) -> float:
        return self.base_value * self.boost


class _Model:
    """Size model for one source/target pair: what each unit needs and can
    use along an axis given the cross dimension. Cached; never measures the
    same thing twice."""

    def __init__(self, source: dict, target: dict, plan: PriorityPlan, measurer):
        self.source, self.plan, self.m = source, plan, measurer
        self.W, self.H = float(target["width"]), float(target["height"])
        self.els = {el["id"]: el for el in source["elements"]}
        sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
        self.sw, self.sh = sw, sh
        self.s_area = math.sqrt(self.W * self.H / (sw * sh))
        self._cache: dict = {}
        head = next((u for u in plan.ranked_units() if u.role == "headline"), None)
        sizes = [self.els[i]["font_size"] for i in head.element_ids if self.els[i]["type"] == "text"] if head else []
        self.head_src = max(sizes) if sizes else None
        self.head_font: float | None = None  # the headline's placed size, once a round has placed it
        self.last_head: float | None = None

    # -- text helpers ------------------------------------------------------
    def floor(self, unit: Unit) -> float:
        return floor_px(unit.role, self.W, self.H)

    def text_cap(self, unit: Unit, head: float | None = None) -> float:
        """Largest size the unit's biggest text may take: legal stays at its
        floor; other text keeps roughly its source ratio to the headline (and
        stays below it) once the headline's size is known."""
        floor = self.floor(unit)
        big = max((self.els[i]["font_size"] for i in unit.element_ids if self.els[i]["type"] == "text"), default=0.0)
        if unit.role == "legal":
            return floor
        cap = big * MAX_TEXT_GROWTH
        head = self.head_font if head is None else head
        if unit.role != "headline" and head and self.head_src:
            cap = min(cap, head * min(HIERARCHY_CAP, RATIO_SLACK * big / self.head_src))
        return max(cap, floor)

    def max_font(self, unit: Unit, el: dict) -> float:
        return self.text_cap(unit)

    def _line(self, el: dict, f: float) -> tuple[float, float]:
        """(widest line, block height) of `el` at size f with no wrapping."""
        key = ("line", el["id"], f)
        if key not in self._cache:
            t = resize_text(el, f)
            self._cache[key] = (max(self.m.line_widths(t, f, float("inf")), default=0.0),
                                self.m.block_height(t, f, float("inf")))
        return self._cache[key]

    def _word(self, el: dict, f: float) -> float:
        key = ("word", el["id"], f)
        if key not in self._cache:
            self._cache[key] = self.m.longest_word_width(resize_text(el, f), f)
        return self._cache[key]

    def _fits(self, el: dict, f: float, w: float, h: float) -> bool:
        return self.m.fits(resize_text(el, f), f, w, h)

    def _text_ext(self, el: dict, f: float, axis: str, cross: float) -> float:
        """Least extent along `axis` for `el` at size f, the cross side fixed."""
        if axis == "y":  # height for a cross-wide box
            if self._word(el, f) > cross + 0.5:
                return math.inf
            return self.m.block_height(resize_text(el, f), f, cross)
        line_w, line_h = self._line(el, f)
        if line_h > cross + 1:
            return math.inf
        lo, hi = self._word(el, f), line_w + 1
        if self._fits(el, f, lo, cross):
            return lo
        if not self._fits(el, f, hi, cross):
            return math.inf
        for _ in range(_SEARCH_STEPS):
            mid = (lo + hi) / 2
            if self._fits(el, f, mid, cross):
                hi = mid
            else:
                lo = mid
        return hi

    def _largest_font(self, el: dict, f_lo: float, f_hi: float, axis: str, cross: float) -> float:
        """Roughly the largest size in [f_lo, f_hi] whose least extent is finite."""
        if math.isfinite(self._text_ext(el, f_hi, axis, cross)):
            return f_hi
        lo, hi = f_lo, f_hi
        for _ in range(8):
            mid = (lo + hi) / 2
            if math.isfinite(self._text_ext(el, mid, axis, cross)):
                lo = mid
            else:
                hi = mid
        return lo

    # -- extents -----------------------------------------------------------
    def unit_ext(self, unit: Unit, axis: str, cross: float) -> tuple[float, float]:
        """(least extent at the floor, most extent it can use) along `axis`."""
        key = ("unit", unit.unit_id, axis, round(cross, 1), self.head_font)
        if key not in self._cache:
            try:
                ext = self._unit_ext(unit, axis, cross, self.floor(unit))
                if not math.isfinite(ext[0]):  # can't reach its floor here: size it at the hard floor
                    ext = self._unit_ext(unit, axis, cross, HARD_FLOOR)
                if not math.isfinite(ext[0]):  # can't be placed this way at all
                    ext = (math.inf, math.inf)
            except Exception:
                ext = (math.inf, math.inf)
            self._cache[key] = ext
        return self._cache[key]

    def _unit_ext(self, unit: Unit, axis: str, cross: float, f0: float) -> tuple[float, float]:
        members = [self.els[i] for i in unit.element_ids]
        texts = [m for m in members if m["type"] == "text"]
        if len(members) == 1 and texts:
            el = texts[0]
            lo = self._text_ext(el, f0, axis, cross)
            if not math.isfinite(lo):
                return lo, lo
            f_hi = max(f0, self.max_font(unit, el))
            f = self._largest_font(el, f0, f_hi, axis, cross)
            if axis == "x":
                # f is the largest size whose single line fits the height: wider than
                # that line, the text can't grow any more
                hi = self._line(el, f)[0] + 1
            else:
                hi = self._text_ext(el, f, axis, cross)
            return lo, max(lo, hi)

        uw, uh = union_box([rotated_aabb(m) for m in members])[2:]
        if uw <= 0 or uh <= 0:
            return 0.0, 0.0
        along, other = (uw, uh) if axis == "x" else (uh, uw)
        if len(members) == 1:  # an image or shape: keeps its aspect
            k_max = cross / other
            if unit.role == "logo":
                k_max = min(k_max, MAX_TEXT_GROWTH, self.logo_cap(uw, uh))
                k_min = min(logo_min_height(self.W, self.H) / uh, k_max)
            else:
                k_min = k_max * HERO_MIN
                if unit.role == "hero_image":  # the product stays a real presence, not a speck
                    k_min = max(k_min, HERO_MIN_SHORT * min(self.W, self.H) / max(uw, uh))
                k_min = min(k_min, k_max)
            return along * k_min, along * k_max

        # A multi-element unit scales rigidly (a pill may also condense round its label).
        k_hi = cross / other
        if texts:
            big = max(t["font_size"] for t in texts)
            k_hi = min(k_hi, MAX_TEXT_GROWTH, self.text_cap(unit) / big)
            k_lo = f0 / big * 1.05
            lo = along * k_lo if other * k_lo <= cross else math.inf
            if unit.kind == "container_label" and not any(m.get("rotation") for m in members):
                label = texts[0]
                line_w, line_h = self._line(label, f0)
                cw, ch = line_w + 1.6 * f0, line_h + 0.4 * f0
                c_along, c_other = (cw, ch) if axis == "x" else (ch, cw)
                if c_other <= cross:
                    lo = min(lo, c_along)
        else:
            k_lo = k_hi * HERO_MIN
            lo = along * k_lo
        return lo, max(lo, along * k_hi) if math.isfinite(lo) else lo

    def item_ext(self, item: _Item, axis: str, cross: float) -> tuple[float, float]:
        """A run of units sits in equal cells on one or more lines, like the engine's place_row."""
        if item.minor:
            return self.minor_ext(item, axis, cross)
        if len(item.units) == 1:
            return self.unit_ext(item.units[0], axis, cross)
        key = ("item", item.key, axis, round(cross, 1), self.head_font)
        if key in self._cache:
            return self._cache[key]
        n = len(item.units)
        best_lo, best_hi = math.inf, 0.0
        for lines in range(1, n + 1):
            per_line = math.ceil(n / lines)
            if axis == "x":  # lines stacked in the cross (height); cells side by side along x
                exts = [self.unit_ext(u, "x", cross / lines) for u in item.units]
                lo, hi = per_line * max(e[0] for e in exts), per_line * max(e[1] for e in exts)
            else:
                exts = [self.unit_ext(u, "y", cross / per_line) for u in item.units]
                lo, hi = lines * max(e[0] for e in exts), lines * max(e[1] for e in exts)
            best_lo = min(best_lo, lo)
            if math.isfinite(hi):
                best_hi = max(best_hi, hi)
        out = (best_lo, max(best_hi, best_lo) if math.isfinite(best_lo) else best_lo)
        self._cache[key] = out
        return out

    # -- the minor zone ----------------------------------------------------
    def minor_size(self, unit: Unit, max_w: float, font: float | None = None) -> tuple[float, float]:
        """A demoted unit's box at its floor (or `font`, if smaller): text on as
        few lines as fit `max_w`, a pill condensed round its label, an image
        one text line tall."""
        members = [self.els[i] for i in unit.element_ids]
        texts = [m for m in members if m["type"] == "text"]
        f0 = self.zone_floor(unit, font)
        if len(members) == 1 and texts:
            el = texts[0]
            line_w, line_h = self._line(el, f0)
            if line_w <= max_w:
                return line_w + 1, line_h
            if self._word(el, f0) > max_w:
                return math.inf, math.inf
            return max_w, self.m.block_height(resize_text(el, f0), f0, max_w)
        uw, uh = union_box([rotated_aabb(m) for m in members])[2:]
        if texts and unit.kind == "container_label" and not any(m.get("rotation") for m in members):
            line_w, line_h = self._line(texts[0], f0)
            return line_w + 1.6 * f0 + 1, line_h + 0.4 * f0 + 1
        if texts:
            k = f0 / max(t["font_size"] for t in texts) * 1.05
            return uw * k + 1, uh * k + 1
        h = max(2.2 * min(floor_px("legal", self.W, self.H), font or math.inf), HARD_FLOOR * 2)
        if unit.role == "hero_image":
            h *= 2.5
        if unit.role == "logo":
            h = max(h, logo_min_height(self.W, self.H))
        return h * uw / uh if uh else h, h

    def logo_cap(self, uw: float, uh: float) -> float:
        """Largest scale for a logo: a modest mark, not a hero."""
        short = min(self.W, self.H)
        h_max = max(logo_min_height(self.W, self.H), LOGO_MAX_SHORT * short)
        return min(h_max / uh, LOGO_MAX_WIDTH * self.W / uw)

    def zone_floor(self, unit: Unit, font: float | None) -> float:
        floor = self.floor(unit)
        return floor if font is None else min(floor, font)

    def shelves(self, units: list[Unit], width: float, gap: float,
                font: float | None = None) -> list[tuple[Unit, Box]]:
        """Shelf packing at the floor: left to right, a new shelf when the row is full."""
        out, x, y, shelf = [], 0.0, 0.0, 0.0
        for unit in units:
            w, h = self.minor_size(unit, width, font)
            if not math.isfinite(w):
                return []
            if x > 0 and x + w > width:
                x, y, shelf = 0.0, y + shelf + gap, 0.0
            out.append((unit, (x, y, w, h)))
            x += w + gap
            shelf = max(shelf, h)
        return out

    def minor_ext(self, item: _Item, axis: str, cross: float) -> tuple[float, float]:
        key = ("minor", item.zone_font, tuple(u.unit_id for u in item.units), axis, round(cross, 1))
        if key in self._cache:
            return self._cache[key]
        gap = 0.02 * min(self.W, self.H)

        def height(width: float) -> float:
            packed = self.shelves(item.units, width, gap, item.zone_font)
            return max((b[1] + b[3] for _, b in packed), default=math.inf) if packed else math.inf

        if axis == "y":
            lo = height(cross)
        else:
            sizes = [self.minor_size(u, math.inf, item.zone_font) for u in item.units]
            lo_w, hi_w = 1.0, sum(w for w, _ in sizes) + gap * len(sizes)
            if height(hi_w) > cross:
                lo = math.inf
            else:
                for _ in range(_SEARCH_STEPS):
                    mid = (lo_w + hi_w) / 2
                    if height(mid) <= cross:
                        hi_w = mid
                    else:
                        lo_w = mid
                lo = hi_w
        out = (lo, lo * 1.15) if math.isfinite(lo) else (math.inf, math.inf)
        self._cache[key] = out
        return out

    def span(self, items: list[_Item], axis: str, cross: float, length: float, gutter: float) -> Span:
        """What a side of a cut needs along `axis`. A side holding several items
        is cut again, either along this axis (their extents add up) or across
        it (each spans the side and they share the cross); it needs the less
        demanding of the two and can use what that arrangement can use."""
        value = sum(i.value for i in items)
        if len(items) == 1:
            lo, hi = self.item_ext(items[0], axis, cross)
            return Span(lo, hi, value) if math.isfinite(lo) else Span(0.0, 0.0, value)
        g = gutter * (len(items) - 1)
        along = [self.item_ext(i, axis, cross) for i in items]
        lo_along, hi_along = sum(e[0] for e in along) + g, sum(e[1] for e in along) + g
        other = "y" if axis == "x" else "x"
        free = cross - g

        def across_need(e: float) -> float:
            return sum(self.item_ext(i, other, e)[0] for i in items)

        lo_across = math.inf
        if free > 0 and across_need(length) <= free:
            lo_e, hi_e = 0.0, length
            for _ in range(10):
                mid = (lo_e + hi_e) / 2
                if across_need(mid) <= free:
                    hi_e = mid
                else:
                    lo_e = mid
            lo_across = hi_e
        if lo_along <= lo_across:
            lo, hi = lo_along, hi_along
        else:
            lo, hi = lo_across, max(self.item_ext(i, axis, free)[1] for i in items)
        if not math.isfinite(lo):
            return Span(0.0, length, value)
        return Span(lo, max(lo, hi) if math.isfinite(hi) else length, value)

    def min_area(self, unit: Unit) -> float:
        """Rough area the unit takes at its floor: text on one line, a pill
        condensed round its label, an image at its least useful size."""
        members = [self.els[i] for i in unit.element_ids]
        texts = [m for m in members if m["type"] == "text"]
        f0 = self.floor(unit)
        content = self.W * self.H
        if len(members) == 1 and texts:
            line_w, line_h = self._line(texts[0], f0)
            return line_w * line_h
        if texts and unit.kind == "container_label":
            line_w, line_h = self._line(texts[0], f0)
            return (line_w + 1.6 * f0) * (line_h + 0.4 * f0)
        uw, uh = union_box([rotated_aabb(m) for m in members])[2:]
        if texts:
            k = f0 / max(t["font_size"] for t in texts)
            return uw * uh * k * k
        if unit.role == "logo":
            h = logo_min_height(self.W, self.H)
            return h * h * (uw / uh if uh else 1.0)
        return (0.08 if unit.role == "hero_image" else 0.02) * content

    # -- value -------------------------------------------------------------
    def need(self, unit: Unit) -> float:
        """Area the unit would like: text at a comfortable size, images at the area-scaled source size."""
        members = [self.els[i] for i in unit.element_ids]
        texts = [m for m in members if m["type"] == "text"]
        f0 = self.floor(unit)
        if len(members) == 1 and texts:
            el = texts[0]
            if unit.role == "legal":
                f = f0
            else:
                f = min(max(el["font_size"] * self.s_area, f0 * COMFORT.get(unit.role, 1.0)),
                        max(f0, el["font_size"] * MAX_TEXT_GROWTH))
            line_w, line_h = self._line(el, f)
            return line_w * line_h
        uw, uh = union_box([rotated_aabb(m) for m in members])[2:]
        area = uw * uh
        if texts:
            big = max(t["font_size"] for t in texts)
            f = min(max(big * self.s_area, f0 * COMFORT.get(unit.role, 1.0)), big * MAX_TEXT_GROWTH)
            return area * (f / big) ** 2
        k = self.s_area if unit.role != "logo" else min(self.s_area, MAX_TEXT_GROWTH)
        return area * k * k


def _items(model: _Model, plan: PriorityPlan) -> list[_Item]:
    groups: list[list[Unit]] = []
    for unit in plan.ranked_units():
        if groups and unit.role not in _NO_CLUSTER and groups[-1][-1].role == unit.role:
            groups[-1].append(unit)
        else:
            groups.append([unit])
    top = max((plan.weight(g[0].unit_id) for g in groups), default=1.0) or 1.0
    items = []
    for key, units in enumerate(groups):
        boxes = [rotated_aabb(model.els[i]) for u in units for i in u.element_ids]
        x, y, w, h = union_box(boxes)
        order = (y + h / 2) / model.sh + 0.25 * (x + w / 2) / model.sw
        weight = max(plan.weight(u.unit_id) for u in units) / top
        need = sum(model.need(u) for u in units)
        value = math.sqrt(need) * weight * ROLE_BOOST.get(units[0].role, 1.0)
        items.append(_Item(units, key, order, min(u.rank for u in units), max(value, 1e-6)))
    items.sort(key=lambda i: (i.order, i.rank))
    return items


def _cut_index(items: list[_Item], order: str) -> int:
    total = sum(i.value for i in items)
    best_k, best_gap, acc = 1, math.inf, 0.0
    gaps = {}
    for k in range(1, len(items)):
        acc += items[k - 1].value
        gaps[k] = abs(acc - total / 2)
        if gaps[k] < best_gap:
            best_k, best_gap = k, gaps[k]
    if order == "balanced":
        return best_k
    top = min(range(len(items)), key=lambda i: (items[i].rank, i))
    if top == 0:
        return 1
    if top == len(items) - 1:
        return len(items) - 1
    return min((top, top + 1), key=lambda k: (gaps[k], k))


def _cut(model: _Model, a: list[_Item], b: list[_Item], box: Box, axis: str, gutter: float):
    """Split `box` along `axis` between sides a and b. Returns (box_a, box_b, room): room is
    the tighter side's size relative to its floor size (see `_room`; higher is roomier)."""
    x, y, w, h = box
    length, cross = (w, h) if axis == "x" else (h, w)
    g = gutter if length > 4 * gutter else 0.0
    sa = model.span(a, axis, cross, length - g, gutter)
    sb = model.span(b, axis, cross, length - g, gutter)
    e = min(max(split_extent(length - g, sa, sb), 0.0), length - g)
    if axis == "x":
        box_a, box_b = (x, y, e, h), (x + e + g, y, w - e - g, h)
    else:
        box_a, box_b = (x, y, w, e), (x, y + e + g, w, h - e - g)
    return box_a, box_b, min(_room(model, a, sa, e, box_a), _room(model, b, sb, length - g - e, box_b))


def _room(model: _Model, items: list[_Item], span: Span, extent: float, box: Box) -> float:
    """How roomy a side is, as a linear scale over its floor size: a lone text
    by the largest size it fits at over its floor (capped where it stops
    growing), anything else by the square root of its extent over its least
    extent. A narrow column that only just holds a paragraph at its floor
    scores low even though it is 'feasible'."""
    if len(items) == 1 and not items[0].minor and len(items[0].units) == 1:
        unit = items[0].units[0]
        members = [model.els[i] for i in unit.element_ids]
        if len(members) == 1 and members[0]["type"] == "text":
            el, floor = members[0], model.floor(unit)
            cap = model.text_cap(unit)
            size = model.m.fit_font_size(el, box[2], box[3], 1.0, max(cap, 1.0)) if min(box[2:]) >= 1 else None
            return (size or 0.0) / floor
    if span.lo <= 0:
        return ROOM_CAP
    return min(ROOM_CAP, math.sqrt(max(extent, 0.0) / span.lo))


def _layout(model: _Model, items: list[_Item], box: Box, order: str, direction: str, main: str,
            gutter: float, depth: int = 0) -> None:
    """Cut recursively. A clearly elongated piece is cut along its long side;
    a squarish one along whichever axis leaves the tighter side more room
    above its floors (near-ties stack top to bottom, the way copy reads).
    `flow` makes the first cut follow the canvas's reading direction
    (left to right on a wide canvas, top down on a tall or square one)."""
    if len(items) == 1:
        items[0].cells = [box]
        return
    w, h = box[2], box[3]
    k = _cut_index(items, order)
    a, b = items[:k], items[k:]
    long_axis = "x" if w > h else "y"
    if direction == "flow" and depth == 0:
        axes = [main]
    elif max(w, h) > AXIS_FREE * min(w, h):
        axes = [long_axis]
    else:
        axes = [long_axis, "y" if long_axis == "x" else "x"]
    best = None
    for axis in axes:
        box_a, box_b, slack = _cut(model, a, b, box, axis, gutter)
        score = slack * (STACK_BIAS if axis == "y" else 1.0)
        if best is None or score > best[0]:
            best = (score, box_a, box_b)
    _layout(model, a, best[1], order, direction, main, gutter, depth + 1)
    _layout(model, b, best[2], order, direction, main, gutter, depth + 1)


def _place_item(b, model: _Model, item: _Item, caps: dict) -> list[tuple[Unit, dict | None, bool]]:
    """Units of one item into its cell: one unit fills it; a run sits in equal
    cells on as few lines as lets every unit reach its floor."""
    if item.minor:
        return _place_minor(b, model, item)
    box = item.cells[0]
    units = item.units
    x, y, w, h = box
    first = None
    gap = max(1.0, 0.015 * min(model.W, model.H))
    for lines in range(1, len(units) + 1):
        per_line = math.ceil(len(units) / lines)
        cw, ch = (w - gap * (per_line - 1)) / per_line, (h - gap * (lines - 1)) / lines
        if per_line > 1 or lines > 1:
            cells = [(x + (i % per_line) * (cw + gap), y + (i // per_line) * (ch + gap), cw, ch)
                     for i in range(len(units))]
        else:
            cells = [box]
        placed = [_place_unit(b, model, u, c, caps) for u, c in zip(units, cells)]
        first = first or placed
        if all(geo is not None and not degraded for _, geo, degraded in placed):
            return placed
    return first


def _place_minor(b, model: _Model, item: _Item) -> list[tuple[Unit, dict | None, bool]]:
    """Demoted units packed in shelves inside the minor cell, each held to its
    floor size; one that doesn't fit goes to the free-space pass."""
    x, y, w, h = item.cells[0]
    gap = 0.02 * min(model.W, model.H)
    packed = model.shelves(item.units, w, gap, item.zone_font)
    if not packed:
        return [(u, None, True) for u in item.units]
    used_w = max(bx + bw for _, (bx, _, bw, _) in packed)
    used_h = max(by + bh for _, (_, by, _, bh) in packed)
    ox, oy = x + max(0.0, (w - used_w) / 2), y + max(0.0, (h - used_h) / 2)
    out = []
    for unit, (bx, by, bw, bh) in packed:
        box = (ox + bx, oy + by, min(bw, w), bh)
        if by + bh > h + 0.5:
            out.append((unit, None, True))
            continue
        floor = model.zone_floor(unit, item.zone_font)
        geo = place_unit(unit, model.els, box, model.m, floor, align="center", max_font=floor)
        out.append((unit, geo, geo is None))
    return out


def _place_unit(b, model: _Model, unit: Unit, box: Box, caps: dict) -> tuple[Unit, dict | None, bool]:
    members = [model.els[i] for i in unit.element_ids]
    texts = [m for m in members if m["type"] == "text"]
    floor = model.floor(unit)
    max_font = model.text_cap(unit, caps.get("headline")) if texts else None
    if box[2] < 1 or box[3] < 1:  # a cut that left this unit no room: the free-space pass takes it
        return unit, None, True
    if unit.role == "logo":
        box = b._cap_logo(unit, box)
        uw, uh = union_box([rotated_aabb(m) for m in members])[2:]
        k = model.logo_cap(uw, uh)
        w, h = min(box[2], uw * k), min(box[3], uh * k)
        box = (*align_in(box, w, h), w, h)
    geo = place_unit(unit, model.els, box, model.m, floor, align="center", max_font=max_font)
    degraded = False
    if geo is None:
        hard_cap = None if max_font is None else max(max_font, HARD_FLOOR)
        geo = place_unit(unit, model.els, box, model.m, HARD_FLOOR, align="center", max_font=hard_cap)
        degraded = True
    if geo is not None and unit.role == "logo":
        el = geo[unit.element_ids[0]]
        if el["height"] < logo_min_height(model.W, model.H) - 0.05:
            degraded = True
    return unit, geo, degraded


def _build(source: dict, target: dict, plan: PriorityPlan, measurer, model: _Model, items: list[_Item],
           params: dict, minor: list[Unit] = ()):
    """One layout for the current item values; `minor` units skip the cuts and
    go to the engine's free-space pass at their floor size. Returns
    (Candidate | None, missed item keys, score)."""
    from engine.pipeline.layout.generate import _Builder  # the engine's own placement bookkeeping

    W, H = model.W, model.H
    short = min(W, H)
    m = params["margin"] * short
    content = (m, m, W - 2 * m, H - 2 * m)
    main = "x" if W > H else "y"
    _layout(model, items, content, params["order"], params["dir"], main, GUTTER * short)

    b = _Builder(source, target, plan, measurer)
    caps: dict = {}
    missed: set[int] = set()
    by_rank = sorted(items, key=lambda i: (i.rank, i.key))
    for item in by_rank:
        for unit, geo, degraded in _place_item(b, model, item, caps):
            if geo is None:
                b.leftover.append(unit)
                missed.add(item.key)
                continue
            b._record(unit, geo, degraded)
            if degraded:
                missed.add(item.key)
            if unit.role == "headline" and not degraded and "headline" not in caps:
                sizes = [g["font_size"] for g in geo.values() if g.get("type") == "text"]
                if sizes:
                    caps["headline"] = max(sizes)
                    model.last_head = caps["headline"]
    b.leftover.extend(minor)
    if b.leftover and not b.place_leftovers(m):
        return None, missed, -math.inf
    cid = f"{NAME}/" + "/".join(f"{k}={v}" for k, v in sorted(params.items()))
    cand = b.finish(cid, NAME, dict(params))
    if cand is None:
        return None, missed, -math.inf
    return cand, missed, _score(cand, plan, model)


def _score(cand: Candidate, plan: PriorityPlan, model: _Model) -> float:
    """Internal pick between repair rounds: priority-weighted readability
    (text at or above its floor, more up to comfort; degraded units earn
    nothing) minus the largest empty rectangle's share of the canvas."""
    els = {e["id"]: e for e in cand.design["elements"]}
    total, got, boxes = 0.0, 0.0, []
    for unit in plan.ranked_units():
        w = plan.weight(unit.unit_id)
        boxes.append(union_box([rotated_aabb(els[i]) for i in unit.element_ids]))
        sizes = [els[i]["font_size"] for i in unit.element_ids if els[i]["type"] == "text"]
        total += w
        if unit.unit_id in cand.degraded_unit_ids:
            continue
        if not sizes:
            got += w
            continue
        floor = model.floor(unit)
        if min(sizes) >= floor:
            got += w * min(1.0, 0.5 + 0.5 * (min(sizes) - floor) / (0.6 * floor))
    legible = got / total if total else 0.0
    return legible - EMPTY_PENALTY * largest_empty_share(boxes, model.W, model.H)


def largest_empty_share(boxes: list[Box], width: float, height: float, grid: int = 48) -> float:
    """Largest axis-aligned rectangle touching no box, as a share of the canvas (coarse grid)."""
    cell = max(width, height) / grid
    cols, rows = max(1, math.ceil(width / cell)), max(1, math.ceil(height / cell))
    busy = [[False] * cols for _ in range(rows)]
    for x, y, w, h in boxes:
        for r in range(max(0, int(y // cell)), min(rows, math.ceil((y + h) / cell))):
            row = busy[r]
            for c in range(max(0, int(x // cell)), min(cols, math.ceil((x + w) / cell))):
                row[c] = True
    best, heights = 0, [0] * cols
    for r in range(rows):
        heights = [0 if busy[r][c] else heights[c] + 1 for c in range(cols)]
        stack: list[int] = []
        for c in range(cols + 1):
            hc = heights[c] if c < cols else 0
            while stack and heights[stack[-1]] >= hc:
                top = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                best = max(best, top * (c - left))
            stack.append(c)
    return min(1.0, best * cell * cell / (width * height))


def _variant(source, target, plan, measurer, model: _Model, params: dict) -> Candidate | None:
    """Repair rounds on the item values; on a crowded canvas the least
    important items are first sent to the free-space pass, one more each time
    a layout can't be built at all."""
    all_items = _items(model, plan)
    if not all_items:
        return None
    m = params["margin"] * min(model.W, model.H)
    content_area = (model.W - 2 * m) * (model.H - 2 * m)
    demote = _demotion_order(all_items)
    start = _crowd_count(model, all_items, demote, max(content_area, 1.0))
    best, best_score = None, -math.inf
    for count in range(start, len(demote) + 1):
        out = {id(i) for i in demote[:count]}
        kept = [i for i in all_items if id(i) not in out]
        demoted = sorted((u for i in demote[:count] for u in i.units), key=lambda u: u.rank)
        if not kept:
            break
        for zone_font in ZONE_FONTS if demoted else (None,):
            items = list(kept)
            if demoted:
                zone = _Item(demoted, -1, math.inf, 10 ** 6, 1.0, minor=True, zone_font=zone_font)
                zone.base_value = 0.3 * math.sqrt(sum(model.min_area(u) for u in demoted))
                items.append(zone)
            for item in items:
                item.boost = 1.0
            model.head_font = None
            main_ok = False
            for _ in range(REPAIR_ROUNDS):
                model.last_head = None
                try:
                    cand, missed, score = _build(source, target, plan, measurer, model, items, params)
                except Exception:
                    cand, missed, score = None, set(), -math.inf
                if cand is not None and score >= best_score - 1e-9:  # ties go to the later, better-informed round
                    best, best_score = cand, score
                if cand is not None and not (missed - {-1}):
                    main_ok = True
                head = model.last_head
                refine = bool(head) and (model.head_font is None or abs(head - model.head_font) > 0.05 * head)
                if refine:  # size the other texts' cells for the headline this round reached
                    model.head_font = head
                if not missed and not refine:
                    break
                for item in items:
                    if item.key in missed:
                        item.boost *= BOOST
            if main_ok:
                return best
    return best


def _demotion_order(items: list[_Item]) -> list[_Item]:
    """Who leaves the cuts first on a crowded canvas: least important first;
    the hero image, logo and CTA only after everything else (the headline never)."""
    protect = {"hero_image": 1, "logo": 2, "cta": 3}
    pool = [i for i in items if i.role != "headline"]
    return sorted(pool, key=lambda i: (protect.get(i.role, 0), -i.rank, -i.key))


def _crowd_count(model: _Model, items: list[_Item], demote: list[_Item], area: float) -> int:
    """How many items to demote before the rest fit at their floors with room to spare."""
    need = {id(i): sum(model.min_area(u) for u in i.units) for i in items}
    total = sum(need.values())
    count = 0
    for item in demote:
        if total * CROWD_SLACK <= area or item.role in ("hero_image", "logo", "cta"):
            break
        total -= need[id(item)]
        count += 1
    return count


def candidates(source: dict, target: dict, plan: PriorityPlan, measurer) -> list[Candidate]:
    """Bisection candidates for this target (ids `bisection/...`), each passing
    the hard gates. [] when the target has the source's aspect (the reflow is
    the faithful answer there) or anything goes wrong."""
    try:
        return _candidates(source, target, plan, measurer)
    except Exception:
        return []


def _candidates(source, target, plan, measurer) -> list[Candidate]:
    from engine.pipeline.grading.gates import check_gates

    if not source or not plan or measurer is None or not plan.ranked_units():
        return []
    src_w, src_h = source["canvas"]["width"], source["canvas"]["height"]
    W, H = target["width"], target["height"]
    if min(src_w, src_h, W, H) <= 0:
        return []
    if abs((W / H) / (src_w / src_h) - 1) <= SAME_ASPECT_TOL:
        return []
    model = _Model(source, target, plan, measurer)
    out, seen = [], set()
    for margin in MARGINS:
        for order in ORDERS:
            for direction in DIRECTIONS:
                params = {"margin": margin, "order": order, "dir": direction}
                try:
                    cand = _variant(source, target, plan, measurer, model, params)
                    if cand is None:
                        continue
                    if check_gates(source, cand.design, target, plan, measurer):
                        continue
                    digest = json.dumps(cand.design, sort_keys=True)
                except Exception:
                    continue
                if digest in seen:
                    continue
                seen.add(digest)
                out.append(cand)
    return out
