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

Bisection v2 (CHAMELEON_BISECT_V2=1; off by default, and with it off the
output is unchanged) fixes the flaws the teacher's critiques name most, all
inside the layout rather than as a penalty afterwards (detectors and their
validation: dev/distill/bisect_flaws.py):
- a safe margin: at least V2_MARGINS of the short side, and at least
  V2_SIDE_MARGIN of the canvas width (height) on the left/right (top/bottom),
  so a wide banner's copy doesn't run into its ends;
- a logo floor tied to how the canvas is seen: the logo's sqrt(area) is at
  least LOGO_MIN_LONG of the long side (it may grow past 1.5x its source
  size to get there), so it's never a speck on a wide strip;
- legal stays fine print: it needs a cell wide enough to sit on at most
  LEGAL_MAX_LINES lines, and a narrow column counts as cramped for it;
- no widow: a multi-line text whose last line is one short word gets a
  balanced (narrower) measure, or failing that a slightly smaller size;
- shared alignment: units stacked in one column are shifted inside their own
  cells onto one axis (their texts' alignment: left, centre or right);
- decorations kept whole: a round shape behind text keeps its proportions
  (never squashed into an oval); a decoration that framed a content unit in
  the source follows that unit (same offset and scale) instead of being
  parked as a fragment; one that can't is parked only where no copy ends up
  half on it (each corner, then smaller), else the v1 way.

Hierarchy lock (CHAMELEON_HIER_LOCK=1; off by default, output unchanged
when off) keeps the source's text hierarchy (docs/prior-work.md, change 1):
- other text keeps at most its source size ratio to the headline
  (LOCK_RATIO_SLACK) and at most LOCK_HIERARCHY_CAP of the placed headline,
  with tighter caps by role (ROLE_CAPS: CTA label 0.6, badge 0.5, legal
  0.45); floors still win over caps;
- a CTA or badge pill grows at most PILL_GROWTH x the headline's scale (both
  relative to the source); past that its label is set in a compact pill;
- where a cap would sit under a role's floor (strips: CTA floor 11 px vs a
  14 px headline), the headline is raised instead (its floor becomes
  floor / cap of the subordinate roles present, legal excepted), if the
  layout still gives every main item its floor that way; otherwise the
  headline keeps its own floor.

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
import os
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

# -- v2 (CHAMELEON_BISECT_V2=1) --------------------------------------------------
V2_ENV = "CHAMELEON_BISECT_V2"
V2_MARGINS = (0.06, 0.09)  # share of the short side ...
V2_SIDE_MARGIN = 0.03      # ... and at least this share of the width (height) at the sides (top/bottom)
LOGO_MIN_LONG = 0.065      # logo sqrt(area) at least this share of the long side ...
LOGO_MIN_CAP = 0.5         # ... capped at this share of the short side (logo height)
LEGAL_MAX_LINES = 3        # legal wants a cell it fits on this many lines
WIDOW_SHARE = 0.4          # a last line of one word narrower than this x the longest line is a widow
WIDOW_SHRINK = 0.88        # a widow may cost at most this much of the font size
FRAME_SCALES = (1.0, 0.9, 0.8, 0.7, 0.6, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5)  # frame sizes tried, x its mapped size
FRAME_SHARE = 0.3          # a decoration covering this share of a content unit (or vice versa) frames it
ALIGN_EPS = 0.5            # px: anchors this close already share an axis
ROUND_SHAPES = ("ellipse", "circle")

# -- hierarchy lock (CHAMELEON_HIER_LOCK=1) ----------------------------------------
LOCK_ENV = "CHAMELEON_HIER_LOCK"
LOCK_HIERARCHY_CAP = 0.75  # no other text past this share of the headline's size ...
LOCK_RATIO_SLACK = 1.0     # ... nor past its source size ratio to the headline
ROLE_CAPS = {"cta": 0.6, "subheadline": 0.75, "badge": 0.5, "legal": 0.45}  # x the placed headline size
PILL_GROWTH = 1.25         # a CTA/badge pill's scale at most this x the headline's (both vs the source)
PILL_ROLES = ("cta", "badge")
LIFT_EXEMPT = ("legal",)   # fine print keeps its floor without raising the headline


def v2_enabled() -> bool:
    return os.environ.get(V2_ENV) == "1"


def lock_enabled() -> bool:
    return os.environ.get(LOCK_ENV) == "1"


def margins(params: dict, width: float, height: float, v2: bool) -> tuple[float, float]:
    """(left/right, top/bottom) margin in px."""
    m = params["margin"] * min(width, height)
    if not v2:
        return m, m
    return max(m, V2_SIDE_MARGIN * width), max(m, V2_SIDE_MARGIN * height)


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
        self.v2 = v2_enabled()
        self.lock = lock_enabled()
        self.lifted = False      # (lock) the headline's floor is raised to `lift` for this run
        self.lift = self._lift(plan) if self.lock else None
        self.layers = None       # (CHAMELEON_LAYERS) the source's layer groups (layout/layer_groups.py)

    def _lift(self, plan: PriorityPlan) -> float | None:
        """(lock) The headline size at which every subordinate text role
        present can sit at its floor without passing its cap; None if that
        is no more than the headline's own floor."""
        head = next((u for u in plan.ranked_units() if u.role == "headline"), None)
        if head is None or not self.head_src:
            return None
        base = floor_px("headline", self.W, self.H)
        need = base
        for u in plan.ranked_units():
            if u is head or u.role in LIFT_EXEMPT:
                continue
            if any(self.els[i]["type"] == "text" for i in u.element_ids):
                need = max(need, floor_px(u.role, self.W, self.H) / ROLE_CAPS.get(u.role, LOCK_HIERARCHY_CAP))
        need = min(need, self.head_src * MAX_TEXT_GROWTH)
        return need if need > base + 1e-6 else None

    def pill_cap(self, unit: Unit, head: float | None = None) -> float:
        """(lock) Largest scale for a CTA/badge pill: PILL_GROWTH x the
        headline's scale over its source size, once that is known."""
        head = self.head_font if head is None else head
        if not self.lock or unit.role not in PILL_ROLES or unit.kind != "container_label" or not head or not self.head_src:
            return math.inf
        return PILL_GROWTH * head / self.head_src

    # -- logo size -----------------------------------------------------------
    def logo_min(self, uw: float, uh: float) -> float:
        """Least logo height: the v1 floor, and in v2 also a sqrt(area) of
        LOGO_MIN_LONG x the long side (how big it looks), capped."""
        base = logo_min_height(self.W, self.H)
        if not self.v2 or uw <= 0 or uh <= 0:
            return base
        want = LOGO_MIN_LONG * max(self.W, self.H) / math.sqrt(uw / uh)
        return max(base, min(want, LOGO_MIN_CAP * min(self.W, self.H), 0.9 * self.W * uh / uw))

    def logo_growth(self, uw: float, uh: float) -> float:
        """Largest growth over the source size: 1.5x, or (v2) what the floor needs."""
        if not self.v2 or uh <= 0:
            return MAX_TEXT_GROWTH
        return max(MAX_TEXT_GROWTH, self.logo_min(uw, uh) / uh)

    # -- text helpers ------------------------------------------------------
    def floor(self, unit: Unit) -> float:
        if self.lifted and unit.role == "headline":
            return max(floor_px(unit.role, self.W, self.H), self.lift)
        return floor_px(unit.role, self.W, self.H)

    def base_floor(self, unit: Unit) -> float:
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
            if self.lock:
                cap = min(cap, head * min(ROLE_CAPS.get(unit.role, LOCK_HIERARCHY_CAP),
                                          LOCK_RATIO_SLACK * big / self.head_src))
            else:
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

    def legal_lines(self, el: dict, f: float) -> int:
        """(v2) Lines legal text may take: LEGAL_MAX_LINES, or as many as it
        needs across 90% of the canvas width if that is more."""
        key = ("legal_lines", el["id"], f)
        if key not in self._cache:
            n = self.m.line_count(resize_text(el, f), f, 0.9 * self.W)
            self._cache[key] = max(LEGAL_MAX_LINES, n)
        return self._cache[key]

    def _legal_width(self, el: dict, f: float, cross: float, lo: float) -> float:
        """(v2) Least width at which legal text sits on at most `legal_lines`
        lines at size f, within `cross` height; `lo` if it can't."""
        t = resize_text(el, f)
        hi = self._line(el, f)[0] + 1
        n = self.legal_lines(el, f)
        if hi <= lo or self.m.line_count(t, f, lo) <= n:
            return lo
        a, b = lo, hi
        for _ in range(_SEARCH_STEPS):
            mid = (a + b) / 2
            if self.m.line_count(t, f, mid) <= n:
                b = mid
            else:
                a = mid
        return b if self._fits(el, f, b, cross) else lo

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
        key = ("unit", unit.unit_id, axis, round(cross, 1), self.head_font, self.lifted)
        if key not in self._cache:
            try:
                ext = self._unit_ext(unit, axis, cross, self.floor(unit))
                if not math.isfinite(ext[0]) and self.floor(unit) > self.base_floor(unit):  # (lock) unlifted
                    ext = self._unit_ext(unit, axis, cross, self.base_floor(unit))
                if not math.isfinite(ext[0]):  # can't reach its floor here: size it at the hard floor
                    ext = self._unit_ext(unit, axis, cross, HARD_FLOOR)
                if not math.isfinite(ext[0]):  # can't be placed this way at all
                    ext = (math.inf, math.inf)
            except Exception:
                ext = (math.inf, math.inf)
            self._cache[key] = ext
        return self._cache[key]

    def _unit_ext(self, unit: Unit, axis: str, cross: float, f0: float) -> tuple[float, float]:
        if self.layers is not None and unit.kind == "layer_group":  # (layers) a container and its contents
            from engine.pipeline.layout.layer_groups import group_ext
            return group_ext(self, unit, axis, cross, f0)
        members = [self.els[i] for i in unit.element_ids]
        texts = [m for m in members if m["type"] == "text"]
        if len(members) == 1 and texts:
            el = texts[0]
            lo = self._text_ext(el, f0, axis, cross)
            if not math.isfinite(lo):
                return lo, lo
            if self.v2 and unit.role == "legal" and axis == "x":
                lo = self._legal_width(el, f0, cross, lo)
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
                k_max = min(k_max, self.logo_growth(uw, uh), self.logo_cap(uw, uh))
                k_min = min(self.logo_min(uw, uh) / uh, k_max)
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
            k_lo = f0 / big * 1.05
            # (lock) the pill cap never asks for less than the label's floor needs
            k_hi = min(k_hi, MAX_TEXT_GROWTH, self.text_cap(unit) / big, max(self.pill_cap(unit), k_lo))
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
        key = ("item", item.key, axis, round(cross, 1), self.head_font, self.lifted)
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
            h = max(h, self.logo_min(uw, uh))
        return h * uw / uh if uh else h, h

    def logo_cap(self, uw: float, uh: float) -> float:
        """Largest scale for a logo: a modest mark, not a hero."""
        short = min(self.W, self.H)
        h_max = max(self.logo_min(uw, uh), LOGO_MAX_SHORT * short)
        k = min(h_max / uh, LOGO_MAX_WIDTH * self.W / uw)
        if self.v2:  # the floor wins over the width cap
            k = max(k, self.logo_min(uw, uh) / uh)
        return k

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
        f0 = self.base_floor(unit)
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
            h = self.logo_min(uw, uh)
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
            room = (size or 0.0) / floor
            if model.v2 and unit.role == "legal" and size:  # a tall narrow column is cramped for fine print
                lines = model.m.line_count(resize_text(el, size), size, box[2])
                room *= min(1.0, model.legal_lines(el, size) / max(lines, 1))
            return room
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
    if model.layers is not None and unit.kind == "layer_group":  # (layers) contents laid out inside the container
        from engine.pipeline.layout.layer_groups import place_group
        return place_group(b, model, unit, box, caps)
    members = [model.els[i] for i in unit.element_ids]
    texts = [m for m in members if m["type"] == "text"]
    floor = model.floor(unit)
    max_font = model.text_cap(unit, caps.get("headline")) if texts else None
    if box[2] < 1 or box[3] < 1:  # a cut that left this unit no room: the free-space pass takes it
        return unit, None, True
    if unit.role == "logo":
        uw, uh = union_box([rotated_aabb(m) for m in members])[2:]
        if model.v2:  # the source-size cap gives way to the v2 floor
            g = model.logo_growth(uw, uh)
            box = (*align_in(box, min(box[2], uw * g), min(box[3], uh * g)), min(box[2], uw * g), min(box[3], uh * g))
        else:
            box = b._cap_logo(unit, box)
        k = model.logo_cap(uw, uh)
        w, h = min(box[2], uw * k), min(box[3], uh * k)
        box = (*align_in(box, w, h), w, h)
    if texts and model.pill_cap(unit, caps.get("headline")) < math.inf:  # (lock) the pill stays modest
        geo = _place_pill(model, unit, members, box, floor, max_font, model.pill_cap(unit, caps.get("headline")))
    else:
        geo = place_unit(unit, model.els, box, model.m, floor, align="center", max_font=max_font)
    if geo is None and floor > model.base_floor(unit):  # (lock) the raised headline doesn't fit: its own floor
        floor = model.base_floor(unit)
        geo = place_unit(unit, model.els, box, model.m, floor, align="center", max_font=max_font)
    degraded = False
    if geo is None:
        hard_cap = None if max_font is None else max(max_font, HARD_FLOOR)
        geo = place_unit(unit, model.els, box, model.m, HARD_FLOOR, align="center", max_font=hard_cap)
        degraded = True
    if geo is not None and unit.role == "logo":
        el = geo[unit.element_ids[0]]
        uw, uh = union_box([rotated_aabb(m) for m in members])[2:]
        if el["height"] < model.logo_min(uw, uh) - 0.05:
            degraded = True
    if model.v2 and geo is not None and len(members) == 1 and texts and unit.role != "legal":
        geo = {texts[0]["id"]: _no_widow(model, geo[texts[0]["id"]], box, HARD_FLOOR if degraded else floor)}
    return unit, geo, degraded


def _place_pill(model: _Model, unit: Unit, members: list[dict], box: Box, floor: float,
                max_font: float | None, k_cap: float) -> dict | None:
    """(lock) A CTA/badge pill at most k_cap x its source size: rigid inside
    a box that size; if its label can't reach the floor that way, a compact
    pill round the label (in the full cell); failing both, the usual
    placement."""
    from engine.pipeline.layout.place import _place_condensed
    uw, uh = union_box([rotated_aabb(m) for m in members])[2:]
    w, h = min(box[2], uw * k_cap), min(box[3], uh * k_cap)
    geo = place_unit(unit, model.els, (*align_in(box, w, h), w, h), model.m, floor, align="center", max_font=max_font)
    if geo is None and unit.kind == "container_label" and any(m["type"] == "shape" for m in members):
        geo = _place_condensed(members, box, model.m, floor, "center", max_font)
    if geo is None:
        geo = place_unit(unit, model.els, box, model.m, floor, align="center", max_font=max_font)
    return geo


# -- v2: widows ----------------------------------------------------------------

def _has_widow(m, el: dict, f: float, w: float) -> bool:
    lines = m.wrap(el, f, w).split("\n")
    if len(lines) < 2 or len(lines[-1].split()) != 1:
        return False
    widths = m.line_widths(el, f, w)
    return bool(widths) and widths[-1] < WIDOW_SHARE * max(widths)


def _balanced_width(m, el: dict, f: float, w: float) -> float:
    """Narrowest width that keeps the line count at w (lines evened out)."""
    n = m.line_count(el, f, w)
    lo, hi = m.longest_word_width(el, f), w
    if lo >= hi:
        return w
    for _ in range(_SEARCH_STEPS):
        mid = (lo + hi) / 2
        if m.line_count(el, f, mid) == n and max(m.line_widths(el, f, mid), default=0.0) <= mid + 0.5:
            hi = mid
        else:
            lo = mid
    return hi


def _no_widow(model: _Model, el: dict, box: Box, floor: float) -> dict:
    """`el` as placed, unless its last line is one short word: then a
    balanced measure (same size and line count, narrower box aligned the way
    the text is), else the largest slightly smaller size that has no widow."""
    m = model.m
    f, w = el["font_size"], el["width"]
    if el.get("rotation") or not _has_widow(m, el, f, w):
        return el

    def placed(text: dict, size: float, width: float) -> dict:
        block = m.block_height(text, size, width)
        align = text.get("text_align") or "left"
        x = el["x"] + ((w - width) / 2 if align == "center" else (w - width) if align == "right" else 0.0)
        return dict(text, x=x, width=width, y=box[1] + (box[3] - block) / 2, height=block)

    size = f
    while size >= max(floor, f * WIDOW_SHRINK):
        text = resize_text(el, size)
        bw = _balanced_width(m, text, size, w)
        if not _has_widow(m, text, size, bw) and m.fits(text, size, bw, box[3]):
            return placed(text, size, bw)
        if not _has_widow(m, text, size, w) and m.fits(text, size, w, box[3]):
            return placed(text, size, w)
        size -= 0.5
    return el


# -- v2: one axis per stack -------------------------------------------------------

def _tighten(m, g: dict) -> dict:
    """A placed text's box shrunk to its widest line (same wrap, same ink position)."""
    if g.get("type") != "text" or g.get("rotation"):
        return g
    f, w = g["font_size"], g["width"]
    nw = max(m.line_widths(g, f, w), default=0.0) + 1.0
    if nw >= w - 0.5 or m.wrap(g, f, nw) != m.wrap(g, f, w) or not m.fits(g, f, nw, g["height"]):
        return g
    align = g.get("text_align") or "left"
    x = g["x"] + ((w - nw) / 2 if align == "center" else (w - nw) if align == "right" else 0.0)
    return dict(g, x=x, width=nw)


def _stacks(boxes: list[Box]) -> list[list[int]]:
    """Groups of boxes one above the other whose x-ranges overlap by at least
    half the narrower (the same rule as dev/distill/bisect_flaws.py)."""
    n = len(boxes)
    adj: dict[int, set] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            ov = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
            vert = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
            if ov >= 0.5 * min(a[2], b[2]) and vert > -0.25 * min(a[3], b[3]):
                adj[i].add(j)
                adj[j].add(i)
    seen, out = set(), []
    for i in range(n):
        if i in seen:
            continue
        comp, todo = [], [i]
        while todo:
            k = todo.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.append(k)
            todo += sorted(adj[k] - seen)
        out.append(sorted(comp))
    return out


def columns(boxes: list[Box], comp: list[int], cap: int = 64) -> list[list[int]]:
    """The ways to read a stack as one column: members side by side (their
    vertical ranges overlap) form a row, and a column takes one member per
    row (a headline over a copy-and-button row gives two columns)."""
    rows: list[list[int]] = []
    for i in sorted(comp, key=lambda k: (boxes[k][1], boxes[k][0])):
        b = boxes[i]
        for row in rows:
            if any(min(b[1] + b[3], boxes[j][1] + boxes[j][3]) - max(b[1], boxes[j][1])
                   > 0.25 * min(b[3], boxes[j][3]) for j in row):
                row.append(i)
                break
        else:
            rows.append([i])
    out: list[list[int]] = [[]]
    for row in rows:
        out = [c + [i] for c in out for i in sorted(row, key=lambda k: boxes[k][0])][:cap]
    return out


def _align_stacks(model: _Model, placed: list) -> list:
    """Units stacked in one column share one axis: their texts' alignment
    (left, centre or right; centre when no text votes), else the next axis
    that every unit can reach. Each unit moves sideways only inside its own
    cell, so nothing new overlaps; runs and the minor zone stay put."""
    out = []
    for item, unit, geo, degraded in placed:
        if not item.minor and len(item.units) == 1 and len(geo) == 1:
            geo = {eid: _tighten(model.m, g) for eid, g in geo.items()}
        out.append([item, unit, geo, degraded])
    rows = [r for r in out if not r[0].minor]
    boxes = [union_box([rotated_aabb(g) for g in r[2].values()]) for r in rows]
    rng = {}
    for i, (item, bx) in enumerate(zip([r[0] for r in rows], boxes)):
        if len(item.units) == 1 and item.cells:
            cx, _, cw, _ = item.cells[0]
            lo, hi = cx - bx[0], cx + cw - bx[0] - bx[2]
            rng[i] = (min(0.0, lo) if lo <= 0 else 0.0, max(0.0, hi) if hi >= 0 else 0.0)
        else:
            rng[i] = (0.0, 0.0)
    for comp in _stacks(boxes):
        if len(comp) < 2:
            continue
        votes = {"left": 0, "center": 0, "right": 0}
        for i in comp:
            for g in rows[i][2].values():
                if g.get("type") == "text":
                    votes[g.get("text_align") if g.get("text_align") in votes else "left"] += 1
        pref = max(votes, key=lambda k: (votes[k], k == "center")) if any(votes.values()) else "center"
        best = None  # (axis rank, total shift, column, target, anchors)
        for rank, axis in enumerate([pref] + [a for a in ("center", "left", "right") if a != pref]):
            at = {"left": 0.0, "center": 0.5, "right": 1.0}[axis]
            for col in columns(boxes, comp):
                if len(col) < 2:
                    continue
                anchors = [boxes[i][0] + at * boxes[i][2] for i in col]
                lo = max(a + rng[i][0] for a, i in zip(anchors, col))
                hi = min(a + rng[i][1] for a, i in zip(anchors, col))
                if lo > hi + 1e-6:
                    continue
                target = min(max(sorted(anchors)[len(anchors) // 2], lo), hi)
                key = (rank, -len(col), sum(abs(target - a) for a in anchors))
                if best is None or key < best[0]:
                    best = (key, col, target, anchors)
            if best is not None:
                break
        if best is None:
            continue
        _, col, target, anchors = best
        for i, a in zip(col, anchors):
            dx = target - a
            if abs(dx) > ALIGN_EPS / 10:
                rows[i][2] = {eid: dict(g, x=g["x"] + dx) for eid, g in rows[i][2].items()}
    return [tuple(r) for r in out]


# -- v2: decorations kept whole ------------------------------------------------------

_V2_BUILDER = None


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


def _v2_builder():
    """The engine's _Builder with layers placed the v2 way (made on first use:
    generate.py is imported lazily, as in _build)."""
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


def _build(source: dict, target: dict, plan: PriorityPlan, measurer, model: _Model, items: list[_Item],
           params: dict, minor: list[Unit] = ()):
    """One layout for the current item values; `minor` units skip the cuts and
    go to the engine's free-space pass at their floor size. Returns
    (Candidate | None, missed item keys, score)."""
    from engine.pipeline.layout.generate import _Builder  # the engine's own placement bookkeeping

    W, H = model.W, model.H
    short = min(W, H)
    mx, my = margins(params, W, H, model.v2)
    m = min(mx, my)
    content = (mx, my, W - 2 * mx, H - 2 * my)
    main = "x" if W > H else "y"
    _layout(model, items, content, params["order"], params["dir"], main, GUTTER * short)

    b = (_v2_builder() if model.v2 else _Builder)(source, target, plan, measurer)
    caps: dict = {}
    missed: set[int] = set()
    by_rank = sorted(items, key=lambda i: (i.rank, i.key))
    placed = []
    for item in by_rank:
        for unit, geo, degraded in _place_item(b, model, item, caps):
            if geo is None:
                b.leftover.append(unit)
                missed.add(item.key)
                continue
            placed.append((item, unit, geo, degraded))
            if not model.v2:
                b._record(unit, geo, degraded)
            if degraded:
                missed.add(item.key)
            if unit.role == "headline" and not degraded and "headline" not in caps:
                sizes = [g["font_size"] for g in geo.values() if g.get("type") == "text"]
                if sizes:
                    caps["headline"] = max(sizes)
                    model.last_head = caps["headline"]
    if model.v2:
        for _, unit, geo, degraded in _align_stacks(model, placed):
            b._record(unit, geo, degraded)
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
        floor = model.base_floor(unit)
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
    """One candidate for `params`. (lock) Also with a raised headline; that
    one is kept only if it costs no other text its legibility (see `_no_loss`)."""
    plain = _variant_rounds(source, target, plan, measurer, model, params)[0]
    if model.lock and model.lift and (plain is None or _head_size(plan, plain) < model.lift - 1e-6):
        model.lifted = True
        try:
            cand, ok = _variant_rounds(source, target, plan, measurer, model, params)
        finally:
            model.lifted = False
        if cand is not None and ok and (plain is None or _no_loss(model, plan, cand, plain)):
            return cand
    return plain


def _head_size(plan: PriorityPlan, cand: Candidate) -> float:
    els = {e["id"]: e for e in cand.design["elements"]}
    return max((els[i]["font_size"] for u in plan.ranked_units() if u.role == "headline"
                for i in u.element_ids if els[i]["type"] == "text"), default=math.inf)


def _no_loss(model: _Model, plan: PriorityPlan, new: Candidate, old: Candidate) -> bool:
    """(lock) Is `new` (raised headline) at least as legible as `old` for every
    other text: nothing newly degraded, nothing newly under its floor, and
    nothing already under it any smaller? The headline must actually be bigger."""
    if set(new.degraded_unit_ids) - set(old.degraded_unit_ids):
        return False
    a = {e["id"]: e for e in new.design["elements"]}
    b = {e["id"]: e for e in old.design["elements"]}
    for unit in plan.ranked_units():
        sa = [a[i]["font_size"] for i in unit.element_ids if a[i]["type"] == "text"]
        sb = [b[i]["font_size"] for i in unit.element_ids if b[i]["type"] == "text"]
        if not sa:
            continue
        if unit.role == "headline":
            if max(sa) <= max(sb) + 1e-6:
                return False
        elif min(sa) < min(min(sb), model.base_floor(unit)) - 1e-6:
            return False
    return True


def _variant_rounds(source, target, plan, measurer, model: _Model, params: dict) -> tuple[Candidate | None, bool]:
    """Repair rounds on the item values; on a crowded canvas the least
    important items are first sent to the free-space pass, one more each time
    a layout can't be built at all. Returns (best, every main item placed at its floor)."""
    all_items = _items(model, plan)
    if not all_items:
        return None, False
    mx, my = margins(params, model.W, model.H, model.v2)
    content_area = (model.W - 2 * mx) * (model.H - 2 * my)
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
                return best, True
    return best, False


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
    layers, gate_plan = None, plan
    if os.environ.get("CHAMELEON_LAYERS") == "1":  # (layers) a container and its contents move as one unit
        from engine.pipeline.layout import layer_groups
        layers = layer_groups.detect(source, plan, target)
        plan = layers.plan  # the layout plan; the gates and the grader keep the real one
    model = _Model(source, target, plan, measurer)
    model.layers = layers
    out, seen = [], set()
    for margin in (V2_MARGINS if model.v2 else MARGINS):
        for order in ORDERS:
            for direction in DIRECTIONS:
                params = {"margin": margin, "order": order, "dir": direction}
                try:
                    cand = _variant(source, target, plan, measurer, model, params)
                    if cand is None:
                        continue
                    if layers is not None:
                        cand.degraded_unit_ids = layer_groups.remap_degraded(layers, cand.degraded_unit_ids,
                                                                             cand.design, W, H)
                    if check_gates(source, cand.design, target, gate_plan, measurer):
                        continue
                    digest = json.dumps(cand.design, sort_keys=True)
                except Exception:
                    continue
                if digest in seen:
                    continue
                seen.add(digest)
                out.append(cand)
    return out
