"""Content-aware slot sizing ("flex"), switched on with CHAMELEON_FLEX=1.

Templates cut the content box into fixed fractions (templates.py). With flex
on, those boxes only give the template's *structure* and its intent; the
sizes along each cut come from what the assigned units need:

1. the slots are decomposed into nested rows and columns (guillotine cuts:
   straight lines that cross no slot), which works for the built-in templates
   and for most floor plans; a plan that can't be cut keeps its fixed boxes;
2. along each cut, every child gets a minimum (its units at their readability
   floor, pills in their compact form; images at half their template share; a
   logo's visible mark at LOGO_MIN_AREA of the canvas, measured on the asset's
   opaque pixels), a preferred size (text at COMFORT x floor, images at their
   template share) and a cap (the lead text at its largest size, other text at
   SUPPORT_CAP x floor so the hierarchy survives, pills at source proportions,
   images filling the cross extent but at most IMAGE_MAX_SHARE of the canvas);
   slots with no units collapse to nothing;
3. space goes to minimums, then to preferred sizes, then to caps, each shared
   by priority weight; whatever is still left widens the gaps evenly, so the
   group stays centred instead of leaving one dead block. When even the
   minimums don't fit (a crowded banner), the logo keeps its minimum (up to
   LOGO_KEEP of the canvas) and the rest follows the template's own fractions;
4. children are separated by one gap size for the whole canvas; an image or
   logo may widen across its slot, up to its parent's extent, to use the
   length it was given.

Only slot geometry changes; assignment, placement, leftovers and layers run
as before. `flex_slots` never raises: on any problem it returns the fixed
slots unchanged.
"""

import math
import os
from dataclasses import dataclass, field, replace

from engine.pipeline.assets import default_assets_dir
from engine.pipeline.geometry import Box, rotated_aabb, union_box
from engine.pipeline.layout.place import (CONDENSED_PADDING, CONDENSED_VPAD, HARD_FLOOR, MAX_TEXT_GROWTH, floor_px,
                                          place_unit, resize_text)
from engine.pipeline.layout.templates import Slot
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import PriorityPlan, Unit
from renderer.converter.assets import asset_path

ENV = "CHAMELEON_FLEX"
COMFORT = 1.6            # preferred text size, x the role's floor (the grader's legibility COMFORT) ...
PREF_SLACK = 1.06        # ... with a little to spare, as fitting rounds sizes down to 0.5 px steps
SUPPORT_CAP = 2.0        # text other than the lead grows to at most this x its floor
GAP_SHORT = 0.025        # gap between slots: this share of the canvas's short side ...
GAP_LONG = 0.012         # ... or this share of its long side, whichever is larger
STACK_GAP = 0.5          # least space between stacked text units, in gaps
IMAGE_MIN = 0.5          # an image may shrink to this share of its template size to make room
IMAGE_MAX_SHARE = 0.6    # an image never takes more than this share of the canvas along a cut
LOGO_MIN_AREA = 0.015    # a logo's least visible size: its mark covers this share of the canvas area ...
LOGO_MIN_WIDTH = 0.15    # ... and is at least this share of the canvas's short side wide
LOGO_PREF = 1.25         # preferred logo size, x its minimum
LOGO_MAX_GROWTH = 1.5    # same cap generate.py applies when placing a logo
LOGO_MAX_WIDTH = 0.35    # a logo asks for at most this share of the canvas width (past its minimum)
LOGO_KEEP = 0.18         # in a crowded cut a logo still keeps its minimum, up to this share of the canvas
PILL_VPAD = 0.45         # a pill's preferred room above and below its label, in label font sizes
MINOR_IMAGE_SHARE = 0.06  # an image in the minor zone asks for at most this share of the canvas
FLOW_GAP = 0.02          # generate.py's gap between minor units, share of the short side
CUT_TOL = 0.005          # slots may overlap this much (content-box fraction) and still be cut apart
SEARCH_STEPS = 14        # binary-search steps when looking for a width
EPS = 1e-6


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


# --- the 1-D solver ---------------------------------------------------------

@dataclass
class Need:
    """What one child wants along a cut: at least `min`, ideally `pref`, at most `cap`
    (a hint, not a limit: leftover past every cap widens the gaps). `weight` shares
    out space; an `empty` child gets nothing, not even a gap. When the minimums don't
    all fit, `keep` (a logo's least visible size) is honoured first and the rest is
    shared in proportion to the larger of `min` and `basis` (the template's size)."""

    min: float
    pref: float
    cap: float
    weight: float
    empty: bool = False
    basis: float = 0.0
    keep: float = 0.0


def _fill(sizes: list[float], targets: list[float], weights: list[float], room: float) -> float:
    """Grow sizes toward targets, sharing `room` by weight (water-filling); returns what's left."""
    while room > EPS:
        open_ = [i for i in range(len(sizes)) if sizes[i] < targets[i] - EPS]
        if not open_:
            break
        total = sum(weights[i] for i in open_)
        step = min((targets[i] - sizes[i]) * total / weights[i] for i in open_)
        step = min(step, room)
        for i in open_:
            sizes[i] = min(targets[i], sizes[i] + step * weights[i] / total)
        room -= step
    return max(room, 0.0)


def _squeeze(room: float, mins: list[float], needs: list[Need]) -> list[float]:
    """Sizes when the minimums don't fit (a crowded banner: something will be degraded
    or moved to the free-space pass whatever we do): kept sizes first, then the rest in
    proportion to the template's own sizes, which the engine's fallbacks are tuned for."""
    keep = [min(max(n.keep, 0.0), m) for n, m in zip(needs, mins)]
    if sum(keep) >= room:
        total = sum(keep)
        return [room * k / total for k in keep] if total > 0 else [room / len(mins)] * len(mins)
    share = [max(n.basis - k, 0.0) for n, k in zip(needs, keep)]
    if sum(share) <= 0:
        share = [max(m - k, 0.0) for m, k in zip(mins, keep)]
    total = sum(share)
    rest = room - sum(keep)
    if total <= 0:
        return [k + rest / len(mins) for k in keep]
    return [k + rest * w / total for k, w in zip(keep, share)]


def distribute(length: float, needs: list[Need], gap: float) -> list[tuple[float, float]]:
    """(offset, size) per child along a cut of `length`: minimums first (scaled down
    together if they don't fit), then preferred sizes, then caps, each by weight; any
    rest is spread evenly over the gaps and both ends. Empty children get size 0."""
    live = [i for i, n in enumerate(needs) if not n.empty]
    out = [(0.0, 0.0)] * len(needs)
    if not live:
        return out
    gaps = gap * (len(live) - 1)
    if gaps >= length:
        gap, gaps = 0.0, 0.0
    room = length - gaps
    mins = [max(0.0, needs[i].min) for i in live]
    prefs = [max(m, needs[i].pref) for m, i in zip(mins, live)]
    caps = [max(p, needs[i].cap) for p, i in zip(prefs, live)]
    weights = [max(needs[i].weight, 1e-3) for i in live]
    rest = 0.0
    if sum(mins) >= room:
        sizes = _squeeze(room, mins, [needs[i] for i in live])
    else:
        sizes = list(mins)
        rest = _fill(sizes, prefs, weights, room - sum(mins))
        rest = _fill(sizes, caps, weights, rest)
    extra = rest / (len(live) + 1)
    cursor = extra
    for i, size in zip(live, sizes):
        out[i] = (cursor, size)
        cursor += size + gap + extra
    # an empty child sits where the next live one starts, with no size
    for i in range(len(needs)):
        if needs[i].empty:
            nxt = next((out[j][0] for j in range(i + 1, len(needs)) if not needs[j].empty), cursor - gap - extra)
            out[i] = (nxt, 0.0)
    return out


# --- structure: guillotine decomposition -----------------------------------

@dataclass
class Node:
    """A slot (`index`) or a cut: `children` laid top to bottom (axis "v") or left to
    right ("h"). `box` is where the template drew it, in content-box fractions."""

    box: Box
    index: int | None = None
    axis: str | None = None
    children: list["Node"] = field(default_factory=list)


class _Uncuttable(Exception):
    pass


def _bbox(items) -> Box:
    return union_box([b for _, b in items])


def _groups(items, axis: str) -> list[list]:
    k = 1 if axis == "v" else 0
    groups, end = [], -math.inf
    for item in sorted(items, key=lambda it: (it[1][k], it[1][k + 2], it[0])):
        start, stop = item[1][k], item[1][k] + item[1][k + 2]
        if groups and start < end - CUT_TOL:
            groups[-1].append(item)
            end = max(end, stop)
        else:
            groups.append([item])
            end = stop
    return groups


def _decompose(items) -> Node:
    if len(items) == 1:
        return Node(items[0][1], index=items[0][0])
    best = None
    for axis in ("v", "h"):
        groups = _groups(items, axis)
        if len(groups) > 1 and (best is None or len(groups) > len(best[1])):
            best = (axis, groups)
    if best is None:
        raise _Uncuttable
    return Node(_bbox(items), axis=best[0], children=[_decompose(g) for g in best[1]])


def decompose(boxes: list[Box]) -> Node | None:
    """Nested rows/columns for slot boxes (content-box fractions), or None if some
    group of slots can't be separated by a straight cut. The root spans the unit square."""
    if not boxes:
        return None
    try:
        root = _decompose(list(enumerate(boxes)))
    except _Uncuttable:
        return None
    root.box = (0.0, 0.0, 1.0, 1.0)
    return root


def _along(box: Box, axis: str) -> tuple[float, float]:
    return (box[1], box[3]) if axis == "v" else (box[0], box[2])


def _frac(child: Node, parent: Node, axis: str) -> tuple[float, float]:
    """The child's start and length along `axis` as fractions of the parent's."""
    p0, pl = _along(parent.box, axis)
    c0, cl = _along(child.box, axis)
    if pl <= EPS:
        return 0.0, 1.0
    return (c0 - p0) / pl, cl / pl


def _other(axis: str) -> str:
    return "h" if axis == "v" else "v"


# --- what the content needs -------------------------------------------------

_ink_cache: dict[tuple, tuple[float, float]] = {}


def ink_share(el: dict, assets_dir: str | None = None) -> tuple[float, float]:
    """Share of an image element's box (width, height) its visible pixels span: a logo
    lockup drawn on a wide transparent canvas shows far less than its box. (1, 1) for
    anything that isn't an image whose asset can be read."""
    if el.get("type") != "image" or not el.get("width") or not el.get("height"):
        return 1.0, 1.0
    assets_dir = str(default_assets_dir() if assets_dir is None else assets_dir)  # resolved per call
    key = (assets_dir, el.get("asset_id"), round(el["width"] / el["height"], 4))
    if key not in _ink_cache:
        share = (1.0, 1.0)
        try:
            path = asset_path(assets_dir, el.get("asset_id"))
            if path is not None:
                from PIL import Image
                with Image.open(path) as img:
                    aw, ah = img.size
                    bbox = img.getchannel("A").getbbox() if "A" in img.getbands() else img.getbbox()
                if bbox and aw and ah:
                    scale = min(el["width"] / aw, el["height"] / ah)  # drawn "contain" in its box
                    share = (min(1.0, (bbox[2] - bbox[0]) * scale / el["width"]),
                             min(1.0, (bbox[3] - bbox[1]) * scale / el["height"]))
                    if share[0] <= 0 or share[1] <= 0:
                        share = (1.0, 1.0)
        except Exception:
            share = (1.0, 1.0)
        _ink_cache[key] = share
    return _ink_cache[key]


def logo_min_width(width: float, height: float, logo_w: float, logo_h: float,
                   ink: tuple[float, float] = (1.0, 1.0)) -> float:
    """Least box width for a logo whose box is logo_w x logo_h (source px) and whose
    visible mark spans `ink` of that box, on a width x height canvas: the mark covers
    LOGO_MIN_AREA of the canvas and LOGO_MIN_WIDTH of its short side, but the box never
    grows past LOGO_MAX_GROWTH x the source."""
    ink_w, ink_h = logo_w * ink[0], logo_h * ink[1]
    aspect = ink_w / ink_h if ink_h > 0 else 1.0
    want = max(LOGO_MIN_WIDTH * min(width, height), math.sqrt(LOGO_MIN_AREA * width * height * aspect))
    return min(want / max(ink[0], EPS), logo_w * LOGO_MAX_GROWTH)


class _Content:
    """Measures what each slot's units need along a cut, given the slot's cross extent."""

    def __init__(self, slots: list[Slot], assignment: dict, els: dict, plan: PriorityPlan,
                 measurer: TextMeasurer, width: float, height: float, content: Box):
        self.slots, self.els, self.plan, self.measurer = slots, els, plan, measurer
        self.width, self.height, self.content = width, height, content
        self.units = {i: [u for item in items for u in item] for i, items in assignment.items()}
        self.gap = max(GAP_SHORT * min(width, height), GAP_LONG * max(width, height))
        self.big = 10 * max(width, height)
        # Only the lead text (the best-ranked unit with plain text) grows past SUPPORT_CAP x
        # floor, so the source's hierarchy survives when every slot has room to spare.
        texts = [u for u in plan.ranked_units() if u.kind != "container_label"
                 and any(els.get(i, {}).get("type") == "text" for i in u.element_ids)]
        self.lead = texts[0].unit_id if texts else None

    def members(self, unit: Unit) -> list[dict]:
        return [self.els[i] for i in unit.element_ids]

    def floor(self, unit: Unit) -> float:
        return floor_px(unit.role, self.width, self.height)

    # one unit

    def _text_width(self, el: dict, size: float, height: float) -> float:
        """Least width at which `el` at `size` fits `height` (wrapping as needed)."""
        text = resize_text(el, size)
        m = self.measurer
        hi = max(m.line_widths(text, size, math.inf), default=0.0) + 1.0
        lo = max(m.longest_word_width(text, size), 1.0)
        if m.fits(text, size, lo, height):
            return lo
        if not m.fits(text, size, hi, height):
            return hi
        for _ in range(SEARCH_STEPS):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if m.fits(text, size, mid, height) else (mid, hi)
        return hi

    def extent(self, unit: Unit, axis: str, cross: float, font_cap: float, roomy: bool = False) -> float | None:
        """Length along `axis` the unit takes in a slot `cross` wide (or tall), with its
        text at most `font_cap` (never below its floor). A pill is measured in its compact
        (condensed) form when that is smaller, unless `roomy`: given room, it keeps the
        source's proportions. None if it can't be measured."""
        members = self.members(unit)
        if any(m.get("rotation") for m in members) or cross <= 0:
            return None
        floor = self.floor(unit)
        cap = max(font_cap, floor)
        if len(members) == 1 and members[0]["type"] == "text":
            el = members[0]
            size = min(cap, el["font_size"] * MAX_TEXT_GROWTH)
            if axis == "h":
                if roomy:  # given room, a CTA or badge reads on one or two lines, not a tall narrow column
                    text = resize_text(el, size)
                    line = size * float(el.get("line_height", 1.2) or 1.2)
                    cross = min(cross, line + self.measurer.block_height(text, size, math.inf))
                return self._text_width(el, size, cross)
        box = (0.0, 0.0, cross, self.big) if axis == "v" else (0.0, 0.0, self.big, cross)
        geometry = None
        for lo in (floor, HARD_FLOOR):
            geometry = place_unit(unit, self.els, box, self.measurer, lo, max_font=cap)
            if geometry is not None:
                break
        rigid = None
        if geometry is not None:
            _, _, w, h = union_box([rotated_aabb(g) for g in geometry.values()])
            rigid = h if axis == "v" else w
        if unit.kind != "container_label":
            return rigid
        condensed = self._condensed(members, axis, cross, cap, floor)
        if roomy and font_cap > COMFORT * PREF_SLACK * floor + EPS and rigid is not None:
            return rigid  # at its largest, a pill keeps the source's proportions
        options = [v for v in (rigid, condensed) if v is not None]
        return min(options) if options else None

    def _condensed(self, members: list[dict], axis: str, cross: float, cap: float, floor: float) -> float | None:
        """A pill rebuilt around its one-line label (place.py's condensed form): far
        shorter than the rigidly scaled source pill when that one is tall for its label.
        At the floor it gets the least padding place.py accepts, above it PILL_VPAD."""
        label = next((m for m in members if m["type"] == "text"), None)
        if label is None:
            return None
        vpad = CONDENSED_VPAD if cap <= floor + EPS else PILL_VPAD
        size = min(cap, label["font_size"] * MAX_TEXT_GROWTH)
        for _ in range(3):  # shrink the label until the pill fits across (text width ~ size)
            text = resize_text(label, size)
            text_w = max(self.measurer.line_widths(text, size, math.inf), default=0.0)
            block = self.measurer.block_height(text, size, text_w)
            pill_w, pill_h = text_w + 2 * CONDENSED_PADDING * size, block + 2 * vpad * size + 1.0
            if (pill_w if axis == "v" else pill_h) <= cross:
                return pill_h if axis == "v" else pill_w
            size = math.floor(size * cross / (pill_w if axis == "v" else pill_h) * 2) / 2
            if size < floor:
                return None
        return None

    def _source_box(self, unit: Unit) -> Box:
        return union_box([rotated_aabb(m) for m in self.members(unit)])

    # one slot

    def image_cross(self, index: int, axis: str, size: float) -> float | None:
        """Cross extent the images (or logos) of a slot need to fill `size` along `axis`;
        None for any other slot."""
        units = self.units.get(index) or []
        if not units or self.slots[index].minor or self.slots[index].stack:
            return None
        members = [m for u in units for m in self.members(u)]
        logos = all(u.role == "logo" for u in units)
        if not logos and any(m["type"] != "image" or m.get("fit") == "cover" or m.get("rotation") for m in members):
            return None
        aspects = []
        for u in units:
            _, _, w, h = self._source_box(u)
            if w <= 0 or h <= 0:
                return None
            aspects.append(w / h)
        n = len(units)
        return n * size * max(aspects) if axis == "v" else size / n / min(aspects)

    def need(self, index: int, axis: str, cross: float, basis: float, room: float | None = None) -> Need:
        units = self.units.get(index) or []
        if not units:
            return Need(0.0, 0.0, 0.0, 0.0, empty=True)
        weight = max(self.plan.weight(u.unit_id) for u in units)  # a slot matters as much as its top unit
        slot = self.slots[index]
        try:
            if slot.minor:
                lo, pref, cap = self._minor(units, axis, cross)
            elif slot.stack:
                lo, pref, cap = self._stack(units, axis, cross)
            elif all(u.role == "logo" for u in units):
                lo, pref, cap = self._logo(units, axis, cross if room is None else max(cross, room))
            elif all(m["type"] == "image" for u in units for m in self.members(u)):
                lo, pref, cap = self._image(units, axis, cross if room is None else max(cross, room), basis)
            else:
                lo, pref, cap = self._row(units, axis, cross)
        except _Unmeasurable:
            lo = pref = cap = basis
        values = (lo, pref, cap)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values):
            lo = pref = cap = basis
        keep = 0.0
        if any(u.role == "logo" for u in units) and not slot.stack:
            keep = min(lo, LOGO_KEEP * (self.height if axis == "v" else self.width))
        return Need(lo, max(lo, pref), max(lo, pref, cap), weight, basis=basis, keep=keep)

    def _measure(self, unit: Unit, axis: str, cross: float, font_cap: float, roomy: bool = False) -> float:
        value = self.extent(unit, axis, cross, font_cap, roomy)
        if value is None:
            raise _Unmeasurable
        return value

    def _levels(self, unit: Unit) -> tuple[float, float, float]:
        floor = self.floor(unit)
        pref = COMFORT * PREF_SLACK * floor
        return floor, pref, math.inf if unit.unit_id == self.lead else max(pref, SUPPORT_CAP * floor)

    def _stack(self, units: list[Unit], axis: str, cross: float) -> tuple[float, float, float]:
        """Text units top to bottom. place_stack gives each unit its height at the floor
        plus a weight share of the spare height, so a slot whose units should all reach a
        size needs enough spare for the lightest unit's share to get there."""
        weights = [max(self.plan.weight(u.unit_id), 1e-3) for u in units]
        total = sum(weights)

        def height(width: float, level: int) -> float:
            base = [self._measure(u, "v", width, self._levels(u)[0]) for u in units]
            inner = self.gap * (STACK_GAP if level == 0 else 1.0) * (len(units) - 1)
            if level == 0:
                return sum(base) + inner
            want = [self._measure(u, "v", width, self._levels(u)[level]) for u in units]
            spare = max(max(0.0, t - b) * total / w for t, b, w in zip(want, base, weights))
            return sum(base) + spare + inner

        if axis == "v":
            return tuple(height(cross, level) for level in range(3))
        widest = self.content[2]
        out = []
        for level in range(3):
            if height(widest, level) > cross:
                out.append(widest)
                continue
            lo, hi = 1.0, widest
            for _ in range(SEARCH_STEPS):
                mid = (lo + hi) / 2
                try:
                    fits = height(mid, level) <= cross
                except _Unmeasurable:  # too narrow for a word: wider
                    fits = False
                lo, hi = (lo, mid) if fits else (mid, hi)
            out.append(hi)
        return tuple(out)

    def _row(self, units: list[Unit], axis: str, cross: float) -> tuple[float, float, float]:
        """Units side by side in equal cells (place_row): pills, CTAs, badges."""
        n = len(units)
        out = []
        for level in range(3):
            roomy = level == 2 if axis == "v" else level >= 1
            if axis == "v":
                out.append(max(self._measure(u, "v", cross / n, self._levels(u)[level], roomy) for u in units))
            else:
                out.append(n * max(self._measure(u, "h", cross, self._levels(u)[level], roomy) for u in units))
        return tuple(out)

    def _minor(self, units: list[Unit], axis: str, cross: float) -> tuple[float, float, float]:
        """The minor zone: one line, everything at its floor (place_flow); it never grows."""
        sizes = []
        for u in units:
            if u.role == "logo":  # a logo dropped to the minor zone still gets its least visible size
                sizes.append(self._logo([u], axis, cross)[0])
            elif all(m["type"] == "image" for m in self.members(u)):
                _, _, w, h = self._source_box(u)
                aspect = w / h if h > 0 else 1.0
                if axis == "v":
                    sizes.append(min(cross / aspect, MINOR_IMAGE_SHARE * self.height))
                else:
                    sizes.append(min(cross * aspect, MINOR_IMAGE_SHARE * self.width))
            else:
                sizes.append(self._measure(u, axis, cross, self.floor(u)))
        if axis == "v":
            size = max(sizes)
            return size, size, size
        flow = sum(sizes) + FLOW_GAP * min(self.width, self.height) * (len(sizes) - 1)
        return max(sizes), flow, flow

    def _logo(self, units: list[Unit], axis: str, cross: float) -> tuple[float, float, float]:
        n = len(units)
        lo = pref = cap = 0.0
        for u in units:
            _, _, w, h = self._source_box(u)
            if w <= 0 or h <= 0:
                raise _Unmeasurable
            aspect = w / h
            members = self.members(u)
            ink = ink_share(members[0]) if len(members) == 1 else (1.0, 1.0)
            least = logo_min_width(self.width, self.height, w, h, ink)
            most = max(least, min(w * LOGO_MAX_GROWTH, LOGO_MAX_WIDTH * self.width))
            sizes = (least, min(least * LOGO_PREF, most), most)
            if axis == "v":  # cross is the width
                got = [min(s, cross / n) / aspect for s in sizes]
                lo, pref, cap = (max(a, b) for a, b in zip((lo, pref, cap), got))
            else:  # cross is the height
                got = [min(s / aspect, cross) * aspect for s in sizes]
                lo, pref, cap = (a + b for a, b in zip((lo, pref, cap), got))
        return lo, pref, cap

    def _image(self, units: list[Unit], axis: str, cross: float, basis: float) -> tuple[float, float, float]:
        """Hero images: preferred at the template's share (or less, if that already fills
        the cross extent), may shrink to IMAGE_MIN of it, may grow to fill the cross extent."""
        n = len(units)
        fills = []
        for u in units:
            members = self.members(u)
            if any(m.get("fit") == "cover" for m in members):
                fills.append(math.inf)
                continue
            _, _, w, h = self._source_box(u)
            if w <= 0 or h <= 0:
                raise _Unmeasurable
            aspect = w / h
            fills.append(cross / n / aspect if axis == "v" else cross * aspect)
        fill = max(fills) if axis == "v" else (n * max(fills) if all(math.isfinite(f) for f in fills) else math.inf)
        canvas = self.height if axis == "v" else self.width
        pref = min(fill, basis)
        cap = min(fill, IMAGE_MAX_SHARE * canvas)
        return IMAGE_MIN * pref, pref, max(pref, cap)


class _Unmeasurable(Exception):
    pass


# --- the tree solve ---------------------------------------------------------

def _node_need(node: Node, axis: str, cross: float, hint: float, ctx: _Content, room: float | None = None) -> Need:
    """What `node` needs along `axis`, `cross` being its extent across it, `hint` its
    provisional extent along it and `room` the cross extent an image may widen to."""
    if node.index is not None:
        return ctx.need(node.index, axis, cross, hint, room)
    if node.axis == axis:
        subs = []
        for child in node.children:
            _, cross_frac = _frac(child, node, _other(axis))
            _, along_frac = _frac(child, node, axis)
            subs.append(_node_need(child, axis, cross * cross_frac, hint * along_frac, ctx, cross))
        live = [s for s in subs if not s.empty]
        if not live:
            return Need(0.0, 0.0, 0.0, 0.0, empty=True)
        gaps = ctx.gap * (len(live) - 1)
        return Need(sum(s.min for s in live) + gaps, sum(s.pref for s in live) + gaps,
                    sum(s.cap for s in live) + gaps, sum(s.weight for s in live),
                    basis=hint, keep=sum(s.keep for s in live))
    # Children are laid across `axis` and share this node's extent along it: settle
    # their sizes across first, then take the largest need along. Their sizes across
    # depend on the extent along (a logo row is wider when taller), so this is done
    # again with the extent along that the first pass asked for.
    first = _across_need(node, axis, cross, hint, ctx)
    if first.empty or first.pref <= hint + EPS:
        return first
    return replace(_across_need(node, axis, cross, first.pref, ctx), basis=hint)


def _across_need(node: Node, axis: str, cross: float, hint: float, ctx: _Content) -> Need:
    across = node.axis
    subs = []
    for child in node.children:
        _, frac_along = _frac(child, node, axis)
        _, frac_across = _frac(child, node, across)
        subs.append(_node_need(child, across, hint * frac_along, cross * frac_across, ctx, hint))
    alloc = distribute(cross, subs, ctx.gap)
    lo = pref = cap = weight = keep = 0.0
    any_live = False
    for child, sub, (_, size) in zip(node.children, subs, alloc):
        if sub.empty:
            continue
        _, frac_along = _frac(child, node, axis)
        frac_along = max(frac_along, 0.05)
        n = _node_need(child, axis, size, hint * frac_along, ctx, cross)
        if n.empty:
            continue
        any_live = True
        lo, pref, cap = max(lo, n.min / frac_along), max(pref, n.pref / frac_along), max(cap, n.cap / frac_along)
        keep = max(keep, n.keep / frac_along)
        weight += n.weight
    if not any_live:
        return Need(0.0, 0.0, 0.0, 0.0, empty=True)
    return Need(lo, pref, cap, weight, basis=hint, keep=keep)


def _solve(node: Node, box: Box, ctx: _Content, out: dict[int, Box]) -> None:
    if node.index is not None:
        out[node.index] = box
        return
    axis, across = node.axis, _other(node.axis)
    start, length = _along(box, axis)
    c_start, c_len = _along(box, across)
    needs, crosses = [], []
    for child in node.children:
        f0, fl = _frac(child, node, across)
        _, al = _frac(child, node, axis)
        crosses.append((c_start + f0 * c_len, fl * c_len))
        needs.append(_node_need(child, axis, fl * c_len, al * length, ctx, c_len))
    for child, (offset, size), (cs, cl) in zip(node.children, distribute(length, needs, ctx.gap), crosses):
        wide = ctx.image_cross(child.index, axis, size) if child.index is not None else None
        if wide is not None and wide > cl:  # an image may widen across, up to its parent, about its centre
            centre, cl = cs + cl / 2, min(wide, c_len)
            cs = min(max(centre - cl / 2, c_start), c_start + c_len - cl)
        if axis == "v":
            child_box = (cs, start + offset, cl, size)
        else:
            child_box = (start + offset, cs, size, cl)
        _solve(child, child_box, ctx, out)


def _crowded(root: Node, content: Box, ctx: _Content) -> bool:
    """True when the units can't all fit at their floors along the template's main cut:
    something will be degraded or moved to the free-space pass whatever the slot sizes,
    and the fixed template is what the engine's fallbacks for that case are tuned on."""
    start, length = _along(content, root.axis)
    _, c_len = _along(content, _other(root.axis))
    live = []
    for child in root.children:
        _, fl = _frac(child, root, _other(root.axis))
        _, al = _frac(child, root, root.axis)
        need = _node_need(child, root.axis, fl * c_len, al * length, ctx, c_len)
        if not need.empty:
            live.append(need)
    return sum(n.min for n in live) + ctx.gap * max(0, len(live) - 1) > length + EPS


def flex_slots(slots: list[Slot], assignment: dict, content: Box, els: dict, plan: PriorityPlan,
               measurer: TextMeasurer, width: float, height: float) -> list[Slot]:
    """`slots` with content-aware boxes inside `content` (same order, names and roles),
    or `slots` unchanged if they can't be decomposed or anything goes wrong."""
    try:
        cx, cy, cw, ch = content
        if cw <= 0 or ch <= 0:
            return slots
        fracs = [((s.box[0] - cx) / cw, (s.box[1] - cy) / ch, s.box[2] / cw, s.box[3] / ch) for s in slots]
        root = decompose(fracs)
        if root is None:
            return slots
        ctx = _Content(slots, assignment, els, plan, measurer, width, height, content)
        if _crowded(root, content, ctx):
            return slots
        boxes: dict[int, Box] = {}
        _solve(root, content, ctx, boxes)
        if sorted(boxes) != list(range(len(slots))):
            return slots
        tol = 0.01
        for x, y, w, h in boxes.values():
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w < -EPS or h < -EPS:
                return slots
            if x < cx - tol or y < cy - tol or x + w > cx + cw + tol or y + h > cy + ch + tol:
                return slots
        return [replace(s, box=(boxes[i][0], boxes[i][1], max(0.0, boxes[i][2]), max(0.0, boxes[i][3])))
                for i, s in enumerate(slots)]
    except Exception:
        return slots
