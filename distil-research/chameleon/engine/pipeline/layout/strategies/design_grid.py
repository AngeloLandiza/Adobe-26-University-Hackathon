"""Strategy S4 `design_grid`: a responsive modular grid plus a modular type scale.

The classic designer recipe. From the target canvas it derives

* a GRID: a baseline unit (short side / 48), margins and gutters that are
  whole numbers of units, and a column count; every box edge in the
  horizontal compositions sits on a column line, every vertical gap is a whole
  number of units;
* a TYPE SCALE: sizes `base * ratio**k` (ratio 1.25 or 1.333) anchored so the
  subheadline floor is a step and the legal floor is reachable. The headline
  takes the top step that fits; every other text takes the step its
  source-size ratio to the headline says, so the source hierarchy survives;
  a size that was limited by the box is snapped DOWN to the next step.

Compositions per aspect class: a vertical flow (header / hero / headline block
/ badges / CTA / footer) for tall and square canvases, a two-column split for
square and landscape, hero | text | CTA columns for wide, and a one-line
logo | hero | text | CTA row for strips. Left-over space becomes equal,
unit-snapped gaps instead of one large hole.

`candidates()` never raises and every returned candidate passes `check_gates`.
"""

import math

from engine.pipeline.geometry import Box, rotated_aabb, union_box
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout.aspect import aspect_class
from engine.pipeline.layout.generate import _Builder
from engine.pipeline.layout.place import HARD_FLOOR, MAX_TEXT_GROWTH, floor_px, place_unit

NAME = "design_grid"
RATIOS = (1.25, 1.333)
MARGIN_UNITS = (3, 4)
FLOW_COLS = 8
ROW_COLS = 12
UNIT_DIV = 48           # baseline unit = short side / 48
GAP_MIN_UNITS = 2       # least gap between vertical bands, in units
GAP_MAX_UNITS = 6       # the most extra space one inner gap absorbs, in units
MIN_HERO = (0.38, 0.55)  # share of the content height the hero band should reach (loose, tight)
MIN_TEXT_COLS = 4
BAND_CAPS = {"header": 0.15, "badges": 0.08, "cta": 0.13, "footer": 0.08}  # band height / content height


# --------------------------------------------------------------------------- grid and scale

class Grid:
    """Columns, gutters, margins and a baseline unit for one canvas."""

    def __init__(self, width: float, height: float, cols: int = 12, margin_units: int = 4):
        short = min(width, height)
        self.width, self.height = width, height
        self.unit = max(2.0, float(round(short / UNIT_DIV)))
        self.margin = min(margin_units * self.unit, math.floor(0.12 * short / self.unit) * self.unit)
        self.margin = max(self.margin, self.unit)
        self.gutter = self.unit * (2 if short >= 120 else 1)
        content = width - 2 * self.margin
        cols = max(1, int(cols))
        while cols > 1 and (content - (cols - 1) * self.gutter) / cols < 2 * self.unit:
            cols -= 1
        self.cols = cols
        self.col_w = (content - (cols - 1) * self.gutter) / cols
        self.pitch = self.col_w + self.gutter

    def col_x(self, i: int) -> float:
        return self.margin + i * self.pitch

    def span_w(self, c0: int, c1: int) -> float:
        """Width of columns c0 .. c1-1 including the gutters between them."""
        n = c1 - c0
        return n * self.col_w + (n - 1) * self.gutter

    @property
    def content(self) -> Box:
        return (self.margin, self.margin, self.width - 2 * self.margin, self.height - 2 * self.margin)

    def cols_for(self, width: float) -> int:
        """Fewest columns whose span is at least `width` (at least 1)."""
        return max(1, min(self.cols, math.ceil((width + self.gutter) / self.pitch - 1e-9)))


def type_scale(width: float, height: float, ratio: float) -> list[float]:
    """Ascending sizes `base * ratio**k` (rounded to 0.5 px), anchored on the
    subheadline floor; the lowest step is at or under the legal floor."""
    base = floor_px("subheadline", width, height)
    steps = sorted({round(base * ratio ** k * 2) / 2 for k in range(-2, 15)})
    return steps


def snap_down(steps: list[float], value: float) -> float:
    """The largest step not above `value` (the smallest step when none is)."""
    best = steps[0]
    for s in steps:
        if s <= value + 1e-6:
            best = s
    return best


# --------------------------------------------------------------------------- units and context

_GROUP_OF_ROLE = {"logo": "logo", "badge": "badge", "hero_image": "hero", "secondary_image": "secondary",
                  "headline": "headline", "subheadline": "sub", "cta": "cta", "legal": "legal"}


def _groups(plan, els: dict | None = None, canvas: dict | None = None) -> dict[str, list]:
    out = {k: [] for k in ("logo", "badge", "hero", "secondary", "headline", "sub", "other", "cta", "legal")}
    for unit in plan.ranked_units():
        out[_GROUP_OF_ROLE.get(unit.role, "other")].append(unit)
    if not out["hero"] and els and canvas:
        # no hero_image in the plan: a big secondary image (a product the labeller called secondary) is the hero
        area = canvas["width"] * canvas["height"]
        big = [u for u in out["secondary"]
               if sum(els[i]["width"] * els[i]["height"] for i in u.element_ids) >= 0.05 * area]
        if big:
            out["hero"] = big
            out["secondary"] = [u for u in out["secondary"] if u not in big]
    return out


class _Ctx:
    """One composition attempt: a builder plus the grid, the scale and the plan's units."""

    def __init__(self, source, target, plan, measurer, ratio: float, margin_units: int, cols: int):
        self.b = _Builder(source, target, plan, measurer)
        self.plan, self.measurer = plan, measurer
        self.W, self.H = self.b.width, self.b.height
        self.ratio = ratio
        self.grid = Grid(self.W, self.H, cols, margin_units)
        self.unit = self.grid.unit
        self.steps = type_scale(self.W, self.H, ratio)
        self.g = _groups(plan, self.b.els, source["canvas"])
        ranked = [u for u in plan.ranked_units() if self._font(u)]
        ref = (self.g["headline"] or ranked or [None])[0]
        self.ref = ref
        self.ref_size = self._font(ref) if ref is not None else 0.0
        aligns = [self.b.els[i].get("text_align") or "left" for u in self.g["headline"] + self.g["sub"]
                  for i in u.element_ids if self.b.els[i]["type"] == "text"]
        centre = aligns.count("center") >= max(1, len(aligns) - aligns.count("center"))
        self.lead = "center" if (centre or not aligns) else ("right" if aligns.count("right") > aligns.count("left") else "left")

    # -- type scale
    def _font(self, unit) -> float:
        sizes = [self.b.els[i]["font_size"] for i in unit.element_ids if self.b.els[i]["type"] == "text"]
        return max(sizes) if sizes else 0.0

    def top_index(self) -> int:
        """Index of the highest step the reference text may take."""
        size = snap_down(self.steps, self.ref_size * MAX_TEXT_GROWTH)
        return self.steps.index(size)

    def cap(self, unit, h_idx: int) -> float | None:
        """Font cap for a text-carrying unit when the headline sits at step `h_idx`."""
        src = self._font(unit)
        if not src:
            return None
        floor = floor_px(unit.role, self.W, self.H)
        if unit is self.ref or not self.ref_size:
            off = 0
        else:
            off = round(math.log(src / self.ref_size) / math.log(self.ratio))
            off = min(off, 0 if unit.role == "headline" else -1)
        size = self.steps[max(0, min(len(self.steps) - 1, h_idx + off))]
        if unit.role in ("legal", "badge"):
            size = min(size, max(src, floor))
        return max(size, floor)

    # -- placing
    def place(self, unit, box: Box, align: str, cap: float | None, snap: bool = True):
        """(geometry, degraded) or (None, False). Text snaps down to a step."""
        box = self.b._cap_logo(unit, box)
        floor = floor_px(unit.role, self.W, self.H)
        if box[2] <= 0 or box[3] <= 0:
            return None, False
        els = self.b.els
        geo = place_unit(unit, els, box, self.measurer, floor, align=align, max_font=cap)
        if geo is None:
            geo = place_unit(unit, els, box, self.measurer, HARD_FLOOR, align=align, max_font=cap)
            return (geo, True) if geo is not None else (None, False)
        members = [els[i] for i in unit.element_ids]
        if snap and len(members) == 1 and members[0]["type"] == "text":
            size = geo[members[0]["id"]]["font_size"]
            target = snap_down(self.steps, size)
            if size - target > 0.01 and target >= floor - 1e-6:
                again = place_unit(unit, els, box, self.measurer, floor, align=align, max_font=target)
                if again is not None:
                    geo = again
        return geo, False

    def put(self, unit, geo, degraded: bool, dx: float, dy: float) -> None:
        shifted = {k: dict(v, x=v["x"] + dx, y=v["y"] + dy) for k, v in geo.items()}
        self.b._record(unit, shifted, degraded)


def _bbox(geo: dict) -> Box:
    return union_box([rotated_aabb(g) for g in geo.values()])


class _Band:
    """Items laid out relative to the band's top-left corner."""

    def __init__(self, kind: str):
        self.kind = kind
        self.items: list = []  # (unit, geometry, degraded)
        self.height = 0.0

    def add(self, unit, geo, degraded, x_shift=0.0):
        bx, by, bw, bh = _bbox(geo)
        self.items.append((unit, geo, degraded, x_shift, -by))
        self.height = max(self.height, bh)


def _row_band(ctx: _Ctx, kind: str, units: list, w: float, h: float, aligns, widths=None, caps=None):
    """Units side by side, top aligned. None if one can't be placed."""
    band = _Band(kind)
    n = len(units)
    if not n:
        return band
    widths = widths or [w / n] * n
    aligns = aligns if isinstance(aligns, (list, tuple)) else [aligns] * n
    x = 0.0
    for i, u in enumerate(units):
        cap = caps[i] if caps else None
        geo, deg = ctx.place(u, (0, 0, widths[i], h), aligns[i], cap)
        if geo is None:
            return None
        band.add(u, geo, deg, x)
        x += widths[i]
    return band


def _text_stack(ctx: _Ctx, units: list, w: float, h_avail: float, h_idx: int):
    """Headline block: units top to bottom with a scale step each. None if it
    doesn't fit `h_avail`."""
    band = _Band("text")
    y, prev_size, head_size = 0.0, None, None
    for u in units:
        cap = ctx.cap(u, h_idx)
        if head_size is not None and cap is not None and u.role != "headline":
            cap = min(cap, max(snap_down(ctx.steps, head_size / ctx.ratio + 0.01), floor_px(u.role, ctx.W, ctx.H)))
        gap = 0.0
        if prev_size is not None:
            gap = max(ctx.unit, math.floor(0.45 * prev_size / ctx.unit) * ctx.unit)
        room = h_avail - y - gap
        if room <= 0:
            return None
        geo, deg = ctx.place(u, (0, 0, w, room), "top", cap)
        if geo is None:
            return None
        bx, by, bw, bh = _bbox(geo)
        y += gap
        band.items.append((u, geo, deg, 0.0, y - by))
        y += bh
        size = max((g["font_size"] for g in geo.values() if "font_size" in g), default=0.0)
        if size:
            prev_size = size
            if head_size is None:
                head_size = size
        if y > h_avail + 1e-6:
            return None
    band.height = y
    return band


# --------------------------------------------------------------------------- vertical flow

def _hero_natural(ctx: _Ctx, units: list, width: float) -> float:
    """Height the hero units need at `width` to keep their proportions."""
    if not units:
        return 0.0
    heights = []
    for u in units:
        ux, uy, uw, uh = union_box([rotated_aabb(ctx.b.els[i]) for i in u.element_ids])
        heights.append(width / max(uw / uh, 1e-6) if len(units) == 1 else uh)
    return max(heights)


def _flow(ctx: _Ctx, box: Box, parts: dict, p: dict) -> bool:
    """Lay the `parts` out top to bottom inside `box`. Whatever can't be
    placed joins the builder's leftovers. False if nothing sensible fits."""
    bx, by, bw, bh = box
    unit = ctx.unit
    hero_inset = p.get("inset", 0)
    if hero_inset and box == ctx.grid.content and ctx.grid.cols >= 4:
        hx, hw = ctx.grid.col_x(hero_inset) - bx, ctx.grid.span_w(hero_inset, ctx.grid.cols - hero_inset)
    else:
        hx, hw = 0.0, bw
    logos, badges = parts.get("logo", []), parts.get("badge", [])
    heroes, text = parts.get("hero", []), parts.get("text", [])
    ctas = parts.get("cta", [])
    foot = parts.get("secondary", []) + parts.get("legal", [])
    hero_first = p.get("hero_first", True)
    gmin = GAP_MIN_UNITS * unit

    def build(h_idx: int, min_hero: float):
        header_units = logos + (badges if len(badges) == 1 else [])
        badge_row = badges if len(badges) > 1 else []
        header = badge_band = cta_band = footer = None
        if header_units:
            if logos and len(header_units) == 2 and badges:
                widths, aligns = [bw * 0.55, bw * 0.45], ["left", "right"]
            else:
                widths, aligns = None, ctx.lead
            caps = [ctx.cap(u, h_idx) for u in header_units]
            header = _row_band(ctx, "header", header_units, bw, BAND_CAPS["header"] * bh, aligns, widths, caps)
            if header is None:
                return None
        if badge_row:
            badge_band = _row_band(ctx, "badges", badge_row, bw, BAND_CAPS["badges"] * bh, "center",
                                   None, [ctx.cap(u, h_idx) for u in badge_row])
            if badge_band is None:
                return None
        if ctas:
            cta_band = _row_band(ctx, "cta", ctas, bw, BAND_CAPS["cta"] * bh, ctx.lead,
                                 None, [ctx.cap(u, h_idx) for u in ctas])
            if cta_band is None:
                return None
        if foot:
            if len(foot) == 2:
                widths = [bw * 0.3, bw * 0.7] if foot[0].role == "secondary_image" else None
            else:
                widths = None
            caps = [floor_px(u.role, ctx.W, ctx.H) if u.role == "legal" else None for u in foot]
            footer = _row_band(ctx, "footer", foot, bw, BAND_CAPS["footer"] * bh, ctx.lead, widths, caps)
            if footer is None:
                return None
        fixed = [b for b in (header, badge_band, cta_band, footer) if b is not None]
        n_other = len(fixed) + (1 if text else 0) + (1 if heroes else 0)
        fixed_h = sum(b.height for b in fixed)
        gaps = max(0, n_other - 1) * gmin
        text_band = None
        if text:
            room = bh - fixed_h - gaps - (min_hero * bh * 0.4 if heroes else 0)
            text_band = _text_stack(ctx, text, bw, max(room, 1.0), h_idx)
            if text_band is None:
                return None
        remaining = bh - fixed_h - (text_band.height if text_band else 0) - gaps
        if remaining < 0:
            return None
        hero_band = None
        if heroes:
            natural = _hero_natural(ctx, heroes, hw)
            if remaining < min(min_hero * bh, natural) - 1e-6:
                return None
            hero_band = _row_band(ctx, "hero", heroes, hw, min(remaining, natural) if natural else remaining,
                                  "center")
            if hero_band is None:
                return None
            for k, it in enumerate(hero_band.items):
                hero_band.items[k] = (it[0], it[1], it[2], it[3] + hx, it[4])
        return header, hero_band, text_band, badge_band, cta_band, footer

    if not (heroes or text or ctas or logos or badges or foot):
        return True
    built = None
    top = ctx.top_index() if ctx.ref is not None else len(ctx.steps) - 1
    for min_hero in (p.get("min_hero", MIN_HERO[0]), 0.18, 0.0):
        for h_idx in range(top, -1, -1):
            built = build(h_idx, min_hero)
            if built is not None:
                break
        if built is not None:
            break
    if built is None:
        for u in logos + badges + heroes + text + ctas + foot:
            ctx.b.leftover.append(u)
        return False
    header, hero_band, text_band, badge_band, cta_band, footer = built
    order = [header, hero_band, text_band] if hero_first else [header, text_band, hero_band]
    order += [badge_band, cta_band, footer]
    bands = [b for b in order if b is not None]
    total = sum(b.height for b in bands)
    n = len(bands)
    spare = bh - total - gmin * max(0, n - 1)
    extra = 0.0
    if n > 1:
        extra = min(GAP_MAX_UNITS * unit, math.floor(max(spare, 0) / n / unit) * unit)
    gap = gmin + extra
    used = total + gap * (n - 1)
    outer = max(0.0, math.floor((bh - used) / 2 / unit) * unit)
    y = by + outer
    for band in bands:
        for u, geo, deg, dx, dy in band.items:
            ctx.put(u, geo, deg, bx + dx, y + dy)
        y += band.height + gap
    return True


# --------------------------------------------------------------------------- compositions

def _parts(ctx: _Ctx, without=()) -> dict:
    g = ctx.g
    parts = {"logo": g["logo"], "badge": g["badge"], "hero": g["hero"], "secondary": g["secondary"],
             "text": g["headline"] + g["sub"] + g["other"], "cta": g["cta"], "legal": g["legal"]}
    # reading order inside the stack follows the plan's ranks, headline first
    parts["text"] = sorted(parts["text"], key=lambda u: (u.role != "headline", u.rank if u.rank is not None else 99))
    for k in without:
        parts[k] = []
    return parts


def _place_zone(ctx: _Ctx, units: list, box: Box, align: str = "center") -> None:
    """Fill a column with side-by-side units (hero images), centred in the box."""
    if not units:
        return
    x, y, w, h = box
    band = _row_band(ctx, "zone", units, w, h, align, None, [ctx.cap(u, ctx.top_index()) for u in units])
    if band is None:
        ctx.b.leftover.extend(units)
        return
    top = y + (h - band.height) / 2
    for u, geo, deg, dx, dy in band.items:
        ctx.put(u, geo, deg, x + dx, top + dy)


def _compose_flow(ctx: _Ctx, p: dict) -> bool:
    return _flow(ctx, ctx.grid.content, _parts(ctx), p)


def _hero_cols(ctx: _Ctx, box_h: float, extra: int, lo: int = 1) -> int:
    heroes = ctx.g["hero"]
    if not heroes:
        return 0
    gr = ctx.grid
    width = max(_hero_width_for(ctx, h, box_h) for h in heroes) * (1 if len(heroes) == 1 else len(heroes))
    hc = gr.cols_for(width) + extra
    return max(lo, min(hc, gr.cols - MIN_TEXT_COLS))


def _hero_width_for(ctx: _Ctx, unit, height: float) -> float:
    ux, uy, uw, uh = union_box([rotated_aabb(ctx.b.els[i]) for i in unit.element_ids])
    return height * uw / max(uh, 1e-6)


def _compose_split(ctx: _Ctx, p: dict) -> bool:
    gr = ctx.grid
    cx, cy, cw, ch = gr.content
    if not ctx.g["hero"]:
        return _flow(ctx, gr.content, _parts(ctx), {"hero_first": True})
    hc = _hero_cols(ctx, ch, p.get("hc_extra", 0))
    tc = gr.cols - hc
    if p.get("mirror"):
        hero_box = (gr.col_x(tc), cy, gr.span_w(tc, gr.cols), ch)
        text_box = (gr.col_x(0), cy, gr.span_w(0, tc), ch)
    else:
        hero_box = (gr.col_x(0), cy, gr.span_w(0, hc), ch)
        text_box = (gr.col_x(hc), cy, gr.span_w(hc, gr.cols), ch)
    _place_zone(ctx, ctx.g["hero"], hero_box)
    return _flow(ctx, text_box, _parts(ctx, without=("hero",)), {"hero_first": True})


def _cta_cols(ctx: _Ctx, box_h: float, lo: int, hi: int) -> int:
    """Fewest columns (lo..hi) in which the CTA reaches (almost) its cap."""
    ctas = ctx.g["cta"]
    if not ctas:
        return 0
    gr = ctx.grid
    top = ctx.top_index() if ctx.ref is not None else len(ctx.steps) - 1
    cap = ctx.cap(ctas[0], top) or 0
    best = lo
    for c in range(lo, hi + 1):
        geo, _ = ctx.place(ctas[0], (0, 0, gr.span_w(0, c), box_h), "center", cap, snap=False)
        best = c
        if geo is not None:
            size = max((g["font_size"] for g in geo.values() if "font_size" in g), default=0.0)
            if size >= 0.9 * cap:
                break
    return best


def _compose_wide(ctx: _Ctx, p: dict) -> bool:
    gr = ctx.grid
    cx, cy, cw, ch = gr.content
    heroes, ctas = ctx.g["hero"], ctx.g["cta"]
    cc = _cta_cols(ctx, ch, 2, max(2, gr.cols // 3)) if ctas else 0
    hc = _hero_cols(ctx, ch, p.get("hc_extra", 0)) if heroes else 0
    while gr.cols - hc - cc < MIN_TEXT_COLS and (hc > 1 or cc > 2):
        if hc > 1:
            hc -= 1
        else:
            cc -= 1
    c = 0
    hero_box = cta_box = None
    if hc:
        hero_box = (gr.col_x(0), cy, gr.span_w(0, hc), ch)
        c = hc
    text_box = (gr.col_x(c), cy, gr.span_w(c, gr.cols - cc), ch)
    if cc:
        cta_box = (gr.col_x(gr.cols - cc), cy, gr.span_w(gr.cols - cc, gr.cols), ch)
    if hero_box:
        _place_zone(ctx, heroes, hero_box)
    if cta_box:
        band = _row_band(ctx, "cta", ctas, cta_box[2], ch, "center", None,
                         [ctx.cap(u, ctx.top_index()) for u in ctas])
        if band is None:
            ctx.b.leftover.extend(ctas)
        else:
            top = cy + (ch - band.height) / 2
            for u, geo, deg, dx, dy in band.items:
                ctx.put(u, geo, deg, cta_box[0] + dx, top + dy)
    return _flow(ctx, text_box, _parts(ctx, without=("hero", "cta")), {"hero_first": True})


def _compose_strip(ctx: _Ctx, p: dict) -> bool:
    """logo | hero | headline block | CTA on one line; the rest waits for the free-space pass."""
    gr = ctx.grid
    cx, cy, cw, ch = gr.content
    logos, heroes, ctas = ctx.g["logo"], ctx.g["hero"] if p.get("hero", True) else [], ctx.g["cta"]
    lc = 0
    if logos:
        lw = max(_hero_width_for(ctx, u, ch) for u in logos) * len(logos)
        lc = min(3, gr.cols_for(lw))
    hc = _hero_cols(ctx, ch, 0) if heroes else 0
    hc = min(hc, 3)
    cc = _cta_cols(ctx, ch, 2, max(2, gr.cols // 3)) if ctas else 0
    while gr.cols - lc - hc - cc < MIN_TEXT_COLS and (hc or lc > 1 or cc > 2):
        if hc:
            hc -= 1
        elif lc > 1:
            lc -= 1
        else:
            cc -= 1
    if gr.cols - lc - hc - cc < 2:
        return False
    c = 0
    if lc:
        _place_zone(ctx, logos, (gr.col_x(0), cy, gr.span_w(0, lc), ch), "left")
        c = lc
    if hc:
        _place_zone(ctx, heroes, (gr.col_x(c), cy, gr.span_w(c, c + hc), ch))
        c += hc
    elif heroes:
        ctx.b.leftover.extend(heroes)
    text_box = (gr.col_x(c), cy, gr.span_w(c, gr.cols - cc), ch)
    if cc:
        cbox = (gr.col_x(gr.cols - cc), cy, gr.span_w(gr.cols - cc, gr.cols), ch)
        _place_zone(ctx, ctas, cbox)
    # everything else: headline block here; badges, legal and secondary images go to the free-space pass
    parts = _parts(ctx, without=("logo", "hero", "cta", "badge", "secondary", "legal"))
    ctx.b.leftover.extend(ctx.g["badge"] + ctx.g["secondary"] + ctx.g["legal"])
    return _flow(ctx, text_box, parts, {"hero_first": True})


# --------------------------------------------------------------------------- driver

def _plans(cls: str) -> list[tuple[str, callable, dict]]:
    """(name, composer, extra params) per aspect class."""
    out = []
    if cls in ("very_tall", "tall", "square"):
        for first in (True, False):
            for inset in (0, 1):
                for mh in (0, 1):
                    out.append(("flow", _compose_flow,
                                {"hero_first": first, "inset": inset, "min_hero": MIN_HERO[mh]}, FLOW_COLS))
    if cls in ("square", "landscape"):
        for mirror in (False, True):
            for extra in (0, 1):
                out.append(("split", _compose_split, {"mirror": mirror, "hc_extra": extra}, ROW_COLS))
    if cls == "landscape":
        for first in (True, False):
            out.append(("flow", _compose_flow, {"hero_first": first, "inset": 0, "min_hero": MIN_HERO[0]}, FLOW_COLS))
    if cls == "wide":
        for extra in (0, 1):
            out.append(("cols", _compose_wide, {"hc_extra": extra}, ROW_COLS))
        for mirror in (False, True):
            out.append(("split", _compose_split, {"mirror": mirror, "hc_extra": 0}, ROW_COLS))
    if cls == "strip":
        for hero in (True, False):
            out.append(("line", _compose_strip, {"hero": hero}, ROW_COLS))
    return out


def _build(source, target, plan, measurer, name, composer, extra, cols, ratio, margin_units):
    ctx = _Ctx(source, target, plan, measurer, ratio, margin_units, cols)
    ok = composer(ctx, extra)
    if ok is False and not ctx.b.placed and not ctx.b.leftover:
        return None
    if not ctx.b.place_leftovers(ctx.grid.margin):
        return None
    params = {"ratio": ratio, "margin_units": margin_units, **{k: (int(v) if isinstance(v, bool) else v)
                                                                 for k, v in extra.items()}}
    cid = f"{NAME}/{name}/" + "/".join(f"{k}={v}" for k, v in sorted(params.items()))
    return ctx.b.finish(cid, NAME, params)


def candidates(source, target, plan, measurer) -> list:
    """Grid-composed candidate layouts for `target`. Never raises."""
    try:
        width, height = target["width"], target["height"]
        if not (width > 0 and height > 0) or not source.get("elements"):
            return []
        cls = aspect_class(width, height)
        out, seen = [], set()
        for name, composer, extra, cols in _plans(cls):
            for ratio in RATIOS:
                for margin_units in MARGIN_UNITS:
                    try:
                        cand = _build(source, target, plan, measurer, name, composer, extra, cols, ratio, margin_units)
                        if cand is None or check_gates(source, cand.design, target, plan, measurer):
                            continue
                        key = repr([(e["x"], e["y"], e["width"], e["height"], e.get("font_size"))
                                    for e in cand.design["elements"]])
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(cand)
                    except Exception:
                        continue
        return out
    except Exception:
        return []
