"""Source layers (CHAMELEON_LAYERS=1; off by default, and with it off nothing
here runs and the engine's output is unchanged).

Real ads are layered: copy sits in a card, on a band, inside a heart or a
label; a photo or pattern fills the whole canvas behind everything. The
layout strategies treat every unit as an independent piece, so a re-layout
pulls the text out of its card (leaving the empty card stranded) and treats
a full-bleed photo as a hero to shrink into a cell. This module reads the
layers off the source and hands the layout a plan that keeps them:

1. **Background layer**: the bottom stack of full-bleed elements, each an
   image or shape covering at least FULL_BLEED of the canvas with nothing
   but other full-bleed layers painted under it (a colour field, then a
   cover photo, then a tint). `background_stack` feeds the priority node
   (priority/units.py), so every such element gets the background role and
   is re-fitted to the new canvas full-bleed by the engine's own
   `place_background`, by every strategy, instead of being placed as a unit.
2. **Containers**: a shape or image (not rotated, not part of a multi-element
   unit, not in the background layer) whose box holds a content unit
   (at least CONTAIN of the unit's box inside it, the container at least
   GROWTH x the unit's area, painted under all of the unit's elements, or
   an image drawn over them: a frame raster round a photo).
   Each unit belongs to the smallest container holding it; a container held
   by a bigger one nests inside its group. A container and everything it
   holds become one unit of kind KIND in the layout plan, so bisection
   places the group in one cell: the contents are laid out inside the
   container (bisection again, in the container's box minus the source's
   padding, scaled with the text), and the container is fitted round them.
   - a rectangle shape or a `cover` image *stretches* to wrap its contents
     plus padding; one that bled off a canvas edge in the source is
     extended to that edge again when its cell reaches the content area's
     edge (a band stays a band);
   - any other image and round shapes keep their aspect (a heart stays a
     heart): the container is fitted to the cell and the contents are laid
     out inside it, padding scaled with the container.
   The padding shrinks (PAD_STEPS) before any content goes below its floor.
3. Decorations inside a container (an ornament on a card) stay decoration
   units; bisection v2's frame rule makes them follow the group they sat in.

Only bisection lays the groups out (strategies/bisection.py hooks: `detect`
in `_candidates`, `group_ext` in `_Model._unit_ext`, `place_group` in
`_place_unit`). With a target, `detect` drops a fixed-aspect container that
can't hold its texts near their floors there (a frame round the whole ad,
on a strip); its contents are laid out as units of their own.

The hard gates run on the original plan. Contents overlapping their own
container break no gate (G9 only flags an element fully under an opaque one
painted after it). With the switch on, grading/gates.py adds two things,
both tied to relations the source itself has:
- G9 lets an (element, image) pair be when the source already draws that
  image over the whole element (`source_cover_pairs`: a frame raster drawn
  round a photo would otherwise count as hiding it);
- G10_layers fails a candidate in which a container kept for that target
  (`detect` with the target) no longer holds every unit it held in the
  source (`broken_containers`): no stranded, empty cards from any strategy.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from engine.pipeline.geometry import Box, fit_aspect, overlap_area, rotated_aabb, union_box
from engine.pipeline.types import ALPHA, PriorityPlan, Unit

ENV = "CHAMELEON_LAYERS"
KIND = "layer_group"
FULL_BLEED = 0.9        # share of the canvas a background-layer element covers
CONTAIN = 0.85          # share of a content unit's box that must lie inside its container
GROWTH = 1.15           # a container's area is at least this x the unit's it holds
PAD_STEPS = (1.0, 0.5, 0.25, 0.0)  # padding tried, x the source's (scaled with the text)
EDGE_TOL = 0.01         # share of the canvas side: a container this close to an edge bled off it
INNER_ROUNDS = 3        # repair rounds for the layout inside a container
HELD_SHARE = 0.5        # (gate) a unit at least this much inside its container's box is still held
FIT_SLACK = 1.2         # a fixed-aspect container is kept if its fit reaches this x the scale its texts need ...
FIT_AREA = 0.82         # ... fitted into this share of each canvas side (the content area inside v2's margins)
_LAYER_ROLES = ("background", "decoration")


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def _area(b: Box) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def _paint_rank(elements: list[dict]) -> dict[str, int]:
    """element id -> paint position (z_index, then list order, as cr-renderer draws)."""
    order = sorted(range(len(elements)), key=lambda i: elements[i].get("z_index") or 0)
    return {elements[i]["id"]: r for r, i in enumerate(order)}


def background_stack(elements: list[dict], width: float, height: float) -> set[str]:
    """Ids of the bottom stack of full-bleed layers: images or shapes, not
    rotated, covering at least FULL_BLEED of the canvas, with nothing but
    other such layers painted under them where they overlap."""
    canvas = (0.0, 0.0, float(width), float(height))
    rank = _paint_rank(elements)
    stack: set[str] = set()
    below: list[dict] = []
    for el in sorted(elements, key=lambda e: rank[e["id"]]):
        box = rotated_aabb(el)
        full = (el.get("type") in ("image", "shape") and not el.get("rotation")
                and overlap_area(box, canvas) >= FULL_BLEED * width * height)
        if full and all(o["id"] in stack for o in below if overlap_area(box, rotated_aabb(o)) > 0):
            stack.add(el["id"])
        below.append(el)
    return stack


@dataclass
class Group:
    """A container and the content units it holds (some may be groups themselves)."""

    uid: str
    container: str                  # element id
    units: list[Unit]               # what it holds, in rank order (original units or nested groups)
    stretch: bool                   # rectangle shape / cover image: wraps its contents
    box: Box                        # container's source box
    pads: tuple[float, float, float, float]  # source px: container edge to the contents' union (l, t, r, b)
    bleed: tuple[bool, bool, bool, bool]     # touched the canvas edge in the source (l, t, r, b)
    f_src: float | None             # largest font among everything it holds
    content_ids: list[str] = field(default_factory=list)  # original content unit ids inside (flattened)


@dataclass
class Layers:
    background: list[str]           # element ids of the background layer
    groups: list[Group]             # every group, nested ones included
    plan: PriorityPlan              # the layout plan: top-level groups replace their members
    by_uid: dict[str, Group] = field(default_factory=dict)

    def top_level(self) -> list[Group]:
        top = {u.unit_id for u in self.plan.units if u.kind == KIND}
        return [g for g in self.groups if g.uid in top]

    def content_ids(self, uid: str) -> list[str]:
        g = self.by_uid.get(uid)
        return list(g.content_ids) if g else [uid]


def _unit_box(els: dict, unit: Unit) -> Box:
    return union_box([rotated_aabb(els[i]) for i in unit.element_ids])


def _covers(cbox: Box, ubox: Box) -> bool:
    return (cbox[0] <= ubox[0] and cbox[1] <= ubox[1]
            and ubox[0] + ubox[2] <= cbox[0] + cbox[2] and ubox[1] + ubox[3] <= cbox[1] + cbox[3])


def _holds(cbox: Box, ubox: Box) -> bool:
    return overlap_area(cbox, ubox) >= CONTAIN * _area(ubox) and _area(cbox) >= GROWTH * _area(ubox) > 0


def detect(source: dict, plan: PriorityPlan, target: dict | None = None) -> Layers:
    """The source's layers and the layout plan that keeps them (see the module
    docstring). With `target`, a fixed-aspect container that can't hold its
    contents at their floors on that canvas (a frame round the whole ad, on a
    728x90 strip) is not grouped: its contents go back to being units of
    their own (or join the next container out)."""
    excluded: set[str] = set()
    while True:
        layers = _detect(source, plan, excluded)
        bad = [g for g in layers.groups if target is not None and not _fits(g, layers, plan, source, target)]
        if not bad:
            return layers
        excluded |= {g.container for g in bad}


def _fits(g: Group, layers: Layers, plan: PriorityPlan, source: dict, target: dict) -> bool:
    """Can a fixed-aspect container, fitted into the content area (FIT_AREA of each
    side), hold its texts at FIT_SLACK x their floors (scaling with it)? A
    container too small to read there is better broken up. Stretching
    containers always can try."""
    from engine.pipeline.layout.place import floor_px

    if g.stretch:
        return True
    W, H = float(target["width"]), float(target["height"])
    k_fit = min(FIT_AREA * W / g.box[2], FIT_AREA * H / g.box[3])
    els = {e["id"]: e for e in source["elements"]}
    by_unit = {u.unit_id: u for u in plan.units}
    need = 0.0
    for cid in g.content_ids:
        u = by_unit.get(cid)
        if u is None:
            continue
        for i in u.element_ids:
            if els[i]["type"] == "text" and els[i]["font_size"] > 0:
                need = max(need, floor_px(u.role, W, H) / els[i]["font_size"])
    return k_fit >= FIT_SLACK * need


def _detect(source: dict, plan: PriorityPlan, excluded: set[str]) -> Layers:
    elements = source["elements"]
    els = {e["id"]: e for e in elements}
    W, H = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    rank = _paint_rank(elements)
    stack = background_stack(elements, W, H)
    background = [e["id"] for e in elements if e["id"] in stack]

    def in_stack(u: Unit) -> bool:
        return u.role == "background" or all(i in stack for i in u.element_ids)

    content = [u for u in plan.units if u.element_ids and u.role not in _LAYER_ROLES and not in_stack(u)]
    holders = [u for u in plan.units
               if len(u.element_ids) == 1 and u.kind == "single" and not in_stack(u)
               and els[u.element_ids[0]]["type"] in ("shape", "image")
               and not els[u.element_ids[0]].get("rotation")
               and els[u.element_ids[0]]["width"] > 0 and els[u.element_ids[0]]["height"] > 0
               and u.element_ids[0] not in excluded]
    hbox = {u.unit_id: rotated_aabb(els[u.element_ids[0]]) for u in holders}
    hrank = {u.unit_id: rank[u.element_ids[0]] for u in holders}

    def holder_of(box: Box, first_paint: int, skip: str) -> str | None:
        best = None
        for h in holders:
            # under it, or an image frame drawn over it (see-through: the designer meant it to show)
            # (a frame drawn over it must hold all of it, as `source_cover_pairs` lets G9 accept)
            under = hrank[h.unit_id] < first_paint
            frame = els[h.element_ids[0]]["type"] == "image" and _covers(hbox[h.unit_id], box)
            if h.unit_id == skip or not (under or frame) or not _holds(hbox[h.unit_id], box):
                continue
            if best is None or _area(hbox[h.unit_id]) < _area(hbox[best]):
                best = h.unit_id
        return best

    parent: dict[str, str] = {}   # content unit id or holder unit id -> holder unit id
    for u in content:
        p = holder_of(_unit_box(els, u), min(rank[i] for i in u.element_ids), u.unit_id)
        if p is not None:
            parent[u.unit_id] = p
    holding = set(parent.values())
    changed = True
    while changed:  # holders inside holders (a card inside a frame), only for holders that hold something
        changed = False
        for h in holders:
            if h.unit_id in holding and h.unit_id not in parent:
                p = holder_of(hbox[h.unit_id], hrank[h.unit_id], h.unit_id)
                if p is not None and not _ancestor(parent, p, h.unit_id):
                    parent[h.unit_id] = p
                    if p not in holding:
                        holding.add(p)
                        changed = True

    by_unit = {u.unit_id: u for u in plan.units}
    children: dict[str, list[str]] = {}
    for child, p in parent.items():
        children.setdefault(p, []).append(child)
    groups: dict[str, Group] = {}

    def build(hid: str) -> Group:
        if hid in groups:
            return groups[hid]
        held: list[Unit] = []
        flat_of: dict[str, list[str]] = {}
        for cid in sorted(children.get(hid, [])):
            if cid in children:  # a container held by this one: its whole group goes in
                sub = build(cid)
                held.append(_group_unit(sub))
                flat_of[sub.uid] = sub.content_ids
            else:
                held.append(by_unit[cid])
        held.sort(key=lambda u: (u.rank if u.rank is not None else math.inf, u.unit_id))
        cid = by_unit[hid].element_ids[0]
        c = els[cid]
        members = [els[i] for u in held for i in u.element_ids]
        cb = hbox[hid]
        ub = union_box([rotated_aabb(m) for m in members])
        pads = (max(0.0, ub[0] - cb[0]), max(0.0, ub[1] - cb[1]),
                max(0.0, cb[0] + cb[2] - ub[0] - ub[2]), max(0.0, cb[1] + cb[3] - ub[1] - ub[3]))
        tol_x, tol_y = EDGE_TOL * W, EDGE_TOL * H
        bleed = (cb[0] <= tol_x, cb[1] <= tol_y, cb[0] + cb[2] >= W - tol_x, cb[1] + cb[3] >= H - tol_y)
        fonts = [m["font_size"] for m in members if m["type"] == "text"]
        stretch = (c["type"] == "shape" and c.get("shape_type", "rectangle") == "rectangle") or \
                  (c["type"] == "image" and c.get("fit") == "cover")
        flat = []
        for u in held:
            flat += flat_of.get(u.unit_id, [u.unit_id])
        g = Group(f"layers:{cid}", cid, held, stretch, cb, pads, bleed, max(fonts) if fonts else None, flat)
        groups[hid] = g
        return g

    tops = sorted(h for h in holding if h not in parent)
    for h in sorted(holding):
        build(h)
    top_units = {h: _group_unit(groups[h]) for h in tops}
    absorbed = set(parent) | set(holding)
    units = []
    for u in plan.units:
        if u.unit_id in top_units:
            units.append(top_units[u.unit_id])
        elif u.unit_id not in absorbed:
            if in_stack(u) and u.role != "background":
                u = Unit(u.unit_id, list(u.element_ids), u.kind, "background", None, list(u.flags) + ["background"])
            units.append(u)
    layout = PriorityPlan(plan.source, units, plan.stats)
    out = Layers(background, list(groups.values()), layout)
    out.by_uid = {g.uid: g for g in groups.values()}
    return out


def _ancestor(parent: dict, node: str, of: str) -> bool:
    """Is `of` an ancestor of `node` (or `node` itself)? Guards against cycles."""
    seen = set()
    while node is not None and node not in seen:
        if node == of:
            return True
        seen.add(node)
        node = parent.get(node)
    return False


def _group_unit(g: Group) -> Unit:
    """The layout-plan unit for a group: the container and everything it holds;
    its role and rank are those of the most important unit inside."""
    ids = [g.container] + [i for u in g.units for i in u.element_ids]
    top = g.units[0]
    flags = sorted({f for u in g.units for f in u.flags if f != "background"})
    return Unit(g.uid, ids, KIND, top.role, top.rank, flags)


def remap_degraded(layers: Layers, degraded: list[str], design: dict, width: float, height: float) -> list[str]:
    """Degraded layout-plan unit ids -> original plan unit ids. A degraded
    group names the units inside it whose text ended below its floor."""
    from engine.pipeline.layout.place import floor_px

    if not degraded:
        return []
    els = {e["id"]: e for e in design["elements"]}
    out = []
    for uid in degraded:
        g = layers.by_uid.get(uid)
        if g is None:
            out.append(uid)
            continue
        for cid in g.content_ids:
            unit = _find_unit(g, cid, layers)
            if unit is None:
                continue
            sizes = [els[i]["font_size"] for i in unit.element_ids if i in els and els[i]["type"] == "text"]
            if sizes and min(sizes) < floor_px(unit.role, width, height) - 1e-6:
                out.append(cid)
    return list(dict.fromkeys(out))


def _find_unit(g: Group, uid: str, layers: Layers) -> Unit | None:
    for u in g.units:
        if u.unit_id == uid:
            return u
        if u.kind == KIND:
            hit = _find_unit(layers.by_uid[u.unit_id], uid, layers)
            if hit is not None:
                return hit
    return None


# ---------------------------------------------------------------- bisection hooks

def _k_range(model, g: Group) -> tuple[float, float]:
    """(scale that puts the contents' texts at their floors, natural scale) vs the source."""
    lo = 0.0
    for u in g.units:
        sizes = [model.els[i]["font_size"] for i in u.element_ids if model.els[i]["type"] == "text"]
        if sizes and u.kind != KIND:
            lo = max(lo, model.floor(u) / max(sizes))
        elif sizes:
            lo = max(lo, _k_range(model, model.layers.by_uid[u.unit_id])[0])
    from engine.pipeline.layout.place import MAX_TEXT_GROWTH
    lo = min(lo, MAX_TEXT_GROWTH) if lo else 0.0
    hi = max(lo, min(max(model.s_area, 1e-6), MAX_TEXT_GROWTH)) if lo else max(model.s_area, 1e-6)
    return lo, hi


def _sub_items(model, g: Group) -> list:
    """Bisection items for the units inside `g`, in source reading order."""
    from engine.pipeline.layout.strategies.bisection import ROLE_BOOST, _Item
    items = []
    for k, u in enumerate(g.units):
        x, y, w, h = _unit_box(model.els, u)
        order = (y + h / 2) / model.sh + 0.25 * (x + w / 2) / model.sw
        weight = (u.rank ** -ALPHA) if u.rank else 0.1
        value = math.sqrt(max(model.need(u), 1e-9)) * weight * ROLE_BOOST.get(u.role, 1.0)
        items.append(_Item([u], -1000 - k, order, u.rank if u.rank is not None else 10 ** 6, max(value, 1e-6)))
    items.sort(key=lambda i: (i.order, i.rank))
    return items


def group_ext(model, unit: Unit, axis: str, cross: float, f0: float) -> tuple[float, float]:
    """(least, most) extent along `axis` for a group, the cross side fixed: the
    contents' own needs (inside the container, padding compressed to the
    smallest step) plus the padding; a fixed-aspect container by its aspect."""
    g = model.layers.by_uid[unit.unit_id]
    k_lo, k_hi = _k_range(model, g)
    base = model.floor(unit)
    if f0 < base and base > 0:  # sized at the hard floor: the contents may go that small too
        k_lo *= f0 / base
    other = "y" if axis == "x" else "x"
    l, t, r, b = g.pads
    pad_along, pad_cross = (l + r, t + b) if axis == "x" else (t + b, l + r)
    along_src, cross_src = (g.box[2], g.box[3]) if axis == "x" else (g.box[3], g.box[2])
    if not g.stretch:
        k_max = cross / cross_src if cross_src > 0 else 0.0
        lo_k = max(k_lo * 1.05, 1e-6)
        if lo_k > k_max:
            return math.inf, math.inf
        return along_src * lo_k, along_src * k_max  # like an image unit, it can use all the room it gets
    p_min = PAD_STEPS[-2]
    inner_cross = cross - pad_cross * k_lo * p_min
    if inner_cross < 1:
        return math.inf, math.inf
    items = _sub_items(model, g)
    length = model.W if axis == "x" else model.H
    from engine.pipeline.layout.strategies.bisection import GUTTER
    gutter = GUTTER * min(model.W, model.H)
    if len(items) == 1:
        lo, hi = model.item_ext(items[0], axis, inner_cross)
    else:
        span = model.span(items, axis, inner_cross, length, gutter)
        lo, hi = span.lo, span.hi
        if lo <= 0:
            return math.inf, math.inf
    if not math.isfinite(lo):
        return math.inf, math.inf
    lo += pad_along * k_lo * p_min
    hi = (hi if math.isfinite(hi) else length) + pad_along * max(k_lo, k_hi)
    return lo, max(lo, hi)


def _inset(box: Box, pads, k: float) -> Box:
    l, t, r, b = (p * k for p in pads)
    return (box[0] + l, box[1] + t, box[2] - l - r, box[3] - t - b)


def _inside(b, model, g: Group, inner: Box, caps: dict):
    """Lay the group's contents out inside `inner` with bisection (a few repair
    rounds). (geometry, degraded, text scale vs the source) or None."""
    from engine.pipeline.layout.strategies import bisection as bs
    if inner[2] < 1 or inner[3] < 1:
        return None
    items = _sub_items(model, g)
    gutter = bs.GUTTER * min(model.W, model.H)
    best = None
    for _ in range(INNER_ROUNDS):
        bs._layout(model, items, inner, "rank", "long", "x" if inner[2] > inner[3] else "y", gutter)
        inner_caps = dict(caps)
        geo: dict[str, dict] = {}
        missed = set()
        for item in sorted(items, key=lambda i: (i.rank, i.key)):
            for u, ugeo, degraded in bs._place_item(b, model, item, inner_caps):
                if ugeo is None:
                    return best
                geo.update(ugeo)
                if degraded:
                    missed.add(item.key)
                if u.role == "headline" and not degraded and "headline" not in inner_caps:
                    sizes = [e["font_size"] for e in ugeo.values() if e.get("type") == "text"]
                    if sizes:
                        inner_caps["headline"] = max(sizes)
        fonts = [e["font_size"] for e in geo.values() if e.get("type") == "text"]
        k = max(fonts) / g.f_src if fonts and g.f_src else None
        best = (geo, bool(missed), k)
        if not missed:
            return best
        for item in items:
            if item.key in missed:
                item.boost *= bs.BOOST
    return best


def _scaled_container(c: dict, box: Box, k: float) -> dict:
    out = dict(c, x=box[0], y=box[1], width=box[2], height=box[3])
    for key in ("corner_radius", "stroke_width"):
        if c.get(key):
            out[key] = c[key] * k
    return out


def place_group(b, model, unit: Unit, box: Box, caps: dict):
    """(unit, geometry, degraded) for a group in its cell (see the module
    docstring); if its contents can't be laid out inside the container, the
    whole group scaled rigidly as a last resort (degraded), so it stays whole."""
    out = _place_group(b, model, unit, box, caps)
    if out[1] is None and box[2] >= 1 and box[3] >= 1:
        from engine.pipeline.layout.place import HARD_FLOOR, place_unit
        geo = place_unit(unit, model.els, box, model.m, HARD_FLOOR)
        if geo is not None:
            return unit, geo, True
    return out


def _place_group(b, model, unit: Unit, box: Box, caps: dict):
    g = model.layers.by_uid[unit.unit_id]
    c = model.els[g.container]
    x, y, w, h = box
    if w < 1 or h < 1:
        return unit, None, True
    k_lo, k_hi = _k_range(model, g)
    best = None
    if g.stretch:
        k0 = min(w / g.box[2], h / g.box[3])
        k0 = min(max(k0, k_lo), max(k_lo, k_hi)) if k_lo else k0
        for p in PAD_STEPS:
            k, res = k0, None
            for _ in range(2):  # the padding follows the text's scale: one correction once it is known
                res = _inside(b, model, g, _inset(box, g.pads, k * p), caps)
                if res is None or res[2] is None or abs(res[2] - k) <= 0.1 * k:
                    break
                k = res[2]
            if res is None:
                continue
            if best is None or (best[1] and not res[1]):
                best = (res[0], res[1], k * p, k)
            if not res[1]:
                break
        if best is None:
            return unit, None, True
        geo, degraded, kp, k = best
        ub = union_box([rotated_aabb(e) for e in geo.values()])
        l, t, r, bt = (pd * kp for pd in g.pads)
        x0, y0 = max(x, ub[0] - l), max(y, ub[1] - t)
        x1, y1 = min(x + w, ub[0] + ub[2] + r), min(y + h, ub[1] + ub[3] + bt)
        edge = _content_edges(model)
        if g.bleed[0] and x <= edge[0] + 0.5:
            x0 = 0.0
        if g.bleed[1] and y <= edge[1] + 0.5:
            y0 = 0.0
        if g.bleed[2] and x + w >= model.W - edge[0] - 0.5:
            x1 = model.W
        if g.bleed[3] and y + h >= model.H - edge[1] - 0.5:
            y1 = model.H
        geo[g.container] = _scaled_container(c, (x0, y0, x1 - x0, y1 - y0), k)
        return unit, geo, degraded
    cx, cy, cw, ch = fit_aspect(g.box[2] / g.box[3], box)
    kc = cw / g.box[2]
    for p in PAD_STEPS[:3]:
        res = _inside(b, model, g, _inset((cx, cy, cw, ch), g.pads, kc * p), caps)
        if res is None:
            continue
        if best is None or (best[1] and not res[1]):
            best = res
        if not res[1]:
            break
    if best is None:
        return unit, None, True
    geo, degraded, _ = best
    geo[g.container] = _scaled_container(c, (cx, cy, cw, ch), kc)
    return unit, geo, degraded


def _content_edges(model) -> tuple[float, float]:
    """The widest margins bisection uses (px, x and y): a cell this close to the
    canvas edge is on the content area's edge."""
    from engine.pipeline.layout.strategies import bisection as bs
    short = min(model.W, model.H)
    m = max(bs.V2_MARGINS + bs.MARGINS) * short
    return max(m, bs.V2_SIDE_MARGIN * model.W), max(m, bs.V2_SIDE_MARGIN * model.H)


# ---------------------------------------------------------------- stats

def profile(source: dict, plan: PriorityPlan) -> dict:
    """Layer counts for one source (aggregate stats, no content): background
    layer kinds, groups by container kind, units held."""
    layers = detect(source, plan)
    els = {e["id"]: e for e in source["elements"]}
    bg = [els[i] for i in layers.background]
    tops = layers.top_level()
    return {
        "background_layers": len(bg),
        "background_image": any(e["type"] == "image" for e in bg),
        "background_cover_image": any(e["type"] == "image" and e.get("fit") == "cover" for e in bg),
        "groups": len(tops),
        "nested_groups": len(layers.groups) - len(tops),
        "stretch_groups": sum(g.stretch for g in tops),
        "fixed_groups": sum(not g.stretch for g in tops),
        "image_containers": sum(els[g.container]["type"] == "image" for g in layers.groups),
        "units_held": sum(len(g.content_ids) for g in tops),
        "content_units": sum(1 for u in plan.units if u.role not in _LAYER_ROLES),
    }


# ---------------------------------------------------------------- gates (CHAMELEON_LAYERS only)

def source_cover_pairs(source: dict) -> frozenset:
    """(element id, image id) pairs where the source already draws an opaque-counted
    image over the whole element (a frame raster round a photo or a line of
    copy): the designer meant it to show through, so G9 lets that pair be."""
    from engine.pipeline.occlusion import contains, opaque, paint_order

    elements = source["elements"]
    order = paint_order(elements)
    out = set()
    for c in elements:
        if c["type"] != "image" or not opaque(c):
            continue
        cb = rotated_aabb(c)
        for e in elements:
            if e is not c and order[id(c)] > order[id(e)] and contains(cb, rotated_aabb(e)):
                out.add((e["id"], c["id"]))
    return frozenset(out)


def broken_containers(source: dict, design: dict, plan: PriorityPlan, target: dict) -> list[str]:
    """Containers kept as groups for this target (see `detect`) that no longer
    hold every unit they held in the source (each at least HELD_SHARE inside)."""
    els = {e["id"]: e for e in design.get("elements") or [] if isinstance(e, dict) and "id" in e}
    out = []
    for g in detect(source, plan, target).groups:
        if g.container not in els:
            continue
        try:
            cb = rotated_aabb(els[g.container])
            for u in g.units:
                ub = union_box([rotated_aabb(els[i]) for i in u.element_ids])
                if overlap_area(cb, ub) < HELD_SHARE * _area(ub):
                    out.append(g.container)
                    break
        except (KeyError, TypeError):
            continue
    return out
