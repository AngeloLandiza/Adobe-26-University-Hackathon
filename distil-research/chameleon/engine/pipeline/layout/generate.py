"""Node 2's candidate generator (docs/pipeline/02-resize-engine.md §4-6).

For the target's aspect class it tries every template x parameter set:
units are assigned to slots in rank order, placed with `place_unit`, and
anything that doesn't fit goes to the leftover free space at a small size
(marked degraded). A proportional "reflow" of the source layout is tried
first. Every candidate is finalised; ones that can't be built are dropped.
"""

import math

from engine.pipeline.geometry import Box, align_in, overlap_area, rotated_aabb, union_box
from engine.pipeline.layout import flex, retrieval, strategy_registry
from engine.pipeline.layout.aspect import aspect_class
from engine.pipeline.layout.layers import fit_inside, place_background, place_decoration
from engine.pipeline.layout.normalize import finalize
from engine.pipeline.layout.place import CONDENSED_PADDING, CONDENSED_VPAD, HARD_FLOOR, floor_px, place_unit
from engine.pipeline.layout.templates import Slot, build_slots, param_grid, templates_for
from engine.pipeline.occlusion import contains, is_covered, paint_order
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Candidate, PriorityPlan, Unit

LOGO_MAX_GROWTH = 1.5   # a logo never grows past 1.5x its source size
FREE_MAX_SHARE = 1 / 3  # a leftover image gets at most a third of the canvas each way
FREE_TRIES = 12         # distinct empty rectangles tried per leftover unit, largest first
FLOW_GAP = 0.02         # space between minor units, as a share of the canvas's short side
SAME_ASPECT_TOL = 0.02  # target aspect within 2% of the source's: the proportional layout is the answer
LEFTOVER_FLOORS = (HARD_FLOOR, 4, 2, 1)  # px; a leftover goes below the hard floor only if nothing else fits
BACKDROP_TOL = 1.0      # px of slack when deciding that a source shape holds a text
# Text roles stack (or flow) one by one; only other roles cluster into a row.
_NO_CLUSTER = {"headline", "subheadline", "other_text", "legal", None}


def assign(plan: PriorityPlan, slots: list[Slot], cutoff: int | None) -> dict[int, list[list[Unit]]]:
    """Slot index -> items, where an item is a unit or a cluster of
    consecutive same-role units. Units take the first open slot that accepts
    their role, in rank order; with a cutoff, units ranked below it go to the
    minor slot, except a CTA."""
    items: list[list[Unit]] = []
    for unit in plan.ranked_units():
        if items and unit.role not in _NO_CLUSTER and items[-1][-1].role == unit.role:
            items[-1].append(unit)
        else:
            items.append([unit])

    minor = next(i for i, s in enumerate(slots) if s.minor)
    taken: set[int] = set()
    out: dict[int, list[list[Unit]]] = {}
    for item in items:
        role, rank = item[0].role, item[0].rank
        target = minor
        if cutoff is None or rank <= cutoff or role == "cta":
            target = next((i for i, s in enumerate(slots)
                           if not s.minor and role in s.accepts and (s.stack or i not in taken)), minor)
        if not slots[target].stack and not slots[target].minor:
            taken.add(target)
        out.setdefault(target, []).append(item)
    return out


def _contains(outer: Box, inner: Box) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and inner[0] + inner[2] <= outer[0] + outer[2] and inner[1] + inner[3] <= outer[1] + outer[3])


def _candidate_id(template: str, params: dict) -> str:
    return f"{template}/" + "/".join(f"{k}={v}" for k, v in sorted(params.items()))


class _Builder:
    """Places one candidate's units, tracking what's placed, the content
    boxes already taken, and which units ended up degraded."""

    def __init__(self, source: dict, target: dict, plan: PriorityPlan, measurer: TextMeasurer):
        self.source, self.plan, self.measurer = source, plan, measurer
        self.width, self.height = target["width"], target["height"]
        self.els = {el["id"]: el for el in source["elements"]}
        self._source_index = {el["id"]: i for i, el in enumerate(source["elements"])}
        self.placed: dict[str, dict] = {}
        self.boxes: list[Box] = []
        self.degraded: list[str] = []
        self.leftover: list[Unit] = []

    def _members(self, unit: Unit) -> list[dict]:
        return [self.els[i] for i in unit.element_ids]

    def _cap_logo(self, unit: Unit, box: Box) -> Box:
        if unit.role != "logo":
            return box
        _, _, uw, uh = union_box([rotated_aabb(m) for m in self._members(unit)])
        w, h = min(box[2], uw * LOGO_MAX_GROWTH), min(box[3], uh * LOGO_MAX_GROWTH)
        return (*align_in(box, w, h), w, h)

    def _record(self, unit: Unit, geometry: dict[str, dict], degraded: bool) -> None:
        self.placed.update(geometry)
        self.boxes.append(union_box([rotated_aabb(g) for g in geometry.values()]))
        if degraded:
            self.degraded.append(unit.unit_id)

    def try_place(self, unit: Unit, box: Box, minor: bool, align: str = "center",
                  scale: float | None = None) -> tuple[dict[str, dict] | None, bool]:
        """Place at the role's floor, else at the hard floor (degraded). Minor
        units are capped at their floor size; with `scale`, text is capped at
        scale x its source size (never below the floor)."""
        floor = floor_px(unit.role, self.width, self.height)
        max_font = floor if minor else None
        if scale is not None and not minor:
            sizes = [m["font_size"] for m in self._members(unit) if m["type"] == "text"]
            max_font = max(scale * max(sizes), floor) if sizes else None
        box = self._cap_logo(unit, box)
        geometry = place_unit(unit, self.els, box, self.measurer, floor, align=align, max_font=max_font)
        if geometry is not None:
            return geometry, False
        return place_unit(unit, self.els, box, self.measurer, HARD_FLOOR, align=align, max_font=max_font), True

    def place_row(self, units: list[Unit], box: Box, minor: bool, align: str = "center") -> None:
        """Units side by side in equal-width cells. If some can't reach their
        floor in one line, the row wraps onto as few lines as lets every unit
        reach it (four badges in a half-width slot); if none does, it stays
        on one line."""
        x, y, w, h = box
        first = None
        for lines in range(1, len(units) + 1):
            per_line = math.ceil(len(units) / lines)
            cw, ch = w / per_line, h / lines
            placed = [self.try_place(unit, (x + (i % per_line) * cw, y + (i // per_line) * ch, cw, ch), minor, align)
                      for i, unit in enumerate(units)]
            first = first or placed
            if all(geometry is not None and not degraded for geometry, degraded in placed):
                break
        else:
            placed = first
        for unit, (geometry, degraded) in zip(units, placed):
            if geometry is None:
                self.leftover.append(unit)
            else:
                self._record(unit, geometry, degraded)

    def place_flow(self, units: list[Unit], box: Box) -> None:
        """The minor zone: units left to right at their natural width (text on
        one line where it fits), each at most its floor size. Whatever doesn't
        fit waits for the free-space pass."""
        x, y, w, h = box
        gap = FLOW_GAP * min(self.width, self.height)
        cursor = x
        for unit in units:
            room = x + w - cursor
            members = self._members(unit)
            if room <= 0:
                self.leftover.append(unit)
                continue
            if len(members) == 1 and members[0]["type"] == "text":
                floor = floor_px(unit.role, self.width, self.height)
                text = dict(members[0], font_size=floor)
                room = min(room, max(self.measurer.line_widths(text, floor, float("inf"))))
            geometry, degraded = self.try_place(unit, (cursor, y, room, h), minor=True, align="left")
            if geometry is None:
                self.leftover.append(unit)
                continue
            self._record(unit, geometry, degraded)
            ux, _, uw, _ = self.boxes[-1]
            cursor = ux + uw + gap

    def _min_height(self, unit: Unit, w: float, h: float, floor: float) -> float:
        """Height the unit needs in a w-wide column with its text at `floor`."""
        geometry = place_unit(unit, self.els, (0, 0, w, h), self.measurer, floor, max_font=floor)
        return math.inf if geometry is None else union_box([rotated_aabb(g) for g in geometry.values()])[3]

    def place_stack(self, units: list[Unit], box: Box) -> None:
        """Units top to bottom, in rank order. Each first gets the height it
        needs at its role's floor; while they don't all fit, the lowest-ranked
        units drop to the hard floor (degraded), and then out to the free-space
        pass. Spare height is shared by priority weight, and the placed blocks
        are re-spaced with equal gaps."""
        x, y, w, h = box
        need = {u.unit_id: self._min_height(u, w, h, floor_px(u.role, self.width, self.height)) for u in units}
        kept = list(units)
        for unit in sorted(units, key=lambda u: -u.rank):
            if sum(need[u.unit_id] for u in kept) <= h:
                break
            need[unit.unit_id] = self._min_height(unit, w, h, HARD_FLOOR)
        for unit in sorted(units, key=lambda u: -u.rank):
            if sum(need[u.unit_id] for u in kept) <= h:
                break
            kept.remove(unit)
            self.leftover.append(unit)

        weights = {u.unit_id: self.plan.weight(u.unit_id) for u in kept}
        spare = h - sum(need[u.unit_id] for u in kept)
        total_weight = sum(weights.values()) or 1.0
        stacked, top = [], y
        for unit in kept:
            sub_h = need[unit.unit_id] + spare * weights[unit.unit_id] / total_weight
            geometry, degraded = self.try_place(unit, (x, top, w, sub_h), minor=False)
            top += sub_h
            if geometry is None:
                self.leftover.append(unit)
            else:
                stacked.append((unit, geometry, degraded))
        used = sum(union_box([rotated_aabb(g) for g in geo.values()])[3] for _, geo, _ in stacked)
        gap = (h - used) / (len(stacked) + 1) if stacked else 0.0
        cursor = y + gap
        for unit, geometry, degraded in stacked:
            bx, by, bw, bh = union_box([rotated_aabb(g) for g in geometry.values()])
            shifted = {eid: dict(g, y=g["y"] + cursor - by) for eid, g in geometry.items()}
            self._record(unit, shifted, degraded)
            cursor += bh + gap

    def free_rects(self, margin: float) -> list[Box]:
        """The largest empty rectangles left on the canvas: a coarse occupancy
        grid (placed content plus a gap, and the margins), scanned row by row
        with the largest-rectangle-in-histogram method."""
        cell = max(4.0, min(self.width, self.height) / 30)
        cols, rows = int(self.width // cell), int(self.height // cell)
        if cols == 0 or rows == 0:
            return []
        busy = [[False] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                x0, y0 = c * cell, r * cell
                if x0 < margin or y0 < margin or x0 + cell > self.width - margin or y0 + cell > self.height - margin:
                    busy[r][c] = True
        for bx, by, bw, bh in self.boxes:
            gap = cell / 4
            c0, c1 = max(0, int((bx - gap) // cell)), min(cols - 1, int((bx + bw + gap) // cell))
            r0, r1 = max(0, int((by - gap) // cell)), min(rows - 1, int((by + bh + gap) // cell))
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    busy[r][c] = True

        found: dict[Box, float] = {}
        heights = [0] * cols
        for r in range(rows):
            heights = [0 if busy[r][c] else heights[c] + 1 for c in range(cols)]
            stack: list[int] = []
            for c in range(cols + 1):
                h = heights[c] if c < cols else 0
                while stack and heights[stack[-1]] >= h:
                    top_h = heights[stack.pop()]
                    left = stack[-1] + 1 if stack else 0
                    if top_h:
                        rect = (left * cell, (r - top_h + 1) * cell, (c - left) * cell, top_h * cell)
                        found[rect] = rect[2] * rect[3]
                stack.append(c)
        # Largest first, skipping any rectangle inside one already kept, so the
        # tries cover different regions rather than shrinking copies of one.
        distinct: list[Box] = []
        for rect in sorted(found, key=lambda r: (-found[r], r)):
            if not any(_contains(kept, rect) for kept in distinct):
                distinct.append(rect)
                if len(distinct) == FREE_TRIES:
                    break
        return distinct

    def _footprint(self, unit: Unit) -> float:
        """Area of a unit's smallest form at the hard floor: a text on one
        line, a pill condensed around its label, any other group shrunk to
        that text size. Images can shrink freely, so 0."""
        members = self._members(unit)
        texts = [m for m in members if m["type"] == "text"]
        if not texts:
            return 0.0
        if len(members) == 1 or unit.kind == "container_label":
            text = dict(texts[0], font_size=HARD_FLOOR)
            width = max(self.measurer.line_widths(text, HARD_FLOOR, float("inf")))
            height = self.measurer.block_height(text, HARD_FLOOR, width)
            if unit.kind == "container_label":
                width += 2 * CONDENSED_PADDING * HARD_FLOOR
                height += 2 * CONDENSED_VPAD * HARD_FLOOR
            return width * height
        _, _, uw, uh = union_box([rotated_aabb(m) for m in members])
        return uw * uh * (HARD_FLOOR / max(t["font_size"] for t in texts)) ** 2

    def place_leftovers(self, margin: float) -> bool:
        """Put each leftover unit into the largest empty rectangle it fits
        (images capped in size, text at most its floor size), biggest
        footprint first so small units don't take the only space a long
        text fits. Text that fits nowhere at the hard floor tries smaller
        sizes (LEFTOVER_FLOORS): a dense design in a small banner keeps its
        top units readable instead of losing the whole template to the
        fallback.

        If that pass fails, the banner is crowded: long low-priority text
        filled every gap before a pill like the CTA got one. The pass is then
        redone in rank order, so the important units get room first. False
        if a unit can't be placed even then."""
        saved = (dict(self.placed), list(self.boxes), list(self.degraded))
        orders = (lambda u: (-self._footprint(u), u.rank),
                  lambda u: (u.rank if u.rank is not None else math.inf, u.unit_id))
        for order in orders:
            self.placed, self.boxes, self.degraded = dict(saved[0]), list(saved[1]), list(saved[2])
            if all(self._place_leftover(unit, margin) for unit in sorted(self.leftover, key=order)):
                self.leftover = []
                return True
        return False

    def _place_leftover(self, unit: Unit, margin: float) -> bool:
        has_image = any(m["type"] == "image" for m in self._members(unit))
        floor = floor_px(unit.role, self.width, self.height)
        rects = self.free_rects(margin)
        for text_floor in LEFTOVER_FLOORS:
            for rect in rects:
                box = rect
                if has_image:  # text and pills are already held to their floor size by max_font
                    w, h = min(rect[2], self.width * FREE_MAX_SHARE), min(rect[3], self.height * FREE_MAX_SHARE)
                    box = (*align_in(rect, w, h), w, h)
                geometry = place_unit(unit, self.els, box, self.measurer, text_floor, max_font=floor)
                if geometry is not None:
                    self._record(unit, geometry, degraded=True)
                    return True
        return False

    def place_layers(self) -> None:
        for unit in self.plan.units:
            if unit.role == "background":
                self.placed.update((m["id"], place_background(m, self.width, self.height)) for m in self._members(unit))
            elif unit.role == "decoration":
                for m in self._members(unit):
                    self.placed[m["id"]] = self._backdrop(m) or self._visible_decoration(m)

    def _ink_box(self, el: dict) -> Box:
        """Where a text's glyphs are drawn: its widest line, placed per
        text_align, and its measured block. A placed text box spans its whole
        slot, which is much wider than the words in it."""
        if el.get("rotation"):
            return rotated_aabb(el)
        width = max(self.measurer.line_widths(el, el["font_size"], el["width"]), default=0.0)
        width = min(width, el["width"])
        align = el.get("text_align") or "left"
        x = el["x"] + (el["width"] - width) / 2 if align == "center" else             el["x"] + el["width"] - width if align == "right" else el["x"]
        return (x, el["y"], width, min(el["height"], self.measurer.block_height(el, el["font_size"], el["width"])))

    def _backdrop(self, shape: dict) -> dict | None:
        """A shape the source paints text on (a panel behind copy, a card under
        a headline) follows that text: its new box is the text's new ink
        bounds plus the source padding, scaled with the text (or no padding,
        if the padded box would overlap other content). None, so it's parked
        like any decoration, if it holds no text, is rotated, or would still
        sit over content painted before it."""
        if shape["type"] != "shape" or shape.get("rotation"):
            return None
        order = paint_order(self.source["elements"])
        shape_box = rotated_aabb(shape)
        held = [t for t in self.source["elements"] if t["type"] == "text" and order[id(t)] > order[id(shape)]
                and contains(shape_box, rotated_aabb(t), BACKDROP_TOL)]
        if not held or any(t["id"] not in self.placed for t in held):
            return None
        k = sum(self.placed[t["id"]]["font_size"] / t["font_size"] for t in held) / len(held)
        sx, sy, sw, sh = shape_box
        tx, ty, tw, th = union_box([self._ink_box(t) for t in held])
        nx, ny, nw, nh = union_box([self._ink_box(self.placed[t["id"]]) for t in held])
        held_ids = {t["id"] for t in held}
        under = [rotated_aabb(g) for eid, g in self.placed.items()
                 if eid not in held_ids and order[id(self.els[eid])] < order[id(shape)]
                 and self.plan.unit_for_element(eid).role not in ("background", "decoration")]
        for pad in (k, 0.0):
            x0, y0 = max(0.0, nx - (tx - sx) * pad), max(0.0, ny - (ty - sy) * pad)
            x1 = min(self.width, nx + nw + (sx + sw - tx - tw) * pad)
            y1 = min(self.height, ny + nh + (sy + sh - ty - th) * pad)
            box = (x0, y0, x1 - x0, y1 - y0)
            if not any(overlap_area(box, other) > 0 for other in under):
                out = dict(shape, x=x0, y=y0, width=box[2], height=box[3])
                for key in ("corner_radius", "stroke_width"):
                    if shape.get(key):
                        out[key] = shape[key] * k
                return out
        return None

    def _visible_decoration(self, el: dict) -> dict:
        """The decoration at its nearest corner, or the first other corner
        where placed content doesn't fully cover it (G9: nothing hidden)."""
        content = [self.placed[e["id"]] for e in self.source["elements"] if e["id"] in self.placed]
        for corner in (None, "tl", "tr", "bl", "br"):
            candidate = place_decoration(el, self.source["canvas"], self.width, self.height, corner)
            elements = content + [candidate]
            ordered = sorted(elements, key=lambda e: self._source_index[e["id"]])
            if not is_covered(candidate, content, paint_order(ordered)):
                return candidate
        return place_decoration(el, self.source["canvas"], self.width, self.height)

    def finish(self, candidate_id: str, template: str, params: dict) -> Candidate | None:
        self.place_layers()
        design = finalize(self.source, self.placed, {"width": self.width, "height": self.height})
        return None if design is None else Candidate(candidate_id, design, template, params, self.degraded, False)


def _template_candidate(template: str, params: dict, source: dict, target: dict, plan: PriorityPlan,
                        measurer: TextMeasurer) -> Candidate | None:
    b = _Builder(source, target, plan, measurer)
    slots = build_slots(template, b.width, b.height, params)
    assignment = assign(plan, slots, params.get("cutoff"))
    if flex.enabled():  # CHAMELEON_FLEX=1: size the slots by what their units need (never raises)
        m = params["margin"] * min(b.width, b.height)
        slots = flex.flex_slots(slots, assignment, (m, m, b.width - 2 * m, b.height - 2 * m),
                                b.els, plan, measurer, b.width, b.height)
    # Regular slots first, then the minor zone, then whatever is left over.
    for index in sorted(assignment, key=lambda i: (slots[i].minor, i)):
        slot, items = slots[index], assignment[index]
        if slot.stack:
            b.place_stack([u for item in items for u in item], slot.box)
        elif slot.minor:
            b.place_flow([u for item in items for u in item], slot.box)
        else:
            for item in items:
                b.place_row(item, slot.box, minor=False, align=slot.align)
    if not b.place_leftovers(params["margin"] * min(b.width, b.height)):
        return None
    return b.finish(_candidate_id(template, params), template, params)


def reflow_candidate(source: dict, target: dict, plan: PriorityPlan, measurer: TextMeasurer) -> Candidate | None:
    """Keep the source arrangement: each unit shrinks by s = min(W/Ws, H/Hs)
    while its centre maps proportionally on each axis. Gaps scale by at least
    s, so this can't create overlaps the source didn't have."""
    b = _Builder(source, target, plan, measurer)
    rx, ry = b.width / source["canvas"]["width"], b.height / source["canvas"]["height"]
    s = min(rx, ry)
    for unit in plan.ranked_units():
        ux, uy, uw, uh = union_box([rotated_aabb(m) for m in b._members(unit)])
        cx, cy = (ux + uw / 2) * rx, (uy + uh / 2) * ry
        # Text scales with the layout and sits at the top of its box, where the
        # renderer draws it, so a same-aspect resize returns the source layout.
        geometry, degraded = b.try_place(unit, (cx - uw * s / 2, cy - uh * s / 2, uw * s, uh * s), minor=False,
                                         align="top", scale=s)
        if geometry is None:
            return None
        inside = fit_inside(list(geometry.values()), b.width, b.height)
        b._record(unit, {g["id"]: g for g in inside}, degraded)
    return b.finish("reflow", "reflow", {})


def generate_candidates(source: dict, target: dict, plan: PriorityPlan, measurer: TextMeasurer) -> list[Candidate]:
    """The reflow candidate first, then every template x parameter set for the
    target's aspect class, in a fixed order. Excludes the safe fallback. A
    template that fails to build is skipped, so one bug can't sink the task.
    When the target has the source's aspect, the proportional reflow is the
    faithful answer and is the only candidate (templates only if it fails)."""
    candidates = []
    try:
        reflow = reflow_candidate(source, target, plan, measurer)
    except Exception:
        reflow = None
    if reflow is not None:
        candidates.append(reflow)
        src_aspect = source["canvas"]["width"] / source["canvas"]["height"]
        if abs(target["width"] / target["height"] / src_aspect - 1) <= SAME_ASPECT_TOL:
            return candidates
    for template in templates_for(aspect_class(target["width"], target["height"])):
        for params in param_grid(template):
            try:
                candidate = _template_candidate(template, params, source, target, plan, measurer)
            except Exception:
                candidate = None
            if candidate is not None:
                candidates.append(candidate)
    if retrieval.enabled():  # CHAMELEON_RETRIEVAL=1: floor plans of the nearest training ads
        try:
            candidates += retrieval.candidates(source, target, plan, measurer)
        except Exception:
            pass  # retrieval never sinks the task
    candidates += strategy_registry.extra_candidates(source, target, plan, measurer)  # CHAMELEON_STRATEGIES
    return candidates
