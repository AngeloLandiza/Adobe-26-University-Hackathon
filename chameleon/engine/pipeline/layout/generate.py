"""Node 2's candidate generator (docs/pipeline/02-resize-engine.md §4-6).

For the target's aspect class it tries every template x parameter set:
units are assigned to slots in rank order, placed with `place_unit`, and
anything that doesn't fit goes to the leftover free space at a small size
(marked degraded). A proportional "reflow" of the source layout is tried
first. Every candidate is finalised; ones that can't be built are dropped.
"""

import math

from engine.pipeline.geometry import Box, align_in, rotated_aabb, union_box
from engine.pipeline.layout.aspect import aspect_class
from engine.pipeline.layout.layers import fit_inside, place_background, place_decoration
from engine.pipeline.layout.normalize import finalize
from engine.pipeline.layout.place import CONDENSED_PADDING, CONDENSED_VPAD, HARD_FLOOR, floor_px, place_unit
from engine.pipeline.layout.templates import PARAM_GRID, TEMPLATES_BY_CLASS, Slot, build_slots
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Candidate, PriorityPlan, Unit

LOGO_MAX_GROWTH = 1.5   # a logo never grows past 1.5x its source size
FREE_MAX_SHARE = 1 / 3  # a leftover image gets at most a third of the canvas each way
FREE_TRIES = 12         # distinct empty rectangles tried per leftover unit, largest first
FLOW_GAP = 0.02         # space between minor units, as a share of the canvas's short side
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

    def try_place(self, unit: Unit, box: Box, minor: bool, align: str = "center") -> tuple[dict[str, dict] | None, bool]:
        """Place at the role's floor, else at the hard floor (degraded). Minor
        units are capped at their floor size."""
        floor = floor_px(unit.role, self.width, self.height)
        max_font = floor if minor else None
        box = self._cap_logo(unit, box)
        geometry = place_unit(unit, self.els, box, self.measurer, floor, align=align, max_font=max_font)
        if geometry is not None:
            return geometry, False
        return place_unit(unit, self.els, box, self.measurer, HARD_FLOOR, align=align, max_font=max_font), True

    def place_row(self, units: list[Unit], box: Box, minor: bool, align: str = "center") -> None:
        """Units side by side in equal-width cells."""
        x, y, w, h = box
        cell = w / len(units)
        for i, unit in enumerate(units):
            geometry, degraded = self.try_place(unit, (x + i * cell, y, cell, h), minor, align)
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
        text fits. False if any unit can't be placed anywhere."""
        for unit in sorted(self.leftover, key=lambda u: (-self._footprint(u), u.rank)):
            has_image = any(m["type"] == "image" for m in self._members(unit))
            for rect in self.free_rects(margin):
                box = rect
                if has_image:  # text and pills are already held to their floor size by max_font
                    w, h = min(rect[2], self.width * FREE_MAX_SHARE), min(rect[3], self.height * FREE_MAX_SHARE)
                    box = (*align_in(rect, w, h), w, h)
                floor = floor_px(unit.role, self.width, self.height)
                geometry = place_unit(unit, self.els, box, self.measurer, HARD_FLOOR, max_font=floor)
                if geometry is not None:
                    self._record(unit, geometry, degraded=True)
                    break
            else:
                return False
        self.leftover = []
        return True

    def place_layers(self) -> None:
        for unit in self.plan.units:
            if unit.role == "background":
                self.placed.update((m["id"], place_background(m, self.width, self.height)) for m in self._members(unit))
            elif unit.role == "decoration":
                self.placed.update((m["id"], place_decoration(m, self.source["canvas"], self.width, self.height))
                                   for m in self._members(unit))

    def finish(self, candidate_id: str, template: str, params: dict) -> Candidate | None:
        self.place_layers()
        design = finalize(self.source, self.placed, {"width": self.width, "height": self.height})
        return None if design is None else Candidate(candidate_id, design, template, params, self.degraded, False)


def _template_candidate(template: str, params: dict, source: dict, target: dict, plan: PriorityPlan,
                        measurer: TextMeasurer) -> Candidate | None:
    b = _Builder(source, target, plan, measurer)
    slots = build_slots(template, b.width, b.height, params)
    assignment = assign(plan, slots, params.get("cutoff"))
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
        geometry, degraded = b.try_place(unit, (cx - uw * s / 2, cy - uh * s / 2, uw * s, uh * s), minor=False)
        if geometry is None:
            return None
        inside = fit_inside(list(geometry.values()), b.width, b.height)
        b._record(unit, {g["id"]: g for g in inside}, degraded)
    return b.finish("reflow", "reflow", {})


def generate_candidates(source: dict, target: dict, plan: PriorityPlan, measurer: TextMeasurer) -> list[Candidate]:
    """The reflow candidate first, then every template x parameter set for the
    target's aspect class, in a fixed order. Excludes the safe fallback. A
    template that fails to build is skipped, so one bug can't sink the task."""
    candidates = []
    try:
        reflow = reflow_candidate(source, target, plan, measurer)
    except Exception:
        reflow = None
    if reflow is not None:
        candidates.append(reflow)
    for template in TEMPLATES_BY_CLASS[aspect_class(target["width"], target["height"])]:
        for params in PARAM_GRID[template]:
            try:
                candidate = _template_candidate(template, params, source, target, plan, measurer)
            except Exception:
                candidate = None
            if candidate is not None:
                candidates.append(candidate)
    return candidates
