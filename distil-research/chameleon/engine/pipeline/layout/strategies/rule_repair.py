"""Strategy S3 `rule_repair`: repair a picked layout by local search against
the mined design rulebook (rules.py, docs/design-rulebook.md).

`refine` hill-climbs over affine moves of the content units, a unit or the
whole content block at a time: translate, scale about the unit's centre, and
for the block also spread (positions only) and grow. Decorations that sit
behind a unit (a circle behind the can, a panel behind the copy) move with
it. Every step takes the move that lowers `rules.penalty` the most among
those that keep every element on the canvas, add no new overlap between
content units or text-across-decoration clash, never break a reading-order
pair that agreed with the source, and pass `check_gates` after rounding.
Text keeps its wrap where it can (font scales with the box) and never grows
past 1.5x its source size, nor any image but the hero past 1.5x its own;
text and logos never shrink, nothing enters the 3% edge band, and no
cluster drifts away from its neighbours. Deterministic: moves are
tried in a fixed order and ties go to the earlier move. Never raises.
"""

import math

from engine.pipeline.geometry import Box, overlap_area, rotated_aabb, union_box
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout.layers import fit_inside
from engine.pipeline.layout.normalize import finalize
from engine.pipeline.layout.place import MAX_TEXT_GROWTH, STEP
from engine.pipeline.layout.strategies import rules
from engine.pipeline.occlusion import paint_order
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Candidate, PriorityPlan, Unit

NAME = "rule_repair"
MAX_STEPS = 30              # accepted moves per refine call
MIN_GAIN = 0.004            # a move must lower the penalty by at least this much
MIN_TOTAL_GAIN = 0.02       # return None unless the whole repair gained this much
TRANSLATE_STEPS = (0.08, 0.03)   # share of the canvas side
UNIT_SCALES = (1.25, 1.1, 0.9)
BLOCK_SCALES = (1.3, 1.12, 0.92)
SPREADS = (1.15, 0.9)
LOGO_MAX_GROWTH = 1.5
LOGO_VS_HEADLINE = 0.25     # a logo may grow only while its area stays under this share of the headline's
LOGO_VS_HEADLINE_HEIGHT = 1.5  # ... and its height under this multiple of the headline block's
EDGE_BAND = 0.03            # no unit moves closer than this share of the short side to an edge (unless it was)
FOLLOW_SHARE = 0.3          # a decoration follows the unit covering this share of it (or holding it)
SAME_ASPECT_TOL = 0.02      # the target has the source's aspect: the reflow is the faithful answer
CLUSTER_GAP = 0.03          # same-role units this close (share of short side) move as one (a chip row)
STRAND_GAP = 0.08           # a cluster may drift from its nearest neighbour only up to this share of the short side
_SIZE_KEYS = ("font_size", "letter_spacing", "corner_radius", "stroke_width")


def refine(source: dict, target: dict, plan: PriorityPlan, measurer: TextMeasurer,
           candidate: Candidate) -> Candidate | None:
    """A repaired copy of `candidate` (id `rule_repair+<id>`), or None if no
    repair lowers the rule penalty enough. Never raises."""
    try:
        return _Repair(source, target, plan, measurer, candidate).run()
    except Exception:
        return None


def order_agreement(source: dict, design: dict, plan: PriorityPlan, measurer: TextMeasurer | None = None) -> float:
    """Share of content-unit pairs whose reading order matches the source's."""
    try:
        measurer = measurer or _default_measurer()
        els = {e["id"]: e for e in design["elements"]}
        src = {e["id"]: e for e in source["elements"]}
        units = rules.content_units(plan, els)
        now = rules.order_pairs(units, rules.unit_boxes(units, els, measurer), design["canvas"]["height"])
        was = rules.order_pairs(units, rules.unit_boxes(units, src, measurer), source["canvas"]["height"])
        return sum(now[k] == was[k] for k in now) / len(now) if now else 1.0
    except Exception:
        return 0.0


_measurer_cache: list[TextMeasurer] = []


def _default_measurer() -> TextMeasurer:
    if not _measurer_cache:
        _measurer_cache.append(TextMeasurer())
    return _measurer_cache[0]


class _Repair:
    def __init__(self, source, target, plan, measurer, candidate):
        self.source, self.plan, self.measurer, self.cand = source, plan, measurer, candidate
        self.target = {"width": target["width"], "height": target["height"]}
        self.width, self.height = float(target["width"]), float(target["height"])
        self.src_els = {e["id"]: e for e in source["elements"]}

    # ------------------------------------------------------------ setup

    def run(self) -> Candidate | None:
        cand = self.cand
        if cand is None or not isinstance(cand.design, dict) or not isinstance(cand.design.get("elements"), list):
            return None
        src_aspect = self.source["canvas"]["width"] / self.source["canvas"]["height"]
        if abs(self.width / self.height / src_aspect - 1) <= SAME_ASPECT_TOL:
            return None
        if check_gates(self.source, cand.design, self.target, self.plan, self.measurer):
            return None
        els = {e["id"]: dict(e) for e in cand.design["elements"]}
        self.units = rules.content_units(self.plan, els)
        if not self.units:
            return None
        self.followers = self._followers(els)
        self.clusters = self._clusters(self._boxes(els))
        self.base_fonts = {e["id"]: e["font_size"] for e in els.values() if e.get("type") == "text"}
        self.order = paint_order(cand.design["elements"])
        boxes = self._boxes(els)
        self.base_overlap = self._overlaps(boxes)
        self.base_clash = self._clashes(els, boxes)
        src_units = rules.unit_boxes(self.units, self.src_els, self.measurer)
        self.src_pairs = rules.order_pairs(self.units, src_units, self.source["canvas"]["height"])
        now = rules.order_pairs(self.units, boxes, self.height)
        self.kept_pairs = {k for k in now if now[k] == self.src_pairs[k]}
        self.base_nn = self._cluster_gaps(boxes)
        self.base_logo = {u.unit_id: boxes[u.unit_id][2:] for u in self.units if u.role == "logo"}
        heads = [boxes[u.unit_id] for u in self.units if u.role == "headline"]
        self.headline_box = max(heads, key=lambda b: b[2] * b[3]) if heads else None
        self.base_edge = {u.unit_id: rules._edge(boxes[u.unit_id], self.width, self.height) for u in self.units}
        self.edge_band = EDGE_BAND * min(self.width, self.height)

        start, _ = self._score(cand.design)
        best_design, best = cand.design, start
        for _ in range(MAX_STEPS):
            step = self._best_move(els, best)
            if step is None:
                break
            els, best_design, best = step
        if start - best < MIN_TOTAL_GAIN or best_design is cand.design:
            return None
        return Candidate(f"{NAME}+{cand.candidate_id}", best_design, cand.template,
                         dict(cand.params, repaired=True), list(cand.degraded_unit_ids), False)

    def _members(self, unit: Unit, els: dict) -> list[dict]:
        return [els[i] for i in unit.element_ids]

    def _boxes(self, els: dict) -> dict[str, Box]:
        return rules.unit_boxes(self.units, els, self.measurer)

    def _followers(self, els: dict) -> dict[str, list[str]]:
        """unit_id -> decoration element ids that move with it: a shape or
        image painted under the unit that holds it, or that the unit covers
        a good share of (the unit with the largest overlap wins)."""
        deco = [e for u in self.plan.units if u.role == "decoration" for e in (els.get(i) for i in u.element_ids) if e]
        boxes = self._boxes(els)
        out: dict[str, list[str]] = {u.unit_id: [] for u in self.units}
        for d in deco:
            db = rotated_aabb(d)
            area = db[2] * db[3]
            if area <= 0 or (db[2] >= 0.95 * self.width and db[3] >= 0.95 * self.height):
                continue
            best, best_share = None, 0.0
            for u in self.units:
                ub = boxes[u.unit_id]
                share = overlap_area(db, ub) / area
                holds = overlap_area(db, ub) >= 0.95 * ub[2] * ub[3] > 0
                score = 1.0 + share if holds else share
                if (holds or share >= FOLLOW_SHARE) and score > best_share:
                    best, best_share = u, score
            if best is not None:
                out[best.unit_id].append(d["id"])
        return out

    def _clusters(self, boxes: dict[str, Box]) -> list[list[Unit]]:
        """Same-role units sitting close together (a row of chips, a badge
        pair) form one cluster; clusters are what single moves act on."""
        short = min(self.width, self.height)
        parent = {u.unit_id: u.unit_id for u in self.units}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for i, a in enumerate(self.units):
            for b in self.units[i + 1:]:
                if a.role == b.role and rules._gap(boxes[a.unit_id], boxes[b.unit_id]) <= CLUSTER_GAP * short:
                    parent[find(b.unit_id)] = find(a.unit_id)
        groups: dict[str, list[Unit]] = {}
        for u in self.units:
            groups.setdefault(find(u.unit_id), []).append(u)
        return list(groups.values())

    def _cluster_gaps(self, boxes: dict[str, Box]) -> list[float]:
        """Each cluster's gap to the nearest other cluster."""
        cb = [union_box([boxes[u.unit_id] for u in c]) for c in self.clusters]
        return [min((rules._gap(a, b) for j, b in enumerate(cb) if j != i), default=0.0) for i, a in enumerate(cb)]

    # ------------------------------------------------------------ constraints

    def _overlaps(self, boxes: dict[str, Box]) -> dict[tuple[str, str], float]:
        ids = [u.unit_id for u in self.units]
        return {(a, b): overlap_area(boxes[a], boxes[b]) for i, a in enumerate(ids) for b in ids[i + 1:]}

    def _clashes(self, els: dict, boxes: dict[str, Box]) -> int:
        """Text units straddling a decoration's edge (partly on it, partly off)."""
        deco = [els[i] for u in self.plan.units if u.role == "decoration" for i in u.element_ids if i in els]
        n = 0
        for u in self.units:
            if not any(els[i].get("type") == "text" for i in u.element_ids):
                continue
            ub = boxes[u.unit_id]
            area = ub[2] * ub[3]
            for d in deco:
                if d["id"] in self.followers.get(u.unit_id, []):
                    continue
                ov = overlap_area(ub, rotated_aabb(d))
                if area > 0 and 0.02 * area < ov < 0.98 * area:
                    n += 1
        return n

    def _allowed(self, els: dict) -> tuple[bool, dict[str, Box]]:
        for e in els.values():
            x, y, w, h = rotated_aabb(e)
            if x < -0.01 or y < -0.01 or x + w > self.width + 0.01 or y + h > self.height + 0.01:
                return False, {}
        boxes = self._boxes(els)
        for k, v in self._overlaps(boxes).items():
            if v > self.base_overlap[k] + 0.5:
                return False, {}
        now = rules.order_pairs(self.units, boxes, self.height)
        if any(now[k] != self.src_pairs[k] for k in self.kept_pairs):
            return False, {}
        if self._clashes(els, boxes) > self.base_clash:
            return False, {}
        for eid, size in self.base_fonts.items():  # text never shrinks: "too small" is the top complaint
            if els[eid]["font_size"] < size - 0.01:
                return False, {}
        for u in self.units:
            b = boxes[u.unit_id]
            # Margins: nothing moves into the edge band it wasn't already in.
            if not rules._is_full_bleed(b, self.width, self.height) and \
                    rules._edge(b, self.width, self.height) < min(self.base_edge[u.unit_id], self.edge_band) - 0.5:
                return False, {}
            # A logo never grows to outweigh the headline.
            if u.role == "logo" and b[2] * b[3] < self.base_logo[u.unit_id][0] * self.base_logo[u.unit_id][1] - 1:
                return False, {}  # nor shrinks: a tiny logo is a top-five complaint
            if u.role == "logo" and self.headline_box is not None:
                lw, lh = self.base_logo[u.unit_id]
                hb = self.headline_box
                if b[2] * b[3] > max(lw * lh, LOGO_VS_HEADLINE * hb[2] * hb[3]) + 1 or \
                        b[3] > max(lh, LOGO_VS_HEADLINE_HEIGHT * hb[3]) + 0.5:
                    return False, {}
        cap = STRAND_GAP * min(self.width, self.height)
        for before, after in zip(self.base_nn, self._cluster_gaps(boxes)):
            if after > max(before, cap) + 0.5:
                return False, {}
        return True, boxes

    # ------------------------------------------------------------ moves

    def _transform(self, els: dict, unit: Unit, k: float, ax: float, ay: float, dx: float, dy: float) -> bool:
        """Apply p -> a + k (p - a) + d to the unit and its followers, in
        place. False if text can't be fitted or a cap is exceeded."""
        for el in self._members(unit, els):
            new = _affine(el, k, ax, ay, dx, dy)
            src = self.src_els[el["id"]]
            if new.get("type") == "text":
                new = self._fit_text(new, src)
                if new is None:
                    return False
            elif unit.role != "hero_image" and k > 1 and max(new["width"] / src["width"] if src["width"] else 1,
                                                        new["height"] / src["height"] if src["height"] else 1) > LOGO_MAX_GROWTH + 1e-6:
                return False
            els[el["id"]] = new
        for fid in self.followers.get(unit.unit_id, []):
            moved = _affine(els[fid], k, ax, ay, dx, dy)
            els[fid] = fit_inside([moved], self.width, self.height)[0]
        return True

    def _fit_text(self, el: dict, src: dict) -> dict | None:
        size = min(el["font_size"], src["font_size"] * MAX_TEXT_GROWTH)
        if size < 1:
            return None
        for attempt in range(8):
            if attempt == 1:  # the exact scaled size didn't fit: fall to the 0.5 px grid
                size = math.floor(size / STEP + 1e-9) * STEP
            text = _resized(el, size, src)
            block = self.measurer.block_height(text, size, text["width"])
            text["height"] = max(text["height"], block) if block <= text["height"] + 1 else block
            if self.measurer.fits(text, size, text["width"], text["height"]):
                return text
            if attempt >= 1:
                size -= STEP
            if size < 1:
                break
        return None

    def _moves(self, boxes: dict[str, Box]):
        """(label, [(unit, k, ax, ay, dx, dy), ...]) in a fixed order."""
        W, H = self.width, self.height
        block = union_box(list(boxes.values()))
        bcx, bcy = block[0] + block[2] / 2, block[1] + block[3] / 2
        for k in BLOCK_SCALES:
            yield f"block*{k}", [(u, k, bcx, bcy, 0.0, 0.0) for u in self.units]
        for k in SPREADS:
            for axis in ("x", "y"):
                ops = []
                for cluster in self.clusters:
                    x, y, w, h = union_box([boxes[u.unit_id] for u in cluster])
                    cx, cy = x + w / 2, y + h / 2
                    dx = (k - 1) * (cx - bcx) if axis == "x" else 0.0
                    dy = (k - 1) * (cy - bcy) if axis == "y" else 0.0
                    ops += [(u, 1.0, cx, cy, dx, dy) for u in cluster]
                yield f"spread{axis}*{k}", ops
        ccx, ccy = W / 2 - bcx, H / 2 - bcy
        yield "centre", [(u, 1.0, 0.0, 0.0, ccx, ccy) for u in self.units]
        for s in TRANSLATE_STEPS:
            for dx, dy in ((s * W, 0), (-s * W, 0), (0, s * H), (0, -s * H)):
                yield f"block+{dx:.0f},{dy:.0f}", [(u, 1.0, 0.0, 0.0, dx, dy) for u in self.units]
        for cluster in self.clusters:
            x, y, w, h = union_box([boxes[u.unit_id] for u in cluster])
            name = "|".join(u.unit_id for u in cluster)
            for k in UNIT_SCALES:
                yield f"{name}*{k}", [(u, k, x + w / 2, y + h / 2, 0.0, 0.0) for u in cluster]
            for s in TRANSLATE_STEPS:
                for dx, dy in ((s * W, 0), (-s * W, 0), (0, s * H), (0, -s * H)):
                    yield f"{name}+{dx:.0f},{dy:.0f}", [(u, 1.0, 0.0, 0.0, dx, dy) for u in cluster]

    def _best_move(self, els: dict, current: float):
        boxes = self._boxes(els)
        options = []
        for index, (label, ops) in enumerate(self._moves(boxes)):
            trial = dict(els)
            if not all(self._transform(trial, *op) for op in ops):
                continue
            ok, _ = self._allowed(trial)
            if not ok:
                continue
            design = {"canvas": dict(self.cand.design["canvas"]), "elements": [trial[e["id"]] for e in self.source["elements"]]}
            score, _ = self._score(design)
            if score < current - MIN_GAIN:
                options.append((score, index, trial))
        for score, _, trial in sorted(options, key=lambda o: (o[0], o[1])):
            design = finalize(self.source, trial, self.target)
            if design is None or check_gates(self.source, design, self.target, self.plan, self.measurer):
                continue
            final, _ = self._score(design)
            if final < current - MIN_GAIN:
                return {e["id"]: dict(e) for e in design["elements"]}, design, final
        return None

    def _score(self, design: dict) -> tuple[float, dict]:
        return rules.penalty(self.source, design, self.plan, self.measurer)


def _affine(el: dict, k: float, ax: float, ay: float, dx: float, dy: float) -> dict:
    new = dict(el, x=ax + (el["x"] - ax) * k + dx, y=ay + (el["y"] - ay) * k + dy,
               width=el["width"] * k, height=el["height"] * k)
    if k != 1:
        for key in _SIZE_KEYS:
            if el.get(key):
                new[key] = el[key] * k
    return new


def _resized(el: dict, size: float, src: dict) -> dict:
    out = dict(el, font_size=size)
    if src.get("letter_spacing"):
        out["letter_spacing"] = src["letter_spacing"] * size / src["font_size"]
    return out
