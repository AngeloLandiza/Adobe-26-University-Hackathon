"""Content-aware slot sizing (engine/pipeline/layout/flex.py), on with CHAMELEON_FLEX=1.

Off (the default) the engine must give exactly what it gave before flex
existed; on, every candidate must still pass the gates, and the layouts must
use the canvas better (more text at its readability floor, less dead space).
"""

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from engine.pipeline.geometry import rotated_aabb, union_box
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.grading.metrics import COMFORT
from engine.pipeline.layout import extra_templates, flex
from engine.pipeline.layout.flex import Need, decompose, distribute, flex_slots, ink_share, logo_min_width
from engine.pipeline.layout.generate import _template_candidate, assign, generate_candidates
from engine.pipeline.layout.place import floor_px, place_unit
from engine.pipeline.layout.templates import Slot, build_slots, PARAM_GRID
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.types import PriorityPlan, Unit
from tests.conftest import assert_valid

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())
BY_NAME = {t["name"]: {"width": t["width"], "height": t["height"]} for t in TARGETS}

# sha256[:16] of every candidate (asdict, sorted keys) generate_candidates gave
# BEFORE flex existed, with no floor plans loaded. Text measurement differs per
# platform, so only the platform these were recorded on checks them.
DIGESTS_BEFORE = {"darwin": {
    "level1|story": "6a829b890cf39881", "level1|instagram_post": "e79dedfe5fd789ca",
    "level1|medium_rectangle": "979351f3c33a0002", "level1|leaderboard": "59bca118f9d482e4",
    "level2|story": "4e087e205b0337f9", "level2|instagram_post": "c11dee7a6dc27d2b",
    "level2|medium_rectangle": "f4184bd9be1edf6c", "level2|leaderboard": "c9ea6767aff6ac26",
    "level3|story": "6a0656a5fb30e8f4", "level3|instagram_post": "19a8e249fff5109b",
    "level3|medium_rectangle": "3a645062173a2329", "level3|leaderboard": "13b32e896e289281"}}


@pytest.fixture
def flex_on(monkeypatch):
    monkeypatch.setenv(flex.ENV, "1")


@pytest.fixture
def flex_off(monkeypatch):
    monkeypatch.delenv(flex.ENV, raising=False)


@pytest.fixture
def no_floor_plans(monkeypatch, tmp_path):
    monkeypatch.delenv(extra_templates.ENV_VAR, raising=False)
    monkeypatch.setattr(extra_templates, "EXTRA_PATH", tmp_path / "absent.json")


def _sample(path):
    return json.loads(Path(path).read_text())["source"]


def _ranked_box(design, unit):
    els = {e["id"]: e for e in design["elements"]}
    return union_box([rotated_aabb(els[i]) for i in unit.element_ids])


# --- the 1-D solver ---------------------------------------------------------

def test_distribute_collapses_empty_items_and_keeps_equal_gaps():
    needs = [Need(10, 10, 10, 1), Need(0, 0, 0, 0, empty=True), Need(20, 20, 20, 1), Need(5, 5, 5, 1)]
    out = distribute(100, needs, gap=4)
    assert out[1][1] == 0
    live = [out[i] for i in (0, 2, 3)]
    gaps = [b[0] - (a[0] + a[1]) for a, b in zip(live, live[1:])]
    assert gaps[0] == pytest.approx(gaps[1]) and gaps[0] >= 4
    # everything at its cap: the remaining space is spread evenly, so the group is centred
    assert live[0][0] == pytest.approx(100 - (live[2][0] + live[2][1]))


def test_distribute_shares_leftover_by_weight_up_to_caps():
    out = distribute(100, [Need(10, 10, 1000, 2), Need(10, 10, 1000, 1)], gap=0)
    assert out[0][1] - 10 == pytest.approx(2 * (out[1][1] - 10))
    assert out[0][1] + out[1][1] == pytest.approx(100)
    capped = distribute(100, [Need(10, 10, 20, 5), Need(10, 10, 1000, 1)], gap=0)
    assert capped[0][1] == pytest.approx(20) and capped[1][1] == pytest.approx(80)


def test_distribute_reaches_preferred_sizes_before_growing_to_caps():
    out = distribute(100, [Need(10, 50, 1000, 1), Need(10, 30, 1000, 100)], gap=0)
    # the heavy item can't take the light one's preferred space
    assert out[0][1] >= 50 - 1e-6 and out[1][1] >= 30 - 1e-6


def test_distribute_scales_minimums_when_they_do_not_fit():
    out = distribute(50, [Need(60, 60, 60, 1), Need(40, 40, 40, 1)], gap=0)
    assert out[0][1] == pytest.approx(30) and out[1][1] == pytest.approx(20)


# --- decomposition ----------------------------------------------------------

def test_decompose_builtin_stack_and_floor_plans():
    slots = build_slots("stack", 1000, 1000, {"margin": 0.0, "hero_first": True})
    root = decompose([tuple(v / 1000 for v in s.box) for s in slots])
    assert root is not None and root.axis == "v" and len(root.children) == 5
    assert root.children[0].axis == "h"  # badge | logo header
    for plan in extra_templates.load_floor_plans(extra_templates.EXTRA_PATH):
        assert decompose([box for _, box in plan.slots]) is not None, plan.name


def test_undecomposable_floor_plan_falls_back_to_fixed_fractions(measurer):
    # a pinwheel: no straight cut separates the slots
    boxes = [(0, 0, 0.6, 0.3), (0.6, 0, 0.4, 0.6), (0.4, 0.6, 0.6, 0.4), (0, 0.3, 0.4, 0.7), (0.4, 0.3, 0.2, 0.3)]
    assert decompose(boxes) is None
    names = ("hero", "text", "logo", "cta", "minor")
    slots = [Slot(n, tuple(v * 1000 for v in b), ("headline",), stack=(n == "text"), minor=(n == "minor"))
             for n, b in zip(names, boxes)]
    src = {"canvas": {"width": 1000, "height": 1000}, "elements": [
        {"id": "t", "type": "text", "content": "Hello", "font_family": "Solstice Serif", "font_size": 40,
         "color": "rgba(0,0,0,1)", "x": 0, "y": 0, "width": 400, "height": 60}]}
    plan = heuristic_plan(src)
    assignment = assign(plan, slots, None)
    out = flex_slots(slots, assignment, (0, 0, 1000, 1000), {"t": src["elements"][0]}, plan, measurer, 1000, 1000)
    assert [s.box for s in out] == [s.box for s in slots]


def test_flex_slots_never_raises(measurer):
    slots = build_slots("stack", 1080, 1920, {"margin": 0.04, "hero_first": True})
    bogus = PriorityPlan("heuristic", [Unit("x", ["missing"], "single", role="headline", rank=1)])
    assignment = assign(bogus, slots, None)
    out = flex_slots(slots, assignment, (0, 0, 1080, 1920), {}, bogus, measurer, 1080, 1920)
    assert [s.box for s in out] == [s.box for s in slots]


# --- slot geometry on real content -----------------------------------------

def _flexed(task, template, params, target, measurer):
    src = task["source"]
    plan = heuristic_plan(src)
    w, h = target["width"], target["height"]
    slots = build_slots(template, w, h, params)
    assignment = assign(plan, slots, params.get("cutoff"))
    m = params["margin"] * min(w, h)
    els = {e["id"]: e for e in src["elements"]}
    return slots, flex_slots(slots, assignment, (m, m, w - 2 * m, h - 2 * m), els, plan, measurer, w, h), assignment


def test_empty_slots_collapse_and_their_space_goes_to_the_rest(l1, measurer):
    # level 1 has no hero image, badge or minor units: those slots get no room
    slots, out, assignment = _flexed(l1, "stack", {"margin": 0.04, "hero_first": True}, BY_NAME["story"], measurer)
    for i, s in enumerate(out):
        if i not in assignment:
            assert s.box[2] * s.box[3] == pytest.approx(0, abs=1e-6), s.name
    # the empty slots' space doesn't stay behind as one dead band: the largest vertical
    # stretch of the content box no used slot covers shrinks
    def widest_band(boxes, top, bottom):
        rows = sorted((b[1], b[1] + b[3]) for b in boxes)
        edges = [top] + [v for r in rows for v in r] + [bottom]
        return max(edges[i + 1] - edges[i] for i in range(0, len(edges), 2))
    m = 0.04 * 1080
    fixed = widest_band([slots[i].box for i in assignment], m, 1920 - m)
    flexed = widest_band([out[i].box for i in assignment], m, 1920 - m)
    assert flexed < fixed * 0.6, (flexed, fixed)
    # what's left once every unit is at its largest is spread evenly between and around them
    rows = sorted((out[i].box[1], out[i].box[1] + out[i].box[3]) for i in assignment)
    inner = [b[0] - a[1] for a, b in zip(rows, rows[1:])]
    assert max(inner) - min(inner) < 0.5


def test_equal_gaps_between_stacked_slots(l2, measurer):
    _, out, assignment = _flexed(l2, "stack", {"margin": 0.04, "hero_first": True}, BY_NAME["story"], measurer)
    rows = sorted({(round(s.box[1], 6), round(s.box[1] + s.box[3], 6)) for i, s in enumerate(out) if i in assignment})
    gaps = [b[0] - a[1] for a, b in zip(rows, rows[1:])]
    assert len(gaps) >= 2 and max(gaps) - min(gaps) < 0.5 and min(gaps) > 0


def test_text_slot_grows_to_fit_its_text_at_the_floor(measurer):
    words = "Seasonal botanical infusions crafted in small batches from hand picked citrus and wild herbs " * 5
    src = {"canvas": {"width": 1080, "height": 1080}, "elements": [
        {"id": "head", "type": "text", "content": words.strip(), "font_family": "Solstice Serif", "font_size": 48,
         "color": "rgba(0,0,0,1)", "x": 60, "y": 500, "width": 960, "height": 400},
        {"id": "pic", "type": "image", "asset_id": "solstice_bottle.png", "x": 290, "y": 40, "width": 500, "height": 400}]}
    target = BY_NAME["half_page"]
    task = {"source": src}
    params = {"margin": 0.04, "hero_first": True}
    slots, out, _ = _flexed(task, "stack", params, target, measurer)
    text = next(s for s in out if s.name == "text")
    floor = floor_px("headline", target["width"], target["height"])
    plan = heuristic_plan(src)
    unit = next(u for u in plan.units if u.unit_id == "head")
    fixed = next(s for s in slots if s.name == "text")
    assert place_unit(unit, {e["id"]: e for e in src["elements"]}, fixed.box, measurer, floor) is None  # the bug
    assert place_unit(unit, {e["id"]: e for e in src["elements"]}, text.box, measurer, floor) is not None


def test_logo_never_below_its_minimum_visible_size(l1, l2, measurer, flex_on, monkeypatch):
    for task in (l1, l2):
        plan = heuristic_plan(task["source"])
        logo_unit = next(u for u in plan.units if u.role == "logo")
        logo_el = next(e for e in task["source"]["elements"] if e["id"] in logo_unit.element_ids)
        ink = ink_share(logo_el)
        assert ink[0] < 0.5  # the sample lockup's mark spans well under half its box
        for name in ("story", "instagram_portrait", "half_page", "medium_rectangle", "youtube_thumbnail"):
            target = BY_NAME[name]
            for c in generate_candidates(task["source"], target, plan, measurer):
                # only where the template gives the logo its own slot (otherwise it shares the minor
                # zone's one line with the units ranked above it)
                if c.template == "reflow" or not any(
                        s.name == "logo" for s in build_slots(c.template, target["width"], target["height"], c.params)):
                    continue
                box = _ranked_box(c.design, logo_unit)
                want = logo_min_width(target["width"], target["height"], logo_el["width"], logo_el["height"], ink)
                # where the logo's row can't fit everything at its minimum, the logo still keeps
                # LOGO_KEEP of the canvas width
                assert box[2] >= min(want, flex.LOGO_KEEP * target["width"]) - 1.0, (name, c.candidate_id, box, want)
                monkeypatch.delenv(flex.ENV)
                fixed = _template_candidate(c.template, c.params, task["source"], target, plan, measurer)
                monkeypatch.setenv(flex.ENV, "1")
                if fixed is not None:  # and never clearly smaller than the fixed template drew it (gaps cost a little)
                    assert box[2] >= 0.95 * _ranked_box(fixed.design, logo_unit)[2] - 1.0, (name, c.candidate_id)


# --- flag off: unchanged ----------------------------------------------------

@pytest.mark.skipif(sys.platform not in DIGESTS_BEFORE, reason="digests recorded on another platform")
def test_flag_off_candidates_identical_to_before(no_floor_plans, flex_off, measurer, monkeypatch):
    calls = []
    monkeypatch.setattr(flex, "flex_slots", lambda *a, **k: calls.append(1) or a[0])
    expected = DIGESTS_BEFORE[sys.platform]
    for path in SAMPLES:
        src = _sample(path)
        plan = heuristic_plan(src)
        level = path.stem.split("_")[2]
        for name in ("story", "instagram_post", "medium_rectangle", "leaderboard"):
            cands = generate_candidates(src, BY_NAME[name], plan, measurer)
            blob = json.dumps([asdict(c) for c in cands], sort_keys=True)
            assert hashlib.sha256(blob.encode()).hexdigest()[:16] == expected[f"{level}|{name}"], (level, name)
    assert calls == []


def test_flag_off_matches_a_run_without_the_hook(flex_off, measurer, monkeypatch):
    """Platform-independent: with the flag unset, the solver is never consulted."""
    def boom(*a, **k):
        raise AssertionError("flex consulted with the flag off")
    for path in SAMPLES:
        src = _sample(path)
        plan = heuristic_plan(src)
        for name in ("story", "leaderboard", "youtube_thumbnail"):
            before = [asdict(c) for c in generate_candidates(src, BY_NAME[name], plan, measurer)]
            monkeypatch.setattr(flex, "flex_slots", boom)
            after = [asdict(c) for c in generate_candidates(src, BY_NAME[name], plan, measurer)]
            monkeypatch.undo()
            monkeypatch.delenv(flex.ENV, raising=False)
            assert before == after


# --- flag on: valid everywhere, better proxies -----------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_flag_on_every_candidate_passes_the_gates(path, target, measurer, flex_on):
    src = _sample(path)
    plan = heuristic_plan(src)
    tgt = {"width": target["width"], "height": target["height"]}
    cands = generate_candidates(src, tgt, plan, measurer)
    assert cands
    for c in cands:
        assert check_gates(src, c.design, tgt, plan, measurer) == [], c.candidate_id
        assert_valid(src, c.design, tgt)


def test_flag_on_changes_template_geometry(l2, measurer, flex_on, monkeypatch):
    plan = heuristic_plan(l2["source"])
    params = {"margin": 0.04, "hero_first": True}
    on = _template_candidate("stack", params, l2["source"], BY_NAME["story"], plan, measurer)
    monkeypatch.delenv(flex.ENV)
    off = _template_candidate("stack", params, l2["source"], BY_NAME["story"], plan, measurer)
    assert on is not None and off is not None and on.design != off.design


def _largest_empty_share(boxes, width, height) -> float:
    """Largest axis-aligned rectangle touching no content box, as a share of the canvas (grid, histogram scan)."""
    cell = min(width, height) / 40
    cols, rows = max(1, int(width // cell)), max(1, int(height // cell))
    busy = [[False] * cols for _ in range(rows)]
    for bx, by, bw, bh in boxes:
        for r in range(max(0, int(by // cell)), min(rows, int((by + bh) // cell) + 1)):
            for c in range(max(0, int(bx // cell)), min(cols, int((bx + bw) // cell) + 1)):
                busy[r][c] = True
    best, heights = 0, [0] * cols
    for r in range(rows):
        heights = [0 if busy[r][c] else heights[c] + 1 for c in range(cols)]
        stack = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            while stack and heights[stack[-1]] >= h:
                top = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                best = max(best, top * (c - left))
            stack.append(c)
    return best / (cols * rows)


def proxy_metrics(design, plan, target) -> dict[str, float]:
    """Per candidate: share of text units at/above their floor and at/above COMFORT x floor (smallest
    size in the unit), the largest empty rectangle's share of the canvas, and the smallest distance
    from content to a canvas edge as a share of the short side."""
    w, h = target["width"], target["height"]
    els = {e["id"]: e for e in design["elements"]}
    units = plan.ranked_units()
    texts = [u for u in units if any(els[i]["type"] == "text" for i in u.element_ids)]
    floor_ok = comfy = 0
    for u in texts:
        size = min(els[i]["font_size"] for i in u.element_ids if els[i]["type"] == "text")
        floor = floor_px(u.role, w, h)
        floor_ok += size >= floor - 1e-6
        comfy += size >= COMFORT * floor - 1e-6
    boxes = [union_box([rotated_aabb(els[i]) for i in u.element_ids]) for u in units]
    edge = min(min(x, y, w - x - bw, h - y - bh) for x, y, bw, bh in boxes) / min(w, h)
    return {"text_at_floor": floor_ok / max(1, len(texts)), "text_comfortable": comfy / max(1, len(texts)),
            "dead_space": _largest_empty_share(boxes, w, h), "min_edge_margin": edge}


def proxies(measurer, monkeypatch, which=lambda c: c.template != "reflow") -> dict[bool, dict[str, float]]:
    """Mean proxy metrics over every template candidate for 3 samples x 13 targets, flag off and on."""
    out = {}
    for flag in (False, True):
        if flag:
            monkeypatch.setenv(flex.ENV, "1")
        else:
            monkeypatch.delenv(flex.ENV, raising=False)
        rows = []
        for path in SAMPLES:
            src = _sample(path)
            plan = heuristic_plan(src)
            for t in TARGETS:
                tgt = {"width": t["width"], "height": t["height"]}
                rows += [proxy_metrics(c.design, plan, tgt) for c in generate_candidates(src, tgt, plan, measurer) if which(c)]
        out[flag] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    monkeypatch.delenv(flex.ENV, raising=False)
    return out


def test_proxies_improve_with_the_flag_on(measurer, monkeypatch):
    p = proxies(measurer, monkeypatch)
    off, on = p[False], p[True]
    print("\nproxy off:", off, "\nproxy on: ", on)
    assert on["text_at_floor"] >= off["text_at_floor"] - 1e-9
    assert on["text_comfortable"] > off["text_comfortable"]
    assert on["dead_space"] < off["dead_space"]
    assert on["min_edge_margin"] >= off["min_edge_margin"] - 1e-3
