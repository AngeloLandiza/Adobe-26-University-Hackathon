"""Bisection v2 (CHAMELEON_BISECT_V2=1) and its flaw detectors
(engine/pipeline/layout/strategies/bisection.py, dev/distill/bisect_flaws.py).

With the switch unset the strategy must produce exactly today's candidates;
with it on, every candidate still passes the hard gates and keeps the
hierarchy, and each fix does what it says on small cases.
"""

import hashlib
import json
from pathlib import Path

import pytest

from engine.pipeline.geometry import overlap_area, rotated_aabb, union_box
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout.strategies import bisection
from engine.pipeline.layout.strategies.bisection import candidates, columns, margins, straddles
from engine.pipeline.priority import heuristic_plan
from tests.conftest import assert_valid

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())

# sha1 over (candidate_id, design) of every bisection candidate for each sample
# on every target, from the strategy as it was before v2 existed.
GOLDEN = {
    "chameleon_sample_level1_warmup": "7558fb19a72a3bd6f22d54978f9c6ca46be3303b",
    "chameleon_sample_level2_core": "e0cf49388358bfb33796a4a89a6a51563d8a7af6",
    "chameleon_sample_level3_advanced": "1938a5b759dfb057dfaa71559d9b0d7d673226a0",
}


def _src(path: Path) -> dict:
    return json.loads(path.read_text())["source"]


def _digest(src: dict, measurer) -> str:
    plan = heuristic_plan(src)
    h = hashlib.sha1()
    for t in TARGETS:
        for c in candidates(src, {"width": t["width"], "height": t["height"]}, plan, measurer):
            h.update(c.candidate_id.encode())
            h.update(json.dumps(c.design, sort_keys=True).encode())
    return h.hexdigest()


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_switch_off_is_byte_identical(path, measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_BISECT_V2", raising=False)
    assert _digest(_src(path), measurer) == GOLDEN[path.stem]


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_switch_other_values_are_off(path, measurer, monkeypatch):
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "0")
    assert _digest(_src(path), measurer) == GOLDEN[path.stem]


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_v2_candidates_pass_the_gates_and_keep_the_hierarchy(path, target, measurer, monkeypatch):
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "1")
    src = _src(path)
    plan = heuristic_plan(src)
    canvas = {"width": target["width"], "height": target["height"]}
    out = candidates(src, canvas, plan, measurer)
    head = next(u for u in plan.units if u.role == "headline")
    content = plan.ranked_units()
    for c in out:
        assert c.candidate_id.startswith("bisection/")
        assert check_gates(src, c.design, canvas, plan, measurer) == [], c.candidate_id
        assert_valid(src, c.design, canvas)
        els = {e["id"]: e for e in c.design["elements"]}
        if head.unit_id not in c.degraded_unit_ids:
            size = els[head.element_ids[0]]["font_size"]
            others = [e["font_size"] for e in els.values() if e["type"] == "text" and e["id"] not in head.element_ids]
            assert all(size >= o - 1e-6 for o in others), c.candidate_id
        boxes = {u.unit_id: union_box([rotated_aabb(els[i]) for i in u.element_ids]) for u in content}
        for i, a in enumerate(content):
            for b in content[i + 1:]:
                assert overlap_area(boxes[a.unit_id], boxes[b.unit_id]) < 1.0, (c.candidate_id, a.unit_id, b.unit_id)


def test_v2_is_deterministic(l3, measurer, monkeypatch):
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "1")
    assert _digest(l3["source"], measurer) == _digest(l3["source"], measurer)


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("name", ["leaderboard", "large_leaderboard", "billboard", "story"])
def test_v2_logo_is_not_a_speck(path, name, measurer, monkeypatch):
    """The logo's sqrt(area) reaches LOGO_MIN_LONG of the long side (as seen), unless degraded."""
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "1")
    src = _src(path)
    plan = heuristic_plan(src)
    t = next(t for t in TARGETS if t["name"] == name)
    logo = next(u for u in plan.units if u.role == "logo")
    el0 = next(e for e in src["elements"] if e["id"] == logo.element_ids[0])
    model = bisection._Model(src, t, plan, measurer)
    need = model.logo_min(el0["width"], el0["height"])
    for c in candidates(src, t, plan, measurer):
        el = next(e for e in c.design["elements"] if e["id"] == logo.element_ids[0])
        assert el["height"] >= need - 0.1 or logo.unit_id in c.degraded_unit_ids, c.candidate_id


def test_margins():
    p = {"margin": 0.06}
    assert margins(p, 970, 90, False) == pytest.approx((0.06 * 90, 0.06 * 90))
    mx, my = margins(p, 970, 90, True)
    assert mx == pytest.approx(0.03 * 970) and my == pytest.approx(0.06 * 90)
    assert margins(p, 1080, 1080, True) == pytest.approx((0.06 * 1080, 0.06 * 1080))


def test_no_widow_balances_the_lines(measurer, monkeypatch):
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "1")
    el = {"id": "h", "type": "text", "content": "Cold brew, bottled this morning", "font_family": "Lora",
          "font_weight": "bold", "font_size": 40, "color": "#000", "text_align": "center",
          "x": 0, "y": 0, "width": 470, "height": 200}
    m = measurer
    # find a width where the default wrap leaves one short word on the last line
    width = next(w for w in range(300, 900, 5) if bisection._has_widow(m, el, 40, w))
    el = dict(el, width=width, height=m.block_height(el, 40, width))
    model = type("M", (), {"m": m})()
    out = bisection._no_widow(model, el, (0, 0, width, 400), 14)
    assert not bisection._has_widow(m, out, out["font_size"], out["width"])
    assert out["font_size"] >= 40 * bisection.WIDOW_SHRINK
    assert m.fits(out, out["font_size"], out["width"], out["height"])
    assert 0 <= out["x"] and out["x"] + out["width"] <= width + 1e-6


def test_straddles_uses_the_ellipse():
    circle = {"shape_type": "ellipse"}
    box = (0, 0, 100, 100)
    assert not straddles(circle, box, (40, 40, 20, 20))       # inside
    assert not straddles(circle, box, (200, 200, 10, 10))     # far away
    assert not straddles(circle, box, (0, 0, 10, 10))         # in the box's corner, off the circle
    assert straddles(circle, box, (80, 40, 40, 20))           # across the rim
    assert straddles({"shape_type": "rectangle"}, box, (0, 0, 10, 200))


def test_columns_read_side_by_side_members_as_alternatives():
    boxes = [(0, 0, 200, 40), (0, 50, 90, 30), (110, 50, 90, 30)]  # headline over a copy+button row
    cols = columns(boxes, [0, 1, 2])
    assert sorted(map(sorted, cols)) == [[0, 1], [0, 2]]


# --- the detectors ------------------------------------------------------------

def test_detectors_on_a_small_design(measurer):
    from dev.distill.bisect_flaws import detect
    from engine.pipeline.types import PriorityPlan, Unit

    src = {"canvas": {"width": 400, "height": 400}, "elements": [
        {"id": "bg", "type": "shape", "shape_type": "rectangle", "fill_color": "#fff", "x": 0, "y": 0, "width": 400, "height": 400},
        {"id": "disc", "type": "shape", "shape_type": "ellipse", "fill_color": "#fc0", "x": 200, "y": 200, "width": 150, "height": 150},
        {"id": "img", "type": "image", "asset_id": "a", "x": 225, "y": 225, "width": 100, "height": 100},
        {"id": "h", "type": "text", "content": "Hello there", "font_family": "Lora", "font_size": 40, "color": "#000",
         "x": 40, "y": 40, "width": 300, "height": 50},
        {"id": "logo", "type": "image", "asset_id": "l", "x": 40, "y": 120, "width": 60, "height": 30},
    ]}
    plan = PriorityPlan("heuristic", [Unit("bg", ["bg"], "single", "background"), Unit("disc", ["disc"], "single", "decoration"),
                                      Unit("h", ["h"], "single", "headline", 1), Unit("img", ["img"], "single", "hero_image", 2),
                                      Unit("logo", ["logo"], "single", "logo", 3)])
    out = json.loads(json.dumps(src))
    els = {e["id"]: e for e in out["elements"]}
    els["disc"].update(x=0, y=300, width=150, height=60)     # squashed, and parked away from the image
    els["logo"].update(width=8, height=4)                     # a speck
    els["h"].update(x=2)                                      # against the left edge
    d = detect(src, out, plan, measurer)
    assert d["squash"]["fire"] and d["decor_orphan"]["fire"] and d["logo_small"]["fire"] and d["edge"]["fire"]
    clean = detect(src, src, plan, measurer)
    assert not clean["squash"]["fire"] and not clean["decor_orphan"]["fire"] and not clean["logo_small"]["fire"]


def test_bisect_paragraph_takes_the_bisection_side():
    from dev.distill.bisect_flaws import bisect_paragraph
    row = {"a": "bisect", "b": "base", "reason": "A keeps the order. The logo is a speck. B scrambles everything."}
    assert bisect_paragraph(row) == "A keeps the order. The logo is a speck. "
    row = {"a": "base", "b": "bisect_chl", "reason": "A: fine. B: the CTA is oversized."}
    assert bisect_paragraph(row) == "B: the CTA is oversized."
