"""Strategy `anchored` (engine/pipeline/layout/strategies/anchored.py):
constraint-style retargeting. Off unless named in CHAMELEON_STRATEGIES; with
it off the engine's output is byte-identical (golden hashes from git HEAD
before the strategy existed)."""

import hashlib
import json
from pathlib import Path

import pytest

from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout import strategy_registry as reg
from engine.pipeline.layout.place import floor_px
from engine.pipeline.layout.strategies import anchored
from engine.pipeline.priority import heuristic_plan
from tests.conftest import assert_valid

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())
V3L = {"CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1", "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1"}

# sha1 over (winner id, design) of resize_with_report on every target, the v3L stack with
# CHAMELEON_STRATEGIES=bisection, from the engine before this strategy existed (git HEAD 81a9b6f).
GOLDEN_V3L = {
    "chameleon_sample_level1_warmup": "000eae1dbcfd82345f9ba4773a2b1bce45ecaecd",
    "chameleon_sample_level2_core": "73f48df675757493066be73e6b330c7c57a4c60d",
    "chameleon_sample_level3_advanced": "eedac0f658e465a874e4bb19df461ce6fdac623c",
}


def _src(path: Path) -> dict:
    return json.loads(path.read_text())["source"]


def _set(monkeypatch, strategies: str, layers: bool = True):
    for k, v in V3L.items():
        if k == "CHAMELEON_LAYERS" and not layers:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    monkeypatch.setenv("CHAMELEON_STRATEGIES", strategies)
    reg._modules.clear()


def _engine_digest(src: dict) -> str:
    from engine.resize import resize_with_report
    h = hashlib.sha1()
    for t in TARGETS:
        design, _, best, _ = resize_with_report(src, {"width": t["width"], "height": t["height"]})
        h.update(best.candidate_id.encode())
        h.update(json.dumps(design, sort_keys=True).encode())
    return h.hexdigest()


# --- switch off: byte-identical --------------------------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_off_is_byte_identical(path, monkeypatch):
    _set(monkeypatch, "bisection")
    called = []
    monkeypatch.setattr(anchored, "candidates", lambda *a, **k: called.append(1) or [])
    assert _engine_digest(_src(path)) == GOLDEN_V3L[path.stem]
    assert not called


# --- anchors ------------------------------------------------------------------------

@pytest.mark.parametrize("lo,size,kind", [
    (40, 220, "start"),      # badge near the left edge
    (380, 320, "centre"),    # logo centred
    (850, 180, "end"),       # watermark near the right edge
    (90, 900, "stretch"),    # spans most of the side
    (300, 200, "centre"),    # off-centre, far from both edges: keeps its offset from the middle
])
def test_anchor_kinds(lo, size, kind):
    assert anchored.anchor(lo, size, 1080) == kind


def test_place_on_axis_pins_edges_and_scales_gaps():
    # start: gap scaled by s; end: measured from the far edge; bled pieces stay on the edge
    assert anchored.place_on_axis("start", 40, 220, 1080, 275, 1350, 1.25, 20)[0] == pytest.approx(50)
    assert anchored.place_on_axis("end", 850, 180, 1080, 225, 1350, 1.25, 20)[0] == pytest.approx(1350 - 225 - 62.5)
    assert anchored.place_on_axis("start", 0, 300, 1080, 375, 1350, 1.25, 20)[:2] == (0.0, 0.0)
    # centre: offset from the middle scaled by s (anchored) or by the axis ratio (spread)
    p = anchored.place_on_axis("centre", 290, 500, 1080, 500, 1350, 1.0, 20)[0]
    assert p + 250 == pytest.approx(675 + 0)


# --- candidates ---------------------------------------------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("layers", [False, True])
def test_candidates_pass_gates_and_respect_floors(path, layers, measurer, monkeypatch):
    _set(monkeypatch, "bisection,anchored", layers)
    src = _src(path)
    plan = heuristic_plan(src)
    roles = {i: u.role for u in plan.units for i in u.element_ids}
    total = 0
    for t in TARGETS + [{"width": 1600, "height": 400}]:
        target = {"width": t["width"], "height": t["height"]}
        cands = anchored.candidates(src, target, plan, measurer)
        assert len(cands) <= anchored.MAX_CANDIDATES
        assert len({json.dumps(c.design, sort_keys=True) for c in cands}) == len(cands)
        for c in cands:
            total += 1
            assert c.candidate_id.startswith("anchored/") and not c.is_fallback
            assert check_gates(src, c.design, target, plan, measurer) == []
            assert_valid(src, c.design, target)
            for el in c.design["elements"]:
                if el["type"] == "text":
                    assert el["font_size"] >= floor_px(roles[el["id"]], *target.values()) - 0.01, el["id"]
    assert total >= 15


def test_same_aspect_gives_nothing(l2, measurer):
    plan = heuristic_plan(l2["source"])
    assert anchored.candidates(l2["source"], {"width": 540, "height": 540}, plan, measurer) == []


def test_deterministic(l2, measurer, monkeypatch):
    _set(monkeypatch, "bisection,anchored")
    plan = heuristic_plan(l2["source"])
    a = anchored.candidates(l2["source"], {"width": 1600, "height": 400}, plan, measurer)
    b = anchored.candidates(l2["source"], {"width": 1600, "height": 400}, plan, measurer)
    assert [json.dumps(c.design, sort_keys=True) for c in a] == [json.dumps(c.design, sort_keys=True) for c in b]


def _boxes(design: dict) -> dict:
    return {e["id"]: (e["x"], e["y"], e["width"], e["height"]) for e in design["elements"]}


def test_portrait_keeps_the_composition_and_fills_the_canvas(l2, measurer, monkeypatch):
    """Square -> 4:5: the largest scale fills the width; top stays top, bottom stays
    bottom, the column keeps its order and the headline stays the largest text."""
    _set(monkeypatch, "bisection,anchored")
    src = l2["source"]
    plan = heuristic_plan(src)
    cands = {c.candidate_id: c for c in anchored.candidates(src, {"width": 1080, "height": 1350}, plan, measurer)}
    top = max(cands.values(), key=lambda c: c.params["r"])
    assert top.params["r"] == pytest.approx(1.25)
    b = _boxes(top.design)
    assert b["logo"][1] < b["product"][1] < b["headline"][1] < b["subhead"][1] <= b["cta_bg"][1]
    assert b["cta_bg"][1] + b["cta_bg"][3] > 1350 - 0.1 * 1350          # still at the bottom
    assert b["badge_bg"][0] + b["badge_bg"][2] < b["logo"][0] + b["logo"][2] / 2   # badge still up left of the logo
    assert b["badge_bg"][1] < b["product"][1]
    els = {e["id"]: e for e in top.design["elements"]}
    texts = [e for e in top.design["elements"] if e["type"] == "text"]
    assert els["headline"]["font_size"] == max(e["font_size"] for e in texts)
    assert els["headline"]["font_size"] == pytest.approx(56 * 1.25, abs=0.05)


def test_wide_strip_turns_the_stack_into_a_row_in_order(l2, measurer, monkeypatch):
    """Square -> 1600x400 at a scale above the uniform fit: the stacked
    logo / product / text column become a left-to-right row, in source order,
    and the text column stays together."""
    _set(monkeypatch, "bisection,anchored")
    src = l2["source"]
    plan = heuristic_plan(src)
    cands = anchored.candidates(src, {"width": 1600, "height": 400}, plan, measurer)
    top = max(cands, key=lambda c: c.params["r"])
    assert top.params["r"] >= 2.0
    b = _boxes(top.design)
    assert b["logo"][0] + b["logo"][2] <= b["product"][0] + 1
    assert b["product"][0] + b["product"][2] <= min(b["headline"][0], b["subhead"][0], b["cta_bg"][0]) + 1
    assert b["headline"][1] < b["subhead"][1] < b["cta_bg"][1]          # the column keeps its stacking


def test_registry_loads_it(monkeypatch, l1, measurer):
    _set(monkeypatch, "bisection,anchored")
    plan = heuristic_plan(l1["source"])
    out = reg.extra_candidates(l1["source"], {"width": 1080, "height": 1350}, plan, measurer)
    assert any(c.candidate_id.startswith("anchored/") for c in out)


@pytest.mark.parametrize("t", [{"width": 1080, "height": 1920}, {"width": 320, "height": 50}, {"width": 160, "height": 600}])
def test_resize_with_anchored_returns_a_valid_design(t, l3, monkeypatch):
    from engine.resize import resize_with_report
    _set(monkeypatch, "bisection,anchored")
    design, plan, best, _ = resize_with_report(l3["source"], t)
    assert_valid(l3["source"], design, t)


def test_never_raises(measurer):
    assert anchored.candidates({}, {"width": 10, "height": 10}, None, measurer) == []
    bad = {"canvas": {"width": 0, "height": 100}, "elements": []}
    assert anchored.candidates(bad, {"width": 100, "height": 100}, heuristic_plan(bad), measurer) == []
