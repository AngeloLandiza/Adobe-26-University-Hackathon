"""Strategy S4 `design_grid`: responsive modular grid + modular type scale
(engine/pipeline/layout/strategies/design_grid.py)."""

import json
import math
from pathlib import Path

import pytest

from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout.place import floor_px
from engine.pipeline.priority import heuristic_plan
from tests.conftest import assert_valid

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())
SOME = [t for t in TARGETS if t["name"] in ("story", "instagram_post", "youtube_thumbnail", "billboard",
                                            "medium_rectangle", "leaderboard", "wide_skyscraper", "mobile_banner")]


def _source(path):
    return json.loads(path.read_text())["source"]


def _mod():
    from engine.pipeline.layout.strategies import design_grid
    return design_grid


def test_type_scale_is_geometric_and_anchored():
    dg = _mod()
    steps = dg.type_scale(1080, 1080, 1.25)
    assert steps == sorted(steps)
    ratios = [b / a for a, b in zip(steps, steps[1:])]
    assert all(r == pytest.approx(1.25, rel=0.06) for r in ratios)  # 0.5 px rounding only
    # the smallest legal text sits on a step at (or just under) its floor, the headline floor is reachable
    assert min(steps) <= floor_px("legal", 1080, 1080) + 1e-6
    assert any(abs(s - floor_px("subheadline", 1080, 1080)) < 0.6 for s in steps)


def test_snap_down_picks_the_largest_step_not_above():
    dg = _mod()
    steps = [10, 12.5, 15.5, 19.5, 24.5]
    assert dg.snap_down(steps, 20) == 19.5
    assert dg.snap_down(steps, 24.5) == 24.5
    assert dg.snap_down(steps, 9) == 10  # below the scale: the smallest step


def test_grid_columns_snap_and_are_proportional():
    dg = _mod()
    g = dg.Grid(1080, 1920, cols=6, margin_units=4)
    assert g.margin > 0 and g.margin % g.unit == pytest.approx(0, abs=1e-6)
    assert g.col_x(0) == pytest.approx(g.margin)
    assert g.col_x(g.cols) - g.gutter == pytest.approx(1080 - g.margin)  # the last column ends on the margin
    assert g.span_w(0, g.cols) == pytest.approx(1080 - 2 * g.margin)


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem[-8:])
def test_candidates_pass_gates_and_ids(path, measurer):
    dg = _mod()
    source = _source(path)
    plan = heuristic_plan(source)
    produced = 0
    for t in SOME:
        target = {"width": t["width"], "height": t["height"]}
        cands = dg.candidates(source, target, plan, measurer)
        for c in cands:
            produced += 1
            assert c.candidate_id.startswith("design_grid/")
            assert check_gates(source, c.design, target, plan, measurer) == [], (t["name"], c.candidate_id)
            assert_valid(source, c.design, target)
            for el in c.design["elements"]:
                if el["type"] == "text":
                    assert el["font_size"] >= 1
    assert produced >= len(SOME)


def test_tall_and_square_cells_get_candidates(l2, measurer):
    dg = _mod()
    plan = heuristic_plan(l2["source"])
    for name in ("story", "instagram_portrait", "instagram_post", "half_page"):
        t = next(t for t in TARGETS if t["name"] == name)
        assert dg.candidates(l2["source"], {"width": t["width"], "height": t["height"]}, plan, measurer), name


def test_text_sizes_come_from_the_type_scale(l2, measurer):
    dg = _mod()
    plan = heuristic_plan(l2["source"])
    target = {"width": 1080, "height": 1920}
    cands = dg.candidates(l2["source"], target, plan, measurer)
    assert cands
    on_scale = 0
    total = 0
    for c in cands:
        steps = dg.type_scale(1080, 1920, c.params["ratio"])
        for el in c.design["elements"]:
            if el["type"] == "text" and plan.unit_for_element(el["id"]).role in ("headline", "subheadline"):
                total += 1
                on_scale += any(abs(el["font_size"] - s) < 0.01 for s in steps)
    assert on_scale / total >= 0.7


def test_deterministic(l3, measurer):
    dg = _mod()
    plan = heuristic_plan(l3["source"])
    target = {"width": 970, "height": 250}
    a = dg.candidates(l3["source"], target, plan, measurer)
    b = dg.candidates(l3["source"], target, plan, measurer)
    assert [(c.candidate_id, c.design) for c in a] == [(c.candidate_id, c.design) for c in b]


def test_never_raises_on_garbage(measurer, l1):
    dg = _mod()
    plan = heuristic_plan(l1["source"])
    assert dg.candidates(None, None, None, None) == []
    assert dg.candidates(l1["source"], {"width": 0, "height": 0}, plan, measurer) == []
    assert dg.candidates({"elements": []}, {"width": 100, "height": 100}, plan, measurer) == []


def test_edges_are_clear(l2, measurer):
    """Design principle: nothing touches the canvas edge (content sits inside the margin)."""
    dg = _mod()
    plan = heuristic_plan(l2["source"])
    target = {"width": 1080, "height": 1350}
    for c in dg.candidates(l2["source"], target, plan, measurer):
        for el in c.design["elements"]:
            if plan.unit_for_element(el["id"]).role in ("background", "decoration"):
                continue
            assert el["x"] >= 0.01 * 1080 and el["y"] >= 0.01 * 1080
            assert el["x"] + el["width"] <= 1080 * 0.99 and el["y"] + el["height"] <= 1350 - 0.01 * 1080
