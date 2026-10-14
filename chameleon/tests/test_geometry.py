import pytest
from engine.pipeline.geometry import element_box, rotated_aabb, overlap_area, union_box, fit_aspect, inside_canvas
from engine.pipeline.types import PriorityPlan, Unit


def test_rotated_aabb_matches_ribbon():
    x, y, w, h = rotated_aabb({"x": 740, "y": 30, "width": 300, "height": 54, "rotation": -12})
    assert w == pytest.approx(304.67, abs=0.05) and h == pytest.approx(115.19, abs=0.05)
    assert x + w / 2 == pytest.approx(890) and y + h / 2 == pytest.approx(57)


def test_unrotated_aabb_is_the_box():
    assert rotated_aabb({"x": 1, "y": 2, "width": 3, "height": 4}) == (1, 2, 3, 4)


def test_overlap_area():
    assert overlap_area((0, 0, 10, 10), (5, 5, 10, 10)) == 25
    assert overlap_area((0, 0, 10, 10), (10, 0, 5, 5)) == 0
    assert overlap_area((0, 0, 10, 10), (20, 20, 5, 5)) == 0  # gap on both axes: negatives must not multiply


def test_fit_aspect():
    assert fit_aspect(1.0, (0, 0, 200, 100)) == (50, 0, 100, 100)
    assert fit_aspect(1.0, (0, 0, 200, 100), align="left") == (0, 0, 100, 100)


def test_inside_canvas_tolerance():
    assert inside_canvas((-0.4, 0, 10, 10), 100, 100)
    assert not inside_canvas((-1, 0, 10, 10), 100, 100)


def test_plan_helpers():
    plan = PriorityPlan("heuristic", [
        Unit("bg", ["bg"], "single", role="background"),
        Unit("h", ["h"], "single", role="headline", rank=1),
        Unit("c", ["c1", "c2"], "container_label", role="cta", rank=2)])
    assert [u.unit_id for u in plan.ranked_units()] == ["h", "c"]
    assert plan.background_unit_ids() == ["bg"]
    assert plan.unit_for_element("c2").unit_id == "c"
    assert plan.weight("h") == 1.0 and plan.weight("bg") == 0.0
    assert plan.weight("c") == pytest.approx(2 ** -0.7)


# --- beyond the brief: every Task 1 function gets a test that can fail ---

def test_element_box_ignores_rotation_and_union_spans_all():
    assert element_box({"x": 1, "y": 2, "width": 3, "height": 4, "rotation": 30}) == (1, 2, 3, 4)
    assert union_box([(0, 0, 10, 10), (20, 5, 5, 10)]) == (0, 0, 25, 15)


def test_fit_aspect_other_alignments():
    assert fit_aspect(1.0, (0, 0, 200, 100), align="right") == (100, 0, 100, 100)
    assert fit_aspect(2.0, (0, 0, 100, 100), align="top") == (0, 0, 100, 50)
    assert fit_aspect(2.0, (0, 0, 100, 100), align="bottom") == (0, 50, 100, 50)
    assert fit_aspect(2.0, (10, 10, 100, 100)) == (10, 35, 100, 50)


def test_inside_canvas_far_edges():
    assert inside_canvas((90.4, 90.4, 10, 10), 100, 100)
    assert not inside_canvas((91, 0, 10, 10), 100, 100)
    assert not inside_canvas((0, 91, 10, 10), 100, 100)


def test_ranked_units_sorted_and_decoration_ids():
    plan = PriorityPlan("llm", [
        Unit("d", ["d"], "single", role="decoration"),
        Unit("b", ["b"], "single", role="badge", rank=2),
        Unit("a", ["a"], "single", role="headline", rank=1)])
    assert [u.unit_id for u in plan.ranked_units()] == ["a", "b"]
    assert plan.decoration_unit_ids() == ["d"]
