import json
import pytest
from pathlib import Path
from engine.pipeline.priority import heuristic_plan, _assemble
from engine.pipeline.priority.units import build_units
from dev.priority_metrics import role_accuracy, kendall_tau, top3_recall

SAMPLES = ["chameleon_sample_level1_warmup", "chameleon_sample_level2_core", "chameleon_sample_level3_advanced"]


def txt(id, y, size, content, x=100, w=800, h=60, **extra):
    return {"id": id, "type": "text", "content": content, "font_family": "Solstice Sans", "font_size": size,
            "color": "rgba(0,0,0,1)", "x": x, "y": y, "width": w, "height": h, **extra}


def img(id, x, y, w, h, z=1, asset="solstice_bottle.png", **extra):
    return {"id": id, "type": "image", "asset_id": asset, "x": x, "y": y, "width": w, "height": h, "z_index": z, **extra}


def roles_of(elements, size=1000):
    plan = heuristic_plan({"canvas": {"width": size, "height": size}, "elements": elements})
    return {u.unit_id: u.role for u in plan.units}


@pytest.mark.parametrize("stem", SAMPLES)
def test_heuristic_matches_golden(stem):
    task = json.loads(Path(f"input/{stem}.json").read_text())
    gold = json.loads(Path(f"dev/golden/{stem}.json").read_text())
    plan = heuristic_plan(task["source"])
    roles = {u.unit_id: u.role for u in plan.units}
    ranking = [u.unit_id for u in plan.ranked_units()]
    assert role_accuracy(roles, gold["roles"]) == 1.0
    assert top3_recall(ranking, gold["ranking"]) == 1.0


def test_heuristic_ranking_is_deterministic(l3):
    a = [u.unit_id for u in heuristic_plan(l3["source"]).ranked_units()]
    assert a == [u.unit_id for u in heuristic_plan(l3["source"]).ranked_units()]
    assert a[-1] == "holiday_watermark"


def test_heuristic_handles_no_text_no_images():
    shapes = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "s", "type": "shape", "shape_type": "ellipse", "fill_color": "rgba(1,2,3,1)",
         "x": 10, "y": 10, "width": 20, "height": 20}]}
    plan = heuristic_plan(shapes)
    assert plan.units[0].role == "decoration" and plan.ranked_units() == []


def test_metrics():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["c", "b", "a"], ["a", "b", "c"]) == -1.0
    assert top3_recall(["a", "x", "b", "c"], ["a", "b", "c"]) == 1.0
    assert role_accuracy({"a": "logo", "b": "cta"}, {"a": "logo", "b": "badge"}) == 0.5


# --- beyond the brief: rules and edges the three samples don't reach ---

def test_metrics_partial_agreement():
    assert kendall_tau(["a", "c", "b"], ["a", "b", "c"]) == pytest.approx(1 / 3)   # 2 concordant, 1 discordant pair
    assert kendall_tau(["a", "x", "b"], ["b", "a", "y"]) == -1.0                    # only a,b are common: reversed
    assert top3_recall(["a", "x", "y", "z", "b"], ["a", "b", "c"]) == pytest.approx(1 / 3)
    assert role_accuracy({"a": "logo"}, {"a": "logo", "b": "cta"}) == 0.5          # missing prediction counts as wrong


def test_badges_order_by_text_prominence(l3):
    # font_size x text length: badge2 15x18=270, badge1 15x15=225, ribbon 20x11=220, badge3 15x11=165
    ranking = [u.unit_id for u in heuristic_plan(l3["source"]).ranked_units()]
    assert [u for u in ranking if u in ("badge1", "badge2", "badge3", "ribbon")] == ["badge2", "badge1", "ribbon", "badge3"]


def test_legal_by_position_without_markers():
    fine_print = "Offer valid in participating stores only while supplies last and subject to local availability rules"
    roles = roles_of([txt("h", 100, 60, "Big summer savings"), txt("fine", 900, 10, fine_print)])
    assert roles["fine"] == "legal"


def test_text_far_smaller_than_headline_is_other_text():
    roles = roles_of([txt("h", 100, 60, "Headline"), txt("sub", 300, 40, "A supporting line of copy"),
                      txt("note", 500, 20, "A minor aside about the product")])
    assert roles == {"h": "headline", "sub": "subheadline", "note": "other_text"}


def test_action_text_is_cta_case_insensitively():
    roles = roles_of([txt("h", 100, 60, "Headline"), txt("go", 500, 20, "GET STARTED!"), txt("no", 600, 20, "Shop")])
    assert roles["go"] == "cta" and roles["no"] == "cta"


def test_logo_prefers_highest_z_then_largest():
    corner = dict(w=120, h=40)  # 0.5% of the canvas, aspect 3, near the top edge
    roles = roles_of([img("a", 20, 20, z=2, **corner), img("b", 800, 20, z=5, **corner), img("hero", 300, 300, 400, 400)])
    assert roles["b"] == "logo" and roles["hero"] == "hero_image" and roles["a"] == "secondary_image"
    roles = roles_of([img("small", 20, 20, 90, 30, z=3), img("big", 800, 20, 150, 50, z=3)])
    assert roles["big"] == "logo"


def test_missing_asset_image_is_never_the_hero():
    # without rule 4 the larger (missing) image would take hero_image and the real one would be demoted
    roles = roles_of([img("present", 100, 100, 200, 200), img("missing", 300, 300, 500, 500, asset="nope.png")])
    assert roles == {"present": "hero_image", "missing": "secondary_image"}


def test_generic_group_with_text_uses_text_rules():
    # an icon + caption group (not rotated, not a label-on-shape) has no rule of its own
    roles = roles_of([txt("h", 100, 60, "Headline"),
                      img("icon", 100, 400, 40, 40, group_id="perk"), txt("cap", 400, 40, "Free shipping on all orders", x=150, group_id="perk")])
    assert roles["perk"] == "subheadline"


def test_assemble_leaves_input_units_untouched(l2):
    units = build_units(l2["source"])
    roles = {u.unit_id: "headline" for u in units}
    _assemble(units, roles, [u.unit_id for u in units], "llm", None)
    assert all(u.rank is None for u in units) and units[1].role is None


def test_legal_text_is_never_the_headline():
    # the only single text is a disclaimer (the real title sits on a pill): it stays legal
    pill = {"id": "pill", "type": "shape", "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)",
            "x": 300, "y": 400, "width": 400, "height": 120, "z_index": 1}
    roles = roles_of([pill, txt("title", 420, 60, "Summer sale", x=320, w=360, h=80, z_index=2),
                      txt("disclaimer", 900, 16, "Please enjoy responsibly.", z_index=1)])
    assert roles["disclaimer"] == "legal"


def test_headline_on_a_background_shape_is_the_headline():
    from tests.designs import BG_SHAPE_HEADLINE
    roles = {u.unit_id: u.role for u in heuristic_plan(BG_SHAPE_HEADLINE).units}
    assert roles == {"bg": "background", "headline": "headline"}
