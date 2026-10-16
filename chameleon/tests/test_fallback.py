import json
import pytest
from pathlib import Path
from engine.pipeline.geometry import rotated_aabb, inside_canvas
from engine.pipeline.layout.layers import place_background, place_decoration
from engine.pipeline.layout.normalize import finalize, EDITABLE
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.priority import heuristic_plan
from tests.conftest import assert_valid

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_fallback_is_valid(path, target):
    source = json.loads(path.read_text())["source"]
    assert_valid(source, safe_fallback(source, target, heuristic_plan(source)).design, target)


def test_background_covers(l2):
    assert place_background(l2["source"]["elements"][0], 1600, 400)["width"] == 1600


def test_decoration_round_and_inside(l2):
    d = place_decoration(l2["source"]["elements"][1], l2["source"]["canvas"], 1600, 400)
    assert d["width"] == pytest.approx(d["height"]) and inside_canvas(rotated_aabb(d), 1600, 400)


def test_finalize_copies_untouched_fields(l2):
    src = l2["source"]
    placed = {e["id"]: dict(e, x=1, y=1, width=10, height=10) for e in src["elements"]}
    out = finalize(src, placed, {"width": 100, "height": 100})
    for a, b in zip(out["elements"], src["elements"]):
        assert {k: v for k, v in a.items() if k not in EDITABLE} == {k: v for k, v in b.items() if k not in EDITABLE}


def test_finalize_rejects_out_of_canvas(l1):
    src = l1["source"]
    placed = {e["id"]: dict(e) for e in src["elements"]}
    assert finalize(src, placed, {"width": 10, "height": 10}) is None


def test_fallback_with_missing_optional_fields():
    src = {"canvas": {"width": 200, "height": 200}, "elements": [
        {"id": "t", "type": "text", "content": "Hi", "font_family": "Solstice Sans", "font_size": 20,
         "color": "rgba(0,0,0,1)", "x": 10, "y": 10, "width": 100, "height": 30}]}
    tgt = {"width": 728, "height": 90}
    assert_valid(src, safe_fallback(src, tgt, heuristic_plan(src)).design, tgt)


# --- beyond the brief ---

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_fallback_text_fits_and_labels_stay_in_their_shapes(path, target, measurer):
    # the fallback must pass the text-fit and group gates too, not just the structural ones
    source = json.loads(path.read_text())["source"]
    plan = heuristic_plan(source)
    els = {e["id"]: e for e in safe_fallback(source, target, plan, measurer).design["elements"]}
    for el in els.values():
        if el["type"] == "text":
            assert measurer.fits(el, el["font_size"], el["width"], el["height"]), el["id"]
    for unit in plan.units:
        if unit.kind == "container_label":
            shape, label = (els[i] for i in unit.element_ids)
            assert shape["x"] - 0.5 <= label["x"] and label["x"] + label["width"] <= shape["x"] + shape["width"] + 0.5
            assert shape["y"] - 0.5 <= label["y"] and label["y"] + label["height"] <= shape["y"] + shape["height"] + 0.5


def test_fallback_is_marked_and_named(l2):
    c = safe_fallback(l2["source"], l2["target_canvas"], heuristic_plan(l2["source"]))
    assert (c.candidate_id, c.template, c.is_fallback, c.degraded_unit_ids) == ("fallback", "fallback", True, [])


def test_finalize_rounds_positions_and_rounds_sizes_up(l1):
    src = l1["source"]
    placed = {e["id"]: dict(e, x=1.23456, y=2.0049, width=10.001, height=5.0) for e in src["elements"]}
    el = finalize(src, placed, {"width": 100, "height": 100})["elements"][0]
    assert (el["x"], el["y"], el["width"], el["height"]) == (1.23, 2.0, 10.01, 5.0)


def test_finalize_never_narrows_a_text_box(l3, measurer):
    # a condensed label is exactly as wide as its text; rounding it down would wrap it onto two lines
    src = l3["source"]
    label = next(e for e in src["elements"] if e["id"] == "cta_label")
    tw = max(measurer.line_widths(dict(label, font_size=11), 11, float("inf")))
    placed = {e["id"]: dict(e, x=1, y=1, width=10, height=10) for e in src["elements"]}
    placed["cta_label"] = dict(label, x=1, y=1, width=tw, height=20, font_size=11)
    out = next(e for e in finalize(src, placed, {"width": 728, "height": 90})["elements"] if e["id"] == "cta_label")
    assert out["width"] >= tw and measurer.line_count(out, 11, out["width"]) == 1


def test_finalize_rejects_a_missing_element(l1):
    src = l1["source"]
    placed = {e["id"]: dict(e) for e in src["elements"][1:]}
    assert finalize(src, placed, {"width": 1080, "height": 1080}) is None


def test_non_cover_background_keeps_its_aspect():
    bg = {"id": "bg", "type": "image", "asset_id": "solstice_bg_gradient.png", "x": 0, "y": 0, "width": 1080, "height": 1080}
    out = place_background(bg, 1600, 400)
    assert out["width"] / out["height"] == pytest.approx(1.0) and inside_canvas(rotated_aabb(out), 1600, 400)
    assert (out["x"], out["height"]) == (600, 400)


def test_decoration_keeps_its_nearest_corner(l2):
    # deco_circle_2 sits 80px in from the bottom-right corner; at scale 400/1080 that becomes 29.63px
    d = place_decoration(l2["source"]["elements"][2], l2["source"]["canvas"], 1600, 400)
    s = 400 / 1080
    assert d["x"] + d["width"] == pytest.approx(1600 - 80 * s) and d["y"] + d["height"] == pytest.approx(400 - 80 * s)


def test_fallback_moves_a_bleeding_button_as_one():
    # the pill starts 40px off the left edge; pulling only the pill in would shift it under its label
    src = {"canvas": {"width": 400, "height": 200}, "elements": [
        {"id": "pill", "type": "shape", "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)",
         "x": -40, "y": 10, "width": 200, "height": 60, "z_index": 1, "group_id": "btn"},
        {"id": "label", "type": "text", "content": "Buy now", "font_family": "Solstice Sans", "font_size": 20,
         "color": "rgba(255,255,255,1)", "x": 0, "y": 20, "width": 100, "height": 40, "z_index": 2, "group_id": "btn"}]}
    els = {e["id"]: e for e in safe_fallback(src, {"width": 400, "height": 200}, heuristic_plan(src)).design["elements"]}
    assert els["pill"]["x"] == 0 and els["label"]["x"] - els["pill"]["x"] == 40


def test_finalize_rounds_text_sizes_down(l1):
    src = l1["source"]
    placed = {e["id"]: dict(e, x=1, y=1, width=10, height=10) for e in src["elements"]}
    placed["headline"] = dict(placed["headline"], font_size=11.006, letter_spacing=0.019)
    out = next(e for e in finalize(src, placed, {"width": 100, "height": 100})["elements"] if e["id"] == "headline")
    assert (out["font_size"], out["letter_spacing"]) == (11.0, 0.01)


def test_fit_inside_shrinks_what_is_too_big():
    from engine.pipeline.layout.layers import fit_inside
    (out,) = fit_inside([{"id": "c", "type": "shape", "x": -50, "y": -50, "width": 500, "height": 500}], 400, 200)
    assert inside_canvas(rotated_aabb(out), 400, 200) and out["width"] == pytest.approx(200) and out["height"] == pytest.approx(200)


def test_fallback_fixes_a_source_text_box_that_is_too_tight(measurer):
    # a 40px headline in a 30px-tall box overflows even before resizing; the fallback must still fit it
    src = {"canvas": {"width": 800, "height": 400}, "elements": [
        {"id": "h", "type": "text", "content": "Summer sale", "font_family": "Solstice Serif", "font_size": 40,
         "color": "rgba(0,0,0,1)", "x": 100, "y": 100, "width": 600, "height": 30}]}
    el = safe_fallback(src, {"width": 800, "height": 400}, heuristic_plan(src), measurer).design["elements"][0]
    assert el["font_size"] < 40 and measurer.fits(el, el["font_size"], el["width"], el["height"])
