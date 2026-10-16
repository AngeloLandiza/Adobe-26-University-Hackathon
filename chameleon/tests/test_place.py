import pytest
from engine.pipeline.geometry import inside_canvas, rotated_aabb
from engine.pipeline.layout.aspect import aspect_class
from engine.pipeline.layout.place import place_unit, floor_px, HARD_FLOOR, MAX_TEXT_GROWTH
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.types import Unit


@pytest.mark.parametrize("w,h,cls", [(160, 600, "very_tall"), (1080, 1350, "tall"), (1080, 1920, "tall"),
    (1080, 1080, "square"), (300, 250, "square"), (1280, 720, "landscape"), (820, 312, "wide"),
    (1600, 400, "wide"), (728, 90, "strip")])
def test_aspect_class(w, h, cls): assert aspect_class(w, h) == cls


def setup(task):
    els = {e["id"]: e for e in task["source"]["elements"]}
    p = heuristic_plan(task["source"])
    return els, {u.unit_id: u for u in p.units}


def test_image_keeps_source_aspect(l2, measurer):
    els, units = setup(l2)
    out = place_unit(units["product"], els, (0, 0, 300, 100), measurer, floor=0)["product"]
    assert out["width"] / out["height"] == pytest.approx(1.0) and out["height"] == 100


def test_text_fits_tight(l2, measurer):
    els, units = setup(l2)
    out = place_unit(units["headline"], els, (0, 0, 600, 120), measurer, floor=14)["headline"]
    assert measurer.fits(out, out["font_size"], out["width"], out["height"])
    assert out["font_size"] <= 56 * MAX_TEXT_GROWTH
    assert out["height"] == pytest.approx(measurer.block_height(out, out["font_size"], 600))


def test_button_scales_rigidly(l2, measurer):
    els, units = setup(l2)
    out = place_unit(units["cta"], els, (0, 0, 300, 80), measurer, floor=8)
    bg, lab = out["cta_bg"], out["cta_label"]
    k = bg["width"] / 420
    assert bg["height"] == pytest.approx(90 * k) and bg["corner_radius"] == pytest.approx(45 * k)
    assert bg["x"] <= lab["x"] and lab["x"] + lab["width"] <= bg["x"] + bg["width"] + 0.01


def test_button_condenses_in_strip(l3, measurer):
    els, units = setup(l3)
    out = place_unit(units["cta"], els, (0, 0, 158, 82), measurer, floor=11)
    assert out is not None
    bg, lab = out["cta_bg"], out["cta_label"]
    assert lab["font_size"] >= 11 and measurer.line_count(lab, lab["font_size"], lab["width"]) == 1
    assert bg["width"] <= 158 and bg["x"] <= lab["x"] and lab["x"] + lab["width"] <= bg["x"] + bg["width"] + 0.01


def test_returns_none_below_floor(l3, measurer):
    els, units = setup(l3)
    assert place_unit(units["headline"], els, (0, 0, 60, 10), measurer, floor=14) is None


def test_overlong_word_never_overflows(l1, measurer):
    els, units = setup(l1)
    els["cta"] = dict(els["cta"], content="Supercalifragilisticexpialidocious")
    out = place_unit(units["cta"], els, (0, 0, 80, 40), measurer, floor=9)
    assert out is None or measurer.longest_word_width(out["cta"], out["cta"]["font_size"]) <= 80


def test_floor_px_scales():
    assert floor_px("headline", 728, 90) == 14
    assert floor_px("headline", 1080, 1920) == pytest.approx(14 * 1080 / 600)


# --- beyond the brief ---

def test_text_and_image_boxes_stay_inside_the_slot(l2, measurer):
    els, units = setup(l2)
    slot = (40, 30, 600, 120)
    for uid in ("headline", "product", "subhead"):
        for el in place_unit(units[uid], els, slot, measurer, floor=8).values():
            x, y, w, h = slot
            assert el["x"] >= x - 0.01 and el["y"] >= y - 0.01
            assert el["x"] + el["width"] <= x + w + 0.01 and el["y"] + el["height"] <= y + h + 0.01


def test_image_alignment_is_passed_through(l2, measurer):
    els, units = setup(l2)
    assert place_unit(units["product"], els, (0, 0, 300, 100), measurer, floor=0, align="left")["product"]["x"] == 0
    assert place_unit(units["product"], els, (0, 0, 300, 100), measurer, floor=0, align="right")["product"]["x"] == 200


def test_cover_image_takes_the_whole_slot(measurer):
    el = {"id": "photo", "type": "image", "asset_id": "solstice_bottle.png", "fit": "cover", "x": 0, "y": 0, "width": 400, "height": 100}
    out = place_unit(Unit("photo", ["photo"], "single", role="hero_image"), {"photo": el}, (10, 20, 150, 150), measurer, floor=0)
    assert (out["photo"]["x"], out["photo"]["y"], out["photo"]["width"], out["photo"]["height"]) == (10, 20, 150, 150)


def test_scaled_letter_spacing_still_fits(measurer):
    # fit_font_size measures with the source spacing; growing the font grows the spacing too
    el = {"id": "t", "type": "text", "content": "MMMMMMMMMM", "font_family": "Solstice Sans", "font_size": 20,
          "letter_spacing": 20, "color": "rgba(0,0,0,1)", "x": 0, "y": 0, "width": 100, "height": 40}
    tight = max(measurer.line_widths(dict(el, letter_spacing=0), 30, float("inf"))) + 9 * 20 + 5  # fits 30px only at the old spacing
    out = place_unit(Unit("t", ["t"], "single", role="headline"), {"t": el}, (0, 0, tight, 300), measurer, floor=6)["t"]
    assert out["letter_spacing"] == pytest.approx(20 * out["font_size"] / 20)
    assert measurer.fits(out, out["font_size"], out["width"], out["height"])


def test_text_units_never_grow_past_the_cap(l2, measurer):
    els, units = setup(l2)
    out = place_unit(units["badge"], els, (0, 0, 1000, 1000), measurer, floor=8)
    assert out["badge_label"]["font_size"] <= 18 * MAX_TEXT_GROWTH + 1e-9
    assert out["badge_bg"]["width"] <= 220 * MAX_TEXT_GROWTH + 1e-9


def test_label_keeps_its_visual_centre_in_the_pill(l2, measurer):
    els, units = setup(l2)
    src_label = els["cta_label"]
    src_centre = (975 - 950 + measurer.block_height(src_label, 22, 420) / 2) / 90  # share of the pill's height
    out = place_unit(units["cta"], els, (0, 0, 300, 80), measurer, floor=8)
    bg, lab = out["cta_bg"], out["cta_label"]
    assert (lab["y"] + lab["height"] / 2 - bg["y"]) / bg["height"] == pytest.approx(src_centre, abs=0.01)
    assert lab["height"] == pytest.approx(measurer.block_height(lab, lab["font_size"], lab["width"]))


def test_rotated_group_stays_inside_its_slot(l3, measurer):
    els, units = setup(l3)
    slot = (10, 5, 200, 60)
    out = place_unit(units["ribbon"], els, slot, measurer, floor=6)
    assert out is not None
    for el in out.values():
        x, y, w, h = rotated_aabb(el)
        assert inside_canvas((x - slot[0], y - slot[1], w, h), slot[2], slot[3])
        assert el["rotation"] == -12


def test_rotated_single_stays_inside_its_slot(measurer):
    el = {"id": "tag", "type": "image", "asset_id": "solstice_logo_lockup.png", "x": 0, "y": 0, "width": 300, "height": 100, "rotation": 30}
    out = place_unit(Unit("tag", ["tag"], "single", role="logo", flags=["rotated"]), {"tag": el}, (0, 0, 200, 100), measurer, floor=0)["tag"]
    assert inside_canvas(rotated_aabb(out), 200, 100) and out["width"] / out["height"] == pytest.approx(3.0)


def test_group_with_text_below_floor_returns_none(measurer):
    icon = {"id": "icon", "type": "image", "asset_id": "solstice_bottle.png", "x": 0, "y": 0, "width": 40, "height": 40, "group_id": "perk"}
    cap = {"id": "cap", "type": "text", "content": "Free shipping on all orders", "font_family": "Solstice Sans", "font_size": 20,
           "color": "rgba(0,0,0,1)", "x": 50, "y": 5, "width": 400, "height": 30, "group_id": "perk"}
    unit = Unit("perk", ["icon", "cap"], "group", role="subheadline")
    assert place_unit(unit, {"icon": icon, "cap": cap}, (0, 0, 45, 5), measurer, floor=10) is None


def test_max_font_caps_text_and_whole_buttons(l2, measurer):
    # minor and free-space units must stay small, however much room there is
    els, units = setup(l2)
    head = place_unit(units["headline"], els, (0, 0, 1000, 400), measurer, floor=6, max_font=10)["headline"]
    assert head["font_size"] == 10
    btn = place_unit(units["cta"], els, (0, 0, 1000, 400), measurer, floor=6, max_font=9)
    assert btn["cta_label"]["font_size"] <= 9 and btn["cta_bg"]["height"] <= 90 * 9 / 22 + 1e-6


def test_condensed_pill_compresses_to_a_short_slot(l3, measurer):
    # at 6px the source pill ratio (50/15) wants a 20px shape; an 18px gap still holds the label plus padding
    els, units = setup(l3)
    out = place_unit(units["badge1"], els, (0, 0, 100, 18), measurer, floor=6, max_font=6)
    assert out is not None
    bg, lab = out["badge1_bg"], out["badge1_label"]
    assert bg["height"] <= 18 and bg["y"] <= lab["y"] and lab["y"] + lab["height"] <= bg["y"] + bg["height"]


def test_rotated_pill_condenses_inside_its_slot(l3, measurer):
    # the -12deg ribbon at 6px: its rotated bounds must fit the slot, not just the unrotated pill
    els, units = setup(l3)
    slot = (0, 0, 100, 26)
    out = place_unit(units["ribbon"], els, slot, measurer, floor=6, max_font=6)
    assert out is not None and out["ribbon_label"]["font_size"] == 6
    for el in out.values():
        assert inside_canvas(rotated_aabb(el), 100, 26) and el["rotation"] == -12


def test_image_never_shrinks_below_the_hard_floor(l3, measurer):
    """A sliver of room must not turn an image into an invisible speck (found
    in the Linux container: the L3 watermark placed at 0.04 x 0.02 px)."""
    els, units = setup(l3)  # holiday_watermark is 180 x 80: 6 px tall at 13.5 px wide
    assert place_unit(units["holiday_watermark"], els, (0, 0, 13.4, 40), measurer, floor=0) is None
    out = place_unit(units["holiday_watermark"], els, (0, 0, 13.6, 40), measurer, floor=0)["holiday_watermark"]
    assert min(out["width"], out["height"]) >= HARD_FLOOR


def test_image_thinner_than_the_floor_keeps_its_own_thickness(measurer):
    el = {"id": "rule", "type": "image", "asset_id": "solstice_bottle.png", "x": 0, "y": 0, "width": 400, "height": 4}
    unit = Unit("rule", ["rule"], "single", role="secondary_image")
    assert place_unit(unit, {"rule": el}, (0, 0, 400, 100), measurer, floor=0)["rule"]["height"] == 4
    assert place_unit(unit, {"rule": el}, (0, 0, 300, 100), measurer, floor=0) is None
