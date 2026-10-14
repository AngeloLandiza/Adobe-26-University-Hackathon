import pytest
from engine.pipeline.priority.units import build_units
from engine.pipeline.priority.features import unit_features


def by_id(units): return {u.unit_id: u for u in units}


def feats_by_id(task): return {f["unit_id"]: f for f in unit_features(task["source"], build_units(task["source"]))}


def text(id, x, y, w, h, z, content="Buy"):
    return {"id": id, "type": "text", "content": content, "font_family": "Solstice Sans", "font_size": 10,
            "color": "rgba(255,255,255,1)", "x": x, "y": y, "width": w, "height": h, "z_index": z}


def rect(id, x, y, w, h, z):
    return {"id": id, "type": "shape", "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)",
            "x": x, "y": y, "width": w, "height": h, "z_index": z}


def test_l2_units(l2):
    units = by_id(build_units(l2["source"]))
    assert len(units) == 9
    assert units["badge"].element_ids == ["badge_bg", "badge_label"]
    assert units["cta"].kind == "container_label"
    assert units["bg"].role == "background" and "background" in units["bg"].flags


def test_l3_flags(l3):
    units = by_id(build_units(l3["source"]))
    assert len(units) == 12
    assert "missing_asset" in units["holiday_watermark"].flags
    assert "rotated" in units["ribbon"].flags and units["ribbon"].kind == "container_label"


def test_every_element_in_exactly_one_unit(l1, l2, l3):
    for task in (l1, l2, l3):
        ids = [e for u in build_units(task["source"]) for e in u.element_ids]
        assert sorted(ids) == sorted(el["id"] for el in task["source"]["elements"])


def test_implicit_label_on_shape():
    src = {"canvas": {"width": 100, "height": 100}, "elements": [rect("s", 10, 10, 50, 20, 1), text("t", 12, 12, 40, 14, 2)]}
    (u,) = build_units(src)
    assert u.kind == "container_label" and u.element_ids == ["s", "t"]


def test_units_tolerate_missing_optional_fields():
    src = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "i", "type": "image", "asset_id": "solstice_bottle.png", "x": 0, "y": 0, "width": 10, "height": 10}]}
    (f,) = unit_features(src, build_units(src))
    assert f["z_index"] == 0 and f["opacity"] == 1 and f["rotation"] == 0


def test_features_l1(l1):
    feats = feats_by_id(l1)
    assert feats["headline"]["font_size_rank"] == 1
    assert feats["bg"]["is_full_canvas"] and feats["bg"]["area_pct"] == 1.0


# --- beyond the brief ---

def test_features_include_pixel_aspect(l1):
    # pre-flight ruling: the heuristic's logo rule needs width/height in px, not of the normalised bbox
    assert feats_by_id(l1)["logo"]["aspect"] == pytest.approx(3.2)


def test_panel_with_several_texts_is_not_a_button():
    # each text covers 37% of the panel, so only the "exactly one text" rule keeps them apart
    src = {"canvas": {"width": 400, "height": 400}, "elements": [
        rect("panel", 0, 0, 300, 200, 1), text("a", 10, 10, 280, 80, 2), text("b", 10, 110, 280, 80, 2)]}
    assert [u.kind for u in build_units(src)] == ["single", "single", "single"]


def test_small_text_in_large_panel_is_not_a_button():
    src = {"canvas": {"width": 400, "height": 400}, "elements": [rect("panel", 0, 0, 300, 200, 1), text("a", 10, 10, 100, 30, 2)]}
    assert [u.kind for u in build_units(src)] == ["single", "single"]


def test_overlay_above_background_is_not_background():
    src = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "bg", "type": "image", "asset_id": "solstice_bg_gradient.png", "fit": "cover",
         "x": 0, "y": 0, "width": 100, "height": 100, "z_index": 0},
        dict(rect("tint", 0, 0, 100, 100, 1), opacity=0.3)]}
    units = by_id(build_units(src))
    assert units["bg"].role == "background" and units["tint"].role is None and "background" not in units["tint"].flags


def test_features_of_a_button_and_a_badge(l2):
    feats = feats_by_id(l2)
    cta = feats["cta"]
    assert cta["types"] == ["shape", "text"] and cta["has_label"] is True
    assert cta["text"] == "Shop the Summer Line" and cta["word_count"] == 4 and cta["font_size"] == 22 and cta["bold"]
    assert cta["asset_id"] is None and cta["missing_asset"] is None
    assert feats["badge"]["all_caps"] and not cta["all_caps"]
    assert [feats[u]["font_size_rank"] for u in ("headline", "subhead", "cta", "badge")] == [1, 2, 3, 4]
    assert feats["logo"]["missing_asset"] is False and feats["logo"]["text"] is None


def test_overlaps_skip_background_and_sort_by_area(l2):
    # product (290,260,500,500) meets deco_circle_1 over 110x140 and deco_circle_2 over 90x60
    assert feats_by_id(l2)["product"]["overlaps"] == ["deco_circle_1", "deco_circle_2"]


def test_area_pct_counts_only_the_visible_part(l2):
    # deco_circle_1 spans (-100,-100)-(400,400) on a 1080 canvas: 400x400 of it is visible
    assert feats_by_id(l2)["deco_circle_1"]["area_pct"] == pytest.approx(400 * 400 / 1080 ** 2)


def test_group_id_clashing_with_an_element_id_gets_a_distinct_unit_id():
    src = {"canvas": {"width": 200, "height": 200}, "elements": [
        rect("logo", 0, 0, 50, 20, 1),
        dict(rect("b", 100, 100, 50, 20, 1), group_id="logo"), dict(text("t", 102, 102, 40, 14, 2), group_id="logo")]}
    ids = [u.unit_id for u in build_units(src)]
    assert len(ids) == len(set(ids)) == 2 and "logo" in ids


def test_missing_asset_flag_in_l3_features(l3):
    assert feats_by_id(l3)["holiday_watermark"]["missing_asset"] is True
