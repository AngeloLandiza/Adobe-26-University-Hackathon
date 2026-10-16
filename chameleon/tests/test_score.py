import copy
import pytest
from engine.pipeline.types import Candidate, PriorityPlan, Unit
from engine.pipeline.grading.metrics import WEIGHTS, geometry_metrics, geometry_total
from engine.pipeline.grading.pixels import contrast_ratio, pixel_metrics
from engine.pipeline.grading.score import pick_best
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.priority import heuristic_plan


def test_contrast_ratio():
    assert contrast_ratio((255, 255, 255), (0, 0, 0)) == pytest.approx(21.0, rel=1e-3)
    assert contrast_ratio((10, 10, 10), (10, 10, 10)) == pytest.approx(1.0)


def test_overlap_and_hierarchy_penalised(l2, measurer):
    src, tgt = l2["source"], l2["target_canvas"]; plan = heuristic_plan(src)
    good = generate_candidates(src, tgt, plan, measurer)[0]
    bad = copy.deepcopy(good); els = {e["id"]: e for e in bad.design["elements"]}
    els["subhead"].update(x=els["headline"]["x"], y=els["headline"]["y"])          # collide
    els["headline"]["font_size"], els["subhead"]["font_size"] = 8, 60               # invert sizes
    g, b = geometry_metrics(src, good, plan), geometry_metrics(src, bad, plan)
    assert b["overlap"] < g["overlap"] and b["hierarchy"] < g["hierarchy"]


def test_white_on_white_scores_low():
    design = {"canvas": {"width": 200, "height": 100}, "elements": [
        {"id": "bg", "type": "shape", "shape_type": "rectangle", "fill_color": "rgba(255,255,255,1)", "x": 0, "y": 0, "width": 200, "height": 100, "z_index": 0},
        {"id": "t", "type": "text", "content": "Hello", "font_family": "Solstice Sans", "font_size": 20, "color": "rgba(255,255,255,1)", "x": 10, "y": 10, "width": 180, "height": 30, "z_index": 1}]}
    assert pixel_metrics(design, heuristic_plan(design))["text_contrast"] < 0.3


def test_pick_best_returns_fallback_when_others_fail(l2, measurer):
    src, tgt = l2["source"], l2["target_canvas"]; plan = heuristic_plan(src)
    fb = safe_fallback(src, tgt, plan)
    broken = Candidate("broken", {"canvas": tgt, "elements": []}, "x", {})
    best, reports = pick_best(src, tgt, plan, [fb, broken], measurer)
    assert best.is_fallback and len(reports) == 2


def test_pick_best_deterministic(l2, measurer):
    src, tgt = l2["source"], l2["target_canvas"]; plan = heuristic_plan(src)
    cands = [safe_fallback(src, tgt, plan)] + generate_candidates(src, tgt, plan, measurer)
    assert pick_best(src, tgt, plan, cands, measurer)[0].candidate_id == pick_best(src, tgt, plan, cands, measurer)[0].candidate_id


# --- beyond the brief: each metric has a case with a hand-derived answer ---

def text_el(id, x, y, w, h, size, color="rgba(0,0,0,1)", z=2):
    return {"id": id, "type": "text", "content": "Sample copy", "font_family": "Solstice Sans", "font_size": size,
            "color": color, "x": x, "y": y, "width": w, "height": h, "z_index": z}


def rect(id, x, y, w, h, fill="rgba(255,255,255,1)", z=0):
    return {"id": id, "type": "shape", "shape_type": "rectangle", "fill_color": fill, "x": x, "y": y, "width": w, "height": h, "z_index": z}


def single(uid, role, rank):
    return Unit(uid, [uid], "single", role=role, rank=rank)


def cand(elements, w=1000, h=1000, degraded=()):
    return Candidate("c", {"canvas": {"width": w, "height": h}, "elements": elements}, "t", {}, list(degraded))


def test_metrics_stay_in_range_and_total_is_the_weighted_sum(l3, measurer):
    src, tgt = l3["source"], l3["target_canvas"]; plan = heuristic_plan(src)
    for c in generate_candidates(src, tgt, plan, measurer)[:3]:
        m = geometry_metrics(src, c, plan)
        assert set(m) == set(WEIGHTS) and all(0 <= v <= 1 for v in m.values())
        assert geometry_total(m) == pytest.approx(sum(WEIGHTS[k] * m[k] for k in WEIGHTS))


def test_legibility_is_zero_at_the_floor_and_one_at_comfort():
    # on a 600px canvas the headline floor is exactly 14px; comfort is 1.6 x 14 = 22.4px
    plan = PriorityPlan("heuristic", [single("h", "headline", 1)])
    src = {"canvas": {"width": 600, "height": 600}, "elements": [text_el("h", 50, 50, 500, 100, 30)]}
    at_floor = cand([text_el("h", 50, 50, 500, 100, 14)], 600, 600)
    comfy = cand([text_el("h", 50, 50, 500, 100, 22.4)], 600, 600)
    assert geometry_metrics(src, at_floor, plan)["legibility"] == pytest.approx(0)
    assert geometry_metrics(src, comfy, plan)["legibility"] == pytest.approx(1)


def test_degradation_is_the_weight_share_of_degraded_units():
    els = [text_el("h", 100, 100, 800, 100, 30), text_el("s", 100, 300, 800, 100, 20)]
    plan = PriorityPlan("heuristic", [single("h", "headline", 1), single("s", "subheadline", 2)])
    src = {"canvas": {"width": 1000, "height": 1000}, "elements": els}
    m = geometry_metrics(src, cand(els, degraded=["s"]), plan)
    assert m["degradation"] == pytest.approx(1 - 2 ** -0.7 / (1 + 2 ** -0.7))


def test_overlap_the_source_already_had_is_not_penalised():
    # text deliberately set on a photo in the original design
    els = [{"id": "photo", "type": "image", "asset_id": "solstice_bottle.png", "x": 0, "y": 0, "width": 600, "height": 600, "z_index": 1},
           text_el("h", 50, 50, 400, 100, 40)]
    src = {"canvas": {"width": 1000, "height": 1000}, "elements": els}
    plan = PriorityPlan("heuristic", [single("h", "headline", 1), single("photo", "hero_image", 2)])
    assert geometry_metrics(src, cand(copy.deepcopy(els)), plan)["overlap"] == pytest.approx(1)


def test_hierarchy_ignores_text_versus_image_size():
    # a big hero image ranked below a small headline is normal, not an inverted hierarchy
    els = [{"id": "photo", "type": "image", "asset_id": "solstice_bottle.png", "x": 0, "y": 200, "width": 800, "height": 800, "z_index": 1},
           text_el("h", 50, 50, 400, 100, 40)]
    src = {"canvas": {"width": 1000, "height": 1000}, "elements": els}
    plan = PriorityPlan("heuristic", [single("h", "headline", 1), single("photo", "hero_image", 2)])
    assert geometry_metrics(src, cand(els), plan)["hierarchy"] == pytest.approx(1)


def test_space_use_band():
    plan = PriorityPlan("heuristic", [single("h", "headline", 1)])
    small = [text_el("h", 400, 400, 100, 100, 30)]                      # covers 1% of the canvas
    snug = [text_el("h", 100, 100, 800, 800, 30)]                       # covers 64%: inside [0.45, 0.85]
    src = {"canvas": {"width": 1000, "height": 1000}, "elements": small}
    assert geometry_metrics(src, cand(small), plan)["space_use"] == pytest.approx(0.01 / 0.45)
    assert geometry_metrics(src, cand(snug), plan)["space_use"] == pytest.approx(1)


def test_alignment_counts_units_sharing_a_line():
    plan = PriorityPlan("heuristic", [single("a", "headline", 1), single("b", "subheadline", 2), single("c", "other_text", 3)])
    lined = [text_el("a", 100, 100, 300, 50, 30), text_el("b", 100, 300, 500, 50, 20), text_el("c", 100, 500, 200, 50, 15)]
    scattered = [text_el("a", 100, 100, 300, 50, 30), text_el("b", 237, 333, 511, 41, 20), text_el("c", 613, 701, 173, 29, 15)]
    src = {"canvas": {"width": 1000, "height": 1000}, "elements": lined}
    assert geometry_metrics(src, cand(lined), plan)["alignment"] == pytest.approx(1)
    assert geometry_metrics(src, cand(scattered), plan)["alignment"] == pytest.approx(0)


def test_contrast_takes_the_worse_end_of_the_backdrop():
    # dark text over a backdrop that is half white, half black
    split = {"canvas": {"width": 200, "height": 100}, "elements": [
        rect("w", 0, 0, 100, 100), rect("k", 100, 0, 100, 100, fill="rgba(0,0,0,1)"),
        text_el("t", 20, 20, 160, 30, 20, z=3)]}
    plain = {"canvas": {"width": 200, "height": 100}, "elements": [rect("w", 0, 0, 200, 100), text_el("t", 20, 20, 160, 30, 20, z=3)]}
    assert pixel_metrics(split, heuristic_plan(split))["text_contrast"] < 0.3
    light = dict(split, elements=split["elements"][:2] + [text_el("t", 20, 20, 160, 30, 20, color="rgba(255,255,255,1)", z=3)])
    assert pixel_metrics(light, heuristic_plan(light))["text_contrast"] < 0.3  # white text is at risk from the white half
    assert pixel_metrics(plain, heuristic_plan(plain))["text_contrast"] == pytest.approx(1)


def test_busy_backdrop_scores_lower_than_a_flat_one():
    stripes = [rect(f"s{i}", i * 10, 0, 5, 100, fill="rgba(0,0,0,1)", z=1) for i in range(20)]
    busy = {"canvas": {"width": 200, "height": 100}, "elements": [rect("w", 0, 0, 200, 100), *stripes, text_el("t", 20, 20, 160, 30, 20, color="rgba(220,40,40,1)", z=3)]}
    flat = {"canvas": {"width": 200, "height": 100}, "elements": [rect("w", 0, 0, 200, 100), text_el("t", 20, 20, 160, 30, 20, color="rgba(220,40,40,1)", z=3)]}
    assert pixel_metrics(busy, heuristic_plan(busy))["busyness"] < pixel_metrics(flat, heuristic_plan(flat))["busyness"] == pytest.approx(1)


def test_placeholder_is_the_area_left_to_missing_assets():
    design = {"canvas": {"width": 1000, "height": 1000}, "elements": [
        {"id": "gone", "type": "image", "asset_id": "nope.png", "x": 0, "y": 0, "width": 200, "height": 500, "z_index": 1},
        text_el("h", 300, 100, 600, 100, 40)]}
    assert pixel_metrics(design, heuristic_plan(design))["placeholder"] == pytest.approx(0.9)


def test_pick_best_renders_only_the_top_k_and_reports_in_input_order(l2, measurer):
    src, tgt = l2["source"], l2["target_canvas"]; plan = heuristic_plan(src)
    cands = [safe_fallback(src, tgt, plan)] + generate_candidates(src, tgt, plan, measurer)
    best, reports = pick_best(src, tgt, plan, cands, measurer, top_k=3)
    assert [r.candidate_id for r in reports] == [c.candidate_id for c in cands]
    rendered = [r for r in reports if r.rendered]
    assert len(rendered) == 3 and best.candidate_id in {r.candidate_id for r in rendered}
    assert min(rendered, key=lambda r: (-r.total, r.candidate_id)).candidate_id == best.candidate_id  # best total, ties by id


def test_degradation_counts_text_below_its_floor_even_unflagged(l3, measurer):
    # the safe fallback shrinks Level 3 to ~7% and flags nothing; its tiny text must still count as degraded
    src, tgt = l3["source"], l3["target_canvas"]; plan = heuristic_plan(src)
    fb = safe_fallback(src, tgt, plan, measurer)
    # every text unit is below its floor, so only the image units (ranks 2, 4 and 11 of 11) still count
    images = 2 ** -0.7 + 4 ** -0.7 + 11 ** -0.7
    assert fb.degraded_unit_ids == []
    assert geometry_metrics(src, fb, plan)["degradation"] == pytest.approx(images / sum(n ** -0.7 for n in range(1, 12)))
