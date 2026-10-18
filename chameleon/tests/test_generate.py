import json
import pytest
from pathlib import Path
from engine.pipeline.geometry import inside_canvas, overlap_area, rotated_aabb, union_box
from engine.pipeline.layout.generate import assign, generate_candidates, reflow_candidate
from engine.pipeline.layout.place import HARD_FLOOR, floor_px
from engine.pipeline.layout.templates import Slot, build_slots, PARAM_GRID
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.types import PriorityPlan, Unit
from tests.conftest import assert_valid

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_candidates_are_valid_and_text_fits(path, target, measurer):
    """Also: no placed image is shrunk below the hard floor (6 px short side,
    or its own thickness if thinner), so no unit is hidden as a speck."""
    src = json.loads(path.read_text())["source"]
    by_id = {e["id"]: e for e in src["elements"]}
    plan = heuristic_plan(src)
    layered = {i for u in plan.units if u.unit_id in plan.background_unit_ids() + plan.decoration_unit_ids() for i in u.element_ids}
    for c in generate_candidates(src, target, plan, measurer):
        assert_valid(src, c.design, target)
        for el in c.design["elements"]:
            if el["type"] == "text":
                assert measurer.fits(el, el["font_size"], el["width"], el["height"]), (c.candidate_id, el["id"])
            elif el["type"] == "image" and el["id"] not in layered:
                s = by_id[el["id"]]
                assert min(el["width"], el["height"]) >= min(HARD_FLOOR, s["width"], s["height"]), (c.candidate_id, el["id"])


def test_ids_unique_and_deterministic(l2, measurer):
    run = lambda: [c.candidate_id for c in generate_candidates(l2["source"], l2["target_canvas"], heuristic_plan(l2["source"]), measurer)]
    a = run()
    assert a == run() and len(a) == len(set(a)) and any(i.startswith("row/") for i in a)


def test_cta_never_degraded(l3, measurer):
    for c in generate_candidates(l3["source"], l3["target_canvas"], heuristic_plan(l3["source"]), measurer):
        assert "cta" not in c.degraded_unit_ids


def test_reflow_present_for_l1(l1, measurer):
    cands = generate_candidates(l1["source"], l1["target_canvas"], heuristic_plan(l1["source"]), measurer)
    assert cands[0].candidate_id == "reflow"


@pytest.mark.parametrize("template", ["stack", "columns", "row"])
@pytest.mark.parametrize("w,h", [(728, 90), (1080, 1350), (1600, 400)])
def test_slots_inside_canvas(template, w, h):
    for params in PARAM_GRID[template]:
        assert all(inside_canvas(s.box, w, h) for s in build_slots(template, w, h, params))


# --- beyond the brief ---

def unit(uid, role, rank):
    return Unit(uid, [uid], "single", role=role, rank=rank)


def test_assign_by_rank_with_cutoff_clusters_and_cta_exemption():
    slots = [Slot("text", (0, 0, 10, 10), ("headline", "subheadline", "other_text"), stack=True),
             Slot("cta", (0, 0, 10, 10), ("cta",)), Slot("minor", (0, 0, 10, 10), (), minor=True)]
    h, s, b1, b2, c = unit("h", "headline", 1), unit("s", "subheadline", 2), unit("b1", "badge", 3), unit("b2", "badge", 4), unit("c", "cta", 5)
    got = assign(PriorityPlan("heuristic", [h, s, b1, b2, c]), slots, cutoff=2)
    ids = {i: [[u.unit_id for u in item] for item in items] for i, items in got.items()}
    assert ids == {0: [["h"], ["s"]], 1: [["c"]], 2: [["b1", "b2"]]}


def test_only_consecutive_same_role_units_cluster():
    slots = [Slot("minor", (0, 0, 10, 10), (), minor=True)]
    plan = PriorityPlan("heuristic", [unit("b1", "badge", 1), unit("o", "other_text", 2), unit("b2", "badge", 3)])
    assert [[u.unit_id for u in item] for item in assign(plan, slots, cutoff=None)[0]] == [["b1"], ["o"], ["b2"]]
    texts = PriorityPlan("heuristic", [unit("s1", "subheadline", 1), unit("s2", "subheadline", 2)])
    assert [[u.unit_id for u in item] for item in assign(texts, slots, cutoff=None)[0]] == [["s1"], ["s2"]]  # text stacks, never clusters


def test_leaderboard_gets_real_candidates_for_level_3(l3, measurer):
    # 17 elements into 728x90: the 150-character headline is degraded where it can't meet its 14px floor, not
    # dropped (rows that give it a whole slot push other units below the hard floor instead)
    cands = generate_candidates(l3["source"], l3["target_canvas"], heuristic_plan(l3["source"]), measurer)
    rows = [c for c in cands if c.template == "row"]
    headline = lambda c: next(e for e in c.design["elements"] if e["id"] == "headline")
    assert rows and all(headline(c)["font_size"] >= 14 or "headline" in c.degraded_unit_ids for c in rows)


def test_minor_units_stay_at_or_below_their_floor(l3, measurer):
    for c in generate_candidates(l3["source"], l3["target_canvas"], heuristic_plan(l3["source"]), measurer):
        legal = next(e for e in c.design["elements"] if e["id"] == "legal_text")
        assert legal["font_size"] <= floor_px("legal", 728, 90) + 1e-9, c.candidate_id
        mark = next(e for e in c.design["elements"] if e["id"] == "holiday_watermark")
        assert mark["height"] <= 90 / 3 + 0.01, c.candidate_id  # a leftover image stays small


def test_logo_never_grows_past_one_and_a_half(l1, measurer):
    story = {"width": 1080, "height": 1920}
    for c in generate_candidates(l1["source"], story, heuristic_plan(l1["source"]), measurer):
        logo = next(e for e in c.design["elements"] if e["id"] == "logo")
        assert logo["width"] <= 320 * 1.5 + 0.01, c.candidate_id


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_reflow_adds_no_new_overlaps(path, target, measurer):
    src = json.loads(path.read_text())["source"]
    plan = heuristic_plan(src)
    c = reflow_candidate(src, target, plan, measurer)
    if c is None:
        return
    src_els = {e["id"]: e for e in src["elements"]}
    out_els = {e["id"]: e for e in c.design["elements"]}
    box = lambda els, u: union_box([rotated_aabb(els[i]) for i in u.element_ids])
    content = plan.ranked_units()
    for i, a in enumerate(content):
        for b in content[i + 1:]:
            if overlap_area(box(src_els, a), box(src_els, b)) == 0:
                assert overlap_area(box(out_els, a), box(out_els, b)) < 1.0, (a.unit_id, b.unit_id)


def test_design_with_missing_optional_fields(measurer):
    src = {"canvas": {"width": 1080, "height": 1080}, "elements": [
        {"id": "t", "type": "text", "content": "Fresh summer flavours", "font_family": "Solstice Serif", "font_size": 60,
         "color": "rgba(0,0,0,1)", "x": 100, "y": 600, "width": 880, "height": 100},
        {"id": "i", "type": "image", "asset_id": "solstice_bottle.png", "x": 290, "y": 60, "width": 500, "height": 500}]}
    for target in ({"width": 728, "height": 90}, {"width": 1080, "height": 1920}, {"width": 1280, "height": 720}):
        cands = generate_candidates(src, target, heuristic_plan(src), measurer)
        assert cands
        for c in cands:
            assert_valid(src, c.design, target)


def test_stack_shrinks_lowest_rank_first_so_everything_fits(l3, measurer):
    # a weighted split gives the headline 76% of a 299x58 column and leaves the subhead too little;
    # both fit if the subhead drops to its minimum first and the headline gets the rest
    from engine.pipeline.layout.generate import _Builder
    plan = heuristic_plan(l3["source"])
    units = {u.unit_id: u for u in plan.units}
    b = _Builder(l3["source"], l3["target_canvas"], plan, measurer)
    b.place_stack([units["headline"], units["subhead"]], (266.7, 3.6, 299.1, 58.0))
    assert b.leftover == [] and {"headline", "subhead"} <= set(b.placed)
    head, sub = b.placed["headline"], b.placed["subhead"]
    assert head["font_size"] >= sub["font_size"] and head["y"] + head["height"] <= sub["y"] + 0.01


def test_strip_logos_align_to_the_top():
    logo = next(s for s in build_slots("row", 728, 90, PARAM_GRID["row"][0]) if s.name == "logo")
    assert logo.align == "top"


def builder(task, target, measurer):
    from engine.pipeline.layout.generate import _Builder
    plan = heuristic_plan(task["source"])
    return _Builder(task["source"], target, plan, measurer), {u.unit_id: u for u in plan.units}


def test_leftovers_go_biggest_first(l3, measurer):
    # one 200x30 gap fits the legal text, one 80x25 gap fits a badge; in rank order the badge
    # would take the big gap first and the legal text would have nowhere to go
    b, units = builder(l3, {"width": 400, "height": 100}, measurer)
    b.boxes = [(200, 0, 200, 60), (0, 30, 300, 70), (380, 60, 20, 40), (300, 85, 80, 15)]
    b.leftover = [units["badge1"], units["legal_text"]]
    assert b.place_leftovers(margin=0) is True


def test_free_rects_are_distinct_regions(l3, measurer):
    b, _ = builder(l3, {"width": 728, "height": 90}, measurer)
    b.boxes = [(4, 30, 100, 30), (150, 4, 80, 82), (300, 4, 250, 50)]
    rects = b.free_rects(margin=4)
    inside = lambda o, i: o[0] <= i[0] and o[1] <= i[1] and i[0] + i[2] <= o[0] + o[2] and i[1] + i[3] <= o[1] + o[3]
    assert rects and not any(inside(a, c) for a in rects for c in rects if a is not c)


def test_leftover_images_stay_small(l3, measurer):
    b, units = builder(l3, {"width": 728, "height": 90}, measurer)
    b.leftover = [units["holiday_watermark"]]
    assert b.place_leftovers(margin=4) and b.placed["holiday_watermark"]["height"] <= 90 / 3 + 0.01


def _centre(el): return (el["x"] + el["width"] / 2, el["y"] + el["height"] / 2)


def _drawn(el, measurer):
    """Where a text's glyphs are drawn: its widest line placed per text_align, and its measured block."""
    w = min(max(measurer.line_widths(el, el["font_size"], el["width"])), el["width"])
    x = {"center": el["x"] + (el["width"] - w) / 2, "right": el["x"] + el["width"] - w}.get(el.get("text_align"), el["x"])
    return (x, el["y"], w, measurer.block_height(el, el["font_size"], el["width"]))


def test_same_aspect_target_keeps_the_source_layout(l2, measurer):
    # a 1080x1080 design resized to 1080x1080 should come back as itself, not rebuilt as a stack
    src = l2["source"]
    cands = generate_candidates(src, dict(src["canvas"]), heuristic_plan(src), measurer)
    assert [c.candidate_id for c in cands] == ["reflow"]
    out = {e["id"]: e for e in cands[0].design["elements"]}
    for el in src["elements"]:
        if not inside_canvas(rotated_aabb(el), src["canvas"]["width"], src["canvas"]["height"]):
            continue  # the corner-bleeding circle must move inside: nothing off-canvas
        o = out[el["id"]]
        # text must be drawn where it was (its box may be tighter); everything else keeps its centre
        if el["type"] == "text":
            moved = (_drawn(o, measurer)[0] - _drawn(el, measurer)[0], o["y"] - el["y"])
        else:
            moved = (_centre(o)[0] - _centre(el)[0], _centre(o)[1] - _centre(el)[1])
        assert abs(moved[0]) <= 2 and abs(moved[1]) <= 2, (el["id"], moved)
        if el["type"] == "text":  # never bigger, except up to the role's readability floor
            role = heuristic_plan(src).unit_for_element(el["id"]).role
            assert o["font_size"] <= max(el["font_size"], floor_px(role, 1080, 1080)) + 1e-6, el["id"]


def test_reflow_scales_text_with_the_layout(l1, measurer):
    # at half size the headline is about half its source size, not filled up to 1.5x the source
    src = l1["source"]
    reflow = reflow_candidate(src, {"width": 540, "height": 540}, heuristic_plan(src), measurer)
    sizes = {e["id"]: e["font_size"] for e in src["elements"] if e["type"] == "text"}
    for el in reflow.design["elements"]:
        if el["type"] == "text":
            assert el["font_size"] <= max(0.5 * sizes[el["id"]], floor_px("cta", 540, 540)) + 1e-6, el["id"]


@pytest.mark.parametrize("w,h", [
    (300, 600), (970, 90),
    # IAB mobile banners: the CTA misses its slot (on Linux at 320x50 too, where text measures wider)
    # and long low-priority text filled every gap before it
    (300, 50), (320, 50),
])
def test_dense_design_in_a_small_format_still_gets_real_layouts(w, h, l3, measurer):
    # L3's 17 elements: a leftover that can't fit at the hard floor goes smaller, rather than sinking
    # every template and leaving only the fallback (all text at ~1 px in a 320x50 mobile banner)
    from engine.resize import resize_with_report
    src, target = l3["source"], {"width": w, "height": h}
    cands = generate_candidates(src, target, heuristic_plan(src), measurer)
    assert any(c.candidate_id != "reflow" for c in cands)
    for c in cands:
        assert_valid(src, c.design, target)
    assert not resize_with_report(src, target)[2].is_fallback


@pytest.mark.parametrize("w,h", [(728, 90), (160, 600), (1584, 396), (1080, 1350)])
@pytest.mark.parametrize("name,held", [("PANEL_BEHIND_COPY", ["headline", "sub"]), ("HEADLINE_CARD", ["headline"])])
def test_a_backdrop_stays_behind_its_copy(name, held, w, h, measurer):
    # a panel or card under text in the source follows that text; parked in a corner it leaves
    # black copy on a dark photo
    from engine.pipeline.geometry import element_box
    from engine.pipeline.occlusion import contains
    import tests.designs as designs
    src, target = getattr(designs, name), {"width": w, "height": h}
    backdrop = src["elements"][2]["id"] if name == "HEADLINE_CARD" else "panel"
    drawn = lambda el: _drawn(el, measurer)

    def drawn_width(els):
        boxes = [drawn(els[t]) for t in held]
        return max(x + w for x, _, w, _ in boxes) - min(x for x, _, _, _ in boxes)

    src_els = {e["id"]: e for e in src["elements"]}
    pad = src_els[backdrop]["width"] - drawn_width(src_els)  # the source's side padding around the text
    cands = generate_candidates(src, target, heuristic_plan(src), measurer)
    assert cands
    for c in cands:
        els = {e["id"]: e for e in c.design["elements"]}
        for t in held:
            assert contains(element_box(els[backdrop]), drawn(els[t]), tol=0.5), (c.candidate_id, t)
        # it hugs the text, padded like the source in proportion to the font, rather than spanning
        # the text's whole slot
        k = max(els[t]["font_size"] / src_els[t]["font_size"] for t in held)
        assert els[backdrop]["width"] <= drawn_width(els) + pad * k + 1, c.candidate_id



def test_crowded_banner_gives_the_cta_room_first(l3, measurer):
    # at 300x50 the usual leftover pass fails in every template; the retry goes in rank order,
    # so the CTA (rank 3) gets a gap before the long subhead and legal text fill them
    src = l3["source"]
    cands = [c for c in generate_candidates(src, {"width": 300, "height": 50}, heuristic_plan(src), measurer)
             if c.candidate_id != "reflow"]
    assert len(cands) >= 10
    for c in cands:
        size = {e["id"]: e["font_size"] for e in c.design["elements"] if e["type"] == "text"}
        assert size["cta_label"] >= max(size["subhead"], size["legal_text"]), c.candidate_id
