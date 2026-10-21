"""Strategy S3 `rule_repair`: the rulebook's penalty function (rules.py) and
the local-search repair pass (rule_repair.refine)."""

import copy
import json

import pytest

from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.layout.strategies import rules
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.types import Candidate
from tests.designs import design, image, shape, text


def _plan(source):
    return heuristic_plan(source)


def _tiny_text_design():
    """A headline and a CTA; the second copy shrinks the CTA label to 4 px."""
    src = design(
        shape("bg", 0, 0, 1080, 1080, z_index=0),
        text("headline", "Summer sale starts now", 90, 300, 900, 300, 100, z_index=1, text_align="center"),
        shape("cta_bg", 390, 760, 300, 90, z_index=1, fill_color="rgba(0,0,0,1)", corner_radius=20),
        text("cta_label", "Shop now", 420, 780, 240, 50, 36, z_index=2, color="rgba(255,255,255,1)",
             text_align="center"),
    )
    return src


# ---------------------------------------------------------------- rules

def test_measure_returns_every_declared_measure(l2, measurer):
    source = l2["source"]
    plan = _plan(source)
    m = rules.measure(source, source, plan, measurer)
    assert set(rules.MEASURES) <= set(m)
    assert all(v is None or isinstance(v, float) for v in m.values())


def test_tiny_text_is_penalised(measurer):
    src = _tiny_text_design()
    plan = _plan(src)
    small = copy.deepcopy(src)
    for el in small["elements"]:
        if el["id"] == "cta_label":
            el["font_size"] = 4
    good, _ = rules.penalty(src, src, plan, measurer)
    bad, parts = rules.penalty(src, small, plan, measurer)
    assert bad > good
    assert parts["R1_text_floor"] > 0


def test_content_against_the_edge_is_penalised(measurer):
    src = _tiny_text_design()
    plan = _plan(src)
    edge = copy.deepcopy(src)
    for el in edge["elements"]:
        if el["id"] == "headline":
            el["y"] = 0
    _, ok = rules.penalty(src, src, plan, measurer)
    _, bad = rules.penalty(src, edge, plan, measurer)
    assert bad["R4_edge_margin"] > ok["R4_edge_margin"]


def test_shrunk_cluster_with_dead_space_is_penalised(measurer):
    """The same content squeezed into the top-left quarter leaves a big empty
    rectangle and looks shrunk."""
    src = _tiny_text_design()
    plan = _plan(src)
    shrunk = copy.deepcopy(src)
    for el in shrunk["elements"]:
        if el["id"] == "bg":
            continue
        for k in ("x", "y", "width", "height"):
            el[k] = el[k] * 0.4
        if el["type"] == "text":
            el["font_size"] *= 0.4
        if el.get("corner_radius"):
            el["corner_radius"] *= 0.4
    good, gp = rules.penalty(src, src, plan, measurer)
    bad, bp = rules.penalty(src, shrunk, plan, measurer)
    assert bad > good
    assert bp["R2_dead_space"] > gp["R2_dead_space"]
    assert bp["R6_looks_shrunk"] > gp["R6_looks_shrunk"]


def test_penalty_never_raises_on_garbage(measurer):
    src = _tiny_text_design()
    plan = _plan(src)
    total, parts = rules.penalty(src, {"canvas": {"width": 10, "height": 10}, "elements": "nope"}, plan, measurer)
    assert total >= 0 and isinstance(parts, dict)


def test_empty_rect_share_of_an_empty_canvas_is_one():
    assert rules.largest_empty_share([], 300, 250) == pytest.approx(1.0)
    assert rules.largest_empty_share([(0, 0, 300, 250)], 300, 250) == pytest.approx(0.0)
    half = rules.largest_empty_share([(0, 0, 150, 250)], 300, 250)
    assert 0.4 < half <= 0.5


# ---------------------------------------------------------------- refine

def _cells(l1, l2, l3, targets):
    out = []
    for task in (l1, l2, l3):
        for t in targets:
            out.append((task["source"], {"width": t["width"], "height": t["height"]}))
    return out


def _winner(source, target, plan, measurer):
    from engine.pipeline.grading.score import pick_best
    cands = [safe_fallback(source, target, plan, measurer)] + generate_candidates(source, target, plan, measurer)
    best, _ = pick_best(source, target, plan, cands, measurer)
    return best


@pytest.fixture(scope="module")
def sample_cells():
    import json as _json
    from pathlib import Path
    tasks = [_json.loads(p.read_text()) for p in sorted(Path("input").glob("*.json"))]
    targets = _json.loads(Path("dev/targets.json").read_text())
    picked = ["instagram_portrait", "story", "leaderboard", "medium_rectangle", "wide_skyscraper", "billboard"]
    return [(t["source"], {"width": g["width"], "height": g["height"]}) for t in tasks for g in targets
            if g["name"] in picked]


def test_refine_output_passes_gates_and_lowers_penalty(sample_cells, measurer):
    from engine.pipeline.layout.strategies import rule_repair

    improved = 0
    for source, target in sample_cells:
        plan = _plan(source)
        cand = _winner(source, target, plan, measurer)
        before, _ = rules.penalty(source, cand.design, plan, measurer)
        out = rule_repair.refine(source, target, plan, measurer, cand)
        if out is None:
            continue
        improved += 1
        assert out.candidate_id == f"rule_repair+{cand.candidate_id}"
        assert check_gates(source, out.design, target, plan, measurer) == []
        after, _ = rules.penalty(source, out.design, plan, measurer)
        assert after < before
    assert improved >= len(sample_cells) // 3


def test_refine_keeps_reading_order_content_and_ids(sample_cells, measurer):
    from engine.pipeline.layout.strategies import rule_repair

    for source, target in sample_cells[::3]:
        plan = _plan(source)
        cand = _winner(source, target, plan, measurer)
        out = rule_repair.refine(source, target, plan, measurer, cand)
        if out is None:
            continue
        assert rule_repair.order_agreement(source, out.design, plan) >= rule_repair.order_agreement(
            source, cand.design, plan) - 1e-9
        frozen = {"x", "y", "width", "height", "font_size", "letter_spacing", "corner_radius", "stroke_width"}
        for a, b in zip(cand.design["elements"], out.design["elements"]):
            assert a["id"] == b["id"]
            if a["type"] == "text":  # text never shrinks
                assert b["font_size"] >= a["font_size"] - 0.01
            assert {k: v for k, v in a.items() if k not in frozen} == {k: v for k, v in b.items() if k not in frozen}


def test_refine_is_deterministic(sample_cells, measurer):
    from engine.pipeline.layout.strategies import rule_repair

    source, target = sample_cells[1]
    plan = _plan(source)
    cand = _winner(source, target, plan, measurer)
    a = rule_repair.refine(source, target, plan, measurer, cand)
    b = rule_repair.refine(source, target, plan, measurer, cand)
    assert (a is None) == (b is None)
    if a is not None:
        assert json.dumps(a.design, sort_keys=True) == json.dumps(b.design, sort_keys=True)


def test_refine_never_raises(measurer):
    from engine.pipeline.layout.strategies import rule_repair

    src = _tiny_text_design()
    plan = _plan(src)
    broken = Candidate("x", {"canvas": {"width": 300, "height": 250}, "elements": None}, "x", {})
    assert rule_repair.refine(src, {"width": 300, "height": 250}, plan, measurer, broken) is None
    assert rule_repair.refine(src, {"width": 300, "height": 250}, plan, measurer, None) is None


def test_refine_fixes_a_shrunk_layout(measurer):
    """The safe fallback of a square ad in a tall canvas is a small band with
    dead space above and below: the repair must grow it."""
    from engine.pipeline.layout.strategies import rule_repair

    src = _tiny_text_design()
    plan = _plan(src)
    target = {"width": 1080, "height": 1920}
    fb = safe_fallback(src, target, plan, measurer)
    shrunk = Candidate("fallback", copy.deepcopy(fb.design), "fallback", {}, [], True)
    # Squeeze the content further so there is clear room to grow.
    for el in shrunk.design["elements"]:
        if el["id"] == "bg":
            continue
        cx, cy = 540, 960
        el["x"], el["y"] = cx + (el["x"] - cx) * 0.6, cy + (el["y"] - cy) * 0.6
        el["width"], el["height"] = el["width"] * 0.6, el["height"] * 0.6
        if el["type"] == "text":
            el["font_size"] = el["font_size"] * 0.6
        if el.get("corner_radius"):
            el["corner_radius"] *= 0.6
    assert check_gates(src, shrunk.design, target, plan, measurer) == []
    out = rule_repair.refine(src, target, plan, measurer, shrunk)
    assert out is not None
    assert check_gates(src, out.design, target, plan, measurer) == []
    size = lambda d, i: next(e for e in d["elements"] if e["id"] == i)["font_size"]
    assert size(out.design, "headline") > size(shrunk.design, "headline")
