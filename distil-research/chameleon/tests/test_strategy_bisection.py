"""Strategy S2 `bisection`: priority-weighted recursive splitting
(engine/pipeline/layout/strategies/bisection.py).

Every returned candidate must pass the hard gates, the function must never
raise, and the layouts must keep the hierarchy (headline biggest text, CTA
readable, logo visible) on every shape, extreme strips included.
"""

import json
from pathlib import Path

import pytest

from engine.pipeline.geometry import overlap_area, rotated_aabb, union_box
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.layout.place import floor_px
from engine.pipeline.layout.strategies import bisection
from engine.pipeline.layout.strategies.bisection import Span, candidates, split_extent
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.types import Candidate, PriorityPlan, Unit
from tests.conftest import assert_valid

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())
BY_NAME = {t["name"]: {"width": t["width"], "height": t["height"]} for t in TARGETS}


def _src(path: Path) -> dict:
    return json.loads(path.read_text())["source"]


def _els(c: Candidate) -> dict:
    return {e["id"]: e for e in c.design["elements"]}


# --- the split rule -----------------------------------------------------------

def test_split_follows_value_when_unconstrained():
    a, b = Span(lo=0, hi=1000, value=3), Span(lo=0, hi=1000, value=1)
    assert split_extent(400, a, b) == pytest.approx(300)


def test_split_respects_minimums():
    a, b = Span(lo=0, hi=1000, value=10), Span(lo=150, hi=1000, value=1)
    assert split_extent(400, a, b) == pytest.approx(250)  # b keeps its 150


def test_split_gives_slack_of_a_saturated_side_to_the_other():
    a, b = Span(lo=0, hi=100, value=10), Span(lo=0, hi=1000, value=1)
    assert split_extent(400, a, b) == pytest.approx(100)  # a can't use more than 100


def test_split_shares_surplus_when_both_saturate():
    a, b = Span(lo=0, hi=100, value=1), Span(lo=0, hi=300, value=1)
    assert split_extent(800, a, b) == pytest.approx(200)  # surplus shared in proportion to max use


def test_split_scales_minimums_when_infeasible():
    a, b = Span(lo=300, hi=1000, value=1), Span(lo=100, hi=1000, value=1)
    assert split_extent(200, a, b) == pytest.approx(150)


# --- the contract ---------------------------------------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_every_candidate_passes_the_gates(path, target, measurer):
    src = _src(path)
    plan = heuristic_plan(src)
    canvas = {"width": target["width"], "height": target["height"]}
    out = candidates(src, canvas, plan, measurer)
    assert isinstance(out, list)
    for c in out:
        assert c.candidate_id.startswith("bisection/")
        assert not c.is_fallback
        assert check_gates(src, c.design, canvas, plan, measurer) == [], c.candidate_id
        assert_valid(src, c.design, canvas)
    assert len({c.candidate_id for c in out}) == len(out)


@pytest.mark.parametrize("name", ["leaderboard", "mobile_banner", "wide_skyscraper", "large_leaderboard",
                                  "story", "linkedin_banner"])
@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_extreme_shapes_get_candidates(path, name, measurer):
    src = _src(path)
    assert candidates(src, BY_NAME[name], heuristic_plan(src), measurer)


def test_deterministic(l3, measurer):
    run = lambda: [(c.candidate_id, json.dumps(c.design, sort_keys=True))
                   for c in candidates(l3["source"], BY_NAME["billboard"], heuristic_plan(l3["source"]), measurer)]
    assert run() == run()


def test_same_aspect_is_left_to_the_reflow(l1, measurer):
    assert candidates(l1["source"], BY_NAME["instagram_post"], heuristic_plan(l1["source"]), measurer) == []


@pytest.mark.parametrize("bad", [
    {},
    {"canvas": {"width": 1080, "height": 1080}, "elements": []},
    {"canvas": {"width": 0, "height": 1080}, "elements": [{"id": "x"}]},
    {"canvas": {"width": 1080, "height": 1080}, "elements": [{"id": "t", "type": "text"}]},
])
def test_never_raises(bad, measurer):
    plan = PriorityPlan("heuristic", [Unit("t", ["t"], "single", role="headline", rank=1)])
    assert candidates(bad, {"width": 300, "height": 250}, plan, measurer) == []
    assert candidates(bad, {"width": 300, "height": 250}, None, None) == []


# --- design quality ---------------------------------------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("name", ["leaderboard", "billboard", "wide_skyscraper", "story", "youtube_thumbnail"])
def test_headline_is_the_biggest_text(path, name, measurer):
    src = _src(path)
    plan = heuristic_plan(src)
    head = next(u for u in plan.units if u.role == "headline")
    floor = floor_px("headline", BY_NAME[name]["width"], BY_NAME[name]["height"])
    for c in candidates(src, BY_NAME[name], plan, measurer):
        els = _els(c)
        size = els[head.element_ids[0]]["font_size"]
        if size < floor:  # a headline that can't be readable must not drag the CTA down with it
            continue
        others = [e["font_size"] for e in els.values() if e["type"] == "text" and e["id"] not in head.element_ids]
        assert all(size >= o - 1e-6 for o in others), c.candidate_id


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_headline_and_cta_reach_their_floor_where_the_templates_do(path, target, measurer):
    """Wherever some engine template gets the headline (or CTA) to its floor,
    some bisection candidate does too."""
    src = _src(path)
    plan = heuristic_plan(src)
    out = candidates(src, target, plan, measurer)
    engine = [c for c in generate_candidates(src, target, plan, measurer) if c.candidate_id != "reflow"]

    def best(cands, unit):
        return max((min(_els(c)[i]["font_size"] for i in unit.element_ids if _els(c)[i]["type"] == "text")
                    for c in cands), default=0.0)

    for role in ("headline", "cta"):
        unit = next((u for u in plan.units if u.role == role), None)
        if unit is None or not out:
            continue
        floor = floor_px(role, target["width"], target["height"])
        if best(engine, unit) >= floor - 1e-6:
            assert best(out, unit) >= floor - 1e-6, role


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_logo_is_visible(path, target, measurer):
    src = _src(path)
    plan = heuristic_plan(src)
    logo = next(u for u in plan.units if u.role == "logo")
    need = bisection.logo_min_height(target["width"], target["height"])
    for c in candidates(src, target, plan, measurer):
        el = _els(c)[logo.element_ids[0]]
        assert el["height"] >= need - 0.05 or logo.unit_id in c.degraded_unit_ids, c.candidate_id


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("name", ["leaderboard", "half_page", "facebook_cover", "instagram_portrait"])
def test_content_units_do_not_collide(path, name, measurer):
    src = _src(path)
    plan = heuristic_plan(src)
    content = plan.ranked_units()
    for c in candidates(src, BY_NAME[name], plan, measurer):
        els = _els(c)
        boxes = {u.unit_id: union_box([rotated_aabb(els[i]) for i in u.element_ids]) for u in content}
        for i, a in enumerate(content):
            for b in content[i + 1:]:
                assert overlap_area(boxes[a.unit_id], boxes[b.unit_id]) < 1.0, (c.candidate_id, a.unit_id, b.unit_id)


CORPUS_TARGETS = ["mobile_banner", "wide_skyscraper", "billboard", "instagram_portrait"]


@pytest.mark.parametrize("name", CORPUS_TARGETS)
def test_corpus_candidates_pass_the_gates(name, measurer):
    """A spread of corpus ads (both brands) on extreme and ordinary shapes."""
    from dev.distill.corpus import load_corpus

    designs = [d for d in load_corpus() if d.get("origin") != "sample"][::15]
    target = BY_NAME[name]
    for d in designs:
        src = d["source"]
        plan = heuristic_plan(src)
        for c in candidates(src, target, plan, measurer):
            assert check_gates(src, c.design, target, plan, measurer) == [], (d["task_id"], c.candidate_id)
