"""The three experiments from docs/prior-work.md, each behind its own switch:
CHAMELEON_HIER_LOCK (bisection), CHAMELEON_NEAR_FIT and CHAMELEON_RELPRESERVE
(pick_best). With every switch off the output must be byte-identical to
before; with one on, the change does what it says."""

import hashlib
import json
from pathlib import Path

import pytest

from engine.pipeline.geometry import rotated_aabb, union_box
from engine.pipeline.grading import relations, score
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout.place import floor_px
from engine.pipeline.layout.strategies import bisection
from engine.pipeline.layout.strategies.bisection import candidates
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.types import PriorityPlan, Unit
from tests.conftest import assert_valid
from tests.test_bisection_v2 import GOLDEN as GOLDEN_V1

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())
SWITCHES = ("CHAMELEON_HIER_LOCK", "CHAMELEON_NEAR_FIT", "CHAMELEON_RELPRESERVE", "CHAMELEON_RELPRESERVE_W")

# sha1 over (candidate_id, design) of every bisection candidate (v2 on) for each sample on every target,
# from the strategy before these switches existed.
GOLDEN_V2 = {
    "chameleon_sample_level1_warmup": "1fdb2a6f77612da82f1ee748da3be31722a0452c",
    "chameleon_sample_level2_core": "5d4dda404fb14e75e86697b8353f2159b32996c5",
    "chameleon_sample_level3_advanced": "a9ae71a596e8d7d9044d8d6d693ee60f2898b6b7",
}
# sha1 over (winner id, design) of resize_with_report on every target, CHAMELEON_STRATEGIES=bisection and
# CHAMELEON_BISECT_V2=1, from the engine before these switches existed.
GOLDEN_ENGINE_B2 = {
    "chameleon_sample_level1_warmup": "e809dd630630a0bc31208dbcb49c87f69db03e64",
    "chameleon_sample_level2_core": "86104d9304b1b743ba4d7579426014124e17b365",
    "chameleon_sample_level3_advanced": "19c331ab61ddc86e59f2b39d7b35fc096d4bc54f",
}


def _src(path: Path) -> dict:
    return json.loads(path.read_text())["source"]


def _clear(monkeypatch):
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)


def _bisection_digest(src: dict, measurer) -> str:
    plan = heuristic_plan(src)
    h = hashlib.sha1()
    for t in TARGETS:
        for c in candidates(src, {"width": t["width"], "height": t["height"]}, plan, measurer):
            h.update(c.candidate_id.encode())
            h.update(json.dumps(c.design, sort_keys=True).encode())
    return h.hexdigest()


# --- switches off: byte-identical ------------------------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("value", [None, "0"])
def test_hier_lock_off_is_byte_identical(path, value, measurer, monkeypatch):
    _clear(monkeypatch)
    if value is not None:
        monkeypatch.setenv("CHAMELEON_HIER_LOCK", value)
    monkeypatch.delenv("CHAMELEON_BISECT_V2", raising=False)
    assert _bisection_digest(_src(path), measurer) == GOLDEN_V1[path.stem]
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "1")
    assert _bisection_digest(_src(path), measurer) == GOLDEN_V2[path.stem]


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_picker_switches_off_are_byte_identical(path, monkeypatch):
    from engine.resize import resize_with_report
    _clear(monkeypatch)
    if "level1" in path.stem:  # any value but "1" is off
        monkeypatch.setenv("CHAMELEON_NEAR_FIT", "0")
        monkeypatch.setenv("CHAMELEON_RELPRESERVE", "0")
        monkeypatch.setenv("CHAMELEON_RELPRESERVE_W", "0.9")
    monkeypatch.setenv("CHAMELEON_STRATEGIES", "bisection")
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "1")
    src = _src(path)
    h = hashlib.sha1()
    for t in TARGETS:
        design, _, best, _ = resize_with_report(src, {"width": t["width"], "height": t["height"]})
        h.update(best.candidate_id.encode())
        h.update(json.dumps(design, sort_keys=True).encode())
    assert h.hexdigest() == GOLDEN_ENGINE_B2[path.stem]


# --- hierarchy lock ------------------------------------------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
@pytest.mark.parametrize("v2", ["0", "1"])
def test_hier_lock_keeps_the_hierarchy(path, target, v2, measurer, monkeypatch):
    """Gates pass; the headline is the biggest text; every other text is
    within its role cap of the headline or at (or under) its floor."""
    _clear(monkeypatch)
    monkeypatch.setenv("CHAMELEON_HIER_LOCK", "1")
    monkeypatch.setenv("CHAMELEON_BISECT_V2", v2)
    src = _src(path)
    plan = heuristic_plan(src)
    canvas = {"width": target["width"], "height": target["height"]}
    head = next(u for u in plan.units if u.role == "headline")
    W, H = canvas["width"], canvas["height"]
    for c in candidates(src, canvas, plan, measurer):
        assert check_gates(src, c.design, canvas, plan, measurer) == [], c.candidate_id
        assert_valid(src, c.design, canvas)
        if head.unit_id in c.degraded_unit_ids:
            continue
        els = {e["id"]: e for e in c.design["elements"]}
        hs = max(els[i]["font_size"] for i in head.element_ids if els[i]["type"] == "text")
        for u in plan.ranked_units():
            if u is head or u.unit_id in c.degraded_unit_ids:
                continue
            sizes = [els[i]["font_size"] for i in u.element_ids if els[i]["type"] == "text"]
            if not sizes:
                continue
            cap = bisection.ROLE_CAPS.get(u.role, bisection.LOCK_HIERARCHY_CAP)
            assert max(sizes) <= hs + 1e-6, (c.candidate_id, u.role)
            assert max(sizes) <= max(cap * hs, floor_px(u.role, W, H)) + 0.51, (c.candidate_id, u.role, sizes, hs)


def test_hier_lock_caps_the_cta_harder_than_v1(l1, measurer, monkeypatch):
    """On a strip the CTA label keeps its source ratio to the headline (v1 lets it reach 1.3x that)."""
    _clear(monkeypatch)
    src = l1["source"]
    plan = heuristic_plan(src)
    t = next(t for t in TARGETS if t["name"] == "leaderboard")
    cta = next(u for u in plan.units if u.role == "cta")
    head = next(u for u in plan.units if u.role == "headline")
    src_els = {e["id"]: e for e in src["elements"]}
    ratio0 = max(src_els[i]["font_size"] for i in cta.element_ids if src_els[i]["type"] == "text") / \
        max(src_els[i]["font_size"] for i in head.element_ids if src_els[i]["type"] == "text")

    def ratios():
        out = []
        for c in candidates(src, t, plan, measurer):
            els = {e["id"]: e for e in c.design["elements"]}
            f = max(els[i]["font_size"] for i in cta.element_ids if els[i]["type"] == "text")
            out.append(f / max(els[i]["font_size"] for i in head.element_ids if els[i]["type"] == "text"))
        return out

    v1 = ratios()
    monkeypatch.setenv("CHAMELEON_HIER_LOCK", "1")
    lock = ratios()
    assert lock and all(r <= min(ratio0, 0.6) + 0.02 or r <= 11 / 14 for r in lock)
    assert max(lock) < max(v1)


def test_lift_is_floor_over_cap(l2, measurer, monkeypatch):
    """With a CTA and a badge present, the headline's raised floor on a 728x90
    strip is the CTA floor / 0.6 (the most demanding role); off, no lift."""
    _clear(monkeypatch)
    src = l2["source"]
    plan = heuristic_plan(src)
    t = {"width": 728, "height": 90}
    assert bisection._Model(src, t, plan, measurer).lift is None
    monkeypatch.setenv("CHAMELEON_HIER_LOCK", "1")
    model = bisection._Model(src, t, plan, measurer)
    assert model.lift == pytest.approx(max(11 / 0.6, 9 / 0.5))
    head = next(u for u in plan.units if u.role == "headline")
    assert model.floor(head) == 14
    model.lifted = True
    assert model.floor(head) == pytest.approx(model.lift)
    assert model.base_floor(head) == 14


# --- relation preservation --------------------------------------------------------

def _plan(*units) -> PriorityPlan:
    return PriorityPlan("heuristic", [Unit(uid, [uid], "single", role, rank) for uid, role, rank in units])


def _text(eid, x, y, w, h, size):
    return {"id": eid, "type": "text", "content": "Hello", "font_family": "Inter", "font_size": size,
            "color": "#000", "x": x, "y": y, "width": w, "height": h}


def _design(w, h, els):
    return {"canvas": {"width": w, "height": h}, "elements": els}


def test_relation_score_counts_kept_relations():
    plan = _plan(("h", "headline", 1), ("c", "cta", 2), ("l", "legal", 3))
    src = _design(400, 400, [_text("h", 20, 20, 200, 60, 40), _text("c", 20, 100, 120, 30, 20),
                             _text("l", 20, 300, 200, 20, 10)])
    assert relations.relation_score(src, src, plan) == 1.0
    # a strip: the stack becomes a row, left to right in the same order, sizes in the same order
    row = _design(800, 100, [_text("h", 10, 20, 200, 60, 30), _text("c", 300, 20, 120, 30, 16),
                             _text("l", 500, 20, 200, 30, 8)])
    parts = relations.relation_parts(src, row, plan)
    assert parts["size"][0] == parts["size"][1] and parts["order"][0] == parts["order"][1]
    # reversed reading order and the CTA bigger than the headline: both families drop
    bad = _design(400, 400, [_text("h", 20, 300, 200, 60, 20), _text("c", 20, 20, 120, 30, 40),
                             _text("l", 20, 200, 200, 20, 10)])
    p = relations.relation_parts(src, bad, plan)
    assert p["size"][0] < p["size"][1] and p["order"][0] < p["order"][1]
    assert relations.relation_score(src, bad, plan) < relations.relation_score(src, row, plan)
    # degraded (skipped) units are left out
    assert relations.relation_score(src, bad, plan, skip=("h", "c")) == 1.0


def test_relation_weight_env(monkeypatch):
    monkeypatch.delenv("CHAMELEON_RELPRESERVE_W", raising=False)
    assert relations.weight() == relations.DEFAULT_WEIGHT
    monkeypatch.setenv("CHAMELEON_RELPRESERVE_W", "0.3")
    assert relations.weight() == 0.3
    monkeypatch.setenv("CHAMELEON_RELPRESERVE_W", "junk")
    assert relations.weight() == relations.DEFAULT_WEIGHT


def test_relpreserve_penalises_the_totals(l3, monkeypatch):
    from engine.resize import resize_with_report
    _clear(monkeypatch)
    canvas = {"width": 728, "height": 90}
    _, _, best0, reports0 = resize_with_report(l3["source"], canvas)
    monkeypatch.setenv("CHAMELEON_RELPRESERVE", "1")
    monkeypatch.setenv("CHAMELEON_RELPRESERVE_W", "0.2")
    _, _, best, reports = resize_with_report(l3["source"], canvas)
    off = {r.candidate_id: r for r in reports0}
    rendered = [r for r in reports if r.rendered]
    assert rendered and all("relation" in r.metrics for r in reports if r.passed_gates)
    for r in rendered:
        if r.candidate_id in off and off[r.candidate_id].rendered:
            assert r.total == pytest.approx(off[r.candidate_id].total - 0.2 * (1 - r.metrics["relation"]))
    assert next(r for r in reports if r.candidate_id == best.candidate_id).total == max(r.total for r in rendered)


# --- near-aspect source fit -------------------------------------------------------

def test_near_fit_ratio(monkeypatch):
    src = {"canvas": {"width": 1080, "height": 1080}}
    monkeypatch.delenv("CHAMELEON_NEAR_FIT", raising=False)
    assert not score.near_fit(src, {"width": 1080, "height": 1350})
    monkeypatch.setenv("CHAMELEON_NEAR_FIT", "1")
    assert score.near_fit(src, {"width": 1080, "height": 1350})    # 1.25
    assert score.near_fit(src, {"width": 300, "height": 250})      # 1.2
    assert not score.near_fit(src, {"width": 1280, "height": 720})  # 1.78
    assert not score.near_fit(src, {"width": 728, "height": 90})


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_near_fit_keeps_the_original_unless_clearly_beaten(path, monkeypatch):
    from engine.resize import resize_with_report
    _clear(monkeypatch)
    monkeypatch.setenv("CHAMELEON_STRATEGIES", "bisection")
    monkeypatch.setenv("CHAMELEON_NEAR_FIT", "1")
    src = _src(path)
    for t in TARGETS:
        canvas = {"width": t["width"], "height": t["height"]}
        _, _, best, reports = resize_with_report(src, canvas)
        if not score.near_fit(src, canvas):
            continue
        fb = next((r for r in reports if r.candidate_id == "fallback"), None)
        if fb is None or not fb.passed_gates:
            continue
        assert fb.rendered
        others = max(r.total for r in reports if r.rendered and r.candidate_id != "fallback")
        assert (best.is_fallback) == (others - fb.total <= score.NEAR_FIT_MARGIN), t["name"]
