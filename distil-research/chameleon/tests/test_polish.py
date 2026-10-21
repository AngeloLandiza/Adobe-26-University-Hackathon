"""The polish pass (CHAMELEON_POLISH=1, engine/pipeline/layout/polish.py) and
the near-fit guard (CHAMELEON_NEAR_FIT_GUARD=1, grading/score.py).

With both switches unset (or "0") the engine's output is byte-identical to
before they existed (golden hash over the 3 samples x every target with the
v3L switches on). With polish on, every polished candidate still passes the
hard gates and keeps its headline size; the official level-2 sample's
squashed badge and speck logo are repaired. The guard keeps the re-layout on
the level-1 sample (1080x1080 -> 1080x1350), where the uniform fit leaves the
headline at 2/3 of its size and an empty band.
"""

import hashlib
import json
from pathlib import Path

import pytest

from engine.pipeline.grading import score
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout import polish
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.priority import heuristic_plan
from engine.resize import resize

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())
V3L = {"CHAMELEON_STRATEGIES": "bisection", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
       "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1", "CHAMELEON_LLM": "heuristic"}
# sha1 over resize() for the 3 samples x (13 targets + the official target), v3L switches on,
# from the engine as it was before CHAMELEON_POLISH / CHAMELEON_NEAR_FIT_GUARD existed.
GOLDEN_V3L = "5ed09a2783cebd68fad792481efb5fe21b6bb174"


def _env(monkeypatch, extra: dict):
    for k, v in {**V3L, **extra}.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


def _sample(stem: str) -> dict:
    return json.loads(Path(f"input/chameleon_sample_{stem}.json").read_text())


@pytest.mark.parametrize("value", [None, "0"])
def test_switches_off_are_byte_identical(value, monkeypatch):
    _env(monkeypatch, {polish.ENV: value, score.GUARD_ENV: value})
    h = hashlib.sha1()
    for path in SAMPLES:
        d = json.loads(path.read_text())
        for t in TARGETS + [d["target_canvas"]]:
            out = resize(d["source"], {"width": t["width"], "height": t["height"]})
            h.update(json.dumps(out, sort_keys=True).encode())
    assert h.hexdigest() == GOLDEN_V3L


def _head(design: dict, plan) -> float:
    ids = {i for u in plan.units if u.role == "headline" for i in u.element_ids}
    return max((e["font_size"] for e in design["elements"] if e["id"] in ids and e["type"] == "text"), default=0.0)


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_polished_candidates_pass_gates_and_keep_the_headline(path, measurer, monkeypatch):
    _env(monkeypatch, {polish.ENV: None})
    src = json.loads(path.read_text())["source"]
    plan = heuristic_plan(src)
    changed = 0
    for t in TARGETS[::2]:
        target = {"width": t["width"], "height": t["height"]}
        for cand in generate_candidates(src, target, plan, measurer):
            if polish._skip(cand) or check_gates(src, cand.design, target, plan, measurer):
                continue
            new = polish.polish(src, target, plan, measurer, cand)
            assert new.candidate_id == cand.candidate_id
            assert not check_gates(src, new.design, target, plan, measurer)
            assert _head(new.design, plan) == _head(cand.design, plan)
            changed += new is not cand
    assert changed > 0


def test_fallback_and_bisection_v2_are_left_alone(measurer, monkeypatch):
    _env(monkeypatch, {polish.ENV: None})
    d = _sample("level2_core")
    target, plan = d["target_canvas"], heuristic_plan(d["source"])
    fb = safe_fallback(d["source"], target, plan, measurer)
    cands = [fb] + [c for c in generate_candidates(d["source"], target, plan, measurer) if c.template == "bisection"]
    assert len(cands) > 1
    assert all(a is b for a, b in zip(cands, polish.polish_all(d["source"], target, plan, measurer, cands)))


def test_level2_row_badge_is_readable(measurer, monkeypatch):
    """1080x1080 -> 1600x400: the row template parks NEW FLAVOR at its 9 px floor; polish never
    shrinks it and, where there is room, lifts it to PILL_MIN_LONG of the long side (17.6 px)."""
    _env(monkeypatch, {polish.ENV: None})
    d = _sample("level2_core")
    src, target = d["source"], d["target_canvas"]
    plan = heuristic_plan(src)
    size = lambda c: next(e for e in c.design["elements"] if e["id"] == "badge_label")["font_size"]
    lifted = 0
    for cand in generate_candidates(src, target, plan, measurer):
        if cand.template != "row" or check_gates(src, cand.design, target, plan, measurer):
            continue
        new = polish.polish(src, target, plan, measurer, cand)
        assert size(new) >= size(cand)
        lifted += size(new) >= polish.PILL_MIN_LONG * 1600 - 0.05 > size(cand)
    assert lifted > 0


def test_logo_floor_is_reached_where_there_is_room(measurer, monkeypatch):
    _env(monkeypatch, {polish.ENV: None})
    from engine.pipeline.geometry import rotated_aabb
    from engine.pipeline.layout.strategies.bisection import logo_min_height
    grew = 0
    for path in SAMPLES:
        src = json.loads(path.read_text())["source"]
        plan = heuristic_plan(src)
        logo = next((u for u in plan.units if u.role == "logo"), None)
        if logo is None:
            continue
        for t in TARGETS:
            target = {"width": t["width"], "height": t["height"]}
            for cand in generate_candidates(src, target, plan, measurer):
                if polish._skip(cand) or check_gates(src, cand.design, target, plan, measurer):
                    continue
                new = polish.polish(src, target, plan, measurer, cand)
                h0 = rotated_aabb(next(e for e in cand.design["elements"] if e["id"] == logo.element_ids[0]))
                h1 = rotated_aabb(next(e for e in new.design["elements"] if e["id"] == logo.element_ids[0]))
                assert h1[3] >= h0[3] - 1e-6 or h1[3] >= polish.logo_floor(
                    t["width"], t["height"], h1[2], h1[3], logo_min_height(t["width"], t["height"])) - 0.5
                grew += h1[3] > h0[3] + 0.5
    assert grew > 0


def test_guard_rule():
    base = {"head_ratio": 0.67, "cta_ratio": 0.67, "empty_src": 0.38, "empty_fb": 0.40, "empty_best": 0.25}
    assert score.guard_blocks(base)
    assert not score.guard_blocks({**base, "head_ratio": 0.9})          # headline barely smaller
    assert not score.guard_blocks({**base, "cta_ratio": 0.95})          # CTA keeps its size
    assert score.guard_blocks({**base, "cta_ratio": None})              # no CTA: the headline decides
    assert not score.guard_blocks({**base, "empty_best": 0.35})         # the re-layout leaves as much empty
    assert not score.guard_blocks({**base, "head_ratio": None})


def test_guard_keeps_the_relayout_on_level1(monkeypatch):
    d = _sample("level1_warmup")
    _env(monkeypatch, {polish.ENV: None, score.GUARD_ENV: None})
    from engine.resize import resize_with_report
    assert resize_with_report(d["source"], d["target_canvas"])[2].is_fallback
    monkeypatch.setenv(score.GUARD_ENV, "1")
    assert not resize_with_report(d["source"], d["target_canvas"])[2].is_fallback
    monkeypatch.delenv("CHAMELEON_NEAR_FIT")  # the guard means nothing without near-fit
    plain = resize_with_report(d["source"], d["target_canvas"])[2]
    monkeypatch.delenv(score.GUARD_ENV)
    assert resize_with_report(d["source"], d["target_canvas"])[2].candidate_id == plain.candidate_id
