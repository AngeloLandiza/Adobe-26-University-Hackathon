"""Sonnet's learned proportions (engine/pipeline/layout/targets.py, CHAMELEON_TARGETS=1 / =winner):
the solver pulls block sizes toward the predicted ones, bisection caps text at them, anchored adds
candidates with every piece at them. With the switch off the engine's output is byte-identical
(golden hashes of the v5 engine taken before the targets existed). Fixtures are synthetic."""

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout import solver, targets
from engine.pipeline.layout import strategy_registry as reg
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.types import Candidate
from tests.conftest import assert_valid
from tests.designs import design, shape, text
from tests.test_solver import GOLDEN_V5, SAMPLES, TARGETS, V5


def _set(monkeypatch, value):
    for k, v in V5.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CHAMELEON_LLM", "heuristic")
    monkeypatch.delenv("CHAMELEON_SOLVER", raising=False)
    if value is None:
        monkeypatch.delenv("CHAMELEON_TARGETS", raising=False)
    else:
        monkeypatch.setenv("CHAMELEON_TARGETS", value)
    reg._modules.clear()


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_off_is_byte_identical(path, monkeypatch):
    from engine.resize import resize_with_report
    _set(monkeypatch, None)
    called = []
    monkeypatch.setattr(targets, "predict", lambda *a, **k: called.append(1) or {})
    src = json.loads(path.read_text())["source"]
    h = hashlib.sha1()
    for t in TARGETS:
        design_, _, best, _ = resize_with_report(src, {"width": t["width"], "height": t["height"]})
        h.update(best.candidate_id.encode())
        h.update(json.dumps(design_, sort_keys=True).encode())
    assert h.hexdigest() == GOLDEN_V5[path.stem]
    assert not called


def test_mode_switch(monkeypatch):
    for v, want in ((None, None), ("1", "1"), ("winner", "winner"), ("yes", None)):
        if v is None:
            monkeypatch.delenv("CHAMELEON_TARGETS", raising=False)
        else:
            monkeypatch.setenv("CHAMELEON_TARGETS", v)
        assert targets.mode() == want


# --- the model -------------------------------------------------------------------------

def test_model_file_is_small_and_confident():
    model = json.loads(targets.MODEL_PATH.read_text())
    assert targets.MODEL_PATH.stat().st_size < 64_000
    assert "font_headline" in model["targets"] and "hero" in model["targets"]
    for name, spec in model["targets"].items():
        assert spec["cv_r2"] >= 0.3, name
        assert len(spec["features"]) == len(spec["w"]) == len(spec["mean"]) == len(spec["std"]) == len(spec["lo"])


SRC = design(
    shape("bg", 0, 0, 1080, 1080, fill_color="rgba(250,240,220,1)", z_index=0),
    text("headline", "Chase the light", 140, 410, 800, 130, 90, z_index=2, text_align="center"),
    text("sub", "Botanical sparkling infusions", 190, 560, 700, 60, 40, z_index=2, text_align="center"),
    shape("pill", 390, 700, 300, 90, fill_color="rgba(200,60,40,1)", corner_radius=45, z_index=2),
    text("cta", "Shop now", 420, 715, 240, 60, 40, z_index=3, text_align="center"))


def test_predict_is_sane_and_never_raises():
    plan = heuristic_plan(SRC)
    for W, H in ((1080, 1080), (1584, 396), (160, 600), (728, 90), (1080, 1920)):
        p = targets.predict(SRC, plan, {"width": W, "height": H})
        assert 0.02 < p["font_headline"] < 0.6
        lo, hi = json.loads(targets.MODEL_PATH.read_text())["targets"]["font_headline"]["range"]
        assert lo <= p["font_headline"] <= hi
    assert targets.predict({"canvas": {}, "elements": []}, plan, {"width": 1, "height": 1}) == {}
    assert targets.predict(SRC, plan, {"width": 0, "height": 100}) == {}


def test_unsure_outside_training_range():
    plan = heuristic_plan(SRC)
    # an absurd target (1 x 50,000) is far outside every aspect the model saw: nothing predicted
    assert "font_headline" not in targets.predict(SRC, plan, {"width": 50000, "height": 1})


def test_measure_reads_roles():
    plan = heuristic_plan(SRC)
    m = targets.measure({e["id"]: e for e in SRC["elements"]}, plan, 1080, 1080)
    assert m["font_headline"] == pytest.approx(90 / 1080)
    assert m["content_w"] > 0.5 and m["content_h"] > 0.3


# --- the solver with targets ------------------------------------------------------------

TARGET = {"width": 1080, "height": 1350}
BIG = {"bg": {"width": 1080, "height": 1350}, "headline": {"x": 90, "y": 400, "width": 900, "height": 160, "font_size": 110},
       "sub": {"y": 640}, "pill": {"y": 800}, "cta": {"y": 815}}


def _cand(moves):
    els = [dict(copy.deepcopy(e), **moves.get(e["id"], {})) for e in SRC["elements"]]
    return Candidate("test", {"canvas": dict(TARGET), "elements": els}, "test", {})


def test_headline_pulled_toward_target(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    plan = heuristic_plan(SRC)
    cand = _cand(BIG)
    assert not check_gates(SRC, cand.design, TARGET, plan, measurer)
    want = 0.07  # of the short side: 76 px, the candidate has 110
    out = solver.solve_candidate(SRC, TARGET, plan, measurer, cand, opts=solver.Opts(targets={"font_headline": want}))
    assert out is not cand
    els = {e["id"]: e for e in out.design["elements"]}
    f = els["headline"]["font_size"]
    assert f < 105 and abs(f - want * 1080) < abs(110 - want * 1080)
    assert all(f >= e["font_size"] - 0.01 for e in els.values() if e["type"] == "text")  # still the largest text
    assert not check_gates(SRC, out.design, TARGET, plan, measurer)
    assert_valid(SRC, out.design, TARGET)


def test_floors_hold(measurer, monkeypatch):
    from engine.pipeline.layout.place import floor_px
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    plan = heuristic_plan(SRC)
    cand = _cand(BIG)
    out = solver.solve_candidate(SRC, TARGET, plan, measurer, cand,
                                 opts=solver.Opts(targets={"font_headline": 0.001, "font_subheadline": 0.001}))
    els = {e["id"]: e for e in out.design["elements"]}
    assert els["headline"]["font_size"] >= min(110, floor_px("headline", 1080, 1350)) - 0.01
    assert els["headline"]["font_size"] >= 110 * solver.T_S_MIN - 0.5   # never below T_S_MIN x
    assert not check_gates(SRC, out.design, TARGET, plan, measurer)


def test_no_targets_is_the_plain_solver(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    plan = heuristic_plan(SRC)
    cand = _cand(BIG)
    a = solver.solve_candidate(SRC, TARGET, plan, measurer, cand)
    b = solver.solve_candidate(SRC, TARGET, plan, measurer, cand, opts=solver.Opts(targets={}))
    assert json.dumps(a.design, sort_keys=True) == json.dumps(b.design, sort_keys=True)


# --- the engine with the switch on ------------------------------------------------------

@pytest.mark.parametrize("mode", ["1", "winner"])
def test_engine_with_targets_is_valid(mode, monkeypatch):
    from engine.resize import _get_measurer, resize_with_report
    _set(monkeypatch, mode)
    src = json.loads(SAMPLES[1].read_text())["source"]
    plan = heuristic_plan(src)
    for t in TARGETS[:8]:
        canvas = {"width": t["width"], "height": t["height"]}
        design_, plan_, best, _ = resize_with_report(src, canvas)
        assert_valid(src, design_, canvas)
        assert not check_gates(src, design_, canvas, plan_, _get_measurer())


def test_winner_mode_keeps_the_pick(monkeypatch):
    from engine.resize import resize_with_report
    src = json.loads(SAMPLES[1].read_text())["source"]
    picks = {}
    for mode in (None, "winner"):
        _set(monkeypatch, mode)
        picks[mode] = [resize_with_report(src, {"width": t["width"], "height": t["height"]})[2].candidate_id
                       for t in TARGETS[:6]]
    assert picks[None] == picks["winner"]


def test_anchored_target_candidates(measurer, monkeypatch):
    from engine.pipeline.layout.strategies import anchored
    src = json.loads(SAMPLES[1].read_text())["source"]
    plan = heuristic_plan(src)
    t = {"width": 1584, "height": 396}
    _set(monkeypatch, None)
    off = [c.candidate_id for c in anchored.candidates(src, t, plan, measurer)]
    _set(monkeypatch, "1")
    on = anchored.candidates(src, t, plan, measurer)
    assert [c.candidate_id for c in on][:len(off)] == off
    extra = [c for c in on if "mode=targets" in c.candidate_id]
    assert all(not check_gates(src, c.design, t, plan, measurer) for c in on)
    assert len(extra) <= anchored.TARGET_PICKS
    assert math.isfinite(len(extra))
