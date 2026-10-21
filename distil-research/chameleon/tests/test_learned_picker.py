"""CHAMELEON_PICKER=learned (engine/pipeline/grading/learned_picker.py): a
pairwise model picks among the grader's shortlist; unset = today's grader,
byte-identical (golden hashes of the v4 engine, 3 samples x every target)."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from engine.pipeline.grading import learned_picker as lp
from engine.pipeline.grading import score
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout import strategy_registry as reg
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Candidate, GradeReport

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())
V4 = {"CHAMELEON_STRATEGIES": "bisection,anchored", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
      "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1"}
# sha1 over (winner id, design) of resize_with_report on every target, the v4 stack (as in test_balance.py).
GOLDEN_V4 = {
    "chameleon_sample_level1_warmup": "000eae1dbcfd82345f9ba4773a2b1bce45ecaecd",
    "chameleon_sample_level2_core": "73f48df675757493066be73e6b330c7c57a4c60d",
    "chameleon_sample_level3_advanced": "5debf254cc3007adba52df1fed9f3047c01769be",
}


def _v4(monkeypatch, picker=None):
    for k, v in V4.items():
        monkeypatch.setenv(k, v)
    for k in ("CHAMELEON_BALANCE", "CHAMELEON_TEXT_SLACK", "CHAMELEON_RELPRESERVE"):
        monkeypatch.delenv(k, raising=False)
    if picker is None:
        monkeypatch.delenv("CHAMELEON_PICKER", raising=False)
    else:
        monkeypatch.setenv("CHAMELEON_PICKER", picker)
    reg._modules.clear()


def _engine_digest(src: dict) -> str:
    from engine.resize import resize_with_report
    h = hashlib.sha1()
    for t in TARGETS:
        design_, _, best, _ = resize_with_report(src, {"width": t["width"], "height": t["height"]})
        h.update(best.candidate_id.encode())
        h.update(json.dumps(design_, sort_keys=True).encode())
    return h.hexdigest()


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_unset_is_byte_identical(path, monkeypatch):
    _v4(monkeypatch)
    monkeypatch.setattr(lp, "choose", lambda *a, **k: (_ for _ in ()).throw(AssertionError("picker ran")))
    assert _engine_digest(json.loads(path.read_text())["source"]) == GOLDEN_V4[path.stem]


# ---------------------------------------------------------------- one real cell

@pytest.fixture(scope="module")
def cell():
    task = json.loads(Path("input/chameleon_sample_level2_core.json").read_text())
    src, tgt = task["source"], task["target_canvas"]
    plan, m = heuristic_plan(src), TextMeasurer()
    cands = [safe_fallback(src, tgt, plan, m)] + generate_candidates(src, tgt, plan, m)
    return src, tgt, plan, m, cands


def _grader(cell, monkeypatch):
    src, tgt, plan, m, cands = cell
    monkeypatch.delenv("CHAMELEON_PICKER", raising=False)
    return score.pick_best(src, tgt, plan, cands, m)


def test_weights_match_the_engines_features(cell, monkeypatch):
    src, tgt, plan, m, cands = cell
    monkeypatch.setenv("CHAMELEON_PICKER", "learned")
    score.pick_best(src, tgt, plan, cands, m)
    assert lp.LAST["names"] == lp.load_model()["names"]
    assert len(lp.LAST["shortlist"]) >= 2


def test_learned_takes_the_models_argmax(cell, monkeypatch):
    src, tgt, plan, m, cands = cell
    grader_pick, reports = _grader(cell, monkeypatch)
    ids = [cands[i].candidate_id for i in lp.shortlist(cands, reports, grader_pick)]
    monkeypatch.setattr(lp, "model_scores", lambda model, rows: [float(j == len(rows) - 1) for j in range(len(rows))])
    monkeypatch.setenv("CHAMELEON_PICKER", "learned")
    best, _ = score.pick_best(src, tgt, plan, cands, m)
    assert best.candidate_id == ids[-1] != grader_pick.candidate_id
    assert check_gates(src, best.design, tgt, plan, m) == []


@pytest.mark.parametrize("breakage", ["load", "names"])
def test_any_error_keeps_the_graders_pick(cell, monkeypatch, breakage):
    src, tgt, plan, m, cands = cell
    grader_pick, _ = _grader(cell, monkeypatch)
    if breakage == "load":
        monkeypatch.setattr(lp, "load_model", lambda *a: (_ for _ in ()).throw(OSError("no weights")))
    else:
        model = dict(lp.load_model(), names=["something", "else"])
        monkeypatch.setattr(lp, "load_model", lambda *a: model)
    monkeypatch.setenv("CHAMELEON_PICKER", "learned")
    best, _ = score.pick_best(src, tgt, plan, cands, m)
    assert best.candidate_id == grader_pick.candidate_id


def test_learned_never_returns_a_gate_failure(cell, monkeypatch):
    src, tgt, plan, m, cands = cell
    broken = json.loads(json.dumps(cands[-1].design))
    broken["elements"] = broken["elements"][1:]
    bad = replace(cands[-1], candidate_id="broken", design=broken)
    monkeypatch.setattr(lp, "model_scores", lambda model, rows: [1.0] * len(rows))
    monkeypatch.setenv("CHAMELEON_PICKER", "learned")
    best, _ = score.pick_best(src, tgt, plan, cands + [bad], m)
    assert best.candidate_id != "broken"
    assert "broken" not in lp.LAST["shortlist"]


def test_near_source_target_keeps_the_graders_pick(monkeypatch):
    task = json.loads(Path("input/chameleon_sample_level2_core.json").read_text())
    src = task["source"]
    tgt = {"width": src["canvas"]["width"], "height": src["canvas"]["height"] + 5}  # within 2% of the source aspect
    plan, m = heuristic_plan(src), TextMeasurer()
    cands = [safe_fallback(src, tgt, plan, m)] + generate_candidates(src, tgt, plan, m)
    grader_pick, _ = _grader((src, tgt, plan, m, cands), monkeypatch)
    monkeypatch.setattr(lp, "model_scores", lambda model, rows: list(range(len(rows))))
    monkeypatch.setenv("CHAMELEON_PICKER", "learned")
    best, _ = score.pick_best(src, tgt, plan, cands, m)
    assert best.candidate_id == grader_pick.candidate_id


# ---------------------------------------------------------------- the shortlist

def _c(cid, template, fallback=False):
    return Candidate(cid, {"canvas": {}, "elements": []}, template, {}, is_fallback=fallback)


def test_shortlist_pick_first_two_per_family_and_the_fallback():
    cands = [_c("fallback", "fallback", True), _c("b/1", "bisection"), _c("b/2", "bisection"),
             _c("b/3", "bisection"), _c("a/1", "anchored"), _c("r/1", "row"), _c("x", "row")]
    totals = {"fallback": 0.1, "b/1": 0.9, "b/2": 0.8, "b/3": 0.85, "a/1": 0.5, "r/1": 0.7, "x": 0.99}
    reports = [GradeReport(c.candidate_id, c.candidate_id != "x", [], {}, c.candidate_id != "fallback",
                           totals[c.candidate_id]) for c in cands]
    got = [cands[i].candidate_id for i in lp.shortlist(cands, reports, cands[4])]
    # a/1 is the pick; then b/1, b/3 (two bisections, b/2 skipped); r/1 is replaced by the fallback; x failed gates
    assert got == ["a/1", "b/1", "b/3", "fallback"]


def test_feature_vector_order():
    metrics = {"b": 0.5, "a": 0.25}
    names = lp.feature_names(metrics)
    x = lp.feature_vector(metrics, 0.75, "columns", 0.9, 0.5)
    assert names[:3] == ["a", "b", "relation"] and len(names) == len(x)
    assert x[:3] == [0.25, 0.5, 0.75]
    assert x[names.index("fam_columns")] == 1.0 and x[names.index("famXaspect_columns")] == 0.5
    assert lp.feature_vector(metrics, 1, "text_first", 0, 1)[names.index("fam_other")] == 1.0
