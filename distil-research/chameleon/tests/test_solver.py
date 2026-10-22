"""Geometry solver pass (engine/pipeline/layout/solver.py, CHAMELEON_SOLVER=1 / =winner)
and the oracle-plan realization (dev/distill/oracle_plan.py). With the switch off
the engine's output is byte-identical (golden hashes of the v5 engine taken before
the pass existed). All fixtures are synthetic (no Crello content)."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from engine.pipeline.geometry import union_box
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout import solver
from engine.pipeline.layout import strategy_registry as reg
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.types import Candidate
from tests.conftest import assert_valid
from tests.designs import design, shape, text

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())
V5 = {"CHAMELEON_STRATEGIES": "bisection,anchored", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
      "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1", "CHAMELEON_TEXT_SLACK": "1"}

# sha1 over (winner id, design) of resize_with_report on every target, the v5 stack,
# from the engine before the solver existed (git HEAD ae9ecc5).
GOLDEN_V5 = {
    "chameleon_sample_level1_warmup": "e690b15585772b5b0bea184f0410d0ad4e509a38",
    "chameleon_sample_level2_core": "f6e59e06a133a8c5be79d392559b646897184631",
    "chameleon_sample_level3_advanced": "fc545c9c4fbd893495d214c84c1530b1f2eb616d",
}


def _set(monkeypatch, value):
    for k, v in V5.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CHAMELEON_LLM", "heuristic")
    if value is None:
        monkeypatch.delenv("CHAMELEON_SOLVER", raising=False)
    else:
        monkeypatch.setenv("CHAMELEON_SOLVER", value)
    reg._modules.clear()


# --- switch off: byte-identical ----------------------------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_off_is_byte_identical(path, monkeypatch):
    from engine.resize import resize_with_report
    _set(monkeypatch, None)
    called = []
    monkeypatch.setattr(solver, "solve_all", lambda *a, **k: called.append(1) or a[-1])
    monkeypatch.setattr(solver, "solve_candidate", lambda *a, **k: called.append(1) or a[4])
    src = json.loads(path.read_text())["source"]
    h = hashlib.sha1()
    for t in TARGETS:
        design_, _, best, _ = resize_with_report(src, {"width": t["width"], "height": t["height"]})
        h.update(best.candidate_id.encode())
        h.update(json.dumps(design_, sort_keys=True).encode())
    assert h.hexdigest() == GOLDEN_V5[path.stem]
    assert not called


# --- the pass on a hand-built candidate ---------------------------------------------

SRC = design(
    shape("bg", 0, 0, 1080, 1080, fill_color="rgba(250,240,220,1)", z_index=0),
    text("headline", "Chase the light", 140, 410, 800, 130, 90, z_index=2, text_align="center"),
    text("sub", "Botanical sparkling infusions", 190, 560, 700, 60, 40, z_index=2, text_align="center"),
    shape("pill", 390, 700, 300, 90, fill_color="rgba(200,60,40,1)", corner_radius=45, z_index=2),
    text("cta", "Shop now", 420, 715, 240, 60, 40, z_index=3, text_align="center"))
TARGET = {"width": 1080, "height": 1350}


def _cand(src: dict, target: dict, moves: dict[str, dict]) -> Candidate:
    els = [dict(copy.deepcopy(e), **moves.get(e["id"], {})) for e in src["elements"]]
    return Candidate("test", {"canvas": dict(target), "elements": els}, "test", {})


def _boxes(design_: dict) -> dict:
    return {e["id"]: e for e in design_["elements"]}


# the column pushed to the top: 100 px above it, ~780 px of empty canvas below
LOPSIDED = {"bg": {"width": 1080, "height": 1350}, "headline": {"y": 100}, "sub": {"y": 300},
            "pill": {"y": 420}, "cta": {"y": 435}}


def test_lopsided_column_is_rebalanced(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    plan = heuristic_plan(SRC)
    cand = _cand(SRC, TARGET, LOPSIDED)
    assert not check_gates(SRC, cand.design, TARGET, plan, measurer)
    out = solver.solve_candidate(SRC, TARGET, plan, measurer, cand)
    assert out is not cand
    els = _boxes(out.design)
    top = els["headline"]["y"]
    bottom = TARGET["height"] - (els["pill"]["y"] + els["pill"]["height"])
    assert abs(top - bottom) < 0.2 * TARGET["height"]          # was 100 vs ~830
    assert not check_gates(SRC, out.design, TARGET, plan, measurer)
    assert_valid(SRC, out.design, TARGET)


def test_text_never_shrinks_and_headline_stays_largest(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    plan = heuristic_plan(SRC)
    cand = _cand(SRC, TARGET, LOPSIDED)
    out = solver.solve_candidate(SRC, TARGET, plan, measurer, cand)
    before, after = _boxes(cand.design), _boxes(out.design)
    for i, e in after.items():
        if e["type"] == "text":
            assert e["font_size"] >= before[i]["font_size"] - 0.01
            assert e["font_size"] <= 1.5 * _boxes(SRC)[i]["font_size"] + 0.01
    assert all(after["headline"]["font_size"] >= e["font_size"] for e in after.values() if e["type"] == "text")


def test_label_stays_in_its_pill(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    plan = heuristic_plan(SRC)
    out = solver.solve_candidate(SRC, TARGET, plan, measurer, _cand(SRC, TARGET, LOPSIDED))
    els = _boxes(out.design)
    p, c = els["pill"], els["cta"]
    assert p["x"] - 0.5 <= c["x"] and c["x"] + c["width"] <= p["x"] + p["width"] + 0.5
    assert p["y"] - 0.5 <= c["y"] and c["y"] + c["height"] <= p["y"] + p["height"] + 0.5


def test_failing_candidate_and_fallback_are_left_alone(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    plan = heuristic_plan(SRC)
    bad = _cand(SRC, TARGET, {**LOPSIDED, "cta": {"y": 435, "width": 20}})  # the label no longer fits
    assert check_gates(SRC, bad.design, TARGET, plan, measurer)
    assert solver.solve_candidate(SRC, TARGET, plan, measurer, bad) is bad
    fb = Candidate("fallback", bad.design, "fallback", {}, [], True)
    assert solver.solve_candidate(SRC, TARGET, plan, measurer, fb) is fb


def test_tighten_text_keeps_the_wrap(measurer):
    el = text("t", "Botanical sparkling infusions crafted for golden hour", 100, 100, 900, 300, 40,
              text_align="center")
    t = solver.tighten_text(el, measurer)
    assert t["width"] < el["width"]
    assert measurer.wrap(t, 40, t["width"]) == measurer.wrap(el, 40, el["width"])
    ink = max(measurer.line_widths(el, 40, el["width"]))
    assert abs((t["x"] + t["width"] / 2) - (el["x"] + el["width"] / 2)) < 0.01  # centred ink stays put
    assert t["width"] >= ink
    assert measurer.fits(t, 40, t["width"], t["height"])


def test_xy_tree_rows_columns_and_pinwheel():
    col = [(0, 0, 10, 10), (0, 20, 10, 10), (0, 40, 10, 10)]
    node = solver.xy_tree([0, 1, 2], col, 100, 100)
    assert node.kind == "stack" and node.axis == "y" and [c.leaf for c in node.children] == [0, 1, 2]
    row = [(0, 0, 10, 10), (20, 0, 10, 10)]
    assert solver.xy_tree([0, 1], row, 100, 100).axis == "x"
    pin = [(0, 0, 60, 20), (70, 0, 30, 60), (40, 70, 60, 30), (0, 30, 30, 70), (35, 30, 30, 30)]
    assert solver.xy_tree(list(range(5)), pin, 100, 100).kind == "free"


@pytest.mark.parametrize("target", ["leaderboard", "story", "billboard"])
def test_every_sample_candidate_solves_or_stays(target, measurer, monkeypatch):
    """The candidate is a feasible point of its own LP: never infeasible; every solved one passes the gates."""
    from engine.pipeline.layout.generate import generate_candidates
    _set(monkeypatch, None)
    t = next(x for x in TARGETS if x["name"] == target)
    tgt = {"width": t["width"], "height": t["height"]}
    src = json.loads(next(p for p in SAMPLES if "core" in p.stem).read_text())["source"]
    plan = heuristic_plan(src)
    cands = generate_candidates(src, tgt, plan, measurer)
    log = []
    out = solver.solve_all(src, tgt, plan, measurer, cands, log)
    statuses = {r.get("status") for r in log}
    assert statuses <= {"solved", "unchanged", "base_failed", None}, statuses
    assert any(r.get("status") == "solved" for r in log)
    for c, d in zip(cands, out):
        if d is not c:
            assert d.candidate_id == c.candidate_id
            assert not check_gates(src, d.design, tgt, plan, measurer)
            assert_valid(src, d.design, tgt)
            for r in log:
                assert r.get("vars", 0) < 400  # blocks, not elements


def test_winner_mode_keeps_the_pick(monkeypatch):
    from engine.resize import resize_with_report
    src = json.loads(next(p for p in SAMPLES if "core" in p.stem).read_text())["source"]
    tgt = {"width": 970, "height": 250}
    _set(monkeypatch, None)
    _, _, off, _ = resize_with_report(src, tgt)
    _set(monkeypatch, "winner")
    design_, plan, on, _ = resize_with_report(src, tgt)
    assert on.candidate_id == off.candidate_id
    assert_valid(src, design_, tgt)


# --- oracle plan: extract a discrete plan, realize it without the source coordinates ----

def test_oracle_plan_extract_and_realize(measurer, monkeypatch):
    from dev.distill import oracle_plan as op
    _set(monkeypatch, None)
    src = SRC
    tgt = {"width": 1584, "height": 396}
    plan = heuristic_plan(src)
    # a hand-made "Sonnet" layout: headline + sub on the left, the button on the right
    layout = design(
        shape("bg", 0, 0, 1584, 396, fill_color="rgba(250,240,220,1)", z_index=0),
        text("headline", "Chase the light", 120, 110, 760, 100, 80, z_index=2, text_align="left"),
        text("sub", "Botanical sparkling infusions", 120, 225, 700, 50, 36, z_index=2, text_align="left"),
        shape("pill", 1130, 150, 330, 96, fill_color="rgba(200,60,40,1)", corner_radius=45, z_index=2),
        text("cta", "Shop now", 1160, 172, 270, 52, 40, z_index=3, text_align="center"),
        width=1584, height=396)
    P = op.extract(src, tgt, layout, plan, measurer)
    assert P.tree[0] == "x"                                     # a row: [text column, button]
    assert P.units["headline"].anchor["x"] == "start"
    assert P.units["headline"].lines == 1
    assert P.units["headline"].kind == "text" and P.units["headline"].bucket > P.units["sub"].bucket
    naive, solved = op.realize(src, tgt, plan, P, measurer)
    for d in (naive, solved):
        assert d is not None
        assert not check_gates(src, d, tgt, plan, measurer)
        assert_valid(src, d, tgt)
    els = _boxes(solved)
    assert els["headline"]["x"] + els["headline"]["width"] <= els["pill"]["x"]  # order kept: text left of button
    assert els["headline"]["font_size"] > els["sub"]["font_size"]
