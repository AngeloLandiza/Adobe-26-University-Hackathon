"""Balance pass (engine/pipeline/layout/balance.py, CHAMELEON_BALANCE=1) and its
detectors (dev/distill/balance_flaws.py). With the switch off the engine's
output is byte-identical (golden hashes of the v4 engine taken before the
pass existed)."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout import balance
from engine.pipeline.layout import strategy_registry as reg
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Candidate
from tests.conftest import assert_valid
from tests.designs import design, shape, text

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())
V4 = {"CHAMELEON_STRATEGIES": "bisection,anchored", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
      "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1"}

# sha1 over (winner id, design) of resize_with_report on every target, the v4 stack, from the
# engine before this pass existed (git HEAD e15fb0f).
GOLDEN_V4 = {
    "chameleon_sample_level1_warmup": "000eae1dbcfd82345f9ba4773a2b1bce45ecaecd",
    "chameleon_sample_level2_core": "73f48df675757493066be73e6b330c7c57a4c60d",
    "chameleon_sample_level3_advanced": "5debf254cc3007adba52df1fed9f3047c01769be",
}


@pytest.fixture(scope="module")
def measurer():
    return TextMeasurer()


def _set(monkeypatch, on: bool):
    for k, v in V4.items():
        monkeypatch.setenv(k, v)
    if on:
        monkeypatch.setenv("CHAMELEON_BALANCE", "1")
    else:
        monkeypatch.delenv("CHAMELEON_BALANCE", raising=False)
    reg._modules.clear()


def _engine_digest(src: dict) -> str:
    from engine.resize import resize_with_report
    h = hashlib.sha1()
    for t in TARGETS:
        design_, _, best, _ = resize_with_report(src, {"width": t["width"], "height": t["height"]})
        h.update(best.candidate_id.encode())
        h.update(json.dumps(design_, sort_keys=True).encode())
    return h.hexdigest()


# --- switch off: byte-identical ----------------------------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_off_is_byte_identical(path, monkeypatch):
    _set(monkeypatch, on=False)
    called = []
    monkeypatch.setattr(balance, "balance_all", lambda *a, **k: called.append(1) or a[-1])
    assert _engine_digest(json.loads(path.read_text())["source"]) == GOLDEN_V4[path.stem]
    assert not called


# --- the steps on hand-built candidates ---------------------------------------------

def _cand(src: dict, target: dict, moves: dict[str, dict]) -> Candidate:
    """The source elements with `moves` applied, on the target canvas."""
    els = []
    for e in src["elements"]:
        els.append(dict(copy.deepcopy(e), **moves.get(e["id"], {})))
    return Candidate("test", {"canvas": dict(target), "elements": els}, "test", {})


SRC = design(
    shape("bg", 0, 0, 1080, 1080, fill_color="rgba(250,240,220,1)", z_index=0),
    shape("band", 0, 0, 1080, 160, fill_color="rgba(20,20,40,1)", z_index=1),
    text("headline", "Chase the light", 140, 410, 800, 130, 90, z_index=2, text_align="center"),
    text("sub", "Botanical sparkling infusions", 190, 560, 700, 60, 40, z_index=2, text_align="center"),
    shape("pill", 390, 700, 300, 90, fill_color="rgba(200,60,40,1)", corner_radius=45, z_index=2),
    text("cta", "Shop now", 420, 715, 240, 60, 40, z_index=3, text_align="center"),
    text("caption", "Ships late June", 390, 810, 300, 40, 26, z_index=2, text_align="center"))
TARGET = {"width": 1600, "height": 900}


def test_band_spans_the_target_again(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    plan = heuristic_plan(SRC)
    cand = _cand(SRC, TARGET, {"bg": {"width": 1600, "height": 900},
                               "band": {"x": 300, "width": 1000, "height": 133},
                               "headline": {"x": 400, "y": 300}, "sub": {"x": 450, "y": 440},
                               "pill": {"x": 650, "y": 560}, "cta": {"x": 680, "y": 575},
                               "caption": {"x": 650, "y": 670}})
    assert not check_gates(SRC, cand.design, TARGET, plan, measurer)
    out = balance.balance(SRC, TARGET, plan, measurer, cand)
    band = next(e for e in out.design["elements"] if e["id"] == "band")
    assert (band["x"], band["width"]) == (0, 1600)
    assert not check_gates(SRC, out.design, TARGET, plan, measurer)
    assert_valid(SRC, out.design, TARGET)


def test_a_caption_that_drifted_comes_back_to_its_button(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    plan = heuristic_plan(SRC)
    cand = _cand(SRC, TARGET, {"bg": {"width": 1600, "height": 900}, "band": {"width": 1600, "height": 133},
                               "headline": {"x": 400, "y": 230}, "sub": {"x": 450, "y": 370},
                               "pill": {"x": 650, "y": 470}, "cta": {"x": 680, "y": 485},
                               "caption": {"x": 650, "y": 790}})
    out = balance.balance(SRC, TARGET, plan, measurer, cand)
    els = {e["id"]: e for e in out.design["elements"]}
    gap = els["caption"]["y"] - (els["pill"]["y"] + els["pill"]["height"])
    assert gap < 60  # it sat 20 px under the button in the source; 230 px in the candidate
    assert not check_gates(SRC, out.design, TARGET, plan, measurer)


def test_a_lopsided_stack_moves_toward_equal_margins(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    monkeypatch.setattr(balance, "FILL_UNUSED", 1.0)  # shift only (growth tested below)
    src = design(shape("bg", 0, 0, 1080, 1080, z_index=0),
                 text("headline", "Chase the light", 140, 380, 800, 130, 90, z_index=1, text_align="center"),
                 text("sub", "Botanical sparkling infusions", 190, 520, 700, 60, 40, z_index=1, text_align="center"),
                 text("legal", "Please enjoy responsibly", 340, 620, 400, 40, 24, z_index=1, text_align="center"))
    target = {"width": 1080, "height": 1920}
    plan = heuristic_plan(src)
    cand = _cand(src, target, {"bg": {"width": 1080, "height": 1920}, "headline": {"y": 200},
                               "sub": {"y": 340}, "legal": {"y": 440}})
    out = balance.balance(src, target, plan, measurer, cand)
    els = {e["id"]: e for e in out.design["elements"]}
    top = els["headline"]["y"]
    bottom = 1920 - (els["legal"]["y"] + 40)
    assert els["headline"]["y"] > 200 and abs(top - bottom) < 0.3 * (top + bottom)
    assert els["sub"]["y"] - els["headline"]["y"] == pytest.approx(140)  # the block moves as one


def test_a_deliberately_lopsided_source_is_left_alone(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    monkeypatch.setattr(balance, "FILL_UNUSED", 1.0)
    src = design(shape("bg", 0, 0, 1080, 1080, z_index=0),
                 text("headline", "Chase the light", 140, 120, 800, 130, 90, z_index=1, text_align="center"),
                 text("sub", "Botanical sparkling infusions", 190, 260, 700, 60, 40, z_index=1, text_align="center"))
    target = {"width": 1080, "height": 1920}
    plan = heuristic_plan(src)
    cand = _cand(src, target, {"bg": {"width": 1080, "height": 1920}, "headline": {"y": 200}, "sub": {"y": 340}})
    out = balance.balance(src, target, plan, measurer, cand)
    assert out is cand


def test_a_narrow_block_grows_about_its_centre(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    src = design(shape("bg", 0, 0, 1080, 1080, z_index=0),
                 text("headline", "Chase the light", 290, 440, 500, 90, 60, z_index=1, text_align="center"),
                 text("sub", "Sparkling infusions", 340, 540, 400, 50, 30, z_index=1, text_align="center"))
    target = {"width": 1080, "height": 1350}
    plan = heuristic_plan(src)
    cand = _cand(src, target, {"bg": {"width": 1080, "height": 1350}, "headline": {"y": 575}, "sub": {"y": 675}})
    out = balance.balance(src, target, plan, measurer, cand)
    els = {e["id"]: e for e in out.design["elements"]}
    assert els["headline"]["font_size"] == pytest.approx(60 * balance.MAX_GROW, abs=0.02)
    assert els["sub"]["font_size"] == pytest.approx(30 * balance.MAX_GROW, abs=0.02)
    assert not check_gates(src, out.design, target, plan, measurer)


def test_the_fallback_and_failing_candidates_are_untouched(measurer, monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    plan = heuristic_plan(SRC)
    moves = {"bg": {"width": 1600, "height": 900}, "band": {"x": 300, "width": 1000, "height": 133},
             "headline": {"x": 400, "y": 300}, "sub": {"x": 450, "y": 440}, "pill": {"x": 650, "y": 560},
             "cta": {"x": 680, "y": 575}, "caption": {"x": 650, "y": 670}}
    fb = _cand(SRC, TARGET, moves)
    fb.is_fallback = True
    bad = _cand(SRC, TARGET, {**moves, "cta": {"x": 680, "y": 575, "font_size": 400}})  # text no longer fits: G7
    assert check_gates(SRC, bad.design, TARGET, plan, measurer)
    out = balance.balance_all(SRC, TARGET, plan, measurer, [fb, bad])
    assert out[0] is fb and out[1] is bad


# --- the whole engine with the switch on ---------------------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_on_every_target_passes_the_gates(path, measurer, monkeypatch):
    from engine.resize import resize_with_report
    _set(monkeypatch, on=True)
    src = json.loads(path.read_text())["source"]
    for t in TARGETS:
        target = {"width": t["width"], "height": t["height"]}
        out, plan, _, _ = resize_with_report(src, target)
        assert not check_gates(src, out, target, plan, measurer), t["name"]
        assert_valid(src, out, target)


def test_on_is_deterministic(monkeypatch):
    _set(monkeypatch, on=True)
    src = json.loads(SAMPLES[-1].read_text())["source"]
    assert _engine_digest(src) == _engine_digest(src)


# --- detectors ---------------------------------------------------------------------

def test_detectors_fire_on_a_broken_band_a_split_pair_and_a_lopsided_stack(measurer):
    from dev.distill.balance_flaws import detect
    plan = heuristic_plan(SRC)
    good = _cand(SRC, TARGET, {"bg": {"width": 1600, "height": 900}, "band": {"width": 1600, "height": 133},
                               "headline": {"x": 400, "y": 290}, "sub": {"x": 450, "y": 430},
                               "pill": {"x": 650, "y": 530}, "cta": {"x": 680, "y": 545},
                               "caption": {"x": 650, "y": 640}}).design
    bad = _cand(SRC, TARGET, {"bg": {"width": 1600, "height": 900}, "band": {"x": 300, "width": 1000, "height": 133},
                              "headline": {"x": 400, "y": 160}, "sub": {"x": 450, "y": 300},
                              "pill": {"x": 650, "y": 400}, "cta": {"x": 680, "y": 415},
                              "caption": {"x": 650, "y": 840}}).design
    d_good, d_bad = detect(SRC, good, plan, measurer), detect(SRC, bad, plan, measurer)
    assert not d_good["band_broken"]["fire"] and d_bad["band_broken"]["fire"]
    assert not d_good["group_split"]["fire"] and d_bad["group_split"]["fire"]
    lop = _cand(SRC, TARGET, {"bg": {"width": 1600, "height": 900}, "band": {"width": 1600, "height": 133},
                              "headline": {"x": 100, "y": 290}, "sub": {"x": 150, "y": 430},
                              "pill": {"x": 350, "y": 530}, "cta": {"x": 380, "y": 545},
                              "caption": {"x": 350, "y": 640}}).design
    assert detect(SRC, lop, plan, measurer)["margin_imbalance"]["fire"]
    assert not d_good["margin_imbalance"]["fire"]


def test_critique_paragraphs_split_by_side():
    from dev.distill.balance_flaws import mentions, side_paragraphs
    p = side_paragraphs("A keeps the order. Its CTA is tiny. B: the dark header bar no longer spans the canvas. "
                        "Both leave the bottom third empty.")
    assert "header bar" in p["b"] and "header bar" not in p["a"]
    assert "bottom third" in p["a"] and "bottom third" in p["b"]
    assert mentions(p["b"])["band_broken"] and not mentions(p["a"])["band_broken"]
