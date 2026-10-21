"""Text robustness pass (CHAMELEON_TEXT_SLACK; engine/pipeline/layout/text_slack.py)
and its study tool (dev/distill/text_robustness.py).

With the switch unset (or 0) the engine must produce exactly what it did
before (golden hashes over the resize() output, taken before the pass
existed); with it on, every text box keeps its wrap with slack_px of room,
pill labels stay on their source line count inside their pill, and every
design passes the hard gates.
"""

import hashlib
import json
from pathlib import Path

import pytest

from dev.distill import text_robustness as tr
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout import text_slack as ts
from engine.pipeline.priority import heuristic_plan

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())
V4 = {"CHAMELEON_STRATEGIES": "bisection,anchored", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
      "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1"}
V4_KEYS = (*V4, "CHAMELEON_TEXT_SLACK", "CHAMELEON_POLISH", "CHAMELEON_PICKER", "CHAMELEON_RELPRESERVE",
           "CHAMELEON_NEAR_FIT_GUARD", "CHAMELEON_FLEX")

# sha1 over json(resize design, sorted keys) for every target in dev/targets.json order,
# from the engine before CHAMELEON_TEXT_SLACK existed: today's engine (no switches) ...
GOLDEN = {
    "chameleon_sample_level1_warmup": "cdaba7a548a24367bdb2cee3b7e1833bc497358f",
    "chameleon_sample_level2_core": "9c11022d6bfdf67aef13f546917a7ab36cd418b9",
    "chameleon_sample_level3_advanced": "d297ff4ffbe391dfa9b9a1603e147192ff330fe2",
}
# ... and the v4 stack (V4 switches).
GOLDEN_V4 = {"chameleon_sample_level2_core": "72baef3e1287c2d37641b9dd6fa12e4e0d9b1693"}


def _clean(monkeypatch, env: dict) -> None:
    for k in V4_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def _digest(src: dict) -> str:
    from engine.resize import resize_with_report
    h = hashlib.sha1()
    for t in TARGETS:
        h.update(json.dumps(resize_with_report(src, {"width": t["width"], "height": t["height"]})[0],
                            sort_keys=True).encode())
    return h.hexdigest()


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_switch_off_is_byte_identical(path, monkeypatch):
    _clean(monkeypatch, {})
    assert _digest(json.loads(path.read_text())["source"]) == GOLDEN[path.stem]


def test_switch_zero_is_byte_identical_on_v4(monkeypatch):
    _clean(monkeypatch, {**V4, "CHAMELEON_TEXT_SLACK": "0"})
    src = json.loads(Path("input/chameleon_sample_level2_core.json").read_text())["source"]
    assert _digest(src) == GOLDEN_V4["chameleon_sample_level2_core"]


def test_mode(monkeypatch):
    for value, want in ((None, None), ("0", None), ("1", "final"), ("all", "all"), ("yes", None)):
        if value is None:
            monkeypatch.delenv(ts.ENV, raising=False)
        else:
            monkeypatch.setenv(ts.ENV, value)
        assert ts.mode() == want


# ---------------------------------------------------------------- synthetic designs

LABEL = "Shop the Summer Line"


def _text(eid, x, y, w, h, size, content, z, **kw):
    return {"id": eid, "type": "text", "x": x, "y": y, "width": w, "height": h, "z_index": z, "content": content,
            "font_family": "Solstice Sans", "font_size": size, "color": "rgba(20,20,20,1)", **kw}


def ad() -> dict:
    """bg; a 2-line left-aligned headline; a right-aligned line of copy; a centred caption; a CTA pill."""
    return {"canvas": {"width": 600, "height": 400}, "elements": [
        {"id": "bg", "type": "shape", "x": 0, "y": 0, "width": 600, "height": 400, "z_index": 0,
         "shape_type": "rectangle", "fill_color": "rgba(240,230,210,1)"},
        _text("headline", 40, 30, 400, 110, 40, "Fresh picks for summer", 1, text_align="left", font_weight="bold"),
        _text("copy", 160, 150, 400, 30, 18, "Small batch, big flavour", 2, text_align="right"),
        _text("caption", 100, 200, 400, 30, 14, "Every week of the season", 3, text_align="center"),
        {"id": "cta_bg", "type": "shape", "x": 150, "y": 300, "width": 300, "height": 60, "z_index": 4,
         "shape_type": "rectangle", "fill_color": "rgba(230,90,40,1)", "corner_radius": 30, "group_id": "cta"},
        _text("cta_label", 175, 315, 250, 30, 20, LABEL, 5, group_id="cta", text_align="center",
              color="rgba(255,255,255,1)"),
    ]}


def _tight(m, el: dict) -> dict:
    """`el` with its box exactly as narrow as its wrap allows (and as tall as its block)."""
    out = dict(el, width=ts._need(m, el, el["font_size"], m.wrap(el, el["font_size"], el["width"])) + 0.001)
    if (el.get("text_align") or "left") == "right":
        out["x"] = el["x"] + el["width"] - out["width"]
    elif el.get("text_align") == "center":
        out["x"] = el["x"] + (el["width"] - out["width"]) / 2
    out["height"] = m.block_height(out, out["font_size"], out["width"])
    return out


def _design(src: dict, changes: dict) -> dict:
    return {"canvas": dict(src["canvas"]), "elements": [dict(e, **changes.get(e["id"], {})) for e in src["elements"]]}


def _els(d: dict) -> dict:
    return {e["id"]: e for e in d["elements"]}


@pytest.fixture
def setup(measurer, monkeypatch):
    _clean(monkeypatch, {})
    src = ad()
    return src, heuristic_plan(src), {"width": 600, "height": 400}


def test_tight_boxes_widen_by_alignment_and_keep_their_wrap(setup, measurer):
    src, plan, target = setup
    els = _els(src)
    tight = {i: _tight(measurer, els[i]) for i in ("headline", "copy", "caption", "cta_label")}
    design = _design(src, tight)
    assert check_gates(src, design, target, plan, measurer) == []
    before = tr.analyze(src, design, plan, measurer)
    assert set(before["fragile_slack"]) == {"headline", "copy", "caption", "cta_label"}  # wrap changes 1.5 px narrower
    res = ts.fix_design(src, target, plan, measurer, design)
    out = _els(res.design)
    assert check_gates(src, res.design, target, plan, measurer) == []
    after = tr.analyze(src, res.design, plan, measurer)
    assert after["fragile_1px"] == after["fragile_slack"] == []
    for i, t in tight.items():
        o = out[i]
        assert o["font_size"] == t["font_size"] and o["content"] == t["content"]
        assert measurer.wrap(o, o["font_size"], o["width"]) == measurer.wrap(t, t["font_size"], t["width"])
        assert o["width"] - max(measurer.line_widths(o, o["font_size"], o["width"])) >= ts.slack_px(o["width"]) - 0.01
    assert out["headline"]["x"] == pytest.approx(tight["headline"]["x"], abs=0.01)                     # left
    assert out["copy"]["x"] + out["copy"]["width"] == pytest.approx(tight["copy"]["x"] + tight["copy"]["width"],
                                                                    abs=0.02)                         # right
    for i in ("caption", "cta_label"):                                                                # centred
        assert out[i]["x"] + out[i]["width"] / 2 == pytest.approx(tight[i]["x"] + tight[i]["width"] / 2, abs=0.02)
    # the pill already had room for its wider label: it is untouched
    assert out["cta_bg"] == els["cta_bg"]
    # idempotent
    again = ts.fix_design(src, target, plan, measurer, res.design)
    assert again.design is res.design and again.changed == []


def _two_line_label(measurer, src: dict, pill_x: float | None = None, pad: float = 2) -> dict:
    """The CTA label wrapped onto two lines (8 px too narrow for one) in a pill `pad` px wider on each side."""
    label = _els(src)["cta_label"]
    one = max(measurer.line_widths(label, 20, 1000))
    w = one - 8
    assert measurer.line_count(label, 20, w) == 2
    h = measurer.block_height(label, 20, w)
    pw = w + 2 * pad
    px = (600 - pw) / 2 if pill_x is None else pill_x
    return {"cta_bg": {"x": px, "y": 290, "width": pw, "height": h + 20},
            "cta_label": {"x": px + pad, "y": 300, "width": w, "height": h}}


def test_two_line_pill_label_goes_back_to_one_line_and_the_pill_widens(setup, measurer):
    src, plan, target = setup
    changes = _two_line_label(measurer, src)
    design = _design(src, changes)
    assert check_gates(src, design, target, plan, measurer) == []
    assert tr.analyze(src, design, plan, measurer)["pill_multiline"] == ["cta_label"]
    res = ts.fix_design(src, target, plan, measurer, design)
    out = _els(res.design)
    assert check_gates(src, res.design, target, plan, measurer) == []
    a = tr.analyze(src, res.design, plan, measurer)
    assert a["pill_multiline"] == a["pill_outside"] == a["fragile_slack"] == []
    lab, pill = out["cta_label"], out["cta_bg"]
    assert lab["font_size"] == 20 and measurer.line_count(lab, 20, lab["width"]) == 1
    old = changes["cta_bg"]
    assert pill["width"] > old["width"]
    assert pill["x"] + pill["width"] / 2 == pytest.approx(old["x"] + old["width"] / 2, abs=0.02)      # symmetric
    assert pill["y"] == old["y"] and pill["height"] == pytest.approx(old["height"], abs=0.01)
    # the one-line label is re-centred on its old ink
    b2 = measurer.block_height(design["elements"][5], 20, changes["cta_label"]["width"])
    assert lab["y"] + measurer.block_height(lab, 20, lab["width"]) / 2 == pytest.approx(300 + b2 / 2, abs=0.02)


def test_pill_at_the_canvas_edge_shrinks_its_label_instead(setup, measurer):
    src, plan, target = setup
    changes = _two_line_label(measurer, src, pill_x=0)   # pill flush with the left edge: it can't grow
    w = changes["cta_label"]["width"]
    design = _design(src, changes)
    assert check_gates(src, design, target, plan, measurer) == []
    res = ts.fix_design(src, target, plan, measurer, design)
    out = _els(res.design)
    assert check_gates(src, res.design, target, plan, measurer) == []
    lab = out["cta_label"]
    from engine.pipeline.layout.place import floor_px
    assert floor_px("cta", 600, 400) <= lab["font_size"] < 20
    assert measurer.line_count(lab, lab["font_size"], lab["width"]) == 1 and ts.robust(measurer, lab)
    assert out["cta_bg"]["x"] == 0 and out["cta_bg"]["width"] == pytest.approx(w + 4, abs=0.01)
    assert tr.analyze(src, res.design, plan, measurer)["pill_outside"] == []


def test_blocked_text_shrinks_at_most_max_shrink_and_never_below_its_floor(measurer, monkeypatch):
    _clean(monkeypatch, {})
    src = ad()
    src["elements"].append({"id": "logo", "type": "image", "x": 480, "y": 20, "width": 90, "height": 30, "z_index": 6,
                            "asset_id": "solstice_logo_lockup.png", "fit": "contain"})
    plan, target = heuristic_plan(src), {"width": 600, "height": 400}
    els = _els(src)
    tight = _tight(measurer, els["copy"])
    tight["x"] = 600 - tight["width"]                   # right-aligned, flush with the right edge ...
    wall = {"x": tight["x"] - 90, "y": tight["y"]}       # ... and the logo flush on its left
    design = _design(src, {"copy": tight, "logo": wall})
    assert check_gates(src, design, target, plan, measurer) == []
    res = ts.fix_design(src, target, plan, measurer, design)
    out = _els(res.design)["copy"]
    assert check_gates(src, res.design, target, plan, measurer) == []
    assert tight["font_size"] * (1 - ts.MAX_SHRINK) - 0.01 <= out["font_size"] < tight["font_size"]
    assert measurer.line_count(out, out["font_size"], out["width"]) == 1 and ts.robust(measurer, out)
    assert out["x"] == pytest.approx(tight["x"], abs=0.01) and out["width"] == pytest.approx(tight["width"], abs=0.01)


def test_short_box_grows_to_its_block(setup, measurer):
    src, plan, target = setup
    els = _els(src)
    cap = dict(els["caption"])
    cap["height"] = measurer.block_height(cap, cap["font_size"], cap["width"]) - 0.8   # G7 allows 1 px short
    design = _design(src, {"caption": cap})
    assert check_gates(src, design, target, plan, measurer) == []
    out = _els(ts.fix_design(src, target, plan, measurer, design).design)["caption"]
    assert out["height"] >= measurer.block_height(out, out["font_size"], out["width"]) - 1e-6
    assert out["y"] == cap["y"]


def test_a_failing_candidate_is_left_alone(setup, measurer):
    src, plan, target = setup
    els = _els(src)
    design = _design(src, {"copy": dict(_tight(measurer, els["copy"]), x=590)})   # off canvas: fails G4
    assert check_gates(src, design, target, plan, measurer)
    assert ts.fix_design(src, target, plan, measurer, design).design is design


def test_apply_keeps_the_candidate_identity(setup, measurer):
    from engine.pipeline.types import Candidate
    src, plan, target = setup
    els = _els(src)
    cand = Candidate("x/1", _design(src, {"caption": _tight(measurer, els["caption"])}), "row", {"a": 1}, ["u"], True)
    out = ts.apply(src, target, plan, measurer, cand)
    assert out is not cand and out.design != cand.design
    assert (out.candidate_id, out.template, out.params, out.degraded_unit_ids, out.is_fallback) == \
           ("x/1", "row", {"a": 1}, ["u"], True)
    same = Candidate("y", _design(src, {}), "row", {}, [], False)
    assert ts.apply(src, target, plan, measurer, same) is same


# ---------------------------------------------------------------- the engine with the switch on

@pytest.mark.parametrize("value", ["1", "all"])
def test_v4_with_text_slack_has_no_fragile_text_on_the_core_sample(value, measurer, monkeypatch):
    from engine.resize import resize_with_report
    from tests.conftest import assert_valid
    _clean(monkeypatch, {**V4, "CHAMELEON_TEXT_SLACK": value})
    src = json.loads(Path("input/chameleon_sample_level2_core.json").read_text())["source"]
    for t in [t for t in TARGETS if t["name"] in ("medium_rectangle", "leaderboard", "half_page", "story")]:
        canvas = {"width": t["width"], "height": t["height"]}
        design, plan, _, _ = resize_with_report(src, canvas)
        assert_valid(src, design, canvas)
        assert check_gates(src, design, canvas, plan, measurer) == []
        a = tr.analyze(src, design, plan, measurer)
        assert a["fragile_1px"] == [] and a["pill_multiline"] == [] and a["pill_outside"] == [], t["name"]


# ---------------------------------------------------------------- the study tool

def test_analyze_counts_fragile_and_pill_problems(setup, measurer):
    src, plan, target = setup
    assert tr.pill_pairs(src, plan) == {"cta_label": "cta_bg"}
    els = _els(src)
    a = tr.analyze(src, src, plan, measurer)
    assert a["texts"] == 4 and a["pills"] == 1 and a["fragile_1px"] == [] and a["pill_outside"] == []
    lab = dict(els["cta_label"], width=max(measurer.line_widths(els["cta_label"], 20, 1000)) - 8, height=60)
    a = tr.analyze(src, _design(src, {"cta_label": lab}), plan, measurer)
    assert a["pill_multiline"] == a["pill_over_src"] == ["cta_label"]
    assert a["pill_outside"] == ["cta_label"]      # the second line runs below the pill (y 300..360)
    rows = [{"cell": "c", "candidate_id": "a", "analysis": a, "design": src}]
    s = tr.summarize(rows)
    assert s["cells"] == 1 and s["pill_multiline"] == 1
    other = [{"cell": "c", "candidate_id": "a", "analysis": a,
              "design": _design(src, {"copy": dict(els["copy"], font_size=17.1)})}]
    c = tr.compare(rows, other)
    assert c["designs_changed"] == 1 and c["picks_changed"] == 0 and c["font_changed_texts"] == 1
    assert c["font_min_pct"] == pytest.approx(-5.0)
