"""Layer preservation (CHAMELEON_LAYERS=1; engine/pipeline/layout/layer_groups.py),
its detectors (dev/distill/layer_flaws.py) and the crello_dev split
(dev/distill/make_crello_dev.py).

With the switch unset the engine must produce exactly what it did before
(golden hashes); with it on, containers keep their contents, round shapes
keep their aspect, full-bleed layers fill the canvas, and every candidate
passes the hard gates. All fixtures here are synthetic (no Crello content).
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from dev.distill import ab_engine as ab
from dev.distill import layer_flaws
from dev.distill import make_crello_dev as mcd
from dev.distill.corpus import load_corpus, load_crello_dev
from engine.pipeline.geometry import overlap_area, rotated_aabb, union_box
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout import layer_groups
from engine.pipeline.layout.strategies.bisection import candidates
from engine.pipeline.priority import heuristic_plan

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())

# sha1 over the heuristic plan and every bisection v2 candidate (CHAMELEON_BISECT_V2=1)
# for each sample on every target, from the engine as it was before CHAMELEON_LAYERS existed.
GOLDEN = {
    "chameleon_sample_level1_warmup": "e1b8a77fc01a89df29f44ba7295c07e6b87225c2",
    "chameleon_sample_level2_core": "bd41f6c4d19dbc3eb0bf4dc4981b1d7a513a0870",
    "chameleon_sample_level3_advanced": "3fe471b5141290d702e001d50224f3c86df16ddf",
}


def _digest(src: dict, measurer) -> str:
    plan = heuristic_plan(src)
    h = hashlib.sha1(json.dumps([asdict(u) for u in plan.units], sort_keys=True).encode())
    for t in TARGETS:
        for c in candidates(src, {"width": t["width"], "height": t["height"]}, plan, measurer):
            h.update(c.candidate_id.encode())
            h.update(json.dumps(c.design, sort_keys=True).encode())
    return h.hexdigest()


@pytest.mark.parametrize("value", [None, "0"])
@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_switch_off_is_byte_identical(path, value, measurer, monkeypatch):
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "1")
    if value is None:
        monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    else:
        monkeypatch.setenv("CHAMELEON_LAYERS", value)
    assert _digest(json.loads(path.read_text())["source"], measurer) == GOLDEN[path.stem]


# ---------------------------------------------------------------- a synthetic layered ad

def _text(eid, x, y, w, h, size, content, z, **kw):
    return {"id": eid, "type": "text", "x": x, "y": y, "width": w, "height": h, "z_index": z, "content": content,
            "font_family": "Solstice Sans", "font_size": size, "color": "rgba(20,20,20,1)", "text_align": "center",
            **kw}


def layered_ad() -> dict:
    """bg colour + full-bleed cover photo; a white card holding a headline, a
    line of copy and a button; a circle holding a price; a logo on the photo."""
    return {"canvas": {"width": 1080, "height": 1080}, "elements": [
        {"id": "bg", "type": "shape", "x": 0, "y": 0, "width": 1080, "height": 1080, "z_index": 0,
         "shape_type": "rectangle", "fill_color": "rgba(240,230,210,1)"},
        {"id": "photo", "type": "image", "x": 0, "y": 0, "width": 1080, "height": 1080, "z_index": 1,
         "asset_id": "solstice_bg_gradient.png", "fit": "cover"},
        {"id": "card", "type": "shape", "x": 90, "y": 560, "width": 900, "height": 430, "z_index": 2,
         "shape_type": "rectangle", "fill_color": "rgba(255,255,255,1)", "corner_radius": 24},
        _text("headline", 140, 600, 800, 90, 64, "SUMMER SALE", 3, font_weight="bold"),
        _text("body", 140, 720, 800, 80, 28, "Fresh picks for warm days, every week of the season.", 4),
        {"id": "cta_bg", "type": "shape", "x": 380, "y": 860, "width": 320, "height": 90, "z_index": 5,
         "shape_type": "rectangle", "fill_color": "rgba(230,90,40,1)", "corner_radius": 45, "group_id": "cta"},
        _text("cta_label", 410, 880, 260, 50, 32, "Shop now", 6, group_id="cta", color="rgba(255,255,255,1)"),
        {"id": "dot", "type": "shape", "x": 760, "y": 80, "width": 240, "height": 240, "z_index": 7,
         "shape_type": "ellipse", "fill_color": "rgba(230,90,40,1)"},
        _text("price", 800, 160, 160, 70, 48, "-30%", 8, color="rgba(255,255,255,1)"),
        {"id": "logo", "type": "image", "x": 90, "y": 90, "width": 300, "height": 100, "z_index": 9,
         "asset_id": "solstice_logo_lockup.png", "fit": "contain"},
    ]}


def test_background_stack_takes_every_full_bleed_layer():
    src = layered_ad()
    assert layer_groups.background_stack(src["elements"], 1080, 1080) == {"bg", "photo"}
    # a full-canvas layer with copy painted under it is not background
    els = [dict(e, z_index=10) if e["id"] == "photo" else e for e in src["elements"]]
    assert layer_groups.background_stack(els, 1080, 1080) == {"bg"}


def test_plan_marks_the_stack_only_with_the_switch(monkeypatch):
    monkeypatch.delenv("CHAMELEON_LAYERS", raising=False)
    off = {u.unit_id: u.role for u in heuristic_plan(layered_ad()).units}
    monkeypatch.setenv("CHAMELEON_LAYERS", "1")
    on = {u.unit_id: u.role for u in heuristic_plan(layered_ad()).units}
    assert off["bg"] == on["bg"] == "background"
    assert off["photo"] != "background" and on["photo"] == "background"
    assert {k: v for k, v in off.items() if k != "photo"} == {k: v for k, v in on.items() if k != "photo"}


def test_detect_finds_the_card_and_the_circle(monkeypatch):
    monkeypatch.setenv("CHAMELEON_LAYERS", "1")
    src = layered_ad()
    layers = layer_groups.detect(src, heuristic_plan(src))
    by = {g.container: g for g in layers.top_level()}
    assert set(by) == {"card", "dot"}
    assert by["card"].stretch and not by["dot"].stretch
    assert set(by["card"].content_ids) == {"headline", "body", "cta"}
    assert by["dot"].content_ids == ["price"]
    assert all(p > 0 for p in by["card"].pads)
    kinds = {u.unit_id: u.kind for u in layers.plan.units}
    assert kinds["layers:card"] == layer_groups.KIND and "headline" not in kinds and "card" not in kinds
    assert {i for u in layers.plan.units for i in u.element_ids} == {e["id"] for e in src["elements"]}


def _held(design: dict, container: str, ids: list[str], tol: float = 0.5) -> bool:
    els = {e["id"]: e for e in design["elements"]}
    c = rotated_aabb(els[container])
    for i in ids:
        b = rotated_aabb(els[i])
        if not (b[0] >= c[0] - tol and b[1] >= c[1] - tol and b[0] + b[2] <= c[0] + c[2] + tol
                and b[1] + b[3] <= c[1] + c[3] + tol):
            return False
    return True


@pytest.mark.parametrize("target", [t for t in TARGETS if abs(t["width"] / t["height"] - 1) > 0.02],
                         ids=lambda t: t["name"])
def test_layers_keep_containers_and_pass_the_gates(target, measurer, monkeypatch):
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "1")
    monkeypatch.setenv("CHAMELEON_LAYERS", "1")
    src = layered_ad()
    plan = heuristic_plan(src)
    canvas = {"width": target["width"], "height": target["height"]}
    cands = candidates(src, canvas, plan, measurer)
    assert cands
    for c in cands:
        assert check_gates(src, c.design, canvas, plan, measurer) == []
        els = {e["id"]: e for e in c.design["elements"]}
        assert _held(c.design, "card", ["headline", "body", "cta_bg", "cta_label"]), c.candidate_id
        assert _held(c.design, "dot", ["price"]), c.candidate_id
        assert abs(els["dot"]["width"] / els["dot"]["height"] - 1) < 0.02
        assert (els["photo"]["x"], els["photo"]["y"], els["photo"]["width"], els["photo"]["height"]) == \
               (0, 0, target["width"], target["height"])
        det = layer_flaws.container_flaws(src, c.design, plan)
        assert not det["container_empty"]["fire"] and not det["container_split"]["fire"]
        assert not layer_flaws.background_flaws(src, c.design)["background_shrunk"]["fire"]
        assert set(c.degraded_unit_ids) <= {u.unit_id for u in plan.units}


def test_layers_are_deterministic(measurer, monkeypatch):
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "1")
    monkeypatch.setenv("CHAMELEON_LAYERS", "1")
    src = layered_ad()
    plan = heuristic_plan(src)
    t = {"width": 728, "height": 90}
    a = [json.dumps(c.design, sort_keys=True) for c in candidates(src, t, plan, measurer)]
    b = [json.dumps(c.design, sort_keys=True) for c in candidates(src, t, plan, measurer)]
    assert a == b and a


def test_resize_with_layers_returns_a_valid_design(measurer, monkeypatch):
    from engine.resize import resize_with_report
    from tests.conftest import assert_valid
    monkeypatch.setenv("CHAMELEON_STRATEGIES", "bisection")
    monkeypatch.setenv("CHAMELEON_BISECT_V2", "1")
    monkeypatch.setenv("CHAMELEON_LAYERS", "1")
    src = layered_ad()
    for t in ({"width": 160, "height": 600}, {"width": 1200, "height": 628}):
        design, plan, best, _ = resize_with_report(src, t)
        assert_valid(src, design, t)
        assert check_gates(src, design, t, plan, measurer) == []


# ---------------------------------------------------------------- detectors

def test_container_detectors_fire_on_a_stranded_card(monkeypatch):
    monkeypatch.setenv("CHAMELEON_LAYERS", "1")
    src = layered_ad()
    plan = heuristic_plan(src)
    moved = {"canvas": src["canvas"], "elements": [
        dict(e, y=e["y"] - 500) if e["id"] in ("headline", "body", "cta_bg", "cta_label") else e
        for e in src["elements"]]}
    det = layer_flaws.container_flaws(src, moved, plan)
    assert det["container_empty"]["fire"] and det["container_empty"]["ids"] == ["card"]
    half = {"canvas": src["canvas"], "elements": [dict(e, y=e["y"] - 500) if e["id"] == "headline" else e
                                                  for e in src["elements"]]}
    det = layer_flaws.container_flaws(src, half, plan)
    assert det["container_split"]["fire"] and not det["container_empty"]["fire"]
    assert not layer_flaws.container_flaws(src, src, plan)["container_empty"]["fire"]


def test_background_detector_separates_avoidable_shrinks():
    src = layered_ad()
    tall = {"width": 540, "height": 1080}
    els = []
    for e in src["elements"]:
        if e["id"] in ("bg", "photo"):
            e = dict(e, x=0, y=270, width=540, height=540)
        els.append(e)
    det = layer_flaws.background_flaws(src, {"canvas": tall, "elements": els})
    assert det["background_shrunk"]["fire"] and det["background_shrunk_avoidable"]["fire"]
    # a contain image of the wrong aspect can't fill the canvas: shrunk, but not avoidably
    contain = {"canvas": {"width": 1080, "height": 1080}, "elements": [
        {"id": "pat", "type": "image", "x": 0, "y": 0, "width": 1080, "height": 1080, "asset_id": "x.png",
         "fit": "contain"}]}
    out = {"canvas": tall, "elements": [dict(contain["elements"][0], y=270, width=540, height=540)]}
    det = layer_flaws.background_flaws(contain, out)
    assert det["background_shrunk"]["fire"] and not det["background_shrunk_avoidable"]["fire"]


# ---------------------------------------------------------------- the crello_dev split

def _write(directory, rec):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{rec['task_id']}.json").write_text(json.dumps(rec))


def test_crello_dev_split_is_separate(tmp_path):
    main, chl, chl2, dev = (tmp_path / n for n in ("designs", "challenge", "challenge2.nosync", "crello_dev.nosync"))
    _write(main, {"task_id": "a", "split": "train"})
    _write(chl, {"task_id": "z", "split": "challenge"})
    _write(chl2, {"task_id": "cr2_01", "split": "challenge2"})
    _write(dev, {"task_id": "crd_01", "split": "crello_dev"})
    (dev / "summary.json").write_text("{}")
    kw = dict(directory=main, challenge_dir=chl, challenge2_dir=chl2, crello_dev_dir=dev)
    assert [d["task_id"] for d in load_corpus(**kw)] == ["a"]
    assert [d["task_id"] for d in load_corpus(include_challenge=True, **kw)] == ["a", "z"]
    assert [d["task_id"] for d in load_corpus("challenge2", **kw)] == ["cr2_01"]
    assert [d["task_id"] for d in load_corpus("crello_dev", **kw)] == ["crd_01"]
    assert load_crello_dev(tmp_path / "missing") == []
    assert "crello_dev" in ab.EVAL_SPLITS
    assert layer_flaws.folders_for("crello_dev")[-1].name == "assets"
    assert layer_flaws.folders_for("train") == ()


def test_crello_dev_pool_excludes_challenge2_families():
    rows = [
        {"crello_id": "c2", "kept": True, "sig": ["f1", "f2"]},           # a challenge2 ad
        {"crello_id": "sib", "kept": True, "sig": ["f2", "f9"]},          # same family as c2
        {"crello_id": "a", "kept": True, "sig": ["f3"]},
        {"crello_id": "a2", "kept": True, "sig": ["f3", "f4"]},           # same family as a
        {"crello_id": "b", "kept": True, "sig": []},
        {"crello_id": "bad", "kept": False, "sig": ["f5"]},
    ]
    picked = {r["crello_id"] for r in mcd.pool(rows, {"c2"}, seed=0)}
    assert "c2" not in picked and "sib" not in picked and "bad" not in picked
    assert "b" in picked and len(picked & {"a", "a2"}) == 1


def test_crello_dev_select_keeps_some_simple_ads():
    rows, profiles = [], {}
    for k in range(40):
        cid = f"id{k}"
        rows.append({"crello_id": cid, "aspect_class": "square" if k < 30 else "wide", "photo": True,
                     "W": 1080, "H": 1080})
        profiles[cid] = {"groups": int(k % 3 != 0), "background_image": False}
    picked = mcd.select(rows, profiles, 12, seed=0)
    assert len(picked) == 12
    layered = sum(mcd.is_layered(profiles[r["crello_id"]]) for r in picked)
    assert 0 < layered < len(picked)  # both kinds; per class, LAYERED_SHARE of the quota is layered
    square = [r for r in picked if r["aspect_class"] == "square"][:2]
    assert sorted(mcd.is_layered(profiles[r["crello_id"]]) for r in square) == [False, True]


CRD = load_crello_dev()


@pytest.mark.skipif(not CRD, reason="crello_dev not built (python -m dev.distill.make_crello_dev)")
def test_crello_dev_ads_are_labelled_resolvable_and_outside_challenge2():
    from dev.distill.assets import asset_exists
    from dev.distill.corpus import load_challenge2

    assert 40 <= len(CRD) <= 80
    chl2 = {d["crello_id"] for d in load_challenge2()}
    for d in CRD:
        assert d["split"] == "crello_dev" and d["origin"] == "crello" and d["task_id"].startswith("crd_")
        assert d["crello_id"] not in chl2
        for el in d["source"]["elements"]:
            if el["type"] == "image":
                assert el["asset_id"].startswith("crd_") and asset_exists(el["asset_id"]), (d["task_id"], el["id"])
