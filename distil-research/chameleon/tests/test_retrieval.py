"""C retrieval: ad vector, library, recommend, holdout evaluate stages. Fake teachers only."""

import json
from pathlib import Path

import numpy as np
import pytest

from dev.distill import floorplans as fp
from dev.distill import retrieval as rt
from dev.distill.judge import RUBRIC_VERSION
from engine.pipeline.layout import extra_templates
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.types import PriorityPlan, Unit

ENV = "CHAMELEON_EXTRA_TEMPLATES"
METRIC_KEYS = ["hierarchy", "legibility", "overlap", "degradation", "faithfulness", "space_use", "alignment",
               "spacing", "balance", "text_contrast", "busyness", "placeholder"]
TARGETS = {t["name"]: t for t in json.loads(Path("dev/targets.json").read_text())}


def slot(name, box):
    return {"name": name, "box": box}


TALL_SLOTS = [slot("logo", [0, 0, 1, 0.07]), slot("hero", [0, 0.08, 1, 0.42]), slot("text", [0, 0.52, 1, 0.26]),
              slot("cta", [0, 0.8, 1, 0.1]), slot("minor", [0, 0.92, 1, 0.08])]
TALL_SLOTS_B = [slot("hero", [0, 0, 1, 0.5]), slot("text", [0, 0.52, 1, 0.3]), slot("minor", [0, 0.9, 1, 0.1])]
SQUARE_SLOTS = [slot("hero", [0, 0, 0.5, 1]), slot("text", [0.52, 0, 0.48, 0.9]), slot("minor", [0.52, 0.92, 0.48, 0.08])]


@pytest.fixture(autouse=True)
def _no_extra_file(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(extra_templates, "EXTRA_PATH", tmp_path / "absent_default.json")


def _metrics(v):
    return {k: v for k in METRIC_KEYS}


def _pool_row(task, target, cid, value, split="holdout"):
    return {"cell": f"{task}@{target}", "task_id": task, "split": split, "brand": "b", "target": target,
            "candidate_id": cid, "png": f"renders.nosync/{task}/{target}/{cid.replace('/', '_')}.png",
            "is_fallback": False, "metrics": _metrics(value)}


# ---------------------------------------------------------------- the vector

def _source():
    return {"canvas": {"width": 1000, "height": 500}, "elements": [
        {"id": "h", "type": "text", "content": "Fresh summer sale", "x": 0, "y": 0, "width": 400, "height": 100},
        {"id": "s", "type": "text", "content": "Now on", "x": 0, "y": 100, "width": 400, "height": 50},
        {"id": "i", "type": "image", "x": 500, "y": 0, "width": 400, "height": 200},
        {"id": "l", "type": "image", "x": 0, "y": 400, "width": 100, "height": 100},
        {"id": "b1", "type": "shape", "x": 0, "y": 300, "width": 100, "height": 40},
        {"id": "b2", "type": "text", "content": "Buy", "x": 10, "y": 305, "width": 80, "height": 30}]}


def _plan():
    return PriorityPlan("heuristic", [
        Unit("h", ["h"], "single", "headline", 1), Unit("s", ["s"], "single", "subheadline", 2),
        Unit("i", ["i"], "single", "hero_image", 3), Unit("l", ["l"], "single", "logo", 4),
        Unit("btn", ["b1", "b2"], "container_label", "cta", 5), Unit("bg", ["x"], "single", "background", None)])


def test_vector_documents_each_dimension_and_scales_into_unit_range():
    v = rt.ad_vector(_source(), _plan())
    assert v.shape == (len(rt.DIMENSIONS),) == (17,)
    d = dict(zip(rt.DIMENSIONS, v))
    assert d["count_headline"] == pytest.approx(1 / 3) and d["count_cta"] == pytest.approx(1 / 3)
    assert d["count_badge"] == 0 and d["count_legal"] == 0  # background units are not counted
    assert d["headline_chars"] == pytest.approx(len("Fresh summer sale") / 80)
    assert d["headline_words"] == pytest.approx(3 / 16)
    assert d["hero_aspect"] == pytest.approx(0.5)      # log2(2) / 2
    assert d["logo_aspect"] == 0                        # square logo
    assert d["source_canvas_aspect"] == pytest.approx(0.5)
    assert d["n_units"] == pytest.approx(5 / 12)
    assert d["has_group"] == 0 and d["has_button"] == 1
    assert np.all(np.abs(v) <= 1)


def test_vector_is_deterministic_and_saturates():
    a, b = rt.ad_vector(_source(), _plan()), rt.ad_vector(_source(), _plan())
    assert np.array_equal(a, b)
    src = _source()
    src["canvas"] = {"width": 100000, "height": 10}
    units = [Unit(f"u{i}", [f"u{i}"], "group", "badge", i + 1) for i in range(20)]
    v = dict(zip(rt.DIMENSIONS, rt.ad_vector(src, PriorityPlan("heuristic", units))))
    assert v["count_badge"] == 1 and v["n_units"] == 1 and v["source_canvas_aspect"] == 1 and v["has_group"] == 1


def test_vector_on_a_real_design(l1):
    v = rt.ad_vector(l1["source"], heuristic_plan(l1["source"], "brand_kit/assets"))
    assert v.shape == (17,) and np.all(np.abs(v) <= 1) and v[rt.DIMENSIONS.index("n_units")] > 0


# ---------------------------------------------------------------- library and recommend

def _lib_inputs(spec):
    """spec: [(task, cls, slots, [verdict confidences...])] -> proposals, builds, judgments."""
    proposals, builds, judgments = [], [], []
    for i, (task, cls, slots, picks) in enumerate(spec):
        target = {"tall": "instagram_portrait", "square": "instagram_square"}[cls]
        cell = f"{task}@{target}"
        pid = f"p{i:08d}"
        plan = {"name": pid, "classes": [cls], "margins": [0.04], "slots": slots}
        proposals.append({"proposal_id": pid, "cell": cell, "task_id": task, "split": "train", "target": target,
                          "aspect_class": cls, "plan": plan})
        builds.append({"proposal_id": pid, "cell": cell, "candidate_id": f"{pid}/margin=0.04", "built": True})
        for picked, conf in picks:
            judgments.append({"cell": cell, "a": "stack/a", "b": f"{pid}/margin=0.04", "winner": "b" if picked else "a",
                              "confidence": conf, "rubric_version": RUBRIC_VERSION, "swapped": False})
    return proposals, builds, judgments


VECS = {"near": np.zeros(17), "mid": np.full(17, 0.2), "far": np.full(17, 0.9), "sq": np.full(17, 0.05),
        "bad": np.full(17, 0.01)}


def _toy_library():
    spec = [("near", "tall", TALL_SLOTS, [(True, 3)]), ("mid", "tall", TALL_SLOTS_B, [(True, 2)]),
            ("far", "tall", TALL_SLOTS_B, [(True, 3)]), ("sq", "square", SQUARE_SLOTS, [(True, 3)]),
            ("bad", "tall", TALL_SLOTS, [(True, 3), (False, 3)])]  # a swapped re-ask disagrees: not a win
    return rt.library(*_lib_inputs(spec), vector_of=VECS.__getitem__)


def test_library_keeps_only_strict_winners_with_vector_and_class():
    lib = _toy_library()
    assert [e["task_id"] for e in lib] == ["near", "mid", "far", "sq"]
    assert all(e["vector"].shape == (17,) and e["aspect_class"] in ("tall", "square") for e in lib)
    assert {e["task_id"]: e["aspect_class"] for e in lib}["sq"] == "square"


def test_library_skips_holdout_and_other_rubric():
    props, builds, judgments = _lib_inputs([("near", "tall", TALL_SLOTS, [(True, 3)])])
    props[0]["split"] = "holdout"
    assert rt.library(props, builds, judgments, vector_of=VECS.__getitem__) == []
    props[0]["split"] = "train"
    judgments[0]["rubric_version"] = "old"
    assert rt.library(props, builds, judgments, vector_of=VECS.__getitem__) == []


def _query_vec_patch(monkeypatch, vec):
    monkeypatch.setattr(rt, "ad_vector", lambda source, plan: np.asarray(vec, dtype=float))


def test_nearest_neighbour_picks_the_closest_ad_within_the_class():
    lib = _toy_library()
    near = rt.nearest_ads(np.full(17, 0.02), "tall", lib, k=2)
    assert [t for t, _ in near] == ["near", "mid"]  # "bad" never won; "sq" is the wrong class
    assert rt.nearest_ads(np.full(17, 0.9), "tall", lib, k=1)[0][0] == "far"
    assert [t for t, _ in rt.nearest_ads(np.zeros(17), "square", lib, k=3)] == ["sq"]  # class filter
    assert rt.nearest_ads(np.zeros(17), "wide", lib) == []
    assert rt.nearest_ads(np.zeros(17), "tall", lib, k=1, exclude_task="near")[0][0] == "mid"


def test_recommend_returns_valid_unique_renamed_plans(monkeypatch):
    lib = _toy_library()
    lib.append(dict(lib[0], proposal_id="p-dup", task_id="mid"))  # 'mid' also owns TALL_SLOTS: a duplicate of near's plan
    _query_vec_patch(monkeypatch, np.zeros(17))
    plans = rt.recommend({"elements": []}, {"width": 1080, "height": 1350}, k=3, lib=lib, plan=object())
    assert len(plans) == 2  # near's and far/mid's TALL_SLOTS_B; the duplicate content is dropped
    names = [p["name"] for p in plans]
    assert len(set(names)) == len(names) and all(n.startswith("r") and len(n) == 9 for n in names)
    seen = {}
    for p in plans:
        extra_templates._parse_plan(p, seen)  # engine-loadable
        assert p["classes"] == ["tall"]
        seen[p["name"]] = p
    again = rt.recommend({"elements": []}, "tall", k=3, lib=lib, plan=object())
    assert [p["name"] for p in again] == names  # deterministic
    assert rt.recommend({"elements": []}, "tall", k=1, lib=lib, plan=object())[0]["slots"] == TALL_SLOTS


def test_recommend_does_not_mutate_the_library(monkeypatch):
    lib = _toy_library()
    before = json.dumps([e["plan"] for e in lib], sort_keys=True)
    _query_vec_patch(monkeypatch, np.zeros(17))
    rt.recommend({"elements": []}, "tall", k=3, lib=lib, plan=object())[0]["slots"][0]["box"][0] = 0.9
    assert json.dumps([e["plan"] for e in lib], sort_keys=True) == before


# ---------------------------------------------------------------- evaluate with fakes

class JudgeTeacher:
    """Whichever image is not a pool render (the retrieved build) wins."""
    name = "fake-judge"

    def __init__(self, confidence=3):
        self.confidence = confidence

    def ask(self, prompt, schema, images, validate=None):
        calls = [Path(p).read_bytes() for p in images]
        n = schema["properties"]["verdicts"]["maxItems"]
        return {"verdicts": [{"pair": i + 1, "winner": "b" if calls[1 + 2 * i].startswith(b"POOL") else "a",
                              "confidence": self.confidence, "score_a": 3, "score_b": 2, "reason": "r"}
                             for i in range(n)]}


def _eval_fixture(tmp_path, l1):
    pool_dir, fp_dir = tmp_path / "pool", tmp_path / "fp"
    pool_rows = [_pool_row("lvl1", "instagram_portrait", "stack/a", 0.5),
                 _pool_row("lvl1", "half_page", "stack/a", 0.5),          # very_tall
                 _pool_row("lvl1", "leaderboard", "stack/a", 0.5),        # strip: not evaluated
                 _pool_row("tr", "instagram_portrait", "stack/a", 0.5, split="train")]
    for r in pool_rows:
        (pool_dir / r["png"]).parent.mkdir(parents=True, exist_ok=True)
        (pool_dir / r["png"]).write_text("POOL")
    (pool_dir / "renders.nosync" / "lvl1" / "source.png").write_text("SOURCE")
    lib = [{"proposal_id": "pa", "task_id": "other", "cell": "other@story", "aspect_class": "tall",
            "vector": np.zeros(17), "plan": {"name": "pa", "classes": ["tall"], "margins": [0.04], "slots": TALL_SLOTS}},
           {"proposal_id": "pb", "task_id": "another", "cell": "another@story", "aspect_class": "tall",
            "vector": np.ones(17) * 0.5, "plan": {"name": "pb", "classes": ["tall"], "margins": [0.07], "slots": TALL_SLOTS_B}}]
    designs = {"lvl1": {"task_id": "lvl1", "split": "holdout", "source": l1["source"]}}
    return pool_dir, fp_dir, pool_rows, lib, designs


def test_evaluate_builds_best_recommended_plan_for_covered_classes_only(l1, tmp_path):
    pool_dir, fp_dir, pool_rows, lib, designs = _eval_fixture(tmp_path, l1)
    out = fp_dir / "retrieval-builds.jsonl"
    rows = rt.evaluate_build(lib, pool_rows, designs, TARGETS, Path("brand_kit/assets"), out, fp_dir, k=2)
    # tall cell has recommendations; very_tall has no library ad; strip is out of scope; train is not holdout
    assert sorted(r["cell"] for r in rows) == ["lvl1@half_page", "lvl1@instagram_portrait"]
    tall = next(r for r in rows if r["cell"] == "lvl1@instagram_portrait")
    assert tall["built"] and len(tall["plans"]) == 2 and tall["plan"] in {p["name"] for p in tall["plans"]}
    assert tall["candidate_id"].startswith(tall["plan"] + "/margin=") and (fp_dir / tall["png"]).exists()
    assert [n["task_id"] for n in tall["neighbors"]] == ["other", "another"]
    assert "engine_picks_new" in tall and tall["winner_id"] == "stack/a"
    totals = [p["total"] for p in tall["plans"] if p["built"]]
    assert tall["total"] == max(totals)  # best by the default grader
    vt = next(r for r in rows if r["cell"] == "lvl1@half_page")
    assert not vt["built"] and "no ad in the library" in vt["reason_if_not"]
    assert rt.evaluate_build(lib, pool_rows, designs, TARGETS, Path("brand_kit/assets"), out, fp_dir) == []  # resumes
    assert ENV not in __import__("os").environ


def test_evaluate_judge_and_summary_with_a_fake_teacher(l1, tmp_path):
    pool_dir, fp_dir, pool_rows, lib, designs = _eval_fixture(tmp_path, l1)
    out = fp_dir / "retrieval-builds.jsonl"
    builds = rt.evaluate_build(lib, pool_rows, designs, TARGETS, Path("brand_kit/assets"), out, fp_dir)
    pairs = rt.evaluate_pairs(builds, fp.cell_winners(pool_rows), pool_dir, fp_dir)
    assert len(pairs) == 1 and pairs[0][0]["candidate_id"] == "stack/a" and pairs[0][1]["cell"] == "lvl1@instagram_portrait"
    rows = fp.judge(pairs, JudgeTeacher(), pool_dir, fp_dir / "retrieval-judgments.jsonl", swap_share=0.0)
    assert len(rows) == 1
    s = rt.evaluate_summary(builds, rows)
    assert s["retrieval"]["all"]["cells"] == 2 and s["retrieval"]["all"]["built"] == 1
    assert s["retrieval"]["all"]["judged"] == 1 and s["retrieval"]["all"]["wins"] == 1
    assert s["retrieval"]["by_class"]["tall"]["win_rate"] == 1.0
    assert "clustered_same_cells" not in s


def test_summary_compares_with_clustered_templates_on_the_same_cells():
    ret = [{"cell": "a@story", "aspect_class": "tall", "built": True, "candidate_id": "r1/margin=0.04", "plan": "r1",
            "engine_picks_new": True},
           {"cell": "b@story", "aspect_class": "tall", "built": True, "candidate_id": "r2/margin=0.04", "plan": "r2",
            "engine_picks_new": True},
           {"cell": "c@story", "aspect_class": "tall", "built": False, "candidate_id": None, "plan": None,
            "engine_picks_new": False}]
    rj = [{"cell": "a@story", "a": "w", "b": "r1/margin=0.04", "winner": "b", "confidence": 3, "rubric_version": RUBRIC_VERSION},
          {"cell": "b@story", "a": "w", "b": "r2/margin=0.04", "winner": "a", "confidence": 3, "rubric_version": RUBRIC_VERSION}]
    eb = [{"cell": "a@story", "plan": "b2_tall_1", "candidate_id": "b2_tall_1/m", "total": 0.9, "built": True, "engine_picks_new": True},
          {"cell": "a@story", "plan": "b2_tall_2", "candidate_id": "b2_tall_2/m", "total": 0.8, "built": True, "engine_picks_new": True},
          {"cell": "b@story", "plan": "b2_tall_1", "candidate_id": "b2_tall_1/m", "total": 0.9, "built": True, "engine_picks_new": True},
          {"cell": "zzz@story", "plan": "b2_tall_1", "candidate_id": "b2_tall_1/m", "total": 0.9, "built": True, "engine_picks_new": True}]
    ej = [{"cell": "a@story", "a": "w", "b": "b2_tall_1/m", "winner": "a", "confidence": 3, "rubric_version": RUBRIC_VERSION},
          {"cell": "a@story", "a": "w", "b": "b2_tall_2/m", "winner": "b", "confidence": 3, "rubric_version": RUBRIC_VERSION},
          {"cell": "b@story", "a": "w", "b": "b2_tall_1/m", "winner": "b", "confidence": 3, "rubric_version": RUBRIC_VERSION},
          {"cell": "zzz@story", "a": "w", "b": "b2_tall_1/m", "winner": "b", "confidence": 3, "rubric_version": RUBRIC_VERSION}]
    s = rt.evaluate_summary(ret, rj, eb, ej)
    a = s["retrieval"]["all"]
    assert (a["picks_changed"], a["picks_changed_wins"], a["picks_changed_loses"]) == (2, 1, 1)
    c = s["clustered_same_cells"]  # zzz is not a retrieval cell, so it is ignored
    assert c["plan_level"] == {"plan_cell_pairs_judged": 3, "wins": 2, "win_rate": pytest.approx(2 / 3)}
    # final pick: cell a -> b2_tall_1 (higher total), which loses; cell b -> b2_tall_1, which wins
    assert c["final_pick_replay"] == {"cells": 2, "picks_changed": 2, "wins": 1, "loses": 1}
    assert "retrieval" in rt.render_report(s) and "Comparison" in rt.render_report(s)


def test_cli_merge_and_shard_files(tmp_path, monkeypatch):
    main = tmp_path / "retrieval-builds.jsonl"
    monkeypatch.setattr(rt, "RET_BUILDS_FILE", main)
    fp.shard_file(main, 0, 2).write_text(json.dumps({"cell": "a@story"}) + "\n")
    assert rt.main(["evaluate", "--stage", "build", "--merge"]) == 0 and len(main.read_text().splitlines()) == 1
