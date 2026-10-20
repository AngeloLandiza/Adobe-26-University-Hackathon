"""Per-ad floor-plan retrieval inside the engine (engine/pipeline/layout/retrieval.py),
switched on with CHAMELEON_RETRIEVAL=1, and its exporter (dev/distill/export_retrieval_library.py).

generate.py is not wired yet (another change is editing it). `_patched_generate()` applies
the exact patch the report proposes (PATCH_IMPORT / PATCH_BLOCK) to generate.py's current
source in memory, so the end to end tests run the code the engine will run. Once the patch
is in generate.py, it returns the real module instead."""

import json
import re
import types
from pathlib import Path

import numpy as np
import pytest

from dev.distill import retrieval as dev_rt
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout import extra_templates, generate
from engine.pipeline.layout import retrieval as rt
from engine.pipeline.layout.extra_templates import FloorPlan, _parse_plan
from engine.pipeline.layout.templates import build_slots as template_build_slots
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.types import PriorityPlan, Unit

TARGETS = {t["name"]: t for t in json.loads(Path("dev/targets.json").read_text())}
SAMPLES = sorted(Path("input").glob("chameleon_sample_*.json"))
E2E_TARGETS = ("story", "instagram_post", "medium_rectangle")


def slot(name, box):
    return {"name": name, "box": box}


TALL_A = {"margins": [0.05, 0.07], "slots": [
    slot("logo", [0, 0, 1, 0.07]), slot("hero", [0, 0.08, 1, 0.42]), slot("text", [0, 0.52, 1, 0.26]),
    slot("cta", [0, 0.8, 1, 0.1]), slot("minor", [0, 0.92, 1, 0.08])]}
TALL_B = {"margins": [0.06], "slots": [
    slot("hero", [0, 0, 1, 0.5]), slot("text", [0, 0.52, 1, 0.3]), slot("minor", [0, 0.9, 1, 0.1])]}
SQUARE_A = {"margins": [0.05], "slots": [
    slot("hero", [0, 0, 0.5, 1]), slot("text", [0.52, 0, 0.48, 0.9]), slot("minor", [0.52, 0.92, 0.48, 0.08])]}


def _vec(x):
    v = [0.0] * len(rt.DIMENSIONS)
    v[0] = x
    return v


def _entry(pid, task, cls, x, plan):
    return {"proposal_id": pid, "task_id": task, "aspect_class": cls, "vector": _vec(x), "plan": plan}


def _library(entries):
    return {"version": rt.LIBRARY_VERSION, "dimensions": list(rt.DIMENSIONS), "entries": entries}


SMALL = _library([
    _entry("p1", "far", "tall", 1.0, TALL_A),
    _entry("p2", "near", "tall", 0.1, TALL_B),
    _entry("p3", "near", "tall", 0.1, TALL_A),
    _entry("p4", "mid", "tall", 0.5, TALL_A),      # same content as p3: dropped as a duplicate
    _entry("p5", "near", "square", 0.1, SQUARE_A),
])


def _source():
    return {"canvas": {"width": 1000, "height": 1000}, "elements": [
        {"id": "h", "type": "text", "content": "Fresh summer sale", "x": 0, "y": 0, "width": 400, "height": 100},
        {"id": "i", "type": "image", "x": 500, "y": 0, "width": 400, "height": 200}]}


def _plan(headlines=0):
    """count_headline (dimension 0) = min(headlines, 3) / 3; everything else is fixed."""
    units = [Unit(f"h{i}", ["h"], "single", "headline", i + 1) for i in range(headlines)]
    return PriorityPlan("heuristic", units)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No shipped extra templates or library leak in unless a test asks; retrieval off by default."""
    monkeypatch.delenv(rt.ENV, raising=False)
    monkeypatch.delenv(rt.LIBRARY_ENV, raising=False)
    monkeypatch.delenv("CHAMELEON_FLEX", raising=False)
    monkeypatch.delenv(extra_templates.ENV_VAR, raising=False)
    monkeypatch.setattr(extra_templates, "EXTRA_PATH", tmp_path / "absent_extra.json")


def use_library(monkeypatch, tmp_path, data, name="lib.json"):
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    monkeypatch.setattr(rt, "LIBRARY_PATH", path)
    return path


PATCH_IMPORT = ("from engine.pipeline.layout import flex\n",
                "from engine.pipeline.layout import flex, retrieval\n")
PATCH_BLOCK = """\
    if retrieval.enabled():  # CHAMELEON_RETRIEVAL=1: floor plans of the nearest training ads
        try:
            candidates += retrieval.candidates(source, target, plan, measurer)
        except Exception:
            pass  # retrieval never sinks the task
"""
_patched = []


def _patched_generate():
    """generate.py with the proposed patch: the import, and PATCH_BLOCK just before
    generate_candidates' final `return candidates` (after the template loop, so the
    same-aspect early return still returns the reflow alone)."""
    if not _patched:
        src = Path(generate.__file__).read_text()
        if "retrieval.candidates(" in src:
            _patched.append(generate)
        else:
            assert src.count(PATCH_IMPORT[0]) == 1 and src.endswith("    return candidates\n"), \
                "generate.py changed shape: update the patch in the report"
            head, _ = src.replace(*PATCH_IMPORT).rsplit("    return candidates\n", 1)
            mod = types.ModuleType("engine.pipeline.layout._generate_patched")
            exec(compile(head + PATCH_BLOCK + "    return candidates\n", "generate_patched.py", "exec"), mod.__dict__)
            _patched.append(mod)
    return _patched[0]


def _ids(cands):
    return [(c.candidate_id, c.design) for c in cands]


# ---------------------------------------------------------------- off

def test_off_returns_nothing_and_reads_nothing(monkeypatch, tmp_path, capsys, l2, measurer):
    use_library(monkeypatch, tmp_path, "{ not json")  # would warn if it were read
    assert not rt.enabled()
    assert rt.extra_plans_for(_source(), {"width": 1080, "height": 1920}, _plan()) == []
    plan = heuristic_plan(l2["source"])
    assert rt.candidates(l2["source"], TARGETS["story"], plan, measurer) == []
    assert capsys.readouterr().err == ""


def test_off_generate_is_unchanged(l2, measurer):
    plan = heuristic_plan(l2["source"])
    target = TARGETS["story"]
    with_patch = _patched_generate().generate_candidates(l2["source"], target, plan, measurer)
    without = generate.generate_candidates(l2["source"], target, plan, measurer)
    assert with_patch and _ids(with_patch) == _ids(without)


def test_only_exactly_1_switches_on(monkeypatch):
    for value in ("0", "", "true", "yes"):
        monkeypatch.setenv(rt.ENV, value)
        assert not rt.enabled()
    monkeypatch.setenv(rt.ENV, "1")
    assert rt.enabled()


# ---------------------------------------------------------------- on: the nearest ads' plans

def test_on_gives_nearest_ads_plans_for_the_target_class(monkeypatch, tmp_path):
    monkeypatch.setenv(rt.ENV, "1")
    use_library(monkeypatch, tmp_path, SMALL)
    tall = rt.extra_plans_for(_source(), {"width": 1080, "height": 1920}, _plan(0), k=2)
    # nearest tall ads to vector 0: near (0.1), mid (0.5); near's plans by proposal_id: p2, p3; mid's p4 == p3
    assert [p.slots for p in tall] == [_parse_plan(dict(TALL_B, name="x", classes=["tall"]), {}).slots,
                                       _parse_plan(dict(TALL_A, name="x", classes=["tall"]), {}).slots]
    assert all(isinstance(p, FloorPlan) and p.classes == ("tall",) for p in tall)
    square = rt.extra_plans_for(_source(), {"width": 1080, "height": 1080}, _plan(0), k=3)
    assert len(square) == 1 and square[0].classes == ("square",)
    assert rt.extra_plans_for(_source(), {"width": 1280, "height": 720}, _plan(0)) == []  # no landscape plans


def test_on_k1_and_far_ad_selection(monkeypatch, tmp_path):
    monkeypatch.setenv(rt.ENV, "1")
    use_library(monkeypatch, tmp_path, SMALL)
    # count_headline = 1.0 (three headlines): the nearest tall ad is now "far" (distance 0)
    plans = rt.extra_plans_for(_source(), {"width": 1080, "height": 1920}, _plan(3), k=1)
    assert [p.margins for p in plans] == [(0.05, 0.07)]


def test_plans_are_valid_named_by_content_and_deterministic(monkeypatch, tmp_path):
    monkeypatch.setenv(rt.ENV, "1")
    use_library(monkeypatch, tmp_path, SMALL)
    target = {"width": 1080, "height": 1920}
    first = rt.extra_plans_for(_source(), target, _plan(), k=3)
    again = rt.extra_plans_for(_source(), target, _plan(), k=3)
    assert first == again and first
    names = [p.name for p in first]
    assert len(set(names)) == len(names) and all(re.fullmatch(r"r[0-9a-f]{8}", n) for n in names)
    raws = rt.recommend(_source(), target, _plan(), k=3)
    assert len(raws) == len(first)
    for p, raw in zip(first, raws):
        assert _parse_plan(raw, {}) == p
        assert p.name == raw["name"] == dev_rt.recommended_name(raw)


def test_recommend_matches_dev_recommend_on_the_same_library(monkeypatch, tmp_path):
    """Same library, same vector: the engine's choices equal dev/distill/retrieval.recommend's."""
    lib = rt.parse_library(json.dumps(SMALL))
    dev_lib = [dict(e, vector=np.asarray(e["vector"])) for e in lib]
    for headlines in range(4):
        for target in ({"width": 1080, "height": 1920}, {"width": 1080, "height": 1080}):
            for k in (1, 2, 3):
                ours = rt.recommend(_source(), target, _plan(headlines), k=k, lib=lib)
                theirs = dev_rt.recommend(_source(), target, k=k, lib=dev_lib, plan=_plan(headlines))
                assert ours == theirs


def test_ad_vector_matches_dev(l1, l2, l3):
    for task in (l1, l2, l3):
        plan = heuristic_plan(task["source"])
        assert np.array_equal(rt.ad_vector(task["source"], plan), dev_rt.ad_vector(task["source"], plan))
    assert rt.DIMENSIONS == dev_rt.DIMENSIONS


# ---------------------------------------------------------------- a bad library never crashes

@pytest.mark.parametrize("content", ["{ not json", "[]", json.dumps({"entries": []}),
                                     json.dumps(dict(SMALL, dimensions=["a"])),
                                     json.dumps(dict(SMALL, version=999)), "\x00\xff"])
def test_bad_library_means_no_retrieval_and_one_warning(monkeypatch, tmp_path, capsys, content):
    monkeypatch.setenv(rt.ENV, "1")
    use_library(monkeypatch, tmp_path, content)
    target = {"width": 1080, "height": 1920}
    assert rt.extra_plans_for(_source(), target, _plan()) == []
    assert rt.extra_plans_for(_source(), target, _plan()) == []
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("[distill]"), err


def test_missing_library_warns_once_only_when_on(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rt, "LIBRARY_PATH", tmp_path / "gone.json")
    assert rt.extra_plans_for(_source(), {"width": 1080, "height": 1920}, _plan()) == []
    assert capsys.readouterr().err == ""
    monkeypatch.setenv(rt.ENV, "1")
    for _ in range(3):
        assert rt.extra_plans_for(_source(), {"width": 1080, "height": 1920}, _plan()) == []
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("[distill]"), err


def test_bad_entries_are_skipped_good_ones_kept(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(rt.ENV, "1")
    bad = [_entry("b1", "x", "tall", 0.0, TALL_A) | {"vector": [0.0] * 3},
           _entry("b2", "x", "nope", 0.0, TALL_A),
           _entry("b3", "x", "tall", 0.0, TALL_A) | {"vector": [float("nan")] * len(rt.DIMENSIONS)},
           _entry("b4", "x", "tall", 0.0, {"margins": [0.05], "slots": [slot("hero", [0, 0, 2, 2])]}),
           "not an entry"]
    use_library(monkeypatch, tmp_path, _library(bad + [_entry("ok", "y", "tall", 0.0, TALL_B)]))
    plans = rt.extra_plans_for(_source(), {"width": 1080, "height": 1920}, _plan(), k=5)
    assert [p.margins for p in plans] == [(0.06,)]
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 5 and all(line.startswith("[distill]") for line in err), err


def test_unexpected_error_inside_never_raises(monkeypatch, tmp_path, capsys, l2, measurer):
    monkeypatch.setenv(rt.ENV, "1")
    use_library(monkeypatch, tmp_path, SMALL)

    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(rt, "ad_vector", boom)
    assert rt.extra_plans_for(_source(), {"width": 1080, "height": 1920}, _plan()) == []
    assert rt.candidates(l2["source"], TARGETS["story"], heuristic_plan(l2["source"]), measurer) == []
    assert "[distill]" in capsys.readouterr().err


def test_explicit_library_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv(rt.ENV, "1")
    monkeypatch.setattr(rt, "LIBRARY_PATH", tmp_path / "gone.json")
    path = tmp_path / "other.json"
    path.write_text(json.dumps(SMALL))
    monkeypatch.setenv(rt.LIBRARY_ENV, str(path))
    assert rt.extra_plans_for(_source(), {"width": 1080, "height": 1920}, _plan(), k=1)


# ---------------------------------------------------------------- slots and candidates

def test_slots_match_templates_build_slots_for_the_same_plan(monkeypatch, tmp_path):
    raw = dict(TALL_A, name="rtest1234", classes=["tall"])
    (tmp_path / "x.json").write_text(json.dumps({"floor_plans": [raw]}))
    monkeypatch.setenv(extra_templates.ENV_VAR, str(tmp_path / "x.json"))
    fplan = _parse_plan(raw, {})
    for margin in fplan.margins:
        params = {"margin": margin}
        assert rt.build_slots(fplan, 1080, 1920, params) == template_build_slots("rtest1234", 1080, 1920, params)
    assert rt.param_grid(fplan) == [{"margin": 0.05}, {"margin": 0.07}]


@pytest.mark.parametrize("flex", ["0", "1"])
def test_candidate_equals_the_same_plan_loaded_as_an_extra_template(monkeypatch, tmp_path, l2, measurer, flex):
    """Guards the mirror of generate._template_candidate: the same plan built through
    retrieval and through the extra-templates file gives the identical candidate."""
    monkeypatch.setenv("CHAMELEON_FLEX", flex)
    raw = dict(TALL_A, name="rtest1234", classes=["tall"])
    (tmp_path / "x.json").write_text(json.dumps({"floor_plans": [raw]}))
    monkeypatch.setenv(extra_templates.ENV_VAR, str(tmp_path / "x.json"))
    fplan = _parse_plan(raw, {})
    plan = heuristic_plan(l2["source"])
    target = {"width": 1080, "height": 1920}
    for params in rt.param_grid(fplan):
        ours = rt.build_candidate(fplan, params, l2["source"], target, plan, measurer)
        theirs = generate._template_candidate("rtest1234", params, l2["source"], target, plan, measurer)
        assert (ours is None) == (theirs is None)
        if ours is not None:
            assert (ours.candidate_id, ours.design, ours.template, ours.params, ours.degraded_unit_ids) == \
                   (theirs.candidate_id, theirs.design, theirs.template, theirs.params, theirs.degraded_unit_ids)


def _shipped_library_or_skip():
    if not rt.LIBRARY_PATH.exists():
        pytest.skip("no shipped retrieval_library.json")


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target_name", E2E_TARGETS)
def test_end_to_end_recommended_candidates_pass_gates(monkeypatch, measurer, sample, target_name):
    _shipped_library_or_skip()
    monkeypatch.setenv(rt.ENV, "1")
    source = json.loads(sample.read_text())["source"]
    target = {"width": TARGETS[target_name]["width"], "height": TARGETS[target_name]["height"]}
    plan = heuristic_plan(source)
    fplans = rt.extra_plans_for(source, target, plan)
    assert fplans, "the shipped library has plans for tall and square targets"
    ours = rt.candidates(source, target, plan, measurer)
    names = {p.name for p in fplans}
    assert ours and all(c.template in names for c in ours)
    assert len({c.candidate_id for c in ours}) == len(ours)
    for c in ours:
        assert check_gates(source, c.design, target, plan, measurer) == [], c.candidate_id
    # through the patched generate: today's candidates, then exactly these (none after a same-aspect early return)
    patched = _patched_generate().generate_candidates(source, target, plan, measurer)
    monkeypatch.delenv(rt.ENV)
    base = _patched_generate().generate_candidates(source, target, plan, measurer)
    assert _ids(base) == _ids(generate.generate_candidates(source, target, plan, measurer))
    src = source["canvas"]["width"] / source["canvas"]["height"]
    if abs(target["width"] / target["height"] / src - 1) <= generate.SAME_ASPECT_TOL and base[0].template == "reflow":
        assert _ids(patched) == _ids(base)
    else:
        assert _ids(patched) == _ids(base) + _ids(ours)
        assert not ({c.candidate_id for c in base} & {c.candidate_id for c in ours})


def test_shipped_library_is_train_only_and_parses(capsys):
    _shipped_library_or_skip()
    raw = json.loads(rt.LIBRARY_PATH.read_text())
    assert raw["split"] == "train"
    lib = rt.parse_library(rt.LIBRARY_PATH.read_text())
    assert lib and len(lib) == len(raw["entries"])
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------- exporter round trip (needs training data)

def test_exporter_round_trip_matches_dev_recommend_on_holdout():
    from dev.distill import export_retrieval_library as ex

    if not ex.HOLDOUT_BUILDS.exists() or not dev_rt.fp.PROPOSALS_FILE.exists():
        pytest.skip("no floor-plan training data")
    _shipped_library_or_skip()
    result = ex.verify(rt.LIBRARY_PATH)
    assert result["mismatches"] == [] and result["cells"] == 85
    assert result["holdout_tasks_in_library"] == []
