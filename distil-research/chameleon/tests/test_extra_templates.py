import json
import os
from pathlib import Path

import pytest

from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout.aspect import ASPECT_CLASSES
from engine.pipeline.layout import extra_templates
from engine.pipeline.layout.extra_templates import load_floor_plans
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.layout.templates import TEMPLATES_BY_CLASS, build_slots, param_grid, templates_for
from engine.pipeline.priority import heuristic_plan

ENV = "CHAMELEON_EXTRA_TEMPLATES"


def slot(name, box):
    return {"name": name, "box": box}


SQUARE_SPLIT = {"name": "square_split", "classes": ["square"], "margins": [0.04, 0.07], "slots": [
    slot("logo", [0, 0, 0.45, 0.1]), slot("text", [0, 0.12, 0.45, 0.5]), slot("cta", [0, 0.64, 0.45, 0.16]),
    slot("minor", [0, 0.82, 0.45, 0.18]), slot("hero", [0.5, 0, 0.5, 1])]}
TALL_SPLIT = {"name": "tall_split", "classes": ["tall"], "slots": [
    slot("logo", [0, 0, 1, 0.07]), slot("hero", [0, 0.08, 1, 0.42]), slot("text", [0, 0.52, 1, 0.26]),
    slot("cta", [0, 0.8, 1, 0.1]), slot("minor", [0, 0.92, 1, 0.08])]}


@pytest.fixture(autouse=True)
def _no_extra_file(tmp_path, monkeypatch):
    """An extra_templates.json may ship later; these tests must not depend on it. The default
    path is pointed at an absent file (silent) and the env var is cleared."""
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(extra_templates, "EXTRA_PATH", tmp_path / "absent_default.json")


def use(monkeypatch, tmp_path, plans, name="plans.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"floor_plans": plans}))
    monkeypatch.setenv(ENV, str(path))


def test_no_file_no_change():
    for c in ASPECT_CLASSES:
        assert templates_for(c) == TEMPLATES_BY_CLASS[c]


def test_missing_default_file_is_silent(capsys):
    assert load_floor_plans() == []
    assert capsys.readouterr().err == ""


def test_missing_env_file_warns_once(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(ENV, str(tmp_path / "typo.json"))
    assert load_floor_plans() == [] and load_floor_plans() == []
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("[distill]")


def _one_line_and_empty(path, capsys):
    assert load_floor_plans(path) == [] and load_floor_plans(path) == []
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("[distill]")


def test_huge_number_in_file_falls_back(tmp_path, capsys):
    path = tmp_path / "p.json"
    path.write_text('{"floor_plans": [{"name": "p", "classes": ["square"], "margins": [1' + "0" * 400 + '], "slots": []}]}')
    # a bad plan inside a readable file is skipped with its own line; no exception escapes
    assert load_floor_plans(path) == []
    assert capsys.readouterr().err.count("[distill]") == 1


def test_deeply_nested_file_falls_back(tmp_path, capsys):
    path = tmp_path / "p.json"
    path.write_text("[" * 200_000)
    _one_line_and_empty(path, capsys)


def test_path_that_is_a_directory_falls_back(tmp_path, capsys):
    _one_line_and_empty(tmp_path, capsys)


def test_symlink_loop_falls_back(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.symlink_to(b); b.symlink_to(a)
    _one_line_and_empty(a, capsys)


def test_same_size_rewrite_with_same_mtime_is_reread(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"floor_plans": [dict(TALL_SPLIT, name="aaa")]}))
    os.utime(path, ns=(1, 5))
    assert [p.name for p in load_floor_plans(path)] == ["aaa"]
    path.write_text(json.dumps({"floor_plans": [dict(TALL_SPLIT, name="bbbb")]}))
    os.utime(path, ns=(1, 5))
    assert [p.name for p in load_floor_plans(path)] == ["bbbb"]


def test_unknown_template_is_harmless(monkeypatch, tmp_path):
    use(monkeypatch, tmp_path, [SQUARE_SPLIT])
    assert param_grid("no_such_plan") == []
    with pytest.raises(ValueError):
        build_slots("no_such_plan", 1000, 1000, {"margin": 0.04})


def test_unreadable_file_warns_once(monkeypatch, tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    monkeypatch.setenv(ENV, str(path))
    assert load_floor_plans() == [] and load_floor_plans() == []
    err = capsys.readouterr().err
    assert err.count("[distill]") == 1


def test_plan_offered_and_scaled(monkeypatch, tmp_path):
    plan = {"name": "square_split", "classes": ["square"], "slots": [
        slot("text", [0, 0, 0.5, 0.85]), slot("minor", [0, 0.85, 0.5, 0.15]), slot("hero", [0.5, 0, 0.5, 1])]}
    use(monkeypatch, tmp_path, [plan])
    assert templates_for("square") == ["stack", "square_split"]
    assert templates_for("tall") == ["stack"]
    assert param_grid("square_split") == [{"margin": 0.04}, {"margin": 0.07}]
    hero = next(s for s in build_slots("square_split", 1000, 1000, {"margin": 0.04}) if s.name == "hero")
    assert hero.box == pytest.approx((500.0, 40.0, 460.0, 920.0))


def _bad(**change):
    plan = json.loads(json.dumps(SQUARE_SPLIT))
    plan.update(change)
    return plan


# (plan, a distinctive part of the warning): each case is rejected by its own rule only.
BAD_PLANS = {
    "builtin_name": (_bad(name="stack"), "already taken"),
    "bad_name_chars": (_bad(name="a/b=c"), "name must match"),
    "bad_class": (_bad(classes=["round"]), "classes must be"),
    "bad_slot_name": (_bad(slots=[*SQUARE_SPLIT["slots"], slot("sticker", [0.455, 0.2, 0.03, 0.1])]),
                      "slot names must come from"),
    "duplicate_slot": (_bad(slots=[*SQUARE_SPLIT["slots"], slot("cta", [0.455, 0.2, 0.03, 0.1])]),
                       "appears twice"),
    "no_minor": (_bad(slots=[s for s in SQUARE_SPLIT["slots"] if s["name"] != "minor"]), "both 'text' and 'minor'"),
    "out_of_bounds": (_bad(slots=[slot("text", [0, 0, 0.5, 0.5]), slot("minor", [0.6, 0.6, 0.5, 0.5])]),
                      "unit square"),
    "overlap": (_bad(slots=[slot("text", [0, 0, 0.5, 0.5]), slot("minor", [0.25, 0.25, 0.5, 0.5])]), "overlap"),
    "repeated_margins": (_bad(margins=[0.04, 0.04]), "must not repeat"),
    "margin_out_of_range": (_bad(margins=[0.04, 0.5]), "margins must be"),
    "negative_margin": (_bad(margins=[-0.1]), "margins must be"),
}


@pytest.mark.parametrize("bad,expected", BAD_PLANS.values(), ids=BAD_PLANS.keys())
def test_invalid_plans_skipped(monkeypatch, tmp_path, capsys, bad, expected):
    good = dict(TALL_SPLIT, name="good_plan")
    use(monkeypatch, tmp_path, [bad, good])
    assert [p.name for p in load_floor_plans()] == ["good_plan"]
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("[distill]") and expected in err[0]


def test_duplicate_name_skipped(monkeypatch, tmp_path):
    use(monkeypatch, tmp_path, [SQUARE_SPLIT, dict(TALL_SPLIT, name="square_split")])
    assert [p.classes for p in load_floor_plans()] == [("square",)]


CELLS = [("chameleon_sample_level1_warmup", 300, 250), ("chameleon_sample_level2_core", 300, 250),
         ("chameleon_sample_level3_advanced", 300, 250), ("chameleon_sample_level1_warmup", 1080, 1350),
         ("chameleon_sample_level2_core", 1080, 1350)]


@pytest.mark.parametrize("stem,w,h", CELLS)
def test_floor_plan_candidates_pass_gates(monkeypatch, tmp_path, measurer, stem, w, h):
    use(monkeypatch, tmp_path, [SQUARE_SPLIT, TALL_SPLIT])
    src = json.loads(Path(f"input/{stem}.json").read_text())["source"]
    target = {"width": w, "height": h}
    plan = heuristic_plan(src)
    name = "square_split" if w == 300 else "tall_split"
    mine = [c for c in generate_candidates(src, target, plan, measurer) if c.candidate_id.startswith(name + "/")]
    assert mine
    for c in mine:
        assert check_gates(src, c.design, target, plan, measurer) == [], c.candidate_id
