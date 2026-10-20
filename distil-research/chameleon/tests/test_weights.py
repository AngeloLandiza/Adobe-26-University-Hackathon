import json
import os

import pytest

from engine.pipeline.grading import weights as weights_module
from engine.pipeline.grading.metrics import WEIGHTS
from engine.pipeline.grading.pixels import PIXEL_WEIGHTS
from engine.pipeline.grading.score import pick_best
from engine.pipeline.grading.weights import DEFAULT_WEIGHTS, GraderWeights, load_weights
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.priority import heuristic_plan

GEO_KEYS = list(WEIGHTS)
PIX_KEYS = list(PIXEL_WEIGHTS)


def doc(**over):
    base = {"geometry": dict(WEIGHTS), "pixels": dict(PIXEL_WEIGHTS), "geometry_share": 0.7, "source": "hand"}
    base.update(over)
    return base


@pytest.fixture(autouse=True)
def _no_weights_file(tmp_path, monkeypatch):
    """A weights.json may ship later; these tests must not depend on it. The default path is
    pointed at an absent file (silent), and the env var is cleared (an env path that is missing warns)."""
    monkeypatch.delenv("CHAMELEON_GRADER_WEIGHTS", raising=False)
    monkeypatch.setattr(weights_module, "WEIGHTS_PATH", tmp_path / "absent_default.json")


def test_defaults_match_constants():
    assert DEFAULT_WEIGHTS == GraderWeights(dict(WEIGHTS), dict(PIXEL_WEIGHTS), 0.7)


def test_missing_default_file_is_silent(capsys):
    assert load_weights() == DEFAULT_WEIGHTS
    assert capsys.readouterr().err == ""


def test_missing_given_file_gives_defaults_with_one_line(tmp_path, capsys):
    assert load_weights(tmp_path / "nope.json") == DEFAULT_WEIGHTS
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("[distill]")


def test_missing_env_file_warns_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHAMELEON_GRADER_WEIGHTS", str(tmp_path / "typo.json"))
    assert load_weights() == DEFAULT_WEIGHTS and load_weights() == DEFAULT_WEIGHTS
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("[distill]")


def test_valid_file_overrides(tmp_path):
    geo = {k: 0.0 for k in GEO_KEYS}
    geo["balance"] = 1.0
    path = tmp_path / "w.json"
    path.write_text(json.dumps(doc(geometry=geo, geometry_share=0.5)))
    got = load_weights(path)
    assert got.geometry == geo and got.pixels == PIXEL_WEIGHTS and got.geometry_share == 0.5


def test_env_var_selects_file(tmp_path, monkeypatch):
    path = tmp_path / "env.json"
    path.write_text(json.dumps(doc(geometry_share=0.2)))
    monkeypatch.setenv("CHAMELEON_GRADER_WEIGHTS", str(path))
    assert load_weights().geometry_share == 0.2


def test_rewritten_file_is_reread(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(doc(geometry_share=0.2)))
    assert load_weights(path).geometry_share == 0.2
    path.write_text(json.dumps(doc(geometry_share=0.4)))
    os.utime(path, ns=(1, 2_000_000_000))
    assert load_weights(path).geometry_share == 0.4


def _bad_docs():
    missing = doc(); del missing["geometry"]["balance"]
    negative = doc(); negative["geometry"]["balance"] = -0.05; negative["geometry"]["hierarchy"] += 0.10
    nan_in_group = doc(); nan_in_group["geometry"]["balance"] = float("nan")
    huge = doc(); huge["geometry"]["balance"] = 10 ** 400
    return {
        "nan_in_group": json.dumps(nan_in_group),
        "huge_int": json.dumps(huge),
        "deeply_nested": "[" * 200_000,
        "invalid_json": "{not json",
        "missing_key": json.dumps(missing),
        "negative": json.dumps(negative),
        "nan": json.dumps(doc(geometry_share=float("nan"))),
        "share_too_big": json.dumps(doc(geometry_share=1.5)),
        "sum_0.9": json.dumps(doc(pixels={"text_contrast": 0.5, "busyness": 0.25, "placeholder": 0.15})),
    }


@pytest.mark.parametrize("name", list(_bad_docs()))
def test_invalid_files_fall_back(tmp_path, capsys, name):
    path = tmp_path / "w.json"
    path.write_text(_bad_docs()[name])
    assert load_weights(path) == DEFAULT_WEIGHTS
    load_weights(path)  # cached: no second warning
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("[distill]")


def test_path_that_is_a_directory_falls_back(tmp_path, capsys):
    assert load_weights(tmp_path) == DEFAULT_WEIGHTS
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("[distill]")


def test_symlink_loop_falls_back(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.symlink_to(b); b.symlink_to(a)
    assert load_weights(a) == DEFAULT_WEIGHTS and load_weights(a) == DEFAULT_WEIGHTS
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("[distill]")


def test_same_size_rewrite_with_same_mtime_is_reread(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(doc(geometry_share=0.2)))
    os.utime(path, ns=(1, 5))
    assert load_weights(path).geometry_share == 0.2
    path.write_text(json.dumps(doc(geometry_share=0.25)))  # longer text, same mtime
    os.utime(path, ns=(1, 5))
    assert load_weights(path).geometry_share == 0.25


def _inputs(request, measurer, fixture_name):
    sample = request.getfixturevalue(fixture_name)
    src, tgt = sample["source"], sample["target_canvas"]
    plan = heuristic_plan(src)
    cands = [safe_fallback(src, tgt, plan)] + generate_candidates(src, tgt, plan, measurer)
    return src, tgt, plan, cands


@pytest.mark.parametrize("fixture_name", ["l1", "l2", "l3"])
def test_pick_best_unchanged_without_file(request, measurer, fixture_name):
    src, tgt, plan, cands = _inputs(request, measurer, fixture_name)
    a, ra = pick_best(src, tgt, plan, cands, measurer, weights=None)
    b, rb = pick_best(src, tgt, plan, cands, measurer, weights=DEFAULT_WEIGHTS)
    assert a.candidate_id == b.candidate_id
    assert [r.total for r in ra] == [r.total for r in rb]


def test_pick_best_uses_given_weights(request, measurer):
    src, tgt, plan, cands = _inputs(request, measurer, "l2")
    geo = {k: 0.0 for k in GEO_KEYS}; geo["overlap"] = 0.6; geo["balance"] = 0.4
    pix = {"text_contrast": 0.2, "busyness": 0.3, "placeholder": 0.5}
    w = GraderWeights(geo, pix, 0.4)
    _, reports = pick_best(src, tgt, plan, cands, measurer, weights=w)
    rendered = [r for r in reports if r.rendered]
    assert rendered
    for r in rendered:
        g = sum(geo[k] * r.metrics[k] for k in geo)
        p = sum(pix[k] * r.metrics[k] for k in pix) if all(k in r.metrics for k in pix) else 0.0
        assert r.total == pytest.approx(0.4 * g + 0.6 * p, abs=1e-9)
