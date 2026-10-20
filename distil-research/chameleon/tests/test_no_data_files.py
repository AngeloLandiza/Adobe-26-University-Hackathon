"""Pins "no data files means no change": with no weights.json and no
extra_templates.json the engine picks the same winners and writes the same
designs as before the distillation hooks existed, even after those files ship.
Expectations were generated from commit 3a6603a (before the hardening wave)."""

import json
from pathlib import Path

import pytest

from engine import resize as R
from engine.pipeline.grading import weights as weights_module
from engine.pipeline.layout import extra_templates

EXPECTED = json.loads(Path("tests/no_data_files_expected.json").read_text())


@pytest.mark.parametrize("path", sorted(Path("input").glob("*.json")), ids=lambda p: p.stem)
def test_no_data_files_means_no_change(path, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CHAMELEON_GRADER_WEIGHTS", raising=False)
    monkeypatch.delenv("CHAMELEON_EXTRA_TEMPLATES", raising=False)
    monkeypatch.setattr(weights_module, "WEIGHTS_PATH", tmp_path / "absent_weights.json")
    monkeypatch.setattr(extra_templates, "EXTRA_PATH", tmp_path / "absent_templates.json")
    task = json.loads(path.read_text())
    design, _plan, best, _reports = R.resize_with_report(task["source"], task["target_canvas"])
    want = EXPECTED[path.stem]
    assert best.candidate_id == want["candidate_id"]
    assert json.loads(json.dumps(design)) == want["design"]
    assert "[distill]" not in capsys.readouterr().err
