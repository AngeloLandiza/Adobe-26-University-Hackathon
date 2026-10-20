"""Pins "no data files means no change" for the distillation hooks.

Two checks:
- On every platform: with no weights.json and no extra_templates.json, the
  engine gives exactly what it gives with *neutral* files present (today's
  default weights, zero floor plans). This compares two runs on the same
  machine, so it never depends on how the platform measures text.
- Where a stored expectation exists for this platform: the output also equals
  what the engine produced before the hooks existed. Text measurement differs
  between macOS, Linux and Windows (see README, "Linux output differs from
  Windows output"), so expectations are stored per platform and the check is
  skipped where none was recorded. To record one for the current platform,
  from distil-research/chameleon/:

      python -m tests.test_no_data_files

  Only do that on a tree whose hooks are known to be inert, e.g. right after
  checking out the commit before a hook change.
"""

import json
import sys
from pathlib import Path

import pytest

from engine import resize as R
from engine.pipeline.grading import weights as weights_module
from engine.pipeline.grading.weights import DEFAULT_WEIGHTS
from engine.pipeline.layout import extra_templates

INPUTS = sorted(Path("input").glob("*.json"))
EXPECTED_FILE = Path("tests/no_data_files_expected") / f"{sys.platform}.json"


def _run(task: dict) -> tuple[str, dict]:
    design, _plan, best, _reports = R.resize_with_report(task["source"], task["target_canvas"])
    return best.candidate_id, json.loads(json.dumps(design))


@pytest.fixture
def point_hooks_at(monkeypatch, tmp_path):
    """Point both hooks' default paths at `weights` / `templates` (None = absent)."""
    monkeypatch.delenv("CHAMELEON_GRADER_WEIGHTS", raising=False)
    monkeypatch.delenv("CHAMELEON_EXTRA_TEMPLATES", raising=False)

    def point(weights: dict | None, templates: dict | None) -> None:
        wpath, tpath = tmp_path / "weights.json", tmp_path / "extra_templates.json"
        for path, data in ((wpath, weights), (tpath, templates)):
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(json.dumps(data))
        monkeypatch.setattr(weights_module, "WEIGHTS_PATH", wpath)
        monkeypatch.setattr(extra_templates, "EXTRA_PATH", tpath)

    return point


NEUTRAL_WEIGHTS = {"geometry": DEFAULT_WEIGHTS.geometry, "pixels": DEFAULT_WEIGHTS.pixels,
                   "geometry_share": DEFAULT_WEIGHTS.geometry_share, "source": "hand"}


@pytest.mark.parametrize("path", INPUTS, ids=lambda p: p.stem)
def test_absent_files_equal_neutral_files(path, point_hooks_at, capsys):
    task = json.loads(path.read_text())
    point_hooks_at(None, None)
    absent = _run(task)
    point_hooks_at(NEUTRAL_WEIGHTS, {"floor_plans": []})
    neutral = _run(task)
    assert absent == neutral
    assert "[distill]" not in capsys.readouterr().err


@pytest.mark.skipif(not EXPECTED_FILE.exists(), reason=f"no stored expectation for platform {sys.platform!r}")
@pytest.mark.parametrize("path", INPUTS, ids=lambda p: p.stem)
def test_absent_files_match_pre_hook_engine(path, point_hooks_at):
    want = json.loads(EXPECTED_FILE.read_text())[path.stem]
    point_hooks_at(None, None)
    candidate_id, design = _run(json.loads(path.read_text()))
    assert candidate_id == want["candidate_id"]
    assert design == want["design"]


if __name__ == "__main__":  # record this platform's expectation (see module docstring)
    import os
    os.environ["CHAMELEON_LLM"] = "heuristic"
    os.environ["CHAMELEON_GRADER_WEIGHTS"] = "/nonexistent/weights.json"
    os.environ["CHAMELEON_EXTRA_TEMPLATES"] = "/nonexistent/extra_templates.json"
    out = {}
    for p in INPUTS:
        cid, design = _run(json.loads(p.read_text()))
        out[p.stem] = {"candidate_id": cid, "design": design}
    EXPECTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_FILE.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {EXPECTED_FILE}")
