"""CHAMELEON_STRATEGIES switches layout strategies on by name: candidate
generators add layouts, refiners may replace the winner. Unset = nothing runs."""

import json
import sys
import types
from dataclasses import replace
from pathlib import Path

from engine.pipeline.layout import strategy_registry as reg
from engine.pipeline.types import Candidate


def _fake(monkeypatch, name, **funcs):
    mod = types.ModuleType(f"engine.pipeline.layout.strategies.{name}")
    for k, v in funcs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    reg._modules.clear()


def test_unset_runs_nothing(monkeypatch):
    monkeypatch.delenv("CHAMELEON_STRATEGIES", raising=False)
    assert reg.enabled() == []
    assert reg.extra_candidates({}, {}, None, None) == []


def test_candidates_strategy_adds_layouts(monkeypatch):
    c = Candidate("fake/1", {"canvas": {}, "elements": []}, "fake", {})
    _fake(monkeypatch, "fakegen", candidates=lambda s, t, p, m: [c])
    monkeypatch.setenv("CHAMELEON_STRATEGIES", "fakegen")
    assert reg.extra_candidates({}, {}, None, None) == [c]


def test_unknown_or_crashing_strategy_never_raises(monkeypatch, capsys):
    def boom(*a):
        raise RuntimeError("bug")
    _fake(monkeypatch, "boomgen", candidates=boom)
    monkeypatch.setenv("CHAMELEON_STRATEGIES", "boomgen, nosuchthing, Bad-Name")
    assert reg.extra_candidates({}, {}, None, None) == []
    assert "[distill]" in capsys.readouterr().err


def test_refiner_replaces_winner_only_when_gates_pass(monkeypatch):
    task = json.loads(Path("input/chameleon_sample_level1_warmup.json").read_text())
    src, tgt = task["source"], task["target_canvas"]
    from engine.pipeline.priority import heuristic_plan
    from engine.pipeline.textmeasure import TextMeasurer
    from engine.resize import resize_with_report
    monkeypatch.delenv("CHAMELEON_STRATEGIES", raising=False)
    _, plan, best, _ = resize_with_report(src, tgt)
    good = replace(best, candidate_id="fixer+" + best.candidate_id)
    broken_design = json.loads(json.dumps(best.design))
    broken_design["elements"] = broken_design["elements"][1:]  # drops an element: fails G1
    broken = replace(best, candidate_id="breaker+x", design=broken_design)
    _fake(monkeypatch, "fixer", refine=lambda s, t, p, m, c: good)
    _fake(monkeypatch, "breaker", refine=lambda s, t, p, m, c: broken)
    measurer = TextMeasurer()
    monkeypatch.setenv("CHAMELEON_STRATEGIES", "fixer")
    assert reg.refine(src, tgt, plan, measurer, best).candidate_id == good.candidate_id
    monkeypatch.setenv("CHAMELEON_STRATEGIES", "breaker")
    assert reg.refine(src, tgt, plan, measurer, best) is best
