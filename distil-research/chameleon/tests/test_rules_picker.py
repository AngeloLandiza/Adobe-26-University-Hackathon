"""CHAMELEON_PICKER=rules: pick the gate-passing candidate with the lowest
rulebook penalty (strategies/rules.py); unset = today's grader."""

import json
from pathlib import Path

from engine.pipeline.grading import score
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.layout.strategies import rules
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.textmeasure import TextMeasurer


def _cell():
    task = json.loads(Path("input/chameleon_sample_level2_core.json").read_text())
    src, tgt = task["source"], task["target_canvas"]
    plan, m = heuristic_plan(src), TextMeasurer()
    cands = [safe_fallback(src, tgt, plan, m)] + generate_candidates(src, tgt, plan, m)
    return src, tgt, plan, m, cands


def test_unset_keeps_todays_pick(monkeypatch):
    src, tgt, plan, m, cands = _cell()
    monkeypatch.delenv("CHAMELEON_PICKER", raising=False)
    a, _ = score.pick_best(src, tgt, plan, cands, m)
    b, _ = score.pick_best(src, tgt, plan, cands, m)
    assert a.candidate_id == b.candidate_id


def test_rules_picker_takes_lowest_penalty(monkeypatch):
    src, tgt, plan, m, cands = _cell()
    target_id = cands[-1].candidate_id  # make the last candidate the clear rulebook favourite
    monkeypatch.setattr(rules, "penalty", lambda s, d, p, mm, weights=None:
                        (0.0, {}) if d is next(c.design for c in cands if c.candidate_id == target_id) else (5.0, {}))
    monkeypatch.setenv("CHAMELEON_PICKER", "rules")
    best, _ = score.pick_best(src, tgt, plan, cands, m)
    assert best.candidate_id == target_id


def test_rules_picker_never_returns_a_gate_failure(monkeypatch):
    src, tgt, plan, m, cands = _cell()
    broken = json.loads(json.dumps(cands[-1].design))
    broken["elements"] = broken["elements"][1:]
    from dataclasses import replace
    bad = replace(cands[-1], candidate_id="broken", design=broken)
    monkeypatch.setattr(rules, "penalty", lambda s, d, p, mm, weights=None: (0.0, {}) if d is broken else (5.0, {}))
    monkeypatch.setenv("CHAMELEON_PICKER", "rules")
    best, _ = score.pick_best(src, tgt, plan, cands + [bad], m)
    assert best.candidate_id != "broken"
