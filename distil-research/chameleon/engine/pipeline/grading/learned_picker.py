"""Research picker CHAMELEON_PICKER=learned: a pairwise logistic model of the
teacher's preference chooses among the grader's shortlist.

The grader (score.pick_best) picks the teacher's favourite among its top
diverse candidates only ~31% of the time (chance ~25%). The model in
picker_weights.json was trained on teacher rankings of exactly such
shortlists (dev/distill/picker_study.py; trained by dev/distill/train_picker.py):
grouped 5-fold CV top-1 43-44% vs the grader's 31%. It ships with the engine
(a few hundred bytes of numbers; no network, no LLM).

How a pick is made, so the features match the training data:
1. pick_best runs as always (gates, geometry, pixels for the top K, near-fit)
   and gives the grader's pick and its reports.
2. `shortlist` rebuilds the study's K: the grader's pick first, then by grader
   total (rendered ones first) with at most MAX_PER_FAMILY per strategy
   family, and the uniform-fit fallback whenever it passes the gates.
3. Each shortlisted candidate gets the grader's metrics (pixel metrics
   computed here when pick_best didn't render it), its relation score
   (grading/relations.py), its family one-hot, the grader's total as
   pick_best left it, the target's log aspect and family x log aspect.
4. The highest model score wins (ties: shortlist order, so the grader's pick).

Out of distribution, the grader's pick stands: fewer than two shortlisted
candidates, or a target within NEAR_SOURCE of the source's aspect (the
training pool has no such cells; the near-identical fallback is right there).

Only gate-passing candidates are ever shortlisted. Any error -> the grader's
pick. `LAST` keeps the last call's shortlist, features and timing (for the
parity check and runtime report; never read by the engine).
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

WEIGHTS_PATH = Path(__file__).with_name("picker_weights.json")
K = 4
MAX_PER_FAMILY = 2
NEAR_SOURCE = 0.02  # target aspect within 2% of the source's: no such cell in the training data (the pool skips
                    # them, as ab_engine.NEAR_SOURCE does), so the grader's pick stands
FAMILIES = ("anchored", "bisection", "fallback", "stack", "row", "columns", "reflow", "other")

LAST: dict = {}
_MODELS: dict[Path, dict] = {}


def load_model(path: Path = WEIGHTS_PATH) -> dict:
    """The model (names, w, mu, sd), read once per process."""
    path = Path(path)
    if path not in _MODELS:
        m = json.loads(path.read_text())
        if not (len(m["names"]) == len(m["w"]) == len(m["mu"]) == len(m["sd"])):
            raise ValueError("picker weights: names, w, mu, sd differ in length")
        _MODELS[path] = m
    return _MODELS[path]


def family(cand) -> str:
    """The candidate's strategy family: its template (row, stack, bisection, ...), or fallback."""
    return "fallback" if cand.is_fallback else (cand.template or cand.candidate_id.split("/")[0])


def shortlist(candidates: list, reports: list, pick, k: int = K, max_per_family: int = MAX_PER_FAMILY) -> list[int]:
    """Indices (into `candidates`) of the K the picker chooses among, grader's
    order: the pick first, then by grader total (fully graded ones first)
    with at most `max_per_family` per family; the fallback is always included
    when it passes the gates (replacing the last non-pick choice if needed).
    The same selection the picker study showed the teacher."""
    by_id = {r.candidate_id: r for r in reports}
    passing = [i for i, c in enumerate(candidates) if by_id.get(c.candidate_id) and by_id[c.candidate_id].passed_gates]
    order = sorted(passing, key=lambda i: (not by_id[candidates[i].candidate_id].rendered,
                                           -by_id[candidates[i].candidate_id].total, candidates[i].candidate_id))
    pick_i = next((i for i in passing if candidates[i] is pick), None)
    if pick_i is None:
        pick_i = next((i for i in passing if candidates[i].candidate_id == pick.candidate_id), None)
    chosen = [pick_i] if pick_i is not None else []
    count = defaultdict(int)
    for i in chosen:
        count[family(candidates[i])] += 1
    for i in order:
        if len(chosen) >= k:
            break
        if i in chosen or count[family(candidates[i])] >= max_per_family:
            continue
        chosen.append(i)
        count[family(candidates[i])] += 1
    fb = next((i for i in passing if candidates[i].is_fallback), None)
    if fb is not None and fb not in chosen:
        if len(chosen) < k:
            chosen.append(fb)
        elif len(chosen) > 1:
            chosen[-1] = fb
    return chosen


def feature_names(metric_keys) -> list[str]:
    return (sorted(metric_keys) + ["relation"] + [f"fam_{f}" for f in FAMILIES] + ["grader_total", "log_aspect"]
            + [f"famXaspect_{f}" for f in FAMILIES])


def feature_vector(metrics: dict, relation: float, fam: str, grader_total: float, log_aspect: float) -> list[float]:
    """The model's input for one candidate, in feature_names(metrics) order."""
    fam = fam if fam in FAMILIES else "other"
    onehot = [float(fam == f) for f in FAMILIES]
    return ([float(metrics[key]) for key in sorted(metrics)] + [float(relation)] + onehot
            + [float(grader_total), log_aspect] + [x * log_aspect for x in onehot])


def model_scores(model: dict, rows: list[list[float]]) -> list[float]:
    w, mu, sd = model["w"], model["mu"], model["sd"]
    return [sum(wi * (xi - mi) / si for wi, xi, mi, si in zip(w, x, mu, sd)) for x in rows]


def candidate_features(source: dict, cand, report, plan, measurer) -> tuple[dict, float]:
    """(metrics with pixel metrics, relation score) for one gate-passing candidate."""
    from engine.pipeline.grading import relations
    metrics = dict(report.metrics)
    if not report.rendered:
        from engine.pipeline.grading.pixels import pixel_metrics
        metrics.update(pixel_metrics(cand.design, plan))
    try:
        rel = relations.relation_score(source, cand.design, plan, measurer, skip=cand.degraded_unit_ids)
    except Exception:
        rel = 1.0  # the training rows default a missing relation score to 1.0
    return metrics, rel


def choose(source: dict, target: dict, plan, candidates: list, reports: list, pick, measurer):
    """The learned pick among the shortlist (the grader's `pick` when fewer than two)."""
    t0 = time.perf_counter()
    LAST.clear()
    model = load_model()
    idx = shortlist(candidates, reports, pick)
    LAST.update(grader_pick=pick.candidate_id, shortlist=[candidates[i].candidate_id for i in idx])
    src = source["canvas"]["width"] / source["canvas"]["height"]
    if len(idx) < 2 or abs((target["width"] / target["height"]) / src - 1) <= NEAR_SOURCE:
        LAST.update(pick=pick.candidate_id, seconds=time.perf_counter() - t0)
        return pick
    by_id = {r.candidate_id: r for r in reports}
    log_aspect = math.log(target["width"] / target["height"])
    rows, names = [], None
    for i in idx:
        cand = candidates[i]
        report = by_id[cand.candidate_id]
        metrics, rel = candidate_features(source, cand, report, plan, measurer)
        these = feature_names(metrics)
        if these != model["names"]:
            raise ValueError(f"picker features {these} do not match the model's {model['names']}")
        names = these
        rows.append(feature_vector(metrics, rel, family(cand), report.total, log_aspect))
    scores = model_scores(model, rows)
    best = max(range(len(idx)), key=lambda j: (scores[j], -j))  # ties: shortlist order
    winner = candidates[idx[best]]
    LAST.update(names=names, X=rows, scores=scores, families=[family(candidates[i]) for i in idx],
                pick=winner.candidate_id, seconds=time.perf_counter() - t0)
    return winner
