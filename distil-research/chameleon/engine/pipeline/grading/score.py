"""Node 3: choose the winning candidate (docs/pipeline/03-grading-node.md).

Gates first, then geometry for every survivor, then pixel metrics for the
top K by geometry (rendering is the expensive step). Only the top K can
win. If no real candidate survives, the safe fallback is returned.
"""

import os

from engine.pipeline.grading.gates import check_gates
from engine.pipeline.grading.metrics import geometry_metrics, geometry_total
from engine.pipeline.grading.pixels import pixel_metrics, pixel_total
from engine.pipeline.grading.weights import GraderWeights, load_weights
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Candidate, GradeReport, PriorityPlan

TOP_K = 6


def pick_best(source: dict, target: dict, plan: PriorityPlan, candidates: list[Candidate],
              measurer: TextMeasurer, top_k: int = TOP_K,
              weights: GraderWeights | None = None) -> tuple[Candidate, list[GradeReport]]:
    """The winner, plus one report per input candidate in input order. Ties
    are broken by candidate_id, so the same input always picks the same winner.
    `weights` None means load_weights(): the weights file if there is a valid
    one, else the hand-set defaults."""
    weights = weights if weights is not None else load_weights()
    reports: list[GradeReport] = []
    passing: list[tuple[int, Candidate, float]] = []
    for cand in candidates:
        failed = check_gates(source, cand.design, target, plan, measurer)
        if failed:
            reports.append(GradeReport(cand.candidate_id, False, failed, {}, False, 0.0))
            continue
        try:
            metrics = geometry_metrics(source, cand, plan)
        except Exception:
            reports.append(GradeReport(cand.candidate_id, False, ["metrics_error"], {}, False, 0.0))
            continue
        total = geometry_total(metrics, weights.geometry)
        reports.append(GradeReport(cand.candidate_id, True, [], metrics, False, total))
        passing.append((len(reports) - 1, cand, total))

    if not any(not cand.is_fallback for _, cand, _ in passing):
        fallback = next((c for c in candidates if c.is_fallback), None)
        return fallback or (passing[0][1] if passing else candidates[0]), reports

    if os.environ.get("CHAMELEON_PICKER") == "rules":
        # Research picker: the rulebook mined from the teacher's critiques
        # (layout/strategies/rules.py) predicts its preference better than the
        # weighted metrics. Every gate-passing candidate competes; the grader's
        # geometry total breaks ties.
        from engine.pipeline.layout.strategies import rules
        ranked = []
        for index, cand, geo in passing:
            pen, _ = rules.penalty(source, cand.design, plan, measurer)
            reports[index].total = geo
            ranked.append((pen, -geo, cand.candidate_id, cand))
        return min(ranked, key=lambda r: r[:3])[3], reports

    scored = []
    for index, cand, geo in sorted(passing, key=lambda p: (-p[2], p[1].candidate_id))[:top_k]:
        try:
            pixels = pixel_metrics(cand.design, plan)
            pix = pixel_total(pixels, weights.pixels)
        except Exception:  # an unrenderable candidate scores nothing on pixels, it doesn't stop the pick
            pixels, pix = {}, 0.0
        report = reports[index]
        report.metrics = {**report.metrics, **pixels}
        report.rendered = True
        report.total = weights.geometry_share * geo + (1 - weights.geometry_share) * pix
        scored.append((report.total, cand))
    winner = min(scored, key=lambda s: (-s[0], s[1].candidate_id))[1]
    return winner, reports
