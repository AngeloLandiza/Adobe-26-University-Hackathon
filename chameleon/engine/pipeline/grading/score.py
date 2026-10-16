"""Node 3: choose the winning candidate (docs/pipeline/03-grading-node.md).

Gates first, then geometry for every survivor, then pixel metrics for the
top K by geometry (rendering is the expensive step). Only the top K can
win. If no real candidate survives, the safe fallback is returned.
"""

from engine.pipeline.grading.gates import check_gates
from engine.pipeline.grading.metrics import geometry_metrics, geometry_total
from engine.pipeline.grading.pixels import pixel_metrics, pixel_total
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Candidate, GradeReport, PriorityPlan

TOP_K = 6
GEOMETRY_SHARE = 0.7  # total = 0.7 x geometry + 0.3 x pixels


def pick_best(source: dict, target: dict, plan: PriorityPlan, candidates: list[Candidate],
              measurer: TextMeasurer, top_k: int = TOP_K) -> tuple[Candidate, list[GradeReport]]:
    """The winner, plus one report per input candidate in input order. Ties
    are broken by candidate_id, so the same input always picks the same winner."""
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
        total = geometry_total(metrics)
        reports.append(GradeReport(cand.candidate_id, True, [], metrics, False, total))
        passing.append((len(reports) - 1, cand, total))

    if not any(not cand.is_fallback for _, cand, _ in passing):
        fallback = next((c for c in candidates if c.is_fallback), None)
        return fallback or (passing[0][1] if passing else candidates[0]), reports

    scored = []
    for index, cand, geo in sorted(passing, key=lambda p: (-p[2], p[1].candidate_id))[:top_k]:
        try:
            pixels = pixel_metrics(cand.design, plan)
            pix = pixel_total(pixels)
        except Exception:  # an unrenderable candidate scores nothing on pixels, it doesn't stop the pick
            pixels, pix = {}, 0.0
        report = reports[index]
        report.metrics = {**report.metrics, **pixels}
        report.rendered = True
        report.total = GEOMETRY_SHARE * geo + (1 - GEOMETRY_SHARE) * pix
        scored.append((report.total, cand))
    winner = min(scored, key=lambda s: (-s[0], s[1].candidate_id))[1]
    return winner, reports
