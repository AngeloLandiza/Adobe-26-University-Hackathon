"""Node 3: choose the winning candidate (docs/pipeline/03-grading-node.md).

Gates first, then geometry for every survivor, then pixel metrics for the
top K by geometry (rendering is the expensive step). Only the top K can
win. If no real candidate survives, the safe fallback is returned.

Research switches (off by default; with them off the pick is unchanged):
- CHAMELEON_RELPRESERVE=1: every total loses w x (1 - relation score), the
  share of the source's pairwise relations the candidate keeps
  (grading/relations.py; w from CHAMELEON_RELPRESERVE_W, default chosen on
  train judgments).
- CHAMELEON_NEAR_FIT=1: when the target's aspect is within NEAR_FIT_RATIO of
  the source's, the safe fallback (the whole original scaled uniformly and
  centred, the background filling the canvas) wins unless the best re-layout
  beats its total by more than NEAR_FIT_MARGIN.
- CHAMELEON_NEAR_FIT_GUARD=1 (only with CHAMELEON_NEAR_FIT=1): the near-fit
  rule does not hand the pick to the fallback when the shrink costs too much:
  its headline (and CTA, if any) is at most GUARD_SIZE x the size the best
  re-layout gives them, and its largest empty rectangle is at least
  GUARD_EMPTY of the canvas larger than the re-layout's (the shrink leaves
  room unused that the re-layout fills). Thresholds set on train judgments
  (dev/distill/nearfit_guard.py; `near_fit_features`).
"""

import os

from engine.pipeline.grading import relations
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.grading.metrics import geometry_metrics, geometry_total
from engine.pipeline.grading.pixels import pixel_metrics, pixel_total
from engine.pipeline.grading.weights import GraderWeights, load_weights
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Candidate, GradeReport, PriorityPlan

TOP_K = 6
NEAR_FIT_ENV = "CHAMELEON_NEAR_FIT"
NEAR_FIT_RATIO = 1.35   # target and source aspects within this factor: the original, shrunk, is a strong answer
NEAR_FIT_MARGIN = 0.10  # ... and keeps the pick unless a re-layout leads by more (set on train judgments)
GUARD_ENV = "CHAMELEON_NEAR_FIT_GUARD"
GUARD_SIZE = 0.8        # fallback headline (and CTA) at most this x the best re-layout's: clearly smaller ...
GUARD_EMPTY = 0.10      # ... and its largest empty rectangle this share of the canvas above the re-layout's
_LAYER_ROLES = ("background", "decoration")


def near_fit(source: dict, target: dict) -> bool:
    """CHAMELEON_NEAR_FIT=1 and the target's aspect is within NEAR_FIT_RATIO of the source's."""
    if os.environ.get(NEAR_FIT_ENV) != "1":
        return False
    a = source["canvas"]["width"] / source["canvas"]["height"]
    b = target["width"] / target["height"]
    return max(a / b, b / a) <= NEAR_FIT_RATIO


def guard_enabled() -> bool:
    return os.environ.get(GUARD_ENV) == "1"


def _role_font(design: dict, plan: PriorityPlan, role: str) -> float | None:
    els = {e["id"]: e for e in design["elements"]}
    sizes = [els[i]["font_size"] for u in plan.units if u.role == role for i in u.element_ids
             if i in els and els[i]["type"] == "text"]
    return max(sizes) if sizes else None


def _empty_share(design: dict, plan: PriorityPlan, measurer: TextMeasurer) -> float:
    """Largest axis-aligned rectangle touching no content unit's ink, as a share of the canvas."""
    from engine.pipeline.geometry import rotated_aabb, union_box
    from engine.pipeline.layout.strategies.bisection import largest_empty_share
    els = {e["id"]: e for e in design["elements"]}

    def ink(el: dict):
        if el["type"] != "text" or el.get("rotation"):
            return rotated_aabb(el)
        widths = measurer.line_widths(el, el["font_size"], el["width"]) or [0.0]
        w = min(max(widths), el["width"])
        align = el.get("text_align") or "left"
        x = el["x"] + (el["width"] - w) / 2 if align == "center" else \
            el["x"] + el["width"] - w if align == "right" else el["x"]
        return (x, el["y"], w, min(el["height"], measurer.block_height(el, el["font_size"], el["width"])))

    boxes = [union_box([ink(els[i]) for i in u.element_ids]) for u in plan.units
             if u.role not in _LAYER_ROLES and u.element_ids and all(i in els for i in u.element_ids)]
    canvas = design["canvas"]
    return largest_empty_share(boxes, canvas["width"], canvas["height"])


def near_fit_features(source: dict, plan: PriorityPlan, measurer: TextMeasurer, fallback: dict, best: dict) -> dict:
    """What the uniform fit costs against a re-layout: headline and CTA size
    ratios (fallback / re-layout; None without that text) and the largest
    empty rectangle's share of the canvas in the source, the fallback and
    the re-layout."""
    ratio = {}
    for role in ("headline", "cta"):
        a, b = _role_font(fallback, plan, role), _role_font(best, plan, role)
        ratio[role] = a / b if a and b else None
    return {"head_ratio": ratio["headline"], "cta_ratio": ratio["cta"],
            "empty_src": _empty_share(source, plan, measurer), "empty_fb": _empty_share(fallback, plan, measurer),
            "empty_best": _empty_share(best, plan, measurer)}


def guard_blocks(f: dict) -> bool:
    """CHAMELEON_NEAR_FIT_GUARD: keep the re-layout when the fit shrinks the
    headline and CTA clearly and leaves much more of the canvas empty than
    the re-layout does."""
    if f["head_ratio"] is None or f["head_ratio"] > GUARD_SIZE:
        return False
    if f["cta_ratio"] is not None and f["cta_ratio"] > GUARD_SIZE:
        return False
    return f["empty_fb"] - f["empty_best"] >= GUARD_EMPTY


def _pixels(cand: Candidate, plan: PriorityPlan, weights: GraderWeights) -> tuple[dict, float]:
    try:
        pixels = pixel_metrics(cand.design, plan)
        return pixels, pixel_total(pixels, weights.pixels)
    except Exception:  # an unrenderable candidate scores nothing on pixels, it doesn't stop the pick
        return {}, 0.0


def pick_best(source: dict, target: dict, plan: PriorityPlan, candidates: list[Candidate],
              measurer: TextMeasurer, top_k: int = TOP_K,
              weights: GraderWeights | None = None) -> tuple[Candidate, list[GradeReport]]:
    """The winner, plus one report per input candidate in input order. Ties
    are broken by candidate_id, so the same input always picks the same winner.
    `weights` None means load_weights(): the weights file if there is a valid
    one, else the hand-set defaults."""
    weights = weights if weights is not None else load_weights()
    rel_w = relations.weight() if relations.enabled() else 0.0
    reports: list[GradeReport] = []
    passing: list[tuple[int, Candidate, float]] = []
    penalty: dict[int, float] = {}  # (relpreserve) report index -> w x (1 - relation score)
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
        if rel_w:
            rel = relations.relation_score(source, cand.design, plan, measurer, skip=cand.degraded_unit_ids)
            metrics = {**metrics, "relation": rel}
            penalty[len(reports)] = rel_w * (1 - rel)
            total -= penalty[len(reports)]
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

    def grade(index: int, cand: Candidate, geo: float) -> float:
        pixels, pix = _pixels(cand, plan, weights)
        report = reports[index]
        report.metrics = {**report.metrics, **pixels}
        report.rendered = True
        if index in penalty:  # (relpreserve) the penalty counts in full on the final total
            geo += penalty[index]
            report.total = weights.geometry_share * geo + (1 - weights.geometry_share) * pix - penalty[index]
        else:
            report.total = weights.geometry_share * geo + (1 - weights.geometry_share) * pix
        return report.total

    scored = []
    for index, cand, geo in sorted(passing, key=lambda p: (-p[2], p[1].candidate_id))[:top_k]:
        scored.append((grade(index, cand, geo), cand))
    winner = min(scored, key=lambda s: (-s[0], s[1].candidate_id))[1]
    if not winner.is_fallback and near_fit(source, target):
        fb = next(((i, c, g) for i, c, g in passing if c.is_fallback), None)
        if fb is not None:
            fb_total = next((t for t, c in scored if c is fb[1]), None)
            if fb_total is None:
                fb_total = grade(*fb)
            best_total = max(t for t, c in scored if not c.is_fallback)
            if best_total - fb_total <= NEAR_FIT_MARGIN:
                if not (guard_enabled() and guard_blocks(near_fit_features(
                        source, plan, measurer, fb[1].design, winner.design))):
                    winner = fb[1]
    return winner, reports
