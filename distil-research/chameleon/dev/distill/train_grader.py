"""Fit the grader's weights from the teacher's pairwise judgements.

    python -m dev.distill.train_grader [--pool training-data/candidate-pool/pool.jsonl]
        [--judgments training-data/judgments/judgments.jsonl] [--teacher NAME] [--write]

A Bradley-Terry model: P(A beats B) = sigmoid(w . (metrics_a - metrics_b)) over
the 12 metrics, no intercept, w >= 0 (projected gradient descent, numpy only).
`--write` only replaces the engine's weights.json when the learned weights beat
the hand-set defaults on the holdout ads by MIN_GAIN, so a noisy fit cannot
make the engine worse.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

from dev.distill.paths import DATA_ROOT, JUDGMENTS_FILE, POOL_DIR
from engine.pipeline.grading.weights import DEFAULT_WEIGHTS, ENV_VAR, WEIGHTS_PATH, GraderWeights

GEOMETRY_NAMES = list(DEFAULT_WEIGHTS.geometry)
PIXEL_NAMES = list(DEFAULT_WEIGHTS.pixels)
METRIC_NAMES = GEOMETRY_NAMES + PIXEL_NAMES
MIN_GAIN = 0.03
ITERATIONS = 3000
REPORT_FILE = DATA_ROOT / "judgments" / "training-report.md"


def _matrix(examples) -> tuple[np.ndarray, np.ndarray]:
    diffs = np.array([[a[k] - b[k] for k in METRIC_NAMES] for a, b, _ in examples], dtype=float)
    return diffs, np.array([y for _, _, y in examples], dtype=float)


def _to_weights(raw: np.ndarray) -> tuple[GraderWeights, list[str]]:
    """Raw non-negative vector -> GraderWeights. A group that sums to 0 keeps
    its default weights (named in the returned notes)."""
    notes = []
    n_geo = len(GEOMETRY_NAMES)
    groups = []
    for names, part, default in ((GEOMETRY_NAMES, raw[:n_geo], DEFAULT_WEIGHTS.geometry),
                                 (PIXEL_NAMES, raw[n_geo:], DEFAULT_WEIGHTS.pixels)):
        total = float(part.sum())
        if total > 0:
            groups.append({k: float(v) / total for k, v in zip(names, part)})
        else:
            groups.append(dict(default))
            notes.append(f"all-zero group fell back to defaults: {names[0]}...")
    geo_sum, total = float(raw[:n_geo].sum()), float(raw.sum())
    share = geo_sum / total if total > 0 else DEFAULT_WEIGHTS.geometry_share
    weights = GraderWeights(groups[0], groups[1], share)
    return weights, notes


def _fit(examples, l2: float, seed: int) -> tuple[GraderWeights, list[str]]:
    if not examples:
        raise ValueError("no training examples")
    diffs, y = _matrix(examples)
    n = len(y)
    # Step 1/L, where L bounds the curvature of the (mean) logistic loss + l2.
    curvature = 0.25 * float(np.linalg.eigvalsh(diffs.T @ diffs / n)[-1]) + l2
    step = 1.0 / max(curvature, 1e-12)
    rng = np.random.default_rng(seed)
    w = rng.random(len(METRIC_NAMES)) * 1e-3
    look, prev, t = w.copy(), w.copy(), 1.0
    for _ in range(ITERATIONS):  # Nesterov-accelerated projected gradient
        p = 1.0 / (1.0 + np.exp(-np.clip(diffs @ look, -30, 30)))
        grad = diffs.T @ (p - y) / n + l2 * look
        w = np.maximum(look - step * grad, 0.0)
        t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
        look = w + (t - 1) / t_next * (w - prev)
        prev, t = w, t_next
    return _to_weights(w)


def fit_weights(examples, l2: float = 1e-2, seed: int = 0) -> GraderWeights:
    """Weights that best reproduce the judged winners. Raises ValueError if there are no examples."""
    return _fit(examples, l2, seed)[0]


def _total(weights: GraderWeights, metrics: dict) -> float:
    """The same total pick_best ranks by."""
    return (weights.geometry_share * sum(v * metrics[k] for k, v in weights.geometry.items())
            + (1 - weights.geometry_share) * sum(v * metrics[k] for k, v in weights.pixels.items()))


def pairwise_accuracy(weights: GraderWeights, examples) -> float:
    """Share of examples where the higher total is the judged winner (a score tie counts as a miss)."""
    if not examples:
        raise ValueError("no examples")
    hits = sum((_total(weights, a) > _total(weights, b)) == (y == 1.0) and _total(weights, a) != _total(weights, b)
               for a, b, y in examples)
    return hits / len(examples)


def _join(pool_rows, judgments) -> tuple[list, list, dict]:
    """(train, holdout, counts). The split comes from the pool row, never from
    the judgement, so a holdout ad cannot leak into training."""
    by_key = {(r["cell"], r["candidate_id"]): r for r in pool_rows}
    train, holdout = [], []
    counts = {"ties": 0, "skipped": 0}
    for j in judgments:
        if j["winner"] == "tie":
            counts["ties"] += 1
            continue
        ra, rb = by_key.get((j["cell"], j["a"])), by_key.get((j["cell"], j["b"]))
        if ra is None or rb is None or ra["split"] != rb["split"]:
            counts["skipped"] += 1
            continue
        example = (ra["metrics"], rb["metrics"], 1.0 if j["winner"] == "a" else 0.0)
        (holdout if ra["split"] == "holdout" else train).append(example)
    return train, holdout, counts


def split_examples(pool_rows, judgments) -> tuple[list, list]:
    """(train, holdout) examples; ties and judgements with unknown candidates are dropped."""
    train, holdout, _ = _join(pool_rows, judgments)
    return train, holdout


def holdout_by_brand(pool_rows, judgments) -> dict[str, list]:
    """Holdout examples grouped by the pool row's `brand` ("solstice",
    "offbrand"; "unknown" for rows written before the field existed), so the
    report can show whether a fit generalises beyond the sample brand."""
    rows = [dict(r, split=f"holdout:{r.get('brand', 'unknown')}") if r["split"] == "holdout" else r
            for r in pool_rows]
    groups: dict[str, list] = {}
    by_key = {(r["cell"], r["candidate_id"]): r for r in rows}
    for j in judgments:
        if j["winner"] == "tie":
            continue
        ra, rb = by_key.get((j["cell"], j["a"])), by_key.get((j["cell"], j["b"]))
        if ra is None or rb is None or ra["split"] != rb["split"] or not ra["split"].startswith("holdout:"):
            continue
        groups.setdefault(ra["split"].split(":", 1)[1], []).append(
            (ra["metrics"], rb["metrics"], 1.0 if j["winner"] == "a" else 0.0))
    return groups


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _weights_table(learned: GraderWeights) -> list[str]:
    lines = ["| metric | default | learned |", "|---|---|---|"]
    for names, attr in ((GEOMETRY_NAMES, "geometry"), (PIXEL_NAMES, "pixels")):
        for k in names:
            lines.append(f"| {attr}.{k} | {getattr(DEFAULT_WEIGHTS, attr)[k]:.3f} | {getattr(learned, attr)[k]:.3f} |")
    lines.append(f"| geometry_share | {DEFAULT_WEIGHTS.geometry_share:.3f} | {learned.geometry_share:.3f} |")
    return lines


def _write_json(path: Path, weights: GraderWeights, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"geometry": weights.geometry, "pixels": weights.pixels,
                                "geometry_share": weights.geometry_share, "source": source}, indent=2) + "\n")


def select_judgments(judgments, teacher=None, min_confidence: int = 2, rubric_version=None):
    """(kept, dropped counts). Keep one teacher's judgements (if given), made
    under `rubric_version` (if given), with confidence >= `min_confidence`.
    The pilot showed confidence-1 answers flip when A and B are swapped, so by
    default they count as "no preference" rather than as training labels."""
    kept, dropped = [], {"teacher": 0, "low_confidence": 0, "rubric_version": 0}
    for j in judgments:
        if teacher and j["teacher"] != teacher:
            dropped["teacher"] += 1
        elif rubric_version and j.get("rubric_version") != rubric_version:
            dropped["rubric_version"] += 1
        elif j.get("confidence", 3) < min_confidence:
            dropped["low_confidence"] += 1
        else:
            kept.append(j)
    return kept, dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", type=Path, default=POOL_DIR / "pool.jsonl")
    ap.add_argument("--judgments", type=Path, default=JUDGMENTS_FILE)
    ap.add_argument("--teacher", default=None, help="use only this teacher's judgements (default: every teacher in the file)")
    ap.add_argument("--min-confidence", type=int, default=2, help="drop judgements below this confidence (1-3)")
    ap.add_argument("--rubric-version", default=None,
                    help="only judgements made under this rubric (default: the current rubric.md; 'any' for all)")
    ap.add_argument("--report", type=Path, default=REPORT_FILE)
    ap.add_argument("--write", action="store_true", help="write the engine's weights.json if holdout accuracy gains >= 0.03")
    args = ap.parse_args(argv)

    from dev.distill.judge import RUBRIC_VERSION
    version = None if args.rubric_version == "any" else (args.rubric_version or RUBRIC_VERSION)
    judgments, dropped = select_judgments(_read_jsonl(args.judgments), args.teacher, args.min_confidence, version)
    per_teacher: dict[str, int] = {}
    for j in judgments:
        per_teacher[j["teacher"]] = per_teacher.get(j["teacher"], 0) + 1
    pool_rows = _read_jsonl(args.pool)
    train, holdout, counts = _join(pool_rows, judgments)
    if not train or not holdout:
        print(f"need both train and holdout examples (train={len(train)}, holdout={len(holdout)})", file=sys.stderr)
        return 1
    learned, notes = _fit(train, 1e-2, 0)
    acc = {(name, split): pairwise_accuracy(w, ex)
           for name, w in (("default", DEFAULT_WEIGHTS), ("learned", learned))
           for split, ex in (("train", train), ("holdout", holdout))}
    gain = acc["learned", "holdout"] - acc["default", "holdout"]

    teachers = ", ".join(f"{t}: {n}" for t, n in sorted(per_teacher.items())) or "none"
    summary = [f"teachers: {teachers}",
               f"filters: rubric {version or 'any'}, confidence >= {args.min_confidence}; dropped {dropped}",
               f"examples: train {len(train)}, holdout {len(holdout)}, ties dropped {counts['ties']}, "
               f"skipped (candidate missing from pool) {counts['skipped']}",
               "| accuracy | train | holdout |", "|---|---|---|"]
    for name in ("default", "learned"):
        summary.append(f"| {name} | {acc[name, 'train']:.3f} | {acc[name, 'holdout']:.3f} |")
    summary += ["", "| holdout brand | examples | default | learned |", "|---|---|---|---|"]
    for brand, examples in sorted(holdout_by_brand(pool_rows, judgments).items(), key=lambda kv: kv[0] != "solstice"):
        summary.append(f"| {brand} | {len(examples)} | {pairwise_accuracy(DEFAULT_WEIGHTS, examples):.3f} | "
                       f"{pairwise_accuracy(learned, examples):.3f} |")
    summary += ["", f"holdout gain: {gain:+.3f} (needs >= {MIN_GAIN})", *notes, "", *_weights_table(learned)]
    print("\n".join(summary))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("# Grader training report\n\n" + "\n".join(summary) + "\n")

    if args.write:
        if gain < MIN_GAIN - 1e-9:
            print(f"refusing to write weights: holdout gain {gain:+.3f} is below {MIN_GAIN}")
            return 0
        target = Path(os.environ.get(ENV_VAR) or WEIGHTS_PATH)
        _write_json(target, learned, f"learned:{datetime.date.today().isoformat()}-{len(train)}")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
