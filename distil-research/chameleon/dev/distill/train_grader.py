"""Fit the grader's weights from the teacher's pairwise judgements.

    python -m dev.distill.train_grader [--pool training-data/candidate-pool/pool.jsonl]
        [--judgments training-data/judgments/judgments.jsonl] [--teacher NAME] [--write]
        [--extra-features training-data/attention/relation_features.jsonl]

A Bradley-Terry model: P(A beats B) = sigmoid(w . (metrics_a - metrics_b)) over
the 12 metrics, no intercept, w >= 0 (projected gradient descent, numpy only).
`--write` only replaces the engine's weights.json when the learned weights beat
the hand-set defaults on the holdout ads by MIN_GAIN, so a noisy fit cannot
make the engine worse.

`--extra-features PATH` (research, attention experiment E1) also fits plain
L2-regularised logistic regressions with signed weights on the 12 metrics
plus per-candidate extra features ({cell, candidate_id, features} rows) and
reports them next to the 12-metric graders. Those models are never written to
the engine: it cannot compute the extra features.
"""

import argparse
import datetime
import json
import math
import os
import sys
from dataclasses import dataclass
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


# --- --extra-features: research only (attention experiment E1) ---------------
# The engine cannot compute extra features, so nothing below ever writes weights.

L2_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)
CV_FOLDS = 5
NEWTON_STEPS = 50


@dataclass(frozen=True)
class LogisticModel:
    """P(A beats B) = sigmoid(sum_k weights[k] * (a[k] - b[k]) / scale[k]). No
    intercept, so swapping A and B flips the probability. `weights` are in
    standardised units (each feature difference divided by its RMS over the
    training pairs), so their sizes are comparable across features."""
    names: tuple[str, ...]
    weights: dict
    scale: dict
    l2: float

    def score(self, m: dict) -> float:
        return sum(self.weights[k] * m[k] / self.scale[k] for k in self.names)


def _diffs(examples, names) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([[a[k] - b[k] for k in names] for a, b, _ in examples], dtype=float),
            np.array([y for _, _, y in examples], dtype=float))


def fit_logistic(examples, names, l2: float = 1e-2) -> LogisticModel:
    """Plain L2-regularised logistic regression on feature differences, signed
    weights (so "less is better" features work), fitted by Newton's method."""
    if not examples:
        raise ValueError("no training examples")
    names = tuple(names)
    x, y = _diffs(examples, names)
    rms = np.sqrt((x ** 2).mean(axis=0))
    scale = np.where(rms > 1e-12, rms, 1.0)
    x = x / scale
    n, d = x.shape
    w = np.zeros(d)
    for _ in range(NEWTON_STEPS):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        grad = x.T @ (p - y) / n + l2 * w
        hess = (x.T * (p * (1 - p))) @ x / n + l2 * np.eye(d)
        step = np.linalg.solve(hess, grad)
        w -= step
        if float(np.abs(step).max()) < 1e-10:
            break
    return LogisticModel(names, {k: float(v) for k, v in zip(names, w)},
                         {k: float(v) for k, v in zip(names, scale)}, l2)


def logistic_accuracy(model: LogisticModel, examples) -> float:
    """Share of examples where the higher score is the judged winner (a tie counts as a miss)."""
    if not examples:
        raise ValueError("no examples")
    return sum(_logistic_hits(model, examples)) / len(examples)


def _logistic_hits(model: LogisticModel, examples) -> list[bool]:
    out = []
    for a, b, y in examples:
        d = model.score(a) - model.score(b)
        out.append(d != 0 and (d > 0) == (y == 1.0))
    return out


def _folds(groups) -> list[list[int]]:
    """CV_FOLDS folds of example indices, whole groups (ads) per fold, deterministic."""
    order = sorted(set(groups))
    fold_of = {g: i % CV_FOLDS for i, g in enumerate(order)}
    return [[i for i, g in enumerate(groups) if fold_of[g] == f] for f in range(CV_FOLDS)]


def cv_accuracy(examples, groups, names, l2: float) -> float:
    """Grouped (by ad) CV_FOLDS-fold accuracy on the training pairs only."""
    hits = 0
    for fold in _folds(groups):
        held = set(fold)
        if not held or len(held) == len(examples):
            continue
        model = fit_logistic([e for i, e in enumerate(examples) if i not in held], names, l2)
        hits += sum(_logistic_hits(model, [examples[i] for i in fold]))
    return hits / len(examples)


def choose_l2(examples, groups, names, grid=L2_GRID) -> tuple[float, dict]:
    """(best l2, {l2: grouped CV accuracy}); ties go to the stronger penalty."""
    cv = {l2: cv_accuracy(examples, groups, names, l2) for l2 in grid}
    return max(grid, key=lambda l2: (cv[l2], l2)), cv


def join_extra_features(pool_rows, extra_rows) -> list[dict]:
    """Pool rows with each candidate's extra features merged into `metrics`;
    rows with no extra features are left out. The inputs are not changed."""
    extra = {(r["cell"], r["candidate_id"]): r["features"] for r in extra_rows}
    return [dict(r, metrics={**r["metrics"], **extra[r["cell"], r["candidate_id"]]})
            for r in pool_rows if (r["cell"], r["candidate_id"]) in extra]


def _pairs(pool_rows, judgments) -> tuple[list, list, list]:
    """(train examples, their task_ids, holdout examples), same rules as _join."""
    by_key = {(r["cell"], r["candidate_id"]): r for r in pool_rows}
    train, groups, holdout = [], [], []
    for j in judgments:
        ra, rb = by_key.get((j["cell"], j["a"])), by_key.get((j["cell"], j["b"]))
        if j["winner"] == "tie" or ra is None or rb is None or ra["split"] != rb["split"]:
            continue
        example = (ra["metrics"], rb["metrics"], 1.0 if j["winner"] == "a" else 0.0)
        if ra["split"] == "holdout":
            holdout.append(example)
        else:
            train.append(example)
            groups.append(ra["task_id"])
    return train, groups, holdout


def _mcnemar_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar (binomial) p-value for paired hit/miss counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def extended_report(pool_rows, judgments, extra_rows, learned: GraderWeights) -> tuple[list[str], dict]:
    """Markdown lines + a JSON-able dict comparing, on the pairs whose two
    candidates both have extra features: the hand-set and learned 12-metric
    graders, and signed logistic models on the 12 metrics, the extras alone and
    both. Each logistic model's l2 is chosen by grouped CV on train only."""
    rows = join_extra_features(pool_rows, extra_rows)
    extra_names = [k for k in extra_rows[0]["features"] if k not in METRIC_NAMES] if extra_rows else []
    train, groups, holdout = _pairs(rows, judgments)
    brands = holdout_by_brand(rows, judgments)
    if not train or not holdout:
        return ["", "## Extended model", "", "not enough pairs with extra features"], {"error": "no pairs"}
    hits: dict[str, list[bool]] = {}
    acc: dict[str, dict] = {}
    models: dict[str, LogisticModel] = {}
    for name, w in (("default", DEFAULT_WEIGHTS), ("learned", learned)):
        def bt_hits(ex, w=w):
            return [(_total(w, a) > _total(w, b)) == (y == 1.0) and _total(w, a) != _total(w, b) for a, b, y in ex]
        hits[name] = bt_hits(holdout)
        acc[name] = {"train": float(np.mean(bt_hits(train))), "holdout": float(np.mean(hits[name])), "cv": None,
                     "l2": None, "by_brand": {b: float(np.mean(bt_hits(ex))) for b, ex in brands.items()}}
    for name, names in (("logistic 12", METRIC_NAMES), ("logistic extra only", extra_names),
                        ("logistic 12 + extra", METRIC_NAMES + extra_names)):
        if not names:
            continue
        l2, cv = choose_l2(train, groups, names)
        model = fit_logistic(train, names, l2)
        models[name] = model
        hits[name] = _logistic_hits(model, holdout)
        acc[name] = {"train": logistic_accuracy(model, train), "holdout": float(np.mean(hits[name])), "cv": cv[l2],
                     "l2": l2, "by_brand": {b: logistic_accuracy(model, ex) for b, ex in brands.items()}}
    order = sorted(brands, key=lambda b: (b != "solstice", b))
    lines = ["", "## Extended model (12 metrics + extra features; research only, never written to the engine)", "",
             f"extra features: {len(extra_names)}; pairs with features: train {len(train)}, holdout {len(holdout)} "
             f"({', '.join(f'{b} {len(brands[b])}' for b in order)})", "",
             "| model | train | train CV (by ad) | holdout | " + " | ".join(f"holdout {b}" for b in order) + " | l2 |",
             "|---|---|---|---|" + "---|" * len(order) + "---|"]
    for name, a in acc.items():
        cv = f"{a['cv']:.3f}" if a["cv"] is not None else "-"
        l2 = f"{a['l2']:g}" if a["l2"] is not None else "-"
        lines.append(f"| {name} | {a['train']:.3f} | {cv} | {a['holdout']:.3f} | "
                     + " | ".join(f"{a['by_brand'][b]:.3f}" for b in order) + f" | {l2} |")
    paired = {}
    if "logistic 12 + extra" in hits:
        lines += ["", "paired holdout comparison against `logistic 12 + extra` (pairs only it gets right / only the other gets right, exact McNemar p):"]
        for other in [k for k in hits if k != "logistic 12 + extra"]:
            only_ext = sum(e and not o for e, o in zip(hits["logistic 12 + extra"], hits[other]))
            only_other = sum(o and not e for e, o in zip(hits["logistic 12 + extra"], hits[other]))
            paired[other] = {"only_extended": only_ext, "only_other": only_other, "p": _mcnemar_p(only_ext, only_other)}
            lines.append(f"- vs {other}: {only_ext} / {only_other}, p = {paired[other]['p']:.3f}")
    top = []
    if "logistic 12 + extra" in models:
        m = models["logistic 12 + extra"]
        top = sorted(((k, m.weights[k]) for k in m.names), key=lambda kv: -abs(kv[1]))
        lines += ["", "largest |weight| in `logistic 12 + extra` (standardised units; + = more is better):", "",
                  "| feature | kind | weight |", "|---|---|---|"]
        lines += [f"| {k} | {'metric' if k in METRIC_NAMES else 'extra'} | {v:+.3f} |" for k, v in top[:15]]
    data = {"extra_feature_count": len(extra_names),
            "examples": {"train": len(train), "holdout": len(holdout), "holdout_by_brand": {b: len(brands[b]) for b in order}},
            "accuracy": acc, "paired_vs_extended": paired,
            "weights": {name: m.weights for name, m in models.items()},
            "top_extra_features": [[k, v] for k, v in top if k not in METRIC_NAMES][:10],
            "top_features": [[k, v] for k, v in top][:15]}
    return lines, data


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
    ap.add_argument("--extra-features", type=Path, default=None,
                    help="jsonl of {cell, candidate_id, features}: also report signed logistic models with them (never written)")
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
    extended = None
    if args.extra_features:
        ext_lines, extended = extended_report(pool_rows, judgments, _read_jsonl(args.extra_features), learned)
        summary += ext_lines
    print("\n".join(summary))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("# Grader training report\n\n" + "\n".join(summary) + "\n")
    by_brand = {brand: {"examples": len(ex), "default": pairwise_accuracy(DEFAULT_WEIGHTS, ex),
                        "learned": pairwise_accuracy(learned, ex)}
                for brand, ex in holdout_by_brand(pool_rows, judgments).items()}
    args.report.with_suffix(".json").write_text(json.dumps({  # machine-readable twin, read by the dashboard
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rubric_version": version, "min_confidence": args.min_confidence, "dropped": dropped,
        "examples": {"train": len(train), "holdout": len(holdout), "ties": counts["ties"], "skipped": counts["skipped"]},
        "accuracy": {f"{name}_{split}": acc[name, split] for name, split in acc}, "holdout_gain": gain,
        "holdout_by_brand": by_brand,
        "weights": {"default": {**DEFAULT_WEIGHTS.geometry, **DEFAULT_WEIGHTS.pixels, "geometry_share": DEFAULT_WEIGHTS.geometry_share},
                    "learned": {**learned.geometry, **learned.pixels, "geometry_share": learned.geometry_share}},
        **({"extended": extended} if extended is not None else {}),
    }, indent=1))

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
