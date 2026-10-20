"""Fitting the grader weights from pairwise judgements (synthetic data only)."""

import json

import numpy as np
import pytest

from dev.distill import train_grader as tg
from engine.pipeline.grading.weights import DEFAULT_WEIGHTS, GraderWeights

NAMES = tg.METRIC_NAMES


def _known() -> GraderWeights:
    geo = dict(zip(DEFAULT_WEIGHTS.geometry, [0.40, 0.20, 0.10, 0.08, 0.07, 0.05, 0.04, 0.04, 0.02]))
    pix = dict(zip(DEFAULT_WEIGHTS.pixels, [0.6, 0.3, 0.1]))
    return GraderWeights(geo, pix, 0.7)  # effective weights: top 3 clearly separated from the rest


def _total(w: GraderWeights, m: dict) -> float:
    return (w.geometry_share * sum(v * m[k] for k, v in w.geometry.items())
            + (1 - w.geometry_share) * sum(v * m[k] for k, v in w.pixels.items()))


def _synthetic(truth: GraderWeights, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = {k: float(rng.random()) for k in NAMES}
        b = {k: float(rng.random()) for k in NAMES}
        out.append((a, b, 1.0 if _total(truth, a) > _total(truth, b) else 0.0))
    return out


def _effective(w: GraderWeights) -> dict:
    eff = {k: w.geometry_share * v for k, v in w.geometry.items()}
    eff.update({k: (1 - w.geometry_share) * v for k, v in w.pixels.items()})
    return eff


def test_recovers_known_weights():
    truth = _known()
    train, holdout = _synthetic(truth, 800, 1), _synthetic(truth, 300, 2)
    fitted = tg.fit_weights(train)
    assert tg.pairwise_accuracy(fitted, holdout) >= 0.9
    top3 = lambda w: sorted(_effective(w), key=_effective(w).get, reverse=True)[:3]
    assert set(top3(fitted)) == set(top3(truth))


def test_weights_nonnegative_and_normalised():
    # labels that push some metrics' weights negative must be clamped to 0
    rng = np.random.default_rng(3)
    examples = []
    for _ in range(200):
        a = {k: float(rng.random()) for k in NAMES}
        b = {k: float(rng.random()) for k in NAMES}
        examples.append((a, b, 1.0 if a[NAMES[0]] < b[NAMES[0]] + a[NAMES[1]] - b[NAMES[1]] else 0.0))
    w = tg.fit_weights(examples)
    assert all(v >= 0 for v in w.geometry.values()) and all(v >= 0 for v in w.pixels.values())
    assert sum(w.geometry.values()) == pytest.approx(1.0)
    assert sum(w.pixels.values()) == pytest.approx(1.0)
    assert 0.0 <= w.geometry_share <= 1.0


def test_fit_is_deterministic():
    ex = _synthetic(_known(), 100, 4)
    assert tg.fit_weights(ex, seed=5) == tg.fit_weights(ex, seed=5)


def _row(cell, cid, split, task="t1", **metrics):
    m = {k: 0.0 for k in NAMES}
    m.update(metrics)
    return {"cell": cell, "task_id": task, "split": split, "candidate_id": cid, "metrics": m}


def _judge(cell, a, b, winner, task="t1", split="train", swapped=False):
    return {"cell": cell, "a": a, "b": b, "winner": winner, "task_id": task, "split": split,
            "teacher": "fake", "swapped": swapped}


def test_ties_dropped():
    pool = [_row("c", "x", "train"), _row("c", "y", "train")]
    train, holdout = tg.split_examples(pool, [_judge("c", "x", "y", "tie"), _judge("c", "x", "y", "a")])
    assert len(train) == 1 and train[0][2] == 1.0 and holdout == []
    train, _ = tg.split_examples(pool, [_judge("c", "x", "y", "b")])
    assert train[0][2] == 0.0


def test_empty_raises():
    with pytest.raises(ValueError):
        tg.fit_weights([])


def test_holdout_never_in_train():
    pool = [_row("c", "x", "train"), _row("c", "y", "train"),
            _row("d", "p", "holdout", task="t2"), _row("d", "q", "holdout", task="t2")]
    judgments = [_judge("c", "x", "y", "a"),
                 _judge("d", "p", "q", "a", task="t2", split="train")]  # mislabelled split must not leak
    train, holdout = tg.split_examples(pool, judgments)
    assert len(train) == 1 and len(holdout) == 1


def test_missing_candidates_skipped():
    pool = [_row("c", "x", "train"), _row("c", "y", "train")]
    train, holdout = tg.split_examples(pool, [_judge("c", "x", "gone", "a")])
    assert train == [] and holdout == []


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _dataset(tmp_path, truth, n_train=60, n_hold=40):
    """Pool + judgements in files, labelled by `truth` (single metric per ad cell)."""
    pool, judgments = [], []
    rng = np.random.default_rng(7)
    for i in range(n_train + n_hold):
        split = "train" if i < n_train else "holdout"
        cell = f"cell{i}"
        ids = []
        for j in "ab":
            m = {k: float(rng.random()) for k in NAMES}
            pool.append({"cell": cell, "task_id": f"t{i}", "split": split, "candidate_id": f"{cell}{j}", "metrics": m})
            ids.append(m)
        win = "a" if _total(truth, ids[0]) > _total(truth, ids[1]) else "b"
        judgments.append(_judge(cell, f"{cell}a", f"{cell}b", win, task=f"t{i}", split=split))
    _write_jsonl(tmp_path / "pool.jsonl", pool)
    _write_jsonl(tmp_path / "judgments.jsonl", judgments)


def test_write_refused_without_gain(tmp_path, monkeypatch, capsys):
    # Judgements made by the default weights: learning cannot beat them by 0.03.
    out = tmp_path / "weights.json"
    monkeypatch.setenv("CHAMELEON_GRADER_WEIGHTS", str(out))
    _dataset(tmp_path, DEFAULT_WEIGHTS)
    code = tg.main(["--rubric-version", "any", "--pool", str(tmp_path / "pool.jsonl"), "--judgments", str(tmp_path / "judgments.jsonl"),
                    "--report", str(tmp_path / "report.md"), "--write"])
    assert code == 0 and not out.exists()
    assert "refus" in capsys.readouterr().out.lower()
    assert (tmp_path / "report.md").exists()


def test_write_when_gain(tmp_path, monkeypatch):
    out = tmp_path / "weights.json"
    monkeypatch.setenv("CHAMELEON_GRADER_WEIGHTS", str(out))
    _dataset(tmp_path, _known(), n_train=400, n_hold=200)
    code = tg.main(["--rubric-version", "any", "--pool", str(tmp_path / "pool.jsonl"), "--judgments", str(tmp_path / "judgments.jsonl"),
                    "--report", str(tmp_path / "report.md"), "--write"])
    assert code == 0 and out.exists()
    data = json.loads(out.read_text())
    assert data["source"].startswith("learned:") and data["source"].endswith("-400")
    from engine.pipeline.grading.weights import load_weights
    assert load_weights(out).geometry_share == pytest.approx(data["geometry_share"])


def test_teacher_filter_and_per_teacher_counts(tmp_path):
    _dataset(tmp_path, _known(), n_train=30, n_hold=20)
    rows = [json.loads(x) for x in (tmp_path / "judgments.jsonl").read_text().splitlines()]
    for i, r in enumerate(rows):
        r["teacher"] = "one" if i % 2 else "two"
    _write_jsonl(tmp_path / "judgments.jsonl", rows)
    common = ["--rubric-version", "any", "--pool", str(tmp_path / "pool.jsonl"), "--judgments", str(tmp_path / "judgments.jsonl")]
    tg.main(common + ["--report", str(tmp_path / "all.md")])
    text = (tmp_path / "all.md").read_text()
    assert "teachers: one: 25, two: 25" in text and "examples: train 30, holdout 20" in text
    tg.main(common + ["--teacher", "one", "--report", str(tmp_path / "one.md")])
    text = (tmp_path / "one.md").read_text()
    assert "teachers: one: 25\n" in text and "two" not in text


def test_report_breaks_holdout_down_by_brand(tmp_path):
    _dataset(tmp_path, _known(), n_train=40, n_hold=30)
    rows = [json.loads(x) for x in (tmp_path / "pool.jsonl").read_text().splitlines()]
    for r in rows:  # holdout ads t40..t69: the first 20 solstice, the last 10 offbrand
        r["brand"] = "offbrand" if int(r["task_id"][1:]) >= 60 else "solstice"
    _write_jsonl(tmp_path / "pool.jsonl", rows)
    tg.main(["--rubric-version", "any", "--pool", str(tmp_path / "pool.jsonl"), "--judgments", str(tmp_path / "judgments.jsonl"),
             "--report", str(tmp_path / "r.md")])
    text = (tmp_path / "r.md").read_text()
    assert "| holdout brand | examples | default | learned |" in text
    assert "| solstice | 20 |" in text and "| offbrand | 10 |" in text


def test_holdout_by_brand_defaults_unknown():
    pool = [_row("d", "p", "holdout"), _row("d", "q", "holdout")]
    groups = tg.holdout_by_brand(pool, [_judge("d", "p", "q", "a")])
    assert list(groups) == ["unknown"] and len(groups["unknown"]) == 1


def test_select_judgments_filters_teacher_confidence_and_rubric_version():
    from dev.distill.train_grader import select_judgments
    rows = [{"teacher": "t1", "confidence": 3, "rubric_version": "v2"},
            {"teacher": "t1", "confidence": 1, "rubric_version": "v2"},   # a guess: dropped
            {"teacher": "t1", "confidence": 2, "rubric_version": "v1"},   # older rubric: dropped
            {"teacher": "t1", "confidence": 2},                            # no version: dropped
            {"teacher": "t2", "confidence": 3, "rubric_version": "v2"}]   # other teacher: dropped
    kept, dropped = select_judgments(rows, teacher="t1", min_confidence=2, rubric_version="v2")
    assert kept == [rows[0]]
    assert dropped == {"teacher": 1, "low_confidence": 1, "rubric_version": 2}
    kept_any, _ = select_judgments(rows, min_confidence=1, rubric_version=None)
    assert kept_any == rows


def test_report_has_a_json_twin(tmp_path):
    _dataset(tmp_path, _known(), n_train=30, n_hold=20)
    tg.main(["--rubric-version", "any", "--pool", str(tmp_path / "pool.jsonl"), "--judgments",
             str(tmp_path / "judgments.jsonl"), "--report", str(tmp_path / "r.md")])
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["examples"]["train"] == 30 and data["examples"]["holdout"] == 20
    assert {"default_holdout", "learned_holdout"} <= set(data["accuracy"])
    assert abs(sum(v for k, v in data["weights"]["learned"].items() if k != "geometry_share") - 2) < 1e-6
