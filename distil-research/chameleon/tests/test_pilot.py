"""Pilot tooling: blind human sheets and the agreement score. No teacher is called."""

import csv
import json

from PIL import Image

import random

from dev.distill.paths import RENDERS_SUBDIR
from dev.distill.pilot import judge_pilot, make_pilot, pick_pilot_pairs, score_pilot


def _row(cell, cid, task="t1"):
    return {"cell": cell, "task_id": task, "split": "train", "target": "300x250", "candidate_id": cid,
            "png": f"renders/{task}/{cid}.png", "is_fallback": False, "metrics": {}}


def _pool(tmp_path, n):
    for p in [f"{RENDERS_SUBDIR}/t1/source.png"] + [f"renders/t1/{c}{i}.png" for i in range(n) for c in "ab"]:
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 40), "white").save(tmp_path / p)
    return [(_row("c1", f"a{i}"), _row("c1", f"b{i}")) for i in range(n)]


def test_make_pilot_writes_pairs_sheets_and_blank_csv(tmp_path):
    pairs = _pool(tmp_path, 3)
    out = tmp_path / "pilot"
    make_pilot(pairs, out, tmp_path)
    meta = json.loads((out / "pairs.json").read_text())
    assert len(meta) == 3
    for pid, m in meta.items():
        assert {m["a"], m["b"]} <= {f"{c}{i}" for c in "ab" for i in range(3)} and m["cell"] == "c1"
        assert (out / "sheets" / f"{pid}.png").exists()
    rows = list(csv.DictReader((out / "human_answers.csv").open()))
    assert [r["pair_id"] for r in rows] == list(meta) and all(r["winner"] == "" for r in rows)


def test_score_pilot_matches_across_display_order(tmp_path):
    pairs = _pool(tmp_path, 2)
    out = tmp_path / "pilot"
    make_pilot(pairs, out, tmp_path)
    meta = json.loads((out / "pairs.json").read_text())
    ids = list(meta)
    with (out / "human_answers.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "winner"])
        w.writerow([ids[0], "a"])
        w.writerow([ids[1], "a"])
    m0, m1 = meta[ids[0]], meta[ids[1]]
    judgments = [  # teacher saw pair 0 in the opposite order and picked the same candidate; disagrees on pair 1
        {"cell": "c1", "a": m0["b"], "b": m0["a"], "winner": "b", "swapped": False, "teacher": "t"},
        {"cell": "c1", "a": m0["a"], "b": m0["b"], "winner": "a", "swapped": True, "teacher": "t"},
        {"cell": "c1", "a": m1["a"], "b": m1["b"], "winner": "b", "swapped": False, "teacher": "t"},
        {"cell": "c1", "a": m1["a"], "b": m1["b"], "winner": "a", "swapped": False, "teacher": "other"},
    ]
    res = score_pilot(out, judgments, "t")
    assert res["n"] == 2 and res["agree"] == 1
    assert res["position_bias"] == 1 / 3 and res["swap_agreement"] == 1.0
    text = (out / "RESULTS.md").read_text()
    assert "1 / 2" in text and "FAIL" in text


KEYS = ["hierarchy", "legibility", "overlap", "degradation", "faithfulness", "space_use", "alignment",
        "spacing", "balance", "text_contrast", "busyness", "placeholder"]


def _multi_pool(tmp_path, cells=6, per_cell=8, holdout_cells=1):
    rng = random.Random(0)
    rows = []
    for c in range(cells):
        split = "holdout" if c < holdout_cells else "train"
        for i in range(per_cell):
            r = _row(f"cell{c}", f"c{c}x{i}")
            r.update(split=split, metrics={k: rng.random() for k in KEYS})
            rows.append(r)
    for p in [f"{RENDERS_SUBDIR}/t1/source.png"] + [r["png"] for r in rows]:
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 40), "white").save(tmp_path / p)
    return rows


def test_pilot_pairs_span_cells_and_only_train(tmp_path):
    rows = _multi_pool(tmp_path)
    pairs = pick_pilot_pairs(rows, 8, seed=1)
    assert len(pairs) == 8
    cells = [a["cell"] for a, _ in pairs]
    assert len(set(cells)) >= 4 and max(cells.count(c) for c in set(cells)) <= 2
    assert all(a["split"] == b["split"] == "train" for a, b in pairs)
    assert pairs == pick_pilot_pairs(rows, 8, seed=1)


class AlwaysA:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def ask(self, prompt, schema, images, validate=None):
        self.calls += 1
        n = schema["properties"]["verdicts"]["maxItems"]
        return {"verdicts": [{"pair": i + 1, "winner": "a", "confidence": 2, "reason": "r"} for i in range(n)]}


def test_make_judge_score_round_trip(tmp_path):
    rows = _multi_pool(tmp_path)
    out = tmp_path / "pilot"
    make_pilot(pick_pilot_pairs(rows, 6, seed=0), out, tmp_path)
    meta = json.loads((out / "pairs.json").read_text())
    teacher = AlwaysA()
    added = judge_pilot(out, rows, teacher, tmp_path, pairs_per_call=4)
    assert len(added) == 12  # every pilot pair, plus every one re-asked swapped
    assert sum(j["swapped"] for j in added) == 6
    stored = [json.loads(x) for x in (out / "teacher_judgments.jsonl").read_text().splitlines()]
    assert stored == added
    assert {(j["cell"], frozenset((j["a"], j["b"]))) for j in stored} == \
        {(m["cell"], frozenset((m["a"], m["b"]))) for m in meta.values()}
    with (out / "human_answers.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "winner"])
        w.writerows([pid, "a"] for pid in meta)
    res = score_pilot(out, stored, teacher.name)
    assert res["n"] == 6
    assert res["swap_agreement"] == 0.0  # AlwaysA contradicts itself when swapped
    calls = teacher.calls
    assert judge_pilot(out, rows, teacher, tmp_path) == [] and teacher.calls == calls  # rerun resumes


def test_blank_or_short_csv_rows_are_not_judged(tmp_path):
    pairs = _pool(tmp_path, 3)
    out = tmp_path / "pilot"
    make_pilot(pairs, out, tmp_path)
    ids = list(json.loads((out / "pairs.json").read_text()))
    (out / "human_answers.csv").write_text(f"pair_id,winner\n{ids[0]},\n{ids[1]}\n\n{ids[2]},tie\n")
    m = json.loads((out / "pairs.json").read_text())[ids[2]]
    judgments = [{"cell": m["cell"], "a": m["a"], "b": m["b"], "winner": "tie", "swapped": False, "teacher": "t"}]
    assert score_pilot(out, judgments, "t")["n"] == 1  # only the row with an answer counts
