"""Train the learned picker (CHAMELEON_PICKER=learned) on the picker studies'
teacher rankings, and check the engine computes the same features.

The model is pairwise logistic: for every pair of shortlisted candidates in
a cell, the teacher-better one should score higher; pairs are weighted by
1 + the teacher's score gap. Features are what the engine has at pick time
(engine/pipeline/grading/learned_picker.py `feature_vector`): the grader's
metrics in sorted key order, the relation score, the family one-hot, the
grader's total, the target's log aspect and family x log aspect, each
standardised (mu, sd) on the training cells.

Data: training-data/picker/{pk_v4_train,pk_v4_train2,pk_v4_crd,pk_v4_crd2}
(train + crello_dev, deduplicated by cell) for training; pk_v4_hold
(holdout) is reported only, never trained on.

    cd distil-research/chameleon
    python -m dev.distill.train_picker train [--lam 3] [--write]   # grouped 5-fold CV + holdout; --write saves the engine weights
    python -m dev.distill.train_picker parity [--workers 3]        # rebuild every ranked cell's shortlist in the engine

`parity` reruns the engine (the studies' env plus CHAMELEON_PICKER=learned)
on every ranked cell and reports: does the shortlist equal the stored one,
the max abs difference between the engine's features and the stored ones,
whether the engine's learned pick equals the model's argmax over the stored
features, and the picker's extra seconds per cell.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dev.distill import ab_engine
from dev.distill.paths import DATA_ROOT
from engine.pipeline.grading import learned_picker as lp

PICKER_ROOT = DATA_ROOT / "picker"
TRAIN_STUDIES = ("pk_v4_train", "pk_v4_crd", "pk_v4_train2", "pk_v4_crd2")
HOLDOUT_STUDY = "pk_v4_hold"
LAM = 3.0
LAMS = (0.3, 1, 3, 10, 30)
TARGETS = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}


def load(name: str, root: Path = PICKER_ROOT) -> list[dict]:
    out = []
    for f in (Path(root) / name / "rankings.jsonl", Path(root) / name / "private.nosync" / "rankings.jsonl"):
        if f.exists():
            out += ab_engine._read_jsonl(f)
    return out


def log_aspect(target: str) -> float:
    t = TARGETS[target]
    return math.log(t["width"] / t["height"])


def stored_features(c: dict, target: str) -> list[float]:
    """The model input of one stored candidate (a rankings.jsonl candidate)."""
    f = c["features"]
    return lp.feature_vector(f["metrics"], f.get("relation", 1.0), c["family"], c["grader_total"], log_aspect(target))


def stored_names(c: dict) -> list[str]:
    return lp.feature_names(c["features"]["metrics"])


def cellset(rows: list[dict]) -> list[dict]:
    import numpy as np
    return [{"cell": r["cell"], "task": r["task_id"],
             "X": np.array([stored_features(c, r["target"]) for c in r["candidates"]], float),
             "rank": np.array([c["teacher_rank"] for c in r["candidates"]]),
             "score": np.array([c["teacher_score"] for c in r["candidates"]], float),
             "pick": np.array([c["is_pick"] for c in r["candidates"]])} for r in rows]


def fit(cells: list[dict], lam: float = LAM, iters: int = 3000, lr: float = 0.05):
    import numpy as np
    all_x = np.vstack([c["X"] for c in cells])
    mu, sd = all_x.mean(0), all_x.std(0) + 1e-6
    diffs, wts = [], []
    for c in cells:
        z = (c["X"] - mu) / sd
        for i in range(len(z)):
            for j in range(len(z)):
                if c["rank"][i] < c["rank"][j]:
                    diffs.append(z[i] - z[j])
                    wts.append(1.0 + abs(c["score"][i] - c["score"][j]))  # bigger score gaps weigh more
    d, wts = np.array(diffs), np.array(wts)
    w = np.zeros(d.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-d @ w))
        g = -(d * ((1 - p) * wts)[:, None]).mean(0) + lam * w / len(d)
        w -= lr * g
    return w, mu, sd


def evaluate(cells: list[dict], choose) -> tuple[float, float, float]:
    """(top-1 agreement with the teacher, teacher's mean score of the pick, oracle mean score)."""
    top = sc = best = 0.0
    for c in cells:
        i = choose(c)
        top += c["rank"][i] == 1
        sc += c["score"][i]
        best += c["score"].max()
    n = len(cells)
    return top / n, sc / n, best / n


def grader(c: dict) -> int:
    import numpy as np
    return int(np.argmax(c["pick"]))


def model(w, mu, sd):
    import numpy as np
    return lambda c: int(np.argmax(((c["X"] - mu) / sd) @ w))


def training_rows(root: Path = PICKER_ROOT) -> list[dict]:
    rows = {}
    for name in TRAIN_STUDIES:
        for r in load(name, root):
            rows[r["cell"]] = r
    return list(rows.values())


def train(lam: float = LAM, write: bool = False, root: Path = PICKER_ROOT, out: Path = lp.WEIGHTS_PATH) -> dict:
    tr_rows, ho_rows = training_rows(root), load(HOLDOUT_STUDY, root)
    tr, ho = cellset(tr_rows), cellset(ho_rows)
    print(f"cells train+crello_dev {len(tr)}, holdout {len(ho)}")
    tasks = sorted({c["task"] for c in tr})
    random.Random(0).shuffle(tasks)
    for lam_ in LAMS:  # grouped 5-fold CV by task: no ad in both train and test folds
        res = []
        for k in range(5):
            te = set(tasks[k::5])
            a, b = [c for c in tr if c["task"] not in te], [c for c in tr if c["task"] in te]
            w, mu, sd = fit(a, lam_)
            res.append((evaluate(b, model(w, mu, sd)), evaluate(b, grader), len(b)))
        n = sum(r[2] for r in res)
        agg = lambda j, t: sum(r[j][t] * r[2] for r in res) / n
        print(f"lam {lam_:5}: CV model top1 {agg(0, 0):.0%} score {agg(0, 1):.2f} | "
              f"grader top1 {agg(1, 0):.0%} score {agg(1, 1):.2f} | oracle {agg(0, 2):.2f}")
    w, mu, sd = fit(tr, lam)
    m, g = evaluate(ho, model(w, mu, sd)), evaluate(ho, grader)
    print(f"HOLDOUT (lam {lam}): model top1 {m[0]:.0%} mean score {m[1]:.2f} | "
          f"grader top1 {g[0]:.0%} mean score {g[1]:.2f} | oracle {m[2]:.2f}")
    names = stored_names(tr_rows[0]["candidates"][0])
    assert all(stored_names(c) == names for r in tr_rows + ho_rows for c in r["candidates"]), "feature sets differ"
    for i in sorted(range(len(w)), key=lambda i: -abs(w[i]))[:12]:
        print(f"   {names[i]:28s} {w[i]:+.2f}")
    result = {"names": names, "w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "lam": lam,
              "trained_on": list(TRAIN_STUDIES), "cells": len(tr)}
    if write:
        Path(out).write_text(json.dumps(result) + "\n")
        print(f"wrote {out}")
    return result


# ---------------------------------------------------------------- parity

def _parity_cell(job: dict) -> dict:
    from dev.distill.picker_study import capture_pick
    row = job["row"]
    canvas = {"width": job["target"]["width"], "height": job["target"]["height"]}
    t0 = time.perf_counter()
    with capture_pick() as box:
        ab_engine._resize(job["source"], canvas)
    seconds = time.perf_counter() - t0
    last = dict(lp.LAST)
    stored_ids = [c["candidate_id"] for c in row["candidates"]]
    out = {"cell": row["cell"], "seconds": seconds, "picker_seconds": last.get("seconds"),
           "grader_pick_same": last.get("grader_pick") == row["grader_pick"],
           "shortlist_same": last.get("shortlist") == stored_ids, "engine_pick": box["winner"].candidate_id,
           "last_pick": last.get("pick")}
    if out["shortlist_same"] and "X" in last:
        stored = [stored_features(c, row["target"]) for c in row["candidates"]]
        out["max_abs_diff"] = max(abs(a - b) for x, y in zip(last["X"], stored) for a, b in zip(x, y))
        scores = lp.model_scores(lp.load_model(), stored)
        out["offline_pick"] = stored_ids[max(range(len(scores)), key=lambda j: (scores[j], -j))]
    return out


def _safe_parity_cell(job: dict) -> dict:
    try:
        return _parity_cell(job)
    except Exception as exc:
        return {"cell": job["row"]["cell"], "error": f"{type(exc).__name__}: {exc}"}


def parity(workers: int = 3, root: Path = PICKER_ROOT, limit: int | None = None) -> list[dict]:
    by_split: dict[str, list[dict]] = {}
    envs = {}
    for name in (*TRAIN_STUDIES, HOLDOUT_STUDY):
        meta = json.loads((Path(root) / name / "study.json").read_text())
        envs.setdefault(meta["split"], meta["env"])
        assert envs[meta["split"]] == meta["env"], "studies of one split ran different engines"
        seen = {r["cell"] for r in by_split.get(meta["split"], [])}
        by_split.setdefault(meta["split"], []).extend(r for r in load(name, root) if r["cell"] not in seen)
    results = []
    for split, rows in by_split.items():
        rows = rows[:limit] if limit else rows
        _, designs, folders = ab_engine.split_cells(split)
        jobs = [{"row": r, "source": designs[r["task_id"]]["source"], "target": TARGETS[r["target"]]} for r in rows]
        env = {**envs[split], "CHAMELEON_PICKER": "learned"}
        with ProcessPoolExecutor(max_workers=min(workers, 3), initializer=ab_engine._worker_init,
                                 initargs=(env, folders)) as ex:
            got = list(ex.map(_safe_parity_cell, jobs))
        print(f"[parity] {split}: {len(got)} cells")
        results += got
    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    full = [r for r in ok if "max_abs_diff" in r]
    diffs = [r["max_abs_diff"] for r in full]
    print(f"cells {len(results)}, errors {len(errors)}")
    for r in errors[:5]:
        print("   error", r["cell"], r["error"])
    print(f"grader pick same as the study: {sum(r['grader_pick_same'] for r in ok)}/{len(ok)}")
    print(f"shortlist same as the study:   {sum(r['shortlist_same'] for r in ok)}/{len(ok)}")
    if diffs:
        print(f"features: max abs diff {max(diffs):.3g} (median {sorted(diffs)[len(diffs) // 2]:.3g}) over {len(full)} cells")
    print(f"engine's learned pick == offline argmax: {sum(r['engine_pick'] == r['offline_pick'] for r in full)}/{len(full)}")
    print(f"picker returned what pick_best returned:  {sum(r['engine_pick'] == r['last_pick'] for r in ok)}/{len(ok)}")
    ps = sorted(r["picker_seconds"] for r in ok if r["picker_seconds"] is not None)
    if ps:
        print(f"picker extra seconds per cell: mean {sum(ps) / len(ps):.3f}, median {ps[len(ps) // 2]:.3f}, "
              f"max {ps[-1]:.3f}; engine seconds per cell mean {sum(r['seconds'] for r in ok) / len(ok):.2f}")
    bad = [r for r in ok if not r["shortlist_same"] or not r["grader_pick_same"]]
    for r in bad[:10]:
        print("   differs:", r["cell"], "grader_same", r["grader_pick_same"], "shortlist_same", r["shortlist_same"])
    print("families picked:", Counter(r["engine_pick"].split("/")[0] for r in ok).most_common())
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    t = sub.add_parser("train")
    t.add_argument("--lam", type=float, default=LAM)
    t.add_argument("--write", action="store_true", help=f"save the model to {lp.WEIGHTS_PATH.name}")
    p = sub.add_parser("parity")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="at most this many cells per split")
    args = ap.parse_args(argv)
    if args.step == "train":
        train(args.lam, args.write)
    else:
        parity(args.workers, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
