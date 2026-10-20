"""One JSON snapshot of the distillation pipeline's progress, for the dashboard
(distil-research/dashboard/). Read-only: it never changes data or touches
running workers.

    python -m dev.distill.status            # print the snapshot
"""

import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from dev.distill.corpus import load_corpus
from dev.distill.judge import (RUBRIC_VERSION, _in_shard, _plan, pick_cells, position_bias, shard_file,
                               swap_agreement)
from dev.distill.paths import DATA_ROOT, JUDGMENTS_FILE, POOL_DIR
from dev.distill.pool import sample_pairs

# The main judging run's parameters (training-data/judgments/README.md).
RUN = {"cells": 150, "per_cell": 4, "swap_share": 0.2, "seed": 0, "shards": 16}
TRAINING_JSON = DATA_ROOT / "judgments" / "training-report.json"
_pool_cache: dict = {}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a worker may be mid-write
    return out


def _pool():
    """Pool rows plus the run's expected items, cached until pool.jsonl changes."""
    path = POOL_DIR / "pool.jsonl"
    if not path.exists():
        return [], []
    key = path.stat().st_mtime_ns
    if _pool_cache.get("key") != key:
        rows = _jsonl(path)
        chosen = pick_cells(rows, RUN["cells"], RUN["seed"])
        pairs = sample_pairs([r for r in rows if r["cell"] in chosen], per_cell=RUN["per_cell"], seed=RUN["seed"])
        _pool_cache.update(key=key, rows=rows, plan=_plan(pairs, RUN["swap_share"], RUN["seed"]))
    return _pool_cache["rows"], _pool_cache["plan"]


def _workers() -> int:
    try:
        out = subprocess.run(["pgrep", "-f", "dev.distill.judge --cells"], capture_output=True, text=True).stdout
        return len(out.split())
    except OSError:
        return 0


def _events(limit: int = 15) -> list[dict]:
    try:
        out = subprocess.run(["git", "log", f"-{limit}", "--format=%h|%cI|%s", "--", "."], capture_output=True,
                             text=True, cwd=DATA_ROOT.parent).stdout
    except OSError:
        return []
    return [dict(zip(("sha", "time", "subject"), line.split("|", 2))) for line in out.splitlines() if line.count("|") >= 2]


def snapshot() -> dict:
    corpus = load_corpus()
    rows, plan = _pool()
    brand_of = {r["task_id"]: r.get("brand", "unknown") for r in rows}
    cells = {r["cell"]: (r["split"], r.get("brand", "unknown")) for r in rows}
    files = [JUDGMENTS_FILE] + [shard_file(JUDGMENTS_FILE, k, RUN["shards"]) for k in range(RUN["shards"])]
    per_file = {f: [j for j in _jsonl(f) if j.get("rubric_version") == RUBRIC_VERSION] for f in files}
    judged = [j for js in per_file.values() for j in js]
    done_keys = {(j["cell"], frozenset((j["a"], j["b"])), j["swapped"]) for j in judged}

    expected = Counter()
    done = Counter()
    shard_expected, shard_done = Counter(), Counter()
    for a, b, swapped in plan:
        group = f'{a["split"]}/{a.get("brand", "unknown")}'
        k = next(k for k in range(RUN["shards"]) if _in_shard(a["cell"], (k, RUN["shards"])))
        expected[group] += 1
        shard_expected[k] += 1
        if (a["cell"], frozenset((a["candidate_id"], b["candidate_id"])), swapped) in done_keys:
            done[group] += 1
            shard_done[k] += 1

    best = {"solstice": Counter(), "offbrand": Counter()}
    for j in judged:
        brand = brand_of.get(j.get("task_id"), "unknown")
        if brand in best:
            best[brand][max(j["score_a"], j["score_b"])] += 1

    recent = []
    for f, js in per_file.items():
        recent += js[-2:]
    recent = [{k: j[k] for k in ("cell", "winner", "confidence", "score_a", "score_b", "reason")} for j in recent[-10:]]

    training = json.loads(TRAINING_JSON.read_text()) if TRAINING_JSON.exists() else None
    return {
        "generated_at": time.time(),
        "rubric_version": RUBRIC_VERSION,
        "corpus": {"total": len(corpus),
                   "by_group": dict(Counter(f'{d["split"]}/{d.get("brand", "solstice")}' for d in corpus)),
                   "by_origin": dict(Counter(d["origin"] for d in corpus))},
        "pool": {"cells": len(cells), "candidates": len(rows),
                 "by_group": dict(Counter(f"{split}/{brand}" for split, brand in cells.values()))},
        "judging": {
            "expected": sum(expected.values()), "done": sum(done.values()), "workers": _workers(),
            "by_group": {g: {"done": done[g], "expected": expected[g]} for g in sorted(expected)},
            "shards": [{"shard": k, "done": shard_done[k], "expected": shard_expected[k]} for k in range(RUN["shards"])],
            "confidence": {str(c): n for c, n in sorted(Counter(j["confidence"] for j in judged).items())},
            "best_score": {b: {str(s): best[b][s] for s in range(1, 6)} for b in best},
            "position_bias": position_bias(judged) if judged else None,
            "swap_agreement": swap_agreement(judged) if judged else None,
            "recent": recent,
        },
        "training": training,
        "events": _events(),
    }


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=1))
