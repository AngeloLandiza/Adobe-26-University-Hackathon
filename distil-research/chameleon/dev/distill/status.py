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


RUN_LABELS = {"judgments/judgments.jsonl": "Main judging run (grader training data)",
              "judgments/d2/judgments.jsonl": "D2: learned vs hand-set picks",
              "floor-plans/judgments.jsonl": "B2: teacher floor plans vs current winner",
              "floor-plans/proposals.jsonl": "B2: floor-plan proposals",
              "floor-plans/builds.jsonl": "B2: proposals built into layouts",
              "floor-plans/eval-builds.jsonl": "B2 holdout check: layouts built from new templates",
              "floor-plans/eval-judgments.jsonl": "B2 holdout check: new templates vs current winner",
              "floor-plans/retrieval-builds.jsonl": "Retrieval: per-ad plans built (holdout)",
              "floor-plans/retrieval-judgments.jsonl": "Retrieval: per-ad plans vs current winner"}


def _dynamic_labels() -> dict:
    """A/B engine runs live in training-data/ab/<variant>/ and ab/<A>__vs__<B>/: list whatever exists."""
    out = {}
    for d in sorted((DATA_ROOT / "ab").glob("*/")):
        if "__vs__" in d.name:
            out[f"ab/{d.name}/judgments.jsonl"] = f"A/B judging: {d.name.replace('__vs__', ' vs ')}"
        else:
            out[f"ab/{d.name}/winners.jsonl"] = f"A/B engine run: {d.name} (winners built)"
    return out


def _runs() -> list[dict]:
    """Every judgement-style file under training-data (shards counted in),
    current rubric only, so new runs show up without dashboard changes."""
    runs = []
    for rel, label in {**RUN_LABELS, **_dynamic_labels()}.items():
        main = DATA_ROOT / rel
        files = [main, *sorted(main.parent.glob(f"{main.stem}.shard*of*{main.suffix}"))]
        rows = [r for f in files for r in _jsonl(f) if r.get("rubric_version", RUBRIC_VERSION) == RUBRIC_VERSION]
        if rows or any(f.exists() for f in files):
            newest = max((f.stat().st_mtime for f in files if f.exists()), default=None)
            runs.append({"name": label, "rows": len(rows), "shards": len(files) - 1, "updated": newest})
    return runs


def _reports() -> list[dict]:
    """Experiment logs and run reports, newest first: title, decision line, short excerpt."""
    research = DATA_ROOT.parent
    paths = sorted((research / "docs" / "attention" / "experiments").glob("20*.md"))
    paths += [p for p in (DATA_ROOT / "judgments" / "d2" / "REPORT.md", DATA_ROOT / "floor-plans" / "REPORT.md",
                          DATA_ROOT / "floor-plans" / "RETRIEVAL.md") if p.exists()]
    paths += sorted((DATA_ROOT / "ab").glob("*__vs__*/REPORT.md"))
    out = []
    for path in paths:
        text = path.read_text()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        title = next((l.lstrip("# ") for l in lines if l.startswith("#")), path.stem)
        decision = ""
        for i, l in enumerate(lines):
            if l.lower().startswith("## decision"):
                decision = next((x for x in lines[i + 1:] if not x.startswith("#")), "")
                break
        body = [l for l in lines if not l.startswith("#") and not l.startswith("```")]
        out.append({"title": title, "path": str(path.relative_to(research)), "updated": path.stat().st_mtime,
                    "decision": decision[:300], "excerpt": "\n".join(body[:8])[:900]})
    return sorted(out, key=lambda r: -r["updated"])


def _active() -> list[dict]:
    """Long jobs running right now, from the process table: one entry per
    module + step, with its worker count and the first worker's flags."""
    try:
        out = subprocess.run(["pgrep", "-fl", r"dev\.(distill|attention)\."], capture_output=True, text=True).stdout
    except OSError:
        return []
    jobs: dict[tuple, dict] = {}
    for line in out.splitlines():
        args = line.split()
        if len(args) < 3 or "ython" not in Path(args[1]).name or "-m" not in args:
            continue  # only real Python workers, not shells whose script text mentions the module
        i = args.index("-m")
        module = args[i + 1].rsplit(".", 1)[-1] if i + 1 < len(args) else "?"
        if module in ("publish_status", "status"):
            continue
        rest = args[i + 2:]
        step = rest[0] if rest and not rest[0].startswith("-") else ""
        flags = {rest[k][2:]: rest[k + 1] for k in range(len(rest) - 1) if rest[k].startswith("--") and not rest[k + 1].startswith("--")}
        job = jobs.setdefault((module, step), {"job": f"{module} {step}".strip(), "workers": 0, "flags": flags})
        job["workers"] += 1
    return sorted(jobs.values(), key=lambda j: -j["workers"])


_b2_expected: dict = {}


def _b2(active: list[dict]) -> dict | None:
    """Progress of the floor-plan pipeline (B2), stage by stage."""
    from dev.distill import floorplans as fp
    if not any(f.exists() or list(f.parent.glob(f"{f.stem}.shard*")) for f in (fp.PROPOSALS_FILE, fp.BUILDS_FILE)):
        return None
    running = [j for j in active if j["job"].startswith("floorplans")]
    for j in running:
        if j["job"] == "floorplans propose" and "cells" in j["flags"]:
            _b2_expected["cells"] = int(j["flags"]["cells"])
    proposals = fp.read_jsonl(fp.PROPOSALS_FILE)
    builds = fp.read_jsonl(fp.BUILDS_FILE)
    judged = [j for j in fp.read_jsonl(fp.JUDGMENTS_FILE) if j.get("rubric_version", RUBRIC_VERSION) == RUBRIC_VERSION]
    built = sum(1 for b in builds if b.get("built"))
    eval_builds = fp.read_jsonl(fp.EVAL_BUILDS_FILE)
    eval_judged = fp.read_jsonl(fp.EVAL_JUDGMENTS_FILE)
    clusters = json.loads(fp.CLUSTERS_FILE.read_text()) if fp.CLUSTERS_FILE.exists() else None
    summary = json.loads(fp.EVAL_SUMMARY_FILE.read_text()) if fp.EVAL_SUMMARY_FILE.exists() else None
    cells = len({p["cell"] for p in proposals})
    return {
        "running": [j["job"] + f" x{j['workers']}" for j in running],
        "stages": [
            {"name": "Cells with proposals", "done": cells,
             "expected": _b2_expected.get("cells") or (cells if not any(j["job"] == "floorplans propose" for j in running) else None)},
            {"name": "Proposals built into layouts", "done": len(builds), "expected": len(proposals) or None,
             "note": f"{built} built, {len(builds) - built} could not build"},
            {"name": "Built layouts judged vs current winner", "done": len(judged),
             "expected": round(built * 1.2) or None, "note": "includes swapped re-asks (~20%)"},
            {"name": "Holdout check: layouts built", "done": len(eval_builds), "expected": None},
            {"name": "Holdout check: judged", "done": len(eval_judged), "expected": None},
        ],
        "clusters": clusters, "eval_summary": summary,
    }


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
        "runs": _runs(),
        "active": (active := _active()),
        "b2": _b2(active),
        "reports": _reports(),
    }


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=1))
