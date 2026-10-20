"""Relation features for every judged candidate (experiment E1).

    python -m dev.attention.features_dataset [--judgments training-data/judgments/judgments.jsonl]
        [--out training-data/attention/relation_features.jsonl] [--workers 4]

The candidate pool stores only each layout's 12 metrics, not the layout. So
for every judged cell this rebuilds the candidates exactly as
dev/distill/pool.py does (heuristic plan over the merged assets folder,
safe_fallback first, then generate_candidates), keeps the judged ones by
candidate_id, and computes relation_features() on them. As a check that the
rebuild matches the pool, each rebuilt candidate's geometry metrics are
compared with the pool row's (`max_metric_diff`). Output, one row per
judged candidate: {"cell", "candidate_id", "features": {name: value}}.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from dev.attention.relations import relation_features
from dev.distill.paths import DATA_ROOT, JUDGMENTS_FILE, POOL_DIR

OUT_FILE = DATA_ROOT / "attention" / "relation_features.jsonl"
TARGETS_PATH = Path(__file__).resolve().parents[1] / "targets.json"
MAX_WORKERS = 4  # the dev laptop also runs judge workers; keep memory modest


def judged_cells(judgments) -> dict[str, set[str]]:
    """{cell: every candidate_id that appears on either side of a judgement}."""
    cells: dict[str, set[str]] = {}
    for j in judgments:
        cells.setdefault(j["cell"], set()).update((j["a"], j["b"]))
    return cells


def features_for_cell(record: dict, target: dict, candidate_ids: set[str], assets, measurer,
                      expected_metrics: dict[str, dict] | None = None) -> tuple[list[dict], dict]:
    """Rebuild the cell's candidates and featurise the ones named in
    `candidate_ids`. Returns (rows, stats); stats["missing"] lists ids the
    rebuild did not produce, stats["max_metric_diff"] the largest gap between a
    rebuilt candidate's geometry metrics and `expected_metrics` (0 if none given)."""
    from engine.pipeline.grading.metrics import geometry_metrics
    from engine.pipeline.layout.fallback import safe_fallback
    from engine.pipeline.layout.generate import generate_candidates
    from engine.pipeline.priority import heuristic_plan

    source = record["source"]
    cell = f"{record['task_id']}@{target['name']}"
    plan = heuristic_plan(source, assets)
    canvas = {"width": target["width"], "height": target["height"]}
    rows, max_diff, found = [], 0.0, set()
    for cand in [safe_fallback(source, canvas, plan, measurer)] + generate_candidates(source, canvas, plan, measurer):
        cid = cand.candidate_id
        if cid not in candidate_ids or cid in found:
            continue
        found.add(cid)
        if expected_metrics and cid in expected_metrics:
            got = geometry_metrics(source, cand, plan)
            max_diff = max([max_diff] + [abs(got[k] - expected_metrics[cid][k]) for k in got if k in expected_metrics[cid]])
        rows.append({"cell": cell, "candidate_id": cid, "features": relation_features(source, cand.design, plan)})
    return rows, {"missing": sorted(candidate_ids - found), "max_metric_diff": max_diff}


@lru_cache(maxsize=1)
def _corpus() -> dict[str, dict]:
    from dev.distill.corpus import load_corpus
    return {d["task_id"]: d for d in load_corpus()}


@lru_cache(maxsize=1)
def _targets() -> dict[str, dict]:
    return {t["name"]: t for t in json.loads(TARGETS_PATH.read_text())}


@lru_cache(maxsize=1)
def _measurer():
    from engine.pipeline.textmeasure import TextMeasurer
    return TextMeasurer()


def _work(job: tuple[str, list[str], dict]) -> tuple[str, list[dict], dict]:
    from dev.distill.assets import merged_assets_dir

    cell, ids, expected = job
    task_id, target_name = cell.rsplit("@", 1)
    try:
        with merged_assets_dir() as assets:
            rows, stats = features_for_cell(_corpus()[task_id], _targets()[target_name], set(ids), assets,
                                            _measurer(), expected)
    except Exception as exc:  # one broken ad must not sink the run
        return cell, [], {"missing": sorted(ids), "max_metric_diff": 0.0, "error": repr(exc)}
    return cell, rows, stats


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build(judgments_path: Path, pool_path: Path, out_path: Path, workers: int = MAX_WORKERS, log=print) -> dict:
    os.environ["CHAMELEON_LLM"] = "heuristic"  # like the pool: never call a model
    cells = judged_cells(_read_jsonl(judgments_path))
    pool = {(r["cell"], r["candidate_id"]): r["metrics"] for r in _read_jsonl(pool_path)}
    jobs = [(cell, sorted(ids), {cid: pool[cell, cid] for cid in ids if (cell, cid) in pool})
            for cell, ids in sorted(cells.items())]
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
        for k, res in enumerate(ex.map(_work, jobs, chunksize=2), 1):
            results.append(res)
            if log and (k % 20 == 0 or k == len(jobs)):
                log(f"{k}/{len(jobs)} cells")
    rows = [r for _, cell_rows, _ in results for r in cell_rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = {"cells": len(jobs), "candidates": len(rows),
               "missing": {cell: s["missing"] for cell, _, s in results if s["missing"]},
               "errors": {cell: s["error"] for cell, _, s in results if "error" in s},
               "max_metric_diff": max((s["max_metric_diff"] for _, _, s in results), default=0.0)}
    if log:
        log(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judgments", type=Path, default=JUDGMENTS_FILE)
    ap.add_argument("--pool", type=Path, default=POOL_DIR / "pool.jsonl")
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args(argv)
    summary = build(args.judgments, args.pool, args.out, args.workers)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
