"""Export the per-ad floor-plan library the engine reads when CHAMELEON_RETRIEVAL=1.

    python -m dev.distill.export_retrieval_library            # write, then verify
    python -m dev.distill.export_retrieval_library --check    # verify the shipped file only

Words: the *library* is every winning teacher floor plan (B2) of a TRAIN ad, with that
ad's vector and the plan's aspect class (dev/distill/retrieval.library decides "winning").
The vectors are computed with the ENGINE's `ad_vector`
(engine/pipeline/layout/retrieval.py; dev may import engine, never the reverse), and
checked equal to dev/distill/retrieval.ad_vector on every ad.

Verify: on every holdout cell of retrieval-builds.jsonl (the offline evaluation that won
29-11), the engine's recommend() over the written file must choose exactly the plans
dev/distill/retrieval.recommend chooses, and the plans that evaluation recorded. Any
mismatch, or a holdout ad in the library, fails (exit 1).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dev.distill import floorplans as fp
from dev.distill import retrieval as dev_rt
from engine.pipeline.layout import retrieval as engine_rt

OUT_PATH = engine_rt.LIBRARY_PATH
HOLDOUT_BUILDS = dev_rt.RET_BUILDS_FILE


def log(msg: str) -> None:
    print(f"[export-retrieval] {msg}", file=sys.stderr)


def _plans_by_task(designs: dict, assets) -> dict:
    from engine.pipeline.priority import heuristic_plan

    cache: dict = {}

    def prio(task_id):
        if task_id not in cache:
            cache[task_id] = heuristic_plan(designs[task_id]["source"], assets)
        return cache[task_id]

    return prio


def holdout_tasks(pool_rows: list[dict] | None = None) -> set[str]:
    rows = fp._pool_rows() if pool_rows is None else pool_rows
    tasks = {r["task_id"] for r in rows if r.get("split") == "holdout"}
    return tasks | {r["task_id"] for r in fp.read_jsonl(HOLDOUT_BUILDS)}


def build_library(designs: dict, assets) -> list[dict]:
    """dev's library() of winning TRAIN plans, vectors from the engine's ad_vector (checked
    equal to dev's)."""
    prio = _plans_by_task(designs, assets)

    def vector_of(task_id):
        source = designs[task_id]["source"]
        v = engine_rt.ad_vector(source, prio(task_id))
        if not np.array_equal(v, dev_rt.ad_vector(source, prio(task_id))):
            raise AssertionError(f"{task_id}: engine and dev ad_vector differ")
        return v

    return dev_rt.library(vector_of=vector_of)


def to_json(lib: list[dict]) -> dict:
    entries = [{"proposal_id": e["proposal_id"], "task_id": e["task_id"], "cell": e["cell"],
                "aspect_class": e["aspect_class"], "vector": [float(x) for x in e["vector"]],
                "plan": {"margins": list(e["plan"]["margins"]),
                         "slots": [{"name": s["name"], "box": list(s["box"])} for s in e["plan"]["slots"]]}}
               for e in sorted(lib, key=lambda e: e["proposal_id"])]
    return {"version": engine_rt.LIBRARY_VERSION, "split": "train",
            "about": "Winning teacher floor plans (B2) of train ads, for per-ad retrieval (CHAMELEON_RETRIEVAL=1). "
                     "Built by dev/distill/export_retrieval_library.py; do not edit by hand.",
            "k": engine_rt.DEFAULT_K, "dimensions": list(engine_rt.DIMENSIONS), "entries": entries}


def verify(path: Path = OUT_PATH, designs: dict | None = None, assets=None) -> dict:
    """Replay every holdout cell of the offline evaluation through the engine's loader and
    recommend(); compare with dev's recommend and with the recorded plans."""
    if assets is None:
        from dev.distill.assets import merged_assets_dir

        with merged_assets_dir() as merged:
            return verify(path, designs, merged)
    designs = fp._designs() if designs is None else designs
    prio = _plans_by_task(designs, assets)
    engine_lib = engine_rt.parse_library(Path(path).read_text())
    dev_lib = dev_rt.library(vector_of=lambda t: dev_rt.ad_vector(designs[t]["source"], prio(t)))
    holdout = holdout_tasks()
    rows = fp.read_jsonl(HOLDOUT_BUILDS)
    mismatches, runtime_differs = [], []
    from engine.pipeline.priority import heuristic_plan

    for row in sorted(rows, key=lambda r: r["cell"]):
        task, cls = row["task_id"], row["aspect_class"]
        source = designs[task]["source"]
        ours = [p["name"] for p in engine_rt.recommend(source, cls, prio(task), engine_rt.DEFAULT_K, engine_lib,
                                                       exclude_task=task)]
        theirs = [p["name"] for p in dev_rt.recommend(source, cls, dev_rt.DEFAULT_K, lib=dev_lib, plan=prio(task),
                                                      exclude_task=task)]
        recorded = [p["name"] for p in row.get("plans", [])]
        if not ours == theirs == recorded:
            mismatches.append({"cell": row["cell"], "engine": ours, "dev": theirs, "recorded": recorded})
        # The engine at run time plans with its default assets folder (brand_kit/assets only);
        # count the cells where that changes the choice (a property of the harness, not a failure).
        runtime = [p["name"] for p in engine_rt.recommend(source, cls, heuristic_plan(source), engine_rt.DEFAULT_K,
                                                          engine_lib)]
        if runtime != ours:
            runtime_differs.append(row["cell"])
    return {"cells": len(rows), "entries": len(engine_lib), "ads": len({e["task_id"] for e in engine_lib}),
            "mismatches": mismatches,
            "holdout_tasks_in_library": sorted({e["task_id"] for e in engine_lib} & holdout),
            "runtime_assets_change_choice": runtime_differs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify the existing file; don't write")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)
    from dev.distill.assets import merged_assets_dir

    designs = fp._designs()
    with merged_assets_dir() as assets:
        if not args.check:
            lib = build_library(designs, assets)
            bad = sorted({e["task_id"] for e in lib} & holdout_tasks())
            if bad:
                log(f"holdout ads in the library: {bad}")
                return 1
            args.out.write_text(json.dumps(to_json(lib), indent=1) + "\n")
            log(f"wrote {len(lib)} plans of {len({e['task_id'] for e in lib})} train ads -> {args.out}")
        result = verify(args.out, designs, assets)
    log(f"verify: {result['cells']} holdout cells, {len(result['mismatches'])} mismatches, "
        f"{len(result['holdout_tasks_in_library'])} holdout ads in the library; "
        f"{len(result['runtime_assets_change_choice'])} cells choose differently with the engine's default assets")
    for m in result["mismatches"][:10]:
        log(f"  mismatch {m}")
    return 0 if not result["mismatches"] and not result["holdout_tasks_in_library"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
