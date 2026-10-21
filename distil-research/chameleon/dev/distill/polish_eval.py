"""Measure the polish pass (CHAMELEON_POLISH=1, engine/pipeline/layout/polish.py)
candidate by candidate, not only on the winners: for every gate-passing
candidate polish may touch (templates, reflow, extra floor plans, other
strategies; not the fallback, not bisection v2), the flaw detectors of
bisect_flaws and layer_flaws (geometry only) before and after polishing.

    python -m dev.distill.polish_eval --split train --cells 60
    python -m dev.distill.polish_eval --split crello_dev --cells 120

CPU only, at most 3 processes. Prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

V3L = {"CHAMELEON_STRATEGIES": "bisection", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
       "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1"}
FAMILIES = ("squash", "logo_small", "decor_orphan", "edge", "text_small", "cta_big", "mixed_align", "floating",
            "container_empty", "container_split")
_STATE = {}


def _init(folders: tuple):
    from dev.distill import ab_engine
    ab_engine._worker_init(V3L, folders)


def _one(job: dict) -> dict:
    from dev.distill.bisect_flaws import detect
    from dev.distill.layer_flaws import container_flaws
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.layout import polish
    from engine.pipeline.layout.generate import generate_candidates
    from engine.pipeline.priority import plan as priority_plan
    from engine.resize import _get_measurer
    src, t = job["source"], job["target"]
    m = _get_measurer()
    plan = priority_plan(src)
    out = Counter()
    for cand in generate_candidates(src, t, plan, m):  # polish is off in the workers: these are the originals
        if polish._skip(cand) or check_gates(src, cand.design, t, plan, m):
            continue
        t0 = time.perf_counter()
        new = polish.polish(src, t, plan, m, cand)
        out["seconds"] += time.perf_counter() - t0
        out["candidates"] += 1
        out["changed"] += new is not cand
        out["gate_fail_after"] += bool(check_gates(src, new.design, t, plan, m))
        for tag, c in (("before", cand), ("after", new)):
            det = {**detect(src, c.design, plan, m), **container_flaws(src, c.design, plan)}
            for fam in FAMILIES:
                out[f"{tag}:{fam}"] += bool(det.get(fam, {}).get("fire"))
            head = [e["font_size"] for u in plan.units if u.role == "headline" for e in c.design["elements"]
                    if e["id"] in u.element_ids and e["type"] == "text"]
            out[f"{tag}:head_px"] += max(head) if head else 0.0
    return dict(out)


def run(split: str, cells: int | None, seed: int = 0, workers: int = 3) -> Counter:
    from dev.distill import ab_engine
    from dev.distill.layer_flaws import cells_for, folders_for
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    rows, designs = cells_for(split, cells, seed)
    jobs = [{"source": designs[c["task_id"]]["source"],
             "target": {"width": targets[c["target"]]["width"], "height": targets[c["target"]]["height"]}}
            for c in rows if c["task_id"] in designs and c["target"] in targets]
    total = Counter()
    with ProcessPoolExecutor(max_workers=min(workers, 3), initializer=_init, initargs=(folders_for(split),)) as ex:
        for out in ex.map(_one, jobs, chunksize=4):
            total.update(out)
    n = total["candidates"] or 1
    print(f"{split}: {len(jobs)} cells, {total['candidates']} polishable gate-passing candidates, "
          f"{total['changed']} changed ({100 * total['changed'] / n:.0f}%), "
          f"{total['gate_fail_after']} fail a gate after, {1000 * total['seconds'] / n:.1f} ms each")
    print("| detector (share of candidates) | before | after |")
    print("|---|---|---|")
    for fam in FAMILIES:
        print(f"| {fam} | {100 * total[f'before:{fam}'] / n:.1f}% | {100 * total[f'after:{fam}'] / n:.1f}% |")
    print(f"| mean headline px | {total['before:head_px'] / n:.2f} | {total['after:head_px'] / n:.2f} |")
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--cells", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    run(args.split, args.cells, args.seed, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
