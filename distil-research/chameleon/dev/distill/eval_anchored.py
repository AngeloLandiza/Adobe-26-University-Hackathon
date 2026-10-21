"""Measure the `anchored` strategy (engine/pipeline/layout/strategies/anchored.py)
inside the whole engine, CPU only (<= 3 processes), no judge.

    # re-run the engine on a split; rows are layer_flaws-compatible (its `rates` works on them)
    python -m dev.distill.eval_anchored rebuild --name anc_tr_base --split train --cells 60 --env CHAMELEON_STRATEGIES=bisection ...
    python -m dev.distill.eval_anchored rebuild --name anc_tr --split train --cells 60 --env CHAMELEON_STRATEGIES=bisection,anchored ...
    python -m dev.distill.eval_anchored stats --name anc_tr_base --name anc_tr
    python -m dev.distill.eval_anchored sheet --base anc_tr_base --name anc_tr --out <.nosync folder>/sheet.png
    python -m dev.distill.eval_anchored changed --run training-data/ab/v3L_anch --ref training-data/ab/v3L

Split `samples` = the 3 official samples (input/*.json) x every target in
dev/targets.json plus 1600x400. Each row also keeps the best anchored
candidate (highest grader total among the gate-passing anchored ones), so
the sheet can show it where it didn't win. Caches go to
training-data/ab/_flaws.nosync/ (gitignored: Crello-derived designs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dev.distill import layer_flaws
from dev.distill.layer_flaws import CACHE_DIR, WORKERS

EXTRA_TARGETS = [{"name": "wide_1600x400", "width": 1600, "height": 400}]
_ANCH = {"calls": 0, "failed": 0, "built": 0}


def _count():
    """Count the hard-gate checks the anchored strategy makes (it drops failures silently)."""
    from engine.pipeline.grading import gates
    real = gates.check_gates
    if getattr(real, "_anch", False):
        return

    def counting(*args, **kwargs):
        failed = real(*args, **kwargs)
        if sys._getframe(1).f_code.co_filename.endswith("anchored.py"):
            _ANCH["calls"] += 1
            _ANCH["failed"] += bool(failed)
        return failed

    counting._anch = True
    counting._counting = getattr(real, "_counting", False)
    gates.check_gates = counting


def _init(env: dict, folders: tuple):
    layer_flaws._init(env, folders)
    _count()


def _one(job: dict) -> dict:
    from dev.distill import ab_engine
    from dev.distill.bisect_flaws import plan_to_dict
    from engine.pipeline.grading.gates import check_gates
    from engine.resize import _get_measurer
    try:
        source, target = job["source"], job["target"]
        canvas = {"width": target["width"], "height": target["height"]}
        before = dict(_ANCH)
        t0 = time.perf_counter()
        design, plan, best, reports = ab_engine._resize(source, canvas)
        secs = time.perf_counter() - t0
        failed = check_gates(source, design, canvas, plan, _get_measurer())
        anch = [r for r in reports if r.candidate_id.startswith("anchored")]
        passing = [r for r in anch if r.passed_gates]
        top = max(passing, key=lambda r: (r.total, r.candidate_id), default=None)
        best_any = max((r.total for r in reports if r.passed_gates), default=None)
        top_design = None
        if top is not None and top.candidate_id != best.candidate_id:
            top_design = _rebuild_candidate(source, canvas, plan, top.candidate_id)
        return {"cell": job["cell"], "task_id": job["task_id"], "target": job["target_name"],
                "candidate_id": best.candidate_id, "is_fallback": bool(best.is_fallback),
                "design_hash": ab_engine.design_hash(design), "seconds": round(secs, 3),
                "failed_gates": failed,
                "anchored_gate_calls": _ANCH["calls"] - before["calls"],
                "anchored_gate_failed": _ANCH["failed"] - before["failed"],
                "anchored_candidates": len(anch),
                "anchored_best_id": top.candidate_id if top else None,
                "anchored_best_total": round(top.total, 4) if top else None,
                "best_total": round(best_any, 4) if best_any is not None else None,
                "anchored_best_design": top_design,
                "design": design, "plan": plan_to_dict(plan)}
    except Exception as exc:
        return {"cell": job["cell"], "error": f"{type(exc).__name__}: {exc}"}


def _rebuild_candidate(source, canvas, plan, cid):
    from engine.pipeline.layout.strategies import anchored
    from engine.resize import _get_measurer
    for c in anchored.candidates(source, canvas, plan, _get_measurer()):
        if c.candidate_id == cid:
            return c.design
    return None


def _sample_cells() -> tuple[list[dict], dict]:
    from dev.distill import ab_engine
    targets = json.loads(ab_engine.TARGETS_PATH.read_text()) + EXTRA_TARGETS
    designs, cells = {}, []
    for path in sorted(Path("input").glob("*.json")):
        designs[path.stem] = {"task_id": path.stem, "source": json.loads(path.read_text())["source"]}
        cells += [{"cell": f"{path.stem}@{t['name']}", "task_id": path.stem, "target": t["name"]} for t in targets]
    return cells, designs


def rebuild(name: str, env: dict, split: str, n: int | None = None, seed: int = 0, workers: int = WORKERS,
            log=print) -> Path:
    from dev.distill import ab_engine
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text()) + EXTRA_TARGETS}
    if split == "samples":
        cells, designs = _sample_cells()
        folders = ()
    else:
        cells, designs = layer_flaws.cells_for(split, n, seed)
        folders = layer_flaws.folders_for(split)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.jsonl"
    done = {json.loads(x)["cell"] for x in path.read_text().splitlines() if x.strip()} if path.exists() else set()
    jobs = [{"cell": c["cell"], "task_id": c["task_id"], "target_name": c["target"],
             "source": designs[c["task_id"]]["source"], "target": targets[c["target"]]}
            for c in cells if c["cell"] not in done and c["task_id"] in designs and c["target"] in targets]
    log(f"[anchored] {name}: {len(jobs)} cells to run ({len(done)} cached)")
    with ProcessPoolExecutor(max_workers=max(1, min(workers, WORKERS)), initializer=_init,
                             initargs=(env, folders)) as ex, path.open("a") as f:
        for row in ex.map(_one, jobs, chunksize=2):
            if "error" in row:
                log(f"[anchored] {row['cell']}: {row['error']}")
                continue
            f.write(json.dumps(row) + "\n")
    (CACHE_DIR / f"{name}.meta.json").write_text(json.dumps({"name": name, "env": env, "split": split,
                                                             "cells": n, "seed": seed}, indent=2))
    return path


def stats(names: list[str]) -> dict:
    out = {}
    for name in names:
        rows = layer_flaws.load_cache(name)
        n = len(rows) or 1
        calls = sum(r.get("anchored_gate_calls", 0) for r in rows.values())
        with_anch = [r for r in rows.values() if r.get("anchored_candidates")]
        close = [r for r in with_anch if r.get("anchored_best_total") is not None and r.get("best_total") is not None
                 and r["best_total"] - r["anchored_best_total"] <= 0.02]
        out[name] = {
            "cells": len(rows),
            "anchored_pick_share": sum(r["candidate_id"].startswith("anchored") for r in rows.values()) / n,
            "bisection_pick_share": sum(r["candidate_id"].startswith("bisection") for r in rows.values()) / n,
            "fallback_share": sum(r["is_fallback"] for r in rows.values()) / n,
            "cells_with_anchored": len(with_anch) / n,
            "anchored_per_cell": sum(r.get("anchored_candidates", 0) for r in rows.values()) / n,
            "anchored_gate_pass": (1 - sum(r.get("anchored_gate_failed", 0) for r in rows.values()) / calls)
            if calls else None,
            "anchored_within_0.02": len(close) / n,
            "winner_gate_failures": sum(bool(r["failed_gates"]) for r in rows.values()),
            "mean_seconds": sum(r["seconds"] for r in rows.values()) / n,
        }
    keys = list(next(iter(out.values())).keys()) if out else []
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in keys:
        cell = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"| {k} | " + " | ".join(cell(out[nm][k]) for nm in names) + " |")
    return out


def by_shape(name: str) -> dict:
    """Anchored pick share by target."""
    rows = layer_flaws.load_cache(name)
    out: dict[str, list[int]] = {}
    for r in rows.values():
        o = out.setdefault(r["target"], [0, 0])
        o[0] += r["candidate_id"].startswith("anchored")
        o[1] += 1
    for t, (a, n) in sorted(out.items()):
        print(f"{t:22s} {a}/{n}")
    return out


def _sources(split: str) -> dict:
    if split == "samples":
        return {k: v["source"] for k, v in _sample_cells()[1].items()}
    return layer_flaws._sources(split)


def sheet(base: str, name: str, out: Path, cells: list[str] | None = None, n: int = 12, seed: int = 0,
          tile: int = 300, pick: str = "best") -> Path:
    """source | base winner | variant winner | best anchored candidate (if it isn't the winner)."""
    import random
    from PIL import Image, ImageDraw
    from dev.distill.assets import merged_assets_dir
    from engine.pipeline.render import render_to_image

    ra, rb = layer_flaws.load_cache(base), layer_flaws.load_cache(name)
    split = layer_flaws._meta(name)["split"]
    common = sorted(set(ra) & set(rb))
    if cells is None:
        pool = [c for c in common if (rb[c]["candidate_id"].startswith("anchored") if pick == "won"
                                      else rb[c].get("anchored_best_id"))]
        cells = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
    srcs = _sources(split)
    folders = () if split == "samples" else layer_flaws.folders_for(split)
    rows = []
    with merged_assets_dir(*folders) as assets:
        for c in cells:
            a, b = ra[c], rb[c]
            imgs = [("source", render_to_image(srcs[a["task_id"]], assets_dir=assets, max_size=tile)),
                    (f"base: {a['candidate_id'][:40]}", render_to_image(a["design"], assets_dir=assets, max_size=tile)),
                    (f"+anch: {b['candidate_id'][:40]}", render_to_image(b["design"], assets_dir=assets, max_size=tile))]
            if b.get("anchored_best_design"):
                gap = (b["best_total"] or 0) - (b["anchored_best_total"] or 0)
                imgs.append((f"best anch: {b['anchored_best_id'][9:]} (-{gap:.3f})",
                             render_to_image(b["anchored_best_design"], assets_dir=assets, max_size=tile)))
            label_h = 30
            w = sum(im.width for _, im in imgs) + 10 * len(imgs)
            h = max(im.height for _, im in imgs) + label_h + 16
            row = Image.new("RGB", (max(w, 400), h), (225, 225, 225))
            d = ImageDraw.Draw(row)
            d.text((4, 2), c, fill="black")
            x = 0
            for lab, im in imgs:
                d.text((x + 4, 16), lab, fill=(40, 40, 160))
                row.paste(im.convert("RGB"), (x, label_h + 12))
                x += im.width + 10
            rows.append(row)
    img = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 6 for r in rows)), "white")
    y = 0
    for r in rows:
        img.paste(r, (0, y))
        y += r.height + 6
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def changed(run: Path, ref: Path) -> dict:
    """Picks changed between two ab_engine runs (design_hash per cell), and the
    run's anchored winners."""
    def rows(p: Path) -> dict:
        return {r["cell"]: r for r in (json.loads(x) for x in (Path(p) / "winners.jsonl").read_text().splitlines()
                                       if x.strip())}
    a, b = rows(run), rows(ref)
    common = sorted(set(a) & set(b))
    diff = [c for c in common if a[c].get("design_hash") != b[c].get("design_hash")]
    anch = [c for c in common if str(a[c].get("candidate_id", "")).startswith("anchored")]
    out = {"cells": len(common), "changed": len(diff), "anchored_winners": len(anch),
           "anchored_changed": len(set(anch) & set(diff)),
           "fallback_run": sum(bool(a[c].get("is_fallback")) for c in common),
           "fallback_ref": sum(bool(b[c].get("is_fallback")) for c in common)}
    print(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    from dev.distill.ab_engine import parse_env
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebuild")
    r.add_argument("--name", required=True)
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--split", required=True, choices=(*layer_flaws.SPLITS, "samples"))
    r.add_argument("--cells", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=WORKERS)
    s = sub.add_parser("stats")
    s.add_argument("--name", action="append", required=True)
    b = sub.add_parser("shapes")
    b.add_argument("--name", required=True)
    c = sub.add_parser("sheet")
    c.add_argument("--base", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--cell", action="append", default=None)
    c.add_argument("--n", type=int, default=12)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--tile", type=int, default=300)
    c.add_argument("--pick", choices=("best", "won"), default="best",
                   help="won: cells anchored won; best: cells with an anchored candidate")
    d = sub.add_parser("changed")
    d.add_argument("--run", required=True)
    d.add_argument("--ref", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "rebuild":
        rebuild(args.name, parse_env(args.env), args.split, args.cells, args.seed, args.workers)
    elif args.cmd == "stats":
        stats(args.name)
    elif args.cmd == "shapes":
        by_shape(args.name)
    elif args.cmd == "sheet":
        print(sheet(args.base, args.name, Path(args.out), args.cell, args.n, args.seed, args.tile, args.pick))
    else:
        changed(Path(args.run), Path(args.ref))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
