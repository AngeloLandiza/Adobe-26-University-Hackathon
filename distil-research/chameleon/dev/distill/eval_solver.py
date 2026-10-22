"""Measure the geometry solver pass (engine/pipeline/layout/solver.py,
CHAMELEON_SOLVER=1), CPU only, no judge.

    # winners, via the eval_anchored caches (layer_flaws / balance_flaws `rates` work on them)
    python -m dev.distill.eval_anchored rebuild --name sol_tr_v5 --split train --cells 60 --env <v5 switches>
    python -m dev.distill.eval_anchored rebuild --name sol_tr_s1 --split train --cells 60 --env <v5 switches> --env CHAMELEON_SOLVER=1
    python -m dev.distill.eval_solver changed --a sol_tr_v5 --b sol_tr_s1          # caches
    python -m dev.distill.eval_solver changed --a v5 --b v5s --ab                  # ab_engine runs
    python -m dev.distill.eval_solver cands --split train --cells 60              # per candidate: status, ms, gates
    python -m dev.distill.eval_solver sheet --a sol_tr_v5 --b sol_tr_s1 --out <.nosync>/x.png [--cell C ...] [--n 8]
    python -m dev.distill.eval_solver visible --a v5 --b v5s [--out cells.txt]     # how many changes a judge can see
    python -m dev.distill.eval_solver absheet --run v5 --run v5s --out <.nosync>/x.png --cell C ...   # ab renders

Caches and sheets for Crello splits are Crello-derived: keep them under a
.nosync folder (training-data/ab/_flaws.nosync/).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from dev.distill import layer_flaws

V5 = {"CHAMELEON_STRATEGIES": "bisection,anchored", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
      "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1", "CHAMELEON_TEXT_SLACK": "1"}


def _classes() -> dict[str, str]:
    from dev.distill import ab_engine
    return ab_engine._target_classes()


def changed(a: dict, b: dict) -> dict:
    """Cells whose winning design differs, split by same candidate re-solved vs a different winner, by aspect class."""
    cls = _classes()
    common = sorted(set(a) & set(b))
    out = {"cells": len(common), "changed": 0, "same_candidate": 0, "different_candidate": 0,
           "fallback_a": sum(bool(a[c].get("is_fallback")) for c in common),
           "fallback_b": sum(bool(b[c].get("is_fallback")) for c in common),
           "gate_failures_b": sum(bool(b[c].get("failed_gates")) for c in common), "by_class": {}}
    by = defaultdict(lambda: [0, 0, 0])
    for c in common:
        k = cls.get(a[c]["target"], "?")
        by[k][0] += 1
        if a[c]["design_hash"] == b[c]["design_hash"]:
            continue
        out["changed"] += 1
        by[k][1] += 1
        if a[c]["candidate_id"] == b[c]["candidate_id"]:
            out["same_candidate"] += 1
        else:
            out["different_candidate"] += 1
            by[k][2] += 1
    out["by_class"] = {k: {"cells": v[0], "changed": v[1], "different_candidate": v[2]} for k, v in sorted(by.items())}
    return out


VISIBLE = 0.01   # a cell changed visibly when this share of its render's pixels moved by more than PIX_DELTA
PIX_DELTA = 40


def visible(a: str, b: str, out: Path | None = None) -> dict:
    """Share of each changed cell's 768 px render whose pixels differ (max channel delta > PIX_DELTA),
    from two ab_engine runs; cells above VISIBLE count as visibly changed. With `out`, writes the
    visibly changed cell ids (one per line), for capping the judge on what a judge can see."""
    import numpy as np
    from PIL import Image
    from dev.distill import ab_engine
    wa, wb = ab_engine.load_winners(a), ab_engine.load_winners(b)
    cls = _classes()
    shares, vis = {}, []
    by = defaultdict(lambda: [0, 0])
    for c in sorted(set(wa) & set(wb)):
        if wa[c]["design_hash"] == wb[c]["design_hash"]:
            continue
        ia = np.asarray(Image.open(ab_engine.AB_ROOT / a / wa[c]["png"]).convert("RGB"), dtype=np.int16)
        ib = np.asarray(Image.open(ab_engine.AB_ROOT / b / wb[c]["png"]).convert("RGB"), dtype=np.int16)
        if ia.shape != ib.shape:
            share = 1.0
        else:
            share = float((np.abs(ia - ib).max(axis=2) > PIX_DELTA).mean())
        shares[c] = share
        k = cls.get(wa[c]["target"], "?")
        by[k][0] += 1
        if share > VISIBLE:
            vis.append(c)
            by[k][1] += 1
    vals = sorted(shares.values())
    res = {"changed": len(shares), "visible": len(vis),
           "visible_same_candidate": sum(wa[c]["candidate_id"] == wb[c]["candidate_id"] for c in vis),
           "median_share": vals[len(vals) // 2] if vals else 0.0,
           "by_class": {k: {"changed": v[0], "visible": v[1]} for k, v in sorted(by.items())}}
    if out:
        Path(out).write_text("".join(c + "\n" for c in vis))
    return res


def _load(name: str, ab: bool) -> dict:
    if ab:
        from dev.distill import ab_engine
        return ab_engine.load_winners(name)
    return layer_flaws.load_cache(name)


def cands(split: str, n: int | None, seed: int = 0, workers: int = 3) -> dict:
    """Every candidate of the v5 engine on `split` (n cells): solver status, ms per candidate, LP size."""
    from concurrent.futures import ProcessPoolExecutor
    from dev.distill.eval_anchored import _sample_cells
    from dev.distill import ab_engine
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    if split == "samples":
        cells, designs = _sample_cells()
        folders = ()
    else:
        cells, designs = layer_flaws.cells_for(split, n, seed)
        folders = layer_flaws.folders_for(split)
    jobs = [{"cell": c["cell"], "source": designs[c["task_id"]]["source"], "target": targets[c["target"]]}
            for c in cells if c["target"] in targets and c["task_id"] in designs]
    status, ms, blocks, vars_, solved_ms = Counter(), [], [], [], []
    with ProcessPoolExecutor(max_workers=min(3, workers), initializer=layer_flaws._init, initargs=(V5, folders)) as ex:
        for rows in ex.map(_cands_one, jobs, chunksize=2):
            for r in rows:
                status[r.get("status", "?")] += 1
                ms.append(r["ms"])
                if r.get("blocks"):
                    blocks.append(r["blocks"])
                    vars_.append(r.get("vars", 0))
                if r.get("status") in ("solved", "gates", "unchanged"):
                    solved_ms.append(r["ms"])
    total = sum(status.values()) or 1
    out = {"cells": len(jobs), "candidates": total, "status": dict(status),
           "solved_share": status["solved"] / total,
           "gate_fail_share_of_attempts": status["gates"] / max(1, status["solved"] + status["gates"]),
           "ms_mean": sum(ms) / max(1, len(ms)), "ms_max": max(ms or [0]),
           "ms_p95": sorted(ms)[int(0.95 * (len(ms) - 1))] if ms else 0,
           "blocks_mean": sum(blocks) / max(1, len(blocks)), "lp_vars_mean": sum(vars_) / max(1, len(vars_))}
    print(json.dumps(out, indent=2))
    return out


def _cands_one(job: dict) -> list[dict]:
    from engine.pipeline.layout import solver
    from engine.pipeline.layout.generate import generate_candidates
    from engine.pipeline.priority import plan as priority_plan
    from engine.resize import _get_measurer
    m = _get_measurer()
    src, t = job["source"], job["target"]
    tgt = {"width": t["width"], "height": t["height"]}
    try:
        plan = priority_plan(src)
        cs = generate_candidates(src, tgt, plan, m)
        layout = solver._layout_plan(src, plan, tgt)
    except Exception:
        return []
    rows = []
    for c in cs:
        st: dict = {}
        t0 = time.perf_counter()
        try:
            solver.solve_candidate(src, tgt, plan, m, c, layout, st)
        except Exception as exc:
            st["status"] = f"error:{type(exc).__name__}"
        st["ms"] = (time.perf_counter() - t0) * 1000
        if c.is_fallback:
            st["status"] = "fallback"
        rows.append(st)
    return rows


def _row_image(label: str, imgs: list, tile: int):
    from PIL import Image, ImageDraw
    label_h = 28
    w = sum(im.width for _, im in imgs) + 10 * len(imgs)
    h = max(im.height for _, im in imgs) + label_h + 14
    row = Image.new("RGB", (max(w, 400), h), (225, 225, 225))
    d = ImageDraw.Draw(row)
    d.text((4, 2), label, fill="black")
    x = 0
    for lab, im in imgs:
        d.text((x + 4, 15), lab[:48], fill=(40, 40, 160))
        row.paste(im.convert("RGB"), (x, label_h + 10))
        x += im.width + 10
    return row


def _stack(rows: list, out: Path) -> Path:
    from PIL import Image
    img = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 6 for r in rows)), "white")
    y = 0
    for r in rows:
        img.paste(r, (0, y))
        y += r.height + 6
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def sheet(a: str, b: str, out: Path, cells: list[str] | None = None, n: int = 8, seed: int = 0, tile: int = 360,
          only_changed: bool = True) -> Path:
    """source | A winner | B winner, from two eval_anchored caches."""
    from dev.distill.assets import merged_assets_dir
    from dev.distill.eval_anchored import _sources
    from engine.pipeline.render import render_to_image
    ra, rb = layer_flaws.load_cache(a), layer_flaws.load_cache(b)
    split = layer_flaws._meta(b)["split"]
    common = sorted(set(ra) & set(rb))
    if cells is None:
        pool = [c for c in common if not only_changed or ra[c]["design_hash"] != rb[c]["design_hash"]]
        cells = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
    srcs = _sources(split)
    folders = () if split == "samples" else layer_flaws.folders_for(split)
    rows = []
    with merged_assets_dir(*folders) as assets:
        for c in cells:
            x, y = ra[c], rb[c]
            imgs = [("source", render_to_image(srcs[x["task_id"]], assets_dir=assets, max_size=tile)),
                    (f"{a}: {x['candidate_id']}", render_to_image(x["design"], assets_dir=assets, max_size=tile)),
                    (f"{b}: {y['candidate_id']}", render_to_image(y["design"], assets_dir=assets, max_size=tile))]
            rows.append(_row_image(c, imgs, tile))
    return _stack(rows, out)


def absheet(runs: list[str], out: Path, cells: list[str], tile: int = 360, sources_from: str | None = None) -> Path:
    """source | each run's rendered winner, from ab_engine runs (renders already on disk)."""
    from PIL import Image
    from dev.distill import ab_engine
    from dev.distill.paths import POOL_DIR, RENDERS_SUBDIR
    W = {r: ab_engine.load_winners(r) for r in runs}
    rows = []
    for c in cells:
        first = W[runs[0]][c]
        src = ab_engine.source_png(ab_engine.AB_ROOT, first["task_id"])
        if not src.exists():
            src = Path(POOL_DIR) / RENDERS_SUBDIR / first["task_id"] / "source.png"
        imgs = []
        if src.exists():
            im = Image.open(src)
            im.thumbnail((tile, tile))
            imgs.append(("source", im))
        for r in runs:
            row = W[r].get(c)
            if row is None:
                continue
            im = Image.open(ab_engine.AB_ROOT / r / row["png"])
            im.thumbnail((tile, tile))
            imgs.append((f"{r}: {row.get('candidate_id', '')}", im))
        rows.append(_row_image(c, imgs, tile))
    return _stack(rows, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dev.distill.eval_solver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("changed")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--ab", action="store_true", help="names are ab_engine runs, not eval_anchored caches")
    k = sub.add_parser("cands")
    k.add_argument("--split", required=True)
    k.add_argument("--cells", type=int, default=None)
    k.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sheet")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--cell", action="append", default=None)
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tile", type=int, default=360)
    v = sub.add_parser("visible")
    v.add_argument("--a", required=True)
    v.add_argument("--b", required=True)
    v.add_argument("--out", default=None)
    t = sub.add_parser("absheet")
    t.add_argument("--run", action="append", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--cell", action="append", required=True)
    t.add_argument("--tile", type=int, default=360)
    args = ap.parse_args(argv)
    if args.cmd == "changed":
        print(json.dumps(changed(_load(args.a, args.ab), _load(args.b, args.ab)), indent=2))
    elif args.cmd == "cands":
        cands(args.split, args.cells, args.seed)
    elif args.cmd == "sheet":
        print(sheet(args.a, args.b, Path(args.out), args.cell, args.n, args.seed, args.tile))
    elif args.cmd == "visible":
        print(json.dumps(visible(args.a, args.b, args.out), indent=2))
    elif args.cmd == "absheet":
        print(absheet(args.run, Path(args.out), args.cell, args.tile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
