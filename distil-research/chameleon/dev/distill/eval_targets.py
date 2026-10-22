"""Teacher distance: how close is each engine variant's geometry to Sonnet's
own layout of the same cell? CPU only, no judge, no LLM calls.

Cells: `holdout` = the Sonnet holdout layouts (qc.nosync/gen_holdout.jsonl,
holdout ads, never used for fitting or tuning); `samples` = the 3 official samples x 6
sizes (contact sheets; train ads); `train` = the train_v2 Sonnet
layouts of one parent-ad CV fold (TUNE_FOLD) — tune on these with the model
fitted without that fold (fit_targets fit --exclude-fold 0 --out ...).

Variants (V5 = the six v5 switches):
- v5, v5sw (CHAMELEON_SOLVER=winner), v5t (CHAMELEON_TARGETS=1), v5tw (=winner);
- plan: Sonnet's discrete plan realized by the solver (oracle_plan.realize);
- plant: the same with the predicted targets as solver pulls;
- sonnet: Sonnet's layout run through llm_layout.execute (the reference).

Per cell, against Sonnet: |log size ratio| of the headline font, hero, CTA
and logo (sqrt of box area), the content bounding box IoU, and the mean
distance between matching unit centres / canvas diagonal.

    python -m dev.distill.eval_targets designs --split holdout --variant v5 [--variant ...] [--workers 3]
        [--set W_TARGET=1000] [--model <tune model.json>] [--tag t1] [--cells 80]
    python -m dev.distill.eval_targets dist --split holdout --variant v5 --variant v5t ...
    python -m dev.distill.eval_targets sheet --split holdout --variant v5 --variant v5t --out <.nosync png> [--cell C ...]

Caches: training-data/ab/_flaws.nosync/targets/<split>/<variant>[.<tag>].jsonl (designs; synthetic ads, but
kept with the other flaw caches).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dev.distill import ab_engine as ab
from dev.distill import fit_targets as ft

V5 = {"CHAMELEON_STRATEGIES": "bisection,anchored", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
      "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1", "CHAMELEON_TEXT_SLACK": "1"}
ENGINE = {"v5": {}, "v5sw": {"CHAMELEON_SOLVER": "winner"}, "v5t": {"CHAMELEON_TARGETS": "1"},
          "v5tw": {"CHAMELEON_TARGETS": "winner"}}
SPECIAL = ("plan", "plant", "sonnet")
ROOT = ab.AB_ROOT / "_flaws.nosync" / "targets"
TUNE_FOLD = 0
SAMPLE_ADS = ("warmup_01", "core_01", "advanced_01")   # the 3 official samples (train ads: in the model's data)
SAMPLE_SIZES = ("story", "linkedin_banner", "leaderboard", "medium_rectangle", "wide_skyscraper", "youtube_thumbnail")
_LAYER = ("background", "decoration")


def cells_for(split: str, n: int | None = None, seed: int = 0) -> list[dict]:
    """[{cell, task_id, target, answer}] Sonnet layouts of the split."""
    if split == "holdout":
        rows = [json.loads(l) for l in open(ft.HOLDOUT)]
    elif split == "samples":
        rows = [r for r in (json.loads(l) for l in open(ft.TRAIN))
                if r["cell"].split("@")[0] in SAMPLE_ADS and r["cell"].split("@")[1] in SAMPLE_SIZES]
    else:
        rows = [json.loads(l) for l in open(ft.TRAIN)]
        groups = [ft.parent(r["cell"].split("@")[0]) for r in rows]
        F = ft._folds(groups)
        rows = [r for r, k in zip(rows, F) if k == TUNE_FOLD]
    rows = [r for r in rows if r.get("answer") and r.get("ok") is not False]
    out = [{"cell": r["cell"], "task_id": r["cell"].split("@")[0], "target": r["cell"].split("@")[1],
            "answer": r["answer"]} for r in rows]
    if n is not None and n < len(out):
        out = sorted(random.Random(seed).sample(out, n), key=lambda c: c["cell"])
    return out


def _path(split: str, variant: str, tag: str | None) -> Path:
    return ROOT / split / (f"{variant}.{tag}.jsonl" if tag else f"{variant}.jsonl")


# ---------------------------------------------------------------- designs (workers)

_W = {}


def _init(env: dict, sets: dict, model_path: str | None):
    os.environ.update(V5)
    os.environ.update(env)
    os.environ["CHAMELEON_LLM"] = "heuristic"
    os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
    from engine.pipeline.layout import solver
    from engine.pipeline.layout import targets as T
    for k, v in sets.items():
        mod = T if hasattr(T, k) and not hasattr(solver, k) else solver
        setattr(mod, k, type(getattr(mod, k))(v))
    if model_path:
        T._MODEL = T.load(Path(model_path))
    designs, targets = ft.design_maps(include_holdout=True)
    _W.update(designs=designs, targets=targets)


def _one(job: dict) -> dict:
    from dev.distill.llm_layout import _compact_to_elements, execute
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.priority import plan as priority_plan
    from engine.resize import _get_measurer, resize_with_report
    c, variant = job["cell"], job["variant"]
    src = _W["designs"][c["task_id"]]["source"]
    t = _W["targets"][c["target"]]
    target = {"width": t["width"], "height": t["height"]}
    m = _get_measurer()
    t0 = time.perf_counter()
    cand_id, status = variant, None
    try:
        if variant in ENGINE:
            design, plan, best, _ = resize_with_report(src, target)
            cand_id = best.candidate_id
        else:
            plan = priority_plan(src)
            design = execute(src, target, plan, m, _compact_to_elements(json.loads(c["answer"])["e"]))
            if variant in ("plan", "plant"):
                from dev.distill import oracle_plan
                from engine.pipeline.layout import targets as T
                P = oracle_plan.extract(src, target, design, plan, m)
                pred = T.predict(src, plan, target) if variant == "plant" else None
                st: dict = {}
                _, design = oracle_plan.realize(src, target, plan, P, m, st, targets=pred)
                status = st.get("status")
        failed = ["no_design"] if design is None else check_gates(src, design, target, plan, m)
    except Exception as exc:
        design, failed, status = None, [f"error:{type(exc).__name__}: {exc}"], "error"
    return {"cell": c["cell"], "task_id": c["task_id"], "target": c["target"], "candidate_id": cand_id,
            "design": design, "failed_gates": failed, "status": status,
            "seconds": round(time.perf_counter() - t0, 3)}


def build(split: str, variant: str, workers: int = 3, sets: dict | None = None, model_path: str | None = None,
          tag: str | None = None, n: int | None = None, log=print) -> Path:
    cells = cells_for(split, n)
    out = _path(split, variant, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = ENGINE.get(variant, {})
    jobs = [{"cell": c, "variant": variant} for c in cells]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(workers, 3), initializer=_init,
                             initargs=(env, sets or {}, model_path)) as ex:
        rows = list(ex.map(_one, jobs, chunksize=2))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fails = sum(bool(r["failed_gates"]) for r in rows)
    log(f"[targets] {split}/{variant}{'.' + tag if tag else ''}: {len(rows)} cells, gate failures {fails}, "
        f"{time.time() - t0:.0f} s, mean {statistics.mean(r['seconds'] for r in rows):.2f} s/cell -> {out}")
    return out


# ---------------------------------------------------------------- distance

def _units(design: dict, plan) -> dict:
    els = {e["id"]: e for e in design["elements"]}
    out = {}
    for u in plan.ranked_units():
        if u.role in _LAYER or not u.element_ids or not all(i in els for i in u.element_ids):
            continue
        xs = [els[i] for i in u.element_ids]
        x0 = min(e["x"] for e in xs)
        y0 = min(e["y"] for e in xs)
        x1 = max(e["x"] + e["width"] for e in xs)
        y1 = max(e["y"] + e["height"] for e in xs)
        out[u.unit_id] = (u.role, (x0, y0, x1 - x0, y1 - y0))
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _content_box(units: dict, W: float, H: float):
    bs = [b for r, b in units.values() if b[2] * b[3] < 0.9 * W * H]
    if not bs:
        return None
    x0 = min(b[0] for b in bs)
    y0 = min(b[1] for b in bs)
    return x0, y0, max(b[0] + b[2] for b in bs) - x0, max(b[1] + b[3] for b in bs) - y0


def distance(design: dict, ref: dict, plan, W: float, H: float) -> dict:
    from engine.pipeline.layout import targets as T
    md = T.measure({e["id"]: e for e in design["elements"]}, plan, W, H)
    mr = T.measure({e["id"]: e for e in ref["elements"]}, plan, W, H)
    ud, ur = _units(design, plan), _units(ref, plan)
    out = {}
    for key, name in (("font_headline", "head"), ("hero", "hero"), ("logo", "logo")):
        if md.get(key) and mr.get(key):
            out[name] = abs(math.log(md[key] / mr[key]))
    cta_d = next((b for r, b in ud.values() if r == "cta"), None)
    cta_r = next((b for r, b in ur.values() if r == "cta"), None)
    if cta_d and cta_r and cta_d[2] * cta_d[3] > 0 and cta_r[2] * cta_r[3] > 0:
        out["cta"] = abs(0.5 * math.log((cta_d[2] * cta_d[3]) / (cta_r[2] * cta_r[3])))
    cd, cr = _content_box(ud, W, H), _content_box(ur, W, H)
    if cd and cr:
        out["iou"] = _iou(cd, cr)
    diag = math.hypot(W, H)
    ds = [math.hypot((b[0] + b[2] / 2) - (ur[k][1][0] + ur[k][1][2] / 2), (b[1] + b[3] / 2) - (ur[k][1][1] + ur[k][1][3] / 2)) / diag
          for k, (_, b) in ud.items() if k in ur]
    if ds:
        out["centre"] = sum(ds) / len(ds)
    # raw proportions, for the medians table
    out["_head_font"] = md.get("font_headline")
    out["_hero_share"] = (md["hero"] ** 2) if md.get("hero") else None
    return out


def _load(split: str, variant: str) -> dict:
    name, _, tag = variant.partition(".")
    p = _path(split, name, tag or None)
    return {r["cell"]: r for r in (json.loads(l) for l in p.open())}


def dist(split: str, variants: list[str], log=print) -> dict:
    from engine.pipeline.priority import heuristic_plan
    ft._setup_env()
    designs, targets = ft.design_maps(include_holdout=True)
    ref = _load(split, "sonnet")
    runs = {v: _load(split, v) for v in variants}
    cells = sorted(c for c in ref if ref[c]["design"] and all(c in r for r in runs.values()))
    plans = {}
    per = {v: {} for v in variants}
    for c in cells:
        tid, tn = c.split("@")
        if tid not in plans:
            plans[tid] = heuristic_plan(designs[tid]["source"])
        t = targets[tn]
        for v in variants:
            d = runs[v][c]["design"]
            if d is not None:
                per[v][c] = distance(d, ref[c]["design"], plans[tid], t["width"], t["height"])
    keys = ("head", "hero", "cta", "logo", "iou", "centre")
    table = {}
    base = variants[0]
    for v in variants:
        row = {"cells": len(per[v]), "gate_fail": sum(bool(runs[v][c]["failed_gates"]) for c in cells)}
        for k in keys:
            vals = [x[k] for x in per[v].values() if k in x]
            if vals:
                row[k] = (round(statistics.mean(vals), 3), round(statistics.median(vals), 3), len(vals))
        for k in ("_head_font", "_hero_share"):
            vals = [x[k] for x in per[v].values() if x.get(k)]
            if vals:
                row[k] = round(statistics.median(vals), 4)
        if v != base:  # share of cells where this variant is closer than the first variant (size errors summed)
            better = worse = 0
            for c, x in per[v].items():
                y = per[base].get(c)
                if not y:
                    continue
                a = sum(x.get(k, 0) for k in ("head", "hero", "cta", "logo"))
                b = sum(y.get(k, 0) for k in ("head", "hero", "cta", "logo"))
                better += a < b - 1e-6
                worse += a > b + 1e-6
            row["closer_vs_" + base] = f"{better}-{worse}"
        table[v] = row
    ref_head = [x for x in (distance(ref[c]["design"], ref[c]["design"], plans[c.split('@')[0]],
                                     targets[c.split('@')[1]]["width"], targets[c.split('@')[1]]["height"])["_head_font"]
                            for c in cells) if x]
    ref_hero = [x for x in (distance(ref[c]["design"], ref[c]["design"], plans[c.split('@')[0]],
                                     targets[c.split('@')[1]]["width"], targets[c.split('@')[1]]["height"])["_hero_share"]
                            for c in cells) if x]
    table["sonnet"] = {"cells": len(cells), "_head_font": round(statistics.median(ref_head), 4),
                       "_hero_share": round(statistics.median(ref_hero), 4)}
    log(f"{'variant':16s} cells fails  head(mean/med)  hero          cta           logo          IoU           centre        headF  hero%  closer")
    for v, row in table.items():
        def f(k):
            return f"{row[k][0]:.3f}/{row[k][1]:.3f}" if k in row else "    -      "
        log(f"{v:16s} {row['cells']:5d} {row.get('gate_fail', 0):5d}  {f('head'):13s} {f('hero'):13s} {f('cta'):13s} "
            f"{f('logo'):13s} {f('iou'):13s} {f('centre'):13s} {row.get('_head_font', 0):.3f}  "
            f"{100 * (row.get('_hero_share') or 0):4.1f}  {row.get('closer_vs_' + base, '')}")
    return table


# ---------------------------------------------------------------- sheet

def sheet(split: str, variants: list[str], out: Path, cells: list[str] | None, n: int = 16, tile: int = 300,
          seed: int = 0) -> Path:
    from dev.distill.assets import merged_assets_dir
    from dev.distill.eval_solver import _row_image, _stack
    from engine.pipeline.render import render_to_image
    ft._setup_env()
    designs, _ = ft.design_maps(include_holdout=True)
    runs = {v: _load(split, v) for v in variants}
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    if cells is None:
        first = runs[variants[0]]
        pool = [c for c in common if any(runs[v][c]["design"] != first[c]["design"] for v in variants[1:])]
        cells = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
    rows = []
    with merged_assets_dir() as assets:
        for c in cells:
            src = designs[c.split("@")[0]]["source"]
            imgs = [("source", render_to_image(src, assets_dir=assets, max_size=tile))]
            for v in variants:
                r = runs[v][c]
                if r["design"] is not None:
                    imgs.append((f"{v}: {r['candidate_id']}", render_to_image(r["design"], assets_dir=assets, max_size=tile)))
            rows.append(_row_image(c, imgs, tile))
    return _stack(rows, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dev.distill.eval_targets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("designs")
    d.add_argument("--split", choices=("holdout", "train", "samples"), required=True)
    d.add_argument("--variant", action="append", required=True)
    d.add_argument("--workers", type=int, default=3)
    d.add_argument("--set", action="append", default=[], help="NAME=value: a solver/targets constant (tuning)")
    d.add_argument("--model", default=None, help="targets model json (tuning)")
    d.add_argument("--tag", default=None)
    d.add_argument("--cells", type=int, default=None)
    s = sub.add_parser("dist")
    s.add_argument("--split", choices=("holdout", "train", "samples"), required=True)
    s.add_argument("--variant", action="append", required=True, help="name[.tag]")
    h = sub.add_parser("sheet")
    h.add_argument("--split", choices=("holdout", "train", "samples"), required=True)
    h.add_argument("--variant", action="append", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--cell", action="append", default=None)
    h.add_argument("--n", type=int, default=16)
    h.add_argument("--tile", type=int, default=300)
    args = ap.parse_args(argv)
    if args.cmd == "designs":
        sets = dict(x.split("=", 1) for x in args.set)
        for v in args.variant:
            build(args.split, v, args.workers, sets, args.model, args.tag, args.cells)
    elif args.cmd == "dist":
        dist(args.split, args.variant)
    else:
        print(sheet(args.split, args.variant, Path(args.out), args.cell, args.n, args.tile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
