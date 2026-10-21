"""Detectors for the layer flaws the teacher names on real (Crello) ads, and
the before/after harness for CHAMELEON_LAYERS (engine/pipeline/layout/layer_groups.py).

The critiques of bisection on real ads say, again and again: the copy was
pulled out of the card / label / band it sat in, leaving the empty shape
stranded; a full-bleed photo or pattern was shrunk to a patch; text landed
on a busy part of a photo it wasn't on. Each detector measures one of these
on a finished design against its source:

- container_empty: a container (a shape or image that held content units in
  the source, layer_groups.detect) holds none of them any more (each unit
  less than HELD_SHARE inside the container's new box): a stranded shape.
- container_split: it still holds some of them but not all.
- background_shrunk: a background-layer element (the full-bleed stack) that
  covered >= 90% of the source canvas covers < 90% of the target;
  background_shrunk_avoidable counts only those that could fill it (shapes,
  cover images, or a contain image whose aspect is within 10% of the target's;
  any other contain image must keep its aspect inside the canvas).
- text_on_busy: a text whose backdrop (render without text, mean Sobel of
  the box's inner area, as the corpus gate measures it) is busier than
  MAX_BUSY when its source backdrop wasn't.
- text_low_contrast: contrast against the backdrop under 3:1 when it wasn't.
The bisection flaw detectors (bisect_flaws.detect) are reported alongside.

    # re-run the engine (CPU only, <= 3 processes); caches under training-data/ab/_flaws.nosync/
    python -m dev.distill.layer_flaws rebuild --name dev_b2 --split crello_dev --env CHAMELEON_STRATEGIES=bisection --env CHAMELEON_BISECT_V2=1
    python -m dev.distill.layer_flaws rebuild --name dev_b2l --split crello_dev --env ... --env CHAMELEON_LAYERS=1
    python -m dev.distill.layer_flaws rates --name dev_b2 --name dev_b2l
    python -m dev.distill.layer_flaws sheet --name dev_b2 --name dev_b2l --out training-data/ab/_flaws.nosync/sheet.png

Caches hold whole designs, so for Crello splits they are Crello-derived:
they stay in the gitignored _flaws.nosync/ folder, like the renders.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dev.distill.paths import DATA_ROOT

AB_ROOT = DATA_ROOT / "ab"
CACHE_DIR = AB_ROOT / "_flaws.nosync"
SPLITS = ("crello_dev", "challenge2", "train", "holdout")
HELD_SHARE = 0.5          # a unit at least this much inside its container's box is still held
BG_COVER = 0.9            # background layer coverage of the canvas
ASPECT_SLACK = 0.1        # a contain image within this aspect of the target can still fill it
MAX_BUSY = 40.0           # as make_corpus.MAX_BUSY
MIN_CONTRAST = 3.0        # as make_corpus.MIN_CONTRAST
LAYER_FAMILIES = ("container_empty", "container_split", "background_shrunk", "background_shrunk_avoidable",
                  "text_on_busy", "text_low_contrast")
WORKERS = 3


def folders_for(split: str) -> tuple:
    from dev.distill import assets
    return {"crello_dev": assets.CRELLO_DEV_FOLDERS, "challenge2": assets.CHALLENGE2_FOLDERS}.get(split, ())


# ============================================================ detectors (geometry)

def _box(el: dict):
    from engine.pipeline.geometry import rotated_aabb
    return rotated_aabb(el)


def _inside_share(outer, inner) -> float:
    from engine.pipeline.geometry import overlap_area
    a = inner[2] * inner[3]
    return overlap_area(outer, inner) / a if a > 0 else 0.0


def container_flaws(source: dict, design: dict, plan) -> dict[str, dict]:
    """container_empty / container_split for one design (every group, nested ones included)."""
    from engine.pipeline.geometry import union_box
    from engine.pipeline.layout.layer_groups import detect

    layers = detect(source, plan)
    els = {e["id"]: e for e in design["elements"]}
    empty, split, n = [], [], 0
    for g in layers.groups:
        if g.container not in els:
            continue
        n += 1
        cbox = _box(els[g.container])
        held = [_inside_share(cbox, union_box([_box(els[i]) for i in u.element_ids])) >= HELD_SHARE
                for u in g.units if all(i in els for i in u.element_ids)]
        if held and not any(held):
            empty.append(g.container)
        elif held and not all(held):
            split.append(g.container)
    return {"container_empty": {"fire": bool(empty), "ids": empty, "containers": n},
            "container_split": {"fire": bool(split), "ids": split, "containers": n}}


def background_flaws(source: dict, design: dict) -> dict[str, dict]:
    from engine.pipeline.geometry import overlap_area
    from engine.pipeline.layout.layer_groups import background_stack

    SW, SH = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    W, H = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    src = {e["id"]: e for e in source["elements"]}
    els = {e["id"]: e for e in design["elements"]}
    shrunk, avoidable = [], []
    for eid in background_stack(source["elements"], SW, SH):
        e0, e1 = src[eid], els.get(eid)
        if e1 is None or overlap_area(_box(e0), (0, 0, SW, SH)) < BG_COVER * SW * SH:
            continue
        if overlap_area(_box(e1), (0, 0, W, H)) >= BG_COVER * W * H:
            continue
        shrunk.append(eid)
        fixed = e0["type"] == "image" and e0.get("fit", "contain") != "cover"
        if not fixed or abs((e0["width"] / e0["height"]) / (W / H) - 1) <= ASPECT_SLACK:
            avoidable.append(eid)
    return {"background_shrunk": {"fire": bool(shrunk), "ids": shrunk},
            "background_shrunk_avoidable": {"fire": bool(avoidable), "ids": avoidable}}


# ============================================================ detectors (pixels)

def text_flaws(source: dict, design: dict, assets, src_backdrops: dict | None = None) -> dict[str, dict]:
    """text_on_busy / text_low_contrast: newly busy or low-contrast backdrops behind text."""
    from dev.distill.make_corpus import text_backdrops

    before = src_backdrops if src_backdrops is not None else text_backdrops(source, assets)
    after = text_backdrops(design, assets)
    busy = [t for t, (_, b) in after.items() if b > MAX_BUSY and before.get(t, (0, 0))[1] <= MAX_BUSY]
    low = [t for t, (r, _) in after.items() if r < MIN_CONTRAST and before.get(t, (99, 0))[0] >= MIN_CONTRAST]
    return {"text_on_busy": {"fire": bool(busy), "ids": busy},
            "text_low_contrast": {"fire": bool(low), "ids": low}}


def detect_all(source: dict, design: dict, plan, measurer, assets=None, src_backdrops=None) -> dict[str, dict]:
    """Every layer detector (pixels only when `assets` is given), plus bisect_flaws' families."""
    from dev.distill.bisect_flaws import detect as bisect_detect
    out = dict(bisect_detect(source, design, plan, measurer))
    out.update(container_flaws(source, design, plan))
    out.update(background_flaws(source, design))
    if assets is not None:
        out.update(text_flaws(source, design, assets, src_backdrops))
    return out


# ============================================================ rebuild (engine re-run)

_GATE_CALLS = {"calls": 0, "failed": 0}


def _count_gates():
    """Wrap the hard-gate check to count how many of bisection's candidates
    fail it (bisection drops those silently); calls from anywhere else are
    passed through uncounted."""
    from engine.pipeline.grading import gates
    if getattr(gates.check_gates, "_counting", False):
        return
    real = gates.check_gates

    def counting(*args, **kwargs):
        failed = real(*args, **kwargs)
        if sys._getframe(1).f_code.co_filename.endswith("bisection.py"):  # only bisection's own checks
            _GATE_CALLS["calls"] += 1
            _GATE_CALLS["failed"] += bool(failed)
        return failed

    counting._counting = True
    gates.check_gates = counting


def _init(env: dict, folders: tuple):
    from dev.distill import ab_engine
    ab_engine._worker_init(env, folders)
    _count_gates()


def _rebuild_one(job: dict) -> dict:
    from dev.distill import ab_engine
    from dev.distill.bisect_flaws import plan_to_dict
    try:
        source, target = job["source"], job["target"]
        canvas = {"width": target["width"], "height": target["height"]}
        before = dict(_GATE_CALLS)
        t0 = time.perf_counter()
        design, plan, best, reports = ab_engine._resize(source, canvas)
        secs = time.perf_counter() - t0
        from engine.pipeline.grading.gates import check_gates
        from engine.resize import _get_measurer
        failed = check_gates(source, design, canvas, plan, _get_measurer())
        return {"cell": job["cell"], "task_id": job["task_id"], "target": job["target_name"],
                "candidate_id": best.candidate_id, "is_fallback": bool(best.is_fallback),
                "design_hash": ab_engine.design_hash(design), "seconds": round(secs, 3),
                "failed_gates": failed, "bisect_gate_calls": _GATE_CALLS["calls"] - before["calls"],
                "bisect_gate_failed": _GATE_CALLS["failed"] - before["failed"],
                "bisect_candidates": sum(r.candidate_id.startswith("bisection") for r in reports),
                "design": design, "plan": plan_to_dict(plan)}
    except Exception as exc:
        return {"cell": job["cell"], "error": f"{type(exc).__name__}: {exc}"}


def cells_for(split: str, n: int | None = None, seed: int = 0) -> tuple[list[dict], dict]:
    from dev.distill import ab_engine
    from dev.distill.corpus import load_corpus
    targets = json.loads(ab_engine.TARGETS_PATH.read_text())
    if split in ab_engine.EVAL_SPLITS:
        designs = load_corpus(split)
        return ab_engine.challenge_cells(designs, targets, n, seed), {d["task_id"]: d for d in designs}
    cells = ab_engine.pool_cells(ab_engine.POOL_DIR, split, n, seed)
    return cells, {d["task_id"]: d for d in load_corpus(split)}


def rebuild(name: str, env: dict, split: str, n: int | None = None, seed: int = 0, workers: int = WORKERS,
            log=print) -> Path:
    from dev.distill import ab_engine
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    cells, designs = cells_for(split, n, seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.jsonl"
    done = {json.loads(x)["cell"] for x in path.read_text().splitlines() if x.strip()} if path.exists() else set()
    jobs = [{"cell": c["cell"], "task_id": c["task_id"], "target_name": c["target"],
             "source": designs[c["task_id"]]["source"], "target": targets[c["target"]]}
            for c in cells if c["cell"] not in done and c["task_id"] in designs and c["target"] in targets]
    log(f"[layers] {name}: {len(jobs)} cells to run ({len(done)} cached)")
    with ProcessPoolExecutor(max_workers=max(1, min(workers, WORKERS)), initializer=_init,
                             initargs=(env, folders_for(split))) as ex, path.open("a") as f:
        for row in ex.map(_rebuild_one, jobs, chunksize=4):
            if "error" in row:
                log(f"[layers] {row['cell']}: {row['error']}")
                continue
            f.write(json.dumps(row) + "\n")
    (CACHE_DIR / f"{name}.meta.json").write_text(json.dumps({"name": name, "env": env, "split": split,
                                                             "cells": n, "seed": seed}, indent=2))
    return path


def load_cache(name: str) -> dict[str, dict]:
    path = CACHE_DIR / f"{name}.jsonl"
    return {r["cell"]: r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def _meta(name: str) -> dict:
    return json.loads((CACHE_DIR / f"{name}.meta.json").read_text())


def _sources(split: str) -> dict:
    from dev.distill.corpus import load_corpus
    return {d["task_id"]: d["source"] for d in load_corpus(split)}


# ============================================================ rates

_DET = {}


def _det_init(split: str):
    from dev.distill.assets import merged_assets_dir
    from engine.pipeline.textmeasure import TextMeasurer
    import atexit
    cm = merged_assets_dir(*folders_for(split))
    _DET["assets"] = cm.__enter__()
    atexit.register(cm.__exit__, None, None, None)
    _DET["m"] = TextMeasurer()
    _DET["srcs"] = _sources(split)
    _DET["backdrops"] = {}


def _det_one(row: dict) -> tuple[str, dict]:
    from dev.distill.bisect_flaws import plan_from_dict
    from dev.distill.make_corpus import text_backdrops
    src = _DET["srcs"][row["task_id"]]
    if row["task_id"] not in _DET["backdrops"]:
        _DET["backdrops"][row["task_id"]] = text_backdrops(src, _DET["assets"])
    det = detect_all(src, row["design"], plan_from_dict(row["plan"]), _DET["m"], _DET["assets"],
                     _DET["backdrops"][row["task_id"]])
    return row["cell"], {k: bool(v["fire"]) for k, v in det.items()}


def detect_cache(name: str, rows: dict | None = None, workers: int = WORKERS) -> dict[str, dict]:
    """cell -> {family: fired}; cached next to the rebuild cache."""
    rows = rows if rows is not None else load_cache(name)
    out_path = CACHE_DIR / f"{name}.layer_det.json"
    cached = json.loads(out_path.read_text()) if out_path.exists() else {}
    cached = {c: v for c, v in cached.items() if c in rows and v.get("_hash") == rows[c]["design_hash"]}
    todo = [r for c, r in sorted(rows.items()) if c not in cached]
    if todo:
        split = _meta(name)["split"]
        todo.sort(key=lambda r: r["task_id"])
        with ProcessPoolExecutor(max_workers=max(1, min(workers, WORKERS)), initializer=_det_init,
                                 initargs=(split,)) as ex:
            for cell, det in ex.map(_det_one, todo, chunksize=8):
                cached[cell] = {**det, "_hash": rows[cell]["design_hash"]}
        out_path.write_text(json.dumps(cached))
    return {c: {k: v for k, v in d.items() if k != "_hash"} for c, d in cached.items()}


def rates(names: list[str], families=None) -> dict[str, dict]:
    out = {}
    for name in names:
        rows = load_cache(name)
        det = detect_cache(name, rows)
        n = len(rows)
        fams = sorted({k for d in det.values() for k in d})
        calls = sum(r.get("bisect_gate_calls", 0) for r in rows.values())
        out[name] = {
            "cells": n,
            "rates": {k: sum(d[k] for d in det.values()) / n for k in fams} if n else {},
            "fallback_share": sum(r["is_fallback"] for r in rows.values()) / n if n else 0.0,
            "bisection_pick_share": sum(r["candidate_id"].startswith("bisection") for r in rows.values()) / n if n else 0.0,
            "winner_gate_failures": sum(bool(r["failed_gates"]) for r in rows.values()),
            "bisect_gate_pass": 1 - sum(r.get("bisect_gate_failed", 0) for r in rows.values()) / calls if calls else None,
            "mean_seconds": sum(r["seconds"] for r in rows.values()) / n if n else 0.0,
        }
    fams = families or [*LAYER_FAMILIES, *sorted({k for o in out.values() for k in o["rates"]} - set(LAYER_FAMILIES))]
    print("| family | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in fams:
        print(f"| {k} | " + " | ".join(f"{100 * out[n]['rates'].get(k, 0):.0f}%" for n in names) + " |")
    for k in ("cells", "fallback_share", "bisection_pick_share", "winner_gate_failures", "bisect_gate_pass",
              "mean_seconds"):
        cell = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"| {k} | " + " | ".join(cell(out[n][k]) for n in names) + " |")
    return out


def changed(a: str, b: str) -> tuple[int, int]:
    """(cells whose winning design differs, cells compared) between two caches."""
    ra, rb = load_cache(a), load_cache(b)
    common = set(ra) & set(rb)
    return sum(ra[c]["design_hash"] != rb[c]["design_hash"] for c in common), len(common)


# ============================================================ contact sheet

def contact_sheet(names: list[str], out: Path, cells: list[str] | None = None, n: int = 16, seed: int = 0,
                  tile: int = 300) -> Path:
    """source | <name 1> | <name 2> ... for `cells` (default: n seeded cells whose
    designs differ between the caches); each tile labelled with its candidate
    and the layer detectors that fire. Crello splits: write it into a .nosync folder."""
    import random
    from PIL import Image, ImageDraw
    from dev.distill.assets import merged_assets_dir
    from engine.pipeline.render import render_to_image

    caches = [load_cache(nm) for nm in names]
    dets = [detect_cache(nm, c) for nm, c in zip(names, caches)]
    common = sorted(set.intersection(*(set(c) for c in caches)))
    if cells is None:
        diff = [c for c in common if len({cache[c]["design_hash"] for cache in caches}) > 1]
        cells = sorted(random.Random(seed).sample(diff, min(n, len(diff))))
    split = _meta(names[0])["split"]
    srcs = _sources(split)
    label_h = 30
    rows = []
    with merged_assets_dir(*folders_for(split)) as assets:
        for c in cells:
            tid = caches[0][c]["task_id"]
            imgs = [("source", render_to_image(srcs[tid], assets_dir=assets, max_size=tile))]
            for nm, cache, det in zip(names, caches, dets):
                fired = ",".join(k for k in LAYER_FAMILIES if det[c].get(k))
                imgs.append((f"{nm}: {cache[c]['candidate_id'][:34]}\n{fired}",
                             render_to_image(cache[c]["design"], assets_dir=assets, max_size=tile)))
            w = sum(im.width for _, im in imgs) + 10 * len(imgs)
            h = max(im.height for _, im in imgs) + label_h + 16
            row = Image.new("RGB", (max(w, 400), h), (225, 225, 225))
            d = ImageDraw.Draw(row)
            d.text((4, 2), c, fill="black")
            x = 0
            for lab, im in imgs:
                d.text((x + 4, 14), lab, fill=(40, 40, 160))
                row.paste(im.convert("RGB"), (x, label_h + 12))
                x += im.width + 10
            rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 6 for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("rebuild")
    r.add_argument("--name", required=True)
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--split", required=True, choices=SPLITS)
    r.add_argument("--cells", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=WORKERS)
    s = sub.add_parser("rates")
    s.add_argument("--name", action="append", required=True)
    c = sub.add_parser("sheet")
    c.add_argument("--name", action="append", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--cell", action="append", default=None)
    c.add_argument("--n", type=int, default=16)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--tile", type=int, default=300)
    d = sub.add_parser("changed")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    args = ap.parse_args(argv)
    if args.step == "rebuild":
        from dev.distill.ab_engine import parse_env
        print(rebuild(args.name, parse_env(args.env), args.split, args.cells, args.seed, args.workers))
    elif args.step == "rates":
        rates(args.name)
    elif args.step == "sheet":
        print(contact_sheet(args.name, args.out, args.cell, args.n, args.seed, args.tile))
    else:
        k, n = changed(args.a, args.b)
        print(f"{k} of {n} cells changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
