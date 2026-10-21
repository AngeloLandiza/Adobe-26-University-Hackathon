"""Measurements for the three experiments from docs/prior-work.md (CPU only,
no teacher calls): hierarchy lock, near-aspect source fit, relation
preservation. See docs/optimization-program.md, "Experiments from prior work".

    # engine re-run with the grader's view of the winner (total, degraded units) stored;
    # rows are readable by bisect_flaws rates / sheet (same cache folder and fields)
    python -m dev.distill.prior_experiments rebuild --name X --env K=V ... --split train --cells 60
    python -m dev.distill.prior_experiments rebuild --name base_jd --env CHAMELEON_FLEX=0 --split judged_ab
    # relation score vs the teacher: train judgments (choose w) and the judged A/B cells (check only)
    python -m dev.distill.prior_experiments relval --a-cache base_jd --b-cache bisect_jd
    # near-aspect fallback margin from train judgments
    python -m dev.distill.prior_experiments nearfit
    # picks that differ between runs (winners.jsonl or caches), by design_hash
    python -m dev.distill.prior_experiments changed --base bisect2 --name b2_hier
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dev.distill.bisect_flaws import CACHE_DIR, cells_for, load_cache, plan_from_dict, plan_to_dict
from dev.distill.paths import DATA_ROOT

AB_ROOT = DATA_ROOT / "ab"
JUDGED_AB = (("base__vs__bisect", "holdout"), ("base_chl__vs__bisect_chl", "challenge"))
TRAIN_JUDGMENTS = DATA_ROOT / "judgments" / "judgments.jsonl"
POOL = DATA_ROOT / "candidate-pool" / "pool.jsonl"
NEAR = 1.35
W_GRID = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
MAX_WORKERS = 3


def aspect_ratio(src_canvas: dict, target: dict) -> float:
    a = src_canvas["width"] / src_canvas["height"]
    b = target["width"] / target["height"]
    return max(a / b, b / a)


# ============================================================ rebuild

def _init(env: dict):
    from dev.distill import ab_engine
    ab_engine._worker_init(env)


def _one(job: dict) -> dict:
    from dev.distill import ab_engine
    try:
        t0 = time.perf_counter()
        design, plan, best, reports = ab_engine._resize(job["source"], {"width": job["target"]["width"],
                                                                         "height": job["target"]["height"]})
        secs = time.perf_counter() - t0
        rep = next((r for r in reports if r.candidate_id == best.candidate_id), None)
        return {"cell": job["cell"], "task_id": job["task_id"], "target": job["target_name"],
                "candidate_id": best.candidate_id, "design_hash": ab_engine.design_hash(design),
                "seconds": round(secs, 3), "design": design, "plan": plan_to_dict(plan),
                "total": rep.total if rep else None, "degraded": list(best.degraded_unit_ids),
                "is_fallback": bool(best.is_fallback),
                "n_bisection": sum(r.candidate_id.startswith("bisection/") for r in reports),
                "n_bisection_pass": sum(r.candidate_id.startswith("bisection/") and r.passed_gates for r in reports)}
    except Exception as exc:
        return {"cell": job["cell"], "error": f"{type(exc).__name__}: {exc}"}


def judged_ab_cells() -> tuple[list[dict], dict]:
    from dev.distill.corpus import load_corpus
    designs = {d["task_id"]: d for d in load_corpus("holdout") + load_corpus("challenge")}
    cells = {}
    for pair, split in JUDGED_AB:
        for line in (AB_ROOT / pair / "judgments.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                tid, tname = r["cell"].split("@")
                cells[r["cell"]] = {"cell": r["cell"], "task_id": tid, "target": tname, "split": split}
    return [cells[c] for c in sorted(cells)], designs


def rebuild(name: str, env: dict, split: str, n: int | None = None, seed: int = 0, workers: int = MAX_WORKERS,
            log=print) -> Path:
    from dev.distill import ab_engine
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    cells, designs = judged_ab_cells() if split == "judged_ab" else cells_for(split, n, seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.jsonl"
    done = {json.loads(x)["cell"] for x in path.read_text().splitlines() if x.strip()} if path.exists() else set()
    jobs = [{"cell": c["cell"], "task_id": c["task_id"], "target_name": c["target"],
             "source": designs[c["task_id"]]["source"], "target": targets[c["target"]]}
            for c in cells if c["cell"] not in done and c["task_id"] in designs and c["target"] in targets]
    log(f"[prior] {name}: {len(jobs)} cells to run ({len(done)} cached)")
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS), initializer=_init, initargs=(env,)) as ex, \
            path.open("a") as f:
        for row in ex.map(_one, jobs):
            if "error" in row:
                log(f"[prior] {row['cell']}: {row['error']}")
                continue
            f.write(json.dumps(row) + "\n")
    (CACHE_DIR / f"{name}.meta.json").write_text(json.dumps({"name": name, "env": env, "split": split,
                                                             "cells": n, "seed": seed}, indent=2))
    return path


# ============================================================ relation score vs the teacher

def _sources() -> dict:
    from dev.distill.corpus import load_corpus
    return {d["task_id"]: d["source"] for d in load_corpus(None, include_challenge=True)}


def _agree(rows: list[dict], w: float) -> tuple[int, int]:
    """(agreements, decided) of the blended score total - w (1 - rel) with the teacher."""
    hit = n = 0
    for r in rows:
        sa = r["total_a"] - w * (1 - r["rel_a"])
        sb = r["total_b"] - w * (1 - r["rel_b"])
        if sa == sb:
            continue
        n += 1
        hit += (sa > sb) == (r["winner"] == "a")
    return hit, n


def _rel_only(rows: list[dict]) -> tuple[int, int]:
    hit = n = 0
    for r in rows:
        if r["rel_a"] == r["rel_b"]:
            continue
        n += 1
        hit += (r["rel_a"] > r["rel_b"]) == (r["winner"] == "a")
    return hit, n


def judged_rows(a_cache: str, b_cache: str, min_conf: int = 1) -> list[dict]:
    """Each judged A/B row (base vs bisect) with both sides rebuilt: grader
    total from the side's own engine run and the relation score."""
    from dev.distill import ab_engine
    from engine.pipeline.grading.relations import relation_score
    from engine.pipeline.textmeasure import TextMeasurer
    m = TextMeasurer()
    srcs = _sources()
    caches = {"base": load_cache(a_cache), "bisect": load_cache(b_cache)}
    winners = {}
    for v in ("base", "bisect", "base_chl", "bisect_chl"):
        winners[v] = ab_engine.load_winners(v)
    out, mismatched = [], 0
    for pair, _ in JUDGED_AB:
        for line in (AB_ROOT / pair / "judgments.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["winner"] not in ("a", "b") or r.get("confidence", 0) < min_conf:
                continue
            sides = {}
            for side in ("a", "b"):
                variant = r[side]
                fam = "bisect" if variant.startswith("bisect") else "base"
                row = caches[fam].get(r["cell"])
                if row is None:
                    break
                if winners[variant].get(r["cell"], {}).get("design_hash") != row["design_hash"]:
                    mismatched += 1
                plan = plan_from_dict(row["plan"])
                sides[side] = (row["total"], relation_score(srcs[row["task_id"]], row["design"], plan, m,
                                                            skip=row.get("degraded", ())), variant)
            if len(sides) < 2:
                continue
            out.append({"cell": r["cell"], "winner": r["winner"], "confidence": r.get("confidence"),
                        "total_a": sides["a"][0], "total_b": sides["b"][0], "rel_a": sides["a"][1],
                        "rel_b": sides["b"][1], "a": sides["a"][2], "b": sides["b"][2]})
    print(f"[relval] judged A/B rows: {len(out)} (sides whose rebuilt hash differs from winners.jsonl: {mismatched})")
    return out


def _train_cell(job: dict) -> list[dict]:
    """Regenerate one train cell's candidates (base engine) and score the judged ones."""
    from engine.pipeline.grading.metrics import geometry_metrics
    from engine.pipeline.grading.relations import relation_score
    from engine.pipeline.layout.fallback import safe_fallback
    from engine.pipeline.layout.generate import generate_candidates
    from engine.pipeline.priority import heuristic_plan
    from engine.resize import _get_measurer
    import os
    m = _get_measurer()
    source, canvas = job["source"], job["canvas"]
    plan = heuristic_plan(source, os.environ.get("CHAMELEON_ASSETS_DIR"))
    cands = {c.candidate_id: c for c in [safe_fallback(source, canvas, plan, m)] + generate_candidates(source, canvas, plan, m)}
    out = {}
    for cid in job["ids"]:
        c = cands.get(cid)
        if c is None:
            continue
        geo = geometry_metrics(source, c, plan)
        out[cid] = {"rel": relation_score(source, c.design, plan, m, skip=c.degraded_unit_ids),
                    "geo_match": all(abs(geo[k] - job["pool"][cid][k]) < 1e-6 for k in geo)}
    return [{"cell": job["cell"], "scores": out}]


def train_rows(workers: int = MAX_WORKERS) -> list[dict]:
    from dev.distill import ab_engine
    from engine.pipeline.grading.metrics import geometry_total
    from engine.pipeline.grading.pixels import pixel_total
    from engine.pipeline.grading.weights import DEFAULT_WEIGHTS as W
    cache = CACHE_DIR / "relval_train.json"
    judg = [json.loads(x) for x in TRAIN_JUDGMENTS.read_text().splitlines() if x.strip()]
    judg = [r for r in judg if r["split"] == "train" and r["winner"] in ("a", "b")]
    pool = {}
    for line in POOL.read_text().splitlines():
        r = json.loads(line)
        if r["split"] == "train":
            pool[(r["cell"], r["candidate_id"])] = r["metrics"]
    if cache.exists():
        scores = json.loads(cache.read_text())
    else:
        targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
        srcs = _sources()
        ids = defaultdict(set)
        for r in judg:
            ids[r["cell"]] |= {r["a"], r["b"]}
        jobs = []
        for cell, cids in sorted(ids.items()):
            tid, tname = cell.split("@")
            t = targets[tname]
            jobs.append({"cell": cell, "source": srcs[tid], "canvas": {"width": t["width"], "height": t["height"]},
                         "ids": sorted(cids), "pool": {c: pool[(cell, c)] for c in cids if (cell, c) in pool}})
        scores = {}
        with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS), initializer=_init,
                                 initargs=({"CHAMELEON_FLEX": "0"},)) as ex:
            for res in ex.map(_safe_train_cell, jobs):
                for row in res:
                    scores[row["cell"]] = row["scores"]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(scores))

    def total(metrics):
        return W.geometry_share * geometry_total(metrics, W.geometry) + (1 - W.geometry_share) * pixel_total(metrics, W.pixels)

    out, missing, geo_ok = [], 0, 0
    for r in judg:
        s = scores.get(r["cell"], {})
        if r["a"] not in s or r["b"] not in s or (r["cell"], r["a"]) not in pool or (r["cell"], r["b"]) not in pool:
            missing += 1
            continue
        geo_ok += s[r["a"]]["geo_match"] and s[r["b"]]["geo_match"]
        out.append({"cell": r["cell"], "winner": r["winner"], "confidence": r.get("confidence"),
                    "total_a": total(pool[(r["cell"], r["a"])]), "total_b": total(pool[(r["cell"], r["b"])]),
                    "rel_a": s[r["a"]]["rel"], "rel_b": s[r["b"]]["rel"], "a": r["a"], "b": r["b"]})
    print(f"[relval] train rows: {len(out)} (missing {missing}; geometry re-computed equal to pool on {geo_ok})")
    return out


def _safe_train_cell(job):
    try:
        return _train_cell(job)
    except Exception as exc:
        print(f"[relval] {job['cell']}: {type(exc).__name__}: {exc}")
        return []


def relval(a_cache: str, b_cache: str) -> dict:
    train = train_rows()

    def table(rows, label):
        print(f"\n{label}: {len(rows)} decided rows")
        g = _agree(rows, 0.0)
        r = _rel_only(rows)
        print(f"  grader total alone: {g[0]}/{g[1]} = {g[0] / max(g[1], 1):.1%}")
        print(f"  relation score alone: {r[0]}/{r[1]} = {r[0] / max(r[1], 1):.1%} "
              f"(ties in relation score: {len(rows) - r[1]})")
        for w in W_GRID[1:]:
            h, n = _agree(rows, w)
            print(f"  total - {w:g} x (1 - rel): {h}/{n} = {h / max(n, 1):.1%}")

    table(train, "TRAIN judgments (template candidates within one engine run)")
    best = max(W_GRID, key=lambda w: (_agree(train, w)[0] / max(_agree(train, w)[1], 1), -w))
    print(f"\nw chosen on train: {best:g}")
    judged = judged_rows(a_cache, b_cache)
    table(judged, "JUDGED A/B cells (base vs bisect winners; check only, nothing tuned here)")
    conf = [r for r in judged if (r["confidence"] or 0) >= 2]
    table(conf, "JUDGED A/B, confidence >= 2")
    bis_won = sum((r["winner"] == "a") == r["a"].startswith("bisect") for r in judged)
    print(f"\n(teacher picked the bisection side in {bis_won}/{len(judged)} judged rows)")
    rel_pref = Counter((r["rel_a"] > r["rel_b"]) == r["a"].startswith("bisect") for r in judged if r["rel_a"] != r["rel_b"])
    print(f"(relation score prefers the bisection side in {rel_pref[True]}/{sum(rel_pref.values())})")
    return {"w": best}


# ============================================================ near-aspect fallback margin

def nearfit() -> None:
    from engine.pipeline.grading.metrics import geometry_total
    from engine.pipeline.grading.pixels import pixel_total
    from engine.pipeline.grading.weights import DEFAULT_WEIGHTS as W
    from dev.distill import ab_engine
    srcs = {tid: s["canvas"] for tid, s in _sources().items()}
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    pool = {}
    for line in POOL.read_text().splitlines():
        r = json.loads(line)
        pool[(r["cell"], r["candidate_id"])] = r["metrics"]

    def total(m):
        return W.geometry_share * geometry_total(m, W.geometry) + (1 - W.geometry_share) * pixel_total(m, W.pixels)

    for split in ("train", "holdout"):
        groups = defaultdict(list)
        for line in TRAIN_JUDGMENTS.read_text().splitlines():
            r = json.loads(line)
            if r["split"] != split or "fallback" not in (r["a"], r["b"]) or r["winner"] not in ("a", "b"):
                continue
            tid, tname = r["cell"].split("@")
            other = r["b"] if r["a"] == "fallback" else r["a"]
            fb_won = (r["winner"] == "a") == (r["a"] == "fallback")
            d = total(pool[(r["cell"], other)]) - total(pool[(r["cell"], "fallback")])
            groups["near" if aspect_ratio(srcs[tid], targets[tname]) <= NEAR else "far"].append((d, fb_won))
        for k, v in sorted(groups.items()):
            print(f"{split} {k}: {len(v)} judged fallback pairs, fallback won {sum(x[1] for x in v)}")
            if k == "near":
                for margin in (0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2):
                    right = sum((d > margin) != fb for d, fb in v)
                    print(f"   margin {margin:.2f}: rule 'fallback unless the other leads by > margin' "
                          f"matches the teacher {right}/{len(v)}")
                print("   (other - fallback grader lead, fallback won): " + ", ".join(f"({d:+.3f},{int(fb)})" for d, fb in sorted(v)))


# ============================================================ changed picks

def _hashes(name: str) -> dict[str, str]:
    from dev.distill import ab_engine
    w = ab_engine.load_winners(name)
    if w:
        return {c: r["design_hash"] for c, r in w.items()}
    return {c: r["design_hash"] for c, r in load_cache(name).items()}


def changed(base: str, names: list[str]) -> None:
    b = _hashes(base)
    for n in names:
        h = _hashes(n)
        common = set(b) & set(h)
        diff = sum(b[c] != h[c] for c in common)
        print(f"{n} vs {base}: {diff} of {len(common)} picks differ")


def summary(names: list[str], base: str | None = None) -> None:
    """Runtime, fallback share, final-design gate failures and bisection gate pass rate per cache."""
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.textmeasure import TextMeasurer
    m = TextMeasurer()
    srcs = _sources()
    from dev.distill import ab_engine
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    ref = load_cache(base) if base else None
    for n in names:
        rows = load_cache(n)
        fails = 0
        for r in rows.values():
            t = targets[r["target"]]
            fails += bool(check_gates(srcs[r["task_id"]], r["design"], {"width": t["width"], "height": t["height"]},
                                      plan_from_dict(r["plan"]), m))
        nb = sum(r.get("n_bisection", 0) for r in rows.values())
        nbp = sum(r.get("n_bisection_pass", 0) for r in rows.values())
        secs = sum(r["seconds"] for r in rows.values()) / max(len(rows), 1)
        fb = sum(r.get("is_fallback", r["candidate_id"] == "fallback") for r in rows.values())
        line = (f"{n}: {len(rows)} cells, {secs:.2f} s/cell, fallback picked {fb}, final designs failing gates {fails}, "
                f"bisection candidates passing gates {nbp}/{nb}")
        if ref is not None:
            common = set(ref) & set(rows)
            line += f", picks differing from {base}: {sum(ref[c]['design_hash'] != rows[c]['design_hash'] for c in common)}/{len(common)}"
        print(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("rebuild")
    r.add_argument("--name", required=True)
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--split", default="train", choices=["judged_ab", "train", "holdout"])
    r.add_argument("--cells", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=MAX_WORKERS)
    v = sub.add_parser("relval")
    v.add_argument("--a-cache", default="base_jd")
    v.add_argument("--b-cache", default="bisect_jd")
    sub.add_parser("nearfit")
    c = sub.add_parser("changed")
    c.add_argument("--base", required=True)
    c.add_argument("--name", action="append", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--name", action="append", required=True)
    s.add_argument("--base", default=None)
    args = ap.parse_args(argv)
    if args.step == "rebuild":
        from dev.distill.ab_engine import parse_env
        print(rebuild(args.name, parse_env(args.env), args.split, args.cells, args.seed, args.workers))
    elif args.step == "relval":
        relval(args.a_cache, args.b_cache)
    elif args.step == "nearfit":
        nearfit()
    elif args.step == "changed":
        changed(args.base, args.name)
    else:
        summary(args.name, args.base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
