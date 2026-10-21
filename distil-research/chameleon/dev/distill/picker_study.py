"""Picker study: when the engine has several good candidates, does pick_best
choose the one the teacher prefers?

For a sample of cells the engine (default: the v3L switches) runs as in the
A/B harness (CHAMELEON_LLM=heuristic, the split's merged assets folder), and
pick_best's inputs and answer are captured. Of the gate-passing candidates,
K (default 4) are kept: the grader's pick, then the best by grader total
(candidates pick_best fully graded, i.e. rendered, first) with at most
MAX_PER_FAMILY from one strategy family (the candidate's template: row,
stack, bisection, ...), and the uniform-fit fallback whenever it passes the
gates. They are rendered, and the teacher RANKS them in one call per cell:
the source render, then the candidates in a seeded random order, labelled
1..K, under the harsh rubric (rubric.md), with a 1-5 score and a one-line
reason each.

    cd distil-research/chameleon
    python -m dev.distill.picker_study prepare --name pk_hold --split holdout --cells 30 [--k 4] [--workers 3] [--env K=V ...]
    python -m dev.distill.picker_study rank --name pk_hold [--shard K/N] [--limit-calls N] [--model claude-opus-5-5]
    python -m dev.distill.picker_study rank --name pk_hold --merge
    python -m dev.distill.picker_study report --name pk_hold

`prepare` is CPU only (<= 3 processes) and resumable; `rank` is resumable
and shardable like judge.py (rankings.shardKofN.jsonl, then --merge).
`report` writes REPORT.md: how often the grader's pick is the teacher's top
choice, the teacher's mean score of the grader's pick vs of its best
candidate (the headroom a better picker could win), and which strategy
families the teacher prefers.

Each rankings.jsonl row carries, per candidate, the features a learned
picker would train on: the grader's metrics and total, the relation score
(grading/relations.py) and its parts, and the bisect_flaws / layer_flaws
detector outputs.

Files: training-data/picker/<name>/ (study.json, REPORT.md); the rows
(candidates.jsonl, rankings*.jsonl) and renders. For Crello splits
(crello_dev) the rows, which quote the teacher's reasons about real ads, go
in <name>/private.nosync/ (gitignored); renders always go in
renders.nosync/. Evaluation-only splits (challenge, challenge2, challenge3)
are refused: a picker must not be studied or trained on them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from dev.distill import ab_engine
from dev.distill import judge as judge_mod
from dev.distill.paths import DATA_ROOT
from dev.distill.teacher import TeacherUnavailable

PICKER_ROOT = DATA_ROOT / "picker"
V3L_ENV = {"CHAMELEON_STRATEGIES": "bisection", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
           "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1"}
SPLITS = ("holdout", "train", "all", "crello_dev", "samples")
CRELLO_SPLITS = ("crello_dev",)
K = 4
MAX_PER_FAMILY = 2
RENDER_SIZE = ab_engine.RENDER_SIZE
RENDERS = "renders.nosync"
WORKERS = 3


# ---------------------------------------------------------------- places

def study_dir(name: str, root: Path = PICKER_ROOT) -> Path:
    return Path(root) / name


def data_dir(name: str, root: Path = PICKER_ROOT, split: str | None = None) -> Path:
    """Where the rows go: <name>/private.nosync/ for Crello splits, else <name>/."""
    if split is None:
        meta = study_dir(name, root) / "study.json"
        split = json.loads(meta.read_text())["split"] if meta.exists() else None
    return study_dir(name, root) / ("private.nosync" if split in CRELLO_SPLITS else "")


def _read_jsonl(path: Path) -> list[dict]:
    return ab_engine._read_jsonl(path)


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ---------------------------------------------------------------- choosing the K

# The shortlist lives in the engine (CHAMELEON_PICKER=learned rebuilds it at pick
# time, so its features match the studies'); the study shows the teacher the same K.
from engine.pipeline.grading.learned_picker import family, shortlist as choose  # noqa: E402,F401


# ---------------------------------------------------------------- prepare (CPU)

@contextlib.contextmanager
def capture_pick():
    """Wrap engine.resize's pick_best: yields a dict that receives the
    candidates, the winner and the reports of the last call."""
    import engine.resize as R
    real, box = R.pick_best, {}

    def wrapped(source, target, plan, candidates, measurer, *a, **kw):
        winner, reports = real(source, target, plan, candidates, measurer, *a, **kw)
        box.update(candidates=list(candidates), winner=winner, reports=reports, plan=plan)
        return winner, reports

    R.pick_best = wrapped
    try:
        yield box
    finally:
        R.pick_best = real


def features(source: dict, cand, report, plan, measurer, assets, src_backdrops) -> dict:
    """What a learned picker would see: grader metrics (pixel metrics added
    when pick_best didn't render this one), relation score and parts, and
    the bisect_flaws / layer_flaws detectors."""
    from dev.distill.layer_flaws import detect_all
    from engine.pipeline.grading import relations

    metrics = dict(report.metrics)
    added = False
    if not report.rendered:
        try:
            from engine.pipeline.grading.pixels import pixel_metrics
            metrics.update(pixel_metrics(cand.design, plan))
            added = True
        except Exception:
            pass
    out = {"metrics": metrics, "pixel_metrics_added": added, "degraded_units": len(cand.degraded_unit_ids)}
    try:
        out["relation"] = relations.relation_score(source, cand.design, plan, measurer, skip=cand.degraded_unit_ids)
        out["relation_parts"] = {k: list(v) for k, v in
                                 relations.relation_parts(source, cand.design, plan, measurer,
                                                          cand.degraded_unit_ids).items()}
    except Exception as exc:
        out["relation_error"] = type(exc).__name__
    try:
        det = detect_all(source, cand.design, plan, measurer, assets, src_backdrops)
        out["flaws"] = {k: bool(v.get("fire")) for k, v in det.items()}
        out["flaw_details"] = _jsonable(det)
    except Exception as exc:
        out["flaws_error"] = f"{type(exc).__name__}: {exc}"
    return _jsonable(out)


def prepare_cell(job: dict) -> dict:
    """Run the engine on one cell, choose and render the K, return its candidates.jsonl row."""
    from dev.distill.make_corpus import text_backdrops
    from engine.pipeline.render import render_to_image
    from engine.resize import _get_measurer

    cell, source, target = job["cell"], job["source"], job["target_canvas"]
    canvas = {"width": target["width"], "height": target["height"]}
    assets = ab_engine._assets()
    t0 = time.perf_counter()
    with capture_pick() as box:
        design, plan, best, _ = ab_engine._resize(source, canvas)
    seconds = time.perf_counter() - t0
    row = {"cell": cell["cell"], "task_id": cell["task_id"], "split": cell["split"], "brand": cell.get("brand"),
           "target": cell["target"], "seconds": round(seconds, 3), "engine_final": best.candidate_id,
           "engine_final_hash": ab_engine.design_hash(design)}
    if "candidates" not in box:
        return {**row, "skip": "pick_best not reached"}
    cands, reports, pick = box["candidates"], box["reports"], box["winner"]
    row.update(grader_pick=pick.candidate_id, n_candidates=len(cands),
               n_passing=sum(r.passed_gates for r in reports))
    chosen = choose(cands, reports, pick, job["k"])
    if len(chosen) < 2:
        return {**row, "skip": f"only {len(chosen)} gate-passing candidate(s)"}
    by_id = {r.candidate_id: r for r in reports}
    measurer = _get_measurer()
    src_backdrops = text_backdrops(source, assets)
    out_dir = Path(job["out_dir"])
    src_png = Path(RENDERS) / cell["task_id"] / "source.png"
    if not (out_dir / src_png).exists():
        (out_dir / src_png).parent.mkdir(parents=True, exist_ok=True)
        render_to_image(source, assets_dir=assets, max_size=RENDER_SIZE).save(out_dir / src_png)
    shown = []
    for j, i in enumerate(chosen):
        c, r = cands[i], by_id[cands[i].candidate_id]
        png = Path(RENDERS) / cell["task_id"] / cell["target"] / f"c{j}.png"
        (out_dir / png).parent.mkdir(parents=True, exist_ok=True)
        render_to_image(c.design, assets_dir=assets, max_size=RENDER_SIZE).save(out_dir / png)
        shown.append({"idx": j, "candidate_id": c.candidate_id, "family": family(c), "template": c.template,
                      "is_fallback": bool(c.is_fallback), "is_pick": c is pick or c.candidate_id == pick.candidate_id,
                      "grader_total": round(r.total, 6), "rendered_by_grader": bool(r.rendered),
                      "design_hash": ab_engine.design_hash(c.design), "png": png.as_posix(),
                      "features": features(source, c, r, plan, measurer, assets, src_backdrops)})
    rng = random.Random(f"{job['seed']}:{cell['cell']}")
    order = list(range(len(shown)))
    rng.shuffle(order)
    return {**row, "source_png": src_png.as_posix(), "candidates": shown, "shown_order": order}


def _safe_prepare(job: dict) -> dict:
    try:
        return prepare_cell(job)
    except Exception as exc:
        return {"cell": job["cell"]["cell"], "error": f"{type(exc).__name__}: {exc}"}


def prepare(name: str, split: str, n: int | None = None, seed: int = 0, k: int = K, env: dict | None = None,
            workers: int = WORKERS, root: Path = PICKER_ROOT, cells: list[dict] | None = None,
            designs: dict | None = None, folders: tuple | None = None, log=print) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"split {split!r} not allowed (evaluation-only splits are never studied); use one of {SPLITS}")
    env = dict(V3L_ENV if env is None else env)
    if cells is None:
        cells, designs, folders = ab_engine.split_cells(split, n, seed)
    folders = folders or ()
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    sdir, ddir = study_dir(name, root), data_dir(name, root, split)
    ddir.mkdir(parents=True, exist_ok=True)
    meta = sdir / "study.json"
    if meta.exists():
        old = json.loads(meta.read_text())
        if old.get("split") != split or old.get("env") != env or old.get("k") != k:
            raise ValueError(f"{name} was prepared with {old}; use another --name")
    meta.write_text(json.dumps({"name": name, "split": split, "env": env, "k": k, "seed": seed,
                                "max_per_family": MAX_PER_FAMILY}, indent=2))
    out = ddir / "candidates.jsonl"
    done = {r["cell"] for r in _read_jsonl(out)}
    jobs = [{"cell": c, "source": designs[c["task_id"]]["source"], "target_canvas": targets[c["target"]],
             "k": k, "seed": seed, "out_dir": str(sdir)}
            for c in cells if c["cell"] not in done and c["task_id"] in designs and c["target"] in targets]
    added = []

    def record(row):
        if "error" in row:
            log(f"[picker] {row['cell']}: error ({row['error']})")
            return
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        added.append(row)
        log(f"[picker] {row['cell']}: " + (row.get("skip") or f"{len(row['candidates'])} candidates"))

    if workers <= 1:
        saved = {key: os.environ.get(key) for key in [*env, "CHAMELEON_LLM", "CHAMELEON_ASSETS_DIR"]}
        ab_engine._worker_init(env, folders)
        try:
            for job in jobs:
                record(_safe_prepare(job))
        finally:
            for key, v in saved.items():
                if v is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = v
    else:
        with ProcessPoolExecutor(max_workers=min(workers, WORKERS), initializer=ab_engine._worker_init,
                                 initargs=(env, folders)) as ex:
            for row in ex.map(_safe_prepare, jobs):
                record(row)
    return added


# ---------------------------------------------------------------- rank (teacher)

def schema(n: int) -> dict:
    cand = {"type": "object", "required": ["candidate", "score", "reason"],
            "properties": {"candidate": {"type": "integer"}, "score": {"enum": [1, 2, 3, 4, 5]},
                           "reason": {"type": "string"}}}
    return {"type": "object", "required": ["candidates", "ranking"],
            "properties": {"candidates": {"type": "array", "items": cand, "minItems": n, "maxItems": n},
                           "ranking": {"type": "array", "items": {"type": "integer"}, "minItems": n, "maxItems": n}}}


def prompt(n: int) -> str:
    return (f"{judge_mod.RUBRIC}\n\nImage 1: the ORIGINAL design. Images 2 to {n + 1}: candidates 1 to {n}, "
            f"all resizes of it to the same new canvas (image k+1 is candidate k).\n\n"
            f"For each candidate give its ship-readiness score (1-5, absolute) and a one-line reason naming its "
            f"specific flaws (which element, what is wrong). Then rank all {n} candidates from best to worst: "
            f"`ranking` lists the candidate numbers, best first, each exactly once. No ties in the ranking: if two "
            f"look equally good, still order them; the ranking must agree with your scores. "
            f"Read all of the listed image files, in parallel in one step, before you answer.")


def validator(n: int) -> Callable[[dict], None]:
    def check(answer: dict) -> None:
        if not isinstance(answer, dict):
            raise ValueError("answer is not an object")
        ranking, cands = answer.get("ranking"), answer.get("candidates")
        if not isinstance(ranking, list) or any(type(x) is not int for x in ranking) or \
                sorted(ranking) != list(range(1, n + 1)):
            raise ValueError(f"ranking {ranking!r} is not a permutation of 1..{n}")
        if not isinstance(cands, list):
            raise ValueError("no candidates list")
        seen = []
        for c in cands:
            if not isinstance(c, dict) or type(c.get("candidate")) is not int:
                raise ValueError(f"bad candidate entry {c!r}")
            if type(c.get("score")) is not int or not 1 <= c["score"] <= 5:
                raise ValueError(f"bad score {c.get('score')!r}")
            if not isinstance(c.get("reason"), str):
                raise ValueError("reason is not a string")
            seen.append(c["candidate"])
        if sorted(seen) != list(range(1, n + 1)):
            raise ValueError(f"candidate numbers {sorted(seen)} do not match 1..{n}")
    return check


def ranking_files(name: str, root: Path = PICKER_ROOT) -> list[Path]:
    main = data_dir(name, root) / "rankings.jsonl"
    return [main, *sorted(main.parent.glob("rankings.shard*of*.jsonl"))]


def load_rankings(name: str, root: Path = PICKER_ROOT, teacher: str | None = None) -> dict[str, dict]:
    out = {}
    for f in ranking_files(name, root):
        for r in _read_jsonl(f):
            if r.get("rubric_version") == judge_mod.RUBRIC_VERSION and (teacher is None or r["teacher"] == teacher):
                out[r["cell"]] = r
    return out


def rank(name: str, teacher, root: Path = PICKER_ROOT, shard=None, limit_calls: int | None = None,
         log=print) -> list[dict]:
    """One teacher call per prepared cell not yet ranked (by this teacher, under this rubric)."""
    sdir, ddir = study_dir(name, root), data_dir(name, root)
    main = ddir / "rankings.jsonl"
    out = judge_mod.shard_file(main, *shard) if shard else main
    done = set(load_rankings(name, root, teacher.name))
    todo = [r for r in _read_jsonl(ddir / "candidates.jsonl")
            if "candidates" in r and r["cell"] not in done and judge_mod._in_shard(r["cell"], shard)]
    added, calls = [], 0
    for row in todo:
        if limit_calls is not None and calls >= limit_calls:
            break
        calls += 1
        n, order = len(row["candidates"]), row["shown_order"]
        sources = [sdir / row["source_png"]] + [sdir / row["candidates"][i]["png"] for i in order]
        with tempfile.TemporaryDirectory(prefix="picker-") as tmp:
            images = []  # neutral names: nothing hints at the grader's pick or the fallback
            for i, src in enumerate(sources, 1):
                dest = Path(tmp) / f"image_{i}.png"
                shutil.copyfile(src, dest)
                images.append(dest)
            check = validator(n)
            try:
                answer = teacher.ask(prompt(n), schema(n), images, validate=check)
                check(answer)
            except TeacherUnavailable as e:
                log(f"[picker] {e}; stopping, progress kept")
                break
            except Exception as e:
                log(f"[picker] teacher gave up on {row['cell']} ({type(e).__name__}: {e}); skipping it")
                continue
        result = ranking_row(row, answer, teacher.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps(result) + "\n")
        added.append(result)
    return added


def ranking_row(row: dict, answer: dict, teacher_name: str) -> dict:
    """Map the teacher's labels (shown positions 1..n) back to the candidates
    (grader order) and put its ranks, scores and reasons beside their features."""
    order = row["shown_order"]  # shown position p (0-based) -> candidate idx
    by_label = {c["candidate"]: c for c in answer["candidates"]}
    ranking = [order[label - 1] for label in answer["ranking"]]
    cands = []
    for c in row["candidates"]:
        label = order.index(c["idx"]) + 1
        cands.append({**c, "label": label, "teacher_score": by_label[label]["score"],
                      "teacher_rank": ranking.index(c["idx"]) + 1, "reason": by_label[label]["reason"]})
    scores = [c["teacher_score"] for c in sorted(cands, key=lambda c: c["teacher_rank"])]
    return {"cell": row["cell"], "task_id": row["task_id"], "split": row["split"], "brand": row.get("brand"),
            "target": row["target"], "teacher": teacher_name, "rubric_version": judge_mod.RUBRIC_VERSION,
            "k": len(cands), "ranking": ranking, "consistent": scores == sorted(scores, reverse=True),
            "grader_pick": row["grader_pick"], "engine_final": row.get("engine_final"), "candidates": cands}


# ---------------------------------------------------------------- report

def kendall_tau(a: list, b: list) -> float:
    """Kendall's tau between two orderings of the same items."""
    pos = {x: i for i, x in enumerate(b)}
    conc = disc = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if pos[a[i]] < pos[a[j]]:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else float("nan")


def compute_report(rows: list[dict]) -> dict:
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    n = len(rows)
    agree, pick_scores, best_scores, pick_ranks, taus, chance, fb_top, fb_seen = [], [], [], [], [], [], 0, 0
    fam = defaultdict(lambda: {"shown": 0, "top": 0, "scores": [], "ranks": [], "picked_by_grader": 0})
    for r in rows:
        cands = r["candidates"]
        pick = next(c for c in cands if c["is_pick"])
        top = min(cands, key=lambda c: c["teacher_rank"])
        agree.append(pick is top)
        chance.append(1 / len(cands))
        pick_scores.append(pick["teacher_score"])
        best_scores.append(max(c["teacher_score"] for c in cands))
        pick_ranks.append(pick["teacher_rank"])
        grader_order = [c["idx"] for c in sorted(cands, key=lambda c: (not c["is_pick"], not c["rendered_by_grader"],
                                                                      -c["grader_total"]))]
        taus.append(kendall_tau(grader_order, r["ranking"]))
        for c in cands:
            f = fam[c["family"]]
            f["shown"] += 1
            f["top"] += c is top
            f["scores"].append(c["teacher_score"])
            f["ranks"].append(c["teacher_rank"])
            f["picked_by_grader"] += c["is_pick"]
            if c["is_fallback"]:
                fb_seen += 1
                fb_top += c is top
    gaps = [b - p for b, p in zip(best_scores, pick_scores)]
    return {"cells": n, "agree": sum(agree), "agree_share": mean(agree), "chance_share": mean(chance),
            "pick_mean_score": mean(pick_scores), "best_mean_score": mean(best_scores), "headroom": mean(gaps),
            "cells_with_headroom": sum(g > 0 for g in gaps), "pick_mean_rank": mean(pick_ranks),
            "mean_kendall_tau": mean([t for t in taus if not math.isnan(t)]),
            "inconsistent_rankings": sum(not r["consistent"] for r in rows),
            "fallback_shown": fb_seen, "fallback_top": fb_top,
            "families": {k: {"shown": v["shown"], "teacher_top": v["top"], "top_share": v["top"] / v["shown"],
                             "grader_picked": v["picked_by_grader"], "mean_score": mean(v["scores"]),
                             "mean_rank": mean(v["ranks"])}
                         for k, v in sorted(fam.items(), key=lambda kv: -kv[1]["top"] / kv[1]["shown"])}}


def write_report(name: str, root: Path = PICKER_ROOT) -> Path:
    rows = list(load_rankings(name, root).values())
    prepared = _read_jsonl(data_dir(name, root) / "candidates.jsonl")
    skipped = sum("skip" in r for r in prepared)
    r = compute_report(rows) if rows else None
    meta = json.loads((study_dir(name, root) / "study.json").read_text())
    f2 = lambda x: f"{x:.2f}"
    lines = [f"# Picker study: {name}", "",
             f"Split `{meta['split']}`, engine env `{meta['env']}`, K = {meta['k']} (at most "
             f"{meta['max_per_family']} per strategy family, fallback included when it passes the gates).", "",
             f"- Cells prepared: {len(prepared)} ({skipped} skipped: fewer than 2 gate-passing candidates)",
             f"- Cells ranked by the teacher: {len(rows)}"]
    if r:
        lines += ["",
                  f"**Grader's pick = teacher's top choice: {r['agree']} of {r['cells']} "
                  f"({100 * r['agree_share']:.0f}%; chance {100 * r['chance_share']:.0f}%).**", "",
                  f"- Teacher's mean score (1-5) of the grader's pick: {f2(r['pick_mean_score'])}",
                  f"- Teacher's mean score of its own best candidate: {f2(r['best_mean_score'])}",
                  f"- **Headroom** (best - pick, mean per cell): {f2(r['headroom'])}; cells where a better "
                  f"candidate scored higher: {r['cells_with_headroom']}",
                  f"- Grader's pick: mean teacher rank {f2(r['pick_mean_rank'])} of K",
                  f"- Mean Kendall tau, grader order vs teacher ranking: {f2(r['mean_kendall_tau'])} "
                  "(1 = same order, 0 = unrelated)",
                  f"- Fallback shown in {r['fallback_shown']} cells, teacher's top in {r['fallback_top']}",
                  f"- Rankings whose order disagrees with their own scores: {r['inconsistent_rankings']}", "",
                  "Strategy families (by how often the teacher ranks them first when shown):", "",
                  "| family | shown | teacher's top | top share | grader's pick | mean score | mean rank |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {k} | {v['shown']} | {v['teacher_top']} | {f2(v['top_share'])} | {v['grader_picked']} | "
                  f"{f2(v['mean_score'])} | {f2(v['mean_rank'])} |" for k, v in r["families"].items()]
    path = study_dir(name, root) / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--name", required=True)
    p.add_argument("--split", default="holdout", choices=SPLITS)
    p.add_argument("--cells", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--env", action="append", default=None, help="KEY=VAL (replaces the v3L env when given)")
    p.add_argument("--workers", type=int, default=WORKERS)
    r = sub.add_parser("rank")
    r.add_argument("--name", required=True)
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--shard", default=None)
    r.add_argument("--merge", action="store_true")
    r.add_argument("--limit-calls", type=int, default=None)
    q = sub.add_parser("report")
    q.add_argument("--name", required=True)
    args = ap.parse_args(argv)

    if args.step == "prepare":
        env = ab_engine.parse_env(args.env) if args.env else None
        rows = prepare(args.name, args.split, args.cells, args.seed, args.k, env, args.workers)
        print(f"prepared {len(rows)} new cells for {args.name} ({sum('skip' in x for x in rows)} skipped)")
    elif args.step == "rank":
        if args.merge:
            print(f"merged {judge_mod.merge_shards(data_dir(args.name) / 'rankings.jsonl')} rows")
            return 0
        from dev.distill.teacher import ClaudeCodeTeacher
        rows = rank(args.name, ClaudeCodeTeacher(args.model), shard=ab_engine.parse_shard(args.shard),
                    limit_calls=args.limit_calls, log=lambda m: print(m, file=sys.stderr))
        print(f"ranked {len(rows)} new cells")
    else:
        path = write_report(args.name)
        print(path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
