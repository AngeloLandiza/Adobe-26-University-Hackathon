"""A/B harness: is engine variant B better than variant A, by the teacher's eye?

A "variant" is the whole engine run under some environment variables (e.g.
CHAMELEON_FLEX=1). Both variants resize the same held-out cells; wherever the
winning designs differ, the teacher judges A vs B.

    python -m dev.distill.ab_engine run --name base --env CHAMELEON_FLEX=0 [--split holdout] [--cells N] [--shard K/N] [--workers 4]
    python -m dev.distill.ab_engine run --name flex --env CHAMELEON_FLEX=1
    python -m dev.distill.ab_engine run --name flex --merge          # fold winners shards
    python -m dev.distill.ab_engine judge --a base --b flex [--shard K/N]
    python -m dev.distill.ab_engine judge --a base --b flex --merge
    python -m dev.distill.ab_engine report --a base --b flex

Every step is resumable (a rerun skips cells already done) and shardable.
"""

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dev.distill import judge as judge_mod
from dev.distill.paths import CORPUS_DIR, DATA_ROOT, POOL_DIR, RENDERS_SUBDIR

AB_ROOT = DATA_ROOT / "ab"
TARGETS_PATH = Path(__file__).resolve().parents[1] / "targets.json"
RENDER_SIZE = 768
MIN_CONFIDENCE = 2


# ---------------------------------------------------------------- helpers

def design_hash(design: dict) -> str:
    return hashlib.sha1(json.dumps(design, sort_keys=True).encode()).hexdigest()


def parse_env(items: list[str]) -> dict[str, str]:
    env = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--env wants KEY=VAL, got {item!r}")
        k, v = item.split("=", 1)
        env[k] = v
    return env


def parse_shard(text: str | None):
    return tuple(int(x) for x in text.split("/")) if text else None


def _read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def pool_cells(pool_dir: Path, split: str = "holdout", n: int | None = None, seed: int = 0) -> list[dict]:
    """One row per cell of the pool (cell, task_id, split, brand, target)."""
    rows = _read_jsonl(Path(pool_dir) / "pool.jsonl")
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    if n is not None:
        chosen = judge_mod.pick_cells(rows, n, seed)
        rows = [r for r in rows if r["cell"] in chosen]
    cells: dict[str, dict] = {}
    for r in rows:
        cells.setdefault(r["cell"], {k: r.get(k, "solstice" if k == "brand" else None)
                                     for k in ("cell", "task_id", "split", "brand", "target")})
    return [cells[c] for c in sorted(cells)]


def winners_files(name: str, root: Path = AB_ROOT) -> list[Path]:
    main = Path(root) / name / "winners.jsonl"
    return [main, *sorted(main.parent.glob("winners.shard*of*.jsonl"))]


def load_winners(name: str, root: Path = AB_ROOT) -> dict[str, dict]:
    out = {}
    for f in winners_files(name, root):
        for r in _read_jsonl(f):
            out[r["cell"]] = r
    return out


# ---------------------------------------------------------------- run

_ASSETS = None  # per-process merged assets dir


def _assets():
    global _ASSETS
    if _ASSETS is None:
        import atexit
        from dev.distill.assets import merged_assets_dir
        cm = merged_assets_dir()
        _ASSETS = cm.__enter__()
        atexit.register(cm.__exit__, None, None, None)
    return _ASSETS


def _resize(source: dict, canvas: dict):
    from engine.resize import resize_with_report
    return resize_with_report(source, canvas)


def _worker_init(env: dict):
    os.environ.update(env)
    os.environ["CHAMELEON_LLM"] = "heuristic"


def run_cell(job: dict) -> dict:
    """Resize one cell, render the winner, return its winners.jsonl row."""
    from engine.pipeline.render import render_to_image
    design, cell, target, name = job["design"], job["cell"], job["target_canvas"], job["name"]
    source = design["source"]
    t0 = time.perf_counter()
    best_design, _plan, best, _reports = _resize(source, {"width": target["width"], "height": target["height"]})
    seconds = time.perf_counter() - t0
    png = Path(RENDERS_SUBDIR) / cell["task_id"] / f"{cell['target']}.png"
    path = Path(job["out_dir"]) / png
    path.parent.mkdir(parents=True, exist_ok=True)
    render_to_image(best_design, assets_dir=_assets(), max_size=RENDER_SIZE).save(path)
    return {"cell": cell["cell"], "task_id": cell["task_id"], "split": cell["split"], "brand": cell["brand"],
            "target": cell["target"], "variant": name, "candidate_id": best.candidate_id,
            "is_fallback": bool(best.is_fallback), "design_hash": design_hash(best_design),
            "png": png.as_posix(), "seconds": round(seconds, 3)}


def _safe_run_cell(job: dict):
    try:
        return run_cell(job)
    except Exception as exc:  # one bad cell must not sink the shard
        return {"error": f"{type(exc).__name__}: {exc}", "cell": job["cell"]["cell"]}


def run_variant(name: str, env: dict, cells: list[dict], designs: dict, targets: dict, root: Path = AB_ROOT,
                shard=None, workers: int = 4, log=print) -> list[dict]:
    out_dir = Path(root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    main = out_dir / "winners.jsonl"
    out = judge_mod.shard_file(main, *shard) if shard else main
    done = {c for c, r in load_winners(name, root).items() if (out_dir / r["png"]).exists()}
    jobs = [{"cell": c, "design": designs[c["task_id"]], "target_canvas": targets[c["target"]],
             "name": name, "out_dir": str(out_dir)}
            for c in cells if c["cell"] not in done and judge_mod._in_shard(c["cell"], shard)
            and c["task_id"] in designs and c["target"] in targets]
    added = []

    def record(row):
        if "error" in row:
            log(f"[ab] {row['cell']}: skipped ({row['error']})")
            return
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        added.append(row)

    if workers <= 1:
        saved = {k: os.environ.get(k) for k in [*env, "CHAMELEON_LLM"]}
        os.environ.update(env)
        os.environ["CHAMELEON_LLM"] = "heuristic"
        try:
            for job in jobs:
                record(_safe_run_cell(job))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    else:
        with ProcessPoolExecutor(max_workers=min(workers, 4), initializer=_worker_init, initargs=(env,)) as ex:
            for row in ex.map(_safe_run_cell, jobs):
                record(row)
    (out_dir / "variant.json").write_text(json.dumps({"name": name, "env": env}, indent=2))
    return added


def merge_winners(name: str, root: Path = AB_ROOT) -> int:
    return judge_mod.merge_shards(Path(root) / name / "winners.jsonl")


# ---------------------------------------------------------------- judge

def changed_pairs(a: str, b: str, root: Path = AB_ROOT) -> tuple[list[tuple[dict, dict]], int, int]:
    """(pairs for cells whose winners differ, cells compared, identical count).
    Pair rows carry pngs relative to `root`."""
    wa, wb = load_winners(a, root), load_winners(b, root)
    common = sorted(set(wa) & set(wb))
    pairs, same = [], 0
    for c in common:
        ra, rb = wa[c], wb[c]
        if ra["design_hash"] == rb["design_hash"]:
            same += 1
            continue
        def side(r, v):
            return {"cell": r["cell"], "task_id": r["task_id"], "split": r["split"], "brand": r.get("brand"),
                    "target": r["target"], "candidate_id": v, "png": f"{v}/{r['png']}", "is_fallback": r["is_fallback"]}
        pairs.append((side(ra, a), side(rb, b)))
    return pairs, len(common), same


def pair_dir(a: str, b: str, root: Path = AB_ROOT) -> Path:
    return Path(root) / f"{a}__vs__{b}"


def judge_variants(a: str, b: str, teacher, root: Path = AB_ROOT, pool_dir: Path = POOL_DIR, shard=None,
                   limit_calls=None, seed: int = 0) -> list[dict]:
    pairs, _, _ = changed_pairs(a, b, root)
    main = pair_dir(a, b, root) / "judgments.jsonl"
    out = judge_mod.shard_file(main, *shard) if shard else main
    return judge_mod.judge_pairs(
        pairs, teacher, lambda t: Path(pool_dir) / RENDERS_SUBDIR / t / "source.png", out,
        pairs_per_call=4, swap_share=0.2, seed=seed, pool_dir=Path(root), limit_calls=limit_calls,
        shard=shard, done_files=[main] if shard else None)


# ---------------------------------------------------------------- report

def cell_verdicts(judgments: list[dict], a: str, b: str) -> dict[str, str]:
    """Per cell: 'a', 'b' or 'unsure'. Rows below MIN_CONFIDENCE and ties count
    for neither; a cell goes to the variant with more confident votes."""
    votes = defaultdict(lambda: {a: 0, b: 0})
    seen = set()
    for j in judgments:
        seen.add(j["cell"])
        if j["winner"] == "tie" or j["confidence"] < MIN_CONFIDENCE:
            continue
        votes[j["cell"]][j["a"] if j["winner"] == "a" else j["b"]] += 1
    out = {}
    for c in seen:
        v = votes[c]
        out[c] = "a" if v[a] > v[b] else "b" if v[b] > v[a] else "unsure"
    return out


def compute_report(a: str, b: str, root: Path = AB_ROOT) -> dict:
    wa, wb = load_winners(a, root), load_winners(b, root)
    pairs, compared, same = changed_pairs(a, b, root)
    pdir = pair_dir(a, b, root)
    js = [j for f in [pdir / "judgments.jsonl", *sorted(pdir.glob("judgments.shard*of*.jsonl"))] for j in _read_jsonl(f)]
    changed_cells = {p[0]["cell"] for p in pairs}
    js = [j for j in js if j["cell"] in changed_cells and j.get("rubric_version") == judge_mod.RUBRIC_VERSION] or \
         [j for j in js if j["cell"] in changed_cells]
    verdicts = cell_verdicts(js, a, b)
    tally = {"a": 0, "b": 0, "unsure": 0}
    by_brand = defaultdict(lambda: {"a": 0, "b": 0, "unsure": 0})
    for c, v in verdicts.items():
        tally[v] += 1
        by_brand[wb[c].get("brand") or "unknown"][v] += 1
    scores = {a: [], b: []}
    for j in js:
        scores[j["a"]].append(j["score_a"])
        scores[j["b"]].append(j["score_b"])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    common = sorted(set(wa) & set(wb))
    return {"a": a, "b": b, "compared": compared, "identical": same, "changed": len(pairs),
            "judged": len(verdicts), "unjudged": len(changed_cells - set(verdicts)),
            "wins": {"a": tally["a"], "b": tally["b"], "unsure": tally["unsure"]},
            "by_brand": {k: dict(v) for k, v in sorted(by_brand.items())},
            "mean_score": {a: mean(scores[a]), b: mean(scores[b])},
            "mean_seconds": {a: mean([wa[c]["seconds"] for c in common]), b: mean([wb[c]["seconds"] for c in common])}}


def verdict_line(rep: dict) -> str:
    a, b, w = rep["a"], rep["b"], rep["wins"]
    decided = w["a"] + w["b"]
    if not rep["judged"]:
        return f"NO VERDICT YET: {rep['changed']} changed cells, none judged."
    if decided == 0 or w["a"] == w["b"]:
        return f"NO CLEAR DIFFERENCE: {a} {w['a']} - {b} {w['b']} ({w['unsure']} unsure) on {rep['judged']} judged changed cells."
    win, lose, wn, ln = (b, a, w["b"], w["a"]) if w["b"] > w["a"] else (a, b, w["a"], w["b"])
    # sign test on decided cells: p < 0.05 two-sided
    from math import comb
    n, k = wn + ln, wn
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    strength = "significantly better" if p < 0.05 else "better, but not significant"
    return f"{win} is {strength} than {lose}: {wn} to {ln} ({w['unsure']} unsure), sign test p={p:.3f}, on {rep['judged']} judged changed cells."


def write_report(a: str, b: str, root: Path = AB_ROOT) -> Path:
    r = compute_report(a, b, root)
    pct = lambda x: f"{100 * x / r['compared']:.0f}%" if r["compared"] else "n/a"
    lines = [f"# A/B: {a} (A) vs {b} (B)", "", f"**Verdict: {verdict_line(r)}**", "",
             f"- Cells compared: {r['compared']}",
             f"- Identical winners: {r['identical']} ({pct(r['identical'])})",
             f"- Changed winners: {r['changed']} ({pct(r['changed'])}); judged {r['judged']}, not yet judged {r['unjudged']}",
             "", f"Teacher verdicts (confidence >= {MIN_CONFIDENCE}; per cell, majority of original and swapped asks):", "",
             f"| | {a} wins (A) | {b} wins (B) | tie / unsure |", "|---|---|---|---|",
             f"| all | {r['wins']['a']} | {r['wins']['b']} | {r['wins']['unsure']} |"]
    lines += [f"| {k} | {v['a']} | {v['b']} | {v['unsure']} |" for k, v in r["by_brand"].items()]
    lines += ["", "Mean ship-readiness score (1-5) on changed cells:", "",
              f"- A ({a}): {r['mean_score'][a]:.2f}", f"- B ({b}): {r['mean_score'][b]:.2f}", "",
              "Mean engine seconds per cell (all compared cells):", "",
              f"- A ({a}): {r['mean_seconds'][a]:.2f}", f"- B ({b}): {r['mean_seconds'][b]:.2f}", ""]
    path = pair_dir(a, b, root) / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--split", default="holdout", choices=["train", "holdout", "all"])
    r.add_argument("--cells", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--shard", default=None)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--merge", action="store_true")
    j = sub.add_parser("judge")
    j.add_argument("--a", required=True)
    j.add_argument("--b", required=True)
    j.add_argument("--model", default="claude-opus-5-5")
    j.add_argument("--shard", default=None)
    j.add_argument("--merge", action="store_true")
    j.add_argument("--limit-calls", type=int, default=None)
    j.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    args = ap.parse_args(argv)

    if args.step == "run":
        if args.merge:
            print(f"merged {merge_winners(args.name)} rows")
            return 0
        from dev.distill.corpus import load_corpus
        cells = pool_cells(POOL_DIR, args.split, args.cells, args.seed)
        designs = {d["task_id"]: d for d in load_corpus(None if args.split == "all" else args.split)}
        targets = {t["name"]: t for t in json.loads(TARGETS_PATH.read_text())}
        rows = run_variant(args.name, parse_env(args.env), cells, designs, targets, shard=parse_shard(args.shard),
                           workers=args.workers)
        print(f"ran {len(rows)} new cells of {len(cells)} for {args.name}")
    elif args.step == "judge":
        if args.merge:
            print(f"merged {judge_mod.merge_shards(pair_dir(args.a, args.b) / 'judgments.jsonl')} rows")
            return 0
        from dev.distill.teacher import ClaudeCodeTeacher
        rows = judge_variants(args.a, args.b, ClaudeCodeTeacher(args.model), shard=parse_shard(args.shard),
                              limit_calls=args.limit_calls, seed=args.seed)
        print(f"judged {len(rows)} new pairs")
    else:
        print(write_report(args.a, args.b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
