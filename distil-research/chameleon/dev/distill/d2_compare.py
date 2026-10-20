"""D2: do the learned grader weights pick better final layouts than the
hand-set ones? Pairwise accuracy on judged pairs is not the same question:
the engine only ships one winner per cell.

    python -m dev.distill.d2_compare picks                 # offline: replay pick_best per cell under each weighting
    python -m dev.distill.d2_compare judge --shard K/N     # teacher judges default pick vs each variant's pick
    python -m dev.distill.d2_compare merge
    python -m dev.distill.d2_compare report                # win rates, by split and brand

Replays pick_best from the pool's stored metrics (top 6 by geometry, then
total), so no rendering is needed for the picks themselves.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from dev.distill.judge import judge_pairs, merge_shards, shard_file
from dev.distill.paths import DATA_ROOT, POOL_DIR, RENDERS_SUBDIR
from engine.pipeline.grading.score import TOP_K
from engine.pipeline.grading.weights import DEFAULT_WEIGHTS, GraderWeights

OUT_DIR = DATA_ROOT / "judgments" / "d2"
PICKS = OUT_DIR / "picks.json"
JUDGMENTS = OUT_DIR / "judgments.jsonl"
TRAINING_JSON = DATA_ROOT / "judgments" / "training-report.json"


def _learned() -> GraderWeights:
    w = json.loads(TRAINING_JSON.read_text())["weights"]["learned"]
    geo = {k: w[k] for k in DEFAULT_WEIGHTS.geometry}
    pix = {k: w[k] for k in DEFAULT_WEIGHTS.pixels}
    return GraderWeights(geo, pix, w["geometry_share"])


def pick(rows: list[dict], weights: GraderWeights, fallback_last: bool = False) -> dict:
    """The candidate pick_best would choose from these pool rows (one cell).
    fallback_last: the safe fallback only wins when it is the only candidate."""
    pool = [r for r in rows if not (fallback_last and r["is_fallback"])] or rows
    geo = lambda r: sum(weights.geometry[k] * r["metrics"][k] for k in weights.geometry)
    pix = lambda r: sum(weights.pixels[k] * r["metrics"][k] for k in weights.pixels)
    top = sorted(pool, key=lambda r: (-geo(r), r["candidate_id"]))[:TOP_K]
    total = lambda r: weights.geometry_share * geo(r) + (1 - weights.geometry_share) * pix(r)
    return min(top, key=lambda r: (-total(r), r["candidate_id"]))


VARIANTS = {"learned": dict(fallback_last=False), "learned_no_fallback": dict(fallback_last=True)}


def picks() -> dict:
    rows = [json.loads(x) for x in (POOL_DIR / "pool.jsonl").read_text().splitlines() if x.strip()]
    cells: dict[str, list] = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r)
    learned = _learned()
    out = {}
    for cell, rs in sorted(cells.items()):
        d = pick(rs, DEFAULT_WEIGHTS)
        entry = {"split": rs[0]["split"], "brand": rs[0].get("brand", "unknown"), "default": d["candidate_id"],
                 "default_fallback": d["is_fallback"]}
        for name, kw in VARIANTS.items():
            p = pick(rs, learned, **kw)
            entry[name], entry[f"{name}_fallback"] = p["candidate_id"], p["is_fallback"]
        out[cell] = entry
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PICKS.write_text(json.dumps(out, indent=1))
    return out


def summary(p: dict) -> str:
    lines = [f"cells: {len(p)}"]
    for name in ["default", *VARIANTS]:
        fb = sum(e[f"{name}_fallback"] for e in p.values())
        changed = sum(e[name] != e["default"] for e in p.values()) if name != "default" else 0
        lines.append(f"{name:20} fallback wins {fb:4}   picks changed vs default {changed:4}")
    return "\n".join(lines)


def _pairs(p: dict) -> list[tuple]:
    rows = [json.loads(x) for x in (POOL_DIR / "pool.jsonl").read_text().splitlines() if x.strip()]
    by = {(r["cell"], r["candidate_id"]): r for r in rows}
    pairs, seen = [], set()
    for cell, e in sorted(p.items()):
        for name in VARIANTS:
            key = (cell, frozenset((e["default"], e[name])))
            if e[name] != e["default"] and key not in seen:
                seen.add(key)
                pairs.append((by[cell, e["default"]], by[cell, e[name]]))
    return pairs


def report(p: dict) -> str:
    js = [json.loads(x) for x in JUDGMENTS.read_text().splitlines() if x.strip()] if JUDGMENTS.exists() else []
    lines = [summary(p), "", "teacher verdicts, variant pick vs default pick (cells where they differ; confidence >= 2):"]
    for name in VARIANTS:
        tally = Counter()
        for j in js:
            if j["confidence"] < 2 or j["winner"] == "tie":
                continue
            e = p[j["cell"]]
            if e[name] == e["default"] or {j["a"], j["b"]} != {e["default"], e[name]}:
                continue
            won = j["a"] if j["winner"] == "a" else j["b"]
            tally[(e["split"], e["brand"], "variant" if won == e[name] else "default")] += 1
        for split in ("holdout", "train"):
            for brand in ("solstice", "offbrand"):
                v, d = tally[(split, brand, "variant")], tally[(split, brand, "default")]
                if v + d:
                    lines.append(f"  {name:20} {split:8} {brand:9} variant wins {v:3} / {v + d:3} = {v / (v + d):.0%}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["picks", "judge", "merge", "report"])
    ap.add_argument("--shard", default=None)
    ap.add_argument("--model", default="claude-opus-5-5")
    args = ap.parse_args()
    if args.step == "picks":
        print(summary(picks()))
    elif args.step == "judge":
        from dev.distill.teacher import ClaudeCodeTeacher
        shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
        out = shard_file(JUDGMENTS, *shard) if shard else JUDGMENTS
        added = judge_pairs(_pairs(json.loads(PICKS.read_text())), ClaudeCodeTeacher(args.model),
                            lambda t: POOL_DIR / RENDERS_SUBDIR / t / "source.png", out, pairs_per_call=4,
                            swap_share=0.2, seed=0, pool_dir=POOL_DIR, shard=shard, done_files=[JUDGMENTS])
        print(f"judged {len(added)}")
    elif args.step == "merge":
        print(f"merged {merge_shards(JUDGMENTS)}")
    else:
        text = report(json.loads(PICKS.read_text()))
        (OUT_DIR / "REPORT.md").write_text("# D2: learned vs hand-set picks\n\n```\n" + text + "\n```\n")
        print(text)


if __name__ == "__main__":
    main()
