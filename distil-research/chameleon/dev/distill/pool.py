"""The candidate pool: every passing layout the engine builds for an ad at a
target size, rendered and measured, plus the pair sampler that picks which
two layouts the teacher is asked to compare.

    python -m dev.distill.pool [--per-ad 6] [--seed 0] [--split train|holdout|all] [--limit-ads N]
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from dev.distill.paths import DATA_ROOT, POOL_DIR
from engine.pipeline.grading.weights import DEFAULT_WEIGHTS

TARGETS_PATH = Path(__file__).resolve().parents[1] / "targets.json"
RENDER_SIZE = 768
TOP_N = 6            # "close contests" are drawn from the top N by today's total
NEAR_SOURCE = 0.02   # a target this close to the source's aspect only gets the reflow: nothing to compare


def _aspect(width: float, height: float) -> float:
    return width / height


def _near_source(target: dict, source_aspect: float) -> bool:
    return abs(_aspect(target["width"], target["height"]) / source_aspect - 1) <= NEAR_SOURCE


def _design_key(design: dict) -> str:
    return hashlib.sha1(json.dumps(design, sort_keys=True).encode()).hexdigest()


def _source_aspect(design: dict) -> float:
    canvas = design["source"]["canvas"]
    return _aspect(canvas["width"], canvas["height"])


def build_pool(designs: list[dict], targets: list[dict] | dict[str, list[dict]], out_dir: Path = POOL_DIR,
               log=None) -> list[dict]:
    """Render and measure every distinct passing candidate for each ad x target
    cell. `targets` is one list for every ad, or {task_id: its targets}. An ad
    that fails to build is reported through `log` and skipped. Rows are also
    written to out_dir/pool.jsonl."""
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.grading.metrics import geometry_metrics
    from engine.pipeline.grading.pixels import pixel_metrics
    from engine.pipeline.layout.fallback import safe_fallback
    from engine.pipeline.layout.generate import generate_candidates
    from engine.pipeline.priority import heuristic_plan
    from engine.pipeline.render import render_to_image
    from engine.pipeline.textmeasure import TextMeasurer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    measurer = TextMeasurer()
    rows: list[dict] = []
    for design in designs:
        task_id, source = design["task_id"], design["source"]
        ad_targets = targets[task_id] if isinstance(targets, dict) else targets
        ad_rows: list[dict] = []
        try:
            plan = heuristic_plan(source)
            ad_dir = out_dir / "renders" / task_id
            ad_dir.mkdir(parents=True, exist_ok=True)
            render_to_image(source, max_size=RENDER_SIZE).save(ad_dir / "source.png")
            for target in ad_targets:
                if _near_source(target, _source_aspect(design)):
                    continue
                canvas = {"width": target["width"], "height": target["height"]}
                candidates = [safe_fallback(source, canvas, plan, measurer)] + generate_candidates(source, canvas, plan, measurer)
                seen: set[str] = set()
                for cand in candidates:
                    if check_gates(source, cand.design, canvas, plan, measurer):
                        continue
                    key = _design_key(cand.design)
                    if key in seen:
                        continue
                    try:
                        metrics = geometry_metrics(source, cand, plan) | pixel_metrics(cand.design, plan)
                        image = render_to_image(cand.design, max_size=RENDER_SIZE)
                    except Exception:  # an unmeasurable or unrenderable candidate can't be judged
                        continue
                    seen.add(key)
                    png = Path("renders") / task_id / target["name"] / f"{len(seen) - 1}.png"
                    (out_dir / png).parent.mkdir(parents=True, exist_ok=True)
                    image.save(out_dir / png)
                    ad_rows.append({"cell": f"{task_id}@{target['name']}", "task_id": task_id, "split": design["split"],
                                    "target": target["name"], "candidate_id": cand.candidate_id, "png": png.as_posix(),
                                    "is_fallback": cand.is_fallback, "metrics": metrics})
        except Exception as exc:
            if log:
                log(f"{task_id}: skipped ({exc!r})")
            continue
        rows += ad_rows
        if log:
            log(f"{task_id}: {len(ad_rows)} candidates over {len({r['cell'] for r in ad_rows})} targets")
    (out_dir / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def _total(metrics: dict) -> float:
    """Today's total: what the engine's own grader would give this layout."""
    w = DEFAULT_WEIGHTS
    geo = sum(weight * metrics[k] for k, weight in w.geometry.items())
    pix = sum(weight * metrics[k] for k, weight in w.pixels.items())
    return w.geometry_share * geo + (1 - w.geometry_share) * pix


def sample_pairs(rows: list[dict], per_cell: int = 6, seed: int = 0) -> list[tuple[dict, dict]]:
    """Up to `per_cell` distinct pairs per cell. About half are close contests
    among the top TOP_N by today's total; the rest pair a top candidate with a
    candidate outside the top (when the cell has one)."""
    cells: dict[str, list[dict]] = {}
    for row in rows:
        cells.setdefault(row["cell"], []).append(row)
    pairs: list[tuple[dict, dict]] = []
    for cell in sorted(cells):
        ranked = sorted(cells[cell], key=lambda r: (-_total(r["metrics"]), r["candidate_id"]))
        top = ranked[:TOP_N]
        rng = random.Random(f"{seed}:{cell}")
        close = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
        mixed = [(a, b) for a in top for b in ranked[TOP_N:]]  # a top candidate vs one outside the top
        rng.shuffle(close)
        rng.shuffle(mixed)
        chosen: list[tuple[dict, dict]] = []
        used: set[frozenset] = set()

        def take(source_pairs: list[tuple[dict, dict]], limit: int) -> None:
            for a, b in source_pairs:
                key = frozenset((a["candidate_id"], b["candidate_id"]))
                if len(chosen) >= limit:
                    return
                if key not in used:
                    used.add(key)
                    chosen.append((a, b) if rng.random() < 0.5 else (b, a))

        take(close, (per_cell + 1) // 2)
        take(mixed, per_cell)
        take(close, per_cell)  # top-up when there are too few top-vs-rest pairs
        pairs += chosen
    return pairs


def choose_targets(task_ids: list[str], targets: list[dict], per_ad: int = 6, seed: int = 0,
                   source_aspects: dict[str, float] | None = None) -> dict[str, list[dict]]:
    """`per_ad` distinct targets for each ad, picked by seed, with every aspect
    class in `targets` given to at least one ad and the targets spread evenly
    across the corpus. `source_aspects` (width/height per task_id) drops targets
    the ad is already shaped like."""
    from engine.pipeline.layout.aspect import aspect_class

    rng = random.Random(seed)
    source_aspects = source_aspects or {}

    def eligible(task_id: str) -> list[dict]:
        aspect = source_aspects.get(task_id)
        return [t for t in targets if aspect is None or not _near_source(t, aspect)]

    chosen: dict[str, list[dict]] = {t: [] for t in task_ids}
    used: dict[str, int] = {t["name"]: 0 for t in targets}

    def give(task_id: str, target: dict) -> None:
        chosen[task_id].append(target)
        used[target["name"]] += 1

    order = list(task_ids)
    rng.shuffle(order)
    classes = sorted({aspect_class(t["width"], t["height"]) for t in targets})
    for k, cls in enumerate(classes):  # coverage first: hand each class to the next ad that can take it
        for step in range(len(order)):
            task_id = order[(k + step) % len(order)]
            options = [t for t in eligible(task_id) if aspect_class(t["width"], t["height"]) == cls
                       and t not in chosen[task_id]]
            if options and len(chosen[task_id]) < per_ad:
                give(task_id, rng.choice(options))
                break
    for task_id in order:
        rest = [t for t in eligible(task_id) if t not in chosen[task_id]]
        rng.shuffle(rest)
        rest.sort(key=lambda t: used[t["name"]])  # least-used first, ties by the seeded shuffle
        for t in rest[: per_ad - len(chosen[task_id])]:
            give(task_id, t)
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the candidate pool for the corpus.")
    parser.add_argument("--per-ad", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "holdout", "all"], default="all")
    parser.add_argument("--limit-ads", type=int, default=None)
    args = parser.parse_args(argv)

    os.environ["CHAMELEON_LLM"] = "heuristic"  # the pool never calls a model
    from dev.distill.corpus import load_corpus

    designs = load_corpus(None if args.split == "all" else args.split)
    if args.limit_ads is not None:
        designs = designs[: args.limit_ads]
    targets = json.loads(TARGETS_PATH.read_text())
    chosen = choose_targets([d["task_id"] for d in designs], targets, args.per_ad, args.seed,
                            {d["task_id"]: _source_aspect(d) for d in designs})
    rows = build_pool(designs, chosen, POOL_DIR, log=print)
    print(f"{len(rows)} candidates over {len({r['cell'] for r in rows})} cells "
          f"-> {POOL_DIR.relative_to(DATA_ROOT.parent)}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
