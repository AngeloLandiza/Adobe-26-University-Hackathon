"""C: per-ad floor-plan retrieval, evaluated offline.

    python -m dev.distill.retrieval evaluate --stage build   [--shard K/N | --merge]
    python -m dev.distill.retrieval evaluate --stage judge   [--shard K/N | --merge] [--limit-calls N]
    python -m dev.distill.retrieval evaluate --stage summary

Words: a *cell* is one ad at one target size; a *floor plan* cuts the canvas into
named slots (see dev/distill/floorplans.py); a plan *wins* when the teacher
clearly prefers its build to the cell's current winner (floorplans' strict rule).

Idea: B2's teacher drew floor plans FOR specific training ads, and many beat the
engine's best. Clustered into generic templates they transfer less well on new
ads. Retrieval keeps the per-ad specificity: describe a new ad with a small
vector (`ad_vector`), find the k most similar training ads that have a winning
plan for the target's aspect class, and reuse those winning plans.

* `ad_vector(source, plan)`: 17 numbers, each scaled into [-1, 1] (see below).
* `library()`: every winning B2 plan with its source ad's vector and aspect class.
* `recommend(source, target, k)`: the winning plans of the k nearest training ads.
* `evaluate`: for HOLDOUT cells of tall / very_tall / square targets, build the
  recommended plans (one temp extra-templates file per cell), keep the best by the
  default grader, and have the teacher judge it against the cell's current winner.
  Output: retrieval-builds.jsonl, retrieval-judgments.jsonl, retrieval-summary.json
  and RETRIEVAL.md (compared with the clustered templates' eval-* files).

Offline only: nothing here changes the engine.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from dev.distill import floorplans as fp
from dev.distill.judge import RUBRIC_VERSION, merge_shards, shard_file
from dev.distill.paths import POOL_DIR, RENDERS_SUBDIR
from engine.pipeline.geometry import element_box, union_box
from engine.pipeline.layout.aspect import aspect_class
from engine.pipeline.layout.extra_templates import _parse_plan

FP_DIR = fp.FP_DIR
RET_BUILDS_FILE = FP_DIR / "retrieval-builds.jsonl"
RET_JUDGMENTS_FILE = FP_DIR / "retrieval-judgments.jsonl"
RET_SUMMARY_FILE = FP_DIR / "retrieval-summary.json"
RET_REPORT_FILE = FP_DIR / "RETRIEVAL.md"

EVAL_CLASSES = ("tall", "very_tall", "square")  # the classes B2 covered
DEFAULT_K = 3

ROLES = ("headline", "subheadline", "cta", "logo", "hero_image", "secondary_image", "badge", "legal", "other_text")
COUNT_CAP = 3        # a role count of COUNT_CAP or more reads as 1.0
HEADLINE_CHARS_CAP = 80
HEADLINE_WORDS_CAP = 16
UNITS_CAP = 12
ASPECT_LOG2_RANGE = 2.0   # aspect ratios from 1:4 to 4:1 span [-1, 1]; beyond is clipped

# One entry per vector dimension, in order.
DIMENSIONS = (
    *(f"count_{r}" for r in ROLES),   # 0-8: units with that role, min(count, 3) / 3
    "headline_chars",                  # 9: characters of the headline text, min(n, 80) / 80
    "headline_words",                  # 10: words of the headline text, min(n, 16) / 16
    "hero_aspect",                     # 11: log2(width/height of the hero image unit) / 2, clipped to [-1, 1]; 0 if none
    "logo_aspect",                     # 12: same for the logo unit
    "source_canvas_aspect",            # 13: same for the source canvas
    "n_units",                         # 14: ranked content units, min(n, 12) / 12
    "has_group",                       # 15: 1 if any unit is a multi-element group
    "has_button",                      # 16: 1 if any unit is a button (a shape with its label text)
)


def log(msg: str) -> None:
    print(f"[retrieval] {msg}", file=sys.stderr)


# ---------------------------------------------------------------- the vector

def _scaled_aspect(w, h) -> float:
    if not w or not h or w <= 0 or h <= 0:
        return 0.0
    return float(max(-1.0, min(1.0, math.log2(w / h) / ASPECT_LOG2_RANGE)))


def _unit_elements(unit, by_id: dict) -> list[dict]:
    return [by_id[i] for i in unit.element_ids if i in by_id]


def _unit_aspect(unit, by_id: dict) -> float:
    els = _unit_elements(unit, by_id)
    if not els:
        return 0.0
    _, _, w, h = union_box([element_box(e) for e in els])
    return _scaled_aspect(w, h)


def ad_vector(source: dict, plan) -> np.ndarray:
    """A small, deterministic description of an ad's content (see DIMENSIONS).
    Every value lies in [-1, 1], so no dimension swamps the Euclidean distance.
    `plan` is the ad's PriorityPlan (roles and ranks of its units)."""
    by_id = {e["id"]: e for e in source.get("elements", [])}
    ranked = plan.ranked_units()
    v = np.zeros(len(DIMENSIONS))
    for i, role in enumerate(ROLES):
        v[i] = min(sum(1 for u in ranked if u.role == role), COUNT_CAP) / COUNT_CAP
    headline = next((u for u in ranked if u.role == "headline"), None)
    if headline is not None:
        text = " ".join(str(e.get("content", "")) for e in _unit_elements(headline, by_id) if e.get("type") == "text")
        v[9] = min(len(text), HEADLINE_CHARS_CAP) / HEADLINE_CHARS_CAP
        v[10] = min(len(text.split()), HEADLINE_WORDS_CAP) / HEADLINE_WORDS_CAP
    hero = next((u for u in ranked if u.role == "hero_image"), None)
    logo = next((u for u in ranked if u.role == "logo"), None)
    v[11] = _unit_aspect(hero, by_id) if hero is not None else 0.0
    v[12] = _unit_aspect(logo, by_id) if logo is not None else 0.0
    canvas = source.get("canvas", {})
    v[13] = _scaled_aspect(canvas.get("width"), canvas.get("height"))
    v[14] = min(len(ranked), UNITS_CAP) / UNITS_CAP
    v[15] = float(any(u.kind == "group" for u in ranked))
    v[16] = float(any(u.kind == "container_label" for u in ranked))
    return v


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


# ---------------------------------------------------------------- the library

def library(proposals: list[dict] | None = None, builds: list[dict] | None = None,
            judgments: list[dict] | None = None, vector_of=None) -> list[dict]:
    """The WINNING B2 plans (same strict rule as floorplans.cluster: every current-rubric
    verdict on the plan picks it with confidence >= 2) of TRAIN ads, one entry each:
    {proposal_id, task_id, cell, aspect_class, vector, plan}. `vector_of(task_id)` gives
    the source ad's vector (default: from the corpus via the heuristic priority plan).
    Ads it cannot describe are skipped. Sorted by proposal_id."""
    proposals = fp.read_jsonl(fp.PROPOSALS_FILE) if proposals is None else proposals
    builds = fp.read_jsonl(fp.BUILDS_FILE) if builds is None else builds
    judgments = fp.read_jsonl(fp.JUDGMENTS_FILE) if judgments is None else judgments
    vector_of = vector_of or _corpus_vector_of()
    by_id = {p["proposal_id"]: p for p in proposals if p.get("split", "train") == "train"}
    ids = {(b["cell"], b["candidate_id"]): b["proposal_id"] for b in builds
           if b.get("built") and b["proposal_id"] in by_id}
    outcomes = fp._outcomes(judgments, ids)
    entries, vectors = [], {}
    for pid in sorted(outcomes):
        if not fp._won(outcomes[pid]):
            continue
        p = by_id[pid]
        if p["task_id"] not in vectors:
            try:
                vectors[p["task_id"]] = vector_of(p["task_id"])
            except Exception as e:  # one undescribable ad must not stop the library
                log(f"{p['task_id']}: no vector ({type(e).__name__}: {e}); skipping its plans")
                vectors[p["task_id"]] = None
        if vectors[p["task_id"]] is None:
            continue
        entries.append({"proposal_id": pid, "task_id": p["task_id"], "cell": p["cell"],
                        "aspect_class": p["aspect_class"], "vector": np.asarray(vectors[p["task_id"]], dtype=float),
                        "plan": p["plan"]})
    return entries


def _corpus_vector_of(assets_dir=None):
    from dev.distill.corpus import load_corpus
    from engine.pipeline.priority import heuristic_plan

    designs = {d["task_id"]: d for d in load_corpus()}
    assets = assets_dir or "brand_kit/assets"

    def vector_of(task_id):
        return ad_vector(designs[task_id]["source"], heuristic_plan(designs[task_id]["source"], assets))

    return vector_of


# ---------------------------------------------------------------- recommend

def _content_key(plan: dict) -> str:
    return json.dumps([plan["classes"], plan["margins"], plan["slots"]], sort_keys=True)


def recommended_name(plan: dict) -> str:
    """A unique, content-derived name (`r` + 8 hex): the same plan always gets the same name,
    different plans never share one, and it never collides with a built-in or B2 name."""
    return "r" + hashlib.sha1(_content_key(plan).encode()).hexdigest()[:8]


def target_class(target) -> str:
    """Aspect class of a target given as a class name, or a dict/obj with width and height."""
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        return aspect_class(target["width"], target["height"])
    return aspect_class(target.width, target.height)


def nearest_ads(vector: np.ndarray, cls: str, lib: list[dict], k: int = DEFAULT_K,
                exclude_task: str | None = None) -> list[tuple[str, float]]:
    """(task_id, distance) of the k nearest library ads that have a winning plan in `cls`
    (ties by task_id)."""
    ads: dict[str, np.ndarray] = {}
    for e in lib:
        if e["aspect_class"] == cls and e["task_id"] != exclude_task:
            ads[e["task_id"]] = e["vector"]
    ranked = sorted(((distance(vector, v), t) for t, v in ads.items()))
    return [(t, d) for d, t in ranked[:k]]


def recommend(source: dict, target, k: int = DEFAULT_K, lib: list[dict] | None = None, plan=None,
              assets_dir=None, exclude_task: str | None = None) -> list[dict]:
    """The winning plans of the k nearest training ads, within the target's aspect class.
    `target` is a dict with width and height (or an aspect class name). Plans are renamed
    `r<hash>` (unique by content), their `classes` set to the target's class, duplicates
    dropped, and each one is valid for `extra_templates._parse_plan`.
    `plan` is the source's PriorityPlan (default: the heuristic one)."""
    if plan is None:
        from engine.pipeline.priority import heuristic_plan

        plan = heuristic_plan(source, assets_dir or "brand_kit/assets")
    lib = library() if lib is None else lib
    cls = target_class(target)
    near = nearest_ads(ad_vector(source, plan), cls, lib, k, exclude_task)
    out, seen = [], set()
    for task, _ in near:
        for e in sorted((e for e in lib if e["task_id"] == task and e["aspect_class"] == cls),
                        key=lambda e: e["proposal_id"]):
            src = e["plan"]
            new = {"name": "", "classes": [cls], "margins": list(src["margins"]),
                   "slots": [{"name": s["name"], "box": list(s["box"])} for s in src["slots"]]}
            key = _content_key(new)
            if key in seen:
                continue
            seen.add(key)
            new["name"] = recommended_name(new)
            try:
                _parse_plan(new, {})
            except ValueError as err:
                log(f"dropping {e['proposal_id']}: {err}")
                continue
            out.append(new)
    return out


# ---------------------------------------------------------------- evaluate (holdout)

def evaluate_build(lib: list[dict], pool_rows: list[dict], designs: dict, targets: dict, assets_dir: Path,
                   out: Path = RET_BUILDS_FILE, fp_dir: Path = FP_DIR, k: int = DEFAULT_K, shard=None,
                   done_files=None, classes=EVAL_CLASSES) -> list[dict]:
    """For each HOLDOUT cell of a target in `classes`: recommend plans from the library, build
    each (alone, at each margin) with the real engine, keep the best by the default grader.
    One row per cell (append-only, resumable): built or not, the plans tried, the nearest ads,
    and whether the engine's own grader would pick the new build over the cell's current winner."""
    from engine.pipeline.priority import heuristic_plan
    from engine.pipeline.textmeasure import TextMeasurer

    winners = fp.cell_winners([r for r in pool_rows if r["split"] == "holdout"])
    done = {r["cell"] for f in [out, *(done_files or [])] for r in fp.read_jsonl(f, shards=False)}
    cells = sorted(c for c in winners if c.split("@")[0] in designs and c.split("@")[1] in targets
                   and fp.aspect_of(c.split("@")[1], targets) in classes)
    if shard is not None:
        cells = fp.in_shard(cells, shard)
    measurer, cache, added = None, {}, []
    for cell in cells:
        if cell in done:
            continue
        task, target = cell.split("@")
        cls = fp.aspect_of(target, targets)
        w = winners[cell]
        base = {"cell": cell, "task_id": task, "split": "holdout", "target": target, "aspect_class": cls}
        tail = {"winner_id": w["candidate_id"], "winner_total": fp.score(w["metrics"])}
        failed = {"plan": None, "plans": [], "neighbors": [], "candidate_id": None, "metrics": None, "total": None,
                  "png": None, "built": False, "n_candidates": 0, "n_passed": 0, "engine_picks_new": False}
        try:
            prio = fp._prio(task, designs, assets_dir, cache)
            source = designs[task]["source"]
            near = nearest_ads(ad_vector(source, prio), cls, lib, k, exclude_task=task)
            plans = recommend(source, cls, k, lib=lib, plan=prio, exclude_task=task)
        except Exception as e:
            row = base | failed | {"reason_if_not": f"error {type(e).__name__}: {e}"} | tail
            fp._append(out, [row])
            added.append(row)
            continue
        neighbors = [{"task_id": t, "distance": round(d, 4)} for t, d in near]
        tried, best = [], None
        if plans:
            measurer = measurer or TextMeasurer()
            with tempfile.TemporaryDirectory(prefix="retrieval-") as tmp:
                path = Path(tmp) / "recommended.json"
                path.write_text(json.dumps({"floor_plans": plans}))
                with fp._env(str(path)):
                    for p in plans:
                        png_rel = f"{RENDERS_SUBDIR}/retrieval/{task}/{target}/{p['name']}.png"
                        try:
                            r = fp._build_one(p["name"], p["margins"], source, targets[target], prio, measurer,
                                              assets_dir, png_rel, fp_dir)
                        except Exception as e:
                            r = {"candidate_id": None, "metrics": None, "total": None, "png": None, "built": False,
                                 "reason_if_not": f"error {type(e).__name__}: {e}", "n_candidates": 0, "n_passed": 0}
                        tried.append({"name": p["name"], "built": r["built"], "total": r["total"],
                                      "candidate_id": r["candidate_id"], "reason_if_not": r["reason_if_not"]})
                        if r["built"] and (best is None or (-r["total"], r["candidate_id"]) < (-best[1]["total"], best[1]["candidate_id"])):
                            best = (p["name"], r)
        if best:
            name, r = best
            picks = (-r["total"], r["candidate_id"]) < fp._rank_key(w)
            row = base | r | {"plan": name, "plans": tried, "neighbors": neighbors, "engine_picks_new": picks} | tail
        else:
            reason = "no ad in the library has a winning plan for this class" if not plans else \
                "no recommended plan built: " + "; ".join(str(t["reason_if_not"]) for t in tried)
            row = base | failed | {"plans": tried, "neighbors": neighbors, "reason_if_not": reason} | tail
        fp._append(out, [row])
        added.append(row)
        print(f"{cell}: {sum(t['built'] for t in tried)}/{len(plans)} recommended plans built"
              + (f", best {best[0]}" if best else ""))
    return added


def evaluate_pairs(builds: list[dict], winners: dict, pool_dir: Path = POOL_DIR, fp_dir: Path = FP_DIR):
    """(current winner, retrieval's best build) for every built holdout row."""
    return fp.judge_pairs_for(builds, winners, pool_dir, fp_dir)


def _verdicts(builds: list[dict], judgments: list[dict]) -> dict:
    """cell -> [(picked new?, confidence)] for each retrieval build."""
    ids = {(b["cell"], b["candidate_id"]): b["cell"] for b in builds if b.get("built")}
    return fp._outcomes(judgments, ids)


def _lost(verdicts) -> bool:
    """Mirror of the win rule: every verdict picks the current winner with confidence >= 2."""
    return bool(verdicts) and all((not picked) and conf >= fp.WIN_CONFIDENCE for picked, conf in verdicts)


def _tally(cells: list[str], built: set, picks: set, verdicts: dict) -> dict:
    judged = [c for c in cells if c in verdicts]
    won = [c for c in judged if fp._won(verdicts[c])]
    changed = [c for c in cells if c in picks and c in built]
    cw = sum(1 for c in changed if c in verdicts and fp._won(verdicts[c]))
    cl = sum(1 for c in changed if c in verdicts and _lost(verdicts[c]))
    return {"cells": len(cells), "built": sum(1 for c in cells if c in built), "judged": len(judged), "wins": len(won),
            "win_rate": len(won) / len(judged) if judged else None,
            "picks_changed": len(changed), "picks_changed_wins": cw, "picks_changed_loses": cl}


def clustered_replay(eval_builds: list[dict], eval_judgments: list[dict], cells: set) -> dict:
    """The clustered templates' results on the same cells: per-plan win rate, and the final-pick
    replay (the engine's grader picks the best of the current winner and every template's build;
    where that changes the pick, the teacher's verdict on that build decides)."""
    rows = [b for b in eval_builds if b["cell"] in cells]
    ids = {(b["cell"], b["candidate_id"]): (b["cell"], b["plan"]) for b in rows if b.get("built")}
    outcomes = fp._outcomes(eval_judgments, ids)
    plan_judged = [k for k in outcomes]
    plan_wins = sum(1 for k in plan_judged if fp._won(outcomes[k]))
    by_cell: dict[str, list[dict]] = {}
    for b in rows:
        if b.get("built") and b.get("engine_picks_new"):
            by_cell.setdefault(b["cell"], []).append(b)
    chosen, verdicts = {}, {}
    for cell, bs in by_cell.items():
        b = min(bs, key=lambda b: (-b["total"], b["candidate_id"]))
        chosen[cell] = b
        if (cell, b["plan"]) in outcomes:
            verdicts[cell] = outcomes[(cell, b["plan"])]
    tcells = sorted({b["cell"] for b in rows})
    tally = _tally(tcells, set(chosen), set(chosen), verdicts)
    return {"plan_level": {"plan_cell_pairs_judged": len(plan_judged), "wins": plan_wins,
                           "win_rate": plan_wins / len(plan_judged) if plan_judged else None},
            "final_pick_replay": {"cells": len(tcells), "picks_changed": tally["picks_changed"],
                                  "wins": tally["picks_changed_wins"], "loses": tally["picks_changed_loses"]}}


def evaluate_summary(builds: list[dict], judgments: list[dict], eval_builds: list[dict] | None = None,
                     eval_judgments: list[dict] | None = None) -> dict:
    """Retrieval's results over all cells and per class, and (when the clustered files are given)
    the clustered templates on exactly the same cells."""
    verdicts = _verdicts(builds, judgments)
    built = {b["cell"] for b in builds if b.get("built")}
    picks = {b["cell"] for b in builds if b.get("engine_picks_new")}
    summary = {"rubric_version": RUBRIC_VERSION,
               "rule": f"win = every verdict picks the retrieved build over the cell's current winner with "
                       f"confidence >= {fp.WIN_CONFIDENCE}; best of the recommended plans by the default grader",
               "retrieval": {"all": _tally(sorted({b["cell"] for b in builds}), built, picks, verdicts), "by_class": {}}}
    for cls in sorted({b["aspect_class"] for b in builds}):
        cells = sorted({b["cell"] for b in builds if b["aspect_class"] == cls})
        summary["retrieval"]["by_class"][cls] = _tally(cells, built, picks, verdicts)
    if eval_builds is not None:
        cells = {b["cell"] for b in builds}
        summary["clustered_same_cells"] = clustered_replay(eval_builds, eval_judgments or [], cells)
        summary["clustered_same_cells"]["cells_with_clustered_builds"] = len(
            {b["cell"] for b in eval_builds if b["cell"] in cells})
    return summary


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def render_report(summary: dict) -> str:
    r = summary["retrieval"]
    a = r["all"]
    lines = ["# Retrieval (C) vs clustered templates (B2), holdout", "",
             "For each holdout cell of a tall / very_tall / square target, the plans of the k nearest training ads "
             "(by `ad_vector`) were built by the engine; the best by the default grader was judged against the "
             "cell's current winner with the harsh rubric. " + summary["rule"] + ".", "",
             "## Retrieval", "", "| class | cells | built | judged | wins | win rate | picks changed | new pick wins - loses |",
             "|---|---|---|---|---|---|---|---|"]
    for name, t in [*r["by_class"].items(), ("all", a)]:
        lines.append(f"| {name} | {t['cells']} | {t['built']} | {t['judged']} | {t['wins']} | {_pct(t['win_rate'])} | "
                     f"{t['picks_changed']} | {t['picks_changed_wins']} - {t['picks_changed_loses']} |")
    c = summary.get("clustered_same_cells")
    if c:
        pl, fpr = c["plan_level"], c["final_pick_replay"]
        lines += ["", "## Clustered templates on the same cells", "",
                  f"- {c['cells_with_clustered_builds']} of the {a['cells']} cells have clustered-template builds.",
                  f"- Per (template, cell) pair: {pl['wins']} of {pl['plan_cell_pairs_judged']} judged won "
                  f"({_pct(pl['win_rate'])}). One plan per cell is not the same unit as retrieval's best of k, "
                  "so read this as the cost of a generic template, not a like-for-like win rate.",
                  f"- Final-pick replay (all 14 templates added): {fpr['picks_changed']} picks changed of "
                  f"{fpr['cells']}, new pick wins - loses {fpr['wins']} - {fpr['loses']}.",
                  "", "## Comparison", "",
                  "| | picks changed | wins - loses | judged win rate |", "|---|---|---|---|",
                  f"| retrieval (k plans, best by grader) | {a['picks_changed']} | {a['picks_changed_wins']} - "
                  f"{a['picks_changed_loses']} | {_pct(a['win_rate'])} |",
                  f"| clustered templates (final pick) | {fpr['picks_changed']} | {fpr['wins']} - {fpr['loses']} | "
                  f"{_pct(pl['win_rate'])} (per pair) |"]
    lines += ["", "Caveat: the teacher's verdicts exist only for what was judged; cells where the grader would keep "
              "the current winner contribute no change to the final pick either way."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["evaluate"])
    ap.add_argument("--stage", choices=["build", "judge", "summary"], default="build")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--swap-share", type=float, default=0.2)
    ap.add_argument("--pairs-per-call", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-calls", type=int, default=None)
    ap.add_argument("--shard", default=None, help="K/N: only shard K's cells, writing <file>.shardKofN.jsonl")
    ap.add_argument("--merge", action="store_true", help="fold this stage's shard files into its main file")
    args = ap.parse_args(argv)
    os.environ["CHAMELEON_LLM"] = "heuristic"
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    main_file = {"build": RET_BUILDS_FILE, "judge": RET_JUDGMENTS_FILE, "summary": None}[args.stage]
    if args.merge:
        if main_file is None:
            ap.error("--merge needs --stage build or judge")
        print(f"merged {merge_shards(main_file)} new rows into {main_file}")
        return 0
    out = shard_file(main_file, *shard) if shard and main_file else main_file
    done_files = [main_file] if shard and main_file else None

    if args.stage == "summary":
        eb, ej = fp.read_jsonl(fp.EVAL_BUILDS_FILE), fp.read_jsonl(fp.EVAL_JUDGMENTS_FILE)
        summary = evaluate_summary(fp.read_jsonl(RET_BUILDS_FILE), fp.read_jsonl(RET_JUDGMENTS_FILE), eb, ej)
        RET_SUMMARY_FILE.write_text(json.dumps(summary, indent=1))
        RET_REPORT_FILE.write_text(render_report(summary))
        print(json.dumps(summary, indent=1))
        return 0

    if args.stage == "judge":
        from dev.distill.teacher import ClaudeCodeTeacher

        winners = fp.cell_winners(fp._pool_rows())
        pairs = evaluate_pairs(fp.read_jsonl(RET_BUILDS_FILE), winners)
        added = fp.judge(pairs, ClaudeCodeTeacher(args.model), POOL_DIR, out, args.swap_share, args.pairs_per_call,
                         args.seed, args.limit_calls, shard=shard, done_files=done_files)
        print(f"judged {len(added)} new rows over {len(pairs)} pairs -> {out}")
        return 0

    from dev.distill.assets import merged_assets_dir

    targets, designs = fp.load_targets(), fp._designs()
    with merged_assets_dir() as assets:
        lib = library(vector_of=_designs_vector_of(designs, assets))
        log(f"library: {len(lib)} winning plans from {len({e['task_id'] for e in lib})} ads")
        added = evaluate_build(lib, fp._pool_rows(), designs, targets, assets, out, FP_DIR, args.k, shard, done_files)
        print(f"built {sum(r['built'] for r in added)} of {len(added)} holdout cells -> {out}")
    return 0


def _designs_vector_of(designs: dict, assets):
    from engine.pipeline.priority import heuristic_plan

    def vector_of(task_id):
        return ad_vector(designs[task_id]["source"], heuristic_plan(designs[task_id]["source"], assets))

    return vector_of


if __name__ == "__main__":
    raise SystemExit(main())
