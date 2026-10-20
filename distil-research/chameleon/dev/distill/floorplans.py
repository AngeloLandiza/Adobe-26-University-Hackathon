"""B2: teacher-designed floor plans -> built by the engine -> judged against the
cell's current winner -> winners clustered into new templates.

    python -m dev.distill.floorplans propose  [--cells 160] [--seed 0] [--limit-calls N] [--shard K/N | --merge]
    python -m dev.distill.floorplans build    [--shard K/N | --merge]
    python -m dev.distill.floorplans judge    [--swap-share 0.2] [--pairs-per-call 4] [--limit-calls N]
                                              [--shard K/N | --merge]
    python -m dev.distill.floorplans cluster
    python -m dev.distill.floorplans evaluate --stage build|judge|summary [--candidate FILE] [--shard K/N | --merge]

Words: a *cell* is one ad at one target size; a *floor plan* cuts the canvas's
content box into named slots (hero, text, logo, cta, badge, minor), in the
format of engine/pipeline/layout/extra_templates.json; the cell's *current
winner* is its best pool candidate by today's default grader.

1. propose: the teacher (a senior layout designer) sees the original ad, the
   current winner, the target size and the ad's units, and draws 3 floor plans.
   Plans the engine would reject are skipped with a logged reason.
2. build: each plan alone is filled by the real engine (one candidate per
   margin); the best gate-passing candidate is measured and rendered.
3. judge: each built plan's best candidate vs the cell's current winner, with
   dev.distill.judge (harsh rubric, swapped re-asks).
4. cluster: plans that clearly beat the winner are grouped per aspect class
   with k-medoids; the medoids (real winning plans) become candidate templates
   in extra_templates.candidate.json, next to the shipped hand-drawn plans.
5. evaluate: rebuilds the candidate file's new plans on HOLDOUT cells and pairs
   them with the current winner for judging (run by the controller).

Every step appends as it goes and skips finished work on a rerun. `--shard K/N`
does only shard K's cells and writes `<file>.shardKofN.jsonl`; `--merge` folds
the shard files into the main file.
"""

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from dev.distill import judge as judge_mod
from dev.distill.judge import RUBRIC, RUBRIC_VERSION, merge_shards, shard_file
from dev.distill.paths import DATA_ROOT, POOL_DIR, RENDERS_SUBDIR
from dev.distill.pool import RENDER_SIZE, TARGETS_PATH, _total
from dev.distill.teacher import TeacherUnavailable
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout import extra_templates
from engine.pipeline.layout.aspect import ASPECT_CLASSES, aspect_class
from engine.pipeline.layout.extra_templates import SLOT_NAMES, _parse_plan
from engine.pipeline.layout.generate import _template_candidate

FP_DIR = DATA_ROOT / "floor-plans"
PROPOSALS_FILE = FP_DIR / "proposals.jsonl"
BUILDS_FILE = FP_DIR / "builds.jsonl"
JUDGMENTS_FILE = FP_DIR / "judgments.jsonl"
CLUSTERS_FILE = FP_DIR / "clusters.json"
CANDIDATE_FILE = FP_DIR / "extra_templates.candidate.json"
EVAL_BUILDS_FILE = FP_DIR / "eval-builds.jsonl"
EVAL_JUDGMENTS_FILE = FP_DIR / "eval-judgments.jsonl"
EVAL_SUMMARY_FILE = FP_DIR / "eval-summary.json"
SHIPPED_FILE = Path(extra_templates.__file__).with_name("extra_templates.json")
ENV_VAR = extra_templates.ENV_VAR

PLANS_PER_CELL = 3
WIN_CONFIDENCE = 2   # a plan "beats" the winner only if every verdict on it picks it with at least this confidence
MAX_K = 5
DIGITS = 4           # slot boxes are rounded to this many decimals


def log(msg: str) -> None:
    print(f"[floorplans] {msg}", file=sys.stderr)


def score(metrics: dict) -> float:
    """Today's default grader total (the same number the pool ranks by)."""
    return _total(metrics)


def aspect_of(target_name: str, targets: dict) -> str:
    t = targets[target_name]
    return aspect_class(t["width"], t["height"])


def load_targets(path: Path = TARGETS_PATH) -> dict:
    return {t["name"]: t for t in json.loads(Path(path).read_text())}


def read_jsonl(main: Path, shards: bool = True) -> list[dict]:
    """Rows of `main` plus (by default) any not-yet-merged shard files."""
    main = Path(main)
    files = [main] + (sorted(main.parent.glob(f"{main.stem}.shard*of*{main.suffix}")) if shards else [])
    return [json.loads(x) for f in files if f.exists() for x in f.read_text().splitlines() if x.strip()]


def _append(out: Path, rows: list[dict]) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def in_shard(cells, shard) -> list:
    """The cells (or rows with a "cell") in shard K of N, by the same stable hash as dev.distill.judge."""
    return [c for c in cells if judge_mod._in_shard(c if isinstance(c, str) else c["cell"], shard)]


# ---------------------------------------------------------------- winners and cells

def _rank_key(row: dict) -> tuple:
    return -score(row["metrics"]), row["candidate_id"]


def cell_winners(pool_rows: list[dict]) -> dict[str, dict]:
    """cell -> its best pool row by the default grader (ties by candidate_id).
    The safe fallback wins only when it is the cell's only candidate."""
    by_cell: dict[str, list[dict]] = {}
    for r in pool_rows:
        by_cell.setdefault(r["cell"], []).append(r)
    winners = {}
    for cell, rows in by_cell.items():
        real = [r for r in rows if not r.get("is_fallback")] or rows
        winners[cell] = min(real, key=_rank_key)
    return winners


def pick_propose_cells(pool_rows: list[dict], n: int, seed: int = 0, targets: dict | None = None,
                       classes=None) -> list[str]:
    """`n` train cells via judge.pick_cells, stratified by brand x aspect class
    (at least one cell per group), so every aspect class appears. `classes`
    restricts to those aspect classes (budget control). Sorted."""
    targets = targets or load_targets()
    rows = [dict(r, brand=f'{r.get("brand", "unknown")}|{aspect_of(r["target"], targets)}')
            for r in pool_rows if r["split"] == "train"
            and (not classes or aspect_of(r["target"], targets) in classes)]
    return sorted(judge_mod.pick_cells(rows, n, seed))


def _element_summary(el: dict) -> dict:
    if el["type"] == "text":
        return {"type": "text", "content": el.get("content", ""), "font_size": el.get("font_size")}
    if el["type"] == "image":
        return {"type": "image", "asset_id": el.get("asset_id"),
                "aspect": round(el["width"] / el["height"], 2) if el.get("height") else None}
    return {"type": el["type"]}


def cell_context(cell: str, winners: dict, designs: dict, targets: dict, assets_dir: Path,
                 pool_dir: Path = POOL_DIR, plan_cache: dict | None = None) -> dict:
    """What the teacher is shown for one cell (see `propose`)."""
    from engine.pipeline.priority import heuristic_plan

    task, target = cell.split("@")
    design = designs[task]
    if plan_cache is not None and task in plan_cache:
        plan = plan_cache[task]
    else:
        plan = heuristic_plan(design["source"], assets_dir)
        if plan_cache is not None:
            plan_cache[task] = plan
    els = {e["id"]: e for e in design["source"]["elements"]}
    units = [{"rank": u.rank, "role": u.role, "kind": u.kind,
              "elements": [_element_summary(els[i]) for i in u.element_ids if i in els]}
             for u in plan.ranked_units()]
    t = targets[target]
    winner = winners[cell]
    return {"cell": cell, "task_id": task, "split": design["split"], "target": target,
            "width": t["width"], "height": t["height"], "aspect_class": aspect_class(t["width"], t["height"]),
            "units": units, "layers": sum(1 for u in plan.units if u.rank is None),
            "source_png": Path(pool_dir) / RENDERS_SUBDIR / task / "source.png",
            "winner_png": Path(pool_dir) / winner["png"], "winner_id": winner["candidate_id"]}


# ---------------------------------------------------------------- propose

def propose_schema() -> dict:
    slot = {"type": "object", "required": ["name", "box"], "additionalProperties": False,
            "properties": {"name": {"enum": list(SLOT_NAMES)},
                           "box": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                                   "minItems": 4, "maxItems": 4}}}
    plan = {"type": "object", "required": ["idea", "margins", "slots"], "additionalProperties": False,
            "properties": {"idea": {"type": "string"},
                           "margins": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 0.2},
                                       "minItems": 1, "maxItems": 2},
                           "slots": {"type": "array", "items": slot, "minItems": 2, "maxItems": len(SLOT_NAMES)}}}
    return {"type": "object", "required": ["plans"],
            "properties": {"plans": {"type": "array", "items": plan,
                                     "minItems": PLANS_PER_CELL, "maxItems": PLANS_PER_CELL}}}


def _unit_line(u: dict) -> str:
    parts = []
    for e in u["elements"]:
        if e["type"] == "text":
            parts.append(f'text "{e["content"]}"' + (f' ({e["font_size"]}px in the original)' if e.get("font_size") else ""))
        elif e["type"] == "image":
            parts.append(f'image {e.get("asset_id")} (aspect w/h {e.get("aspect")})')
        else:
            parts.append(e["type"])
    return f'- rank {u["rank"]}: {u["role"] or "unlabelled"} ({u["kind"]}): ' + "; ".join(parts)


def propose_prompt(ctx: dict) -> str:
    w, h = ctx["width"], ctx["height"]
    short = min(w, h)
    cw, ch = round(w - 2 * 0.04 * short), round(h - 2 * 0.04 * short)
    units = "\n".join(_unit_line(u) for u in ctx["units"]) or "- (none)"
    return f"""You are a senior layout designer. A machine resizes finished marketing designs to new canvas sizes by filling a FLOOR PLAN: a set of named, non-overlapping boxes ("slots") that the design's pieces are placed into. Your job: draw {PLANS_PER_CELL} floor plans that would make an excellent resize of this ad to the target below, better than what the machine does now.

Your layouts will be judged by this rubric, so design to it:

----- RUBRIC -----
{RUBRIC}
----- END RUBRIC -----

Image 1 is the ORIGINAL ad. Image 2 is what the engine does now for this target (its current best layout): beat it. Read both image files, in parallel in one step, before you answer.

TARGET: {w} x {h} px, aspect class "{ctx["aspect_class"]}" (width/height = {w / h:.2f}).

THE AD'S PIECES, most important first (rank 1 = most important):
{units}
Plus {ctx["layers"]} background/decoration layer(s), which are placed automatically across the canvas: do not give them slots.

HOW A FLOOR PLAN IS FILLED (so you can predict the result):
- Boxes are [x, y, w, h] as fractions (0-1) of the CONTENT BOX: the canvas minus a margin on every side, where margin = a "margins" value x the canvas's shorter side. At margin 0.04 the content box here is about {cw} x {ch} px, so a box [0, 0, 1, 0.5] is about {cw} x {round(ch / 2)} px.
- Pieces are handed out in rank order; consecutive pieces of the same role (e.g. several badges in a row of the list) travel together as one group, laid out side by side in one slot.
- "hero": the most important image (hero_image or secondary_image) or image group, scaled to fit, keeping its proportions. Only one image or group gets the hero slot.
- "text": the headline, subheadline and other copy, stacked top to bottom in rank order; text wraps and shrinks to fit, so a small text box means small, hard-to-read text.
- "logo", "cta", "badge": the first piece (or group) of that role, scaled to fit.
- "minor": required. Takes everything no other slot accepts (legal text, extra images, extra badges...), set small.
- A piece whose role has no slot is squeezed small into leftover space: give every important piece a proper slot. Leave out slots for roles this ad does not have (except text and minor, which are always required).
- Each slot name at most once. Slots must stay inside the unit square and must not overlap (more than 1% of the smaller slot's area is rejected).
- "margins": one or two margin values to try (0.04 and 0.07 are typical).

Give {PLANS_PER_CELL} genuinely different structures for this exact shape and this ad (not three variations of one idea), each with a one-sentence "idea". Think about hierarchy, reading order, grid alignment, whitespace, grouping and fit to format as the rubric describes."""


def _validate_answer(answer: dict) -> None:
    """Structural check only: plans the engine would reject are skipped one by one, not retried."""
    plans = answer.get("plans") if isinstance(answer, dict) else None
    if not isinstance(plans, list) or not plans:
        raise ValueError("no plans list")
    for p in plans:
        if not isinstance(p, dict) or not isinstance(p.get("slots"), list):
            raise ValueError("a plan without a slots list")


def _round(v):
    return round(float(v), DIGITS) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def _normalise(raw: dict, aspect: str) -> dict:
    slots = [{"name": s.get("name"), "box": [_round(v) for v in s.get("box", [])] if isinstance(s.get("box"), list)
              else s.get("box")} if isinstance(s, dict) else s for s in raw.get("slots", [])]
    margins = raw.get("margins", [0.04, 0.07])
    margins = [_round(m) for m in margins] if isinstance(margins, list) else margins
    return {"classes": [aspect], "margins": margins, "slots": slots}


def _plan_id(cell: str, teacher: str, plan: dict) -> str:
    key = json.dumps([cell, teacher, plan["margins"], plan["slots"]], sort_keys=True)
    return "p" + hashlib.sha1(key.encode()).hexdigest()[:8]


def _done_cells(files, teacher_name: str) -> set:
    return {r["cell"] for f in files for r in read_jsonl(f, shards=False)
            if r.get("teacher") == teacher_name and r.get("rubric_version") == RUBRIC_VERSION}


def propose(cells: list[str], context, teacher, out: Path = PROPOSALS_FILE, limit_calls: int | None = None,
            done_files=None) -> list[dict]:
    """Ask the teacher for floor plans for each cell not yet done by this teacher;
    append the valid ones to `out`. `context(cell)` gives what the teacher sees.
    Returns the rows added."""
    done = _done_cells([out, *(done_files or [])], teacher.name)
    added, calls = [], 0
    for cell in cells:
        if cell in done:
            continue
        if limit_calls is not None and calls >= limit_calls:
            break
        calls += 1
        try:
            ctx = context(cell)
            answer = teacher.ask(propose_prompt(ctx), propose_schema(), [ctx["source_png"], ctx["winner_png"]],
                                 validate=_validate_answer)
            _validate_answer(answer)
        except TeacherUnavailable as e:
            log(f"{e}; stopping, progress kept")
            break
        except Exception as e:  # finished cells are on disk; a rerun retries this one
            log(f"{cell}: teacher gave up ({type(e).__name__}: {e}); skipping it")
            continue
        rows, names = [], set()
        for i, raw in enumerate(answer["plans"], 1):
            plan = _normalise(raw if isinstance(raw, dict) else {}, ctx["aspect_class"])
            name = _plan_id(cell, teacher.name, plan)
            plan = {"name": name, **plan}
            try:
                _parse_plan(plan, {})
            except ValueError as err:
                log(f"skipping plan {i} for {cell}: {err}")
                continue
            if name in names:
                log(f"skipping plan {i} for {cell}: duplicate of an earlier plan")
                continue
            names.add(name)
            rows.append({"proposal_id": name, "cell": cell, "task_id": ctx["task_id"], "split": ctx["split"],
                         "target": ctx["target"], "aspect_class": ctx["aspect_class"],
                         "idea": raw.get("idea") if isinstance(raw, dict) else None, "plan": plan,
                         "winner_id": ctx.get("winner_id"), "teacher": teacher.name,
                         "rubric_version": RUBRIC_VERSION})
        if rows:
            _append(out, rows)
        print(f"{cell}: {len(rows)} valid plan(s)")
        added += rows
    return added


# ---------------------------------------------------------------- build

@contextlib.contextmanager
def _env(value: str):
    old = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = old


def _build_one(name: str, margins, source: dict, target: dict, prio, measurer, assets_dir: Path,
               png_rel: str, fp_dir: Path) -> dict:
    """Fill floor plan `name` (already loadable) at each margin; the best gate-passing
    candidate by the default grader is measured, rendered and described."""
    from engine.pipeline.grading.metrics import geometry_metrics
    from engine.pipeline.grading.pixels import pixel_metrics
    from engine.pipeline.render import render_to_image

    canvas = {"width": target["width"], "height": target["height"]}
    built, failures, n_cand = [], [], 0
    for m in margins:
        try:
            cand = _template_candidate(name, {"margin": m}, source, canvas, prio, measurer)
        except Exception as e:
            failures.append(f"margin={m}: build error {type(e).__name__}: {e}")
            continue
        if cand is None:
            continue
        n_cand += 1
        failed = check_gates(source, cand.design, canvas, prio, measurer)
        if failed:
            failures.append(f"margin={m}: {', '.join(failed)}")
            continue
        try:
            metrics = geometry_metrics(source, cand, prio) | pixel_metrics(cand.design, prio, assets_dir)
        except Exception as e:
            failures.append(f"margin={m}: unmeasurable ({type(e).__name__})")
            continue
        built.append((cand, metrics))
    row = {"candidate_id": None, "metrics": None, "total": None, "png": None, "built": False,
           "reason_if_not": None, "n_candidates": n_cand, "n_passed": len(built)}
    if not built:
        row["reason_if_not"] = ("every candidate failed the gates: " + "; ".join(failures) if n_cand
                                else "no candidate built" + (f" ({'; '.join(failures)})" if failures else ""))
        return row
    cand, metrics = min(built, key=lambda cm: (-score(cm[1]), cm[0].candidate_id))
    try:
        image = render_to_image(cand.design, assets_dir=assets_dir, max_size=RENDER_SIZE)
    except Exception as e:
        row["reason_if_not"] = f"best candidate unrenderable ({type(e).__name__}: {e})"
        return row
    (Path(fp_dir) / png_rel).parent.mkdir(parents=True, exist_ok=True)
    image.save(Path(fp_dir) / png_rel)
    row.update(candidate_id=cand.candidate_id, metrics=metrics, total=score(metrics), png=png_rel, built=True)
    return row


def _prio(task: str, designs: dict, assets_dir: Path, cache: dict):
    from engine.pipeline.priority import heuristic_plan

    if task not in cache:
        cache[task] = heuristic_plan(designs[task]["source"], assets_dir)
    return cache[task]


def build(proposals: list[dict], designs: dict, targets: dict, assets_dir: Path, out: Path = BUILDS_FILE,
          fp_dir: Path = FP_DIR, done_files=None) -> list[dict]:
    """Fill each not-yet-built proposal's plan, alone, with the real engine;
    append one row per proposal (built or not) to `out`. Returns the rows added."""
    from engine.pipeline.textmeasure import TextMeasurer

    done = {r["proposal_id"] for f in [out, *(done_files or [])] for r in read_jsonl(f, shards=False)}
    measurer, cache, added = None, {}, []
    for p in proposals:
        if p["proposal_id"] in done or p["task_id"] not in designs:
            continue
        measurer = measurer or TextMeasurer()
        plan = p["plan"]
        base = {"proposal_id": p["proposal_id"], "cell": p["cell"], "task_id": p["task_id"], "split": p["split"],
                "target": p["target"], "aspect_class": p.get("aspect_class")}
        png_rel = f"{RENDERS_SUBDIR}/{p['task_id']}/{p['target']}/{p['proposal_id']}.png"
        with tempfile.TemporaryDirectory(prefix="floorplan-") as tmp:
            path = Path(tmp) / f"{plan['name']}.json"
            path.write_text(json.dumps({"floor_plans": [plan]}))
            with _env(str(path)):
                try:
                    prio = _prio(p["task_id"], designs, assets_dir, cache)
                    result = _build_one(plan["name"], plan.get("margins", [0.04, 0.07]), designs[p["task_id"]]["source"],
                                        targets[p["target"]], prio, measurer, assets_dir, png_rel, fp_dir)
                except Exception as e:  # one bad ad must not stop the run
                    result = {"candidate_id": None, "metrics": None, "total": None, "png": None, "built": False,
                              "reason_if_not": f"error {type(e).__name__}: {e}", "n_candidates": 0, "n_passed": 0}
        row = base | result
        _append(out, [row])
        added.append(row)
        print(f"{p['cell']} {p['proposal_id']}: " + ("built" if row["built"] else f"not built ({row['reason_if_not']})"))
    return added


# ---------------------------------------------------------------- judge

def judge_pairs_for(builds: list[dict], winners: dict, pool_dir: Path = POOL_DIR, fp_dir: Path = FP_DIR):
    """(current winner, built plan's best candidate) per built row, with absolute png paths."""
    pairs = []
    for b in sorted(builds, key=lambda b: (b["cell"], b.get("proposal_id") or "", b.get("plan") or "")):
        if not b.get("built") or b["cell"] not in winners:
            continue
        w = winners[b["cell"]]
        winner = dict(w, png=str(Path(pool_dir) / w["png"]))
        new = {"cell": b["cell"], "task_id": b["task_id"], "split": b["split"], "candidate_id": b["candidate_id"],
               "png": str(Path(fp_dir) / b["png"]), "proposal_id": b.get("proposal_id"), "plan": b.get("plan")}
        pairs.append((winner, new))
    return pairs


def judge(pairs, teacher, pool_dir: Path = POOL_DIR, out: Path = JUDGMENTS_FILE, swap_share: float = 0.2,
          pairs_per_call: int = 4, seed: int = 0, limit_calls: int | None = None, shard=None,
          done_files=None) -> list[dict]:
    """dev.distill.judge.judge_pairs on these pairs (one cell per call, original first)."""
    return judge_mod.judge_pairs(pairs, teacher, lambda t: Path(pool_dir) / RENDERS_SUBDIR / t / "source.png", out,
                                 pairs_per_call, swap_share, seed, Path(pool_dir), limit_calls, shard=shard,
                                 done_files=done_files)


def _outcomes(judgments: list[dict], ids: dict) -> dict:
    """key -> [(picked the new candidate?, confidence)] for every current-rubric verdict
    involving a candidate in `ids` ((cell, candidate_id) -> key)."""
    out: dict = {}
    for j in judgments:
        if j.get("rubric_version") != RUBRIC_VERSION:
            continue
        for side, other in (("a", "b"), ("b", "a")):
            key = ids.get((j["cell"], j[side]))
            if key is not None:
                out.setdefault(key, []).append((j["winner"] == side, j["confidence"]))
                break
    return out


def _won(verdicts) -> bool:
    return bool(verdicts) and all(picked and conf >= WIN_CONFIDENCE for picked, conf in verdicts)


# ---------------------------------------------------------------- cluster

def vectorise(plan: dict) -> np.ndarray:
    """For each slot name in SLOT_NAMES order: [present, x, y, w, h] (zeros when absent)."""
    boxes = {s["name"]: s["box"] for s in plan["slots"]}
    return np.array([v for n in SLOT_NAMES for v in ([1.0, *map(float, boxes[n])] if n in boxes else [0.0] * 5)])


def pairwise(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))


def _assign(D: np.ndarray, medoids: list[int]) -> np.ndarray:
    return np.argmin(D[:, medoids], axis=1)  # ties -> the earlier medoid


def _cost(D: np.ndarray, medoids: list[int]) -> float:
    return float(D[:, medoids].min(axis=1).sum())


def kmedoids(D: np.ndarray, k: int) -> tuple[list[int], list[int]]:
    """PAM (greedy BUILD, then best-improvement SWAP); deterministic, ties to the lowest index.
    Returns (sorted medoid indices, label per point indexing into the medoid list)."""
    n = len(D)
    medoids = [int(np.argmin(D.sum(axis=1)))]
    while len(medoids) < k:
        nearest = D[:, medoids].min(axis=1)
        gains = [-1.0 if c in medoids else float(np.maximum(nearest - D[:, c], 0).sum()) for c in range(n)]
        medoids.append(int(np.argmax(gains)))
    best = _cost(D, medoids)
    while True:
        swap = None
        for i in range(k):
            for c in range(n):
                if c in medoids:
                    continue
                trial = medoids[:i] + [c] + medoids[i + 1:]
                cost = _cost(D, trial)
                if cost < best - 1e-12:
                    best, swap = cost, trial
        if swap is None:
            break
        medoids = swap
    medoids = sorted(medoids)
    return medoids, [int(x) for x in _assign(D, medoids)]


def silhouette(D: np.ndarray, labels) -> float:
    labels = np.asarray(labels)
    s = []
    for i in range(len(D)):
        own = labels == labels[i]
        if own.sum() <= 1:
            s.append(0.0)
            continue
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, labels == c].mean() for c in set(labels.tolist()) if c != labels[i])
        s.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return float(np.mean(s))


def choose_k(D: np.ndarray, max_k: int = MAX_K):
    """k in 2..max_k by best silhouette (ties -> smaller k), with k-medoids.
    One cluster when there are fewer than 3 points or fewer than 2 distinct ones.
    Returns (k, medoids, labels, {k: silhouette})."""
    D = np.asarray(D, dtype=float)
    n = len(D)
    distinct = sum(1 for i in range(n) if not any(D[i, j] == 0 for j in range(i)))
    top = min(max_k, n - 1, distinct)
    sil, fits = {}, {}
    for k in range(2, top + 1):
        fits[k] = kmedoids(D, k)
        sil[k] = silhouette(D, fits[k][1])
    if not sil:
        return 1, [int(np.argmin(D.sum(axis=1)))], [0] * n, {}
    k = max(sil, key=lambda k: (sil[k], -k))
    return k, fits[k][0], fits[k][1], sil


def cluster(proposals: list[dict], builds: list[dict], judgments: list[dict], clusters_path: Path = CLUSTERS_FILE,
            candidate_path: Path = CANDIDATE_FILE, shipped_path: Path = SHIPPED_FILE) -> dict:
    """Keep the plans that beat the current winner (every verdict on them picks them with
    confidence >= WIN_CONFIDENCE, so a disagreeing swapped re-ask cancels a win); per aspect
    class, cluster the winners with k-medoids. Writes clusters.json and the candidate
    extra_templates file (medoids + the shipped plans). Returns the clusters report."""
    by_id = {p["proposal_id"]: p for p in proposals}
    ids = {(b["cell"], b["candidate_id"]): b["proposal_id"] for b in builds if b.get("built")}
    outcomes = _outcomes(judgments, ids)
    report = {"rubric_version": RUBRIC_VERSION,
              "rule": f"a plan wins when every verdict on it picks it over the cell's current winner with "
                      f"confidence >= {WIN_CONFIDENCE}; medoids by k-medoids per aspect class, k by silhouette",
              "classes": {}}
    new_plans = []
    for cls in ASPECT_CLASSES:
        props = sorted((p for p in proposals if p["aspect_class"] == cls), key=lambda p: p["proposal_id"])
        if not props:
            continue
        built = {b["proposal_id"] for b in builds if b.get("built")} & {p["proposal_id"] for p in props}
        judged = sorted(pid for pid in outcomes if pid in by_id and by_id[pid]["aspect_class"] == cls)
        winners = [pid for pid in judged if _won(outcomes[pid])]
        entry = {"n_proposals": len(props), "n_built": len(built), "n_judged": len(judged), "n_wins": len(winners),
                 "win_rate": len(winners) / len(judged) if judged else None, "k": 0, "silhouette": {}, "medoids": []}
        if winners:
            X = np.stack([vectorise(by_id[pid]["plan"]) for pid in winners])
            k, medoids, labels, sil = choose_k(pairwise(X))
            # every judged plan of the class (winners and losers) counts for its nearest medoid's win rate
            J = np.stack([vectorise(by_id[p]["plan"]) for p in judged])
            judged_labels = np.argmin(np.sqrt(((J[:, None, :] - X[medoids][None, :, :]) ** 2).sum(-1)), axis=1)
            groups = []
            for c, m in enumerate(medoids):
                members = sum(1 for lab in labels if lab == c)
                n_j = int(sum(1 for lab in judged_labels if lab == c))
                groups.append({"proposal_id": winners[m], "cell": by_id[winners[m]]["cell"], "members": members,
                               "judged": n_j, "win_rate": members / n_j if n_j else None,
                               "member_ids": [winners[i] for i, lab in enumerate(labels) if lab == c]})
            groups.sort(key=lambda g: (-g["members"], g["proposal_id"]))
            for i, g in enumerate(groups, 1):
                src = by_id[g["proposal_id"]]["plan"]
                plan = {"name": f"b2_{cls}_{i}", "classes": [cls], "margins": src["margins"], "slots": src["slots"]}
                g["name"], g["plan"] = plan["name"], plan
                new_plans.append(plan)
            entry.update(k=k, silhouette={str(kk): v for kk, v in sil.items()}, medoids=groups)
        report["classes"][cls] = entry
    shipped = json.loads(Path(shipped_path).read_text()).get("floor_plans", []) if Path(shipped_path).exists() else []
    candidate = {"note": "B2 candidate floor plans: teacher-plan medoids, then the shipped hand-drawn plans. "
                         "Not shipped: the controller decides after holdout evaluation.",
                 "floor_plans": new_plans + shipped}
    Path(clusters_path).parent.mkdir(parents=True, exist_ok=True)
    Path(clusters_path).write_text(json.dumps(report, indent=1))
    Path(candidate_path).write_text(json.dumps(candidate, indent=1))
    return report


# ---------------------------------------------------------------- evaluate (holdout)

def evaluate_build(candidate_path: Path, shipped_path: Path, pool_rows: list[dict], designs: dict, targets: dict,
                   assets_dir: Path, out: Path = EVAL_BUILDS_FILE, fp_dir: Path = FP_DIR, shard=None,
                   done_files=None) -> list[dict]:
    """For each HOLDOUT cell and each NEW plan in `candidate_path` (one not in the shipped
    file) covering the cell's aspect class: build that plan's best candidate. Records
    whether the engine's default grader would pick it over the cell's current winner."""
    from engine.pipeline.textmeasure import TextMeasurer

    shipped = set()
    if Path(shipped_path).exists():
        shipped = {p.get("name") for p in json.loads(Path(shipped_path).read_text()).get("floor_plans", [])}
    new = [p for p in extra_templates._parse(Path(candidate_path).read_text()).values() if p.name not in shipped]
    winners = cell_winners([r for r in pool_rows if r["split"] == "holdout"])
    done = {(r["cell"], r["plan"]) for f in [out, *(done_files or [])] for r in read_jsonl(f, shards=False)}
    cells = sorted(c for c in winners if c.split("@")[0] in designs)
    if shard is not None:
        cells = in_shard(cells, shard)
    measurer, cache, added = None, {}, []
    with _env(str(Path(candidate_path).resolve())):
        for cell in cells:
            task, target = cell.split("@")
            cls = aspect_of(target, targets)
            for plan in new:
                if cls not in plan.classes or (cell, plan.name) in done:
                    continue
                measurer = measurer or TextMeasurer()
                png_rel = f"{RENDERS_SUBDIR}/eval/{task}/{target}/{plan.name}.png"
                try:
                    result = _build_one(plan.name, plan.margins, designs[task]["source"], targets[target],
                                        _prio(task, designs, assets_dir, cache), measurer, assets_dir, png_rel, fp_dir)
                except Exception as e:
                    result = {"candidate_id": None, "metrics": None, "total": None, "png": None, "built": False,
                              "reason_if_not": f"error {type(e).__name__}: {e}", "n_candidates": 0, "n_passed": 0}
                w = winners[cell]
                picks = bool(result["built"]) and (-result["total"], result["candidate_id"]) < _rank_key(w)
                row = {"cell": cell, "task_id": task, "split": "holdout", "target": target, "aspect_class": cls,
                       "plan": plan.name, **result, "winner_id": w["candidate_id"], "winner_total": score(w["metrics"]),
                       "engine_picks_new": picks}
                _append(out, [row])
                added.append(row)
    return added


def evaluate_pairs(builds: list[dict], winners: dict, pool_dir: Path = POOL_DIR, fp_dir: Path = FP_DIR):
    """(current winner, new plan's best candidate) for every built holdout row."""
    return judge_pairs_for(builds, winners, pool_dir, fp_dir)


def evaluate_summary(builds: list[dict], judgments: list[dict]) -> dict:
    """Per new plan: holdout cells tried, built, judged, wins (same rule as cluster), and how
    often the engine's own grader would pick it (and win with it)."""
    ids = {(b["cell"], b["candidate_id"]): (b["cell"], b["plan"]) for b in builds if b.get("built")}
    outcomes = _outcomes(judgments, ids)
    summary = {}
    for b in builds:
        s = summary.setdefault(b["plan"], {"cells": 0, "built": 0, "judged": 0, "wins": 0, "win_rate": None,
                                           "engine_picks_new": 0, "engine_picks_new_and_wins": 0})
        s["cells"] += 1
        s["built"] += bool(b.get("built"))
        verdicts = outcomes.get((b["cell"], b["plan"]))
        if verdicts:
            s["judged"] += 1
            s["wins"] += _won(verdicts)
        s["engine_picks_new"] += bool(b.get("engine_picks_new"))
        s["engine_picks_new_and_wins"] += bool(b.get("engine_picks_new")) and _won(verdicts or [])
    for s in summary.values():
        s["win_rate"] = s["wins"] / s["judged"] if s["judged"] else None
    return summary


# ---------------------------------------------------------------- CLI

def _pool_rows() -> list[dict]:
    return read_jsonl(POOL_DIR / "pool.jsonl", shards=False)


def _designs() -> dict:
    from dev.distill.corpus import load_corpus

    return {d["task_id"]: d for d in load_corpus()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["propose", "build", "judge", "cluster", "evaluate"])
    ap.add_argument("--cells", type=int, default=160, help="propose: train cells to sample")
    ap.add_argument("--classes", default=None,
                    help="propose: comma-separated aspect classes to sample from (default: all six)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-calls", type=int, default=None, help="propose/judge: stop after this many teacher calls")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--swap-share", type=float, default=0.2)
    ap.add_argument("--pairs-per-call", type=int, default=4)
    ap.add_argument("--print-prompt", action="store_true", help="propose: print the first cell's prompt and stop")
    ap.add_argument("--candidate", type=Path, default=CANDIDATE_FILE, help="evaluate: candidate extra_templates file")
    ap.add_argument("--stage", choices=["build", "judge", "summary"], default="build", help="evaluate: which stage")
    ap.add_argument("--shard", default=None, help="K/N: only shard K's cells, writing <file>.shardKofN.jsonl")
    ap.add_argument("--merge", action="store_true", help="fold this step's shard files into its main file")
    args = ap.parse_args(argv)
    os.environ["CHAMELEON_LLM"] = "heuristic"  # node 1 never calls a model here
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None

    main_file = {"propose": PROPOSALS_FILE, "build": BUILDS_FILE, "judge": JUDGMENTS_FILE, "cluster": None,
                 "evaluate": {"build": EVAL_BUILDS_FILE, "judge": EVAL_JUDGMENTS_FILE, "summary": None}[args.stage]
                 }[args.step]
    if args.merge:
        if main_file is None:
            ap.error("--merge needs a step that writes a .jsonl file")
        print(f"merged {merge_shards(main_file)} new rows into {main_file}")
        return 0
    out = shard_file(main_file, *shard) if shard and main_file else main_file
    done_files = [main_file] if shard and main_file else None

    if args.step == "cluster":
        report = cluster(read_jsonl(PROPOSALS_FILE), read_jsonl(BUILDS_FILE), read_jsonl(JUDGMENTS_FILE))
        for cls, e in report["classes"].items():
            print(f"{cls}: {e['n_proposals']} proposed, {e['n_built']} built, {e['n_judged']} judged, "
                  f"{e['n_wins']} wins -> k={e['k']} " + ", ".join(f"{m['name']} ({m['members']})" for m in e["medoids"]))
        print(f"wrote {CLUSTERS_FILE} and {CANDIDATE_FILE}")
        return 0

    if args.step == "judge" or (args.step == "evaluate" and args.stage == "judge"):
        from dev.distill.teacher import ClaudeCodeTeacher

        winners = cell_winners(_pool_rows())
        if args.step == "judge":
            pairs = judge_pairs_for(read_jsonl(BUILDS_FILE), winners)
        else:
            pairs = evaluate_pairs(read_jsonl(EVAL_BUILDS_FILE), winners)
        added = judge(pairs, ClaudeCodeTeacher(args.model), POOL_DIR, out, args.swap_share, args.pairs_per_call,
                      args.seed, args.limit_calls, shard=shard, done_files=done_files)
        print(f"judged {len(added)} new rows over {len(pairs)} pairs -> {out}")
        return 0

    if args.step == "evaluate" and args.stage == "summary":
        summary = evaluate_summary(read_jsonl(EVAL_BUILDS_FILE), read_jsonl(EVAL_JUDGMENTS_FILE))
        EVAL_SUMMARY_FILE.write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        return 0

    from dev.distill.assets import merged_assets_dir

    targets, designs = load_targets(), _designs()
    with merged_assets_dir() as assets:
        if args.step == "propose":
            from dev.distill.teacher import ClaudeCodeTeacher

            rows = _pool_rows()
            winners = cell_winners(rows)
            classes = args.classes.split(",") if args.classes else None
            cells = pick_propose_cells(rows, args.cells, args.seed, targets, classes)
            per_class = {}
            for c in cells:
                per_class[aspect_of(c.split("@")[1], targets)] = per_class.get(aspect_of(c.split("@")[1], targets), 0) + 1
            print(f"{len(cells)} train cells: " + ", ".join(f"{k} {v}" for k, v in sorted(per_class.items())))
            cells = in_shard(cells, shard) if shard else cells
            cache: dict = {}

            def context(cell):
                return cell_context(cell, winners, designs, targets, assets, POOL_DIR, cache)

            if args.print_prompt:
                print(propose_prompt(context(cells[0])))
                return 0
            added = propose(cells, context, ClaudeCodeTeacher(args.model), out, args.limit_calls, done_files)
            print(f"proposed {len(added)} new plans over {len(cells)} cells -> {out}")
        elif args.step == "build":
            proposals = [p for p in read_jsonl(PROPOSALS_FILE) if p.get("rubric_version") == RUBRIC_VERSION]
            proposals = in_shard(proposals, shard) if shard else proposals
            added = build(proposals, designs, targets, assets, out, FP_DIR, done_files)
            print(f"built {sum(r['built'] for r in added)} of {len(added)} new proposals -> {out}")
        else:  # evaluate build
            added = evaluate_build(args.candidate, SHIPPED_FILE, _pool_rows(), designs, targets, assets, out, FP_DIR,
                                   shard=shard, done_files=done_files)
            print(f"built {sum(r['built'] for r in added)} of {len(added)} holdout (cell, plan) rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
