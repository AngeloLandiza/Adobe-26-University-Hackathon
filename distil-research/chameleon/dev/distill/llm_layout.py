"""Ceiling test: can a strong LLM lay out the resized ad itself?

The teacher (Claude Opus) sees the source ad (JSON + render) and the target
canvas and proposes every element's box (and font size for text). The engine
then executes the plan: text is fitted into its box with the official word
wrap (largest size <= the proposed one that fits, never below the role floor),
images keep their source proportions inside their box, and the hard gates are
checked. A plan that fails a gate falls back to the given baseline variant's
design, as a real engine would. The result is written as an A/B variant, so
`ab_engine judge --a <baseline> --b <name>` compares it.

    python -m dev.distill.llm_layout run --name llm_hold --baseline v5 --cells 20
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from dev.distill import ab_engine as ab
from dev.distill.paths import POOL_DIR, RENDERS_SUBDIR

SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"},
                           "width": {"type": "number"}, "height": {"type": "number"},
                           "font_size": {"type": "number"}, "z_index": {"type": "integer"}},
            "required": ["id", "x", "y", "width", "height"]}},
        "notes": {"type": "string"},
    },
    "required": ["elements"],
}

KEEP = ("id", "type", "x", "y", "width", "height", "z_index", "content", "font_family", "font_size", "font_weight",
        "text_align", "shape_type", "asset_id", "fit", "group_id", "rotation")

PROMPT = """You are a senior visual designer resizing a finished ad to a new canvas. Produce the new layout.

Goal, in the organisers' words: a new version that fits the new shape and still looks like a designer made it: nothing overlapping that shouldn't, text readable, the brand intact, everything arranged with intent. Rearrange the design rather than simply shrinking or stretching it, and keep it unmistakably the same design.
Judged on: no unwanted collisions (intentional layering like text on its button or a card is fine; text never on a busy image); everything readable (large enough, nothing cut off); on brand (logo, colours, fonts intact and in proportion); intentional (alignment, even spacing, the most important element stands out, pictures keep their proportions).

Source canvas: {sw} x {sh}. Target canvas: {tw} x {th}.
Source elements (JSON, coordinates in source pixels; z_index = stacking order):
{elements}

Rules:
- Return every element id exactly once with its new box (x, y, width, height) in TARGET pixels. Never drop an element.
- Text: also give font_size in target pixels. The box must be wide and tall enough for the text at that size (wrapping happens only at spaces; line height 1.2). Text inside a button stays on one line.
- Images keep their source aspect ratio (the box you give will be shrunk to the source ratio, centred).
- Full-canvas backgrounds cover the whole target canvas exactly (x=0, y=0, width and height = the canvas).
- Every element must lie fully inside the target canvas; nothing may bleed off an edge (unlike some decorations in the source).
- Keep a safe margin of about 4-6% of the shorter side for content near the edges.
- You may change z_index only if needed.
The source ad image is attached (read it first)."""


def _source_png(cell: dict) -> Path:
    return Path(POOL_DIR) / RENDERS_SUBDIR / cell["task_id"] / "source.png"


def pick_cells(baseline: str, n: int, seed: int = 0) -> list[dict]:
    """n holdout cells spread over targets and ads (round-robin by target)."""
    rows = ab.load_winners(baseline)
    by_t = defaultdict(list)
    for r in sorted(rows.values(), key=lambda r: r["cell"]):
        by_t[r["target"]].append(r)
    import random
    rng = random.Random(seed)
    for v in by_t.values():
        rng.shuffle(v)
    out, used = [], set()
    while len(out) < n and any(by_t.values()):
        for t in sorted(by_t):
            while by_t[t] and by_t[t][0]["task_id"] in used:
                by_t[t].pop(0)
            if by_t[t] and len(out) < n:
                r = by_t[t].pop(0); out.append(r); used.add(r["task_id"])
    return out


def execute(source: dict, target: dict, plan, measurer, answer: dict) -> dict:
    """Turn the teacher's boxes into a design the renderer can draw."""
    from engine.pipeline.layout.place import floor_px
    roles = {eid: u.role for u in plan.ranked_units() for eid in u.element_ids}
    proposed = {e["id"]: e for e in answer["elements"]}
    W, H = target["width"], target["height"]
    out = []
    for el in source["elements"]:
        e = dict(el)
        p = proposed.get(el["id"])
        if p is None:
            raise ValueError(f"missing element {el['id']}")
        x, y, w, h = float(p["x"]), float(p["y"]), max(1.0, float(p["width"])), max(1.0, float(p["height"]))
        if el["type"] == "image" and el.get("fit") != "cover" and el.get("width") and el.get("height"):  # cover crops, so any box keeps proportions
            ar = el["width"] / el["height"]
            if w / h > ar:
                nw = h * ar; x += (w - nw) / 2; w = nw
            else:
                nh = w / ar; y += (h - nh) / 2; h = nh
        if w > W or h > H:  # too big for the canvas: shrink uniformly about its centre
            k = min(W / w, H / h); cx, cy = x + w / 2, y + h / 2; w, h = w * k, h * k; x, y = cx - w / 2, cy - h / 2
        x, y = min(max(0.0, x), W - w), min(max(0.0, y), H - h)  # then shift inside
        e.update(x=x, y=y, width=w, height=h)
        if "z_index" in p:
            e["z_index"] = int(p["z_index"])
        if el["type"] == "text":
            floor = floor_px(roles.get(el["id"], "body"), W, H)
            want = float(p.get("font_size") or el["font_size"])
            fs = measurer.fit_font_size(e, w * 0.985, h, min(floor, want), max(want, min(floor, want)))
            e["font_size"] = fs if fs is not None else min(floor, want)
            e["height"] = max(h, measurer.block_height(e, e["font_size"], w))
        if el["type"] == "shape" and "corner_radius" in el and el.get("height"):
            e["corner_radius"] = el["corner_radius"] * h / el["height"]
        out.append(e)
    return {"canvas": {"width": W, "height": H}, "elements": out}


def run(name: str, baseline: str, n: int, model: str, image: bool = True) -> None:
    os.environ["CHAMELEON_LLM"] = "heuristic"
    os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
    from dev.distill.corpus import load_corpus
    from dev.distill.teacher import ClaudeCodeTeacher
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.priority import plan as priority_plan
    from engine.pipeline.render import render_to_image
    from engine.resize import _get_measurer

    designs = {d["task_id"]: d for d in load_corpus(None)}
    targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
    base_dir = ab.AB_ROOT / baseline
    out_dir = ab.AB_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "variant.json").write_text(json.dumps({"name": name, "env": {"LLM_LAYOUT": model, "fallback": baseline, "image": image}}, indent=2))
    teacher = ClaudeCodeTeacher(model)
    measurer = _get_measurer()
    rows, stats = [], defaultdict(int)
    for r in pick_cells(baseline, n):
        source = designs[r["task_id"]]["source"]
        t = targets[r["target"]]
        target = {"width": t["width"], "height": t["height"]}
        plan = priority_plan(source)
        els = [{k: e[k] for k in KEEP if k in e} for e in source["elements"]]
        prompt = (PROMPT if image else PROMPT.replace("\nThe source ad image is attached (read it first).", "")).format(sw=source["canvas"]["width"], sh=source["canvas"]["height"], tw=t["width"], th=t["height"],
                               elements=json.dumps(els, separators=(",", ":")))
        png = Path(r["png"])
        try:
            answer = teacher.ask(prompt, SCHEMA, images=[_source_png(r)] if image else [])
            design = execute(source, target, plan, measurer, answer)
            failed = check_gates(source, design, target, plan, measurer)
        except Exception as exc:  # a bad plan falls back like a failed gate
            design, failed = None, [f"error:{type(exc).__name__}"]
        if failed:
            stats["fallback"] += 1
            stats.update({f: stats[f] + 1 for f in failed})
            row = {**r, "variant": name, "candidate_id": "fallback-" + baseline, "failed_llm_gates": failed}
            src_png = base_dir / r["png"]
            (out_dir / png).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / png).write_bytes(src_png.read_bytes())
        else:
            stats["llm"] += 1
            (out_dir / png).parent.mkdir(parents=True, exist_ok=True)
            render_to_image(design, assets_dir=ab._assets(), max_size=ab.RENDER_SIZE).save(out_dir / png)
            row = {**r, "variant": name, "candidate_id": "llm", "is_fallback": False, "design_hash": ab.design_hash(design)}
            (out_dir / "designs.nosync").mkdir(exist_ok=True)
            (out_dir / "designs.nosync" / f"{r['cell'].replace('@', '__')}.json").write_text(json.dumps(design))
        rows.append(row)
        print(f"[llm] {r['cell']}: {'gates fail ' + ','.join(failed) if failed else 'ok'}", flush=True)
    (out_dir / "winners.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    cost = sum(c["cost_usd"] for c in teacher.calls)
    print(json.dumps({"cells": len(rows), **stats, "teacher_cost_usd": round(cost, 2)}))


def _parse_json(text: str) -> dict:
    """The model's answer: the first top-level JSON object in its output."""
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return json.loads(text[start:i + 1])
    raise ValueError("no complete JSON object")


def ingest(outputs: Path, name: str, baseline: str, repair: bool = False) -> None:
    """Student layouts from Colab (eval_outputs.jsonl) -> an A/B variant, like `run` but without a teacher."""
    os.environ["CHAMELEON_LLM"] = "heuristic"
    os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
    from dev.distill.corpus import load_corpus
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.priority import plan as priority_plan
    from engine.pipeline.render import render_to_image
    from engine.resize import _get_measurer

    designs = {d["task_id"]: d for d in load_corpus(None)}
    targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
    base = ab.load_winners(baseline)
    out_dir = ab.AB_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "variant.json").write_text(json.dumps({"name": name, "env": {"LLM_LAYOUT": str(outputs), "fallback": baseline}}, indent=2))
    measurer = _get_measurer()
    rows, stats = [], defaultdict(int)
    for o in map(json.loads, outputs.open()):
        r = base[o["cell"]]
        source = designs[r["task_id"]]["source"]
        t = targets[r["target"]]
        target = {"width": t["width"], "height": t["height"]}
        plan = priority_plan(source)
        png = Path(r["png"])
        (out_dir / png).parent.mkdir(parents=True, exist_ok=True)
        try:
            design = execute(source, target, plan, measurer, _parse_json(o["output"]))
            if repair:  # the engine's geometry passes on the LLM's plan: balance (fill, groups, margins) + text slack
                from engine.pipeline.layout import balance, text_slack
                from engine.pipeline.types import Candidate
                cand = Candidate("student", design, "student", {})
                cand = balance.balance_all(source, target, plan, measurer, [cand])[0]
                design = text_slack.apply(source, target, plan, measurer, cand).design
            failed = check_gates(source, design, target, plan, measurer)
        except Exception as exc:
            failed = [f"error:{type(exc).__name__}"]
        if failed:
            stats["fallback"] += 1
            stats.update({f: stats[f] + 1 for f in failed})
            (out_dir / png).write_bytes((ab.AB_ROOT / baseline / r["png"]).read_bytes())
            rows.append({**r, "variant": name, "candidate_id": "fallback-" + baseline, "failed_llm_gates": failed})
        else:
            stats["student"] += 1
            render_to_image(design, assets_dir=ab._assets(), max_size=ab.RENDER_SIZE).save(out_dir / png)
            rows.append({**r, "variant": name, "candidate_id": "student", "is_fallback": False, "design_hash": ab.design_hash(design)})
        print(f"[ingest] {o['cell']}: {'fallback ' + ','.join(failed) if failed else 'ok'}", flush=True)
    (out_dir / "winners.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps({"cells": len(rows), **stats}))


# ---------------------------------------------------------------- compact, all-sizes-per-call generation

COMPACT_OUT = ('Answer format (integers only, no spaces needed): {"e": [[id, x, y, width, height, font_size], ...]} '
               "with font_size only for text elements, e.g. [\"headline\",90,40,900,150,64] or [\"logo\",40,40,200,60].")


def student_prompt_v2(source: dict, target: dict) -> str:
    """Single-target prompt with the compact answer format (what the student is trained and run on)."""
    return student_prompt(source, target).replace("- You may change z_index only if needed.\n", "") + "\n" + COMPACT_OUT


def _compact_to_elements(rows: list) -> dict:
    out = []
    for r in rows:
        e = {"id": str(r[0]), "x": r[1], "y": r[2], "width": r[3], "height": r[4]}
        if len(r) > 5 and r[5] is not None:
            e["font_size"] = r[5]
        out.append(e)
    return {"elements": out}


GEN_SCHEMA = {"type": "object", "properties": {"layouts": {"type": "array", "items": {
    "type": "object", "properties": {"t": {"type": "string"}, "e": {"type": "array", "items": {"type": "array"}}},
    "required": ["t", "e"]}}}, "required": ["layouts"]}


def _gen_prompt(source: dict, tl: list[dict]) -> str:
    base = student_prompt(source, tl[0]).replace("- You may change z_index only if needed.\n", "")
    sizes = "; ".join(f'{t["name"]}={t["width"]}x{t["height"]}' for t in tl)
    base = base.replace(f'Target canvas: {tl[0]["width"]} x {tl[0]["height"]}.',
                        f"Target canvases (name=width x height): {sizes}. Design EACH one for its own shape.")
    return base + ('\nReturn {"layouts": [{"t": <target name>, "e": [[id, x, y, width, height, font_size], ...]}, ...]}, '
                   "one entry per target, integers only, font_size only for text.")


def gen(split: str, out: Path, model: str, effort: str, workers: int, ads: int | None, budget: float,
        baseline: str | None, name: str | None) -> None:
    """All 13 target sizes per call, compact answers, lean low-effort teacher, a hard budget cap.
    Writes student examples (gate-passing only) to out/gen_<split>.jsonl; with --baseline, also an A/B
    variant (gate failures fall back to the baseline) for the cells the baseline has."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    os.environ["CHAMELEON_LLM"] = "heuristic"
    os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
    from dev.distill.corpus import load_corpus
    from dev.distill.teacher import ClaudeCodeTeacher
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.priority import plan as priority_plan
    from engine.pipeline.render import render_to_image
    from engine.resize import _get_measurer

    designs = sorted(load_corpus(split), key=lambda d: d["task_id"])[:ads] if ads else sorted(load_corpus(split), key=lambda d: d["task_id"])
    targets = json.loads(ab.TARGETS_PATH.read_text())
    out.mkdir(parents=True, exist_ok=True)
    keep_f = out / f"gen_{split}.jsonl"
    done = {json.loads(l)["task_id"] for l in keep_f.open()} if keep_f.exists() else set()
    teacher = ClaudeCodeTeacher(model, effort=effort)
    measurer = _get_measurer()
    lock = threading.Lock()
    spent = [0.0]
    base = ab.load_winners(baseline) if baseline else {}
    vrows = []

    def one(d):
        tid = d["task_id"]
        with lock:
            if spent[0] >= budget or tid in done:
                return []
        source = d["source"]
        before = len(teacher.calls)
        try:
            ans = teacher.ask(_gen_prompt(source, targets), GEN_SCHEMA)
        except Exception as exc:
            return [{"task_id": tid, "ok": False, "why": f"error:{type(exc).__name__}"}]
        with lock:
            spent[0] += sum(c["cost_usd"] for c in teacher.calls[before:])
        got = {l.get("t"): l for l in ans.get("layouts", [])}
        plan = priority_plan(source)
        rows = []
        for t in targets:
            lay = got.get(t["name"])
            tgt = {"width": t["width"], "height": t["height"]}
            cell = f"{tid}@{t['name']}"
            if lay is None:
                rows.append({"cell": cell, "task_id": tid, "ok": False, "why": "missing"}); continue
            try:
                design = execute(source, tgt, plan, measurer, _compact_to_elements(lay["e"]))
                failed = check_gates(source, design, tgt, plan, measurer)
            except Exception as exc:
                design, failed = None, [f"error:{type(exc).__name__}"]
            rows.append({"cell": cell, "task_id": tid, "target": t["name"], "ok": not failed, "why": ",".join(failed),
                         "prompt": student_prompt_v2(source, tgt),
                         "answer": json.dumps({"e": lay["e"]}, separators=(",", ":")), "_design": design})
        return rows

    stats = defaultdict(int)
    with keep_f.open("a") as kf, (out / f"gen_{split}_rejected.jsonl").open("a") as rf, ThreadPoolExecutor(workers) as ex:
        for rows in ex.map(one, designs):
            for r in rows:
                design = r.pop("_design", None)
                stats["ok" if r["ok"] else "rejected"] += 1
                (kf if r["ok"] else rf).write(json.dumps(r) + "\n")
                if baseline and r.get("cell") in base:
                    vrows.append((r, design))
            kf.flush(); rf.flush()
            print(f"[gen] {dict(stats)} spent ${spent[0]:.2f}", flush=True)
    if baseline:
        out_dir = ab.AB_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "variant.json").write_text(json.dumps({"name": name, "env": {"LLM_GEN": model, "effort": effort, "fallback": baseline}}, indent=2))
        rows = []
        for r, design in vrows:
            b = base[r["cell"]]; png = Path(b["png"])
            (out_dir / png).parent.mkdir(parents=True, exist_ok=True)
            if r["ok"]:
                render_to_image(design, assets_dir=ab._assets(), max_size=ab.RENDER_SIZE).save(out_dir / png)
                rows.append({**b, "variant": name, "candidate_id": "llm", "is_fallback": False, "design_hash": ab.design_hash(design)})
            else:
                (out_dir / png).write_bytes((ab.AB_ROOT / baseline / b["png"]).read_bytes())
                rows.append({**b, "variant": name, "candidate_id": "fallback-" + baseline})
        (out_dir / "winners.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        print(f"variant {name}: {len(rows)} cells")
    print(json.dumps({"ads": len(designs), **stats, "spent_usd": round(spent[0], 2),
                      "usd_per_layout": round(spent[0] / max(1, stats["ok"] + stats["rejected"]), 4)}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--baseline", default="v5")
    r.add_argument("--cells", type=int, default=20)
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--no-image", action="store_true", help="JSON only: do not show the source render")
    d = sub.add_parser("datagen")
    d.add_argument("--split", default="train")
    d.add_argument("--out", default=str(ab.DATA_ROOT / "llm-distill"))
    d.add_argument("--model", default="claude-sonnet-5-5")
    d.add_argument("--per-call", type=int, default=3)
    d.add_argument("--workers", type=int, default=6)
    d.add_argument("--limit", type=int, default=None)
    e = sub.add_parser("eval-prompts")
    e.add_argument("--baseline", default="v5")
    e.add_argument("--cells", type=int, default=30)
    e.add_argument("--out", default=str(ab.DATA_ROOT / "llm-distill" / "eval_prompts.jsonl"))
    g = sub.add_parser("ingest")
    g.add_argument("--outputs", required=True)
    g.add_argument("--name", default="llm_student")
    g.add_argument("--baseline", default="v5")
    g.add_argument("--repair", action="store_true", help="run the engine's balance + text-slack passes on each plan")
    q = sub.add_parser("gen", help="all sizes per call, compact, lean low-effort teacher, budget cap")
    q.add_argument("--split", default="train")
    q.add_argument("--out", default=str(ab.DATA_ROOT / "llm-distill"))
    q.add_argument("--model", default="claude-sonnet-5-5")
    q.add_argument("--effort", default="low")
    q.add_argument("--workers", type=int, default=6)
    q.add_argument("--ads", type=int, default=None)
    q.add_argument("--budget", type=float, default=5.0, help="stop starting new calls after this many USD-equivalent")
    q.add_argument("--baseline", default=None, help="also write an A/B variant vs this baseline (for a quality check)")
    q.add_argument("--name", default=None)
    a = p.parse_args()
    if a.cmd == "gen":
        gen(a.split, Path(a.out), a.model, a.effort, a.workers, a.ads, a.budget, a.baseline, a.name)
    elif a.cmd == "ingest":
        ingest(Path(a.outputs), a.name, a.baseline, repair=a.repair)
    elif a.cmd == "run":
        run(a.name, a.baseline, a.cells, a.model, image=not a.no_image)
    elif a.cmd == "datagen":
        datagen(a.split, Path(a.out), a.model, a.per_call, a.workers, a.limit)
    else:
        eval_prompts(a.baseline, a.cells, Path(a.out))



# ---------------------------------------------------------------- distillation data

BATCH_SCHEMA = {
    "type": "object",
    "properties": {"layouts": {"type": "array", "items": {
        "type": "object",
        "properties": {"target": {"type": "string"}, "elements": SCHEMA["properties"]["elements"]},
        "required": ["target", "elements"]}}},
    "required": ["layouts"],
}

STUDENT_SYSTEM = "You are an ad layout engine. Given a source ad as JSON and a target canvas, return the resized layout as JSON."


def student_prompt(source: dict, target: dict) -> str:
    """The single-target, JSON-only prompt the small model is trained and run on."""
    els = [{k: e[k] for k in KEEP if k in e} for e in source["elements"]]
    return PROMPT.replace("\nThe source ad image is attached (read it first).", "").format(
        sw=source["canvas"]["width"], sh=source["canvas"]["height"], tw=target["width"], th=target["height"],
        elements=json.dumps(els, separators=(",", ":")))


def _batch_prompt(source: dict, targets: list[dict]) -> str:
    base = student_prompt(source, targets[0])
    sizes = "; ".join(f'"{t["name"]}": {t["width"]} x {t["height"]}' for t in targets)
    return base.replace(f'Target canvas: {targets[0]["width"]} x {targets[0]["height"]}.',
                        f"Produce one layout for EACH of these target canvases (name: width x height): {sizes}. "
                        "Design each one for its own shape.") + '\nReturn {"layouts": [{"target": <name>, "elements": [...]}, ...]}.'


def datagen(split: str, out: Path, model: str, per_call: int, workers: int, limit: int | None) -> None:
    """Teacher layouts for every pool cell of `split`, a few targets per call; keep only plans that pass the gates."""
    from concurrent.futures import ThreadPoolExecutor
    os.environ["CHAMELEON_LLM"] = "heuristic"
    os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
    from dev.distill.corpus import load_corpus
    from dev.distill.teacher import ClaudeCodeTeacher
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.priority import plan as priority_plan
    from engine.resize import _get_measurer

    designs = {d["task_id"]: d for d in load_corpus(split)}
    targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
    by_task = defaultdict(list)
    for c in ab.pool_cells(POOL_DIR, split):
        by_task[c["task_id"]].append(c["target"])
    jobs = [(tid, tl[i:i + per_call]) for tid, tl in sorted(by_task.items()) if tid in designs
            for i in range(0, len(tl), per_call)]
    jobs = jobs[:limit] if limit else jobs
    out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["cell"] for f in out.glob("*.jsonl") for l in f.open()}
    teacher = ClaudeCodeTeacher(model)
    measurer = _get_measurer()

    def one(job):
        tid, names = job
        if all(f"{tid}@{n}" in done for n in names):
            return []
        source = designs[tid]["source"]
        tl = [{"name": n, **targets[n]} for n in names]
        try:
            ans = teacher.ask(_batch_prompt(source, tl), BATCH_SCHEMA)
        except Exception as exc:
            return [{"cell": f"{tid}@{n}", "ok": False, "why": f"error:{type(exc).__name__}"} for n in names]
        got = {l.get("target"): l for l in ans.get("layouts", [])}
        plan = priority_plan(source)
        rows = []
        for t in tl:
            lay = got.get(t["name"])
            cell = f"{tid}@{t['name']}"
            if lay is None:
                rows.append({"cell": cell, "ok": False, "why": "missing"}); continue
            tgt = {"width": t["width"], "height": t["height"]}
            try:
                failed = check_gates(source, execute(source, tgt, plan, measurer, lay), tgt, plan, measurer)
            except Exception as exc:
                failed = [f"error:{type(exc).__name__}"]
            rows.append({"cell": cell, "ok": not failed, "why": ",".join(failed), "task_id": tid, "target": t["name"],
                         "prompt": student_prompt(source, tgt),
                         "answer": json.dumps({"elements": lay["elements"]}, separators=(",", ":"))})
        return rows

    keep = (out / "train.jsonl").open("a"); rej = (out / "rejected.jsonl").open("a")
    stats = defaultdict(int)
    with ThreadPoolExecutor(workers) as ex:
        for rows in ex.map(one, jobs):
            for r in rows:
                stats["ok" if r["ok"] else "rejected"] += 1
                (keep if r["ok"] else rej).write(json.dumps(r) + "\n")
            keep.flush(); rej.flush()
            print(f"[datagen] {dict(stats)}", flush=True)
    cost = sum(c["cost_usd"] for c in teacher.calls)
    print(json.dumps({"jobs": len(jobs), **stats, "teacher_cost_usd": round(cost, 2)}))


def eval_prompts(baseline: str, n: int, out: Path) -> None:
    """Student prompts for n holdout cells (the same picker as the ceiling test), for Colab inference."""
    from dev.distill.corpus import load_corpus
    designs = {d["task_id"]: d for d in load_corpus(None)}
    targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in pick_cells(baseline, n):
            t = targets[r["target"]]
            f.write(json.dumps({"cell": r["cell"], "prompt": student_prompt(designs[r["task_id"]]["source"],
                                                                            {"width": t["width"], "height": t["height"]})}) + "\n")
    print(f"wrote {n} prompts to {out}")


if __name__ == "__main__":
    main()
