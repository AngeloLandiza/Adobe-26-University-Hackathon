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


def run(name: str, baseline: str, n: int, model: str) -> None:
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
    (out_dir / "variant.json").write_text(json.dumps({"name": name, "env": {"LLM_LAYOUT": model, "fallback": baseline}}, indent=2))
    teacher = ClaudeCodeTeacher(model)
    measurer = _get_measurer()
    rows, stats = [], defaultdict(int)
    for r in pick_cells(baseline, n):
        source = designs[r["task_id"]]["source"]
        t = targets[r["target"]]
        target = {"width": t["width"], "height": t["height"]}
        plan = priority_plan(source)
        els = [{k: e[k] for k in KEEP if k in e} for e in source["elements"]]
        prompt = PROMPT.format(sw=source["canvas"]["width"], sh=source["canvas"]["height"], tw=t["width"], th=t["height"],
                               elements=json.dumps(els, separators=(",", ":")))
        png = Path(r["png"])
        try:
            answer = teacher.ask(prompt, SCHEMA, images=[_source_png(r)])
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--baseline", default="v5")
    r.add_argument("--cells", type=int, default=20)
    r.add_argument("--model", default="claude-opus-5-5")
    a = p.parse_args()
    run(a.name, a.baseline, a.cells, a.model)


if __name__ == "__main__":
    main()
