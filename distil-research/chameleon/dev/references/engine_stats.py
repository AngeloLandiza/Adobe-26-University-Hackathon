"""Same layout statistics for today's engine ("base") and bisection, on the holdout cells of training-data/ab/<name>/winners.jsonl.

    python -m dev.references.engine_stats base|bisect      (project python, run from chameleon/)
Designs are regenerated (resize_with_report, heuristic planner, no teacher) and roles come from the engine's own plan:
headline->headline, hero_image->product, cta->cta, logo->logo, subheadline/other_text/legal/badge->body,
secondary_image/decoration->decoration, background->background. Output: training-data/references/engine_layout_stats_<name>.json
"""
import json
import os
import sys
from pathlib import Path

from dev.distill import ab_engine as ab
from dev.distill.corpus import load_corpus
from dev.references.layout_stats import aggregate

MAP = {"headline": "headline", "hero_image": "product", "cta": "cta", "logo": "logo", "subheadline": "body",
       "other_text": "body", "legal": "body", "badge": "body", "secondary_image": "decoration",
       "decoration": "decoration", "background": "background"}


def layout_of(design, plan):
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    role_of = {}
    for u in plan.units:
        for e in u.element_ids:
            role_of[e] = MAP.get(u.role or "decoration", "decoration")
    items = []
    for e in design["elements"]:
        if e.get("opacity", 1) is not None and e.get("opacity", 1) < 0.05:
            continue
        it = {"role": role_of.get(e["id"], "decoration"), "x0": e["x"], "y0": e["y"],
              "x1": e["x"] + e["width"], "y1": e["y"] + e["height"]}
        if e["type"] == "text":
            it["font"] = e.get("font_size")
        items.append(it)
    # a container_label unit (button + label): the CTA box is the union of its elements
    return {"W": W, "H": H, "items": items}


def main(name):
    env = json.loads((ab.AB_ROOT / name / "variant.json").read_text())["env"]
    os.environ.update(env)
    os.environ["CHAMELEON_LLM"] = "heuristic"
    os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
    designs = {d["task_id"]: d for d in load_corpus("holdout")}
    targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
    rows = ab.load_winners(name)
    by = {}
    for cell, r in sorted(rows.items()):
        if r["split"] != "holdout" or r["task_id"] not in designs:
            continue
        t = targets[r["target"]]
        design, plan, best, _ = ab._resize(designs[r["task_id"]]["source"], {"width": t["width"], "height": t["height"]})
        by.setdefault(r["target"], []).append(layout_of(design, plan))
    out = {"variant": name, "env": env, "targets": {k: aggregate(v) for k, v in by.items()}}
    p = Path(ab.DATA_ROOT) / "references" / f"engine_layout_stats_{name}.json"
    p.write_text(json.dumps(out, indent=1))
    print(name, {k: v["n_layouts"] for k, v in out["targets"].items()})


if __name__ == "__main__":
    main(sys.argv[1])
