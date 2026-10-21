"""Step 1b (project python): holdout layouts of ab/<name> (heuristic planner, no teacher) -> archetypes.nosync/engine_<name>.json."""
import json
import os
import sys
from pathlib import Path

from dev.distill import ab_engine as ab
from dev.distill.corpus import load_corpus
from dev.references.engine_stats import layout_of


def main(name):
    env = json.loads((ab.AB_ROOT / name / "variant.json").read_text())["env"]
    os.environ.update(env)
    os.environ["CHAMELEON_LLM"] = "heuristic"
    os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
    designs = {d["task_id"]: d for d in load_corpus("holdout")}
    targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
    out = []
    for cell, r in sorted(ab.load_winners(name).items()):
        if r["split"] != "holdout" or r["task_id"] not in designs:
            continue
        t = targets[r["target"]]
        design, plan, best, _ = ab._resize(designs[r["task_id"]]["source"], {"width": t["width"], "height": t["height"]})
        lay = layout_of(design, plan)
        lay.update(cell=cell, target=r["target"])
        out.append(lay)
    d = Path(ab.DATA_ROOT) / "references" / "archetypes.nosync"
    d.mkdir(exist_ok=True)
    (d / f"engine_{name}.json").write_text(json.dumps(out))
    print(name, len(out))


if __name__ == "__main__":
    main(sys.argv[1])
