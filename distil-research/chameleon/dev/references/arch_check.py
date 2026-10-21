"""Step 5 (project python, no teacher): build archetype floor-plan candidates on holdout cells, report gate pass rate, render a contact sheet.
    python -m dev.references.arch_check [cells_per_class=20] [sheet_rows_per_class=4]
Every plan of a class is tried on that class's cells (what the engine would do with CHAMELEON_EXTRA_TEMPLATES set).
Output: archetypes.nosync/check.json, renders/, sheet.png
"""
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from dev.distill import ab_engine as ab
from dev.distill import floorplans as fp
from dev.distill.corpus import load_corpus
from engine.pipeline.layout.aspect import aspect_class
from engine.pipeline.layout.extra_templates import load_floor_plans

REF = Path(ab.DATA_ROOT) / "references"
OUT = REF / "archetypes.nosync"


def main(per_class=20, rows_per_class=4):
    os.environ["CHAMELEON_EXTRA_TEMPLATES"] = str(REF / "archetype_floorplans.json")
    os.environ.update(json.loads((ab.AB_ROOT / "base" / "variant.json").read_text())["env"])
    os.environ["CHAMELEON_LLM"] = "heuristic"
    assets = ab._assets()
    os.environ["CHAMELEON_ASSETS_DIR"] = str(assets)
    from engine.pipeline.render import render_to_image
    from engine.pipeline.textmeasure import TextMeasurer
    plans = load_floor_plans()
    src = json.loads((OUT / "floorplan_sources.json").read_text())
    designs = {d["task_id"]: d for d in load_corpus("holdout")}
    targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
    base, bis = ab.load_winners("base"), ab.load_winners("bisect")
    by_class = {}
    for cell, r in sorted(base.items()):
        by_class.setdefault(aspect_class(targets[r["target"]]["width"], targets[r["target"]]["height"]), []).append(cell)
    rng = random.Random(0)
    measurer, cache, rows = TextMeasurer(), {}, []
    (OUT / "renders").mkdir(exist_ok=True)
    chosen = {}
    for cls, cells in by_class.items():
        rng.shuffle(cells)
        chosen[cls] = cells[:per_class]
    for cls, cells in chosen.items():
        cls_plans = [p for p in plans if cls in p.classes]
        for cell in cells:
            r = base[cell]
            t = targets[r["target"]]
            for p in cls_plans:
                png_rel = f"renders/{r['task_id']}/{r['target']}/{p.name}.png"
                prio = fp._prio(r["task_id"], designs, assets, cache)
                row = fp._build_one(p.name, list(p.margins), designs[r["task_id"]]["source"], t, prio, measurer, assets, png_rel, OUT)
                rows.append({"cell": cell, "class": cls, "target": r["target"], "plan": p.name,
                             "own_size": r["target"] in src[p.name]["sizes"], **{k: row[k] for k in ("built", "n_candidates", "n_passed", "total", "png", "reason_if_not")}})
                print(cell, p.name, row["built"], row["n_passed"], "/", row["n_candidates"], flush=True)
    (OUT / "check.json").write_text(json.dumps(rows, indent=1))
    # contact sheet
    CELL = 230
    sheet_cells = [c for cls in chosen for c in chosen[cls][:rows_per_class]]
    heads = ["original", "today (base)", "bisection", "best archetype"]
    img = Image.new("RGB", (4 * CELL + 150, len(sheet_cells) * CELL + 24), "white")
    d = ImageDraw.Draw(img)
    for j, h in enumerate(heads):
        d.text((150 + j * CELL + 4, 4), h, fill="black")

    def put(path_or_img, i, j):
        im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img).convert("RGB")
        im.thumbnail((CELL - 6, CELL - 6))
        img.paste(im, (150 + j * CELL + 3, 24 + i * CELL + 3))

    for i, cell in enumerate(sheet_cells):
        r = base[cell]
        mine = [x for x in rows if x["cell"] == cell and x["built"]]
        best = max(mine, key=lambda x: x["total"]) if mine else None
        d.text((4, 28 + i * CELL), f"{r['target']}\n{r['task_id'][:18]}\n{best['plan'] if best else 'NO CANDIDATE'}", fill="black")
        put(render_to_image(designs[r["task_id"]]["source"], assets_dir=assets, max_size=CELL * 2), i, 0)
        put(ab.AB_ROOT / "base" / r["png"], i, 1)
        put(ab.AB_ROOT / "bisect" / bis[cell]["png"], i, 2)
        if best:
            put(OUT / best["png"], i, 3)
    img.save(OUT / "sheet.png")
    n = len(rows)
    print("pairs", n, "built", sum(x["built"] for x in rows), "candidates", sum(x["n_candidates"] for x in rows),
          "passed", sum(x["n_passed"] for x in rows))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
