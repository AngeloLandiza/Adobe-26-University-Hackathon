"""Offline evaluation of layout strategy S2 `bisection` (no teacher calls).

For every cell (3 samples x 13 targets, plus ~20 corpus ads x 13 targets) it
runs the current engine (`resize_with_report`) and the strategy, checks every
strategy candidate against the hard gates, picks the strategy's best with the
engine's own grader, and records geometry proxies for both:

- text_at_floor: share of text units at or above their role's floor
- empty_rect:    largest empty rectangle (no content unit in it), share of the canvas
- edge_margin:   smallest gap from a content unit to a canvas edge, share of the short side
- logo_share:    logo area, share of the canvas

    python -m dev.distill.eval_bisection run     # writes rows.jsonl (+ designs) to OUT
    python -m dev.distill.eval_bisection sheet   # contact sheet: current winner vs bisection
    python -m dev.distill.eval_bisection table   # the proxy table (markdown)
"""

import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dev.distill.paths import DATA_ROOT

OUT = DATA_ROOT / "strategies.nosync" / "bisection"
TARGETS = json.loads((Path(__file__).resolve().parents[1] / "targets.json").read_text())
INPUT_DIR = Path(__file__).resolve().parents[2] / "input"
WORKERS = 3
GRID = 120  # empty-rectangle grid cells along the long side


def cells() -> list[dict]:
    from dev.distill.corpus import load_corpus

    designs = [{"task_id": p.stem.replace("chameleon_sample_", "sample_"), "brand": "solstice", "split": "sample",
                "source": json.loads(p.read_text())["source"]} for p in sorted(INPUT_DIR.glob("*.json"))]
    corpus = [d for d in load_corpus() if d.get("origin") != "sample"]
    picked = []
    for brand in ("solstice", "offbrand"):
        for split in ("train", "holdout"):
            pool = [d for d in corpus if d["brand"] == brand and d["split"] == split]
            step = max(1, len(pool) // 5)
            picked += pool[::step][:5]
    designs += [{"task_id": d["task_id"], "brand": d["brand"], "split": d["split"], "source": d["source"]}
                for d in picked]
    return [{**d, "target": t} for d in designs for t in TARGETS]


def _largest_empty(boxes: list, width: float, height: float) -> float:
    cell = max(width, height) / GRID
    cols, rows = max(1, int(math.ceil(width / cell))), max(1, int(math.ceil(height / cell)))
    busy = np.zeros((rows, cols), dtype=bool)
    for x, y, w, h in boxes:
        c0, c1 = int(max(0, x // cell)), int(min(cols, math.ceil((x + w) / cell)))
        r0, r1 = int(max(0, y // cell)), int(min(rows, math.ceil((y + h) / cell)))
        busy[r0:r1, c0:c1] = True
    best, heights = 0, np.zeros(cols, dtype=int)
    for r in range(rows):
        heights = np.where(busy[r], 0, heights + 1)
        stack: list[int] = []
        for c in range(cols + 1):
            hc = heights[c] if c < cols else 0
            while stack and heights[stack[-1]] >= hc:
                top = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                best = max(best, int(top) * (c - left))
            stack.append(c)
    return min(1.0, best * cell * cell / (width * height))


def proxies(design: dict, plan) -> dict:
    from engine.pipeline.geometry import rotated_aabb, union_box
    from engine.pipeline.layout.place import floor_px

    width, height = design["canvas"]["width"], design["canvas"]["height"]
    els = {e["id"]: e for e in design["elements"]}
    units = [u for u in plan.ranked_units() if all(i in els for i in u.element_ids)]
    boxes = {u.unit_id: union_box([rotated_aabb(els[i]) for i in u.element_ids]) for u in units}
    texts = [(u, [els[i]["font_size"] for i in u.element_ids if els[i]["type"] == "text"]) for u in units]
    texts = [(u, s) for u, s in texts if s]
    at_floor = [min(s) >= floor_px(u.role, width, height) - 1e-6 for u, s in texts]
    short = min(width, height)
    margin = min((min(x, y, width - x - w, height - y - h) for x, y, w, h in boxes.values()), default=0.0)
    logo = sum(boxes[u.unit_id][2] * boxes[u.unit_id][3] for u in units if u.role == "logo")
    head = next((min(s) for u, s in texts if u.role == "headline"), None)
    return {"text_at_floor": sum(at_floor) / len(at_floor) if at_floor else 1.0,
            "empty_rect": _largest_empty(list(boxes.values()), width, height),
            "edge_margin": max(0.0, margin) / short,
            "logo_share": logo / (width * height),
            "headline_px": head}


def run_cell(cell: dict) -> dict:
    os.environ["CHAMELEON_LLM"] = "heuristic"
    from engine.pipeline.grading.gates import check_gates
    from engine.pipeline.grading.score import pick_best
    from engine.pipeline.layout.strategies.bisection import candidates
    from engine.resize import _get_measurer, resize_with_report

    source, t = cell["source"], cell["target"]
    canvas = {"width": t["width"], "height": t["height"]}
    measurer = _get_measurer()
    design, plan, best, reports = resize_with_report(source, canvas)
    winner_total = next((r.total for r in reports if r.candidate_id == best.candidate_id), 0.0)
    mine = candidates(source, canvas, plan, measurer)
    gate_fail = sum(bool(check_gates(source, c.design, canvas, plan, measurer)) for c in mine)
    row = {"task_id": cell["task_id"], "brand": cell["brand"], "split": cell["split"], "target": t["name"],
           "n_mine": len(mine), "gate_fail": gate_fail, "winner": best.candidate_id,
           "winner_total": winner_total, "winner_proxies": proxies(design, plan),
           "winner_design": design}
    if mine:
        pick, mine_reports = pick_best(source, canvas, plan, mine, measurer)
        row.update(mine_best=pick.candidate_id,
                   mine_total=next(r.total for r in mine_reports if r.candidate_id == pick.candidate_id),
                   mine_proxies=proxies(pick.design, plan), mine_design=pick.design)
    return row


def run(samples_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cells() if not samples_only or c["split"] == "sample"]
    with Pool(WORKERS) as pool, (OUT / "rows.jsonl").open("w") as fh:
        for i, row in enumerate(pool.imap(run_cell, todo, chunksize=2)):
            fh.write(json.dumps(row) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(todo)}", file=sys.stderr, flush=True)


def load_rows() -> list[dict]:
    return [json.loads(line) for line in (OUT / "rows.jsonl").read_text().splitlines() if line.strip()]


def table(rows: list[dict] | None = None) -> str:
    rows = rows or load_rows()
    have = [r for r in rows if r["n_mine"]]
    n_cand = sum(r["n_mine"] for r in rows)
    fails = sum(r["gate_fail"] for r in rows)
    lines = [f"cells: {len(rows)}; cells with bisection candidates: {len(have)}; "
             f"candidates: {n_cand} (mean {n_cand / max(1, len(rows)):.2f}/cell); "
             f"gate failures among returned: {fails}",
             f"engine grader prefers bisection's best over the current winner in {sum(r['mine_total'] > r['winner_total'] for r in have)}/{len(have)} cells",
             "", "| group | cells | text at floor (win -> bis) | largest empty rect | min edge margin | logo share | headline px (median) |",
             "|---|---|---|---|---|---|---|"]
    groups = {"all": have}
    for key in ("strip", "wide", "landscape", "square", "tall", "very_tall"):
        from engine.pipeline.layout.aspect import aspect_class
        groups[key] = [r for r in have if aspect_class(*_wh(r["target"])) == key]
    for name, rs in groups.items():
        if not rs:
            continue
        def mean(side, key):
            return sum(r[f"{side}_proxies"][key] for r in rs) / len(rs)

        def med(side):
            vals = sorted(r[f"{side}_proxies"]["headline_px"] or 0 for r in rs)
            return vals[len(vals) // 2]

        lines.append(f"| {name} | {len(rs)} | {mean('winner', 'text_at_floor'):.3f} -> {mean('mine', 'text_at_floor'):.3f} "
                     f"| {mean('winner', 'empty_rect'):.3f} -> {mean('mine', 'empty_rect'):.3f} "
                     f"| {mean('winner', 'edge_margin'):.3f} -> {mean('mine', 'edge_margin'):.3f} "
                     f"| {mean('winner', 'logo_share'):.4f} -> {mean('mine', 'logo_share'):.4f} "
                     f"| {med('winner'):.1f} -> {med('mine'):.1f} |")
    return "\n".join(lines)


def _wh(name: str) -> tuple[int, int]:
    t = next(t for t in TARGETS if t["name"] == name)
    return t["width"], t["height"]


SHEET_CELLS = [  # (task_id prefix, target): diverse shapes and ads
    ("sample_level3", "leaderboard"), ("sample_level3", "wide_skyscraper"), ("sample_level2", "billboard"),
    ("sample_level2", "mobile_banner"), ("sample_level1", "story"), ("sample_level3", "youtube_thumbnail"),
    ("sample_level2", "half_page"), ("sample_level3", "medium_rectangle"),
]


def sheet(rows: list[dict] | None = None, extra: int = 6) -> Path:
    from PIL import Image, ImageDraw

    from dev.distill.assets import merged_assets_dir
    from engine.pipeline.render import render_to_image

    rows = [r for r in (rows or load_rows()) if r["n_mine"]]
    chosen = []
    for prefix, target in SHEET_CELLS:
        r = next((r for r in rows if r["task_id"].startswith(prefix) and r["target"] == target), None)
        if r:
            chosen.append(r)
    corpus = [r for r in rows if not r["task_id"].startswith("sample_")]
    shapes = ["large_leaderboard", "linkedin_banner", "facebook_cover", "instagram_portrait", "wide_skyscraper",
              "mobile_banner", "story", "billboard"]
    seen = set()
    for i, shape in enumerate(shapes):
        pool = [r for r in corpus if r["target"] == shape and r["task_id"] not in seen]
        if pool and len(chosen) < len(SHEET_CELLS) + extra:
            r = pool[(i * 7) % len(pool)]
            seen.add(r["task_id"])
            chosen.append(r)
    tile = 340
    tiles = []
    with merged_assets_dir() as assets:
        for r in chosen:
            pair = []
            for side in ("winner", "mine"):
                img = render_to_image(r[f"{side}_design"], assets_dir=assets, max_size=tile)
                img.thumbnail((tile, tile))
                pair.append(img)
            w = sum(p.width for p in pair) + 30
            h = max(p.height for p in pair) + 34
            t = Image.new("RGB", (w, h), "white")
            d = ImageDraw.Draw(t)
            d.text((4, 2), f"{r['task_id']} | {r['target']}", fill="black")
            d.text((4, 16), f"engine {r['winner'][:34]} {r['winner_total']:.3f}", fill=(120, 0, 0))
            d.text((pair[0].width + 24, 16), f"bisection {r['mine_total']:.3f}", fill=(0, 90, 0))
            t.paste(pair[0], (4, 32))
            t.paste(pair[1], (pair[0].width + 24, 32))
            tiles.append(t)
    cols = 2
    cw, ch = max(t.width for t in tiles) + 8, max(t.height for t in tiles) + 8
    rows_n = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cw * cols, ch * rows_n), (200, 200, 200))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * cw + 4, (i // cols) * ch + 4))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sheet.png"
    out.save(path)
    return path


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run(samples_only="samples" in sys.argv[2:])
        print(table())
    elif cmd == "sheet":
        print(sheet())
    else:
        print(table())
