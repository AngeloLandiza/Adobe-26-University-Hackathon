"""
THIS IS THE FILE YOU IMPLEMENT.

Everything else in this repo (input loading, output writing, rendering,
the Dockerfile) is plumbing so you can focus on exactly one function:
`resize()`. See RULES.md and the design schema for the full contract.

Our implementation is a three-node pipeline (docs/pipeline/README.md):
node 1 ranks the elements (`priority.plan`), node 2 builds candidate
layouts (`generate_candidates`, plus `safe_fallback`), node 3 grades them
and picks one (`pick_best`). Every step has a fallback, so `resize()`
always returns a valid design.
"""

import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from engine.pipeline.grading.score import pick_best
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.priority import plan as priority_plan
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Candidate, GradeReport, PriorityPlan

DEBUG_DIR = Path("output/debug")
_measurer: TextMeasurer | None = None


def _get_measurer() -> TextMeasurer:
    global _measurer
    if _measurer is None:
        _measurer = TextMeasurer()
    return _measurer


def _log_plan(plan: PriorityPlan) -> None:
    s = plan.stats
    print(f"[priority] source={plan.source} provider={s.provider if s else 'none'} model={s.model if s else '-'} "
          f"latency={s.latency_s if s else 0:.2f}s cost=${s.cost_usd if s else 0:.4f} "
          f"error={(s.error if s and s.error else '-')}", file=sys.stderr)


def resize_with_report(source: dict, target_canvas: dict) -> tuple[dict, PriorityPlan, Candidate, list[GradeReport]]:
    """Run the three nodes. Returns (design, plan, winning candidate, reports)."""
    measurer = _get_measurer()
    plan = priority_plan(source)  # never raises: LLM -> heuristic fallback inside
    _log_plan(plan)

    candidates = [safe_fallback(source, target_canvas, plan, measurer)]  # always first, always valid
    try:
        candidates += generate_candidates(source, target_canvas, plan, measurer)
    except Exception:
        pass  # a node 2 bug never sinks the task
    try:
        best, reports = pick_best(source, target_canvas, plan, candidates, measurer)
    except Exception:
        best, reports = candidates[0], []

    if os.environ.get("CHAMELEON_DEBUG") == "1":
        try:
            _debug_dump(source, target_canvas, plan, candidates, reports)
        except Exception as exc:  # debug output must never change the result
            print(f"[debug] dump failed: {exc!r}", file=sys.stderr)
    return best.design, plan, best, reports


def _debug_dump(source: dict, target: dict, plan: PriorityPlan, candidates: list[Candidate], reports: list[GradeReport]) -> None:
    """plan.json, reports.json (best first) and a contact sheet of the rendered top candidates."""
    from PIL import Image, ImageDraw

    from engine.pipeline.render import render_to_image

    key = hashlib.sha1((json.dumps(source, sort_keys=True) + json.dumps(target, sort_keys=True)).encode()).hexdigest()[:10]
    folder = DEBUG_DIR / key
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plan.json").write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
    ranked = sorted(reports, key=lambda r: -r.total)
    (folder / "reports.json").write_text(json.dumps([asdict(r) for r in ranked], indent=2), encoding="utf-8")

    by_id = {c.candidate_id: c for c in candidates}
    tiles = []
    for report in [r for r in ranked if r.rendered]:
        img = render_to_image(by_id[report.candidate_id].design, max_size=480)
        label = Image.new("RGB", (img.width, 18), "white")
        ImageDraw.Draw(label).text((4, 3), f"{report.total:.3f}  {report.candidate_id}", fill="black")
        tile = Image.new("RGB", (img.width, img.height + 18), "white")
        tile.paste(label, (0, 0))
        tile.paste(img, (0, 18))
        tiles.append(tile)
    if not tiles:
        tiles = [Image.new("RGB", (240, 40), "white")]
    columns = 3
    cell_w, cell_h = max(t.width for t in tiles) + 8, max(t.height for t in tiles) + 8
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_w * min(columns, len(tiles)), cell_h * rows), (230, 230, 230))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * cell_w + 4, (i // columns) * cell_h + 4))
    sheet.save(folder / "contact_sheet.png")


def _last_resort(source: dict, target_canvas: dict) -> dict:
    """The starter's stretch, with each element shrunk and moved until its
    rotated bounds are inside the canvas. Ugly, but it depends on nothing
    that could fail."""
    width, height = target_canvas["width"], target_canvas["height"]
    scale_x = width / source["canvas"]["width"]
    scale_y = height / source["canvas"]["height"]
    elements = []
    for el in source["elements"]:
        el = dict(el)
        w, h = el["width"] * scale_x, el["height"] * scale_y
        rad = math.radians(el.get("rotation") or 0)
        cos, sin = abs(math.cos(rad)), abs(math.sin(rad))
        rot_w, rot_h = w * cos + h * sin, w * sin + h * cos
        shrink = min(1.0, width / rot_w if rot_w else 1.0, height / rot_h if rot_h else 1.0)
        w, h, rot_w, rot_h = w * shrink, h * shrink, rot_w * shrink, rot_h * shrink
        cx = min(max(el["x"] * scale_x + w / 2, rot_w / 2), width - rot_w / 2)
        cy = min(max(el["y"] * scale_y + h / 2, rot_h / 2), height - rot_h / 2)
        el.update(x=cx - w / 2, y=cy - h / 2, width=w, height=h)
        if el["type"] == "text":
            el["font_size"] = max(1.0, el["font_size"] * min(scale_x, scale_y) * shrink)
        elements.append(el)
    return {"canvas": {"width": width, "height": height}, "elements": elements}


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """
    Args:
        source: {"canvas": {"width": int, "height": int}, "elements": [...]}
                 — the design at its original size. See
                 chameleon_design_schema_v0.9.json for the full element spec
                 (text / image / shape, position, size, rotation, opacity,
                 z_index, group_id, and type-specific fields).
        target_canvas: {"width": int, "height": int} — the shape you're
                 resizing into.

    Returns:
        A design dict with the SAME shape as `source`:
        {"canvas": target_canvas, "elements": [...]}

        Hard requirements (see RULES.md for the full list):
        - Every element `id` in `source` must appear exactly once in your
          output. Do not add, remove, or invent elements.
        - Do not regenerate or redraw any asset's pixel content — you are
          repositioning and resizing the elements you were given, nothing
          more. This is a disqualifying rule, not a style preference.
        - Non-background image elements should be kept in proportion
          (matching the source aspect ratio within ~5%) unless they are a
          full-bleed / `fit: "cover"` element, which is expected to crop.
        - `font_size` may change to help text fit. `font_family` and
          `content` should not, for this stage of scoring (see RULES.md,
          "Adaptive typography and copy").
        - Nothing should end up outside `target_canvas`'s bounds.

    This is where your actual approach goes: constraint solving, a learned
    layout model, heuristics, whatever you choose. Latency is not a scoring
    concern (see RULES.md) — spend your effort on quality, not speed.
    """
    try:
        return resize_with_report(source, target_canvas)[0]
    except Exception as exc:
        print(f"[resize] pipeline failed ({exc!r}); using the safe fallback", file=sys.stderr)
    try:
        return safe_fallback(source, target_canvas, heuristic_plan(source)).design
    except Exception as exc:
        print(f"[resize] safe fallback failed ({exc!r}); using the last resort", file=sys.stderr)
    return _last_resort(source, target_canvas)
