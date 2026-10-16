"""The safe fallback candidate (docs/pipeline/02-resize-engine.md §8): the
whole source scaled uniformly to fit the target and centred. It's ugly on
extreme targets but passes every gate, so resize() always has something
valid to return."""

from engine.pipeline.layout.layers import fit_inside, place_background, scale_about
from engine.pipeline.layout.normalize import finalize
from engine.pipeline.layout.place import resize_text
from engine.pipeline.textmeasure import TextMeasurer
from engine.pipeline.types import Candidate, PriorityPlan

_default_measurer: TextMeasurer | None = None


def _get_measurer() -> TextMeasurer:
    global _default_measurer
    if _default_measurer is None:
        _default_measurer = TextMeasurer()
    return _default_measurer


def _keep_text_fitting(el: dict, measurer: TextMeasurer) -> dict:
    """Uniform scaling usually keeps the wrap, but the 1 px font minimum and
    tiny-size glyph rounding can overflow a scaled box; shrink if they do."""
    if measurer.fits(el, el["font_size"], el["width"], el["height"]):
        return el
    size = measurer.fit_font_size(el, el["width"], el["height"], 1.0, el["font_size"])
    return el if size is None else resize_text(el, size)


def safe_fallback(source: dict, target: dict, plan: PriorityPlan, measurer: TextMeasurer | None = None) -> Candidate:
    measurer = measurer or _get_measurer()
    width, height = target["width"], target["height"]
    src_w, src_h = source["canvas"]["width"], source["canvas"]["height"]
    s = min(width / src_w, height / src_h)
    ox, oy = (width - src_w * s) / 2, (height - src_h * s) / 2
    by_id = {el["id"]: el for el in source["elements"]}

    groups = [(u.role == "background", [by_id[i] for i in u.element_ids]) for u in plan.units]
    planned = {i for u in plan.units for i in u.element_ids}
    groups += [(False, [el]) for el in source["elements"] if el["id"] not in planned]

    placed: dict[str, dict] = {}
    for is_background, members in groups:
        if is_background:
            placed.update((m["id"], place_background(m, width, height)) for m in members)
            continue
        scaled = []
        for m in members:
            new = scale_about(m, s, 0, 0)
            new["x"] += ox
            new["y"] += oy
            scaled.append(_keep_text_fitting(new, measurer) if m["type"] == "text" else new)
        # Elements that bled off the source canvas get pulled in, one unit at a time.
        placed.update((e["id"], e) for e in fit_inside(scaled, width, height))

    design = finalize(source, placed, target)
    if design is None:
        raise RuntimeError("safe fallback left an element outside the canvas")
    return Candidate("fallback", design, "fallback", {}, [], True)
