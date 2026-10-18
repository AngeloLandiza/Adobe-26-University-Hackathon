"""Layout templates: the slot boxes each target shape offers
(docs/pipeline/02-resize-engine.md §3). Stage 1 uses fixed fractions of the
content box (canvas minus margins), as listed in the plan's Task 9 table."""

from dataclasses import dataclass
from itertools import product

from engine.pipeline.geometry import Box
from engine.pipeline.layout.aspect import aspect_class

IMAGE_ROLES = ("hero_image", "secondary_image")
TEXT_ROLES = ("headline", "subheadline", "other_text")


@dataclass
class Slot:
    """A region units can be placed in. `stack` slots hold several units top to
    bottom; the `minor` slot takes whatever no other slot accepts. `align`
    says where a unit sits when it doesn't fill the slot."""

    name: str
    box: Box
    accepts: tuple[str, ...]
    stack: bool = False
    minor: bool = False
    align: str = "center"


TEMPLATES_BY_CLASS = {
    "very_tall": ["stack"], "tall": ["stack"], "square": ["stack"],
    "landscape": ["columns"], "wide": ["row"], "strip": ["row"],
}

_MARGINS = (0.04, 0.07)
_SPLITS = (0.35, 0.45, 0.55)
PARAM_GRID: dict[str, list[dict]] = {
    "stack": [{"margin": m, "hero_first": h} for m, h in product(_MARGINS, (True, False))],
    "columns": [{"margin": m, "split": s, "mirror": r} for m, s, r in product(_MARGINS, _SPLITS, (False, True))],
    "row": [{"margin": m, "split": s, "mirror": r, "cutoff": c}
            for m, s, r, c in product(_MARGINS, _SPLITS, (False, True), (None, 4, 3))],
}


def _split(box: Box, parts: list[tuple[str, float]], vertical: bool) -> list[tuple[str, Box]]:
    """Cut `box` into consecutive named pieces sized by fraction, top to bottom or left to right."""
    x, y, w, h = box
    out, offset = [], 0.0
    for name, frac in parts:
        if vertical:
            out.append((name, (x, y + offset * h, w, frac * h)))
        else:
            out.append((name, (x + offset * w, y, frac * w, h)))
        offset += frac
    return out


def _slot(name: str, box: Box, align: str = "center") -> Slot:
    if name == "text":
        return Slot(name, box, TEXT_ROLES, stack=True)
    if name == "minor":
        return Slot(name, box, (), minor=True)
    if name == "hero":
        return Slot(name, box, IMAGE_ROLES)
    return Slot(name, box, (name,), align=align)  # logo, cta, badge


def build_slots(template: str, width: float, height: float, params: dict) -> list[Slot]:
    margin = params["margin"] * min(width, height)
    content = (margin, margin, width - 2 * margin, height - 2 * margin)

    if template == "stack":
        middle = [("hero", 0.40), ("text", 0.30)]
        if not params["hero_first"]:
            middle.reverse()
        slots = []
        for name, box in _split(content, [("header", 0.10), *middle, ("cta", 0.12), ("minor", 0.08)], vertical=True):
            if name == "header":
                slots += [_slot(n, b) for n, b in _split(box, [("badge", 0.5), ("logo", 0.5)], vertical=False)]
            else:
                slots.append(_slot(name, box))
        return slots

    if template == "columns":
        split = params["split"]
        columns = [("text", 1 - split), ("hero", split)]
        if params["mirror"]:
            columns.reverse()
        slots = []
        for name, box in _split(content, columns, vertical=False):
            if name == "text":
                inner = [("logo", 0.15), ("text", 0.55), ("cta", 0.18), ("minor", 0.12)]
                slots += [_slot(n, b) for n, b in _split(box, inner, vertical=True)]
            else:
                slots.append(_slot(name, box))
        return slots

    if template == "row":
        strip = aspect_class(width, height) == "strip"
        hero, cta = params["split"] * 0.5, 0.22
        if strip:
            columns = [("logo", 0.14), ("hero", hero), ("text", 1 - 0.14 - hero - cta), ("cta", cta)]
            inner = [("text", 0.70), ("minor", 0.30)]
        else:  # wide: the logo sits on top of the text column instead of in its own
            columns = [("hero", hero), ("text", 1 - hero - cta), ("cta", cta)]
            inner = [("logo", 0.25), ("text", 0.60), ("minor", 0.15)]
        if params["mirror"]:
            columns.reverse()
        slots = []
        for name, box in _split(content, columns, vertical=False):
            if name == "text":
                slots += [_slot(n, b) for n, b in _split(box, inner, vertical=True)]
            else:
                # In a strip the logo sits at the top, so the space it leaves pools below it
                # in one usable block instead of two thin bands.
                slots.append(_slot(name, box, "top" if strip and name == "logo" else "center"))
        return slots

    raise ValueError(f"unknown template {template!r}")
