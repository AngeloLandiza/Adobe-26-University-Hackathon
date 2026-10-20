"""Extra layout templates ("floor plans"), optionally loaded from a data file
(extra_templates.json). The three built-in templates stay in templates.py; a
floor plan is a fixed set of slot boxes, given as fractions of the content box.
A missing file is the normal state and is silent; a bad file or a bad plan never
stops the engine: it is skipped with one `[distill]` line to stderr.
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from engine.pipeline.datafile import load_data_file
from engine.pipeline.layout.aspect import ASPECT_CLASSES

EXTRA_PATH = Path(__file__).with_name("extra_templates.json")
ENV_VAR = "CHAMELEON_EXTRA_TEMPLATES"
BUILTIN_NAMES = ("stack", "columns", "row")
SLOT_NAMES = ("hero", "text", "logo", "cta", "badge", "minor")
TOL = 1e-6
NAME_RE = re.compile(r"[a-z0-9_]+")
MAX_OVERLAP = 0.01  # of the smaller slot's area


@dataclass(frozen=True)
class FloorPlan:
    name: str
    classes: tuple[str, ...]
    slots: tuple[tuple[str, tuple[float, float, float, float]], ...]  # (name, (x, y, w, h)) in content-box fractions
    margins: tuple[float, ...] = (0.04, 0.07)


_cache: dict[tuple, dict[str, FloorPlan]] = {}


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def _overlap(a, b) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def _parse_plan(raw, seen) -> FloorPlan:
    if not isinstance(raw, dict):
        raise ValueError("a floor plan must be an object")
    name = raw.get("name")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError(f"name must match {NAME_RE.pattern} (it appears in candidate ids)")
    if name in BUILTIN_NAMES or name in seen:
        raise ValueError(f"name {name!r} is already taken")
    classes = raw.get("classes")
    if not isinstance(classes, list) or not classes or any(c not in ASPECT_CLASSES for c in classes):
        raise ValueError(f"classes must be a non-empty list drawn from {list(ASPECT_CLASSES)}")
    margins = raw.get("margins", [0.04, 0.07])
    if not isinstance(margins, list) or not margins or not all(_number(m) and 0 <= m < 0.5 for m in margins):
        raise ValueError("margins must be a non-empty list of numbers in [0, 0.5)")
    if len(set(margins)) != len(margins):
        raise ValueError("margins must not repeat (candidate ids would collide)")
    raw_slots = raw.get("slots")
    if not isinstance(raw_slots, list):
        raise ValueError("slots must be a list")
    slots, names = [], set()
    for item in raw_slots:
        slot_name = item.get("name") if isinstance(item, dict) else None
        if slot_name not in SLOT_NAMES:
            raise ValueError(f"slot names must come from {list(SLOT_NAMES)}")
        if slot_name in names:
            raise ValueError(f"slot {slot_name!r} appears twice")
        names.add(slot_name)
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 4 or not all(_number(v) for v in box):
            raise ValueError(f"slot {slot_name!r}: box must be four numbers [x, y, w, h]")
        x, y, w, h = box
        if w <= 0 or h <= 0 or x < -TOL or y < -TOL or x + w > 1 + TOL or y + h > 1 + TOL:
            raise ValueError(f"slot {slot_name!r}: box must have w, h > 0 and lie inside the unit square")
        slots.append((slot_name, (float(x), float(y), float(w), float(h))))
    if "text" not in names or "minor" not in names:
        raise ValueError("slots must include both 'text' and 'minor'")
    for i, (na, a) in enumerate(slots):
        for nb, b in slots[i + 1:]:
            if _overlap(a, b) > MAX_OVERLAP * min(a[2] * a[3], b[2] * b[3]):
                raise ValueError(f"slots {na!r} and {nb!r} overlap")
    return FloorPlan(name, tuple(classes), tuple(slots), tuple(float(m) for m in margins))


def _parse(text: str) -> dict[str, FloorPlan]:
    raw = json.loads(text)
    if not isinstance(raw, dict) or not isinstance(raw.get("floor_plans"), list):
        raise ValueError('top level must be an object with a "floor_plans" list')
    plans: dict[str, FloorPlan] = {}
    for entry in raw["floor_plans"]:
        try:
            plan = _parse_plan(entry, plans)
        except ValueError as err:
            label = entry.get("name") if isinstance(entry, dict) else None
            print(f"[distill] skipping floor plan {label!r}: {err}", file=sys.stderr)
            continue
        plans[plan.name] = plan
    return plans


def load_floor_plans(path: Path | None = None) -> list[FloorPlan]:
    """Floor plans from `path`, else $CHAMELEON_EXTRA_TEMPLATES, else EXTRA_PATH, in file order.
    A fresh list each call; the file is re-read when it changes (see engine.pipeline.datafile)."""
    return list(_plans(path).values())


def floor_plan(name: str, path: Path | None = None) -> FloorPlan | None:
    """The loaded floor plan called `name`, or None."""
    return _plans(path).get(name)


def _plans(path: Path | None) -> dict[str, FloorPlan]:
    return load_data_file(path, ENV_VAR, EXTRA_PATH, _parse, {}, "extra templates", _cache)
