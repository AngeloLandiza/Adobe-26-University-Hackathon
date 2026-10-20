"""Per-ad floor-plan retrieval ("C"), switched on with CHAMELEON_RETRIEVAL=1.

Words: a *floor plan* is a fixed set of named slot boxes, as fractions of the
content box (extra_templates.FloorPlan). The research teacher drew floor plans
for specific training ads; the ones that clearly beat the engine's own pick
make up the *library* (retrieval_library.json, built by
dev/distill/export_retrieval_library.py from TRAIN ads only).

At run time, with the switch on, a new ad is described by a small vector
(`ad_vector`, 17 numbers in [-1, 1]); the k nearest library ads (Euclidean
distance) that have a winning plan for the target's aspect class lend their
plans as extra candidate templates. This mirrors
dev/distill/retrieval.recommend exactly (the exporter checks that on every
holdout cell), without importing dev/.

Off (the default), nothing here runs and nothing is read. A missing or bad
library, or any error at all, means no extra plans and one `[distill]` line to
stderr; it never stops the engine.
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from engine.pipeline.datafile import load_data_file
from engine.pipeline.geometry import element_box, union_box
from engine.pipeline.layout.aspect import ASPECT_CLASSES, aspect_class
from engine.pipeline.layout.extra_templates import FloorPlan, _parse_plan
from engine.pipeline.layout.templates import Slot, _slot

ENV = "CHAMELEON_RETRIEVAL"
LIBRARY_ENV = "CHAMELEON_RETRIEVAL_LIBRARY"   # optional path override, like CHAMELEON_EXTRA_TEMPLATES
LIBRARY_PATH = Path(__file__).with_name("retrieval_library.json")
LIBRARY_VERSION = 1
DEFAULT_K = 3

# ---------------------------------------------------------------- the vector (same as dev/distill/retrieval.py)

ROLES = ("headline", "subheadline", "cta", "logo", "hero_image", "secondary_image", "badge", "legal", "other_text")
COUNT_CAP = 3        # a role count of COUNT_CAP or more reads as 1.0
HEADLINE_CHARS_CAP = 80
HEADLINE_WORDS_CAP = 16
UNITS_CAP = 12
ASPECT_LOG2_RANGE = 2.0   # aspect ratios from 1:4 to 4:1 span [-1, 1]; beyond is clipped

DIMENSIONS = (
    *(f"count_{r}" for r in ROLES),   # 0-8: units with that role, min(count, 3) / 3
    "headline_chars",                  # 9: characters of the headline text, min(n, 80) / 80
    "headline_words",                  # 10: words of the headline text, min(n, 16) / 16
    "hero_aspect",                     # 11: log2(width/height of the hero image unit) / 2, clipped; 0 if none
    "logo_aspect",                     # 12: same for the logo unit
    "source_canvas_aspect",            # 13: same for the source canvas
    "n_units",                         # 14: ranked content units, min(n, 12) / 12
    "has_group",                       # 15: 1 if any unit is a multi-element group
    "has_button",                      # 16: 1 if any unit is a button (a shape with its label text)
)


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def _warn(msg: str) -> None:
    print(f"[distill] {msg}", file=sys.stderr)


def _scaled_aspect(w, h) -> float:
    if not w or not h or w <= 0 or h <= 0:
        return 0.0
    return float(max(-1.0, min(1.0, math.log2(w / h) / ASPECT_LOG2_RANGE)))


def _unit_elements(unit, by_id: dict) -> list[dict]:
    return [by_id[i] for i in unit.element_ids if i in by_id]


def _unit_aspect(unit, by_id: dict) -> float:
    els = _unit_elements(unit, by_id)
    if not els:
        return 0.0
    _, _, w, h = union_box([element_box(e) for e in els])
    return _scaled_aspect(w, h)


def ad_vector(source: dict, plan) -> np.ndarray:
    """A small, deterministic description of an ad's content (see DIMENSIONS); every value
    lies in [-1, 1]. `plan` is the ad's PriorityPlan (roles and ranks of its units)."""
    by_id = {e["id"]: e for e in source.get("elements", [])}
    ranked = plan.ranked_units()
    v = np.zeros(len(DIMENSIONS))
    for i, role in enumerate(ROLES):
        v[i] = min(sum(1 for u in ranked if u.role == role), COUNT_CAP) / COUNT_CAP
    headline = next((u for u in ranked if u.role == "headline"), None)
    if headline is not None:
        text = " ".join(str(e.get("content", "")) for e in _unit_elements(headline, by_id) if e.get("type") == "text")
        v[9] = min(len(text), HEADLINE_CHARS_CAP) / HEADLINE_CHARS_CAP
        v[10] = min(len(text.split()), HEADLINE_WORDS_CAP) / HEADLINE_WORDS_CAP
    hero = next((u for u in ranked if u.role == "hero_image"), None)
    logo = next((u for u in ranked if u.role == "logo"), None)
    v[11] = _unit_aspect(hero, by_id) if hero is not None else 0.0
    v[12] = _unit_aspect(logo, by_id) if logo is not None else 0.0
    canvas = source.get("canvas", {})
    v[13] = _scaled_aspect(canvas.get("width"), canvas.get("height"))
    v[14] = min(len(ranked), UNITS_CAP) / UNITS_CAP
    v[15] = float(any(u.kind == "group" for u in ranked))
    v[16] = float(any(u.kind == "container_label" for u in ranked))
    return v


def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


# ---------------------------------------------------------------- the library file

_cache: dict = {}
_warned_missing: set[str] = set()


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _parse_entry(raw) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("an entry must be an object")
    pid, task, cls, vec, plan = (raw.get(k) for k in ("proposal_id", "task_id", "aspect_class", "vector", "plan"))
    if not isinstance(pid, str) or not isinstance(task, str):
        raise ValueError("proposal_id and task_id must be strings")
    if cls not in ASPECT_CLASSES:
        raise ValueError(f"aspect_class must be one of {list(ASPECT_CLASSES)}")
    if not isinstance(vec, list) or len(vec) != len(DIMENSIONS) or not all(_finite(x) for x in vec):
        raise ValueError(f"vector must be {len(DIMENSIONS)} finite numbers")
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")
    plan = {"margins": plan.get("margins"), "slots": plan.get("slots")}
    _parse_plan({"name": "check", "classes": [cls], **plan}, {})  # raises ValueError on a bad plan
    return {"proposal_id": pid, "task_id": task, "aspect_class": cls,
            "vector": np.asarray(vec, dtype=float), "plan": json.loads(json.dumps(plan))}


def parse_library(text: str) -> tuple[dict, ...]:
    """The library's entries ({proposal_id, task_id, aspect_class, vector, plan}). A wrong header
    rejects the whole file (ValueError); a bad entry is skipped with one `[distill]` line."""
    raw = json.loads(text)
    if not isinstance(raw, dict) or raw.get("version") != LIBRARY_VERSION:
        raise ValueError(f"top level must be an object with version {LIBRARY_VERSION}")
    if raw.get("dimensions") != list(DIMENSIONS):
        raise ValueError("dimensions differ from this engine's ad_vector")
    if not isinstance(raw.get("entries"), list):
        raise ValueError('"entries" must be a list')
    entries = []
    for item in raw["entries"]:
        try:
            entries.append(_parse_entry(item))
        except (ValueError, TypeError, AttributeError) as err:
            label = item.get("proposal_id") if isinstance(item, dict) else None
            _warn(f"skipping retrieval library entry {label!r}: {err}")
    return tuple(entries)


def load_library(path: Path | None = None) -> tuple[dict, ...]:
    """Entries from `path`, else $CHAMELEON_RETRIEVAL_LIBRARY, else LIBRARY_PATH; () on any problem.
    Cached per file version (engine.pipeline.datafile), so each warning prints once."""
    explicit = path or os.environ.get(LIBRARY_ENV)
    if not explicit:
        try:
            missing = not Path(LIBRARY_PATH).is_file()
        except OSError:
            missing = False  # let load_data_file report it
        if missing:  # load_data_file is silent here; asking for retrieval without a library is worth a line
            if str(LIBRARY_PATH) not in _warned_missing:
                _warned_missing.add(str(LIBRARY_PATH))
                _warn(f"retrieval library {LIBRARY_PATH} not found; no retrieval")
            return ()
    return load_data_file(path, LIBRARY_ENV, LIBRARY_PATH, parse_library, (), "retrieval library", _cache)


# ---------------------------------------------------------------- recommend (same as dev/distill/retrieval.py)

def _content_key(plan: dict) -> str:
    return json.dumps([plan["classes"], plan["margins"], plan["slots"]], sort_keys=True)


def recommended_name(plan: dict) -> str:
    """`r` + 8 hex of the plan's content: same plan, same name; never a built-in or B2 name."""
    return "r" + hashlib.sha1(_content_key(plan).encode()).hexdigest()[:8]


def target_class(target) -> str:
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        return aspect_class(target["width"], target["height"])
    return aspect_class(target.width, target.height)


def nearest_ads(vector, cls: str, lib, k: int = DEFAULT_K, exclude_task: str | None = None) -> list[tuple[str, float]]:
    """(task_id, distance) of the k nearest library ads with a plan in `cls` (ties by task_id)."""
    ads = {e["task_id"]: e["vector"] for e in lib if e["aspect_class"] == cls and e["task_id"] != exclude_task}
    ranked = sorted((distance(vector, v), t) for t, v in ads.items())
    return [(t, d) for d, t in ranked[:k]]


def recommend(source: dict, target, plan, k: int = DEFAULT_K, lib=None, exclude_task: str | None = None) -> list[dict]:
    """The plans of the k nearest library ads within the target's aspect class, as floor-plan
    dicts valid for extra_templates._parse_plan: renamed `r<hash>`, `classes` set to the
    target's class, duplicates dropped. `plan` is the source's PriorityPlan."""
    lib = load_library() if lib is None else lib
    cls = target_class(target)
    out, seen = [], set()
    for task, _ in nearest_ads(ad_vector(source, plan), cls, lib, k, exclude_task):
        for e in sorted((e for e in lib if e["task_id"] == task and e["aspect_class"] == cls),
                        key=lambda e: e["proposal_id"]):
            src = e["plan"]
            new = {"name": "", "classes": [cls], "margins": list(src["margins"]),
                   "slots": [{"name": s["name"], "box": list(s["box"])} for s in src["slots"]]}
            key = _content_key(new)
            if key in seen:
                continue
            seen.add(key)
            new["name"] = recommended_name(new)
            try:
                _parse_plan(new, {})
            except ValueError as err:
                _warn(f"retrieval: dropping {e['proposal_id']}: {err}")
                continue
            out.append(new)
    return out


def extra_plans_for(source: dict, target: dict, plan, k: int = DEFAULT_K, path: Path | None = None) -> list[FloorPlan]:
    """With CHAMELEON_RETRIEVAL=1: the recommended floor plans for this ad and target, in
    recommendation order. Otherwise, or on any problem, []. Never raises."""
    if not enabled():
        return []
    try:
        lib = load_library(path)
        if not lib:
            return []
        return [_parse_plan(p, {}) for p in recommend(source, target, plan, k, lib)]
    except Exception as err:  # deliberately broad: retrieval must never stop the engine
        _warn(f"retrieval failed ({err!r}); no retrieval plans")
        return []


# ---------------------------------------------------------------- slots and candidates

def param_grid(fplan: FloorPlan) -> list[dict]:
    """Same as templates.param_grid for a floor plan: one parameter set per margin."""
    return [{"margin": m} for m in fplan.margins]


def build_slots(fplan: FloorPlan, width: float, height: float, params: dict) -> list[Slot]:
    """Same as templates.build_slots for a loaded floor plan: fractions of the content box
    (canvas minus margin x short side) mapped to pixels."""
    margin = params["margin"] * min(width, height)
    cx, cy, cw, ch = margin, margin, width - 2 * margin, height - 2 * margin
    return [_slot(name, (cx + x * cw, cy + y * ch, w * cw, h * ch)) for name, (x, y, w, h) in fplan.slots]


def build_candidate(fplan: FloorPlan, params: dict, source: dict, target: dict, plan, measurer):
    """One candidate from a per-ad floor plan: the body of generate._template_candidate with
    `build_slots` swapped for this module's (the plan is not in extra_templates.json, so
    templates.build_slots can't find it). None if it can't be built. A test pins it to
    _template_candidate's output for the same plan loaded as an extra template."""
    from engine.pipeline.layout import flex, generate  # here, so generate may import this module

    b = generate._Builder(source, target, plan, measurer)
    slots = build_slots(fplan, b.width, b.height, params)
    assignment = generate.assign(plan, slots, params.get("cutoff"))
    if flex.enabled():
        m = params["margin"] * min(b.width, b.height)
        slots = flex.flex_slots(slots, assignment, (m, m, b.width - 2 * m, b.height - 2 * m),
                                b.els, plan, measurer, b.width, b.height)
    for index in sorted(assignment, key=lambda i: (slots[i].minor, i)):
        slot, items = slots[index], assignment[index]
        if slot.stack:
            b.place_stack([u for item in items for u in item], slot.box)
        elif slot.minor:
            b.place_flow([u for item in items for u in item], slot.box)
        else:
            for item in items:
                b.place_row(item, slot.box, minor=False, align=slot.align)
    if not b.place_leftovers(params["margin"] * min(b.width, b.height)):
        return None
    return b.finish(generate._candidate_id(fplan.name, params), fplan.name, params)


def candidates(source: dict, target: dict, plan, measurer, k: int = DEFAULT_K) -> list:
    """With CHAMELEON_RETRIEVAL=1: a candidate for every recommended plan x margin that builds,
    in a fixed order. [] when off. Never raises; a plan that fails to build is skipped."""
    out = []
    for fplan in extra_plans_for(source, target, plan, k):
        for params in param_grid(fplan):
            try:
                cand = build_candidate(fplan, params, source, target, plan, measurer)
            except Exception:
                cand = None
            if cand is not None:
                out.append(cand)
    return out
