"""Sonnet's learned proportions (CHAMELEON_TARGETS; off by default, and with it
off nothing here runs).

The LLM designer (Sonnet) beats v5 by ~1 point of 5, and its advantage lives in
its exact geometry: a bigger hero image and a smaller headline than v5 (median
headline font / short side 0.088 vs 0.110, hero 6.4% vs 4.8% of the canvas).
Those proportions are predictable from a handful of source + target features
(grouped CV R^2 0.9 for the headline font, see dev/distill/fit_targets.py), so
small ridge models, fitted offline on 2,659 Sonnet layouts of train ads, give
the size each piece *should* have on a new canvas. The solver
(layout/solver.py) pulls block scales toward them; anchored and bisection use
them where they choose text and hero sizes.

Targets (all relative to the target canvas; S = its short side):

- `font_<role>`: the largest font of a role's text / S (headline,
  subheadline, other_text, cta, legal, badge);
- `cta_h` / `cta_w`: the CTA pill's height / S and width / W;
- `logo`, `hero`: sqrt(box area / canvas area) of the logo unit and the hero
  image;
- `content_w`, `content_h`: the content's bounding box / W and / H;
- `margin`: the content box's smallest edge gap / S;
- `gap`: mean nearest-neighbour gap between content units / headline font.

`predict` returns only the targets the model is sure about: the role exists
in the source, every base feature is within the training range (widened
by EXTRAP of it on each side; pairwise products are not checked), and the target passed the CV bar when the models were fitted (R^2 >=
0.3; weaker targets are not written to sonnet_targets.json). Numpy only;
never raises (an empty dict when anything goes wrong).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

ENV = "CHAMELEON_TARGETS"
MODEL_PATH = Path(__file__).with_name("sonnet_targets.json")
EXTRAP = 0.25         # a base feature beyond the training range by more than this x the range: unsure
TEXT_ROLES = ("headline", "subheadline", "other_text", "cta", "legal", "badge")
_CONTENT_SKIP = ("background", "decoration")
CONFLICT = 0.1       # units of one piece wanting to grow and shrink by more than this: no target for it
TEXT_SLACK = 1.1      # (bisection) a text may take at most this x its predicted size (floors still win)
_MODEL = None


def mode() -> str | None:
    """CHAMELEON_TARGETS=1: pulls on every candidate before grading (and the strategies' sizes);
    =winner: only the picked layout is pulled (after pick_best). None: off."""
    v = os.environ.get(ENV)
    return v if v in ("1", "winner") else None


def enabled() -> bool:
    return mode() == "1"


def winner_only() -> bool:
    return mode() == "winner"


# ---------------------------------------------------------------- measuring a layout

def _box(e: dict) -> tuple[float, float, float, float]:
    return float(e["x"]), float(e["y"]), float(e["width"]), float(e["height"])


def _union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return math.hypot(dx, dy)


def measure(elements: dict, plan, W: float, H: float) -> dict[str, float]:
    """The target quantities read off a layout: `elements` is {id: element
    (x, y, width, height, font_size for text)}; missing pieces are left out."""
    S = min(W, H)
    out: dict[str, float] = {}
    units = [u for u in plan.ranked_units() if u.element_ids and all(i in elements for i in u.element_ids)]
    for role in TEXT_ROLES:
        fonts = [float(elements[i]["font_size"]) for u in units if u.role == role for i in u.element_ids
                 if elements[i].get("font_size")]
        if fonts:
            out[f"font_{role}"] = max(fonts) / S
    cta = next((u for u in units if u.role == "cta"), None)
    if cta is not None:
        shapes = [elements[i] for i in cta.element_ids if elements[i].get("type", "shape") == "shape"
                  and not elements[i].get("font_size")]
        if shapes:
            b = _union([_box(e) for e in shapes])
            out["cta_h"], out["cta_w"] = b[3] / S, b[2] / W
    for role, key in (("logo", "logo"), ("hero_image", "hero")):
        u = next((u for u in units if u.role == role), None)
        if u is not None:
            b = _union([_box(elements[i]) for i in u.element_ids])
            out[key] = math.sqrt(max(0.0, b[2] * b[3]) / (W * H))
    content = []
    for u in units:
        if u.role in _CONTENT_SKIP:
            continue
        b = _union([_box(elements[i]) for i in u.element_ids])
        if b[2] * b[3] < 0.9 * W * H:
            content.append(b)
    if content:
        U = _union(content)
        out["content_w"], out["content_h"] = U[2] / W, U[3] / H
        out["margin"] = max(0.0, min(U[0], U[1], W - U[0] - U[2], H - U[1] - U[3])) / S
        head = out.get("font_headline")
        if head and len(content) > 1:
            gaps = [min(_gap(a, b) for j, b in enumerate(content) if j != i) for i, a in enumerate(content)]
            out["gap"] = (sum(gaps) / len(gaps)) / (head * S)
    return out


def unit_ratio(unit, els: dict, T: dict, W: float, H: float) -> float | None:
    """Scale that brings one priority-plan unit, drawn as `els` (id -> element with x, y, width, height,
    font_size), to its predicted size; None if nothing is predicted for it. Text by its largest font; a CTA
    also by its pill height (geometric mean); a logo or hero image by sqrt(box area / canvas area)."""
    members = [els[i] for i in unit.element_ids if i in els]
    if not members or not T:
        return None
    S = min(W, H)
    role = unit.role
    out = []
    fonts = [e["font_size"] for e in members if e.get("type") == "text" and e.get("font_size")]
    key = f"font_{role}"
    if fonts and key in T:
        out.append(T[key] * S / max(fonts))
    if role == "cta" and "cta_h" in T:
        pills = [e for e in members if e.get("type") == "shape"]
        if pills:
            h = _union([_box(e) for e in pills])[3]
            if h > 0:
                out.append(T["cta_h"] * S / h)
    if role in ("logo", "hero_image") and not fonts:
        key = "logo" if role == "logo" else "hero"
        if key in T:
            b = _union([_box(e) for e in members])
            cur = math.sqrt(max(0.0, b[2] * b[3]) / (W * H))
            if cur > 0:
                out.append(T[key] / cur)
    out = [r for r in out if r > 0 and math.isfinite(r)]
    if not out:
        return None
    return math.exp(sum(math.log(r) for r in out) / len(out))


def group_ratio(units: list, els: dict, T: dict, W: float, H: float) -> float | None:
    """unit_ratio of several units moving as one piece: their geometric mean weighted by priority
    (rank ** -0.7); None when they disagree (one should grow and another shrink by more than CONFLICT,
    e.g. a headline set on the hero photo), since one scale can't serve both."""
    rs, ws = [], []
    for u in units:
        r = unit_ratio(u, els, T, W, H)
        if r is not None:
            rs.append(math.log(r))
            ws.append((u.rank or 10) ** -0.7)
    if not rs or (max(rs) > math.log(1 + CONFLICT) and min(rs) < -math.log(1 + CONFLICT)):
        return None
    return math.exp(sum(a * w for a, w in zip(rs, ws)) / sum(ws))


# ---------------------------------------------------------------- features

_PROD = ("la_t", "la_s", "lA", "lS", "lc_head", "f_head", "hero_src", "r_src", "r_chars")


def _chars(source: dict, plan, role: str) -> float:
    els = {e["id"]: e for e in source["elements"]}
    n = [len(str(els[i].get("content") or "")) for u in plan.ranked_units() if u.role == role
         for i in u.element_ids if i in els and els[i]["type"] == "text"]
    return math.log(max(n) + 1) if n else 0.0


def source_stats(source: dict, plan) -> dict:
    """What the features need from the source alone (one pass per source)."""
    sW, sH = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    els = {e["id"]: e for e in source["elements"]}
    src = measure(els, plan, sW, sH)
    chars = {r: _chars(source, plan, r) for r in TEXT_ROLES}
    roles = {u.role for u in plan.ranked_units()}
    return {"sW": sW, "sH": sH, "m": src, "chars": chars, "n_el": len(source["elements"]),
            "n_txt": sum(1 for e in source["elements"] if e["type"] == "text"),
            "has_hero": float("hero_image" in roles), "roles": sorted(r for r in roles if r)}


def _role_of(name: str) -> str | None:
    if name.startswith("font_"):
        return name[5:]
    return {"cta_h": "cta", "cta_w": "cta", "logo": "logo", "hero": "hero_image"}.get(name)


def features(st: dict, W: float, H: float, name: str) -> dict[str, float] | None:
    """Feature dict for one target, or None when the source lacks what it needs."""
    sW, sH = st["sW"], st["sH"]
    S, sS = min(W, H), min(sW, sH)
    m = st["m"]
    role = _role_of(name)
    if role is not None and role not in st["roles"]:
        return None
    if "font_headline" not in m:
        return None
    r_src = m.get(name)
    if r_src is None:
        return None
    if role in TEXT_ROLES:
        r_chars = st["chars"].get(role, 0.0)
    else:
        r_chars = st["chars"].get("headline", 0.0)
    f = {"la_t": math.log(W / H), "la_s": math.log(sW / sH), "lA": math.log(W * H / 1e6), "lS": math.log(S / sS),
         "ls0": math.log(min(W / sW, H / sH)), "lfill": math.log(max(W / sW, H / sH)),
         "lc_head": st["chars"].get("headline", 0.0), "f_head": m["font_headline"],
         "n_el": st["n_el"] / 10.0, "n_txt": st["n_txt"] / 10.0, "has_hero": st["has_hero"],
         "hero_src": m.get("hero", 0.0), "r_src": r_src, "r_chars": r_chars}
    for i, a in enumerate(_PROD):
        for b in _PROD[i:]:
            f[f"{a}*{b}"] = f[a] * f[b]
    return f


# ---------------------------------------------------------------- the models

def load(path: Path | None = None) -> dict:
    global _MODEL
    if path is not None:
        return json.loads(Path(path).read_text())
    if _MODEL is None:
        try:
            _MODEL = json.loads(MODEL_PATH.read_text())
        except Exception:
            _MODEL = {"targets": {}}
    return _MODEL


def _inverse(z: float, transform: str) -> float:
    return math.exp(z) if transform == "log" else z


def predict_one(spec: dict, f: dict) -> float | None:
    names = spec["features"]
    try:
        z = 0.0
        for k, mu, sd, w, lo, hi in zip(names, spec["mean"], spec["std"], spec["w"], spec["lo"], spec["hi"]):
            v = f[k]
            if "*" not in k and not (lo - EXTRAP * (hi - lo) - 1e-9 <= v <= hi + EXTRAP * (hi - lo) + 1e-9):
                return None
            z += w * (v - mu) / sd
        y = _inverse(z + spec["b"], spec["transform"])
    except (KeyError, ValueError, OverflowError, ZeroDivisionError):
        return None
    lo, hi = spec["range"]
    if not math.isfinite(y):
        return None
    return min(max(y, lo), hi)


def predict(source: dict, plan, target: dict, model: dict | None = None, st: dict | None = None) -> dict[str, float]:
    """{target name: predicted value} for the targets the model is sure about (see the docstring)."""
    try:
        model = model or load()
        st = st or source_stats(source, plan)
        W, H = float(target["width"]), float(target["height"])
        out = {}
        for name, spec in model.get("targets", {}).items():
            f = features(st, W, H, name)
            if f is None:
                continue
            y = predict_one(spec, f)
            if y is not None:
                out[name] = y
        return out
    except Exception:
        return {}
