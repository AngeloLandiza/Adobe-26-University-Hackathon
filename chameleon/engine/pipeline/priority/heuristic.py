"""Rule-based roles and ranking (docs/pipeline/01-priority-node.md, "Heuristic
ranker"). Node 1 falls back to this whenever the LLM path can't run, and it is
the baseline the LLM has to beat."""

import re

ROLE_ORDER = ("headline", "hero_image", "cta", "logo", "subheadline", "badge", "secondary_image", "other_text", "legal")
ACTION_VERBS = frozenset({
    "shop", "buy", "get", "order", "try", "learn", "discover", "sign", "book",
    "start", "explore", "join", "download", "subscribe", "save", "claim",
})
LEGAL_MARKERS = ("responsibly", "terms", "©", "rights reserved", "void where", "conditions apply")

LOGO_MAX_AREA = 0.06
LOGO_MIN_ASPECT = 1.5
EDGE_BAND = 0.2            # "near an edge": centre within this share of the canvas from a side
LEGAL_MIN_WORDS = 12
LEGAL_MIN_CENTER_Y = 0.8
CTA_MAX_WORDS = 6
SUBHEAD_MIN_RATIO = 0.5    # of the headline's font size


def _first_word(text: str | None) -> str:
    match = re.match(r"\W*(\w+)", text or "")
    return match.group(1).lower() if match else ""


def _legal_marked(f: dict) -> bool:
    return any(marker in (f["text"] or "").lower() for marker in LEGAL_MARKERS)


def _near_edge(f: dict) -> bool:
    cx, cy = f["center"]
    return min(cx, cy) < EDGE_BAND or max(cx, cy) > 1 - EDGE_BAND


def heuristic_roles(features: list[dict]) -> dict[str, str]:
    roles: dict[str, str] = {}
    remaining = []
    for f in features:
        uid = f["unit_id"]
        if "background" in f["flags"]:
            roles[uid] = "background"
        elif all(t == "shape" for t in f["types"]):
            roles[uid] = "decoration"
        elif f["kind"] == "container_label" or (f["kind"] == "group" and "rotated" in f["flags"]):
            roles[uid] = "cta" if _first_word(f["text"]) in ACTION_VERBS else "badge"
        else:
            remaining.append(f)

    images = [f for f in remaining if all(t == "image" for t in f["types"])]
    texts = [f for f in remaining if "text" in f["types"]]
    _assign_image_roles(images, roles)
    _assign_text_roles(texts, roles)
    for f in remaining:  # e.g. an image + shape group: no rule names it
        roles.setdefault(f["unit_id"], "secondary_image")
    return roles


def _assign_image_roles(images: list[dict], roles: dict[str, str]) -> None:
    for f in images:
        if f["missing_asset"]:
            roles[f["unit_id"]] = "secondary_image"
    logos = [f for f in images if f["unit_id"] not in roles and f["area_pct"] < LOGO_MAX_AREA
             and (f["aspect"] or 0) >= LOGO_MIN_ASPECT and _near_edge(f)]
    if logos:
        roles[max(logos, key=lambda f: (f["z_index"], f["area_pct"]))["unit_id"]] = "logo"
    rest = sorted((f for f in images if f["unit_id"] not in roles), key=lambda f: (-f["area_pct"], f["unit_id"]))
    for i, f in enumerate(rest):
        roles[f["unit_id"]] = "hero_image" if i == 0 else "secondary_image"


def _assign_text_roles(texts: list[dict], roles: dict[str, str]) -> None:
    """Text units here are singles plus generic groups that carry text (an
    icon + caption, say); only a single text without legal wording can be the
    headline."""
    if not texts:
        return
    singles = [f for f in texts if f["kind"] == "single" and not _legal_marked(f)]
    headline = max(singles, key=lambda f: (f["font_size"], bool(f["bold"])), default=None)
    if headline is not None:
        roles[headline["unit_id"]] = "headline"
    head_size = headline["font_size"] if headline is not None else max(f["font_size"] for f in texts)

    pending = [f for f in texts if f is not headline]
    for f in pending:
        if _legal_marked(f):
            roles[f["unit_id"]] = "legal"
    pending = [f for f in pending if f["unit_id"] not in roles]
    smallest = min(pending, key=lambda f: f["font_size"], default=None)
    if smallest is not None and smallest["word_count"] > LEGAL_MIN_WORDS and smallest["center"][1] > LEGAL_MIN_CENTER_Y:
        roles[smallest["unit_id"]] = "legal"

    for f in pending:
        if f["unit_id"] not in roles and f["word_count"] <= CTA_MAX_WORDS and _first_word(f["text"]) in ACTION_VERBS:
            roles[f["unit_id"]] = "cta"

    pending = [f for f in pending if f["unit_id"] not in roles]
    largest = max(pending, key=lambda f: f["font_size"], default=None)
    for f in pending:
        big_enough = f is largest or f["font_size"] >= SUBHEAD_MIN_RATIO * head_size
        roles[f["unit_id"]] = "subheadline" if big_enough else "other_text"


def _prominence(f: dict) -> float:
    if f["text"] is not None:
        return f["font_size"] * f["text_len"]
    return f["area_pct"]


def heuristic_ranking(features: list[dict], roles: dict[str, str]) -> list[str]:
    """Content units (not background or decoration), by role order, then
    prominence, then reading order, then id."""
    content = [f for f in features if roles[f["unit_id"]] in ROLE_ORDER]
    content.sort(key=lambda f: (ROLE_ORDER.index(roles[f["unit_id"]]), -_prominence(f),
                                f["center"][1], f["center"][0], f["unit_id"]))
    return [f["unit_id"] for f in content]
