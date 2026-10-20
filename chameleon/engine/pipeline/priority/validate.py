"""Check and repair an LLM's roles and ranking against the design's units
(docs/pipeline/01-priority-node.md §5). Gaps are filled from the heuristic;
the returned fraction says how much had to be fixed."""

from engine.pipeline.types import ROLES, Unit

MAX_REPAIR_FRACTION = 0.3
_LAYER_ROLES = ("background", "decoration")
_IMAGE_ROLES = ("hero_image", "logo", "secondary_image")


def repair(raw: dict, units: list[Unit], heur_roles: dict[str, str],
           heur_ranking: list[str]) -> tuple[dict[str, str], list[str], float]:
    """Return (roles, ranking, repaired fraction).

    Background units are a hard signal, not the LLM's call: they stay
    background, the LLM may skip them, and the LLM may not label any other
    unit "background" (node 2 would stretch it over the canvas). Nor may it
    call text or an image "decoration" (node 2 would park it in a corner at
    a fraction of its size): only all-shape units, which the heuristic always
    labels decoration, can be. The LLM still meant "least important", so such
    a unit becomes secondary_image if the heuristic sees an image, else
    other_text, and goes last unless the LLM ranked it. The fraction is the
    share of units involved in any fix; ids the design doesn't have are
    dropped without counting."""
    unit_ids = [u.unit_id for u in units]
    known = set(unit_ids)
    pinned = {u.unit_id for u in units if "background" in u.flags}
    fixed: set[str] = set()

    roles: dict[str, str] = {}
    demoted: set[str] = set()
    seen: set[str] = set()
    for item in raw.get("roles") or []:
        if not isinstance(item, dict) or item.get("unit_id") not in known:
            continue
        uid, role = item["unit_id"], item.get("role")
        if uid in seen:
            if uid not in pinned:
                fixed.add(uid)
            continue
        seen.add(uid)
        if uid in pinned:
            continue
        if role == "decoration" and heur_roles.get(uid) != "decoration":
            role = "secondary_image" if heur_roles.get(uid) in _IMAGE_ROLES else "other_text"
            demoted.add(uid)
            fixed.add(uid)
        if role in ROLES and role != "background":
            roles[uid] = role
    for uid in unit_ids:
        if uid in pinned:
            roles[uid] = "background"
        elif uid not in roles:
            roles[uid] = heur_roles[uid]
            fixed.add(uid)

    content = [uid for uid in unit_ids if roles[uid] not in _LAYER_ROLES]
    content_set = set(content)
    ranking: list[str] = []
    for uid in raw.get("ranking") or []:
        if uid not in known or uid in pinned:
            continue
        if uid not in content_set or uid in ranking:
            fixed.add(uid)
            continue
        ranking.append(uid)

    missing = ([uid for uid in heur_ranking if uid in content_set and uid not in ranking]
               + [uid for uid in content if uid not in heur_ranking and uid not in ranking])
    for uid in missing:
        fixed.add(uid)
        if uid in demoted:
            continue
        if uid not in heur_ranking:
            ranking.append(uid)
            continue
        before = heur_ranking[:heur_ranking.index(uid)]
        anchor = next((p for p in reversed(before) if p in ranking), None)
        ranking.insert(ranking.index(anchor) + 1 if anchor is not None else 0, uid)
    ranking += [uid for uid in missing if uid in demoted]

    return roles, ranking, len(fixed) / len(units) if units else 0.0
