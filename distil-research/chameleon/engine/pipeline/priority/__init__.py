"""Node 1: label each unit with a role and rank the content units
(docs/pipeline/01-priority-node.md)."""

import time
from dataclasses import replace
from pathlib import Path

from engine.pipeline.priority.features import unit_features
from engine.pipeline.priority.heuristic import heuristic_ranking, heuristic_roles
from engine.pipeline.priority.prompt import SYSTEM_PROMPT, build_user_message, response_schema
from engine.pipeline.priority.providers import cached_complete, get_provider
from engine.pipeline.priority.units import build_units
from engine.pipeline.priority.validate import MAX_REPAIR_FRACTION, repair
from engine.pipeline.types import LLMCallStats, PriorityPlan, Unit

_LAYER_ROLES = ("background", "decoration")


def _assemble(units: list[Unit], roles: dict[str, str], ranking: list[str], label: str,
              stats: LLMCallStats | None) -> PriorityPlan:
    """Build the plan from roles and a ranking without touching `units`.
    Hard signals win: a background unit stays background, and units whose
    asset is missing go to the end. Content units the ranking left out are
    appended; ranks are renumbered 1..n."""
    planned = [
        replace(u, role="background" if "background" in u.flags else roles.get(u.unit_id, u.role),
                rank=None, flags=list(u.flags))
        for u in units
    ]
    by_id = {u.unit_id: u for u in planned}
    content = [u.unit_id for u in planned if u.role not in _LAYER_ROLES]
    content_set = set(content)
    order = [uid for uid in dict.fromkeys(ranking) if uid in content_set]
    order += [uid for uid in content if uid not in order]
    order = ([uid for uid in order if "missing_asset" not in by_id[uid].flags]
             + [uid for uid in order if "missing_asset" in by_id[uid].flags])
    for rank, uid in enumerate(order, start=1):
        by_id[uid].rank = rank
    return PriorityPlan(label, planned, stats)


def _heuristic_parts(source: dict, assets_dir: str | Path):
    units = build_units(source, assets_dir)
    features = unit_features(source, units)
    roles = heuristic_roles(features)
    return units, features, roles, heuristic_ranking(features, roles)


def heuristic_plan(source: dict, assets_dir: str | Path | None = None) -> PriorityPlan:
    units, _, roles, ranking = _heuristic_parts(source, assets_dir)
    return _assemble(units, roles, ranking, "heuristic", None)


def _bare_plan(source, exc: Exception) -> PriorityPlan:
    """Last resort when the design can't even be analysed: one unranked unit
    per element, so every id is still accounted for."""
    elements = source.get("elements") or [] if isinstance(source, dict) else []
    units = [Unit(str(el.get("id")), [el.get("id")], "single") for el in elements if isinstance(el, dict)]
    return PriorityPlan("heuristic", units, LLMCallStats("none", "-", error=f"could not analyse design: {exc!r}"))


def plan(source: dict, assets_dir: str | Path | None = None, provider=None) -> PriorityPlan:
    """Node 1's entry point; never raises. `provider` is a provider object or
    a name for `get_provider` (None follows env CHAMELEON_LLM). Without a
    provider, or when the LLM fails or its answer needs too many repairs,
    the heuristic plan is returned and `stats.error` says why."""
    try:
        units, features, heur_roles, heur_ranking = _heuristic_parts(source, assets_dir)
    except Exception as exc:
        return _bare_plan(source, exc)

    llm = None
    start = time.perf_counter()
    try:
        llm = get_provider(provider) if provider is None or isinstance(provider, str) else provider
        if llm is None:
            return _assemble(units, heur_roles, heur_ranking, "heuristic", None)
        result = cached_complete(llm, SYSTEM_PROMPT, build_user_message(features),
                                 response_schema([u.unit_id for u in units]))
        roles, ranking, fraction = repair(result.data, units, heur_roles, heur_ranking)
    except Exception as exc:
        who = llm if llm is not None else provider
        name = getattr(who, "name", None) or (who if isinstance(who, str) else "none")
        stats = LLMCallStats(name, getattr(who, "model", "-"), latency_s=time.perf_counter() - start,
                             error=str(exc) or type(exc).__name__)
        return _assemble(units, heur_roles, heur_ranking, "heuristic", stats)

    stats = result.stats
    stats.repaired_fraction = fraction
    if fraction > MAX_REPAIR_FRACTION:
        stats.error = "too_many_repairs"
        return _assemble(units, heur_roles, heur_ranking, "heuristic", stats)
    return _assemble(units, roles, ranking, "llm", stats)
