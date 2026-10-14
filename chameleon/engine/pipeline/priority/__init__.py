"""Node 1: label each unit with a role and rank the content units
(docs/pipeline/01-priority-node.md)."""

from dataclasses import replace
from pathlib import Path

from engine.pipeline.priority.features import unit_features
from engine.pipeline.priority.heuristic import heuristic_ranking, heuristic_roles
from engine.pipeline.priority.units import build_units
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


def heuristic_plan(source: dict, assets_dir: str | Path = "brand_kit/assets") -> PriorityPlan:
    units = build_units(source, assets_dir)
    features = unit_features(source, units)
    roles = heuristic_roles(features)
    return _assemble(units, roles, heuristic_ranking(features, roles), "heuristic", None)
