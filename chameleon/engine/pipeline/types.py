"""Data passed between the three nodes (see docs/pipeline/README.md §4).

Node 1 produces a `PriorityPlan`, node 2 turns it into `Candidate`s, and
node 3 grades each candidate into a `GradeReport`.
"""

from dataclasses import dataclass, field

ROLES: tuple[str, ...] = (
    "background",
    "decoration",
    "headline",
    "subheadline",
    "cta",
    "logo",
    "hero_image",
    "badge",
    "secondary_image",
    "legal",
    "other_text",
)

# Exponent for a unit's priority weight: weight = rank ** -ALPHA.
ALPHA = 0.7


@dataclass
class Unit:
    """Elements that move as one piece. `kind` is single | container_label | group."""

    unit_id: str
    element_ids: list[str]
    kind: str
    role: str | None = None
    rank: int | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class LLMCallStats:
    provider: str
    model: str
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    repaired_fraction: float = 0.0
    cached: bool = False
    error: str | None = None


@dataclass
class PriorityPlan:
    """Node 1's output. `source` is "llm" or "heuristic"; ranks are 1..n over content units."""

    source: str
    units: list[Unit]
    stats: LLMCallStats | None = None

    def ranked_units(self) -> list[Unit]:
        return sorted((u for u in self.units if u.rank is not None), key=lambda u: u.rank)

    def background_unit_ids(self) -> list[str]:
        return [u.unit_id for u in self.units if u.role == "background"]

    def decoration_unit_ids(self) -> list[str]:
        return [u.unit_id for u in self.units if u.role == "decoration"]

    def unit_for_element(self, element_id: str) -> Unit:
        for unit in self.units:
            if element_id in unit.element_ids:
                return unit
        raise KeyError(element_id)

    def weight(self, unit_id: str) -> float:
        for unit in self.units:
            if unit.unit_id == unit_id:
                return 0.0 if unit.rank is None else unit.rank ** -ALPHA
        raise KeyError(unit_id)


@dataclass
class Candidate:
    """One full layout for the target canvas, produced by node 2."""

    candidate_id: str
    design: dict
    template: str
    params: dict
    degraded_unit_ids: list[str] = field(default_factory=list)
    is_fallback: bool = False


@dataclass
class GradeReport:
    candidate_id: str
    passed_gates: bool
    failed_gates: list[str]
    metrics: dict[str, float]
    rendered: bool
    total: float
