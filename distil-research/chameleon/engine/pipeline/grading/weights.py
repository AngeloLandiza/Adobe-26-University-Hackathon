"""Grader weights, optionally loaded from a data file (weights.json).

The hand-set constants stay the defaults. A learned weights file can replace
them, but a missing or bad file must never stop the engine: it falls back to
the defaults and prints one `[distill]` line to stderr.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from engine.pipeline.datafile import load_data_file
from engine.pipeline.grading.metrics import WEIGHTS
from engine.pipeline.grading.pixels import PIXEL_WEIGHTS

WEIGHTS_PATH = Path(__file__).with_name("weights.json")
ENV_VAR = "CHAMELEON_GRADER_WEIGHTS"
SUM_TOL = 1e-6


@dataclass(frozen=True)
class GraderWeights:
    geometry: dict[str, float]
    pixels: dict[str, float]
    geometry_share: float  # total = share x geometry + (1 - share) x pixels


DEFAULT_WEIGHTS = GraderWeights(dict(WEIGHTS), dict(PIXEL_WEIGHTS), 0.7)

_cache: dict[tuple, GraderWeights] = {}


def _group(raw, keys, name: str) -> dict[str, float]:
    """One weight group: exactly the expected keys, each finite and >= 0, summing to 1."""
    if not isinstance(raw, dict) or set(raw) != set(keys):
        raise ValueError(f"{name} must have exactly the keys {sorted(keys)}")
    out = {}
    for key in keys:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name}.{key} must be a finite number >= 0")
        out[key] = float(value)
    if abs(sum(out.values()) - 1.0) > SUM_TOL:
        raise ValueError(f"{name} weights must sum to 1")
    return out


def _parse(text: str) -> GraderWeights:
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("top level must be an object")
    share = raw.get("geometry_share")
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share <= 1:
        raise ValueError("geometry_share must be a number in [0, 1]")
    return GraderWeights(_group(raw.get("geometry"), WEIGHTS, "geometry"),
                         _group(raw.get("pixels"), PIXEL_WEIGHTS, "pixels"), float(share))


def load_weights(path: Path | None = None) -> GraderWeights:
    """Weights from `path`, else $CHAMELEON_GRADER_WEIGHTS, else WEIGHTS_PATH; the
    defaults if the file is missing or bad (see engine.pipeline.datafile)."""
    return load_data_file(path, ENV_VAR, WEIGHTS_PATH, _parse, DEFAULT_WEIGHTS, "grader weights", _cache)
