"""CHAMELEON_STRATEGIES=name,name: switch layout strategies on by name
(modules in engine/pipeline/layout/strategies/). A strategy exposes
`candidates(source, target, plan, measurer)` (extra layouts for the grader
to choose from) and/or `refine(source, target, plan, measurer, best)` (a
possibly better version of the winner). Unset = nothing runs. A missing or
crashing strategy never stops the engine: one `[distill]` line, then skipped.
"""

import importlib
import os
import re
import sys

_modules: dict = {}
_NAME = re.compile(r"[a-z_][a-z0-9_]*")


def enabled() -> list[str]:
    raw = os.environ.get("CHAMELEON_STRATEGIES", "")
    return [n.strip() for n in raw.split(",") if n.strip()]


def _warn(msg: str) -> None:
    print(f"[distill] strategy {msg}", file=sys.stderr)


def _load(name: str):
    if name not in _modules:
        mod = None
        if not _NAME.fullmatch(name):
            _warn(f"{name!r}: not a valid name; skipped")
        else:
            try:
                mod = importlib.import_module(f"engine.pipeline.layout.strategies.{name}")
            except Exception as e:
                _warn(f"{name!r}: could not load ({type(e).__name__}: {e}); skipped")
        _modules[name] = mod
    return _modules[name]


def extra_candidates(source, target, plan, measurer) -> list:
    out = []
    for name in enabled():
        mod = _load(name)
        if mod is None or not hasattr(mod, "candidates"):
            continue
        try:
            out += list(mod.candidates(source, target, plan, measurer) or [])
        except Exception as e:
            _warn(f"{name!r}: candidates failed ({type(e).__name__}: {e})")
    return out


def refine(source, target, plan, measurer, best):
    """Each enabled refiner in turn; a refinement is kept only if it passes every hard gate."""
    from engine.pipeline.grading.gates import check_gates
    for name in enabled():
        mod = _load(name)
        if mod is None or not hasattr(mod, "refine"):
            continue
        try:
            new = mod.refine(source, target, plan, measurer, best)
            if new is not None and not check_gates(source, new.design, target, plan, measurer):
                best = new
        except Exception as e:
            _warn(f"{name!r}: refine failed ({type(e).__name__}: {e})")
    return best
