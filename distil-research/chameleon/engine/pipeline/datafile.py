"""Shared loader for the optional data files (weights.json, extra_templates.json).

A data file is never required and never allowed to stop the engine. Whatever
goes wrong (missing file, unreadable file, bad JSON, a value the parser
rejects, even a RecursionError or OverflowError), the caller gets its default
and one `[distill]` line goes to stderr. The outcome is cached per version of
the file, so the line prints once per version, not once per task.
"""

import os
import sys
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


def load_data_file(path: Path | None, env_var: str, default_path: Path, parse: Callable[[str], T],
                   default: T, label: str, cache: dict) -> T:
    """`parse(text)` result for the file at `path`, else $env_var, else `default_path`; `default` on any failure.

    A missing file at `default_path` is the normal state and is silent. A missing
    file that was asked for (`path` or the env var) is probably a typo, so it
    prints one line. Cached in `cache` by (resolved path, mtime, size)."""
    explicit = path or os.environ.get(env_var)
    chosen = Path(explicit) if explicit else Path(default_path)
    key = None
    try:
        try:
            stat = chosen.stat()
            key = (str(chosen.resolve()), stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            key = (str(chosen.resolve()), None, None)
        if key in cache:
            return cache[key]
        if key[1] is None:
            if explicit:
                print(f"[distill] {label} file {chosen} not found; using defaults", file=sys.stderr)
            result = default
        else:
            result = parse(chosen.read_text())
    except Exception as err:  # deliberately broad: nothing about a data file may raise
        raw_key = ("unresolved", str(chosen))
        if key is None and raw_key in cache:  # e.g. a symlink loop, already reported
            return cache[raw_key]
        print(f"[distill] ignoring {label} {chosen}: {err!r}; using defaults", file=sys.stderr)
        result = default
        key = key or raw_key
    cache[key] = result
    return result
