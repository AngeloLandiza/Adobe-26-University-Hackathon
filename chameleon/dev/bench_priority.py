"""Compare node 1's providers on the golden set: ranking quality, latency and cost.

    python -m dev.bench_priority [--claude-models claude-opus-5-5 ...] [--gemini-models gemini-3.8-flash ...]
                                 [--runs 3] [--samples input/a.json ...]
                                 [--results-dir dev/bench_results | --out some/file.csv]

Without --claude-models / --gemini-models, each provider uses the model the
engine would: CHAMELEON_CLAUDE_MODEL / CHAMELEON_GEMINI_MODEL from the
environment or chameleon/.env, else the built-in default.

Every run is kept: it gets a new timestamped folder under --results-dir with
runs.csv (one row per provider x sample x run, including the roles and ranking
the plan ended up with) and summary.csv (the table printed at the end). Rows
are written as they finish, so a run that dies midway keeps what it measured.
--out writes the rows to one file instead, with no folder or summary.

The heuristic always runs as the baseline. An LLM provider is skipped (with a
note) when its API key isn't set. The response cache is switched off for the
run, so every call is real and its latency and cost are measured.
"""

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

from dev.priority_metrics import kendall_tau, role_accuracy, top3_recall
from engine.pipeline.priority import plan
from engine.pipeline.priority.providers import ClaudeProvider, GeminiProvider

BENCH_COLUMNS = ["provider", "model", "sample", "run", "source", "latency_s", "input_tokens", "output_tokens",
                 "cost_usd", "role_accuracy", "kendall_tau", "top3_recall", "repaired_fraction", "error",
                 "roles", "ranking"]
NUMERIC = ["latency_s", "input_tokens", "output_tokens", "cost_usd", "role_accuracy", "kendall_tau", "top3_recall",
           "repaired_fraction"]
GOLDEN_DIR = Path("dev/golden")
RESULTS_DIR = Path("dev/bench_results")


def _row(provider: str, model: str, sample: str, run: int, result, latency: float, gold: dict | None) -> dict:
    stats = result.stats
    roles = {u.unit_id: u.role for u in result.units}
    ranking = [u.unit_id for u in result.ranked_units()]
    nan = float("nan")
    return {
        "provider": provider, "model": model, "sample": sample, "run": run, "source": result.source,
        "latency_s": latency,
        "input_tokens": stats.input_tokens if stats else 0,
        "output_tokens": stats.output_tokens if stats else 0,
        "cost_usd": stats.cost_usd if stats else 0.0,
        "role_accuracy": role_accuracy(roles, gold["roles"]) if gold else nan,
        "kendall_tau": kendall_tau(ranking, gold["ranking"]) if gold else nan,
        "top3_recall": top3_recall(ranking, gold["ranking"]) if gold else nan,
        "repaired_fraction": stats.repaired_fraction if stats else 0.0,
        "error": (stats.error or "") if stats else "",
        # What the plan ended up with: the LLM's answer after repair, or the heuristic's if it fell back (see source).
        "roles": json.dumps(roles, sort_keys=True),
        "ranking": json.dumps(ranking),
    }


def bench(providers: list, samples: list[Path], runs: int, out_csv: Path) -> list[dict]:
    """One row per provider x sample x run; `providers` holds provider
    objects and/or "heuristic". Latency is wall-clock around plan(). Each row
    is written to `out_csv` as soon as it's measured."""
    saved_cache = os.environ.pop("CHAMELEON_LLM_CACHE", None)
    rows = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BENCH_COLUMNS)
            writer.writeheader()
            for provider in providers:
                name, model = ("heuristic", "-") if provider == "heuristic" else (provider.name, provider.model)
                for path in samples:
                    source = json.loads(Path(path).read_text())["source"]
                    gold_path = GOLDEN_DIR / f"{Path(path).stem}.json"
                    gold = json.loads(gold_path.read_text()) if gold_path.exists() else None
                    for run in range(runs):
                        start = time.perf_counter()
                        result = plan(source, provider=provider)
                        row = _row(name, model, Path(path).stem, run, result, time.perf_counter() - start, gold)
                        rows.append(row)
                        writer.writerow(row)
                        f.flush()
    finally:
        if saved_cache is not None:
            os.environ["CHAMELEON_LLM_CACHE"] = saved_cache
    return rows


def new_run_dir(root: Path) -> Path:
    """A fresh folder named after the current local time; a second run in the same second gets _2, _3..."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path, n = root / stamp, 1
    while path.exists():
        n += 1
        path = root / f"{stamp}_{n}"
    path.mkdir(parents=True)
    return path


def write_summary(summary: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["provider", "model", "runs"])
        writer.writeheader()
        writer.writerows(summary)


def _mean(values: list[float]) -> float:
    values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else float("nan")


def summarize(rows: list[dict]) -> list[dict]:
    """Per provider x model: the mean of each numeric column, the projected
    cost of 1,000 tasks, and the fallback rate (share of LLM runs that ended
    on the heuristic anyway: errors, refusals, too many repairs)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["provider"], row["model"]), []).append(row)
    summary = []
    for (provider, model), group in groups.items():
        entry = {"provider": provider, "model": model, "runs": len(group)}
        entry.update({col: _mean([r[col] for r in group]) for col in NUMERIC})
        entry["projected_cost_per_1000"] = round(entry["cost_usd"] * 1000, 6)
        entry["fallback_rate"] = None if provider == "heuristic" else (
            sum(r["source"] != "llm" for r in group) / len(group))
        summary.append(entry)
    return summary


def print_summary(summary: list[dict]) -> None:
    header = f"{'provider':10} {'model':24} {'runs':>4} {'latency':>8} {'in tok':>7} {'out tok':>7} {'$/task':>8} " \
             f"{'$/1000':>8} {'roles':>6} {'tau':>6} {'top3':>6} {'repair':>6} {'fallbk':>6}"
    print(header)
    print("-" * len(header))
    for s in summary:
        fallback = "-" if s["fallback_rate"] is None else f"{s['fallback_rate']:.2f}"
        print(f"{s['provider']:10} {s['model']:24} {s['runs']:>4} {s['latency_s']:>7.2f}s {s['input_tokens']:>7.0f} "
              f"{s['output_tokens']:>7.0f} {s['cost_usd']:>8.4f} {s['projected_cost_per_1000']:>8.2f} "
              f"{s['role_accuracy']:>6.2f} {s['kendall_tau']:>6.2f} {s['top3_recall']:>6.2f} "
              f"{s['repaired_fraction']:>6.2f} {fallback:>6}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Default None runs one provider with no model given, so it reads the env (or .env) like the engine does.
    parser.add_argument("--claude-models", nargs="*", default=[None])
    parser.add_argument("--gemini-models", nargs="*", default=[None])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--samples", type=Path, nargs="*", default=sorted(Path("input").glob("*.json")))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=None, help="write the rows to this one file instead of a new folder")
    args = parser.parse_args(argv)

    providers: list = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers += [ClaudeProvider(model=m) for m in args.claude_models]
    else:
        print("skipping claude: ANTHROPIC_API_KEY is not set")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        providers += [GeminiProvider(model=m) for m in args.gemini_models]
    else:
        print("skipping gemini: GEMINI_API_KEY / GOOGLE_API_KEY is not set")
    providers.append("heuristic")

    run_dir = new_run_dir(args.results_dir) if args.out is None else None
    out = args.out or run_dir / "runs.csv"
    rows = bench(providers, args.samples, args.runs, out)
    summary = summarize(rows)
    print_summary(summary)
    if run_dir is not None:
        write_summary(summary, run_dir / "summary.csv")
    print(f"\n{len(rows)} rows written to {out}")


if __name__ == "__main__":
    main()
