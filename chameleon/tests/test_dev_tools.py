import csv
import json
from pathlib import Path
import pytest
from dev.run_matrix import run_matrix


def test_run_matrix_smoke(tmp_path):
    inp = [Path("input/chameleon_sample_level1_warmup.json")]
    tg = [{"name": "leaderboard", "width": 728, "height": 90}, {"name": "story", "width": 1080, "height": 1920}]
    winners = run_matrix(inp, tg, tmp_path)
    assert set(winners["chameleon_sample_level1_warmup"]) == {"leaderboard", "story"}
    assert (tmp_path / "chameleon_sample_level1_warmup_sheet.png").exists()
    assert json.loads((tmp_path / "winners.json").read_text()) == winners
    # beyond the brief: one render per target, and the winners are real candidate ids
    assert {p.name for p in (tmp_path / "chameleon_sample_level1_warmup").iterdir()} == {"leaderboard.png", "story.png"}
    ids = winners["chameleon_sample_level1_warmup"].values()
    assert all(i == "fallback" or i == "reflow" or "/" in i for i in ids)


def test_bench_heuristic_only(tmp_path):
    from dev.bench_priority import bench, BENCH_COLUMNS
    rows = bench(["heuristic"], [Path("input/chameleon_sample_level2_core.json")], runs=2, out_csv=tmp_path / "b.csv")
    assert len(rows) == 2 and rows[0]["source"] == "heuristic" and rows[0]["top3_recall"] == 1.0
    assert (tmp_path / "b.csv").read_text().splitlines()[0].split(",") == BENCH_COLUMNS


# --- beyond the brief: the LLM path, without network ---

L2_GOLDEN_ANSWER = {"summary": "s", "roles": [
    {"unit_id": u, "role": r} for u, r in [("deco_circle_1", "decoration"), ("deco_circle_2", "decoration"),
    ("logo", "logo"), ("product", "hero_image"), ("headline", "headline"), ("subhead", "subheadline"), ("badge", "badge"), ("cta", "cta")]],
    "ranking": ["headline", "product", "cta", "logo", "subhead", "badge"]}


class FakeProvider:
    name, model = "fake", "fake-1"
    def __init__(self): self.calls = 0
    def complete(self, system, user, schema):
        from engine.pipeline.priority.providers import LLMResult
        from engine.pipeline.types import LLMCallStats
        self.calls += 1
        return LLMResult(L2_GOLDEN_ANSWER, LLMCallStats("fake", "fake-1", latency_s=0.5, input_tokens=2000, output_tokens=500, cost_usd=0.02))


def test_bench_records_llm_runs_and_summarizes(tmp_path):
    from dev.bench_priority import bench, summarize
    rows = bench([FakeProvider(), "heuristic"], [Path("input/chameleon_sample_level2_core.json")], runs=2, out_csv=tmp_path / "b.csv")
    fake = [r for r in rows if r["provider"] == "fake"]
    assert len(rows) == 4 and all(r["source"] == "llm" and r["role_accuracy"] == 1.0 for r in fake)
    assert (fake[0]["input_tokens"], fake[0]["output_tokens"], fake[0]["cost_usd"]) == (2000, 500, 0.02)
    summary = {(s["provider"], s["model"]): s for s in summarize(rows)}
    assert summary[("fake", "fake-1")]["projected_cost_per_1000"] == 20.0
    assert summary[("fake", "fake-1")]["fallback_rate"] == 0.0 and summary[("heuristic", "-")]["cost_usd"] == 0.0


def test_bench_never_uses_the_cache(tmp_path, monkeypatch):
    from dev.bench_priority import bench
    import os
    monkeypatch.setenv("CHAMELEON_LLM_CACHE", str(tmp_path / "cache"))
    fake = FakeProvider()
    bench([fake], [Path("input/chameleon_sample_level2_core.json")], runs=3, out_csv=tmp_path / "b.csv")
    assert fake.calls == 3 and os.environ["CHAMELEON_LLM_CACHE"] == str(tmp_path / "cache")


def test_cli_skips_providers_without_keys(tmp_path, monkeypatch, capsys):
    from dev.bench_priority import main
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    out = tmp_path / "b.csv"
    main(["--runs", "1", "--samples", "input/chameleon_sample_level1_warmup.json", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "skipping claude" in printed and "skipping gemini" in printed and "heuristic" in printed
    assert len(out.read_text().splitlines()) == 2  # header + one heuristic row


def _bench_models(tmp_path, monkeypatch, keys: dict, argv: list[str]) -> dict[str, list[str]]:
    """Run the benchmark CLI offline with only `keys` set; provider name -> models in its CSV rows."""
    from dev.bench_priority import main
    from engine.pipeline.priority.providers import ClaudeProvider, GeminiProvider, ProviderError
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in keys.items():
        monkeypatch.setenv(k, v)
    def offline(self, *args):
        raise ProviderError("offline")  # plan() falls back to the heuristic; the row still names the model
    for cls in (ClaudeProvider, GeminiProvider):
        monkeypatch.setattr(cls, "complete", offline)
    out = tmp_path / "b.csv"
    main(["--runs", "1", "--samples", "input/chameleon_sample_level1_warmup.json", "--out", str(out), *argv])
    models: dict[str, list[str]] = {}
    for row in csv.DictReader(out.read_text(encoding="utf-8").splitlines()):
        models.setdefault(row["provider"], []).append(row["model"])
    return models


@pytest.mark.parametrize("key, var, model, name", [
    ("ANTHROPIC_API_KEY", "CHAMELEON_CLAUDE_MODEL", "claude-sonnet-5-5", "claude"),
    ("GEMINI_API_KEY", "CHAMELEON_GEMINI_MODEL", "gemini-3.5-flash-lite", "gemini"),
])
def test_cli_benchmarks_the_model_named_in_the_env(tmp_path, monkeypatch, key, var, model, name):
    models = _bench_models(tmp_path, monkeypatch, {key: "test-key", var: model}, [])
    assert models[name] == [model]


def test_cli_model_flags_override_the_env(tmp_path, monkeypatch):
    keys = {"ANTHROPIC_API_KEY": "test-key", "CHAMELEON_CLAUDE_MODEL": "claude-sonnet-5-5"}
    models = _bench_models(tmp_path, monkeypatch, keys, ["--claude-models", "claude-haiku-5-5", "claude-opus-5-5"])
    assert models["claude"] == ["claude-haiku-5-5", "claude-opus-5-5"]
