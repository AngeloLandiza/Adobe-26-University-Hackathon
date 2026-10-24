# Offline Node 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure three offline ways to label and rank a design's units (better rules, a small model trained on LLM labels, a local LLM) on 100 held-out real ads, then make the winner node 1's default path in the no-network scored run.

**Architecture:**
- **Data.** The friend's converter (`angelo/distill`) turns Crello ad templates into our schema, in a gitignored `chameleon/dev/realads/`. Ids are anonymised before any model sees an ad. Hosted LLMs label the ads at dev time.
- **Candidates.** Rules v2 live in `heuristic.py`. The trained model is evaluated with numpy alone, from a committed weights JSON, behind a `learned` path in `plan()`. The local LLM is a dev-only provider class used by the benchmark.
- **Decision.** A paired-bootstrap comparison on the held-out ads decides. Only then does `CHAMELEON_LLM=auto` change.

**Tech Stack:**
- Runtime: Python 3.11 in the image, numpy 2.4.6.
- Dev only: scikit-learn, jsonschema, huggingface_hub, pyarrow, llama-cpp-python. The friend's converter runs from a git worktree.

**Spec:** `docs/superpowers/specs/2026-10-23-offline-priority-node-design.md`

All commands run from `chameleon/` unless a step says otherwise. Python is `.venv/Scripts/python` (Windows; `.venv/bin/python` elsewhere).

## Global Constraints

- **Runtime:** no network, no run-time downloads, no keys in the image. `requirements.txt` gains nothing; dev tools go in `requirements-research.txt`.
- **Engine code** must run on Python 3.11 (the image); the dev venv is 3.12. The learned model is evaluated with numpy only.
- **`plan()` never raises.** The rules remain the last fallback; `stats.error` says why a path was abandoned.
- **Nothing containing Crello content is committed.**
  - `chameleon/dev/realads/` is gitignored.
  - Benchmark runs on real ads write per-ad rows under `dev/realads/bench_results/`; only aggregate `summary.csv` files are copied to `dev/bench_results/`.
- **Held-out discipline:**
  - Held-out = challenge2 + challenge3 (100 real ads).
  - Rules v2, model selection and hyperparameters use only `train` / `val`.
  - Each candidate's held-out score is read once, in Task 14. No per-ad held-out output is inspected before then.
- **Labelling:**
  - Element and group ids are anonymised (`e1…`, `g1…`) before any hosted or local model labels a real or synthetic ad.
  - Teacher: `gemini-4-argon`. Gold voters: `claude-sonnet-5-5`, `claude-opus-5-5`, `gemini-3.8-flash`, in that order for ties.
  - Total API spend < $15.
- **Official-sample floor for any winner:** role accuracy 1.00 on all three, and Kendall τ ≥ 1.00 (L1), 1.00 (L2), 0.93 (L3) against `dev/golden`.
- **Decision rule:**
  1. B ships if it beats A by ≥ 0.03 τ with a paired bootstrap 95% CI excluding 0; otherwise A ships.
  2. C replaces the winner only if it beats it by ≥ 0.05 τ (CI excludes 0), with median latency ≤ 30 s per ad in the amd64 container on the Mac, and ≤ 3 GB added to the image.
  3. Every winner holds the floor.
- **Downloads** (Crello shards, GGUF weights) happen only after the user confirms file names and sizes.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The branch is `offline-priority`.

## Review Focus

1. **A missing-asset image under the learned path** (L3's `holiday_watermark`) must keep role `secondary_image` and rank last, as the rules do. The test is in Task 12.
2. **A design with no content units** (only a background and decorations) must give a learned plan with no ranked units, not an exception. The test is in Task 12.
3. **Text with missing optional fields** (no `opacity`, `z_index` or `font_weight`, empty `content`) must vectorise without NaN or exceptions. The test is in Task 9.
4. **A weights file from another feature version** must fail to load and fall back to the rules with a `stats.error` naming the feature mismatch. The test is in Task 12.
5. **A group id that equals an element id outside the group** must anonymise to distinct ids with the same unit partition. The test is in Task 2.

---

### Task 1: Guard the provider SDK imports

A broken `anthropic` or `google-genai` install must not stop node 1's offline path from loading (deferred review item).

**Files:**
- Modify: `engine/pipeline/priority/providers.py` (module-level SDK imports, `ClaudeProvider._get_client`, `GeminiProvider._get_client`)
- Test: `tests/test_providers.py`

**Interfaces:**
- Produces: module globals `anthropic`, `genai`, `types` that are `None` when their import fails. `_get_client()` raises `ProviderError("<provider>: SDK unavailable")` in that case.

- [ ] **Step 1: Write the failing test**

```python
def test_plan_survives_a_broken_sdk_install(monkeypatch, l2):
    import importlib, sys
    import engine.pipeline.priority.providers as providers
    from engine.pipeline.priority import plan
    monkeypatch.setitem(sys.modules, "anthropic", None)      # `import anthropic` now raises
    monkeypatch.setitem(sys.modules, "google.genai", None)
    try:
        reloaded = importlib.reload(providers)
        assert reloaded.anthropic is None and reloaded.genai is None
        for cls in (reloaded.ClaudeProvider, reloaded.GeminiProvider):
            p = plan(l2["source"], provider=cls())
            assert p.source == "heuristic" and "SDK unavailable" in p.stats.error
    finally:
        monkeypatch.undo()
        importlib.reload(providers)
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_providers.py::test_plan_survives_a_broken_sdk_install -q`. Expected: FAIL (ImportError on reload).
- [ ] **Step 3:** Wrap each SDK import in `try: … except Exception: … = None`, and raise the `ProviderError` from `_get_client` when the module is `None`. Existing monkeypatch targets (`providers.anthropic.Anthropic`, …) keep working.
- [ ] **Step 4:** Run `.venv/Scripts/python -m pytest tests/test_providers.py tests/test_plan.py -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `fix(priority): a broken LLM SDK install can't take down node 1`.

---

### Task 2: Id anonymiser, real-ad loader, research requirements

**Files:**
- Create: `dev/node1/__init__.py` (empty), `dev/node1/anon.py`, `dev/node1/data.py`, `requirements-research.txt`
- Modify: `.gitignore` (add `dev/realads/`)
- Test: `tests/test_node1_data.py`

**Interfaces:**
- Produces `anonymize_source(source: dict) -> tuple[dict, dict[str, dict[str, str]]]`:
  - returns a deep copy with element ids `e1…eN` in element order, and group ids `g1…` in order of first appearance;
  - the map is `{"elements": {old: new}, "groups": {old: new}}`;
  - the input is never mutated.
- Produces `data.REALADS = Path("dev/realads")` and `load_manifest(root: Path = REALADS) -> dict`.
- Produces `load_split(name: str, root: Path = REALADS) -> list[dict]`:
  - returns ads `{"task_id", "split", "origin", "source"}` sorted by `task_id`;
  - `"official"` loads `input/*.json` (task_id = file stem) and `"devsamples"` loads `dev/samples/*.json`; every other name comes from `root/manifest.json["splits"][name]` and `root/ads/<task_id>.json`;
  - an unknown name raises `KeyError`.
- Produces `gold_for(task_id: str, gold_dir: Path) -> dict | None`: reads `gold_dir/<task_id>.json` if present.
- `requirements-research.txt` pins exact current versions of `scikit-learn`, `jsonschema`, `huggingface_hub` and `pyarrow` (install with `-r requirements.txt -r requirements-dev.txt -r requirements-research.txt`).

- [ ] **Step 1: Write the failing tests**

```python
import copy, json, subprocess
import pytest
from dev.node1.anon import anonymize_source
from dev.node1 import data
from engine.pipeline.priority.units import build_units
from tests.designs import design, shape, text

def test_anonymize_renames_ids_and_keeps_the_units(l3):
    anon, id_map = anonymize_source(l3["source"])
    assert [e["id"] for e in anon["elements"]] == [f"e{i}" for i in range(1, len(anon["elements"]) + 1)]
    before, after = build_units(l3["source"]), build_units(anon)
    assert [[id_map["elements"][i] for i in u.element_ids] for u in before] == [u.element_ids for u in after]
    assert [(u.kind, u.flags) for u in before] == [(u.kind, u.flags) for u in after]

def test_anonymize_is_deterministic_and_pure(l2):
    original = copy.deepcopy(l2["source"])
    assert anonymize_source(l2["source"]) == anonymize_source(l2["source"])
    assert l2["source"] == original

def test_group_id_that_names_another_element_stays_distinct():
    src = design(text("badge", "New", 10, 10, 100, 30, 20),
                 shape("pill", 500, 500, 200, 60, group_id="badge"),
                 text("label", "Shop now", 510, 510, 180, 40, 24, group_id="badge"))
    anon, id_map = anonymize_source(src)
    assert id_map == {"elements": {"badge": "e1", "pill": "e2", "label": "e3"}, "groups": {"badge": "g1"}}
    assert [u.unit_id for u in build_units(anon)] == ["e1", "g1"]

def test_load_split_reads_the_manifest(tmp_path):
    (tmp_path / "ads").mkdir()
    for tid in ("b", "a"):
        (tmp_path / "ads" / f"{tid}.json").write_text(json.dumps(
            {"task_id": tid, "split": "val", "origin": "crello", "source": {"canvas": {}, "elements": []}}))
    (tmp_path / "manifest.json").write_text(json.dumps({"splits": {"val": ["b", "a"]}}))
    assert [a["task_id"] for a in data.load_split("val", tmp_path)] == ["a", "b"]
    with pytest.raises(KeyError):
        data.load_split("nope", tmp_path)

def test_official_split_is_the_three_samples():
    assert [a["task_id"] for a in data.load_split("official")] == [
        "chameleon_sample_level1_warmup", "chameleon_sample_level2_core", "chameleon_sample_level3_advanced"]

def test_realads_folder_is_gitignored():
    assert subprocess.run(["git", "check-ignore", "-q", "dev/realads/ads/x.json"]).returncode == 0
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_node1_data.py -q`. Expected: FAIL (`dev.node1` missing).
- [ ] **Step 3:** Implement `anon.py`, `data.py`, the `.gitignore` line and `requirements-research.txt`. Install it.
- [ ] **Step 4:** Run `.venv/Scripts/python -m pytest tests/test_node1_data.py -q`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(dev): id anonymiser and real-ad split loader for the node 1 bake-off`.

---

### Task 3: Fetch Crello and build the real-ad sets

**Files:**
- Create: `dev/node1/families.py`, `dev/node1/fetch_crello.py`, `dev/node1/convert_crello.py`
- Test: `tests/test_node1_families.py`

**Interfaces:**
- `families.py` is pure and imports nothing from `engine` or `dev`, because `convert_crello.py` loads it by file path:
  - `signature(texts: list[str]) -> list[str]`: the friend's family key. For each text of ≥ 8 characters after `" ".join(t.lower().split())`, take `sha1(...).hexdigest()[:12]`; return them sorted and unique.
  - `source_signature(source: dict) -> list[str]`: `signature` of the source's text contents.
  - `select_rows(rows: list[dict], excluded: set[str], limit: int, seed: int) -> list[dict]`. Each row has `"crello_id"` and `"sig"`. The function:
    - orders rows by `sha1(f"{seed}:{crello_id}")`;
    - drops rows sharing any sig with `excluded`;
    - keeps at most one row per family (no shared sig with an earlier pick);
    - stops at `limit`.
- `fetch_crello.py` CLI:
  - `--list` prints every `data/*.parquet` file of `cyberagent/crello` at revision `5.1.0`, with its size and the total;
  - `--get test|train --dest DIR [--max-shards N]` downloads with `hf_hub_download`.
- `convert_crello.py` is a standalone script, run with cwd = the friend's `distil-research/chameleon`. It puts that cwd first on `sys.path` and imports `dev.distill.make_real_challenge as mrc`, `dev.distill.make_corpus.measurer` and `dev.distill.corpus.load_challenge2/load_challenge3/load_crello_dev`. For every `train-*.parquet` row group, it:
  1. reads rows with `mrc.COLS` and keeps `mrc.is_ad` rows;
  2. computes `sig = mrc.text_signature(row, F)`;
  3. excludes every sig of the challenge2 / challenge3 / crello_dev templates (crello ids → `challenge2.nosync/scan.jsonl` sigs);
  4. runs `select_rows`, then `mrc.convert(row, F, task_id, assets_dir, tm)` with task ids `crt_{k:04d}_{W}x{H}`, catching `mrc.Skip` and any exception as a skip reason;
  5. writes `--out/ads/<task_id>.json` (with `"split": "train"`), the PNG rasters to `--out/assets/`, and `--out/summary.json` (counts and skip reasons only).

- [ ] **Step 1: Write the failing tests**

```python
from dev.node1.families import select_rows, signature, source_signature

def test_signature_matches_the_friends_rule():
    assert signature(["Shop  the SUMMER line", "Hi"]) == signature(["shop the summer line"])
    assert signature(["short"]) == []

def test_select_rows_drops_excluded_families_and_keeps_one_per_family():
    rows = [{"crello_id": str(i), "sig": [s]} for i, s in enumerate(["a", "b", "b", "c", "x"])]
    picked = select_rows(rows, excluded={"x"}, limit=10, seed=0)
    assert sorted(r["sig"][0] for r in picked) == ["a", "b", "c"]
    assert select_rows(rows, excluded={"x"}, limit=10, seed=0) == picked      # deterministic
    assert len(select_rows(rows, excluded=set(), limit=2, seed=0)) == 2

def test_source_signature_reads_text_elements(l2):
    assert source_signature(l2["source"]) == signature(
        [e["content"] for e in l2["source"]["elements"] if e["type"] == "text"])
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_node1_families.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `families.py`, `fetch_crello.py` and `convert_crello.py`.
- [ ] **Step 4:** Run `.venv/Scripts/python -m pytest tests/test_node1_families.py -q`. Expected: PASS. Commit: `feat(dev): Crello fetch and training-pool converter for node 1`.
Paths below: `WT` = `C:\dev\work\distill-wt` (the friend's worktree), `FRIEND` = `WT\distil-research\chameleon`, `PY` = `C:\dev\work\Adobe-26-University-Hackathon\chameleon\.venv\Scripts\python`.

- [ ] **Step 5: Set up the friend's tree** (from the repo root): `git fetch origin angelo/distill`, then `git worktree add ../distill-wt origin/angelo/distill`. Expected: `FRIEND\dev\distill\make_real_challenge.py` exists.
- [ ] **Step 6: Ask before downloading.** Run `.venv/Scripts/python -m dev.node1.fetch_crello --list`. Show the user the file names and total size, and wait for a yes.
- [ ] **Step 7: Download.**
  - Test shards: `--get test --dest WT\distil-research\training-data\references\crello.nosync\data`.
  - Train shards, enough for ≥ 3,000 ad templates: `--get train --max-shards N --dest WT\distil-research\training-data\references\crello.nosync\train`.
- [ ] **Step 8: Build the held-out and validation sets.** With cwd = `FRIEND`, run `PY -m dev.distill.make_real_challenge`, then `PY -m dev.distill.make_crello_dev`, then `PY -m dev.distill.make_challenge3`.
  - Expected: 50 `cr2_*.json`, 60 `crd_*.json` and 50 `cr3_*.json` in their `.nosync` folders.
  - If `symlink_to` fails on Windows, either enable Developer Mode and rerun, or ask the user to get Angelo's three `.nosync` folders. The converter is deterministic, so they're the same sets.
- [ ] **Step 9: Build the training pool.** With cwd = `FRIEND`, run `PY C:\dev\work\Adobe-26-University-Hackathon\chameleon\dev\node1\convert_crello.py --shards-dir ..\training-data\references\crello.nosync\train --out C:\dev\work\Adobe-26-University-Hackathon\chameleon\dev\realads\raw_train --limit 1500 --seed 0`. Expected: `summary.json` reports ≥ 600 converted ads. If fewer, raise `--max-shards`.

---

### Task 4: Intake into `dev/realads/`

**Files:**
- Create: `dev/node1/intake.py`
- Test: `tests/test_node1_intake.py`

**Interfaces:**
- Consumes: `anonymize_source`, `source_signature`, `data.REALADS`.
- Produces `build(sources: dict[str, list[Path]], asset_dirs: list[Path], out: Path = REALADS, schema_path: Path = Path("chameleon_design_schema_v0.9.json")) -> dict`. `sources` maps split name → design-file folders. It:
  - validates each record's `source` with `jsonschema` (draft 2020-12);
  - anonymises it;
  - writes `out/ads/<task_id>.json` as `{"task_id", "split", "origin", "source"}`;
  - copies every file of `asset_dirs` into `out/assets/`;
  - drops any `train`/`synth_train` ad whose `source_signature` intersects `heldout` or `val`;
  - writes and returns `manifest.json` = `{"splits": {split: [task_ids]}, "rejected": [{"task_id", "reason"}], "dropped_for_family": [task_ids], "coverage": {split: {"ads", "with_group_id", "with_z_index", "with_opacity", "assets_present"}}}`.
- CLI `python -m dev.node1.intake --distill-root C:\dev\work\distill-wt\distil-research --raw-train dev/realads/raw_train` wires the splits:
  - `heldout` = `corpus/challenge2.nosync` + `corpus/challenge3.nosync`;
  - `val` = `corpus/crello_dev.nosync`;
  - `train` = `raw_train/ads`;
  - `synth_train` / `synth_holdout` = `corpus/designs` by each record's `split` field, without task ids `warmup_01`, `core_01` and `advanced_01`, plus `corpus/challenge` in `synth_holdout`;
  - assets = `brand_kit/assets`, `corpus/assets`, each `.nosync/assets` and `raw_train/assets`.

- [ ] **Step 1: Write the failing tests.** Build tiny record files in `tmp_path` from `tests/designs.py` builders.

```python
def test_intake_anonymises_and_splits(tmp_path): ...
    # manifest["splits"] == {"heldout": ["h1"], "val": ["v1"], "train": ["t1"]}
    # every written ad's element ids start with "e"; ad files carry no other keys than task_id/split/origin/source
def test_intake_rejects_schema_invalid_ads(tmp_path): ...
    # a record whose text element lacks "font_family" -> in manifest["rejected"] with a reason naming the field
def test_intake_drops_training_ads_that_share_a_family_with_heldout(tmp_path): ...
    # train ad with the same 8+ char headline as the heldout ad -> manifest["dropped_for_family"] == ["t1"]
def test_intake_copies_assets_and_reports_coverage(tmp_path): ...
    # out/assets/<file> exists; coverage["heldout"]["assets_present"] == 1.0
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_node1_intake.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `intake.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Commit: `feat(dev): real-ad intake with anonymised ids and family-safe splits`.
- [ ] **Step 5: Run it on the real data.** Run the CLI, then `git status --short`. Expected: heldout 100, val 60, train ≥ 600; `git status` shows nothing under `dev/realads/`. Report the coverage to the user. If `with_group_id` or `with_z_index` coverage is below 50% in `heldout`, stop and raise it.

---

### Task 5: Labelling tool (teacher labels, consensus gold, review sheets)

**Files:**
- Create: `dev/node1/label.py`
- Test: `tests/test_node1_label.py`

**Interfaces:**
- Consumes: `data.load_split`, `engine.pipeline.priority.plan`, providers, `engine.pipeline.render.render_to_image(design, assets_dir, max_size)`.
- Produces `label_ads(ads: list[dict], provider, out_dir: Path, assets_dir: Path, workers: int = 8) -> dict`:
  - skips ads whose `out_dir/<task_id>.json` exists;
  - writes `{"roles": {unit_id: role}, "ranking": [unit_id], "model": str, "cost_usd": float, "repaired_fraction": float}` only when `plan(...).source == "llm"`;
  - returns `{"labelled": int, "skipped": int, "failed": [[task_id, error]], "cost_usd": float}`;
  - `CHAMELEON_LLM_CACHE` defaults to `dev/realads/.cache`.
- Produces `consensus(votes: list[dict]) -> dict`:
  - `votes` are label dicts in voter order;
  - role = majority over the voters that labelled the unit, with ties going to the earliest voter;
  - content units are those whose consensus role isn't background or decoration;
  - each unit's position is its index in each voter's ranking (`len(ranking)` when missing);
  - the ranking sorts by (mean position, unit_id);
  - returns `{"roles", "ranking", "voters": [model names], "reviewed": False}`.
- Produces `review_sheet(ad: dict, gold: dict, assets_dir: Path, out_png: Path) -> None`: the render, with each unit's box outlined and labelled `"<rank>. <role>"`.
- CLI:
  - `python -m dev.node1.label teacher --provider gemini --model gemini-4-argon --split train val synth_train synth_holdout` writes to `dev/realads/labels/<model>/`;
  - `python -m dev.node1.label vote --provider claude --model claude-sonnet-5-5 --split heldout`, likewise for each gold voter;
  - `python -m dev.node1.label gold --split heldout` writes `dev/realads/gold/` from the three voter dirs;
  - `python -m dev.node1.label review --split heldout` writes `dev/realads/review/<task_id>.png`.

- [ ] **Step 1: Write the failing tests** (use a fake provider like `tests/test_dev_tools.py::FakeProvider`).

```python
def test_label_ads_writes_llm_answers_and_skips_done_ones(tmp_path, l2): ...
    # first call: 1 file written, labelled == 1; second call: provider.calls unchanged, skipped == 1
def test_label_ads_records_fallbacks_as_failures(tmp_path, l2): ...
    # provider raising ProviderError -> no file, failed == [[task_id, "<error text>"]]
def test_consensus_majority_and_tie_order():
    votes = [{"roles": {"a": "headline", "b": "cta"}, "ranking": ["a", "b"], "model": "s"},
             {"roles": {"a": "subheadline", "b": "cta"}, "ranking": ["b", "a"], "model": "o"},
             {"roles": {"a": "headline", "b": "badge"}, "ranking": ["a", "b"], "model": "f"}]
    g = consensus(votes)
    assert g["roles"] == {"a": "headline", "b": "cta"} and g["ranking"] == ["a", "b"]
    assert consensus(votes[:2])["roles"]["a"] == "headline"          # 1-1 tie -> earliest voter
def test_consensus_ranks_units_a_voter_left_out_last(): ...
def test_review_sheet_writes_a_png(tmp_path, l1): ...
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_node1_label.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `label.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Commit: `feat(dev): teacher labelling, gold consensus and review sheets`.

---

### Task 6: Benchmark on splits, and the paired comparison

**Files:**
- Modify: `dev/bench_priority.py`, `dev/priority_metrics.py`
- Create: `dev/node1/compare.py`
- Test: `tests/test_dev_tools.py`, `tests/test_node1_compare.py`

**Interfaces:**
- `priority_metrics.paired_bootstrap(diffs: list[float], n_resamples: int = 10000, seed: int = 0) -> tuple[float, float, float]`: the mean and the 2.5 / 97.5 percentiles of resampled means.
- `bench_priority` gains:
  - `--split NAME` (ads from `data.load_split`; sample = task_id);
  - `--gold-dir DIR` (default `dev/golden`);
  - `--assets-dir DIR` (default `brand_kit/assets`; use `dev/realads/assets` with real or synthetic splits);
  - `--no-hosted`;
  - `--learned [MODEL_JSON ...]` (Task 12) and `--local-models GGUF ...` (Task 13).
  - `bench()` takes `gold_dir` and `assets_dir` arguments with the current defaults, so the existing tests keep passing.
- `compare.py`:
  - `score_rows(rows: list[dict], gold_dir: Path) -> dict[tuple[str, str], dict[str, dict[str, float]]]` recomputes role accuracy / τ / top-3 per (provider, model) and sample, from each row's `roles`/`ranking` columns, averaging runs;
  - `compare(csv_paths: list[Path], gold_dir: Path, a: str, b: str) -> dict` reports mean τ for each, the paired bootstrap of (b − a) over shared samples, and n;
  - provider keys are `"provider:model"`; `heuristic` is `"heuristic:-"`;
  - CLI: `python -m dev.node1.compare RUNS.csv [...] --gold-dir D --a heuristic:- --b learned:<ver>`.

- [ ] **Step 1: Write the failing tests**

```python
def test_paired_bootstrap_is_deterministic_and_centred():
    assert paired_bootstrap([0.0] * 20) == (0.0, 0.0, 0.0)
    m, lo, hi = paired_bootstrap([0.1] * 15 + [0.0] * 5)
    assert m == pytest.approx(0.075) and 0 < lo <= m <= hi
    assert paired_bootstrap([0.1, -0.2, 0.3], seed=1) == paired_bootstrap([0.1, -0.2, 0.3], seed=1)

def test_bench_reads_a_split_and_its_gold_dir(tmp_path, monkeypatch): ...
    # realads root with one ad + gold file; bench(["heuristic"], split ads, 1, out, gold_dir=..., assets_dir=...) -> kendall_tau not NaN

def test_compare_scores_rows_against_a_gold_dir_and_pairs_samples(tmp_path): ...
    # two runs.csv files (heuristic and fake learned) over 3 samples; result["n"] == 3 and the diff sign is right
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_dev_tools.py tests/test_node1_compare.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full `.venv/Scripts/python -m pytest -q`. Expected: all pass. Commit: `feat(bench): split/gold-dir benchmarking and paired bootstrap comparison`.

---

### Task 7: Labels, gold, review and frozen baselines (operational)

No new code. The commands come from Tasks 4–6.

- [ ] **Step 1: Teacher labels.** Run `python -m dev.node1.label teacher --provider gemini --model gemini-4-argon --split train val synth_train synth_holdout`. Expected: `failed` ≤ 2%, reported cost ≤ $11.
- [ ] **Step 2: Gold votes.** Run `label vote` for `claude-sonnet-5-5`, `claude-opus-5-5` and `gemini-3.8-flash` on `heldout`, then `label gold --split heldout` and `label review --split heldout`. Expected: 100 gold files and 100 PNGs; total spend for Steps 1–2 under $15.
- [ ] **Step 3: Teammate review (blocking).** Ask the user for a teammate to review the 50 `cr3_*` review sheets, and spot-check `cr2_*`. They correct `dev/realads/gold/<id>.json` by hand and set `"reviewed": true`. Wait for confirmation before Task 14. Tasks 8–13 may proceed meanwhile.
- [ ] **Step 4: Freeze R0 before any rules change.** Run `python -m dev.bench_priority --split heldout --no-hosted --runs 1 --assets-dir dev/realads/assets --results-dir dev/realads/bench_results`, and the same for `--split val`.
  - Note the held-out run folder name for Task 14.
  - Don't open its per-ad rows.
  - Copy only val's `summary.csv` to `dev/bench_results/<stamp>-r0-val/`.
- [ ] **Step 5: Hosted ceiling outputs.** Run `python -m dev.bench_priority --split heldout --gemini-models gemini-4-argon --claude-models --runs 1 --assets-dir dev/realads/assets --results-dir dev/realads/bench_results`. `--claude-models` with no value adds no Claude provider. Record the folder name for Task 14. The gold voters' own scores come from their `labels/<model>/` files via `compare.py`.
- [ ] **Step 6:** Commit `dev/bench_results/<stamp>-r0-val/summary.csv`: `data(bench): R0 rules on the real-ad validation split`.

---

### Task 8: Rules v2 (candidate A)

**Files:**
- Create: `engine/pipeline/priority/clues.py`, `dev/node1/misses.py`
- Modify: `engine/pipeline/priority/heuristic.py`
- Test: `tests/test_heuristic.py`, `tests/test_node1_misses.py`

**Interfaces:**
- `clues.py` (pure functions of a text string, shared with Task 9):
  - `first_word(text) -> str`;
  - `ACTION_VERBS: frozenset[str]`: the current set ∪ the friend's `ACTION` list from `dev/references/crello_roles.py`;
  - `is_action(text) -> bool`;
  - `is_discount(text) -> bool`: `%`, a currency symbol followed by a digit, or "off"/"save" next to a number;
  - `is_url(text) -> bool`: `www.`, `http`, `@`, `.com`, `.net`, `.org`, `.co`;
  - `has_legal_marker(text) -> bool`: the current markers plus `rebate`, `™`, `®`, `offer valid`, `see terms`, `while supplies last`.
- `heuristic.py` keeps `heuristic_roles(features)` and `heuristic_ranking(features, roles)` signatures. New constants: `FAINT_OPACITY = 0.35`, `WORDMARK_MAX_WORDS = 2`, `WORDMARK_MAX_CHARS = 18`, `WORDMARK_MAX_FONT_RATIO = 0.7`.
- `misses.py`: `python -m dev.node1.misses --split val --provider heuristic --gold-dir DIR` prints the per-ad role and ranking differences. It exits with an error for `--split heldout`.

- [ ] **Step 1: Write the failing tests** (in `tests/test_heuristic.py`, using `tests/designs.py`).

```python
def test_faint_text_is_never_the_headline():
    roles = _roles(GHOST_WORD)   # helper: heuristic roles keyed by unit id
    assert roles["headline"] == "headline" and roles["ghost_word"] == "other_text"

def test_find_is_an_action_verb():
    assert _roles(design(shape("b", 400, 900, 300, 80, group_id="cta"),
                         text("l", "Find your shade", 410, 915, 280, 50, 28, group_id="cta"),
                         text("h", "Golden hour", 90, 200, 900, 120, 90)))["cta"] == "cta"

def test_discount_sticker_on_a_button_is_a_badge():
    assert _roles(design(shape("s", 800, 80, 200, 200, group_id="st", rotation=-12),
                         text("t", "SAVE 20%", 810, 150, 180, 60, 36, group_id="st", rotation=-12),
                         text("h", "Golden hour", 90, 400, 900, 120, 90)))["st"] == "badge"

def test_url_is_other_text():
    assert _roles(design(text("h", "Golden hour", 90, 200, 900, 120, 90),
                         text("u", "www.solstice.com", 90, 1000, 400, 30, 20)))["u"] == "other_text"

def test_short_edge_text_is_a_wordmark_logo():
    assert _roles(design(text("w", "SOLSTICE", 40, 30, 200, 40, 30),
                         text("h", "Golden hour", 90, 400, 900, 120, 90)))["w"] == "logo"

def test_rebate_and_trademark_lines_are_legal(): ...      # "Rebate offer valid through 12/31", "Solstice® is a trademark" -> legal
def test_numbered_steps_rank_in_reading_order(): ...      # three other_text steps, the lowest one the most prominent -> ranked top to bottom
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_heuristic.py -q`. Expected: the new tests FAIL; `test_heuristic_matches_golden` passes.
- [ ] **Step 3:** Implement `clues.py`, then change `heuristic.py`:
  - a text unit with opacity < `FAINT_OPACITY` can't be the headline or subheadline, and becomes `other_text`;
  - a discount clue beats a verb for container labels and rotated groups;
  - URL-like text goes to `other_text`;
  - the wordmark rule applies only when no image unit was labelled `logo`;
  - legal markers come from `clues.has_legal_marker`;
  - `heuristic_ranking` sorts `other_text` and `legal` units by reading order (y, then x) instead of prominence.
- [ ] **Step 4:** Run `.venv/Scripts/python -m pytest -q`. Expected: all pass, including the three official goldens.
- [ ] **Step 5:** Commit: `feat(priority): rules v2 (faint text, wider verbs, discounts, URLs, wordmarks, legal, reading order)`.
- [ ] **Step 6: Iterate on train/val only.** Write `misses.py` and its test (`--split heldout` exits non-zero), then run it on `val` and `train` against the teacher labels.
  - Each further rule needs its own test in `tests/test_heuristic.py` and must keep the official goldens.
  - Stop when one more pass finds no fix worth ≥ 2 ads.
  - Bench `--split val` and copy its `summary.csv` to `dev/bench_results/<stamp>-rules-v2-val/`.
  - Commit each rule separately.

---

### Task 9: Feature vectors for the learned model

**Files:**
- Create: `engine/pipeline/priority/learned_features.py`
- Test: `tests/test_learned.py`

**Interfaces:**
- Consumes: `unit_features()` output, `clues.*`, heuristic roles.
- Produces:
  - `FEATURE_VERSION: int = 1` and `FEATURE_NAMES: tuple[str, ...]`;
  - `scorable(f: dict, heur_role: str) -> bool`: not a background, not `decoration`;
  - `vectorize(features: list[dict], heur_roles: dict[str, str]) -> tuple[list[str], np.ndarray]`: the unit ids of the scorable units in feature order, and a float64 matrix of shape `(n, len(FEATURE_NAMES))` with no NaN.
- **Starting feature set** (the builder may extend it; bump `FEATURE_VERSION` on any change):
  - kind and member-type one-hots; area share; log aspect; centre x/y; distance to the nearest edge; full-canvas flag;
  - z rank / unit count; opacity (missing → 1); rotated;
  - font size ÷ the design's max font; font size rank; is-max-font; log1p word count; log1p text length; bold; all caps;
  - `is_action`, `is_discount`, `is_url`, `has_legal_marker`, has a digit;
  - missing asset; fit cover; image-area rank; is the largest image;
  - overlap count; log unit count;
  - a one-hot of the heuristic role (9 content roles).
  - Never the unit id or the text itself.

- [ ] **Step 1: Write the failing tests**

```python
def test_vectorize_shape_and_order(l3):
    units, features, roles = _parts(l3)          # helper around build_units / unit_features / heuristic_roles
    ids, X = vectorize(features, roles)
    assert X.shape == (len(ids), len(FEATURE_NAMES)) and not np.isnan(X).any()
    assert ids == [f["unit_id"] for f in features if scorable(f, roles[f["unit_id"]])]

def test_vectorize_ignores_ids(l3):
    anon, _ = anonymize_source(l3["source"])
    assert np.array_equal(vectorize(*_parts_src(l3["source"]))[1], vectorize(*_parts_src(anon))[1])

def test_vectorize_tolerates_missing_optional_fields():
    d = design(text("a", "", 0, 0, 100, 40, 20), text("b", "Hello there", 0, 100, 400, 60, 40))
    for el in d["elements"]:
        for k in ("opacity", "z_index", "font_weight"):
            el.pop(k, None)
    ids, X = vectorize(*_parts_src(d))
    assert not np.isnan(X).any()

def test_headline_is_the_max_font(l2):
    ids, X = vectorize(*_parts_src(l2["source"]))
    assert X[ids.index("headline"), FEATURE_NAMES.index("is_max_font")] == 1.0
```

(`_parts_src` is a test helper in `tests/test_learned.py`; `vectorize(*…)` takes its `(features, roles)` pair.)

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_learned.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `learned_features.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Commit: `feat(priority): feature vectors for a learned ranker`.

---

### Task 10: Linear student (candidate B) and its numpy evaluator

**Files:**
- Create: `engine/pipeline/priority/learned.py`, `dev/node1/train.py`, `engine/pipeline/priority/model/priority_model.json` (generated), `tests/fixtures/learned_expected.json` (generated)
- Test: `tests/test_learned.py`

**Interfaces:**
- `learned.py`:
  - `MODEL_PATH = Path(__file__).parent / "model" / "priority_model.json"`.
  - `class LearnedRanker`: `name = "learned"`; `model: str` (the spec's `version`); `__init__(self, spec: dict)`.
  - `LearnedRanker.load(path: Path | None = None) -> LearnedRanker` (classmethod, cached per path, with `LearnedRanker.load.cache_clear()` available to tests). `None` means the module's `MODEL_PATH`, read at call time so tests can monkeypatch it. It raises `ValueError("feature mismatch: …")` when `spec["feature_version"] != FEATURE_VERSION` or `spec["feature_names"] != list(FEATURE_NAMES)`.
  - `predict(self, units: list[Unit], features: list[dict], heur_roles: dict[str, str]) -> tuple[dict[str, str], list[str]]`. Non-scorable units keep their heuristic role. A scorable unit flagged `missing_asset` gets `secondary_image`. Every other unit gets the argmax role. The ranking is the scorable units by score, descending, with ties broken by unit_id.
  - `role_proba(self, X) -> np.ndarray` and `scores(self, X) -> np.ndarray`, dispatching on `spec["role_model"]["kind"]` / `spec["rank_model"]["kind"]`. This task implements `"linear"`.
- **Spec JSON:** `{"version": "lin-<sha8 of weights>", "feature_version": 1, "feature_names": [...], "roles": [9 content roles], "mean": [...], "scale": [...], "role_model": {"kind": "linear", "coef": [[...]], "intercept": [...]}, "rank_model": {"kind": "linear", "coef": [...]}}`.
  - The rank model's input is the standardised features concatenated with the 9 role probabilities, trained pairwise: logistic regression on (x_i − x_j) for each label-ordered pair in an ad, no intercept.
- `dev/node1/train.py`:
  - `build_dataset(parts: list[tuple[str, Path | None, float]], assets_dir: Path) -> Dataset`, where each part is (split name, label dir or `None` for `dev/golden`, sample weight). `Dataset` holds `X`, `y_role`, `ad_index`, `pairs`, `w`.
  - `train_linear(ds: Dataset, C: float) -> dict` (the spec).
  - `evaluate(spec: dict, ads: list[dict], gold_dir: Path, assets_dir: Path) -> dict` returns mean role accuracy / τ / top-3. For each ad it calls `engine.pipeline.priority._heuristic_parts(source, assets_dir)`, then `LearnedRanker(spec).predict(units, features, heur_roles)`, then `_assemble(units, roles, ranking, "learned", None)`. That's exactly what Task 12's `plan()` branch will do.
  - `write_fixture(spec, path)`: role probabilities and scores computed by sklearn for `input/*.json` and `dev/samples/*.json`.
  - CLI: `python -m dev.node1.train --kind linear --C 0.1 0.3 1 3 10 --out engine/pipeline/priority/model/priority_model.json --fixture tests/fixtures/learned_expected.json`.
    - Train on `train` + `synth_train` (teacher, weight 1) + `official` (weight 10) + `devsamples` (weight 3).
    - Pick C by val τ (tie → role accuracy). Print the val metrics for every C.
- [ ] **Step 1: Write the failing tests**

```python
def test_linear_student_learns_a_tiny_labelled_set():
    spec = train_linear(_tiny_dataset(), C=100.0)       # 4 designs from tests/designs.py with hand labels
    ranker = LearnedRanker(spec)
    for src, gold in _TINY:
        roles, ranking = ranker.predict(*_ranker_inputs(src))
        assert {u: roles[u] for u in gold["roles"]} == gold["roles"] and ranking == gold["ranking"]

def test_load_rejects_another_feature_version(tmp_path):
    bad = {**json.loads(MODEL_PATH.read_text()), "feature_version": 999}
    (tmp_path / "m.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="feature mismatch"):
        LearnedRanker.load(tmp_path / "m.json")

def test_numpy_evaluator_matches_the_training_library():
    fx = json.loads(Path("tests/fixtures/learned_expected.json").read_text())
    ranker = LearnedRanker.load()
    assert fx["version"] == ranker.model
    for case in fx["cases"]:
        X = _matrix(case["sample"])
        np.testing.assert_allclose(ranker.role_proba(X), case["role_proba"], atol=1e-9)
        np.testing.assert_allclose(ranker.scores(X), case["scores"], atol=1e-9)
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_learned.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `learned.py` (numpy only) and `train.py` (sklearn).
- [ ] **Step 4:** Run the training CLI on the real labels. Expected: a per-C val table, plus the written model and fixture.
- [ ] **Step 5:** Run `.venv/Scripts/python -m pytest -q`. Expected: all pass.
- [ ] **Step 6:** Commit the model, fixture and code: `feat(priority): linear learned ranker trained on teacher labels (val τ <x>)`. The model JSON holds only weights; check `git diff --stat` shows no `dev/realads` paths.

---

### Task 11 (conditional): Tree-ensemble student

**Do this only if** Task 10's best val role accuracy is < 0.95 or val τ is < 0.90 against the teacher. Otherwise skip it and record the skip in the write-up.

**Files:**
- Modify: `engine/pipeline/priority/learned.py` (`"gbt"` kind), `dev/node1/train.py` (`train_gbt`)
- Test: `tests/test_learned.py`

**Interfaces:**
- `train_gbt(ds, n_estimators: int, max_depth: int, learning_rate: float) -> dict`:
  - roles from `GradientBoostingClassifier`;
  - rank as `GradientBoostingRegressor` on the normalised label position (0 = first; score = −prediction).
- Exported per tree: `{"feature": [...], "threshold": [...], "left": [...], "right": [...], "value": [...]}`, plus `init` and `learning_rate`, under `{"kind": "gbt", ...}`.

- [ ] **Step 1:** Test: `test_gbt_export_matches_sklearn`. Fit on random data (seed 0, 200×len(FEATURE_NAMES)); `role_proba` and `scores` match `predict_proba` / `-predict` to 1e-9.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement the export and the numpy traversal.
- [ ] **Step 4:** Run the tests, then train with `--kind gbt` and a small grid. Keep whichever of linear/gbt has the better val τ as `priority_model.json`, and regenerate the fixture.
- [ ] **Step 5:** Run the full suite. Expected: all pass. Commit: `feat(priority): gradient-boosted learned ranker (val τ <x>)`.

---

### Task 12: `learned` path in `plan()` (not yet the default)

**Files:**
- Modify: `engine/pipeline/priority/__init__.py` (`plan`), `engine/pipeline/priority/providers.py` (`get_provider`), `engine/pipeline/types.py` (`PriorityPlan` docstring), `dev/bench_priority.py` (`--learned`)
- Test: `tests/test_plan.py`, `tests/test_resize.py`

**Interfaces:**
- `get_provider("learned")` returns `LearnedRanker.load()`. `auto` is unchanged in this task.
- In `plan()`: a provider object with a `predict` attribute takes the learned branch inside the existing `try`. It calls `roles, ranking = llm.predict(units, features, heur_roles)`, then returns `_assemble(units, roles, ranking, "learned", LLMCallStats("learned", llm.model, latency_s=…))`. Any exception falls back to the rules with `stats.error = str(exc)`.
- `bench_priority --learned [PATH ...]` adds one `LearnedRanker.load(path)` per path (default `MODEL_PATH`); its rows' `source` is `"learned"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_learned_plan_holds_the_official_floor(l1, l2, l3):
    floor = {"chameleon_sample_level1_warmup": 1.0, "chameleon_sample_level2_core": 1.0,
             "chameleon_sample_level3_advanced": 0.93}
    for stem, task in zip(floor, (l1, l2, l3)):
        gold = json.loads(Path(f"dev/golden/{stem}.json").read_text())
        p = plan(task["source"], provider="learned")
        assert p.source == "learned" and p.stats.provider == "learned"
        assert role_accuracy({u.unit_id: u.role for u in p.units}, gold["roles"]) == 1.0
        assert kendall_tau([u.unit_id for u in p.ranked_units()], gold["ranking"]) >= floor[stem]

def test_learned_keeps_a_missing_asset_secondary_and_last(l3):
    p = plan(l3["source"], provider="learned")
    assert p.unit_for_element("holiday_watermark").role == "secondary_image"
    assert p.ranked_units()[-1].unit_id == "holiday_watermark"

def test_learned_plan_of_a_design_with_no_content():
    p = plan(design(shape("bg", 0, 0, 1080, 1080, z_index=0), shape("dot", 10, 10, 40, 40, z_index=1)),
             provider="learned")
    assert p.source == "learned" and p.ranked_units() == []

def test_corrupt_weights_fall_back_to_rules(monkeypatch, tmp_path, l2):
    (tmp_path / "m.json").write_text("{")
    monkeypatch.setattr(learned, "MODEL_PATH", tmp_path / "m.json"); LearnedRanker.load.cache_clear()
    p = plan(l2["source"], provider="learned")
    assert p.source == "heuristic" and p.stats.error

def test_feature_mismatch_falls_back_with_a_clear_error(monkeypatch, tmp_path, l2): ...
    # weights with feature_version 999 -> source == "heuristic" and "feature mismatch" in p.stats.error

def test_priority_line_for_the_learned_path(capsys, monkeypatch, l1):   # in tests/test_resize.py
    monkeypatch.setenv("CHAMELEON_LLM", "learned")
    R.resize_with_report(l1["source"], l1["target_canvas"])
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert line.startswith(f"[priority] source=learned provider=learned model={LearnedRanker.load().model} ")
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_plan.py tests/test_resize.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full suite. Expected: all pass. If the floor test fails, B misses the floor: record it, and Task 14 ships A.
- [ ] **Step 5:** Commit: `feat(priority): learned path in plan() behind CHAMELEON_LLM=learned`.

---

### Task 13: Local LLM provider (candidate C) and its timings

**Files:**
- Create: `dev/node1/local_llm.py`, `dev/node1/Dockerfile.local`, `requirements-local-llm.txt` (pinned `llama-cpp-python`)
- Modify: `dev/bench_priority.py` (`--local-models`)
- Test: `tests/test_node1_local.py`

**Interfaces:**
- `class LocalLLMProvider`:
  - `name = "local"`; `model = Path(model_path).stem`;
  - `__init__(self, model_path: str, n_threads: int | None = None, n_ctx: int = 8192)`; `llama_cpp` is imported lazily on first `complete`.
  - `complete(self, system: str, user: str, schema: dict) -> LLMResult`:
    - calls `create_chat_completion(messages=[system, user], response_format={"type": "json_object", "schema": schema}, temperature=0, max_tokens=1024)`;
    - for Qwen3 models, appends ` /no_think` to the user message;
    - parses with `providers._parse_json` (raising `ProviderError`);
    - stats carry latency and tokens, with `cost_usd=0.0`.
- *Deviation from the spec:* `local` is not added to `get_provider` (the engine must not import `dev/`). Only the benchmark constructs it.
- `Dockerfile.local`: `FROM` the engine image built from `chameleon/Dockerfile`, then copy `dev/` + `requirements-local-llm.txt`, `pip install` it, with `ENTRYPOINT ["python", "-m", "dev.bench_priority"]`. The GGUF is mounted, never copied.

- [ ] **Step 1: Write the failing tests** (inject a fake `llama_cpp` module into `sys.modules`).

```python
def test_local_provider_request_shape_and_stats(monkeypatch): ...
    # fake Llama records kwargs and returns {"choices":[{"message":{"content": '{"summary":"x","roles":[],"ranking":[]}'}}],
    #                                        "usage":{"prompt_tokens":10,"completion_tokens":5}}
    # assert kwargs["response_format"]["schema"] == schema and kwargs["temperature"] == 0
    # assert result.stats.provider == "local" and result.stats.input_tokens == 10 and result.stats.cost_usd == 0.0
def test_local_provider_bad_json_falls_back(monkeypatch, l2): ...
    # content "not json" -> plan(..., provider=LocalLLMProvider("x.gguf")).source == "heuristic"
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_node1_local.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement, plus `--local-models` in the benchmark.
- [ ] **Step 4:** Run the tests. Expected: PASS. Commit: `feat(dev): local LLM provider for the node 1 bake-off`.
- [ ] **Step 5: Ask before downloading.** Propose a Qwen3-4B instruct GGUF (Q4_K_M, and Q8_0 if time allows): give the user the exact Hugging Face repo, file names and sizes, and wait for a yes. Store the files under `dev/realads/models/` (gitignored).
- [ ] **Step 6: Validation run.** Install `requirements-local-llm.txt`, then run `python -m dev.bench_priority --split val --no-hosted --local-models <gguf> --runs 1 --assets-dir dev/realads/assets --results-dir dev/realads/bench_results`. Score it with `dev.node1.compare` against the teacher labels. Record val τ and the native median latency.
- [ ] **Step 7: Mac timing (teammate).** Build `docker build --platform linux/amd64 -t chameleon-local -f dev/node1/Dockerfile.local .` on the Mac, after building the engine image there. Run the bench on the committed designs only (`--samples input/*.json dev/samples/*.json --no-hosted --local-models /models/<gguf> --runs 1`) with the model mounted at `/models`.
  - Record the median latency.
  - Record the image growth: `docker image ls` size of `chameleon-local` minus the engine image, plus the GGUF size.
  - Ask the user to relay the numbers.
- [ ] **Step 8:** Commit the val and Mac summaries (aggregates only) under `dev/bench_results/<stamp>-local-*`.

---

### Task 14: Held-out bake-off, decision, default switch

**Files:**
- Modify: `engine/pipeline/priority/providers.py` (`get_provider` `auto`), `tests/test_providers.py`, `tests/test_plan.py`
- Create: `dev/bench_results/<stamp>-bakeoff/summary.csv`, `dev/bench_results/<stamp>-bakeoff/decision.json`

- [ ] **Step 1: Preconditions.** The teammate review from Task 7 Step 3 is confirmed. Rules v2 and the model are committed.
- [ ] **Step 2: One held-out run per new candidate.** Run `python -m dev.bench_priority --split heldout --no-hosted --learned --runs 1 --assets-dir dev/realads/assets --results-dir dev/realads/bench_results`; this covers rules v2 (as `heuristic`) and B. Run the same with `--local-models <gguf>` for C.
- [ ] **Step 3: Score everything against the reviewed gold** with `dev.node1.compare`, reusing R0's and the ceiling's held-out runs from Task 7:
  - A vs R0 (for the write-up);
  - B vs A;
  - C vs the better of A/B.
  - Write `summary.csv` (mean role accuracy / τ / top-3 per candidate) and `decision.json`: `{"comparisons": [{"a", "b", "mean_diff", "ci": [lo, hi], "n"}], "c_latency_s_mac", "c_image_growth_gb", "winner", "reason"}`.
  - Apply the decision rule in Global Constraints. Commit them.
- [ ] **Step 4: Switch the default to the winner.**
  - **B wins:** `get_provider()` with `auto` returns `LearnedRanker.load()` when it loads, else `None`. It ignores API keys; `claude`/`gemini` run only when named.
    - Update `test_get_provider_auto`: with and without keys → `.name == "learned"`; with the model path monkeypatched to a missing file → `None`.
    - Add `test_auto_ignores_api_keys`.
  - **A wins:** delete `engine/pipeline/priority/model/`, the learned branch, `learned*.py` and their tests. `auto` returns `None` regardless of keys, and the `auto` tests are updated the same way.
  - **C wins:** stop. Report to the user. Baking a model into the image needs its own plan.
- [ ] **Step 5:** Run `.venv/Scripts/python -m pytest -q`. Expected: all pass. Commit: `feat(priority): <winner> is node 1's default offline path`.

---

### Task 15: Write-up and end-to-end verification

**Files:**
- Modify: `docs/pipeline/01-priority-node.md` (new "Offline bake-off" section), `docs/pipeline/README.md` (§1 diagram, §8 env vars, §9 decisions, §11 missing), `chameleon/README.md` (settings, status line, how to rebuild data/labels/weights)

- [ ] **Step 1: Write-up** (aggregate counts only, no ad text):
  - data counts per split and coverage;
  - labels (teacher, voters, review outcome, spend);
  - a results table for R0, A, B, C, the ceiling and the voters;
  - comparisons with CIs; the decision and why; deviations (local not in `get_provider`; Task 11 run or skipped).
- [ ] **Step 2:** Run `.venv/Scripts/python -m dev.run_matrix`. Expected: all 39 cells pass every gate, and the fallback never wins (`output/matrix/winners.json`). Compare the official samples' contact sheets with a run on `testing-llms` and note any changed winners.
- [ ] **Step 3: Docker, no network.** Run `docker build --platform linux/amd64 -t my-chameleon-engine .`, then the README's `--network none` run on `input/`.
  - Expected: each task prints `[priority] source=<winner> …`, with node 1 adding < 50 ms per task for B.
  - Then run the README's container matrix. Expected: 39/39 pass the gates.
- [ ] **Step 4:** Run `.venv/Scripts/python -m pytest -q`, both locally and in the container (README command). Expected: all pass.
- [ ] **Step 5:** Commit: `docs: offline node 1 bake-off results and the shipped path`.
