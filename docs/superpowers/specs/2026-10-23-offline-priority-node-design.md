# Offline node 1: bake-off, then ship the winner (design)

**Status (2026-10-23):** design approved; implementation plan next. Branch `offline-priority`, cut from `testing-llms` (PR #5).

## Context

Stage-one scoring runs our Docker image with `--network none`, so node 1's Claude and Gemini path can never run there. In the whole engine, the LLM only does node 1's job: give each unit a role (headline, logo, CTA, legal…) and rank the content units. Offline, `plan()` already falls back to the rule-based ranker, so today the scored engine *is* the rules.

The rules match the LLMs on the 3 official samples. They fall apart on designs they weren't written for: Kendall τ 0.749 against 0.95–0.98 for the hosted models on the 10-design benchmark (`chameleon/dev/bench_results/2026-10-19_213054`). Each miss is a specific gap:
- opacity is ignored, so a faint 280 px word becomes the headline;
- "find" isn't an action verb;
- "SAVE 20%" is read as a button;
- the legal vocabulary is short;
- numbered steps aren't ranked in reading order.

Roles and ranks drive node 2's slot assignment, per-role text floors and CTA protection, and node 3's weights, so wrong labels change the final picture.

**What the friend's branch (`angelo/distill`, `distil-research/`) adds:**
- **It already planned exactly this as "student A":** a role classifier + ranker over `unit_features`, trained on teacher labels, shipping only if it beats the rules on holdout ads (`training-data/role-labels/README.md`). It was never built.
- **Its local-model work was about layout, not labels.** A Qwen3-4B fine-tune, run as Q4 GGUF on CPU, at best tied their rule-based pipeline. Their own offline research (`docs/research/beating-v5-offline.md`) ranks "a gradient-boosted per-element classifier" (under 50 ms) ahead of a 0.5–1.7B LLM (an estimated 1–25 s on CPU, unmeasured).
- **A Crello-based role heuristic scored poorly on real templates** (`dev/references/crello_roles.py`, `labelling_accuracy.json`): about 80% headline accuracy, but CTA and logo precision only around 25–30%. On real ads, hand rules fall short.
- **Real ads exist, but none are committed** (the VistaCreate licence is unverified): `make_real_challenge.py` converts Crello 5.1.0 ad templates into our schema.
  - 1,606 ad-format templates were scanned; 327 passed their layout-quality gate.
  - Fixed sets were drawn from that pool: challenge2 (50), crello_dev (60) and challenge3 (50, untouched).
  - Sets are kept apart by design family: Crello stores one design in several formats.
- **Their converted ads carry semantic element ids** taken from that heuristic (`headline`, `cta`, `logo`, `product`, `body_N`, `deco_N`). Our LLM prompt includes unit ids, so a teacher would read the answer straight from the id.

**Intent (user's words, paraphrased):** first research what's best for offline node 1, an algorithm or a local LLM, then rebuild node 1 around the winner. Scope is node 1 only; node 2 (layout) and node 3 (grading) stay as they are.

## Goals, non-goals, anti-goals

**Goals**
- A node 1 that runs fully offline inside the existing image and gets as close as possible to the hosted LLMs' roles and ranking on real ads it was never tuned on.
- An evidence-based choice between: better rules (A), a small model trained on LLM labels (B), and a local LLM (C). The numbers are committed and written up.
- The winner becomes the default node 1 path in the scored run.

**Non-goals**
- Changing node 2 or node 3, or porting the friend's layout pipeline.
- Using a model for layout or grading.
- Removing the hosted LLM code: it stays as a development tool (labelling, benchmark) and for any later stage that allows network.

**Anti-goals** (failure even if it "works")
- Tuning anything on the held-out ads. The held-out score is read once per candidate, at the end.
- Any regression on the 3 official samples, which are closest to the hidden tasks: role accuracy stays 1.00 and τ stays ≥ the current rules (1.00, 1.00, 0.93).
- Any path where node 1 can crash or block a task. `plan()` still never raises, and the rules remain the last fallback.
- Network access, a download at run time, or keys in the image.
- Committing anything that contains Crello content (user decision; matches the friend's licence rule).

## Constraints (from the repo)

- `docker run --network none`, `linux/amd64` image (`chameleon/Dockerfile`), reference machine Apple M5 Pro (likely emulated amd64, CPU only). The image size limit is unknown (`docs/pipeline/README.md` §10 q3).
- `plan()` contract: never raises; background and missing assets are code decisions (`units.py`, `_assemble` in `priority/__init__.py`); all-shape units are decoration; the LLM can't make text or images decoration (`validate.repair`).
- Tests force `CHAMELEON_LLM=heuristic` (autouse fixture); the suite takes about a minute.
- Node 1 is pure Python and doesn't measure text, so it gives the same answer on Linux and Windows.

## Approaches (user picked B, after a bake-off)

| | What | Cost in the image | Trade-off |
|---|---|---|---|
| **A. Better rules** | Fix the measured gaps in `heuristic.py` | none | Predictable and debuggable; plateaus on the long tail of real ads (the friend's Crello rules: CTA/logo ~25–30% precision) |
| **B. Small trained model** ✅ picked | A hosted LLM labels real + synthetic ads at dev time; train a small role classifier + ranker on node 1's features plus A's fixes; ship a weights file evaluated with numpy | a few KB–MB, milliseconds | Needs a training pipeline; reads text only through surface clues |
| **C. Local LLM** | Quantized 3–4B model (e.g. Qwen3-4B, which the friend already runs as GGUF on CPU) via llama.cpp, JSON-schema output, behind the existing prompt/repair path | +2–3 GB, seconds per ad, emulated CPU | Least new engine code; quality and speed unproven; the smallest hosted model we tried only reached τ 0.83 |

**Why B:** it gets the LLMs' judgement into the image without running an LLM, keeps the image small and deterministic, keeps the rules as the fallback, and matches the friend's own ranking of options. Because the user asked for research first, A, B and C are all measured on the same held-out ads before anything becomes the default.

**Timing (user decision):** one decision at the end. Nothing becomes the default until every candidate is scored on held-out ads.

## Design

### 1. Data
**Real ads, rebuilt locally and gitignored.**
- Run the friend's converter (`distil-research/chameleon/dev/distill/make_real_challenge.py` and `make_challenge3.py`, seed 0, from a worktree of `angelo/distill`) against the 4 Crello 5.1.0 parquet shards (downloaded at dev time).
- The output lands in `chameleon/dev/realads/` (gitignored, never committed): `ads/*.json`, `assets/`, `splits.json`, `families.json`. If the friend sends their `.nosync` folders instead, use those; the sets are identical because the converter is deterministic.
- **Splits, by design family** (the friend's rule: a shared text of ≥ 8 characters):
  - **held-out** = challenge3 (50, untouched by anyone) + challenge2 (50; the friend used it to choose layouts, never labels). 100 real ads.
  - **validation** = crello_dev (60).
  - **train** = every other converted ad template whose family isn't in held-out or validation.
- The training pool may go beyond the friend's pixel/contrast gate, which judges layout quality and doesn't matter to node 1. It needs only schema validity and the structural checks. Target ≥ 600 training ads; report the count.
- Intake validates every ad against `chameleon_design_schema_v0.9.json` and reports how many keep `group_id`, `z_index`, `opacity` and real asset files.

**Synthetic ads (committed already).** Use the friend's `training-data/corpus/` designs with their own train/holdout split, plus our 7 `chameleon/dev/samples/`. They add to training and act as a secondary held-out set; they're never the decision set.

**The official samples.** The 3 `chameleon/input/` samples and their goldens go into B's training, weighted up, since the hidden tasks share the official brand kit. They're also the floor check in the anti-goals.

**Anonymised ids.** Before any ad reaches a teacher, a model or the benchmark, element ids and group ids are rewritten to neutral ones (`e1…`, `g1…`) with a reversible map. Without this, the friend's semantic ids (`headline`, `cta`…) leak the heuristic's answer to the teacher. The official samples, whose ids are also semantic (`headline`, `cta_bg`), are anonymised the same way when they're labelled or scored. The engine itself is unchanged: B never uses ids as features.

### 2. Labels
- **Training labels:** one teacher, `gemini-4-argon` (best τ on our bench, 0.983, ~$6/1,000), run through the existing `plan()` path (prompt, `repair()`, cache) on anonymised ads. Stored per ad in golden format (`roles`, `ranking`) with teacher metadata, under `chameleon/dev/realads/labels/` (gitignored).
- **Held-out gold, without the teacher:** consensus of `claude-sonnet-5-5`, `claude-opus-5-5` and `gemini-3.8-flash`: majority role per unit, and the ranking by mean position. A teammate reviews challenge3's 50 and spot-checks challenge2. The teacher stays out of the gold, so B (trained to copy it) gets no edge over A from self-agreement.
- **Ceiling:** the teacher, scored against that gold. B can only approach it.
- **Estimated API spend:** < $15.
- *My call:* the teacher and gold models; the builder may swap in an equally strong model if one is unavailable.

### 3. Candidates
- **R0: current rules,** measured first, before any code change.
- **A: rules v2.** Developed against train and validation only. Known fixes:
  - low-opacity text can't be the headline;
  - a wider set of CTA verbs (the friend's `ACTION` list in `crello_roles.py` is a good start);
  - %, prices and "save N" mean a badge, not a CTA;
  - URL-like text is other text;
  - a short text wordmark near an edge counts as the logo;
  - more legal markers (rebate, ™, ®, "offer valid", "see terms");
  - same-role sequences rank in reading order.
  - Further fixes come from the training misses.
  - The official goldens must still match: `chameleon/tests/test_heuristic.py::test_heuristic_matches_golden`.
- **B: learned model.**
  - Label space: the 9 content roles. Background stays a code flag and all-shape units stay decoration, mirroring `repair()`.
  - Inputs: `unit_features()` plus new features (A's clues as booleans, relative font size, opacity, relative z, edge bands, digits/%/currency/URL, word count…). No ids.
  - Output: role per unit + an importance score; the ranking sorts by score, trained pairwise on teacher rankings.
  - Model family: anything whose trained form evaluates with numpy alone (linear model or a tree ensemble exported to JSON); the training library is dev-only. Selected on validation.
  - Training data: train + synthetic train + the 3 official samples and 7 dev designs (weighted up).
  - No rule overrides on top of B beyond the existing hard signals. If B misses the official-sample floor, A ships.
- **C: local LLM.**
  - A dev-only `LocalLLMProvider` (llama-cpp-python, GGUF path from env), with the same `SYSTEM_PROMPT`, `build_user_message`, `response_schema` and `repair()`.
  - Try 1–2 instruct models at Q4/Q8 (Qwen3-4B first, since the friend's `student_cpu_bench.py` already runs it on CPU).
  - Measure quality on held-out. Measure latency on this x86 machine natively, and **on a teammate's Apple Silicon Mac in the amd64 container** (model mounted, not baked), which is closer to the M5 Pro scorer.
  - Not added to the image unless it wins.
- **Hosted:** the teacher as the ceiling, and the three gold models (reported, but biased as voters).

### 4. Decision rule (*my call*; change it if you disagree)
Score every candidate on the 100 held-out real ads with the existing metrics (`chameleon/dev/priority_metrics.py`: role accuracy, Kendall τ, top-3 recall), plus a paired bootstrap 95% CI on per-ad τ differences.
1. B ships if it beats A by ≥ 0.03 τ with a CI that excludes 0. Otherwise A ships alone, because it's simpler.
2. C replaces the winner only if all three hold:
   - it beats the winner by ≥ 0.05 τ, with a CI that excludes 0;
   - its median latency is ≤ 30 s per ad in the amd64 container on the Mac;
   - it adds ≤ 3 GB to the image.
3. Any winner must also hold the official-sample floor in the anti-goals.

### 5. Engine integration (for B; A or C slot in the same way)
- New `chameleon/engine/pipeline/priority/learned.py`: loads `engine/pipeline/priority/model/priority_model.json` once and predicts roles + ranking from features. Load or predict failures raise inside `plan()`'s try, so they fall back to the rules with `stats.error` set.
- `plan()` gains a learned path. It returns `_assemble(units, roles, ranking, "learned", stats)`, where `stats.provider="learned"`, `stats.model=<model version/hash>` and `stats.latency_s` is set. `PriorityPlan.source` adds `"learned"` (docstring in `types.py`).
- `get_provider`/`CHAMELEON_LLM` gains `learned` (and `local`, dev-only). **`auto` now means learned if the weights load, else rules.** Hosted models run only when named (`claude` / `gemini`). *My call:* dev runs then behave like the scored run even with keys in `.env`.
- The provider SDK imports move inside `_get_client` (deferred review item), so a broken SDK install can't take down the offline path.
- No Dockerfile change: the weights live under `engine/`, which is already copied. The `[priority]` status line prints `source=learned`.
- The weights are committed (they hold no Crello content). The README explains how to rebuild the data, labels and weights.

### 6. Dev tooling
- `dev/realads.py` (or similar): intake from the friend's converter output. Validation, family-based splits, id anonymisation, and a gitignore check.
- `dev/label_ads.py`: teacher labelling over a split. Concurrent, resumable, cached, reports cost. A gold mode builds the 3-model consensus.
- `dev/train_priority.py`: features → train → validation report → export weights + a committed fixture of expected predictions (built from official and synthetic designs only, so it carries no Crello content).
- `dev/bench_priority.py`:
  - accepts `learned`, `rules` and `local` providers;
  - takes a golden directory per sample set;
  - prints the bootstrap CI;
  - summaries committed under `dev/bench_results/`. Per-ad rows from the real ads quote Crello text, so they stay gitignored; only aggregates are committed.
- Dev-only dependencies (training library, llama-cpp-python) go in a separate dev/research requirements file, never `requirements.txt`.

### 7. Write-up
A "Offline bake-off" section in `docs/pipeline/01-priority-node.md`: data (aggregate counts only), labels, a results table for every candidate, the decision and why. Plus updates to `docs/pipeline/README.md` (§1 diagram, §8 env vars, §9 decisions, §11 missing) and `chameleon/README.md` (settings, status line, how to rebuild).

### Left to the builder
- Exact feature set and model family; the ranking formulation; hyperparameters.
- Which local GGUF models, quantization and llama.cpp settings.
- How the friend's converter is run (a worktree of `angelo/distill`, or a minimal port of its scan into `chameleon/dev/`); names and CLI flags of dev tools; test organisation.
- Whether rules v2 lands in `heuristic.py` directly (the R0 numbers are recorded first).

## Branch
*My call:* a new branch `offline-priority`, cut from `testing-llms` (PR #5's head), because it builds on PR #5's benchmark and `.env` work.

## Verification
- `.venv/Scripts/python -m pytest -q` from `chameleon/`: all existing tests pass, including the official goldens, plus new tests:
  - the numpy evaluator matches the training library's predictions on the committed fixture;
  - `plan()` with corrupt or missing weights falls back to the rules and never raises;
  - `auto` picks learned with and without keys set;
  - id anonymisation round-trips;
  - each rules-v2 fix has its own test.
- `python -m dev.bench_priority` on held-out with every candidate. Aggregates are committed, and the decision rule is applied in the write-up.
- `python -m dev.run_matrix`: all 39 sample × size outputs pass every gate, and the fallback never wins. Contact sheets are compared with the current run for the 3 official samples.
- Docker: build, then `docker run --network none` on `input/`. Each task prints `[priority] source=learned …`, the added time per task is small (target < 50 ms for B), and all gates pass in the container matrix (README command).
