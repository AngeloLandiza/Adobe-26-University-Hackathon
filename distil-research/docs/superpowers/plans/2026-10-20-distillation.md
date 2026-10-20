# Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Location (2026-10-20):** this plan now lives in the research sandbox `distil-research/`. Every path below (`chameleon/...`, `docs/...`) is relative to `distil-research/`, not the repo root. Tasks F1 and F2 are done: see `distil-research/README.md`.

**Goal:** Use big AI models (the "teacher") on our own machines to produce data, then turn that data into small, offline pieces of the engine (the "students") that make better layouts and pick better winners in the scored, no-network Docker run.

**Architecture:** Everything that talks to an AI lives in `chameleon/dev/` and never ships. The engine gets two small hooks that read data files: grader weights (`weights.json`) and extra floor plans (`extra_templates.json`). The students' output is written into those files, so what ships is data plus plain Python, with no model at run time.

**Tech Stack:** Python 3.11 (Docker) / 3.14 (local venv), numpy 2.4.6, the vendored renderer, Claude Code CLI (`claude -p`, Opus 5.5) as the first teacher. Gemini and others are optional extra teachers.

**Spec:** this plan's "Design" section (agreed in conversation on 2026-10-19/20) and the engine docs in [`docs/pipeline/`](../../pipeline/README.md). The engine itself is PR #4; this branch (`angelo/distill`) merges it.

---

## Design (plain words)

**Words used below:**
- **Teacher:** a big AI model (Claude Opus 5.5 through Claude Code; later maybe Gemini) used only on our laptops. It produces data and never ships.
- **Student:** something small we build from the teacher's data that *does* ship: a set of numbers, a few floor plans, or a tiny model.
- **Cell:** one ad resized to one target size, e.g. "Level 2 ad → LinkedIn banner 1584×396".
- **Candidate:** one layout the engine made for a cell. A cell has 2–38 of them.
- **Pair judgement:** the teacher sees the original ad and two candidates (A and B) and says which is better and why.
- **Floor plan / template:** the engine's way of cutting the canvas into named boxes (slots) such as `hero`, `text`, `logo`, `cta`, `badge` and `minor`, before pieces are placed in them.
- **Holdout ads:** ads we never train on, kept aside to check whether a student really helps.

**The four students.** The letters match the conversation.

| | Student | Replaces or adds | What ships | Teacher data it needs |
|---|---|---|---|---|
| **D** | learned grader | replaces the hand-guessed weights in node 3 | `engine/pipeline/grading/weights.json` (12 numbers + 1 share) | pair judgements of existing candidates |
| **B** | new floor plans | adds templates to node 2 | `engine/pipeline/layout/extra_templates.json` | teacher-designed floor plans + pair judgements, then clustering |
| **A** | distilled ranker | replaces the rule-based ranker in node 1 *if it beats it* | a small JSON model + plain-Python scorer | teacher role/rank labels for many ads |
| **C** | floor-plan recommender | adds per-ad candidates to node 2 | a JSON table of (ad description → winning floor plan) | B's winning floor plans |

**Why this shape:**
- The scored run has no network, so no AI runs there.
- Students that are data files can't crash the engine: a missing or broken file falls back to today's behaviour.
- The hard gates (node 3, G1–G9) stay unchanged, so no student can cause a rule violation.

**Order and parallel work:**

```
            ┌─ F1 grader-weights hook ──────────────────────────────┐
 Lane 1  ───┤                                                       ├─► D  learned grader ──► ship weights.json
 (engine)   └─ F2 floor-plan hook ─► B1 hand-drawn floor plans ─────┤
                                                                    ├─► B2 teacher floor plans + clustering ─► C recommender
 Lane 2  ─── F3 ad corpus (≥30 ads) ────────────────────────────────┤
 (data)                                                             │
 Lane 3  ─── F4 teacher runner ─► F5 candidate pool ─► F6 judge + pilot
 (teacher)                                                          │
                                                                    └─► A  distilled ranker (needs F3 + F4 only; gated on beating the rules)
```

| Work | Can start | Blocked by | Parallel with |
|---|---|---|---|
| F1, F2 (engine hooks) | now | nothing | F3, F4 |
| B1 (hand-drawn floor plans) | after F2 | F2 | everything after it |
| F3 (ad corpus) | now | nothing (better with the official rubric in hand) | F1, F2, F4 |
| F4 → F5 → F6 (teacher pipeline) | now | nothing (F5 uses today's weights to pick close contests) | F1, F2, F3 |
| D | after F1 + F3 + F6 | judgements | B2, A |
| B2 | after F2 + F3 + F6 | judgements | D, A |
| A | after F3 + F4 | none of the engine hooks | D, B2 |
| C | after B2 | B2's winning floor plans | — |

**External blockers** (not code):
1. **The official rubric** (main brief, Section 3). F6's `rubric.md` is a provisional guess until we have it.
2. **The code-freeze date.** If it's only days away, ship F1, F2, B1 and D only.
3. **Claude Code usage limits.** The default sizes below (about 300 teacher calls for D) are chosen to fit in a few sessions.
4. **Extra teachers.** Gemini and GPT need their own keys; the pipeline works with Claude alone.

---

## Global Constraints

- The scored run is `docker run --network none`: nothing in `engine/` may import from `dev/`, call a network, or need an API key.
- Engine code must run on Python 3.11 (Docker image). No 3.12+ syntax in `engine/`.
- With no `weights.json` and no `extra_templates.json`, the engine's output must be byte-identical to today's: all 463 existing tests keep passing unchanged.
- A data file never makes the engine raise. A missing file at the default path is silent; a missing env-var path, an unreadable file or an invalid file prints one `[distill] …` line to stderr, and the engine uses the defaults.
- The gates G1–G9 and their tolerances are not changed by this plan.
- Never regenerate or replace pixel content. Corpus ads may only reference images in `brand_kit/assets/`, or in `dev/corpus/assets/` for imported ads, which never ship.
- Teacher outputs, renders and judgements go in `chameleon/dev/data/` (gitignored). Only the students' output files are committed.
- Tests use the rule-based ranker (existing autouse fixture). No test may call `claude` or any API: the teacher is injected and faked in tests.
- Deterministic: every sampling step takes a `seed` (default 0). Ties are broken by `candidate_id`, as in `pick_best`.

## Review Focus

1. **A broken or hand-edited `weights.json`** (negative weight, missing key, `geometry_share` 1.5, invalid JSON) must give the default weights and one warning. Pinned in Task F1.
2. **A bad floor plan in `extra_templates.json`** (box outside 0–1, unknown slot name, no `minor` slot, overlapping slots, duplicate name) must be skipped while the valid plans in the file still load. Pinned in Task F2.
3. **The teacher CLI failing** (rate limit, timeout, `is_error: true`, no `structured_output`) must not lose finished work. Judging resumes from `judgments.jsonl` and skips pairs already judged. Pinned in Tasks F4 and F6.
4. **Position bias:** a judge that prefers whichever layout is shown first. A and B are shuffled by seed, a share is re-asked with the order swapped, and the run reports a bias figure. Pinned in Task F6.
5. **Holdout leakage:** a holdout ad used in training would make the evaluation meaningless. `train_grader` refuses judgements from holdout ads in its training set. Pinned in Task D1.

---

## Phase F: foundation

### Task F1: grader weights hook

**Files:**
- Create: `chameleon/engine/pipeline/grading/weights.py`
- Modify: `chameleon/engine/pipeline/grading/metrics.py:211-212` (`geometry_total`), `chameleon/engine/pipeline/grading/pixels.py:99-100` (`pixel_total`), `chameleon/engine/pipeline/grading/score.py` (`pick_best`)
- Test: `chameleon/tests/test_weights.py`

**Interfaces:**
- Produces:
  - `GraderWeights(geometry: dict[str, float], pixels: dict[str, float], geometry_share: float)` (frozen dataclass)
  - `DEFAULT_WEIGHTS: GraderWeights`, equal to today's `metrics.WEIGHTS`, `pixels.PIXEL_WEIGHTS` and `0.7`
  - `WEIGHTS_PATH = Path(__file__).with_name("weights.json")`, overridable by env `CHAMELEON_GRADER_WEIGHTS`
  - `load_weights(path: Path | None = None) -> GraderWeights`
  - `geometry_total(m, weights: dict[str, float] | None = None) -> float` and `pixel_total(m, weights=None) -> float`; `None` means today's constants
  - `pick_best(..., weights: GraderWeights | None = None)`; `None` means `load_weights()`, cached per path
- File format: `{"geometry": {9 keys}, "pixels": {3 keys}, "geometry_share": 0.7, "source": "hand" | "learned:<run id>"}`. Each group's weights are ≥ 0 and sum to 1 (±1e-6); `0 ≤ geometry_share ≤ 1`.

- [ ] **Step 1: Write the failing tests** in `tests/test_weights.py`:
  - `test_missing_file_gives_defaults`: `load_weights(tmp_path / "nope.json") == DEFAULT_WEIGHTS`
  - `test_valid_file_overrides`: a file with `balance: 1.0` and the other geometry weights 0, plus `geometry_share: 0.5`, loads exactly those values
  - `test_invalid_files_fall_back`, parametrised over: invalid JSON, a missing key, a negative weight, a NaN, `geometry_share: 1.5`, sums of 0.9 → `DEFAULT_WEIGHTS`, and `capsys` shows exactly one line starting `[distill]`
  - `test_pick_best_unchanged_without_file`: for each sample input at its own target, `pick_best(..., weights=None)` with no file picks the same `candidate_id` and the same `total`s as `weights=DEFAULT_WEIGHTS`
  - `test_pick_best_uses_given_weights`: with custom weights, each rendered report satisfies `total == share × Σ geometry + (1 − share) × Σ pixels`, computed from `report.metrics` (abs tol 1e-9)
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_weights.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `weights.py`, then thread `weights` through `geometry_total`, `pixel_total` and `pick_best`. The top-K selection uses the geometry weights being passed in.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: 463 + the new tests pass.
- [ ] **Step 5: Commit** `feat(grading): load grader weights from weights.json with safe defaults`

### Task F2: floor-plan hook

**Files:**
- Create: `chameleon/engine/pipeline/layout/extra_templates.py`
- Modify: `chameleon/engine/pipeline/layout/templates.py` (add `templates_for`, `param_grid`, and `build_slots` support), `chameleon/engine/pipeline/layout/generate.py:443-444`
- Test: `chameleon/tests/test_extra_templates.py`

**Interfaces:**
- Produces:
  - `FloorPlan(name: str, classes: tuple[str, ...], slots: tuple[tuple[str, tuple[float, float, float, float]], ...], margins: tuple[float, ...] = (0.04, 0.07))`. Slot boxes are `(x, y, w, h)` as fractions of the content box (the canvas minus the margin).
  - `EXTRA_PATH = Path(__file__).with_name("extra_templates.json")`, overridable by env `CHAMELEON_EXTRA_TEMPLATES`
  - `load_floor_plans(path: Path | None = None) -> list[FloorPlan]`, cached per path
  - In `templates.py`: `templates_for(aspect: str) -> list[str]` (built-ins first, then floor plans in file order) and `param_grid(template: str) -> list[dict]` (built-ins unchanged; floor plans get `[{"margin": m} for m in plan.margins]`). `build_slots` accepts a floor-plan name.
- File format: `{"floor_plans": [{"name": "square_split", "classes": ["square"], "margins": [0.04, 0.07], "slots": [{"name": "text", "box": [0, 0, 0.5, 0.85]}, …]}]}`
- Validation (skip the plan with one `[distill]` warning, keep the others):
  - the name is unique and isn't `stack`, `columns` or `row`;
  - the classes are valid aspect classes;
  - slot names come from {`hero`, `text`, `logo`, `cta`, `badge`, `minor`}, each at most once;
  - `text` and `minor` are both present;
  - each box has `w, h > 0`, `x, y ≥ 0` and `x + w, y + h ≤ 1` (1e-6 tolerance);
  - no two slots overlap by more than 1% of the smaller slot's area.

- [ ] **Step 1: Write the failing tests:**
  - `test_no_file_no_change`: `templates_for(c)` equals today's `TEMPLATES_BY_CLASS[c]` for all six classes
  - `test_plan_offered_and_scaled`: a `square_split` plan with slot `hero [0.5, 0, 0.5, 1]`; `build_slots("square_split", 1000, 1000, {"margin": 0.04})` gives a hero box of `(500.0, 40.0, 460.0, 920.0)`
  - `test_invalid_plans_skipped`, parametrised over the six validation failures: each bad plan is skipped and a valid plan in the same file still loads
  - `test_floor_plan_candidates_pass_gates`: with a valid square-split and a tall-split plan, every sample at 300×250 and 1080×1350 yields at least one candidate whose id starts with the plan's name, and `check_gates` returns `[]` for each
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_extra_templates.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `generate.py`, replace the `TEMPLATES_BY_CLASS[...]` / `PARAM_GRID[template]` lookups with `templates_for(...)` / `param_grid(template)`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(layout): load extra floor plans from extra_templates.json`

### Task F3: ad corpus

**Files:**
- Create: `chameleon/dev/corpus/designs/*.json`, `chameleon/dev/corpus/README.md`
- Modify: `chameleon/.gitignore` (add `dev/data/`)
- Test: `chameleon/tests/test_corpus.py`

**Interfaces:**
- Produces: `dev/corpus/designs/<task_id>.json` = `{"task_id": str, "source": <design>, "split": "train" | "holdout", "origin": "sample" | "hand" | "claude-drafted" | "imported", "expects_missing_asset": bool}`, plus `dev.corpus.load_corpus(split: str | None = None) -> list[dict]` in `chameleon/dev/corpus/__init__.py`.
- What to build: **at least 30 ads, at least 8 of them holdout.**
  - The 3 samples go in, as `train`.
  - The rest are drafted by Claude in this session, or by `claude -p`, in the schema, using only `brand_kit/assets` and `brand_kit/fonts`. Angelo reviews each render and rejects ugly or broken sources.
  - Vary the source size (square, story, banner, portrait), the element count (3–20), with and without groups, with no image, with a long legal line, with 1–5 badges, with a rotated element, and with a missing asset.
  - Importing real ads from the Crello dataset is a separate spike, only if 30 drafted ads aren't enough.

- [ ] **Step 1: Write the failing tests:**
  - `test_corpus_size`: ≥ 30 designs, ≥ 8 holdout, unique `task_id`s
  - `test_assets_and_fonts_exist`: every `asset_id` exists (unless `expects_missing_asset`) and every `font_family` is in the brand kit
  - `test_designs_plan_and_render`: `heuristic_plan(source)` and `render_to_image(source, max_size=256)` both succeed for each design
- [ ] **Step 2: Run them.** Expected: FAIL (no corpus yet).
- [ ] **Step 3: Draft the designs,** render each to `dev/data/corpus_review/` and review them; keep the good ones.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/test_corpus.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(dev): ad corpus for distillation (N train, M holdout)`

### Task F4: teacher runner

**Files:**
- Create: `chameleon/dev/distill/__init__.py`, `chameleon/dev/distill/teacher.py`
- Test: `chameleon/tests/test_teacher.py`

**Interfaces:**
- Produces:
  - `TeacherError(Exception)`
  - `ClaudeCodeTeacher(model: str = "claude-opus-5-5", timeout_s: float = 600, cache_dir: Path = Path("dev/data/teacher_cache"), run=subprocess.run)`, with `.name == f"claude-code:{model}"` and `.ask(prompt: str, schema: dict, images: Sequence[Path] = ()) -> dict`
- Command (checked against CLI 2026-10-20: the answer is in the top-level `structured_output` key):
  `["claude", "-p", <prompt + "\n\nRead these image files:\n" + absolute paths>, "--model", model, "--allowedTools", "Read", "--json-schema", json.dumps(schema), "--output-format", "json", "--max-turns", "8"]`
- Behaviour:
  - parse stdout as JSON; `is_error` true, a non-zero exit code or no `structured_output` → `TeacherError`;
  - 2 retries after 30 s and 120 s;
  - cache key = sha256 of (name, prompt, schema, the bytes of each image); a cache hit doesn't run the command.

- [ ] **Step 1: Write the failing tests,** using a fake `run` that returns canned stdout:
  - `test_returns_structured_output`
  - `test_error_result_raises` (the call is retried twice, then raises `TeacherError`)
  - `test_cache_hit_skips_run`
  - `test_changed_image_misses_cache`
  - `test_image_paths_are_absolute_in_prompt`
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** Expected: PASS. Then do one real smoke call by hand (Haiku, tiny prompt) and check it returns a dict.
- [ ] **Step 5: Commit** `feat(dev): Claude Code teacher runner with cache and retries`

### Task F5: candidate pool and pair sampling

**Files:**
- Create: `chameleon/dev/distill/pool.py`
- Test: `chameleon/tests/test_pool.py`

**Interfaces:**
- Consumes: `load_corpus` (F3), `generate_candidates`, `safe_fallback`, `check_gates`, `geometry_metrics`, `pixel_metrics`, `render_to_image`
- Produces:
  - `build_pool(designs: list[dict], targets: list[dict], out_dir: Path) -> list[dict]` writes one PNG per distinct passing candidate (`max_size=768`) to `out_dir/renders/<task_id>/<target>/<n>.png`, and one row per candidate to `out_dir/pool.jsonl`: `{"cell": "<task_id>@<target>", "task_id", "split", "target", "candidate_id", "png", "is_fallback", "metrics": {12 keys}}`. Exact duplicate designs are dropped. Each ad's original is also rendered once, to `out_dir/renders/<task_id>/source.png`, for the judge (F6).
  - `sample_pairs(rows: list[dict], per_cell: int = 6, seed: int = 0) -> list[tuple[dict, dict]]`. Pairs stay within a cell, and nothing is paired with itself. About half are among the top 6 by today's total (close contests); the rest pair a top-6 candidate with a random one.
- Defaults: 6 of the 13 targets per ad, chosen by seed, always covering every aspect class across the corpus. For 30 ads that's about 30 × 6 × 6 ≈ 1,080 pairs.

- [ ] **Step 1: Write the failing tests:**
  - `test_pool_rows_have_metrics_and_pngs` (Level 1 sample × 1 target)
  - `test_sample_pairs_deterministic`
  - `test_pairs_within_cell_no_self_pairs`
  - `test_per_cell_cap`
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dev): candidate pool and pair sampling for distillation`

### Task F6: judge, rubric and pilot

**Files:**
- Create: `chameleon/dev/distill/rubric.md`, `chameleon/dev/distill/judge.py`
- Test: `chameleon/tests/test_judge.py`

**Interfaces:**
- Consumes: a teacher with `.name` and `.ask(prompt, schema, images)` (F4), pairs from F5
- Produces:
  - `judge_pairs(pairs, teacher, source_png: Callable[[str], Path], out: Path, pairs_per_call: int = 4, swap_share: float = 0.2, seed: int = 0) -> list[dict]` appends to `out` (JSONL) `{"cell", "a", "b", "winner": "a" | "b" | "tie", "confidence": 1-3, "reason", "teacher", "swapped": bool}`. It resumes by skipping pairs this teacher already judged in `out`.
  - `position_bias(judgments) -> float`: the share of non-tie answers where the first-shown candidate won.
  - `swap_agreement(judgments) -> float`: agreement between each pair and its swapped re-ask.
- Each call shows the original ad plus `pairs_per_call` pairs (so 1 + 8 images), and the rubric from `rubric.md`. The order inside each pair is shuffled by seed.
- **Provisional rubric** (replace with the official one when we get it):
  1. all text readable at the target size;
  2. every piece visible, none cropped awkwardly;
  3. the most important pieces (headline, product, button) are the most prominent;
  4. still recognisably the same ad as the original;
  5. balanced use of space, with no large accidental gaps and nothing crammed;
  6. nothing looks broken or overlapping by accident.

- [ ] **Step 1: Write the failing tests,** using a fake teacher:
  - `test_resume_skips_judged_pairs`
  - `test_order_shuffled_and_recorded`
  - `test_swap_share_reasks_swapped`
  - `test_position_bias_and_swap_agreement` (pure functions on hand-made judgements)
  - `test_teacher_failure_keeps_progress`: the teacher raises on the second batch, and the first batch is still in `out`
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Pilot.**
  1. Make 25 pairs from the 3 samples.
  2. Claude judges them in-session, and Angelo judges the same 25 blind.
  3. Accept the rubric if Claude and Angelo agree on ≥ 18 of 25, the position bias is between 0.35 and 0.65, and the swap agreement is ≥ 0.8. Otherwise revise `rubric.md` and repeat.
  4. Record the numbers in `dev/distill/PILOT.md`.
- [ ] **Step 5: Commit** `feat(dev): pairwise judge with resume, swap checks and provisional rubric`

---

## Phase B1: hand-drawn floor plans (quick win, start right after F2)

### Task B1: 2–3 floor plans for tall, square and very-tall targets

**Files:**
- Create: `chameleon/engine/pipeline/layout/extra_templates.json`
- Test: extend `chameleon/tests/test_extra_templates.py`

- [ ] **Step 1: Write a failing test,** `test_shipped_floor_plans_valid`: the shipped file loads with no warnings, and it has ≥ 1 plan for each of `square`, `tall` and `very_tall`.
- [ ] **Step 2: Draw the plans.**
  - **Square split:** text and CTA on one side, product on the other, logo top-left.
  - **Poster:** logo on top, a big hero band, the headline over a calm area below it, then the CTA.
  - **Tall split:** hero on top, text and CTA side by side under it.
  - Render each on all 3 samples, and pick slot fractions by eye using `python -m dev.run_matrix`.
- [ ] **Step 3: Check.**
  1. `.venv/bin/python -m pytest -q` passes.
  2. `python -m dev.run_matrix` shows each plan winning at least 1 cell and looking right on the contact sheet.
  3. Drop any plan that never wins.
- [ ] **Step 4: Commit** `feat(layout): hand-drawn floor plans for tall and square targets`

---

## Phase D: learned grader (after F1, F3, F6)

### Task D1: fit the weights

**Files:**
- Create: `chameleon/dev/distill/train_grader.py`
- Test: `chameleon/tests/test_train_grader.py`

**Interfaces:**
- Consumes: pool rows (F5), judgements (F6), `GraderWeights` and `DEFAULT_WEIGHTS` (F1)
- Produces:
  - `fit_weights(examples: list[tuple[dict, dict, float]], l2: float = 1e-2, seed: int = 0) -> GraderWeights`. Each example is `(metrics_a, metrics_b, y)` with `y` = 1 if A won and 0 if B won; ties are dropped.
  - `pairwise_accuracy(weights: GraderWeights, examples) -> float`
  - `split_examples(pool_rows, judgments) -> tuple[list, list]`, as (train, holdout) by the ad's `split`. A holdout ad never appears in train.
  - CLI: `python -m dev.distill.train_grader --pool dev/data/pool.jsonl --judgments dev/data/judgments.jsonl [--write]`
- Method: a Bradley–Terry / logistic model on `metrics_a − metrics_b` over the 12 metric names, with no intercept and weights kept ≥ 0 (projected gradient descent, numpy only). Then convert to `GraderWeights`:
  - `geometry_share` = raw geometry sum ÷ raw total;
  - each group normalised to sum to 1.
- `--write` writes `engine/pipeline/grading/weights.json` (with `source: "learned:<date>-<n examples>"`) **only if** holdout accuracy beats `DEFAULT_WEIGHTS` by ≥ 0.03. It always prints both accuracies and the weights table.

- [ ] **Step 1: Write the failing tests:**
  - `test_recovers_known_weights`: synthetic examples labelled by a known `GraderWeights` → holdout accuracy ≥ 0.9, and the top 3 weights match
  - `test_weights_nonnegative_and_normalised`
  - `test_ties_dropped`
  - `test_empty_raises` (`ValueError`)
  - `test_holdout_never_in_train`
  - `test_write_refused_without_gain` (uses `tmp_path` via `CHAMELEON_GRADER_WEIGHTS`)
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dev): fit grader weights from teacher judgements`

### Task D2: run, evaluate, ship

- [ ] **Step 1: Build the pool** for the train and holdout ads (F5), then judge the sampled pairs (F6). That's about 270 calls at 4 pairs per call; it's resumable, so it can run over several sessions.
- [ ] **Step 2: Train** with `python -m dev.distill.train_grader … --write`. Expected: holdout accuracy, learned vs default, is printed, and the file is written only on a ≥ 0.03 gain.
- [ ] **Step 3: Compare.** Run `python -m dev.run_matrix` with and without the file. Angelo reviews every cell whose winner changed; record wins, losses and draws in `dev/distill/RESULTS.md`.
- [ ] **Step 4: Check in Docker.** Run the matrix in the Linux container (README command). Every output must pass every gate.
- [ ] **Step 5: Commit** `feat(grading): learned grader weights (holdout acc X vs Y)`

---

## Later phases: one plan each, written when they start

Each of these gets its own plan file once its inputs exist. What's decided now:

### B2: teacher floor plans + clustering (after F2, F3, F6; parallel with D and A)
1. **Propose.** For each train ad × aspect class, the teacher proposes 3 floor plans in F2's JSON format. It's given the target size and the roles present (from `heuristic_plan`), and a JSON schema enforces the format.
2. **Fill and judge.** The plans are loaded through `CHAMELEON_EXTRA_TEMPLATES` pointing at a temporary file, filled with the real pieces (F5), and judged against that cell's current winner (F6).
3. **Keep and cluster.** Keep plans that beat the current winner in ≥ 50% of their cells. Turn each one into a vector: for each slot name, `[present, x, y, w, h]`. Cluster per aspect class with k-medoids, k = 2–5 chosen by silhouette score.
4. **Promote and test.** The medoids (the most typical real member of each cluster) become entries in `extra_templates.json`. Keep a medoid only if it improves holdout win rate against the built-ins.

### A: distilled ranker (after F3, F4; parallel with D and B2; gated)
1. **Label.** The teacher labels every corpus ad with roles and a ranking, through `claude -p` with node 1's existing `prompt.py` schema. Check its labels against the 3 hand labels in `dev/golden/`.
2. **Train.** Fit small models on `unit_features`: a role classifier, and a pairwise ranker for importance. Ship them as JSON weights evaluated in plain Python, with no new runtime dependency.
3. **Ship only if it wins.** It ships only if it beats `heuristic_plan` on the holdout ads (role accuracy and Kendall τ via `dev/priority_metrics.py`). Otherwise the rules stay, and the result is written up.

### C: floor-plan recommender (after B2)
- **Ships:** a JSON table of (ad description vector, aspect class, winning floor plan) from B2's judged winners. The ad description includes counts per role, headline length, hero aspect, badge count, and so on.
- **At run time:** the k = 3 nearest training ads with the same aspect class contribute their floor plans as extra candidates. This is deterministic and nothing new is installed.
- **Later:** a trained neural layout generator is the finale stretch goal, not part of this plan.
