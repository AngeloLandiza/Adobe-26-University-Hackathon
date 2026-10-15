# Stage-1 Three-Node Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the uniform-stretch baseline in `chameleon/engine/resize.py` with the three-node pipeline: priority (API LLM, with a heuristic fallback) → resize engine → grading. The plan also includes a benchmark that measures the cost, latency and ranking quality of Claude vs Gemini vs the heuristic.

**Architecture:**
- **Node 1** groups elements into units and asks an API LLM (Claude or Gemini, using JSON-schema structured output) for roles and a ranking. Any failure, or no key or network, falls back to a deterministic heuristic.
- **Node 2** builds candidate layouts from fixed slot templates per aspect class, plus a proportional reflow and a safe fallback.
- **Node 3** gates every candidate, scores geometry, renders the top 6 for pixel metrics, and returns the best.

`engine/main.py` and `renderer/` are untouched.

**Tech Stack:**
- Python 3.11 (Docker) / 3.12 (local venv)
- Pillow, numpy
- cr-renderer via the vendored `renderer/converter`
- `anthropic` SDK (Claude), `google-genai<3.0.0` (Gemini)
- pytest

**Spec:** [docs/pipeline/README.md](../../pipeline/README.md), [01-priority-node.md](../../pipeline/01-priority-node.md), [02-resize-engine.md](../../pipeline/02-resize-engine.md), [03-grading-node.md](../../pipeline/03-grading-node.md)

**Branch:** `stage1-three-node`. The plan was merged to `main` in PR #3. Tasks 1–6 are in [PR #4](https://github.com/AngeloLandiza/Adobe-26-University-Hackathon/pull/4). Implement every task on this branch.

**Status:** in progress. The team chose **native execution** (one implementer, test-first, one whole-branch review at the end). See "Execution status" below.

## Execution status (as of 2026-10-14)

| Task | Status | Commit | Tests after the task |
|---|---|---|---|
| 1 Scaffolding, types, geometry | ✅ done | `490f1ae` | 10 |
| 2 Text measurement + render helper | ✅ done | `93244fa` | 19 |
| 3 Units, hard signals, features | ✅ done | `a937af4` | 34 |
| 4 Heuristic ranker, golden labels | ✅ done | `b0208af` | 50 |
| 5 Claude + Gemini providers | ✅ done (live tests not run: no keys) | `31eb2b4` | 66 (+2 live skipped) |
| 6 Validate/repair, `plan()`, cache | ✅ done | `722b29f` | 83 (+2 live skipped) |
| 7–9 Node 2 (resize engine) | ⏳ next | | |
| 10–11 Node 3 (grading) | ⏳ | | |
| 12 Wire `resize()` | ⏳ | | |
| 13–15 Matrix, benchmark, Docker | ⏳ | | |
| Final whole-branch review | ⏳ | | |

Each task went test-first: every test was seen failing before its code was written. Each important decision was also broken on purpose, to confirm a test catches it. Most tasks ended up with more tests than the plan lists.

### Deviations in Tasks 1–6 that later tasks must follow

These change interfaces or facts the later task text assumes. **Where this list and a task's text disagree, this list wins.**

1. **Text height model (Task 2).**
   - `TextMeasurer.block_height` = `(lines − 1) × font_size × line_height + glyph height`, where glyph height is descent − ascent of the font cr-renderer draws, at the whole-number font size it uses.
   - The plan's `lines × font_size × line_height` failed a real-render test, and the plan's test pinning `72.0` was replaced.
   - Tasks 7–11 just call the measurer and need no change. Never hand-compute text height.
2. **`fits()` checks drawn line widths, including `letter_spacing`** (Task 2), not only the longest word. A new public method, `TextMeasurer.line_widths(el, font_size, width)`, returns each line's drawn width.
3. **No new `text_width` method in Task 7.** Single-line width is `max(measurer.line_widths(el, f, float("inf")))`. The pre-flight plan to add one is superseded.
4. **Unit features gained `aspect` (pixel w/h) and `flags`** (Tasks 3–4).
5. **Detected buttons** need exactly one text covering ≥ 25% of the shape (Task 3).
6. **Clashing group ids:** a `group_id` equal to another element's id gets unit id `<group_id>#group` (Task 3).
7. **`_assemble` returns new `Unit` copies** and never changes the units it's given (Task 4).
8. **Background is a hard signal in `repair()`** (Task 6). The LLM can't label a unit `background`, and leaving background units out isn't counted as a repair.
9. **`plan()` never raises,** even on a design it can't analyse: it returns one unranked unit per element (Task 6). Task 12 can still keep its own fallback.
10. **`textmeasure.BOLD_WEIGHTS`** is public (renamed from `_BOLD_WEIGHTS`) and reused by `features.py`.
11. **`requirements.txt` already lists** `numpy`, `anthropic` and `google-genai<3.0.0` (Task 5). Task 15 only pins versions.

All the rulings with their reasons are in [docs/pipeline/01-priority-node.md](../../pipeline/01-priority-node.md) and the [PR #4 description](https://github.com/AngeloLandiza/Adobe-26-University-Hackathon/pull/4).

## Stage-1 reality check (read first)

Stage-one scoring runs `docker run --network none`. **An API-key LLM can't be reached during scoring.** This plan therefore builds node 1 so that:

- **in development**, with a key and network, the API LLM ranks units, and the benchmark (Task 14) measures it against the heuristic;
- **in the scored Docker run**, with no key and no network, node 1 falls back to the heuristic automatically and the engine still produces valid, good output.

If the benchmark shows the API LLM ranks clearly better than the heuristic, getting that quality into stage-1 scoring means either a local model (a later plan) or the organisers allowing network access. That decision comes after Task 14's numbers.

## Scope

- **In:** all three nodes; Claude + Gemini providers; the heuristic fallback; the dev matrix contact sheets; the provider benchmark; the Docker no-network smoke test.
- **Out (later plans):**
  - local LLM (llama.cpp);
  - grader-weight calibration with an LLM judge;
  - VLM judge;
  - LoRA;
  - finale add-on.
- **Simplifications vs the spec:**
  - templates use fixed fractional slot layouts instead of a flexbox solver;
  - background and decoration ids are derived from unit roles rather than stored as separate lists;
  - the pixel metrics render each top-K candidate once, with text removed (the placeholder metric is computed from geometry);
  - units that can't fit at their role's readability floor are retried at a 6 px hard floor and marked degraded, rather than dropping the candidate.

## Recommendations and decisions

Everything below is a recommendation or a call made while writing this plan. Challenge any of it in review; the tasks implement it as written unless the team changes it here first.

### Decisions baked into the tasks

1. **API LLM in development, heuristic in the scored run.** Node 1 calls Claude or Gemini whenever a key and network are available. Stage-1 scoring has neither, so the rule-based ranker produces the plan there. Both paths go through the same validation, so node 2 can't tell them apart.
2. **Claude settings:**
   - default model `claude-opus-5-5`;
   - `output_config.effort = "low"`: a labelling task doesn't need deep reasoning, and low effort keeps tokens and latency down;
   - `max_tokens = 4096`, leaving room for thinking plus the JSON;
   - structured output through `output_config.format` (JSON schema);
   - **no `temperature`** (rejected with a 400 on Opus 5.5).
3. **Claude refusal fallback is on.** `fallbacks: "default"` with beta `server-side-fallback-2026-07-01`: if Claude declines a request, Anthropic re-runs it on its recommended fallback model inside the same call. A refusal that still comes back is treated as a provider error and the heuristic takes over. Drop this if the team prefers fewer beta features. Refusals on marketing copy should be very rare either way.
4. **Gemini settings:**
   - default model `gemini-3.8-flash`, which Google currently recommends for new projects;
   - `google-genai` pinned below 3.0.0, as the SDK's README advises;
   - structured output through `response_mime_type="application/json"` + `response_json_schema`.
5. **Models are swappable without code changes.** `CHAMELEON_CLAUDE_MODEL` and `CHAMELEON_GEMINI_MODEL` override the defaults, and the benchmark takes a list of models per provider. This lets the team test cheaper models (e.g. `claude-haiku-5-5`, `gemini-3.5-flash-lite`) or stronger ones without editing code.
6. **Provider choice order for `CHAMELEON_LLM=auto`:** Claude if `ANTHROPIC_API_KEY` is set, otherwise Gemini if `GEMINI_API_KEY`/`GOOGLE_API_KEY` is set, otherwise the heuristic.
7. **One schema, two providers.** The JSON schema is built per request from that design's unit ids (as an `enum`), so neither model can invent ids. Both providers receive the identical system prompt, user message and schema, which keeps the benchmark comparison fair.
8. **The LLM labels; code decides the facts** (from the design docs):
   - units, not raw elements, are ranked;
   - the ranking doesn't depend on the target size;
   - background detection and missing-asset demotion override whatever the LLM says;
   - output needing too many repairs (> 30% of units) is thrown away in favour of the heuristic.
9. **The heuristic is the baseline the LLM must beat.** If the benchmark doesn't show a clear win, the heuristic stays the node 1 answer.
10. **Response cache for development** (`CHAMELEON_LLM_CACHE`): repeated runs over the same designs don't pay again. The benchmark disables it so latency and cost are real.
11. **Readability floors.** Per-role floors (headline 14 px, cta 11, subheadline 10, badge 9, legal 7, others 9, scaled up on large canvases) plus a 6 px hard floor. A unit that only fits at the hard floor is kept and marked degraded, which costs score in proportion to its rank. This matters for Level 3's 150-character headline in 728×90.
12. **Condensed buttons.** When rigid scaling would shrink a button label below its floor, the pill is rebuilt around a one-line label instead.
13. **Never hide elements.** No `opacity` changes, no zero-size boxes, nothing fully covered by another element (gate G9).
14. **Styling stays fixed.** `text_align`, colours, `line_height` and the other style fields are frozen; only geometry, `font_size`, `letter_spacing`, `corner_radius` and `stroke_width` change.
15. **Grader weights are placeholders.** The starting weights in Task 11 come from the design doc. Calibrating them against an LLM judge is a later plan; until then, the Task 13 contact sheets are how we judge quality.
16. **Keys stay out of git and out of the image.** Keys come only from environment variables. `.env` is gitignored (Task 1). Keys are never baked into the Docker image, and are passed with `-e` only for the optional dev run in Task 15.

### Cost estimate (rough, before measurement)

Only node 1 calls an LLM, once per task. Nodes 2 and 3 run locally and cost nothing in API fees.

The table assumes about **2,000 input tokens and 1,000 output tokens per task** (output includes thinking tokens, which both providers bill as output). Prices are per 1M tokens, from Anthropic's model table and Google's pricing page as of 2026-10-13. **Task 14 replaces these estimates with measured numbers.**

| Model | Input / output price | ≈ per task | ≈ per 1,000 tasks |
|---|---|---|---|
| `claude-opus-5-5` (default) | $4.00 / $20.00 | $0.028 | $28 |
| `claude-sonnet-5-5` | $2.00 / $10.00 | $0.014 | $14 |
| `claude-haiku-5-5` | $0.10 / $0.50 | $0.0007 | $0.70 |
| `gemini-3.8-flash` (default) | $0.75 / $3.75 | $0.005 | $5.25 |
| `gemini-3.5-flash-lite` | $0.30 / $2.50 | $0.003 | $3.10 |
| `gemini-4-argon` | $2.00 / $10.00 | $0.014 | $14 |

Scheduled price changes on Google's page: Gemini 3.8 Flash rises to $1.50 / $7.50 from 2027-01-01, and Gemini 4 Argon to $4.00 / $20.00 from 2026-12-01. Update `pricing.py` if those dates pass during the hackathon.

At these sizes API cost is small for development. The real questions are ranking quality vs the heuristic, and whether an API can be used in scoring at all.

### Decision point after Task 14 (the benchmark)

1. **Get more test data first.** Three sample designs make the comparison anecdotal. Before deciding anything, add **at least 5 more source designs** to `input/`, each with a golden label file in `dev/golden/` (same format as Task 4). Vary them: no groups, no images, many badges, long copy, different source sizes.
2. **Read the benchmark summary** (mean per provider/model) on role accuracy, Kendall τ, top-3 recall, latency and cost.
3. **Decide.** Suggested threshold: the LLM "clearly wins" if its mean top-3 recall or role accuracy beats the heuristic by **≥ 0.10** on the expanded golden set.
   - **LLM doesn't clearly win:** keep the heuristic as node 1 for stage 1. Use the LLM's disagreements with the heuristic to improve the heuristic rules.
   - **LLM clearly wins:** choose between:
     - (a) asking the organisers whether stage-1 scoring can have network access and an API key (organiser question 0 below);
     - (b) a separate plan for a local model baked into the Docker image (llama.cpp, as in the original design docs), with the same benchmark run against it;
     - (c) feeding the LLM's labels back into better heuristic rules.
4. **Record the benchmark summary and the decision** in the PR description, and in `docs/pipeline/README.md` §9.

### Questions for the organisers (most important first)

0. **Is network access available during stage-1 scoring?** Could the engine call an external LLM API with our key? This decides whether the API path can ever count in stage 1.
1. Can decorative shapes bleed off the canvas (Level 2's corner circle), or must everything stay inside? This plan clamps everything inside.
2. How small is acceptable for minor text (legal, badges) in extreme formats like 728×90?
3. What's the Docker image size limit? This matters if we go the local-model route.
4. What exactly is the reference hardware (cores, RAM, whether amd64 runs emulated on the M5 Pro)?
5. Will hidden tasks use assets outside `brand_kit/`?
6. What is the full scoring rubric? Grader weights should mirror it.

### Unverified details, checked during execution

| Item | Where it's checked | Result |
|---|---|---|
| Gemini `usage_metadata` field names (`prompt_token_count`, `candidates_token_count`, `thoughts_token_count`) | Task 5 | ✅ **Confirmed** in the installed `google-genai` 2.30.1, and by a real-SDK test against a canned API response |
| Unit of the Gemini SDK's HTTP timeout (`types.HttpOptions(timeout=...)`) | Task 5 | ✅ **Milliseconds**, per the SDK's field description; `GeminiProvider` converts seconds |
| Whether the installed `anthropic` SDK accepts `fallbacks="default"` on `client.beta.messages.create` | Task 5 | ✅ **Yes**: `anthropic` 1.13.0 types it as a list of fallbacks or the literal `"default"`. A real-SDK test confirms it's sent in the request body. |
| cr-renderer's vertical text placement matches `lines × font_size × line_height` | Task 2 | ❌ **It doesn't.** The last line reaches the full glyph height and nothing is clipped. Fixed in `block_height` (see "Execution status"). |
| The live APIs accept our schema (Claude: enums under structured output; Gemini: `additionalProperties: false`), plus real latency and cost | Task 5 live tests, Task 14 | ⏳ **Not run yet** (no keys). Run `pytest tests/test_providers.py -m live -s` with keys. If an API rejects the schema, `plan()` falls back to the heuristic. |
| Template slot fractions and readability floors produce good layouts | Task 13 contact sheets | ⏳ Tune the values in Tasks 7 and 9 and update this plan's tables |

### Execution approach

**Chosen: Native** (team decision, 2026-10-14). One person (or Claude in one session) implements the tasks in order, and one fresh reviewer checks the whole branch at the end.
- It's the cheaper option, which fits a cost-conscious team.
- Every task's tests pin the interfaces the next task depends on.

The alternative is **subagent-driven**: a fresh agent implements each task and a fresh reviewer checks it before the next starts. It's more thorough, at the cost of a new context per task and per review. It wasn't chosen.

---

## Global Constraints

- All commands run from `chameleon/` using its venv:
  - `.venv/Scripts/python` on Windows, `.venv/bin/python` elsewhere;
  - in the steps, `python` means that interpreter.
- Code must run on Python 3.11 (Docker base image `python:3.11-slim`). No 3.12-only syntax.
- Never edit `engine/main.py` or anything under `renderer/`. Import from `renderer.converter`.
- Output design: same element ids, same count and order as the source; `canvas == target_canvas` (ints).
- Every element's rotated bounding box must sit inside `[0, W] × [0, H]` with 0.5 px tolerance.
- Non-`cover` images keep the source box aspect within 2% internally (the official limit is ~5%).
- Never change `content`, `font_family`, `font_weight`, `font_style`, `type`, `asset_id`, `fit`, `z_index`, `rotation`, `opacity`, `color`, `fill_color`, `stroke_color`, `style_runs`, `text_align`, `line_height`, `max_lines`, `group_id`, or unknown keys such as `_note`.
- Editable fields only: `x, y, width, height, font_size, letter_spacing, corner_radius, stroke_width`. `font_size ≥ 1`.
- Never regenerate, redraw or replace asset pixels (disqualifying).
- Missing optional fields default to: `z_index` 0, `rotation` 0, `opacity` 1, `line_height` 1.2, `letter_spacing` 0, `font_weight` `"regular"`, `font_style` `"normal"`, `fit` `"contain"`.
- `resize()` never raises and always returns schema-valid JSON.
- Node 1 with no key or no network must fall back to the heuristic within the provider timeout (60 s default).
- **Claude:**
  - default model `claude-opus-5-5` (env `CHAMELEON_CLAUDE_MODEL` overrides);
  - call `client.beta.messages.create(...)` with `betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`, `output_config={"effort": "low", "format": {"type": "json_schema", "schema": ...}}`, `max_tokens=4096`;
  - **never pass `temperature`** (400 on Opus 5.5);
  - check `stop_reason` before reading content.
- **Gemini:**
  - default model `gemini-3.8-flash` (env `CHAMELEON_GEMINI_MODEL` overrides);
  - call `client.models.generate_content(model=..., contents=..., config=types.GenerateContentConfig(system_instruction=..., response_mime_type="application/json", response_json_schema=...))`.
- Provider selection: env `CHAMELEON_LLM` ∈ `auto` (default) | `claude` | `gemini` | `heuristic`. `auto` picks:
  - Claude if `ANTHROPIC_API_KEY` is set;
  - otherwise Gemini if `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set;
  - otherwise the heuristic.
- Determinism: fixed enumeration order; ties are broken by `candidate_id` ascending.
- Debug files are written only when `CHAMELEON_DEBUG=1`.
- Commit messages end with the `Co-Authored-By` trailer this repo's sessions use.

## Review Focus

1. **A design with no text, or no images.** The heuristic finds no headline or hero, the plan is still valid, and `resize()` returns gate-passing output. Tests: Task 4 `test_heuristic_handles_no_text_no_images`, Task 12 `test_resize_text_only_and_image_only`.
2. **A single word wider than its slot even at the floor font.** The unit is relocated or the candidate dropped; text never overflows its box. Test: Task 7 `test_overlong_word_never_overflows`.
3. **LLM output cut off (`stop_reason == "max_tokens"`), refused, or not JSON.** The heuristic plan is used and `stats.error` records why. Tests: Task 5 `test_claude_refusal_and_truncation_raise`, Task 6 `test_provider_error_falls_back`.
4. **Elements missing optional fields** (`z_index`, `rotation`, `opacity`, `line_height`). The defaults above are used everywhere, with no `KeyError`. Tests: Task 3 `test_units_tolerate_missing_optional_fields`, Task 8 `test_fallback_with_missing_optional_fields`.
5. **Ties between same-role units** (three badges). The same ranking and byte-identical output on every run. Tests: Task 4 `test_heuristic_ranking_is_deterministic`, Task 12 `test_resize_is_deterministic`.

---

### Task 1: Scaffolding, shared types, geometry helpers

**Files:**
- Create: `chameleon/engine/pipeline/__init__.py` (empty), `chameleon/engine/pipeline/types.py`, `chameleon/engine/pipeline/geometry.py`
- Create: `chameleon/tests/__init__.py` (empty, so tests can `from tests.conftest import ...`), `chameleon/tests/conftest.py`, `chameleon/tests/test_geometry.py`, `chameleon/pytest.ini`, `chameleon/requirements-dev.txt`
- Create: `chameleon/dev/__init__.py` (empty), `chameleon/dev/targets.json`
- Modify: `chameleon/.gitignore` (add `.cache/` and `.env`)

**Interfaces:**
- Produces in `types.py`:
  - `ROLES: tuple[str, ...] = ("background","decoration","headline","subheadline","cta","logo","hero_image","badge","secondary_image","legal","other_text")`
  - `ALPHA = 0.7`
  - `@dataclass Unit(unit_id: str, element_ids: list[str], kind: str, role: str | None = None, rank: int | None = None, flags: list[str] = [])`. `kind` ∈ `single|container_label|group`.
  - `@dataclass LLMCallStats(provider: str, model: str, latency_s: float = 0.0, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0, repaired_fraction: float = 0.0, cached: bool = False, error: str | None = None)`
  - `@dataclass PriorityPlan(source: str, units: list[Unit], stats: LLMCallStats | None = None)` with methods:
    - `ranked_units() -> list[Unit]`: rank not None, sorted by rank;
    - `background_unit_ids() -> list[str]`;
    - `decoration_unit_ids() -> list[str]`;
    - `unit_for_element(element_id: str) -> Unit`;
    - `weight(unit_id: str) -> float`: `rank ** -ALPHA`, or 0.0 when unranked.
  - `@dataclass Candidate(candidate_id: str, design: dict, template: str, params: dict, degraded_unit_ids: list[str] = [], is_fallback: bool = False)`
  - `@dataclass GradeReport(candidate_id: str, passed_gates: bool, failed_gates: list[str], metrics: dict[str, float], rendered: bool, total: float)`
- Produces in `geometry.py`:
  - `Box = tuple[float, float, float, float]` (x, y, w, h)
  - `element_box(el: dict) -> Box`
  - `rotated_aabb(el: dict) -> Box`: rotation in degrees about the element centre
  - `overlap_area(a: Box, b: Box) -> float`
  - `union_box(boxes: list[Box]) -> Box`
  - `inside_canvas(box: Box, width: float, height: float, tol: float = 0.5) -> bool`
  - `fit_aspect(aspect: float, slot: Box, align: str = "center") -> Box`: the largest box with that w/h inside the slot; `align` ∈ `center|left|right|top|bottom`.
- Produces in `conftest.py`:
  - fixtures `l1`, `l2`, `l3` (the parsed task dicts from `input/`);
  - `measurer` (session-scoped; defined in Task 2, so add it then);
  - `targets` (list from `dev/targets.json`);
  - `element(task, id) -> dict`, which returns the source element.

- [ ] **Step 1: Write `dev/targets.json` and the failing tests**

`dev/targets.json` is a list of `{"name", "width", "height"}`:

| name | width × height |
|---|---|
| instagram_post | 1080×1080 |
| instagram_portrait | 1080×1350 |
| story | 1080×1920 |
| youtube_thumbnail | 1280×720 |
| linkedin_banner | 1584×396 |
| facebook_cover | 820×312 |
| leaderboard | 728×90 |
| medium_rectangle | 300×250 |
| wide_skyscraper | 160×600 |
| billboard | 970×250 |

```python
# tests/test_geometry.py
import pytest
from engine.pipeline.geometry import rotated_aabb, overlap_area, fit_aspect, inside_canvas
from engine.pipeline.types import PriorityPlan, Unit

def test_rotated_aabb_matches_ribbon():
    x, y, w, h = rotated_aabb({"x": 740, "y": 30, "width": 300, "height": 54, "rotation": -12})
    assert w == pytest.approx(304.67, abs=0.05) and h == pytest.approx(115.19, abs=0.05)
    assert x + w / 2 == pytest.approx(890) and y + h / 2 == pytest.approx(57)

def test_unrotated_aabb_is_the_box():
    assert rotated_aabb({"x": 1, "y": 2, "width": 3, "height": 4}) == (1, 2, 3, 4)

def test_overlap_area():
    assert overlap_area((0, 0, 10, 10), (5, 5, 10, 10)) == 25
    assert overlap_area((0, 0, 10, 10), (10, 0, 5, 5)) == 0

def test_fit_aspect():
    assert fit_aspect(1.0, (0, 0, 200, 100)) == (50, 0, 100, 100)
    assert fit_aspect(1.0, (0, 0, 200, 100), align="left") == (0, 0, 100, 100)

def test_inside_canvas_tolerance():
    assert inside_canvas((-0.4, 0, 10, 10), 100, 100)
    assert not inside_canvas((-1, 0, 10, 10), 100, 100)

def test_plan_helpers():
    plan = PriorityPlan("heuristic", [
        Unit("bg", ["bg"], "single", role="background"),
        Unit("h", ["h"], "single", role="headline", rank=1),
        Unit("c", ["c1", "c2"], "container_label", role="cta", rank=2)])
    assert [u.unit_id for u in plan.ranked_units()] == ["h", "c"]
    assert plan.background_unit_ids() == ["bg"]
    assert plan.unit_for_element("c2").unit_id == "c"
    assert plan.weight("h") == 1.0 and plan.weight("bg") == 0.0
    assert plan.weight("c") == pytest.approx(2 ** -0.7)
```

`pytest.ini`:
- `testpaths = tests`
- `markers = live: calls real LLM APIs (needs keys + network)`
- `addopts = -m "not live"`

`requirements-dev.txt`: `pytest`.

- [ ] **Step 2:** Run `python -m pip install -r requirements-dev.txt && python -m pytest tests/test_geometry.py -v`. Expected: FAIL (`ModuleNotFoundError: engine.pipeline.types`).
- [ ] **Step 3:** Implement `types.py` and `geometry.py` per the Interfaces block.
- [ ] **Step 4:** Run `python -m pytest tests/test_geometry.py -v`. Expected: 6 passed.
- [ ] **Step 5: Commit**

```bash
git add .gitignore pytest.ini requirements-dev.txt engine/pipeline dev tests
git commit -m "feat(pipeline): shared types, geometry helpers, test scaffolding"
```

---

### Task 2: Renderer-exact text measurement and render helper

**Files:**
- Create: `chameleon/engine/pipeline/textmeasure.py`, `chameleon/engine/pipeline/render.py`
- Modify: `chameleon/tests/conftest.py` (add the session `measurer` fixture)
- Test: `chameleon/tests/test_textmeasure.py`

**Interfaces:**
- Consumes: `renderer.converter.brand_fonts.load_brand_font_manager(fonts_dir)`, `renderer.converter.wrap.char_widths(fm, family, weight, style, size, text)`, `renderer.converter.wrap.wrap_text(text, widths, max_width)`, `renderer.converter.convert.render_design_to_png(design, assets_dir, png_path, max_size=...)`.
- Produces: `class TextMeasurer(fonts_dir: str | Path = "brand_kit/fonts")` with:
  - `wrap(el: dict, font_size: float, width: float) -> str`
  - `line_count(el, font_size, width) -> int`
  - `block_height(el, font_size, width) -> float`: `line_count × font_size × line_height`, with `line_height` defaulting to 1.2
  - `longest_word_width(el, font_size) -> float`
  - `fits(el, font_size, width, height) -> bool`: `block_height ≤ height + 1`, the longest word ≤ `width`, and `line_count ≤ max_lines` when set
  - `fit_font_size(el, width, height, lo: float, hi: float) -> float | None`: the largest multiple of 0.5 in `[lo, hi]` that fits, or `None` if `lo` doesn't fit
- Produces: `render_to_image(design: dict, assets_dir: str | Path = "brand_kit/assets", max_size: int = 1024) -> PIL.Image.Image` (RGB). It renders to a temp file and reads it back. The scale is `img.width / design["canvas"]["width"]`.

Weight and style must be derived exactly as `renderer/converter/convert.py` does (`_normalize_weight`, `_weight_style_to_font_style`). Mirror the logic; don't import the private names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_textmeasure.py
import numpy as np, pytest
from engine.pipeline.render import render_to_image
from tests.conftest import element

def test_short_headline_one_line(measurer, l1):
    assert measurer.line_count(element(l1, "headline"), 60, 800) == 1

def test_narrow_box_wraps(measurer, l1):
    assert measurer.line_count(element(l1, "headline"), 60, 300) >= 2

def test_block_height_uses_line_height(measurer, l1):
    assert measurer.block_height(element(l1, "headline"), 60, 800) == pytest.approx(72.0)

def test_fit_font_size_is_maximal(measurer, l3):
    el = element(l3, "headline")
    f = measurer.fit_font_size(el, 380, 70, lo=6, hi=60)
    assert f is not None and measurer.fits(el, f, 380, 70)
    assert f == 60 or not measurer.fits(el, f + 0.5, 380, 70)

def test_fit_font_size_none_when_floor_fails(measurer, l3):
    assert measurer.fit_font_size(element(l3, "headline"), 100, 10, lo=14, hi=40) is None

def test_rendered_text_stays_inside_box(measurer, l3):
    text = dict(element(l3, "headline"), x=100, y=10, width=380, height=70, z_index=1)
    text["font_size"] = measurer.fit_font_size(text, 380, 70, lo=6, hi=60)
    text["height"] = measurer.block_height(text, text["font_size"], 380)
    bg = {"id": "bg", "type": "shape", "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)",
          "x": 0, "y": 0, "width": 728, "height": 90, "z_index": 0}
    img = render_to_image({"canvas": {"width": 728, "height": 90}, "elements": [bg, text]})
    s = img.width / 728
    ys, xs = np.nonzero(np.asarray(img.convert("L")) > 128)
    assert xs.min() >= (100 - 2) * s and xs.max() <= (480 + 2) * s
    assert ys.min() >= (10 - 2) * s and ys.max() <= (10 + text["height"] + 2) * s
```

- [ ] **Step 2:** Run `python -m pytest tests/test_textmeasure.py -v`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `TextMeasurer` and `render_to_image`. Load the font manager once per instance and cache `char_widths` results by `(family, weight, style, size, text)`. Run `python -m pip install numpy` (needed by the test, and by Task 11).
- [ ] **Step 4:** Run `python -m pytest tests/test_textmeasure.py -v`. Expected: 6 passed.

  If `test_rendered_text_stays_inside_box` fails vertically, cr-renderer's vertical text model differs from `lines × size × line_height`. Adjust `block_height` (e.g. add a measured top offset or descender) until it passes. That is the spec's "verify against cr-renderer" item. **Do not loosen the test.**
- [ ] **Step 5: Commit**: `git add engine/pipeline/textmeasure.py engine/pipeline/render.py tests && git commit -m "feat(pipeline): renderer-exact text measurement + render helper"`

---

### Task 3: Units, hard signals, features (node 1 pre-pass)

**Files:**
- Create: `chameleon/engine/pipeline/priority/__init__.py` (empty for now), `units.py`, `features.py`
- Test: `chameleon/tests/test_units.py`

**Interfaces:**
- Produces: `build_units(source: dict, assets_dir: str | Path = "brand_kit/assets") -> list[Unit]`.
  - **Explicit groups:** a shared non-null `group_id` gives `unit_id = group_id`. Kind is `container_label` when the group is exactly one shape + one text with the text box ≥ 90% inside the shape box; otherwise `group`.
  - **Implicit groups:** an ungrouped text ≥ 90% inside an ungrouped shape, with a higher `z_index`, joins it as `container_label` with `unit_id` = the shape's id.
  - **Everything else:** `single`, with `unit_id` = the element id.
  - **Flags:**
    - `"background"`: an image or shape covering ≥ 90% of the canvas, with the lowest `z_index` among the elements it overlaps; also sets `role="background"`;
    - `"missing_asset"`: the `asset_id` file isn't in `assets_dir`;
    - `"rotated"`: any member has `rotation != 0`.
  - **Order:** units follow the first appearance of their members in `source["elements"]`.
- Produces: `unit_features(source: dict, units: list[Unit]) -> list[dict]`, one dict per unit, in the same order. Keys:
  - identity: `unit_id, kind, types`
  - geometry: `bbox` (normalised `[x, y, w, h]`), `area_pct`, `center`, `z_index`, `opacity`, `rotation`, `is_full_canvas`
  - text: `text` (≤ 160 chars), `text_len`, `font_size`, `font_size_rank` (1 = largest among units with text), `bold`, `all_caps`, `word_count`
  - image: `asset_id`, `fit`, `missing_asset`
  - shape: `shape_type`, `has_label`
  - relations: `overlaps` (up to 3 unit_ids by overlap area, descending)

  Absent attributes are `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_units.py
from engine.pipeline.priority.units import build_units
from engine.pipeline.priority.features import unit_features

def by_id(units): return {u.unit_id: u for u in units}

def test_l2_units(l2):
    units = by_id(build_units(l2["source"]))
    assert len(units) == 9
    assert units["badge"].element_ids == ["badge_bg", "badge_label"]
    assert units["cta"].kind == "container_label"
    assert units["bg"].role == "background" and "background" in units["bg"].flags

def test_l3_flags(l3):
    units = by_id(build_units(l3["source"]))
    assert len(units) == 12
    assert "missing_asset" in units["holiday_watermark"].flags
    assert "rotated" in units["ribbon"].flags and units["ribbon"].kind == "container_label"

def test_every_element_in_exactly_one_unit(l1, l2, l3):
    for task in (l1, l2, l3):
        ids = [e for u in build_units(task["source"]) for e in u.element_ids]
        assert sorted(ids) == sorted(el["id"] for el in task["source"]["elements"])

def test_implicit_label_on_shape():
    src = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "s", "type": "shape", "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)",
         "x": 10, "y": 10, "width": 50, "height": 20, "z_index": 1},
        {"id": "t", "type": "text", "content": "Buy", "font_family": "Solstice Sans", "font_size": 10,
         "color": "rgba(255,255,255,1)", "x": 12, "y": 12, "width": 40, "height": 14, "z_index": 2}]}
    (u,) = build_units(src)
    assert u.kind == "container_label" and u.element_ids == ["s", "t"]

def test_units_tolerate_missing_optional_fields():
    src = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "i", "type": "image", "asset_id": "solstice_bottle.png", "x": 0, "y": 0, "width": 10, "height": 10}]}
    (f,) = unit_features(src, build_units(src))
    assert f["z_index"] == 0 and f["opacity"] == 1 and f["rotation"] == 0

def test_features_l1(l1):
    feats = {f["unit_id"]: f for f in unit_features(l1["source"], build_units(l1["source"]))}
    assert feats["headline"]["font_size_rank"] == 1
    assert feats["bg"]["is_full_canvas"] and feats["bg"]["area_pct"] == 1.0
```

- [ ] **Step 2:** Run `python -m pytest tests/test_units.py -v`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `build_units` and `unit_features` per the Interfaces block.
- [ ] **Step 4:** Run `python -m pytest tests/test_units.py -v`. Expected: 6 passed.
- [ ] **Step 5: Commit**: `git commit -am "feat(priority): unit building, hard signals, features"` (after `git add` of the new files).

---

### Task 4: Heuristic ranker, golden labels, `heuristic_plan()`

**Files:**
- Create: `chameleon/engine/pipeline/priority/heuristic.py`, `chameleon/dev/priority_metrics.py`
- Create: `chameleon/dev/golden/chameleon_sample_level1_warmup.json`, `..._level2_core.json`, `..._level3_advanced.json`
- Modify: `chameleon/engine/pipeline/priority/__init__.py`
- Test: `chameleon/tests/test_heuristic.py`

**Interfaces:**
- Produces in `heuristic.py`:
  - `ROLE_ORDER = ("headline","hero_image","cta","logo","subheadline","badge","secondary_image","other_text","legal")`
  - `ACTION_VERBS` (shop, buy, get, order, try, learn, discover, sign, book, start, explore, join, download, subscribe, save, claim)
  - `LEGAL_MARKERS` ("responsibly", "terms", "©", "rights reserved", "void where", "conditions apply")
  - `heuristic_roles(features: list[dict]) -> dict[str, str]`
  - `heuristic_ranking(features: list[dict], roles: dict[str, str]) -> list[str]`: content units only, ordered by `ROLE_ORDER`, then prominence (area for images; `font_size × text_len` for text) descending, then `(center y, center x)`, then `unit_id`.
- **Role rules**, applied in order:
  1. a `background` flag → `background`;
  2. a shape-only unit with no label → `decoration`;
  3. a `container_label` or rotated group: label's first word in `ACTION_VERBS` → `cta`, otherwise `badge`;
  4. an image with `missing_asset` → `secondary_image`;
  5. the image with `area_pct < 0.06`, aspect ≥ 1.5 and centre within 20% of an edge, with the highest `z_index` (ties go to the largest), → `logo`;
  6. the largest remaining image → `hero_image`; other images → `secondary_image`;
  7. single text with the largest `font_size` (bold breaks ties) → `headline`;
  8. text containing a `LEGAL_MARKERS` term, or the smallest font with `word_count > 12` and centre y > 0.8 → `legal`;
  9. text with ≤ 6 words whose first word is in `ACTION_VERBS` → `cta`;
  10. remaining text: the largest font → `subheadline`; others with font ≥ 0.5× the headline's → `subheadline`; otherwise `other_text`.
- Produces in `priority/__init__.py`:
  - `heuristic_plan(source: dict, assets_dir="brand_kit/assets") -> PriorityPlan` with `source="heuristic"` and `stats=None`;
  - `_assemble(units, roles, ranking, label, stats) -> PriorityPlan`: sets `role` and `rank`, keeps background roles, moves `missing_asset` units to the end, and renumbers ranks `1..n`. Task 6 reuses it.
- Produces in `dev/priority_metrics.py`:
  - `role_accuracy(pred: dict[str,str], gold: dict[str,str]) -> float`
  - `kendall_tau(pred: list[str], gold: list[str]) -> float`: over ids common to both, in `[-1, 1]`
  - `top3_recall(pred: list[str], gold: list[str]) -> float`: 1.0 if all of gold[:3] are in pred[:4], else the fraction found
- Golden files: `{"roles": {unit_id: role}, "ranking": [unit_ids]}`. **The team reviews these; they are judgements, not facts.**
  - L1:
    - roles `bg` background, `logo` logo, `headline` headline, `cta` cta;
    - ranking `[headline, cta, logo]`.
  - L2:
    - roles `bg` background, `deco_circle_1`/`deco_circle_2` decoration, `logo` logo, `product` hero_image, `headline` headline, `subhead` subheadline, `badge` badge, `cta` cta;
    - ranking `[headline, product, cta, logo, subhead, badge]`.
  - L3:
    - roles `bg` background, `logo` logo, `product` hero_image, `headline` headline, `subhead` subheadline, `badge1`/`badge2`/`badge3`/`ribbon` badge, `cta` cta, `legal_text` legal, `holiday_watermark` secondary_image;
    - ranking `[headline, product, cta, logo, subhead, ribbon, badge2, badge1, badge3, legal_text, holiday_watermark]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_heuristic.py
import json, pytest
from pathlib import Path
from engine.pipeline.priority import heuristic_plan
from dev.priority_metrics import role_accuracy, kendall_tau, top3_recall

SAMPLES = ["chameleon_sample_level1_warmup", "chameleon_sample_level2_core", "chameleon_sample_level3_advanced"]

@pytest.mark.parametrize("stem", SAMPLES)
def test_heuristic_matches_golden(stem):
    task = json.loads(Path(f"input/{stem}.json").read_text())
    gold = json.loads(Path(f"dev/golden/{stem}.json").read_text())
    plan = heuristic_plan(task["source"])
    roles = {u.unit_id: u.role for u in plan.units}
    ranking = [u.unit_id for u in plan.ranked_units()]
    assert role_accuracy(roles, gold["roles"]) == 1.0
    assert top3_recall(ranking, gold["ranking"]) == 1.0

def test_heuristic_ranking_is_deterministic(l3):
    a = [u.unit_id for u in heuristic_plan(l3["source"]).ranked_units()]
    assert a == [u.unit_id for u in heuristic_plan(l3["source"]).ranked_units()]
    assert a[-1] == "holiday_watermark"

def test_heuristic_handles_no_text_no_images():
    shapes = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "s", "type": "shape", "shape_type": "ellipse", "fill_color": "rgba(1,2,3,1)",
         "x": 10, "y": 10, "width": 20, "height": 20}]}
    plan = heuristic_plan(shapes)
    assert plan.units[0].role == "decoration" and plan.ranked_units() == []

def test_metrics():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["c", "b", "a"], ["a", "b", "c"]) == -1.0
    assert top3_recall(["a", "x", "b", "c"], ["a", "b", "c"]) == 1.0
    assert role_accuracy({"a": "logo", "b": "cta"}, {"a": "logo", "b": "badge"}) == 0.5
```

- [ ] **Step 2:** Run `python -m pytest tests/test_heuristic.py -v`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `heuristic.py`, `priority_metrics.py`, `heuristic_plan` and `_assemble`, and write the three golden files.
- [ ] **Step 4:** Run `python -m pytest tests/test_heuristic.py -v`. Expected: 6 passed.
- [ ] **Step 5: Commit**: `git commit -m "feat(priority): heuristic ranker, golden labels, ranking metrics"` (after `git add` of the new files).

---

### Task 5: LLM providers (Claude + Gemini), prompt, schema, pricing

**Files:**
- Create: `chameleon/engine/pipeline/priority/prompt.py`, `providers.py`, `pricing.py`
- Modify: `chameleon/requirements.txt` (add `anthropic`, `google-genai<3.0.0`, `numpy`)
- Test: `chameleon/tests/test_providers.py`

**Interfaces:**
- Produces in `prompt.py`:
  - `SYSTEM_PROMPT: str` covers:
    - the task: label and rank units of a marketing graphic;
    - one line per role in `ROLES`, with the meanings from 01-priority-node.md;
    - the importance guidance: message, product, call to action and brand first; supporting copy next; badges, secondary images, fine print and decoration last;
    - "write a ≤ 30-word summary first".
  - `build_user_message(features: list[dict]) -> str`: `json.dumps({"units": [...]}, sort_keys=True, separators=(",", ":"))`, sending the features of non-background units plus `"background_unit_ids"`.
  - `response_schema(unit_ids: list[str]) -> dict`: a JSON Schema object with `required` `summary` (string), `roles` (array of `{unit_id: enum unit_ids, role: enum ROLES}`) and `ranking` (array of enum unit_ids). `additionalProperties: false` on every object.
- Produces in `pricing.py`:
  - `PRICES_PER_MTOK` (USD input, output; as of 2026-10-13):
    - `claude-opus-5-5` (4.00, 20.00)
    - `claude-sonnet-5-5` (2.00, 10.00)
    - `claude-haiku-5-5` (0.10, 0.50)
    - `gemini-3.8-flash` (0.75, 3.75)
    - `gemini-3.5-flash-lite` (0.30, 2.50)
    - `gemini-4-argon` (2.00, 10.00)
  - `cost_usd(model: str, input_tokens: int, output_tokens: int) -> float`: `nan` for unknown models.
- Produces in `providers.py`:
  - `class ProviderError(Exception)`
  - `@dataclass LLMResult(data: dict, stats: LLMCallStats)`
  - `class ClaudeProvider(model: str | None = None, timeout_s: float = 60.0)`:
    - `name = "claude"`;
    - `complete(system: str, user: str, schema: dict) -> LLMResult`;
    - uses `anthropic.Anthropic(timeout=timeout_s, max_retries=1)` and the call shape in Global Constraints;
    - raises `ProviderError` when `stop_reason` is `"refusal"` or `"max_tokens"`, or when the first text block isn't JSON;
    - tokens come from `usage.input_tokens` / `usage.output_tokens`.
  - `class GeminiProvider(model: str | None = None, timeout_s: float = 60.0)`:
    - `name = "gemini"`;
    - same `complete`;
    - uses `genai.Client()` (key from `GEMINI_API_KEY`/`GOOGLE_API_KEY`) with an HTTP timeout. Verify the `types.HttpOptions(timeout=...)` unit (ms) against the installed SDK;
    - parses `response.text`;
    - tokens: `usage_metadata.prompt_token_count` in; `candidates_token_count + (thoughts_token_count or 0)` out (thinking is billed as output).
  - `get_provider(name: str | None = None) -> ClaudeProvider | GeminiProvider | None`: `name` defaults to env `CHAMELEON_LLM`, else `"auto"`; follows the selection rule in Global Constraints. Returns `None` for `heuristic`.
  - Module-level `import anthropic` and `from google import genai` / `from google.genai import types`, so tests can monkeypatch `providers.anthropic.Anthropic` and `providers.genai.Client`.

- [ ] **Step 1: Write the failing tests** (fake SDK clients; no network)

```python
# tests/test_providers.py
import json, os, pytest
from types import SimpleNamespace as NS
from engine.pipeline.priority import providers, prompt
from engine.pipeline.priority.providers import ClaudeProvider, GeminiProvider, ProviderError, get_provider

PAYLOAD = {"summary": "s", "roles": [{"unit_id": "a", "role": "headline"}], "ranking": ["a"]}

class FakeAnthropic:
    def __init__(self, *a, stop="end_turn", text=json.dumps(PAYLOAD), **k):
        self.kw = None; self.beta = NS(messages=NS(create=self._create)); self._stop, self._text = stop, text
    def _create(self, **kw):
        self.kw = kw
        return NS(stop_reason=self._stop, content=[NS(type="text", text=self._text)],
                  usage=NS(input_tokens=1000, output_tokens=200))

def test_schema_enums_unit_ids():
    s = prompt.response_schema(["a", "b"])
    assert s["properties"]["ranking"]["items"]["enum"] == ["a", "b"]
    assert s["additionalProperties"] is False

def test_claude_request_shape_and_cost(monkeypatch):
    fake = FakeAnthropic()
    monkeypatch.setattr(providers.anthropic, "Anthropic", lambda *a, **k: fake)
    res = ClaudeProvider(model="claude-opus-5-5").complete("sys", "user", prompt.response_schema(["a"]))
    kw = fake.kw
    assert "temperature" not in kw and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["format"]["type"] == "json_schema" and kw["output_config"]["effort"] == "low"
    assert res.data == PAYLOAD and res.stats.cost_usd == pytest.approx(1000 * 4 / 1e6 + 200 * 20 / 1e6)

@pytest.mark.parametrize("stop,text", [("refusal", ""), ("max_tokens", '{"summ'), ("end_turn", "not json")])
def test_claude_refusal_and_truncation_raise(monkeypatch, stop, text):
    monkeypatch.setattr(providers.anthropic, "Anthropic", lambda *a, **k: FakeAnthropic(stop=stop, text=text))
    with pytest.raises(ProviderError):
        ClaudeProvider().complete("sys", "user", prompt.response_schema(["a"]))

def test_gemini_request_shape_and_cost(monkeypatch):
    seen = {}
    def generate_content(model, contents, config):
        seen.update(model=model, config=config)
        return NS(text=json.dumps(PAYLOAD), usage_metadata=NS(prompt_token_count=1000, candidates_token_count=150, thoughts_token_count=50))
    monkeypatch.setattr(providers.genai, "Client", lambda *a, **k: NS(models=NS(generate_content=generate_content)))
    res = GeminiProvider(model="gemini-3.8-flash").complete("sys", "user", {"type": "object"})
    assert seen["config"].response_mime_type == "application/json"
    assert seen["config"].response_json_schema == {"type": "object"}
    assert res.stats.output_tokens == 200
    assert res.stats.cost_usd == pytest.approx(1000 * 0.75 / 1e6 + 200 * 3.75 / 1e6)

def test_get_provider_auto(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "CHAMELEON_LLM"):
        monkeypatch.delenv(k, raising=False)
    assert get_provider() is None
    monkeypatch.setenv("GEMINI_API_KEY", "x"); assert get_provider().name == "gemini"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x"); assert get_provider().name == "claude"
    assert get_provider("heuristic") is None

@pytest.mark.live
@pytest.mark.parametrize("name", ["claude", "gemini"])
def test_live_provider_on_l2(name, l2):
    from engine.pipeline.priority.units import build_units
    from engine.pipeline.priority.features import unit_features
    units = build_units(l2["source"]); ids = [u.unit_id for u in units]
    res = get_provider(name).complete(prompt.SYSTEM_PROMPT, prompt.build_user_message(unit_features(l2["source"], units)), prompt.response_schema(ids))
    assert {r["unit_id"] for r in res.data["roles"]} <= set(ids)
    print(res.stats)
```

- [ ] **Step 2:** Run `python -m pip install anthropic "google-genai<3.0.0" && python -m pytest tests/test_providers.py -v`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `prompt.py`, `pricing.py` and `providers.py` per the Interfaces block. Add the three packages to `requirements.txt` (exact pins come in Task 15).
- [ ] **Step 4:** Run `python -m pytest tests/test_providers.py -v`. Expected: 7 passed, 2 deselected (live).

  Then, with keys set, run `python -m pytest tests/test_providers.py -m live -s`. Expected: 2 passed, with stats printed. In the Gemini output, **confirm the `usage_metadata` field names**; fix `GeminiProvider` if they differ.
- [ ] **Step 5: Commit**: `git commit -m "feat(priority): Claude + Gemini providers with structured output and cost tracking"` (after `git add` of the new files and `requirements.txt`).

---

### Task 6: Validate/repair, LLM path in `plan()`, response cache

**Files:**
- Create: `chameleon/engine/pipeline/priority/validate.py`
- Modify: `chameleon/engine/pipeline/priority/__init__.py`, `chameleon/engine/pipeline/priority/providers.py` (add `cached_complete`)
- Test: `chameleon/tests/test_plan.py`

**Interfaces:**
- Consumes: `build_units`, `unit_features`, `heuristic_roles`, `heuristic_ranking`, `_assemble` (Tasks 3–4); `get_provider`, `LLMResult`, `ProviderError`, `prompt.*` (Task 5).
- Produces:
  - `MAX_REPAIR_FRACTION = 0.3`
  - `repair(raw: dict, units: list[Unit], heur_roles: dict[str,str], heur_ranking: list[str]) -> tuple[dict[str,str], list[str], float]`:
    - a missing or invalid role gets the heuristic role;
    - duplicates in the ranking are dropped (first occurrence kept);
    - unknown ids are dropped;
    - `background`/`decoration` units are removed from the ranking;
    - each missing content unit is inserted right after the nearest unit that precedes it in `heur_ranking` and is already present (at the front if there's none).
    - The fraction returned is the number of distinct units involved in any fix divided by the number of units.
  - `cached_complete(provider, system, user, schema) -> LLMResult`: when env `CHAMELEON_LLM_CACHE` names a directory, it reads and writes `<dir>/<sha256(provider.name|provider.model|system|user|json(schema))>.json` and sets `stats.cached=True` on a hit.
  - `plan(source: dict, assets_dir="brand_kit/assets", provider=None) -> PriorityPlan`:
    - `provider` is a provider object or a name for `get_provider`;
    - no provider → `heuristic_plan`;
    - otherwise call through `cached_complete` → `repair`;
    - a repair fraction > `MAX_REPAIR_FRACTION` → heuristic plan with `stats.error="too_many_repairs"`;
    - any exception → heuristic plan with `stats.error=str(exc)`;
    - success → `_assemble(..., label="llm", stats=...)`, with `stats.repaired_fraction` set.
  - **`plan()` never raises.**

- [ ] **Step 1: Write the failing tests** (fake provider object: `name`, `model`, `complete()` returning `LLMResult(data, LLMCallStats("fake", "m"))` or raising)

```python
# tests/test_plan.py
import pytest
from engine.pipeline.types import LLMCallStats
from engine.pipeline.priority import plan
from engine.pipeline.priority.providers import LLMResult, ProviderError

class Fake:
    name, model = "fake", "m"
    def __init__(self, data=None, exc=None): self.data, self.exc, self.calls = data, exc, 0
    def complete(self, system, user, schema):
        self.calls += 1
        if self.exc: raise self.exc
        return LLMResult(self.data, LLMCallStats("fake", "m"))

L2_GOOD = {"summary": "s", "roles": [
    {"unit_id": u, "role": r} for u, r in [("bg", "background"), ("deco_circle_1", "decoration"), ("deco_circle_2", "decoration"),
    ("logo", "logo"), ("product", "hero_image"), ("headline", "headline"), ("subhead", "subheadline"), ("badge", "badge"), ("cta", "cta")]],
    "ranking": ["headline", "product", "cta", "logo", "subhead", "badge"]}

def test_good_llm_output_is_used(l2):
    p = plan(l2["source"], provider=Fake(L2_GOOD))
    assert p.source == "llm" and [u.unit_id for u in p.ranked_units()][:3] == ["headline", "product", "cta"]

def heur(task):
    from engine.pipeline.priority.units import build_units
    from engine.pipeline.priority.features import unit_features
    from engine.pipeline.priority.heuristic import heuristic_roles, heuristic_ranking
    units = build_units(task["source"]); feats = unit_features(task["source"], units)
    roles = heuristic_roles(feats)
    return units, roles, heuristic_ranking(feats, roles)

def test_repair_fixes_duplicates_and_gaps(l2):
    from engine.pipeline.priority.validate import repair
    units, hr, hk = heur(l2)
    _, ranking, frac = repair(dict(L2_GOOD, ranking=["headline", "product", "cta", "logo", "subhead", "subhead"]), units, hr, hk)
    assert ranking == ["headline", "product", "cta", "logo", "subhead", "badge"] and frac == pytest.approx(2 / 9)
    _, ranking, frac = repair(dict(L2_GOOD, ranking=["headline", "product", "cta", "deco_circle_1", "logo", "subhead", "badge"]), units, hr, hk)
    assert "deco_circle_1" not in ranking and frac == pytest.approx(1 / 9)

def test_too_broken_output_falls_back(l2):
    p = plan(l2["source"], provider=Fake({"summary": "", "roles": [], "ranking": []}))
    assert p.source == "heuristic" and p.stats.error == "too_many_repairs"

def test_provider_error_falls_back(l2):
    p = plan(l2["source"], provider=Fake(exc=ProviderError("refusal")))
    assert p.source == "heuristic" and "refusal" in p.stats.error

def test_llm_cannot_relabel_background(l2):
    data = dict(L2_GOOD, roles=[{"unit_id": "bg", "role": "hero_image"}] + L2_GOOD["roles"][1:])
    assert plan(l2["source"], provider=Fake(data)).unit_for_element("bg").role == "background"

def test_missing_asset_forced_last(l3):
    from engine.pipeline.priority import _assemble
    units, roles, ranking = heur(l3)
    ranking = ["holiday_watermark"] + [u for u in ranking if u != "holiday_watermark"]
    assert _assemble(units, roles, ranking, "llm", None).ranked_units()[-1].unit_id == "holiday_watermark"

def test_auto_without_keys_uses_heuristic(monkeypatch, l2):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "CHAMELEON_LLM"):
        monkeypatch.delenv(k, raising=False)
    assert plan(l2["source"]).source == "heuristic"

def test_cache_hit_skips_provider(tmp_path, monkeypatch, l2):
    monkeypatch.setenv("CHAMELEON_LLM_CACHE", str(tmp_path))
    fake = Fake(L2_GOOD)
    plan(l2["source"], provider=fake); p = plan(l2["source"], provider=fake)
    assert fake.calls == 1 and p.stats.cached
```

- [ ] **Step 2:** Run `python -m pytest tests/test_plan.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `validate.py`, `cached_complete` and `plan()` per the Interfaces block.
- [ ] **Step 4:** Run `python -m pytest tests/test_plan.py tests/test_heuristic.py -v`. Expected: all pass.
- [ ] **Step 5: Commit**: `git commit -m "feat(priority): validate/repair, LLM path with heuristic fallback, response cache"` (after `git add` of the new files).

---

### Task 7: Aspect classes and per-unit placement

**Files:**
- Create: `chameleon/engine/pipeline/layout/__init__.py` (empty), `aspect.py`, `place.py`
- Test: `chameleon/tests/test_place.py`

**Interfaces:**
- Produces: `aspect_class(width: float, height: float) -> str`, with `r = w/h`:
  - `< 0.55` very_tall
  - `< 0.85` tall
  - `< 1.25` square
  - `< 2.2` landscape
  - `≤ 5` wide
  - otherwise strip
- Produces in `place.py`:
  - `FLOORS = {"headline": 14, "cta": 11, "subheadline": 10, "badge": 9, "legal": 7}`; any other role gets 9
  - `HARD_FLOOR = 6`: the last-resort size; a unit placed at it is marked degraded (Task 9)
  - `MAX_TEXT_GROWTH = 1.5`
  - `floor_px(role: str, width: float, height: float) -> float`: `FLOORS.get(role, 9) × max(1, min(width, height) / 600)`
  - `place_unit(unit: Unit, elements: dict[str, dict], box: Box, measurer: TextMeasurer, floor: float, align: str = "center") -> dict[str, dict] | None`: returns `{element_id: updated copy}`, or `None` when the unit can't fit `box` at `floor`.
- **Placement rules:**
  - **Image:** `fit_aspect(source w/h, box, align)`. A non-background `cover` image takes the whole box.
  - **Single text:**
    - `font = fit_font_size(el, box.w, box.h, floor, source_font × MAX_TEXT_GROWTH)`;
    - box width = slot width; height = `block_height`; vertically centred in the slot;
    - `letter_spacing` × (font / source font).
  - **`container_label` / `group`, rigid mode (try first):**
    - `k = min(box.w / unit_w, box.h / unit_h)` over the unit's union box; scale member offsets and sizes by `k`, then centre the unit in `box`;
    - `corner_radius`, `stroke_width` × `k`;
    - each text member: start at `source_font × k`, then fit inside its scaled box. If below `floor`, a `container_label` falls through to condensed mode; a `group` returns `None`.
  - **`container_label`, condensed mode** (pills in tight banners, where the source label box is much wider than its text):
    - search the largest label font `f` (0.5 px steps) in `[floor, source_font × min(1.5, box.h / unit_h)]` where:
      - shape height `= f × (shape_h_src / label_font_src)` ≤ `box.h`;
      - shape width `= single-line text width at f + 1.6 f` ≤ `box.w`;
    - the label sits on one line, inset `0.8 f` horizontally and vertically centred;
    - `corner_radius` × (new shape height / source shape height);
    - `None` if no `f` works.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_place.py
import pytest
from engine.pipeline.layout.aspect import aspect_class
from engine.pipeline.layout.place import place_unit, floor_px, MAX_TEXT_GROWTH
from engine.pipeline.priority import heuristic_plan

@pytest.mark.parametrize("w,h,cls", [(160, 600, "very_tall"), (1080, 1350, "tall"), (1080, 1920, "tall"),
    (1080, 1080, "square"), (300, 250, "square"), (1280, 720, "landscape"), (820, 312, "wide"),
    (1600, 400, "wide"), (728, 90, "strip")])
def test_aspect_class(w, h, cls): assert aspect_class(w, h) == cls

def setup(task):
    els = {e["id"]: e for e in task["source"]["elements"]}
    p = heuristic_plan(task["source"])
    return els, {u.unit_id: u for u in p.units}

def test_image_keeps_source_aspect(l2, measurer):
    els, units = setup(l2)
    out = place_unit(units["product"], els, (0, 0, 300, 100), measurer, floor=0)["product"]
    assert out["width"] / out["height"] == pytest.approx(1.0) and out["height"] == 100

def test_text_fits_tight(l2, measurer):
    els, units = setup(l2)
    out = place_unit(units["headline"], els, (0, 0, 600, 120), measurer, floor=14)["headline"]
    assert measurer.fits(out, out["font_size"], out["width"], out["height"])
    assert out["font_size"] <= 56 * MAX_TEXT_GROWTH
    assert out["height"] == pytest.approx(measurer.block_height(out, out["font_size"], 600))

def test_button_scales_rigidly(l2, measurer):
    els, units = setup(l2)
    out = place_unit(units["cta"], els, (0, 0, 300, 80), measurer, floor=8)
    bg, lab = out["cta_bg"], out["cta_label"]
    k = bg["width"] / 420
    assert bg["height"] == pytest.approx(90 * k) and bg["corner_radius"] == pytest.approx(45 * k)
    assert bg["x"] <= lab["x"] and lab["x"] + lab["width"] <= bg["x"] + bg["width"] + 0.01

def test_button_condenses_in_strip(l3, measurer):
    els, units = setup(l3)
    out = place_unit(units["cta"], els, (0, 0, 158, 82), measurer, floor=11)
    assert out is not None
    bg, lab = out["cta_bg"], out["cta_label"]
    assert lab["font_size"] >= 11 and measurer.line_count(lab, lab["font_size"], lab["width"]) == 1
    assert bg["width"] <= 158 and bg["x"] <= lab["x"] and lab["x"] + lab["width"] <= bg["x"] + bg["width"] + 0.01

def test_returns_none_below_floor(l3, measurer):
    els, units = setup(l3)
    assert place_unit(units["headline"], els, (0, 0, 60, 10), measurer, floor=14) is None

def test_overlong_word_never_overflows(l1, measurer):
    els, units = setup(l1)
    els["cta"] = dict(els["cta"], content="Supercalifragilisticexpialidocious")
    out = place_unit(units["cta"], els, (0, 0, 80, 40), measurer, floor=9)
    assert out is None or measurer.longest_word_width(out["cta"], out["cta"]["font_size"]) <= 80

def test_floor_px_scales():
    assert floor_px("headline", 728, 90) == 14
    assert floor_px("headline", 1080, 1920) == pytest.approx(14 * 1080 / 600)
```

- [ ] **Step 2:** Run `python -m pytest tests/test_place.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `aspect.py` and `place.py` per the Interfaces block.
- [ ] **Step 4:** Run `python -m pytest tests/test_place.py -v`. Expected: 16 passed.
- [ ] **Step 5: Commit**: `git commit -m "feat(layout): aspect classes and per-unit placement"` (after `git add`).

---

### Task 8: Layers, normalisation, safe fallback

**Files:**
- Create: `chameleon/engine/pipeline/layout/layers.py`, `normalize.py`, `fallback.py`
- Test: `chameleon/tests/test_fallback.py`

**Interfaces:**
- Produces in `layers.py`:
  - `place_background(el: dict, width: float, height: float) -> dict`: the box becomes `(0, 0, W, H)`.
  - `place_decoration(el: dict, src_canvas: dict, width: float, height: float) -> dict`:
    - uniform scale `s = min(W/Ws, H/Hs)`;
    - keep the offset from the source's nearest corner, scaled by `s`;
    - then clamp the rotated bounding box inside the canvas (shrinking if it's still too big).
- Produces in `normalize.py`:
  - `EDITABLE = {"x","y","width","height","font_size","letter_spacing","corner_radius","stroke_width"}`
  - `finalize(source: dict, placed: dict[str, dict], target: dict) -> dict | None`:
    - deep-copies each source element (in source order) and overwrites only `EDITABLE` keys from `placed`;
    - rounds to 2 decimals;
    - sets `canvas` to the int target;
    - returns `None` if an element is missing from `placed` or any rotated bounding box is outside the canvas (0.5 px tolerance).
- Produces in `fallback.py`: `safe_fallback(source: dict, target: dict, plan: PriorityPlan) -> Candidate` with `candidate_id="fallback"`, `template="fallback"`, `is_fallback=True`:
  - backgrounds via `place_background`;
  - everything else scales uniformly by `s` around the source canvas centre, then is centred on the target; `font_size = max(1, font_size × s)`; `corner_radius`, `stroke_width`, `letter_spacing` × `s`;
  - any rotated bounding box still outside is moved, then shrunk, until it's inside.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fallback.py
import copy, json, pytest
from pathlib import Path
from engine.pipeline.geometry import rotated_aabb, inside_canvas
from engine.pipeline.layout.layers import place_background, place_decoration
from engine.pipeline.layout.normalize import finalize, EDITABLE
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.priority import heuristic_plan

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())

def assert_valid(source, design, target):
    assert design["canvas"] == {"width": target["width"], "height": target["height"]}
    assert [e["id"] for e in design["elements"]] == [e["id"] for e in source["elements"]]
    src = {e["id"]: e for e in source["elements"]}
    for el in design["elements"]:
        assert inside_canvas(rotated_aabb(el), target["width"], target["height"])
        if el["type"] == "image" and el.get("fit", "contain") != "cover":
            s = src[el["id"]]
            assert (el["width"] / el["height"]) / (s["width"] / s["height"]) == pytest.approx(1, abs=0.02)
        if el["type"] == "text": assert el["font_size"] >= 1

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_fallback_is_valid(path, target):
    source = json.loads(path.read_text())["source"]
    assert_valid(source, safe_fallback(source, target, heuristic_plan(source)).design, target)

def test_background_covers(l2):
    assert place_background(l2["source"]["elements"][0], 1600, 400)[ "width"] == 1600

def test_decoration_round_and_inside(l2):
    d = place_decoration(l2["source"]["elements"][1], l2["source"]["canvas"], 1600, 400)
    assert d["width"] == pytest.approx(d["height"]) and inside_canvas(rotated_aabb(d), 1600, 400)

def test_finalize_copies_untouched_fields(l2):
    src = l2["source"]
    placed = {e["id"]: dict(e, x=1, y=1, width=10, height=10) for e in src["elements"]}
    out = finalize(src, placed, {"width": 100, "height": 100})
    for a, b in zip(out["elements"], src["elements"]):
        assert {k: v for k, v in a.items() if k not in EDITABLE} == {k: v for k, v in b.items() if k not in EDITABLE}

def test_finalize_rejects_out_of_canvas(l1):
    src = l1["source"]
    placed = {e["id"]: dict(e) for e in src["elements"]}
    assert finalize(src, placed, {"width": 10, "height": 10}) is None

def test_fallback_with_missing_optional_fields():
    src = {"canvas": {"width": 200, "height": 200}, "elements": [
        {"id": "t", "type": "text", "content": "Hi", "font_family": "Solstice Sans", "font_size": 20,
         "color": "rgba(0,0,0,1)", "x": 10, "y": 10, "width": 100, "height": 30}]}
    tgt = {"width": 728, "height": 90}
    assert_valid(src, safe_fallback(src, tgt, heuristic_plan(src)).design, tgt)
```

- [ ] **Step 2:** Run `python -m pytest tests/test_fallback.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `layers.py`, `normalize.py` and `fallback.py` per the Interfaces block. Move `assert_valid` into `tests/conftest.py` so Tasks 9 and 12 can reuse it.
- [ ] **Step 4:** Run `python -m pytest tests/test_fallback.py -v`. Expected: 35 passed (30 matrix + 5).
- [ ] **Step 5: Commit**: `git commit -m "feat(layout): background/decoration layers, normalisation, safe fallback"` (after `git add`).

---

### Task 9: Templates, slot assignment, candidate generation, reflow

**Files:**
- Create: `chameleon/engine/pipeline/layout/templates.py`, `generate.py`
- Test: `chameleon/tests/test_generate.py`

**Interfaces:**
- Consumes: `aspect_class`, `place_unit`, `floor_px` (Task 7); `place_background`, `place_decoration`, `finalize` (Task 8); `PriorityPlan`, `Candidate`, `TextMeasurer`.
- Produces in `templates.py`:
  - `@dataclass Slot(name: str, box: Box, accepts: tuple[str, ...], stack: bool = False, minor: bool = False)`
  - `TEMPLATES_BY_CLASS = {"very_tall": ["stack"], "tall": ["stack"], "square": ["stack"], "landscape": ["columns"], "wide": ["row"], "strip": ["row"]}`
  - `PARAM_GRID: dict[str, list[dict]]`, the Cartesian products in this order:
    - `stack`: margin ∈ (0.04, 0.07) × hero_first ∈ (True, False);
    - `columns`: margin × split ∈ (0.35, 0.45, 0.55) × mirror ∈ (False, True);
    - `row`: margin × split × mirror × cutoff ∈ (None, 4, 3).
  - `build_slots(template: str, width: float, height: float, params: dict) -> list[Slot]`: margins are `margin × min(W, H)`; fractions below are of the content box.

  | Template | Slots |
  |---|---|
  | `stack` (top → bottom) | header 0.10: `badge` slot on the left half (accepts badge), `logo` slot on the right half (accepts logo); `hero` 0.40 (hero_image, secondary_image); `text` 0.30, stack (headline, subheadline, other_text); `cta` 0.12 (cta); `minor` 0.08, minor (any role). `hero_first=False` swaps hero and text. |
  | `columns` | text column of width `1 − split`, containing `logo` 0.15 (logo), `text` 0.55 stack, `cta` 0.18 (cta) and `minor` 0.12; hero column of width `split`, full height (hero_image, secondary_image). `mirror` swaps the columns. |
  | `row` (left → right) | `logo` 0.14 (logo); `hero` `split × 0.5` (hero_image, secondary_image); `text` stack (the remaining width minus `cta`); `cta` 0.22 (cta). In `strip`, the bottom 30% of the text column is the `minor` zone. In `wide`, `logo` sits at the top 25% of the text column instead of its own column, and `minor` is a bottom band 0.15 tall under the text column. `mirror` reverses column order. |

- Produces in `generate.py`:
  - `generate_candidates(source: dict, target: dict, plan: PriorityPlan, measurer: TextMeasurer) -> list[Candidate]`: the reflow candidate first, then every template × param combination for `aspect_class(target)`. Drop combinations that `finalize` rejects. Excludes the fallback.
  - `candidate_id = f"{template}/" + "/".join(f"{k}={v}" for k, v in sorted(params.items()))`; reflow uses `"reflow"`.
  - `reflow_candidate(source, target, plan, measurer) -> Candidate | None`:
    - each content unit scales by `s = min(W/Ws, H/Hs)` about its own centre; centre maps to `(cx × W/Ws, cy × H/Hs)`;
    - text is refit with `fit_font_size(..., floor, source × MAX_TEXT_GROWTH)`;
    - overlapping content units are pushed apart along the axis with more room;
    - `None` if `finalize` rejects it.
- **Assignment algorithm** (`_assign`):
  1. Walk `plan.ranked_units()`. A unit takes the first slot whose `accepts` contains its role and that is empty (stack slots take any number).
  2. Units of the same role with consecutive ranks share one slot, laid out as a row (columns of equal width).
  3. With `cutoff=k`, units ranked > k go to `minor`. **`cta` is never sent to `minor`.**
  4. Stack slots split their height by `plan.weight` and place each unit with `place_unit`; leftover height becomes equal gaps.
  5. Minor units are laid out as a row of equal widths at `floor_px(role)`.
  6. If `place_unit` returns `None` at `floor_px(role)`, retry the same slot at `HARD_FLOOR` and add the unit to `degraded_unit_ids`.
  7. Units that still don't fit go to a free-space scan: a 4 px grid, searching for the first box that holds the unit at `HARD_FLOOR` without overlapping a placed content box. They're added to `degraded_unit_ids`.
  8. If a unit can't be placed at all, drop the candidate.
  9. Background and decoration units always go through `layers.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generate.py
import json, pytest
from pathlib import Path
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.layout.templates import build_slots, PARAM_GRID
from engine.pipeline.geometry import inside_canvas
from engine.pipeline.priority import heuristic_plan
from tests.conftest import assert_valid

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_candidates_are_valid_and_text_fits(path, target, measurer):
    src = json.loads(path.read_text())["source"]
    for c in generate_candidates(src, target, heuristic_plan(src), measurer):
        assert_valid(src, c.design, target)
        for el in c.design["elements"]:
            if el["type"] == "text":
                assert measurer.fits(el, el["font_size"], el["width"], el["height"]), (c.candidate_id, el["id"])

def test_ids_unique_and_deterministic(l2, measurer):
    run = lambda: [c.candidate_id for c in generate_candidates(l2["source"], l2["target_canvas"], heuristic_plan(l2["source"]), measurer)]
    a = run()
    assert a == run() and len(a) == len(set(a)) and any(i.startswith("row/") for i in a)

def test_cta_never_degraded(l3, measurer):
    for c in generate_candidates(l3["source"], l3["target_canvas"], heuristic_plan(l3["source"]), measurer):
        assert "cta" not in c.degraded_unit_ids

def test_reflow_present_for_l1(l1, measurer):
    cands = generate_candidates(l1["source"], l1["target_canvas"], heuristic_plan(l1["source"]), measurer)
    assert cands[0].candidate_id == "reflow"

@pytest.mark.parametrize("template", ["stack", "columns", "row"])
@pytest.mark.parametrize("w,h", [(728, 90), (1080, 1350), (1600, 400)])
def test_slots_inside_canvas(template, w, h):
    for params in PARAM_GRID[template]:
        assert all(inside_canvas(s.box, w, h) for s in build_slots(template, w, h, params))
```

- [ ] **Step 2:** Run `python -m pytest tests/test_generate.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `templates.py` and `generate.py` per the Interfaces block. Tune slot fractions only if the tests force it, and record any change in this plan's Task 9 table.
- [ ] **Step 4:** Run `python -m pytest tests/test_generate.py -v`. Expected: all pass.
- [ ] **Step 5: Commit**: `git commit -m "feat(layout): slot templates, rank-driven assignment, reflow candidates"` (after `git add`).

---

### Task 10: Grading gates

**Files:**
- Create: `chameleon/engine/pipeline/grading/__init__.py` (empty), `gates.py`
- Test: `chameleon/tests/test_gates.py`

**Interfaces:**
- Produces:
  - `GATES = ("G1_ids","G2_schema","G3_canvas","G4_in_canvas","G5_aspect","G6_frozen","G7_text_fit","G8_groups","G9_hidden")`
  - `ASPECT_TOL = 0.02`
  - `check_gates(source: dict, design: dict, target: dict, plan: PriorityPlan, measurer: TextMeasurer) -> list[str]`: the failing gate codes, in `GATES` order; empty means pass.
- **Gate definitions:**
  - **G2:** fields required per type per `chameleon_design_schema_v0.9.json`; all numbers finite; `width`, `height` > 0; `font_size` ≥ 1.
  - **G6:** every key not in `EDITABLE` (Task 8) equals the source.
  - **G7:** `measurer.fits` for every text element.
  - **G8:** each `container_label`'s text box is inside its shape box (0.5 px tolerance).
  - **G9:** no element's box is fully contained in the box of a higher-`z_index` element of a unit whose role isn't `decoration`/`background`, where that covering element is opaque (`opacity == 1`; for a shape, fill alpha 1; any image).

- [ ] **Step 1: Write the failing tests:** one mutation of the L2 fallback design per gate. Each must fail exactly its gate. Then a matrix test: the fallback passes all gates.

```python
# tests/test_gates.py
import copy, json, pytest
from pathlib import Path
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.priority import heuristic_plan

def base(l2):
    src, tgt = l2["source"], l2["target_canvas"]
    plan = heuristic_plan(src)
    return src, tgt, plan, safe_fallback(src, tgt, plan).design

def el(design, id): return next(e for e in design["elements"] if e["id"] == id)

MUTATIONS = {
    "G1_ids": lambda d: d["elements"].pop(),
    "G2_schema": lambda d: el(d, "headline").pop("color"),
    "G3_canvas": lambda d: d["canvas"].update(width=999),
    "G4_in_canvas": lambda d: el(d, "logo").update(x=-50),
    "G5_aspect": lambda d: el(d, "product").update(width=el(d, "product")["width"] * 1.5),
    "G6_frozen": lambda d: el(d, "headline").update(content="Changed"),
    "G7_text_fit": lambda d: el(d, "subhead").update(height=1),
    "G8_groups": lambda d: el(d, "cta_label").update(x=el(d, "cta_bg")["x"] + el(d, "cta_bg")["width"]),
    # product (z 3) shrunk to a 2x2 square fully inside the logo image (z 4)
    "G9_hidden": lambda d: el(d, "product").update(x=el(d, "logo")["x"] + 1, y=el(d, "logo")["y"] + 1, width=2, height=2),
}

@pytest.mark.parametrize("gate", list(MUTATIONS))
def test_each_gate_catches_its_mutation(gate, l2, measurer):
    src, tgt, plan, design = base(l2)
    bad = copy.deepcopy(design); MUTATIONS[gate](bad)
    assert gate in check_gates(src, bad, tgt, plan, measurer)

@pytest.mark.parametrize("path", sorted(Path("input").glob("*.json")), ids=lambda p: p.stem)
@pytest.mark.parametrize("target", json.loads(Path("dev/targets.json").read_text()), ids=lambda t: t["name"])
def test_fallback_passes_all_gates(path, target, measurer):
    src = json.loads(path.read_text())["source"]; plan = heuristic_plan(src)
    assert check_gates(src, safe_fallback(src, target, plan).design, target, plan, measurer) == []
```

  Some mutations also trip other gates (e.g. G8 moves the label off-canvas); the assertion checks membership, not exactness.
- [ ] **Step 2:** Run `python -m pytest tests/test_gates.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `gates.py` per the Interfaces block.
- [ ] **Step 4:** Run `python -m pytest tests/test_gates.py -v`. Expected: all pass.
- [ ] **Step 5: Commit**: `git commit -m "feat(grading): hard gates"` (after `git add`).

---

### Task 11: Geometry metrics, pixel metrics, `pick_best`

**Files:**
- Create: `chameleon/engine/pipeline/grading/metrics.py`, `pixels.py`, `score.py`
- Test: `chameleon/tests/test_score.py`

**Interfaces:**
- Produces in `metrics.py`:
  - `WEIGHTS = {"hierarchy": 0.20, "legibility": 0.20, "overlap": 0.15, "degradation": 0.10, "faithfulness": 0.08, "space_use": 0.08, "alignment": 0.07, "spacing": 0.07, "balance": 0.05}`
  - `geometry_metrics(source: dict, cand: Candidate, plan: PriorityPlan) -> dict[str, float]`, each value in `[0, 1]`, with definitions as in 03-grading-node.md §2:
    - legibility `comfort = 1.6 × floor_px(role)`;
    - space_use band `[0.45, 0.85]`;
    - alignment tolerance `0.005 × max(W, H)`.
  - `geometry_total(m: dict[str, float]) -> float`
- Produces in `pixels.py`:
  - `PIXEL_WEIGHTS = {"text_contrast": 0.6, "busyness": 0.25, "placeholder": 0.15}`
  - `contrast_ratio(rgb_a, rgb_b) -> float` (WCAG relative luminance)
  - `pixel_metrics(design: dict, plan: PriorityPlan, assets_dir="brand_kit/assets") -> dict[str, float]`:
    - render the design with its text elements removed (`render_to_image`);
    - per text: the 10th-percentile-luminance pixel inside the box vs the text colour; score `min(1, ratio / 4.5)`;
    - busyness: `1 − clamp(mean Sobel magnitude / 64)`;
    - placeholder: `1 −` the area fraction of `missing_asset` images.

    Each is averaged by `plan.weight` where per-unit.
  - `pixel_total(m) -> float`
- Produces in `score.py`:
  - `TOP_K = 6`
  - `pick_best(source: dict, target: dict, plan: PriorityPlan, candidates: list[Candidate], measurer: TextMeasurer, top_k: int = TOP_K) -> tuple[Candidate, list[GradeReport]]`:
    1. gates for all candidates;
    2. geometry for those that pass;
    3. sort by `(-geometry_total, candidate_id)`;
    4. pixel metrics for the top K;
    5. `total = 0.7 × geo + 0.3 × pix` for the top K. Only the top K are eligible. The winner is the max total, ties broken by `candidate_id`.
    6. If no non-fallback candidate passes, return the fallback.
  - The returned reports cover every input candidate.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_score.py
import copy, pytest
from engine.pipeline.types import Candidate
from engine.pipeline.grading.metrics import geometry_metrics
from engine.pipeline.grading.pixels import contrast_ratio, pixel_metrics
from engine.pipeline.grading.score import pick_best
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.priority import heuristic_plan

def test_contrast_ratio():
    assert contrast_ratio((255, 255, 255), (0, 0, 0)) == pytest.approx(21.0, rel=1e-3)
    assert contrast_ratio((10, 10, 10), (10, 10, 10)) == pytest.approx(1.0)

def test_overlap_and_hierarchy_penalised(l2, measurer):
    src, tgt = l2["source"], l2["target_canvas"]; plan = heuristic_plan(src)
    good = generate_candidates(src, tgt, plan, measurer)[0]
    bad = copy.deepcopy(good); els = {e["id"]: e for e in bad.design["elements"]}
    els["subhead"].update(x=els["headline"]["x"], y=els["headline"]["y"])          # collide
    els["headline"]["font_size"], els["subhead"]["font_size"] = 8, 60               # invert sizes
    g, b = geometry_metrics(src, good, plan), geometry_metrics(src, bad, plan)
    assert b["overlap"] < g["overlap"] and b["hierarchy"] < g["hierarchy"]

def test_white_on_white_scores_low():
    design = {"canvas": {"width": 200, "height": 100}, "elements": [
        {"id": "bg", "type": "shape", "shape_type": "rectangle", "fill_color": "rgba(255,255,255,1)", "x": 0, "y": 0, "width": 200, "height": 100, "z_index": 0},
        {"id": "t", "type": "text", "content": "Hello", "font_family": "Solstice Sans", "font_size": 20, "color": "rgba(255,255,255,1)", "x": 10, "y": 10, "width": 180, "height": 30, "z_index": 1}]}
    assert pixel_metrics(design, heuristic_plan(design))["text_contrast"] < 0.3

def test_pick_best_returns_fallback_when_others_fail(l2, measurer):
    src, tgt = l2["source"], l2["target_canvas"]; plan = heuristic_plan(src)
    fb = safe_fallback(src, tgt, plan)
    broken = Candidate("broken", {"canvas": tgt, "elements": []}, "x", {})
    best, reports = pick_best(src, tgt, plan, [fb, broken], measurer)
    assert best.is_fallback and len(reports) == 2

def test_pick_best_deterministic(l2, measurer):
    src, tgt = l2["source"], l2["target_canvas"]; plan = heuristic_plan(src)
    cands = [safe_fallback(src, tgt, plan)] + generate_candidates(src, tgt, plan, measurer)
    assert pick_best(src, tgt, plan, cands, measurer)[0].candidate_id == pick_best(src, tgt, plan, cands, measurer)[0].candidate_id
```

- [ ] **Step 2:** Run `python -m pytest tests/test_score.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `metrics.py`, `pixels.py` and `score.py` per the Interfaces block.
- [ ] **Step 4:** Run `python -m pytest tests/test_score.py -v`. Expected: 5 passed.
- [ ] **Step 5: Commit**: `git commit -m "feat(grading): geometry + pixel metrics, pick_best"` (after `git add`).

---

### Task 12: Wire `resize()`, failure policy, debug output, end-to-end run

**Files:**
- Modify: `chameleon/engine/resize.py` (replace the baseline body; keep the signature and docstring)
- Test: `chameleon/tests/test_resize.py`

**Interfaces:**
- Consumes: `plan`, `heuristic_plan` (Tasks 4/6); `safe_fallback`, `generate_candidates` (Tasks 8/9); `pick_best` (Task 11); `TextMeasurer` (Task 2).
- Produces:
  - `resize_with_report(source: dict, target_canvas: dict) -> tuple[dict, PriorityPlan, Candidate, list[GradeReport]]`:
    - implements README §5's failure policy exactly, with a module-level lazily created `TextMeasurer`;
    - prints one line to stderr: `[priority] source=<llm|heuristic> provider=<name|none> model=<m|-> latency=<s> cost=$<usd> error=<e|->`.
  - `resize(source, target_canvas) -> dict`: returns `resize_with_report(...)[0]` and catches *everything*.
    - If `resize_with_report` raises, return `safe_fallback(source, target_canvas, heuristic_plan(source)).design`.
    - If that also raises, return `_last_resort(source, target_canvas)`: today's baseline stretch (kept from the current `resize.py`), with each box clamped into the canvas.
  - When `CHAMELEON_DEBUG=1`, write the following to `output/debug/<sha1(json(source)+json(target))[:10]>/`:
    - `plan.json`;
    - `reports.json` (sorted by total);
    - `contact_sheet.png` (the top K renders tiled, labelled with candidate_id and total).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_resize.py
import json, pytest
from pathlib import Path
from engine import resize as R
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.priority import heuristic_plan
from tests.conftest import assert_valid

@pytest.mark.parametrize("path", sorted(Path("input").glob("*.json")), ids=lambda p: p.stem)
def test_resize_outputs_pass_gates(path, measurer):
    task = json.loads(path.read_text()); src, tgt = task["source"], task["target_canvas"]
    design, plan, best, reports = R.resize_with_report(src, tgt)
    assert check_gates(src, design, tgt, plan, measurer) == []

def test_resize_survives_node_failures(monkeypatch, l2):
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(R, "generate_candidates", boom); monkeypatch.setattr(R, "pick_best", boom)
    assert_valid(l2["source"], R.resize(l2["source"], l2["target_canvas"]), l2["target_canvas"])

def test_resize_text_only_and_image_only():
    text_only = {"canvas": {"width": 1080, "height": 1080}, "elements": [
        {"id": "t", "type": "text", "content": "Hello world", "font_family": "Solstice Serif", "font_size": 60,
         "color": "rgba(0,0,0,1)", "x": 100, "y": 400, "width": 880, "height": 100}]}
    image_only = {"canvas": {"width": 1080, "height": 1080}, "elements": [
        {"id": "i", "type": "image", "asset_id": "solstice_bottle.png", "x": 290, "y": 290, "width": 500, "height": 500}]}
    for src in (text_only, image_only):
        for tgt in ({"width": 728, "height": 90}, {"width": 1080, "height": 1920}):
            assert_valid(src, R.resize(src, tgt), tgt)

def test_resize_is_deterministic(l3):
    a = json.dumps(R.resize(l3["source"], l3["target_canvas"]))
    assert a == json.dumps(R.resize(l3["source"], l3["target_canvas"]))

def test_debug_dump_only_when_enabled(monkeypatch, l1):
    debug_dirs = lambda: set(Path("output/debug").glob("*"))
    monkeypatch.delenv("CHAMELEON_DEBUG", raising=False)
    before = debug_dirs(); R.resize(l1["source"], l1["target_canvas"])
    assert debug_dirs() == before
    monkeypatch.setenv("CHAMELEON_DEBUG", "1"); R.resize(l1["source"], l1["target_canvas"])
    new = debug_dirs() - before
    assert len(new) <= 1 and all((d / "reports.json").exists() for d in debug_dirs())
```

- [ ] **Step 2:** Run `python -m pytest tests/test_resize.py -v`. Expected: FAIL (`resize_with_report` missing).
- [ ] **Step 3:** Implement `resize_with_report`, `resize` and the debug dump in `engine/resize.py`. Import `generate_candidates` and `pick_best` into the module namespace so the failure test can patch them.
- [ ] **Step 4:** Run `python -m pytest -v`. Expected: the whole suite passes.

  Then run `python -m engine.main`. Expected: three `[...] OK` lines (Level 2 may also show the self-check note) and three `[priority] source=...` lines. **Open the three `output/*_output.png` files and compare them with the baseline renders.** Text must be readable and laid out by priority, not stretched.
- [ ] **Step 5: Commit**: `git commit -m "feat(engine): wire three-node pipeline into resize() with failure policy"` (after `git add`).

---

### Task 13: Dev matrix and contact sheets

**Files:**
- Create: `chameleon/dev/run_matrix.py`
- Test: `chameleon/tests/test_dev_tools.py`

**Interfaces:**
- Consumes: `resize_with_report` (Task 12), `render_to_image` (Task 2), `dev/targets.json`.
- Produces: `run_matrix(inputs: list[Path], targets: list[dict], out_dir: Path) -> dict`. It writes:
  - `out_dir/<stem>/<target>.png`;
  - `out_dir/<stem>_sheet.png`: all targets tiled, labelled with target name, winner `candidate_id` and plan source;
  - `out_dir/winners.json`: `{stem: {target: candidate_id}}`.

  It returns the winners dict. CLI: `python -m dev.run_matrix [--out output/matrix] [--inputs input/*.json]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dev_tools.py
import json
from pathlib import Path
from dev.run_matrix import run_matrix

def test_run_matrix_smoke(tmp_path):
    inp = [Path("input/chameleon_sample_level1_warmup.json")]
    tg = [{"name": "leaderboard", "width": 728, "height": 90}, {"name": "story", "width": 1080, "height": 1920}]
    winners = run_matrix(inp, tg, tmp_path)
    assert set(winners["chameleon_sample_level1_warmup"]) == {"leaderboard", "story"}
    assert (tmp_path / "chameleon_sample_level1_warmup_sheet.png").exists()
    assert json.loads((tmp_path / "winners.json").read_text()) == winners
```

- [ ] **Step 2:** Run `python -m pytest tests/test_dev_tools.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `run_matrix` and its CLI.
- [ ] **Step 4:** Run `python -m pytest tests/test_dev_tools.py -v`. Expected: PASS.

  Then run `python -m dev.run_matrix`. Expected: 3 sheets in `output/matrix/`. Review them as a team; this is the main quality loop.
- [ ] **Step 5: Commit**: `git commit -m "feat(dev): source x target matrix with contact sheets"` (after `git add`).

---

### Task 14: Provider benchmark (cost, latency, ranking quality)

**Files:**
- Create: `chameleon/dev/bench_priority.py`
- Modify: `chameleon/tests/test_dev_tools.py`

**Interfaces:**
- Consumes: `plan` (Task 6), `ClaudeProvider`, `GeminiProvider` (Task 5), `role_accuracy`, `kendall_tau`, `top3_recall` (Task 4), `dev/golden/*.json`.
- Produces:
  - `BENCH_COLUMNS = ["provider","model","sample","run","source","latency_s","input_tokens","output_tokens","cost_usd","role_accuracy","kendall_tau","top3_recall","repaired_fraction","error"]`
  - `bench(providers: list, samples: list[Path], runs: int, out_csv: Path) -> list[dict]`: `providers` holds provider objects and/or the string `"heuristic"`. One row per provider × sample × run. Wall-clock latency is measured around `plan()`. The heuristic runs as a provider entry with `model="-"` and cost 0.
  - **The cache is disabled** (it unsets `CHAMELEON_LLM_CACHE` during the run).
  - It prints a summary table: the mean of each numeric column per provider × model, plus a projected cost for 1,000 tasks.
  - CLI: `python -m dev.bench_priority --claude-models claude-opus-5-5 --gemini-models gemini-3.8-flash --runs 3 --out output/bench/priority.csv`. Each `--*-models` takes several values; a provider is skipped (with a printed note) when its key is missing.

- [ ] **Step 1: Write the failing test** (heuristic only; no network)

```python
def test_bench_heuristic_only(tmp_path):
    from dev.bench_priority import bench, BENCH_COLUMNS
    rows = bench(["heuristic"], [Path("input/chameleon_sample_level2_core.json")], runs=2, out_csv=tmp_path / "b.csv")
    assert len(rows) == 2 and rows[0]["source"] == "heuristic" and rows[0]["top3_recall"] == 1.0
    assert (tmp_path / "b.csv").read_text().splitlines()[0].split(",") == BENCH_COLUMNS
```

- [ ] **Step 2:** Run `python -m pytest tests/test_dev_tools.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `bench` and its CLI per the Interfaces block.
- [ ] **Step 4:** Run `python -m pytest tests/test_dev_tools.py -v`. Expected: PASS.

  Then, with keys set, run the CLI command above. Expected:
  - a CSV of 3 providers × 3 samples × 3 runs = 27 rows;
  - a printed summary.

  **Record the summary in the PR description.** This is the API-vs-heuristic comparison. Then follow "Decision point after Task 14" at the top of this plan, starting with expanding the golden set.
- [ ] **Step 5: Commit**: `git commit -m "feat(dev): priority provider benchmark (cost, latency, quality)"` (after `git add`).

---

### Task 15: Docker build and no-network smoke test

**Files:**
- Modify: `chameleon/requirements.txt` (exact pins), `chameleon/README.md` (short "Our engine" section: the 3 nodes, env vars, how to run tests, the matrix and the bench)

**Interfaces:**
- Consumes: everything above. `Dockerfile` stays unchanged: it already copies `engine/`, `renderer/` and `brand_kit/`. `dev/` and `tests/` stay out of the image.

- [ ] **Step 1:** Pin `anthropic`, `google-genai` and `numpy` in `requirements.txt` to the versions in the venv (`python -m pip freeze | grep -iE "^(anthropic|google-genai|numpy)=="`). Leave Pillow and cr-renderer lines as they are.
- [ ] **Step 2:** Run `docker build --platform linux/amd64 -t chameleon-engine .`. Expected: the build succeeds.
- [ ] **Step 3:** Run `docker run --rm --network none -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" chameleon-engine`. Expected:
  - three `[...] OK` lines;
  - three `[priority] source=heuristic provider=none` lines;
  - `output/*_output.{json,png}` regenerated.

  **Record the wall time per task.**
- [ ] **Step 4 (optional, dev only):** Run the same command without `--network none` and with `-e ANTHROPIC_API_KEY`. Expected: `source=llm provider=claude`. This shows the API path works in the container. **It is never how scoring runs.**
- [ ] **Step 5: Commit**: `git commit -m "build: pin pipeline deps, document engine usage, verify no-network Docker run"` (after `git add`).
