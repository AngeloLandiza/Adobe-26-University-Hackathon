# CHAMELEON — starter repo

This is the scaffolding for your engine. It handles input loading, output
writing, rendering, and Docker packaging — so the only file you need to
edit is **`engine/resize.py`**.

```
input:  a task JSON (source design + target canvas)
              |
              v
        engine/resize.py   <-- YOU IMPLEMENT THIS
              |
              v
output: <task>_output.json  (schema-valid resized design)
        <task>_output.png   (rendered preview)
```

Both the JSON and the PNG are what get evaluated — see `RULES.md`.

## Our engine (work in progress)

We're replacing the baseline with a three-node pipeline:

1. a **priority** node works out what each element is and ranks them;
2. a **resize engine** builds candidate layouts from that ranking;
3. a **grading** node picks the best candidate.

Design: [`docs/pipeline/`](../docs/pipeline/README.md). Implementation plan and status: [`docs/superpowers/plans/`](../docs/superpowers/plans/2026-10-13-stage1-three-node-pipeline.md).

**Status:** all three nodes are built and wired into `engine/resize.py`, so `python -m engine.main` now produces real layouts instead of the stretch baseline. Each task prints one `[priority] …` line saying whether the LLM or the rule-based ranker ranked it. The provider benchmark, the Docker no-network check and the whole-branch review (with its fixes) are done. What's missing: [`docs/pipeline/README.md` §11](../docs/pipeline/README.md#11-whats-missing-as-of-2026-10-17).

**Run the tests** from this folder:

```bash
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/Scripts/python -m pytest -q
```

(On macOS/Linux, use `.venv/bin/python`.)

**See the results across 13 real ad sizes** (renders and one contact sheet per input, written to `output/matrix/`; about 95 s):

```bash
.venv/Scripts/python -m dev.run_matrix
```

For one task's candidates and scores, set `CHAMELEON_DEBUG=1` before `python -m engine.main`; results go to `output/debug/`.

**Compare the priority providers** (ranking quality, latency, cost per call and projected cost per 1,000 designs; writes `output/bench/priority.csv`). The rule-based ranker always runs; Claude and Gemini only run when their API key is set:

```bash
.venv/Scripts/python -m dev.bench_priority --runs 3
```

**LLM settings for the priority node** (optional; without them it uses its built-in rule-based ranker, which is also what runs in the no-network scored Docker run). Put them in `chameleon/.env`: copy `.env.example` to `.env` and fill in your keys. The file is read automatically and is gitignored.

- `ANTHROPIC_API_KEY` and/or `GEMINI_API_KEY` enable Claude or Gemini.
- `CHAMELEON_LLM` = `auto` (default), `claude`, `gemini` or `heuristic`.
- `CHAMELEON_LLM_CACHE=<dir>` caches LLM answers between runs.

The full list is in [`docs/pipeline/README.md` §8](../docs/pipeline/README.md#8-development-loop). Live API tests are opt-in:

```bash
.venv/Scripts/python -m pytest tests/test_providers.py -m live -s
```

Never commit API keys. `.env` is gitignored, and keys must not be baked into the Docker image.

## Repo layout

```
engine/
  resize.py        <- implement your approach here
  main.py           <- orchestrator (input -> resize() -> output). Don't need to edit this.
  pipeline/         <- ours: shared types/geometry/text measurement, priority/ (node 1), layout/ (node 2), grading/ (node 3)
tests/              <- ours: pytest suite (run from this folder)
dev/                <- ours, dev-only (not in the Docker image): target sizes, golden labels, metrics, run_matrix
renderer/           <- the real renderer (transformer-convertor + cr-renderer), already vendored
brand_kit/
  assets/           <- image files referenced by asset_id in the sample designs
  fonts/            <- font files matching the schema's font_family values
input/               <- drop task JSON files here for local testing (3 samples included)
output/              <- results land here
chameleon_design_schema_v0.9.json   <- full field-by-field spec for the design format
Dockerfile
RULES.md
```

## Quick start (no Docker needed for iterating on `resize()`)

```bash
pip install -r requirements.txt
python -m engine.main
```

This processes every `*.json` file in `input/` (the 3 sample designs are
already there — one per difficulty level) and writes an `_output.json` and
`_output.png` for each into `output/`.

The included baseline is a **uniform stretch** — deliberately the wrong
approach (see the main brief: "a naive program stretches the picture; a
good one moves the furniture"). It's there so the pipeline runs on day
one. Run it once, look at `output/*_output.png`, and you'll see exactly
why it's not good enough — that's the point. Replace the body of
`resize()` in `engine/resize.py`, not just its constants.

## What the self-check does (and doesn't do)

`main.py` runs a quick structural check after every task — same element
`id`s as the source, required fields present, nothing off-canvas. **This
is not the official scorer.** It exists to catch obvious mistakes before
you waste a submission on them. See `RULES.md` for what's actually
checked and graded.

A self-check warning does not fail the run — your JSON and PNG are still
written either way. Read the warning; decide whether it's a real problem
or, like the Level 2 sample's intentionally corner-bleeding decorative
circle, an expected one.

## The renderer

`main.py` calls `renderer/converter/convert.py`'s `render_design_to_png()`
(a vendored copy of `transformer-convertor`, wrapping `cr-renderer`) to
produce each task's PNG — already wired in, no setup needed. It reads
fonts/assets from `brand_kit/` and writes a real rendered preview, not a
placeholder. If you ever see plain labeled boxes instead of real text and
images in `output/*.png`, that means the import in `main.py` failed (e.g.
`cr-renderer` didn't install — check `requirements.txt` installed cleanly)
and it silently fell back to the box-and-label placeholder; don't judge
your layout quality by that fallback.

## Docker (how you'll actually be run)

```bash
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --network none \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  my-chameleon-engine
```

`--network none` matches how stage-one scoring runs: everything your
engine needs must be installed at **build** time (add it to
`requirements.txt`), not fetched at run time. If you're using a locally
hosted model, bake its weights into the image during the build step.

**Our engine in Docker (checked 2026-10-17, Docker 29.8.1).** The build takes about 80 s from scratch. With `--network none`, each task prints `[priority] source=heuristic provider=none`: no API can be reached, so the rule-based ranker takes over. The three samples take 3.3 s, 2.4 s and 1.9 s, about 15 s for the whole run including container start. All 39 sample × size outputs pass every hard gate inside the container.

On Windows (Git Bash), turn off path conversion and use Windows paths for the mounts:

```bash
MSYS_NO_PATHCONV=1 docker run --rm --network none -v "$(pwd -W)/input:/app/input" -v "$(pwd -W)/output:/app/output" my-chameleon-engine
```

**Linux output differs from Windows output.** Skia measures text in whole pixels on Linux and in fractions of a pixel on Windows. The Linux widths can be up to about 6% wider (a badge label measures 120.6 px on Windows and 127 px on Linux). Text can therefore wrap differently, and the engine may pick a different layout. Line breaks still match the renderer on each platform, because our measurer calls the renderer's own wrap code. Scoring runs on Linux, so check final layouts in the container. `dev/` isn't in the image, so mount it:

```bash
docker run --rm --network none -v "$(pwd)/input:/app/input:ro" -v "$(pwd)/dev:/app/dev:ro" -v "$(pwd)/output:/app/output" --entrypoint python my-chameleon-engine -m dev.run_matrix --out output/matrix_linux
```

Run the test suite there too. It needs network to install pytest, so this run isn't the scored setup. On Linux the suite caught a bug that Windows had hidden: an image shrunk to a speck.

```bash
docker run --rm -v "$(pwd)/input:/app/input:ro" -v "$(pwd)/dev:/app/dev:ro" -v "$(pwd)/tests:/app/tests:ro" -v "$(pwd)/pytest.ini:/app/pytest.ini:ro" -v "$(pwd)/requirements-dev.txt:/app/requirements-dev.txt:ro" --entrypoint sh my-chameleon-engine -c "pip install -q -r requirements-dev.txt && python -m pytest -q"
```

Only mount files that exist: Docker creates an empty folder on the host for a missing mount source.

Never pass API keys to the scored image. For a local check of the LLM path, leave out `--network none` and add `-e ANTHROPIC_API_KEY` (or `-e GEMINI_API_KEY`).

## See also

- `chameleon_design_schema_v0.9.json` — full input/output field spec
- `RULES.md` — what's checked, what's graded, and what disqualifies a submission
- FAQ — for anything not covered above
