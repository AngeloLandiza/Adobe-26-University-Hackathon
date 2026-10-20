# distil-research

**Research only. This is not our submission.** The real engine lives in [`../chameleon/`](../chameleon/). Nothing in this folder is used by it, built into its Docker image, or scored.

## What this is

A sandbox for trying **distillation**:
1. Big AI models (the "teacher", e.g. Claude Opus 5.5 through Claude Code) look at many resized ads and produce data: new floor plans, and "layout A beats layout B" judgements.
2. We turn that data into small pieces that could run offline in the engine: learned grader weights, new templates, maybe a small ranker.

The plan, with what each piece is and which can run in parallel, is in [`docs/superpowers/plans/2026-10-20-distillation.md`](docs/superpowers/plans/2026-10-20-distillation.md).

## What's in here

The folder mirrors the repo layout, so every path and link inside the copied code and docs still works:

```
distil-research/
  chameleon/   a full copy of Esteban's three-node engine (branch stage1-three-node, PR #4, at 5e3660e)
               + two research hooks:
                 F1  grader weights read from chameleon/engine/pipeline/grading/weights.json
                 F2  extra floor plans read from chameleon/engine/pipeline/layout/extra_templates.json
               With neither file present, the engine behaves exactly like Esteban's.
  docs/        Esteban's pipeline design docs, Angelo's ideas doc, and the distillation plan
```

**Where it came from:** the exact commits (Esteban's work, then F1 `551c463`, F2 `3a6603a` and review fixes `5e13fe8`) are kept on the local branch `angelo/distill-backup`.

## Running it

Everything runs from `distil-research/chameleon/`. The main project's virtual environment has every package needed:

```bash
cd distil-research/chameleon
../../chameleon/.venv/bin/python -m pytest -q          # 517 tests, ~20 s
../../chameleon/.venv/bin/python -m engine.main        # resize the 3 samples into output/
../../chameleon/.venv/bin/python -m dev.run_matrix     # 3 samples × 13 ad sizes, contact sheets in output/matrix/
```

Python imports `engine` from the folder you run in, so these commands use this copy, never `../chameleon/`.

## Known issue

`chameleon/tests/test_no_data_files.py` compares output against `tests/no_data_files_expected.json`, which was generated on macOS. Text is measured slightly differently on Linux and Windows (see `chameleon/README.md`, "Linux output differs from Windows output"), so this test will likely fail there.

The fix: compare two runs on the same machine (with and without the hooks) instead of a stored file.

## Rules for this folder

- Keep research changes in here. Don't edit `../chameleon/` from research work.
- If something here proves worth shipping, port it to the real engine through a normal, reviewed PR.
- Teacher outputs, renders and judgements go in `chameleon/dev/data/` (gitignored). Never commit API keys.
