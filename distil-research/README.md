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
  training-data/  every dataset the distillation loop produces, one folder each, each with a README
                  (corpus of ads, candidate pool, teacher judgements, floor plans, role labels)
```

**The research tools** live in `chameleon/dev/distill/`; run them from `distil-research/chameleon/`:

| Command | What it does | Writes to |
|---|---|---|
| `python -m dev.distill.make_corpus` | builds the ad collection | `training-data/corpus/` |
| `python -m dev.distill.pool` | resizes every ad to several sizes, scores and renders every candidate | `training-data/candidate-pool/` |
| `python -m dev.distill.pilot make` / `score` | 25-pair calibration check: a person and Claude judge the same pairs | `training-data/judgments/pilot/` |
| `python -m dev.distill.judge` | Claude Code (Opus 5.5) judges candidate pairs against the official rubric | `training-data/judgments/judgments.jsonl` |
| `python -m dev.distill.train_grader [--write]` | learns the grader's weights from the judgements | `training-data/judgments/training-report.md`, and with `--write` `chameleon/engine/pipeline/grading/weights.json` |

The judging criteria are the challenge's official rubric (the brief, Section 2.5), quoted in `chameleon/dev/distill/rubric.md`.

**Progress dashboard (local):** `python -m dev.distill.serve_dashboard`, then open http://127.0.0.1:8765. It shows data collection, judging and training; see [`dashboard/README.md`](dashboard/README.md).

**Attention research** (does a model that sees how the pieces relate to each other judge and plan layouts better?) has its own folder: [`docs/attention/`](docs/attention/README.md), tracking GitHub issue #1.

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

## Differences from Esteban's engine

- **Three extra floor plans ship here** (`chameleon/engine/pipeline/layout/extra_templates.json`): `split_hero_left`, `text_first` and `story_centered`. They were hand-drawn for the square, tall and very-tall shapes that only had the `stack` template; see `training-data/floor-plans/README.md`. Delete the file to get Esteban's behaviour back exactly.
- **`tests/test_no_data_files.py`** checks that with no data files the engine is unchanged. It runs on every platform: it compares two runs on the same machine. The stored expectation is per platform, in `tests/no_data_files_expected/<platform>.json`, so a missing platform is skipped rather than failing.

## Rules for this folder

- Keep research changes in here. Don't edit `../chameleon/` from research work.
- If something here proves worth shipping, port it to the real engine through a normal, reviewed PR.
- Teacher outputs, renders and judgements go in `chameleon/dev/data/` (gitignored). Never commit API keys.
