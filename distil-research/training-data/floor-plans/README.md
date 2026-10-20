# floor-plans: teacher-designed templates (student B2) and hand-drawn ones (B1)

**What it is:** for training ads at each target shape class, the teacher (Claude Opus 5.5 via Claude Code, acting as a senior layout designer) proposes floor plans. A floor plan is a way to cut the canvas into named slots (`hero`, `text`, `logo`, `cta`, `badge`, `minor`), in the same JSON format as `chameleon/engine/pipeline/layout/extra_templates.json`. The engine fills each one with the real pieces, and the teacher judges the result against that cell's current winner (its best candidate by today's default grader).

**Why:** judging showed the engine's best candidate scored 5/5 ship-readiness 0 times and 4/5 only 29 times in 715 pairs: the engine lacks good candidates, especially for tall, square and very tall targets (only `stack` plus the 3 hand-drawn plans). Clustering the floor plans that win (grouping similar winners and taking the most typical real member of each group) gives new templates found from data rather than guessed.

**Feeds:** candidate entries for `extra_templates.json` (B2; the controller ships them only after the holdout evaluation), then a per-ad floor-plan recommender (C).

## B2 files (made by `chameleon/dev/distill/floorplans.py`)

| File | One row / what | Made by |
|---|---|---|
| `proposals.jsonl` | one valid plan: `{proposal_id, cell, task_id, split, target, aspect_class, idea, plan, winner_id, teacher, rubric_version}`; `plan` is engine-loadable, named `p<hash8>` | `propose` |
| `builds.jsonl` | one per proposal, built or not: `{proposal_id, cell, candidate_id, metrics, total, png, built, reason_if_not, n_candidates, n_passed, ...}` | `build` |
| `renders.nosync/<task>/<target>/<proposal_id>.png` | each built plan's best candidate (gitignored, regenerable) | `build` |
| `judgments.jsonl` | `dev.distill.judge` rows: built plan's best candidate vs the cell's current winner | `judge` |
| `clusters.json` | per aspect class: proposed/built/judged/won counts, k, silhouette per k, medoids with member counts and win rates | `cluster` |
| `extra_templates.candidate.json` | the medoids (named `b2_<class>_<n>`) plus the shipped hand-drawn plans, in the engine's format. **Not** the shipped file. | `cluster` |
| `eval-builds.jsonl`, `eval-judgments.jsonl`, `eval-summary.json` | holdout check of a candidate file's new plans vs the current winner, per plan | `evaluate` |
| `b1_hand_drawn.json` | B1's hand-drawn plans (below) | by hand |

**How it is made** (run from `distil-research/chameleon/` with `../../chameleon/.venv/bin/python -m dev.distill.floorplans ...`):

1. `propose [--cells 160] [--seed 0] [--classes tall,very_tall,square] [--limit-calls N] [--print-prompt]`: samples train cells from `candidate-pool/pool.jsonl` with `judge.pick_cells`, stratified by brand x aspect class (every class appears), and asks the teacher for 3 plans per cell. The teacher sees the original ad, the current winner ("beat it"), the target size and class, the ad's ranked units (from `heuristic_plan`), how slots are filled, and the harsh rubric (`dev/distill/rubric.md`). Plans the engine would reject (`extra_templates._parse_plan`: overlap, no `minor`, box outside the square, ...) and duplicates are skipped with a `[floorplans] skipping` line.
2. `build`: each plan alone is filled by the real engine at each of its margins (through `CHAMELEON_EXTRA_TEMPLATES` pointing at a temporary file); gate-passing candidates are measured and the best by the default grader is rendered. A plan that builds nothing is recorded with the reason.
3. `judge [--swap-share 0.2] [--pairs-per-call 4]`: pairs each built plan with the cell's current winner and judges with the harsh rubric; 20% of pairs are re-asked in swapped order.
4. `cluster`: a plan **wins** when every verdict on it (original and swapped re-ask) picks it with confidence >= 2. Per aspect class each winning plan becomes a vector (for each slot name, `[present, x, y, w, h]`); k-medoids (PAM, numpy only) with k = 2..5 chosen by silhouette (k = 1 under 3 winners). A medoid's win rate = its member winners / all judged plans of the class nearest to it.
5. `evaluate --stage build|judge|summary [--candidate FILE]`: for HOLDOUT cells, builds each new plan of the candidate file for the classes it covers, records whether the engine's default grader would pick it over the current winner (`engine_picks_new`), pairs it with the current winner for judging, and summarises wins per plan.

Every step appends as it goes and skips finished work on a rerun. `--shard K/N` does shard K's cells only (writing `<file>.shardKofN.jsonl`, run N in parallel); `--merge` folds the shards into the main file.

## B1: hand-drawn floor plans (2026-10-20)

`b1_hand_drawn.json` records five hand-drawn floor plans for the shapes that had only the `stack` template (square, tall, very tall), plus landscape. Each was tested on the 3 samples × the 7 square/tall/very-tall/landscape target sizes, with the current hand-set grader picking the winner:

| Plan | Shape classes | Wins | Kept |
|---|---|---|---|
| `split_hero_left` | square, landscape | Level 3 → YouTube 1280×720 (big bottle left, text right; old winner had cramped text) | yes |
| `text_first` | tall, very tall | Level 1 → Instagram portrait (headline properly sized; the reflow had shrunk it) | yes |
| `story_centered` | very tall | Level 2 → half page 300×600 | yes |
| `hero_band` | tall, square | none | no |
| `split_hero_right` | square, landscape | none | no |

The kept three ship in `chameleon/engine/pipeline/layout/extra_templates.json`.

**Caveat:** "wins" is judged by the *uncalibrated* grader. Once the learned weights from student D exist, re-run the comparison (`b1_hand_drawn.json` keeps the rejected plans for exactly that).
