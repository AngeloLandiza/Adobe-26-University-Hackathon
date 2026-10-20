# candidate-pool: every layout the engine makes

**What it is:** for each corpus ad × target size (a "cell"), every candidate layout Esteban's engine generates that passes the hard gates, with the grader's 12 scores and a rendered PNG.

**Why it exists:** the teacher judges pairs of these candidates. Student D then learns how much each of the 12 scores should count, so that the grader prefers what the teacher prefers.

**Files:**
- `pool.jsonl`, one row per candidate:
  ```json
  {"cell": "c07_story_sale@leaderboard", "task_id": "c07_story_sale", "split": "train", "target": "leaderboard",
   "candidate_id": "row/cutoff=3/margin=0.04/mirror=False/split=0.35", "png": "renders.nosync/c07_story_sale/leaderboard/4.png",
   "is_fallback": false, "metrics": {"hierarchy": 0.91, "...": "12 keys: 9 geometry + 3 pixel"}}
  ```
- `renders.nosync/` (gitignored): `<task_id>/<target>/<n>.png` per candidate, plus `<task_id>/source.png`, the original ad shown to the judge. Regenerate with the command below.

**How it's made:** `python -m dev.distill.pool` (from `distil-research/chameleon/`). It's deterministic for a given seed: target sizes are sampled per ad from `dev/targets.json`, covering every shape class across the corpus.

**Feeds:** `judgments/` (which pairs to show) and student D (the 12 scores of each judged candidate).

## Generation log

- **2026-10-20:** `python -m dev.distill.pool --per-ad 4 --seed 0 --split all` over the 153-ad corpus gave **612 cells and 10,568 candidates** (mean 17.3 per cell). The cells split as train 420 (216 Solstice, 204 off-brand) and holdout 192 (116 Solstice, 76 off-brand). No ad failed to build. `pool.jsonl` is 6.5 MB; the renders (310 MB) are in `renders.nosync/`, gitignored.
