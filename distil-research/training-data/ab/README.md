# ab: does an engine change make the final ads better?

**What it is:** the whole engine run under two configurations ("variants", set by environment variables such as `CHAMELEON_FLEX=1`) on the same held-out cells. Both winners are rendered, and a teacher model judges A vs B on every cell where the winning design differs.

**Why it exists:** a change like content-aware slot sizing can look good in unit tests and still not change what ships. This measures the end result, by the teacher's taste, on holdout ads only.

**Files:**
- `<NAME>/winners.jsonl`: one row per cell: `{cell, task_id, split, brand, target, variant, candidate_id, is_fallback, design_hash, png, seconds}`. `seconds` is engine time (not rendering). `<NAME>/variant.json` records the env. Renders: `<NAME>/renders.nosync/<task>/<target>.png` (768 px).
- `<A>__vs__<B>/judgments.jsonl`: same rows as `training-data/judgments` (`a`/`b` hold the variant names; swapped re-asks included).
- `<A>__vs__<B>/REPORT.md`: verdict line, identical vs changed winners, wins overall and by brand (confidence ≥ 2; per cell, majority of original and swapped asks), mean ship-readiness, seconds per cell, sign-test p-value; plus wins by target aspect class and by brand family (solstice / offbrand / challenge), each with B's win share, and the **generalisation score**: the lowest B win share over all those slices that have at least 5 decided cells (n/a if none), next to the overall B win share. A change that wins on average but loses a whole slice shows up here.
- `_sources.nosync/<task>/source.png`: the "original" image the judge sees for challenge ads (they are not in the candidate pool).

**How it's made:** `python -m dev.distill.ab_engine run | judge | report` (see the module docstring). Every step is resumable and shardable (`--shard K/N`, then `--merge`). The judge uses the same harsh rubric, swap share 0.2 and 4 pairs per call as `dev/distill/judge.py`.

**Feeds:** go/no-go decisions on engine changes. Not training data.

**Challenge set (generalisation):** `run --split challenge` runs the 40 evaluation-only ads in `corpus/challenge/` (see its README) on every target except those within 2% of the ad's own aspect (499 cells), built from the ads and `dev/targets.json`, not from `pool.jsonl`. Use separate names (e.g. `base_chl`, `flex_chl`) so challenge and holdout winners never mix:

```
python -m dev.distill.ab_engine run --split challenge --name base_chl --env CHAMELEON_FLEX=0
python -m dev.distill.ab_engine run --split challenge --name flex_chl --env CHAMELEON_FLEX=1
python -m dev.distill.ab_engine judge --a base_chl --b flex_chl
python -m dev.distill.ab_engine report --a base_chl --b flex_chl
```

Never tune or select anything on challenge results; they are the final yardstick between candidate approaches.

**Caveat:** holdout runs compare only cells in `candidate-pool/pool.jsonl` (targets near the source's aspect are not in it). The teacher is Claude's taste against the official rubric, not the official scorer.
