# ab: does an engine change make the final ads better?

**What it is:** the whole engine run under two configurations ("variants", set by environment variables such as `CHAMELEON_FLEX=1`) on the same held-out cells. Both winners are rendered, and a teacher model judges A vs B on every cell where the winning design differs.

**Why it exists:** a change like content-aware slot sizing can look good in unit tests and still not change what ships. This measures the end result, by the teacher's taste, on holdout ads only.

**Files:**
- `<NAME>/winners.jsonl`: one row per cell: `{cell, task_id, split, brand, target, variant, candidate_id, is_fallback, design_hash, png, seconds}`. `seconds` is engine time (not rendering). `<NAME>/variant.json` records the env. Renders: `<NAME>/renders.nosync/<task>/<target>.png` (768 px).
- `<A>__vs__<B>/judgments.jsonl`: same rows as `training-data/judgments` (`a`/`b` hold the variant names; swapped re-asks included).
- `<A>__vs__<B>/REPORT.md`: verdict line, identical vs changed winners, wins overall and by brand (confidence ≥ 2; per cell, majority of original and swapped asks), mean ship-readiness, seconds per cell, sign-test p-value.

**How it's made:** `python -m dev.distill.ab_engine run | judge | report` (see the module docstring). Every step is resumable and shardable (`--shard K/N`, then `--merge`). The judge uses the same harsh rubric, swap share 0.2 and 4 pairs per call as `dev/distill/judge.py`.

**Feeds:** go/no-go decisions on engine changes. Not training data.

**Caveat:** only cells in `candidate-pool/pool.jsonl` are compared (targets near the source's aspect are not in it). The teacher is Claude's taste against the official rubric, not the official scorer.
