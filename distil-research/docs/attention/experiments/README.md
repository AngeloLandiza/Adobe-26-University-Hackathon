# Attention experiments log

One file per experiment run, named `YYYY-MM-DD-E<n>-<short-name>.md` (e.g. `2026-10-22-E1-relation-features.md`). The experiment numbers (E0–E5) are defined in [`../README.md` §7](../README.md#7-experiments-in-order).

Write a log even when a run fails or gets abandoned. The point is that the next person doesn't repeat it.

## Template

```markdown
# E<n>: <what was tried>

**Date / who:**
**Commit:** <git sha the run used>
**Question:** <the one thing this run answers>

## Setup
- Data: corpus version (number of ads, train/holdout), judgements file + teacher, corruption types (if any)
- Model: layers, width, heads, parameters; features on/off
- Training: epochs, learning rate, seed(s)
- Command: `...`

## Results
| Model | Holdout acc (all) | Solstice holdout | Off-brand holdout | Train acc |
|---|---|---|---|---|
| baseline (E0 / E1) | | | | |
| this run | | | | |

## What we learned
- ...

## Decision
Go / no-go / change and rerun, and why.
```
