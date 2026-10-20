# D2: learned vs hand-set picks

```
cells: 612
default              fallback wins   40   picks changed vs default    0
learned              fallback wins   59   picks changed vs default  224
learned_no_fallback  fallback wins    1   picks changed vs default  249

teacher verdicts, variant pick vs default pick (cells where they differ; confidence >= 2):
  learned              holdout  solstice  variant wins  19 /  40 = 48%
  learned              holdout  offbrand  variant wins  13 /  22 = 59%
  learned              train    solstice  variant wins  49 /  79 = 62%
  learned              train    offbrand  variant wins  32 /  51 = 63%
  learned_no_fallback  holdout  solstice  variant wins  19 /  45 = 42%
  learned_no_fallback  holdout  offbrand  variant wins  13 /  27 = 48%
  learned_no_fallback  train    solstice  variant wins  43 /  80 = 54%
  learned_no_fallback  train    offbrand  variant wins  31 /  61 = 51%
```

## Conclusions (2026-10-20)

**Question:** the learned weights agree with the teacher more often on judged pairs (holdout 70.4% → 77.8%). Do they also make the engine's *final pick* better?

**Method:** replay `pick_best` from the pool's stored scores for all 612 cells under three weightings. Then the teacher (harsh rubric) judges the hand-set pick against each variant's pick, in every cell where they differ: 330 judgements, 16 workers, 0 failures.

**Findings:**
1. **Learned weights don't improve the final pick on the ads that matter.** On holdout Solstice ads the learned pick wins 19 of 40 (48%), a coin flip. Off-brand holdout: 13 of 22 (59%, small sample). On train ads: about 62%. The pairwise gain doesn't carry over to choosing one winner per cell.
2. **"Never pick the fallback" makes things worse:** holdout 42% (Solstice) and 48% (off-brand).
3. **When today's grader picks the plain-shrink fallback, the teacher agrees: 34 to 3** (9 unsure). The shrink averages 2.8/5 ship-readiness, the engine's real layouts in those cells 2.1/5. In those cells, the engine's re-arranged layouts are worse than not re-arranging at all.

**Decision:**
- **Do not ship the learned weights.**
- **Do not force a real layout over the fallback** (this contradicts the "Level 3 at 300×250 must not pick the fallback" test from the engine's own review: the teacher disagrees with that test's premise for these ads; raise with Esteban).
- **Next:** put effort into better *candidates* (B2 floor plans; attention-based scoring later), not better picking among today's candidates. Re-run D2 once new floor plans exist.
