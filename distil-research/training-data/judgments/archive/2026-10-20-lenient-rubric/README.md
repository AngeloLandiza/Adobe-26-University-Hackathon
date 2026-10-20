# Archived: judgements under the first (lenient) rubric

**What:** 180 pairwise judgements by Claude Opus 5.5 (Claude Code), made 2026-10-20 with the first version of `chameleon/dev/distill/rubric.md`. That version quoted the official criteria but asked for no professional design critique and no absolute scores.

**Why archived, not used:**
1. The pilot showed this rubric's lowest-confidence answers flipped when A and B were swapped.
2. The team decided the judge must critique like a demanding senior designer, by UI/UX and professional design principles. The rubric was rewritten that way, and each candidate now also gets a 1–5 ship-readiness score.

These rows have no `rubric_version` and no scores, so `train_grader` never uses them by default. They are kept only for comparing lenient vs harsh judging.
