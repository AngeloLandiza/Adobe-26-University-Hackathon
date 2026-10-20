# Training data

All data for the distillation research lives here, one folder per dataset. **Nothing here is used by the real engine (`../../chameleon/`) or shipped in its Docker image.**

## How the pieces fit

```
corpus/ ──► candidate-pool/ ──► judgments/ ──► learned grader weights (student D)
  ads          every layout the      "A beats B"      chameleon/engine/pipeline/grading/weights.json
               engine makes for      from the teacher
               each ad × size,       (+ a human pilot)
               with its scores
corpus/ ──► floor-plans/  (teacher-designed templates → clustering → new templates, student B2)
corpus/ ──► role-labels/  (teacher's "what is each piece / how important" → small ranker, student A)
teacher-cache/            raw teacher answers, so reruns cost nothing (gitignored)
```

- The **teacher** is a big AI model run on a laptop, currently Claude Opus 5.5 through Claude Code (`claude -p`).
- A **student** is what we build from the data that could later ship: a data file, never a model call.
- The full plan: [`../docs/superpowers/plans/2026-10-20-distillation.md`](../docs/superpowers/plans/2026-10-20-distillation.md).

| Folder | What it is | Made by | Feeds | Committed? |
|---|---|---|---|---|
| [corpus/](corpus/) | source ads (design JSON), split train/holdout and tagged `brand` solstice/offbrand: originals, seeded augmented variants (same split as their parent) and off-brand ads; `corpus/assets/` holds 25 code-drawn `syn_*` stand-in images | hand + Claude, reviewed; `make_corpus`, `augment`, `make_offbrand`, `synth_images` | everything | yes |
| [candidate-pool/](candidate-pool/) | every candidate layout per ad × target size, with the grader's 12 scores and the ad's `brand` | `python -m dev.distill.pool` | judgments, student D | `pool.jsonl` yes, renders no |
| [judgments/](judgments/) | pairwise "which layout is better, and why" | the teacher (+ human pilot) | student D (and B2) | yes |
| [floor-plans/](floor-plans/) | teacher-designed floor plans and their win rates | not generated yet (B2) | student B2, then C | yes |
| [role-labels/](role-labels/) | teacher's role + importance labels per ad | not generated yet (A) | student A | yes |
| [teacher-cache/](teacher-cache/) | raw teacher responses keyed by prompt hash | `dev/distill/teacher.py` | reruns only | no |

## Rules

- **Holdout ads are never trained on.** They exist to check whether a student really helps on ads it hasn't seen. The brief warns that "anything tuned to one design or one size will fail the hidden and unseen cases."
- Only images from `../chameleon/brand_kit/assets/` or `corpus/assets/`. `corpus/assets/syn_*` are simple code-drawn stand-ins (gradients, patterns, product silhouettes; no generative AI, no downloads) used only as research training inputs; the engine never generates, edits or ships imagery, which is the challenge's disqualifying rule.
- **A variant of a holdout ad is holdout.** Augmented ads inherit their parent's split; the grader report breaks holdout accuracy down by `brand` (Solstice vs off-brand).
- No API keys or personal data in any file here.
- Commands run from `distil-research/chameleon/` with `../../chameleon/.venv/bin/python`.
