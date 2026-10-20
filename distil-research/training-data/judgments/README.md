# judgments: "which layout is better, and why"

**What it is:** pairwise preferences between two candidate layouts of the same cell. Each judgement includes the original ad, the winner (`a`, `b` or `tie`), a confidence of 1–3, and a short reason.

**Why it exists:** it's the teacher's taste turned into data. Student D fits the grader's weights so that the engine picks what the teacher would pick. The judging criteria are the challenge's official rubric (brief Section 2.5), quoted in `chameleon/dev/distill/rubric.md`, because the official scorer is itself an AI judge using that rubric.

**Files:**
- `judgments.jsonl`, one row per judged pair:
  ```json
  {"cell": "...", "task_id": "...", "split": "train", "a": "<candidate_id>", "b": "<candidate_id>",
   "winner": "a", "confidence": 2, "score_a": 3, "score_b": 2,
   "reason": "A: … specific flaws …  B: … specific flaws …",
   "teacher": "claude-code:claude-opus-5-5", "rubric_version": "f80dbb11", "swapped": false}
  ```
  `score_a` / `score_b` are absolute **ship-readiness** scores, 1–5: 5 means a senior designer would ship it as is; 2 means clearly broken. `rubric_version` is a hash of `rubric.md` at judging time. Training uses only the current version.
  `swapped: true` rows re-ask a pair with the display order reversed. They measure position bias (a judge preferring whatever is shown first) and self-consistency.
- `pilot/`: the 25-pair calibration check. Claude and a person judge the same pairs blind; if they agree on ≥ 18 of 25, the rubric and prompt are trusted.

**How it's made:** `python -m dev.distill.judge` (resumable: re-running skips pairs already judged). Four pairs per teacher call; the A/B order is shuffled by seed.

**Feeds:** student D, `python -m dev.distill.train_grader`. Later also B2, scoring teacher-designed floor plans.

**Caveat:** this is Claude's taste against the official rubric, not the official scorer itself. Results are checked on holdout ads before anything is trusted.

## Generation log

- **2026-10-20, pilot:** `python -m dev.distill.pilot make` picked 25 pairs from 25 different train cells (`pilot/pairs.json`).
  - **The person's half:** open `pilot/page.nosync/judge.html` (built by `python -m dev.distill.pilot_page`). It shows each pair at real size next to the original, with A / B / tie buttons. Download the CSV over `pilot/human_answers.csv`.
  - **Claude's half:** `python -m dev.distill.pilot judge` (every pair is also re-asked swapped) writes `pilot/teacher_judgments.jsonl`, kept separate from training data.
  - **Scoring:** `python -m dev.distill.pilot score` writes `pilot/RESULTS.md`.
- **2026-10-20, main run:** `python -m dev.distill.judge --cells 150 --per-cell 4 --pairs-per-call 4 --swap-share 0.2 --split all --seed 0`, with Claude Opus 5.5 through Claude Code.

## Decisions (2026-10-20)

- **The judge is a harsh professional critic.** `rubric.md` now asks for a strict senior-designer review. It keeps the official criteria verbatim, adds explicit UI/UX and design principles (hierarchy, reading order, grid and alignment, spacing, proximity and grouping, typography, contrast at size, balance, fidelity, fit to format), and adds a 1–5 ship-readiness score per candidate.
- **Claude's judgements are the label.** The team chose Claude's judgements over a human gate. The pilot is kept as a record (`pilot/`, made under the earlier rubric): Claude agreed with the person on 12 of 14 pairs where the person picked a winner. The disagreements were pairs the person called a tie, where Claude answered at confidence 1.
- **Training uses confidence ≥ 2 only** (`train_grader --min-confidence 2`, the default). Under the earlier rubric, confidence-1 answers flipped when A and B were swapped (68% consistent overall; 86% once they count as ties).
- **The earlier rubric's 180 judgements are archived** in `archive/2026-10-20-lenient-rubric/` and never used for training.
- **The main run under the harsh rubric:** `python -m dev.distill.judge --cells 150 --per-cell 4 --pairs-per-call 4 --swap-share 0.2 --split all --seed 0` (rubric `f80dbb11`).
