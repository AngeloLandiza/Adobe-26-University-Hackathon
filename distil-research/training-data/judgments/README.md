# judgments: "which layout is better, and why"

**What it is:** pairwise preferences between two candidate layouts of the same cell. Each judgement includes the original ad, the winner (`a`, `b` or `tie`), a confidence of 1–3, and a short reason.

**Why it exists:** it's the teacher's taste turned into data. Student D fits the grader's weights so that the engine picks what the teacher would pick. The judging criteria are the challenge's official rubric (brief Section 2.5), quoted in `chameleon/dev/distill/rubric.md`, because the official scorer is itself an AI judge using that rubric.

**Files:**
- `judgments.jsonl`, one row per judged pair:
  ```json
  {"cell": "...", "a": "<candidate_id>", "b": "<candidate_id>", "winner": "a", "confidence": 2,
   "reason": "headline readable, product not cramped", "teacher": "claude-code:claude-opus-5-5", "swapped": false}
  ```
  `swapped: true` rows re-ask a pair with the display order reversed. They measure position bias (a judge preferring whatever is shown first) and self-consistency.
- `pilot/`: the 25-pair calibration check. Claude and a person judge the same pairs blind; if they agree on ≥ 18 of 25, the rubric and prompt are trusted.

**How it's made:** `python -m dev.distill.judge` (resumable: re-running skips pairs already judged). Four pairs per teacher call; the A/B order is shuffled by seed.

**Feeds:** student D, `python -m dev.distill.train_grader`. Later also B2, scoring teacher-designed floor plans.

**Caveat:** this is Claude's taste against the official rubric, not the official scorer itself. Results are checked on holdout ads before anything is trusted.

## Generation log
