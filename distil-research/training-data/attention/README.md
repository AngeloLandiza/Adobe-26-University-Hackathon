# attention: data for the attention research

**What it is:** inputs for the attention research in [`../../docs/attention/README.md`](../../docs/attention/README.md). For now it holds one file: the hand-written **relation features** of every candidate layout the teacher has judged. A relation feature measures how two specific pieces of a layout relate. For example, `logo_cta_gap` is how far the logo sits from the call-to-action button, and `reading_order` is how much of the source's top-to-bottom, left-to-right order survives the resize.

**Why it exists:** experiment E1 (attention README §7) asks whether a linear grader that sees these relations predicts the teacher's pairwise judgements better than one that sees only the grader's 12 summary metrics. The answer sets the bar a future attention model has to beat.

**Files:**
- `relation_features.jsonl`, one row per judged candidate:
  ```json
  {"cell": "c01_square_golden_hour@billboard", "candidate_id": "row/cutoff=4/margin=0.04/mirror=True/split=0.45",
   "features": {"has_headline": 1.0, "logo_cta_gap": 0.5, "reading_order": 0.83, "...": "29 keys, each in [0, 1]"}}
  ```
  The 29 features (6 `has_<role>` flags + 23 relations) are defined in the docstring table of `chameleon/dev/attention/relations.py`. When an ad lacks a role a feature needs (say, it has no CTA), the feature gets a neutral default and `has_cta` is 0.

**How it's made:** `python -m dev.attention.features_dataset` (from `distil-research/chameleon/`; no teacher calls, about 4 seconds with 4 worker processes). The pool stores only each candidate's 12 metrics, not the layout itself. So for every cell in `judgments/judgments.jsonl` the script rebuilds the candidates exactly as `dev/distill/pool.py` does: the heuristic plan over the merged assets folder, `safe_fallback`, then `generate_candidates`. It keeps the judged candidates by `candidate_id` and computes the features. As a check, the script recomputes each rebuilt candidate's geometry metrics and compares them with the pool row. Rerun it whenever `judgments.jsonl` grows.

**Feeds:** `python -m dev.distill.train_grader --extra-features ../training-data/attention/relation_features.jsonl` (from `chameleon/`), which reports signed logistic models on the 12 metrics plus these features next to the 12-metric graders. Those models are never written to the engine, because the engine cannot compute these features yet. Later, `relations.py` can serve as a probe for the attention model's H4 head.

## Generation log

- **2026-10-20:** `python -m dev.attention.features_dataset --workers 4` over the 715 judgements (rubric `f80dbb11`) covered **149 cells and 882 candidates**. Every judged candidate was rebuilt (0 missing, 0 errors), and the rebuilt geometry metrics match the pool exactly (largest difference 0.0). The file is 832 KB.
