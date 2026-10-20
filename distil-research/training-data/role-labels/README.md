# role-labels: teacher's reading of each ad (student A), not generated yet

**What it will be:** for each corpus ad, the teacher's label for every unit (headline, logo, CTA, hero image, legal, badge, ...) and its importance ranking. It uses the same prompt and JSON schema as the engine's own LLM path (`chameleon/engine/pipeline/priority/prompt.py`), and the same format as `chameleon/dev/golden/*.json`.

**Why:** in the scored run there's no network, so the engine ranks pieces with hand-written rules. A small model trained on these labels could bring LLM-quality ranking offline. It only ships if it beats the rules on holdout ads.

**Planned files:** `labels/<task_id>.json` (`{"roles": {...}, "ranking": [...]}`), plus `agreement.md` comparing the teacher's labels with the hand labels on the 3 samples.

**Feeds:** student A (a role classifier + ranker over `unit_features`).
