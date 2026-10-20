# floor-plans: teacher-designed templates (student B2), not generated yet

**What it will be:** for each training ad and target shape class, the teacher proposes a few floor plans. A floor plan is a way to cut the canvas into named slots (`hero`, `text`, `logo`, `cta`, `badge`, `minor`), in the same JSON format as `chameleon/engine/pipeline/layout/extra_templates.json`. The engine fills each one with the real pieces, and the teacher judges it against that cell's current winner.

**Why:** today the engine has only 3 floor plans (`stack`, `columns`, `row`), and tall and square targets get just one. Clustering the floor plans that win (grouping similar winners and taking the most typical member of each group) gives new templates found from data rather than guessed.

**Planned files:** `proposals.jsonl` (each proposed plan, with the teacher, ad and target), `wins.jsonl` (judged results), `clusters.json` (the chosen templates and their win rates on holdout ads).

**Feeds:** new entries in `extra_templates.json` (B2), then a per-ad floor-plan recommender (C).

## B1: hand-drawn floor plans (2026-10-20)

`b1_hand_drawn.json` records five hand-drawn floor plans for the shapes that had only the `stack` template (square, tall, very tall), plus landscape. Each was tested on the 3 samples × the 7 square/tall/very-tall/landscape target sizes, with the current hand-set grader picking the winner:

| Plan | Shape classes | Wins | Kept |
|---|---|---|---|
| `split_hero_left` | square, landscape | Level 3 → YouTube 1280×720 (big bottle left, text right; old winner had cramped text) | yes |
| `text_first` | tall, very tall | Level 1 → Instagram portrait (headline properly sized; the reflow had shrunk it) | yes |
| `story_centered` | very tall | Level 2 → half page 300×600 | yes |
| `hero_band` | tall, square | none | no |
| `split_hero_right` | square, landscape | none | no |

The kept three ship in `chameleon/engine/pipeline/layout/extra_templates.json`.

**Caveat:** "wins" is judged by the *uncalibrated* grader. Once the learned weights from student D exist, re-run the comparison (`b1_hand_drawn.json` keeps the rejected plans for exactly that).
