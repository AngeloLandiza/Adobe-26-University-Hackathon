# Research notes: node 1 without network (2026-10-23)

**Question:** stage-one scoring runs our image with `docker run --network none`, so the Claude/Gemini path can't run there. What should label and rank a design's units offline: an algorithm, or a local LLM?

**Short answer:** most likely a small model trained on LLM labels, built on improved rules. But the user asked for evidence first, so all three options (better rules, trained model, local LLM) get measured on the same held-out real ads before any of them becomes the default.
- Design: [`docs/superpowers/specs/2026-10-23-offline-priority-node-design.md`](../superpowers/specs/2026-10-23-offline-priority-node-design.md)
- Plan: [`docs/superpowers/plans/2026-10-23-offline-priority-node.md`](../superpowers/plans/2026-10-23-offline-priority-node.md)

## 1. What the LLM actually does in our engine

The LLM is only used in node 1 (`chameleon/engine/pipeline/priority/`): it gives each unit a role and ranks the content units. Node 2 (layout) and node 3 (grading) are pure Python. With no key, or no network, `plan()` already falls back to the rule-based ranker (`heuristic.py`). So **the scored engine today is the rules.**

The labels change the final layout:
- node 2 gives slots by role and in rank order (`layout/generate.py`);
- each role has its own minimum text size (`place.floor_px`);
- the CTA is never cut to the minor zone;
- node 3 weights its scores by `rank ** -0.7` (`grading/metrics.py`, `pixels.py`).

A wrong headline gets the best slot.

## 2. How good are the rules? (PR #5's benchmark, `dev/bench_results/2026-10-19_213054`)

The rules average τ 0.749 against 0.83–0.98 for six hosted models. Per design, the rules' role accuracy / Kendall τ:

| design | rules | best hosted τ |
|---|---|---|
| L1 warmup, L2 core | 1.00 / 1.00 | 1.00 |
| L3 advanced | 1.00 / 0.93 | 0.96 |
| dev 04 flat, no groups | 1.00 / 1.00 | 1.00 |
| dev 05 type only | 0.71 / 0.33 | 1.00 |
| dev 06 rebate fine print | 0.83 / 0.78 | 1.00 |
| dev 07 rotated sticker | 0.62 / 0.71 | 1.00 |
| dev 08 ghost word | 0.57 / −0.07 | 1.00 |
| dev 09 plain-text CTA | 0.71 / 0.80 | 1.00 |
| dev 10 wide lineup | 1.00 / 1.00 | 1.00 |

Every miss is a specific rule gap:
- **05:** the CTA starts with "find", which isn't an action verb, so it becomes a badge. The text wordmark isn't recognised as the logo.
- **06:** the rebate and trademark lines carry no legal marker. The numbered steps are ranked by prominence instead of reading order.
- **07:** "SAVE 20%" is read as a button (its first word, "save", is a verb). The callout and subhead are swapped.
- **08:** a 280 px "SUMMER" at opacity 0.08 is the biggest font, so it becomes the headline and ranks first. Opacity is ignored.
- **09:** a plain-text CTA is not detected (it becomes a subheadline).

**Caveat:** the 3 official samples are what the rules were written against, and the 7 dev designs were written to break them. Their labels were drafted by Claude and haven't been reviewed. Neither set is a fair test, which is why the bake-off uses real ads.

## 3. What the friend's research found (`angelo/distill` → `distil-research/`, [artifact](https://claude.ai/artifact/9bZLwhuod195ZncaMfXVAQ))

- **Their local model was for layout, not labels.**
  - A small open model fine-tuned on layouts Claude designed reached 11–13 against their rule-based "v5" pipeline at best.
  - A model twice the size tied 8–8, and preference training made it worse (7–15).
  - Learned pickers and graders were coin flips.
  - The rule-based v5 pipeline beat our current layout code 70–17 on 50 unseen real ads. That's a layout result, outside this work's scope.
- **They planned this exact role model.** "Student A" is a role classifier + ranker over `unit_features`, trained on teacher labels, shipping only if it beats the rules on held-out ads (`training-data/role-labels/README.md`). It was never built.
- **Their offline research ranks a tree model first** (`docs/research/beating-v5-offline.md`): "a gradient-boosted per-element classifier" (under 50 ms) ahead of a 0.5–1.7B LLM (an estimated 1–25 s on CPU, unmeasured). They ran a Qwen3-4B fine-tune as Q4 GGUF on CPU (`dev/distill/student_cpu_bench.py`) and saw coordinate drift.
- **Hand rules do worse on real ads.** Their Crello role heuristic (`dev/references/crello_roles.py`), checked by eye on 30 real templates, scored about 80% on headlines but only ~25–30% precision on CTAs and logos (`labelling_accuracy.json`).
- **Real ads exist but aren't committed.** `make_real_challenge.py` converts Crello 5.1.0 ad templates (CDLA-Permissive-2.0; the VistaCreate template terms are unverified, so nothing with Crello content is committed).
  - 1,606 ad-format templates were scanned (test split); 327 passed their layout-quality gate.
  - Fixed sets: challenge2 (50), crello_dev (60), challenge3 (50, untouched), kept apart by design family.
  - The conversion is deterministic (seed 0), so we can rebuild the same sets.
- **Their converted ads carry semantic ids** (`headline`, `cta`, `logo`, `product`, `body_N`), set by that heuristic. Our LLM prompt includes unit ids, so ids must be anonymised before any model labels these ads.

## 4. Local LLM: what it would cost

- **Candidates:** CPU-only guides put 3–4B models in the practical sweet spot (Qwen3.5 4B, Gemma 4 E4B, Phi-4-mini, SmolLM3 3B, Llama 3.2 3B), with 1–2B "fast but weaker at reasoning" ([popularai.org](https://www.popularai.org/p/best-cpu-only-local-llm-2026)). We'd use llama.cpp with JSON-schema-constrained output, behind the existing prompt/repair/fallback path.
- **Image:** +2–3 GB at Q4. The image size limit is still unknown (`docs/pipeline/README.md` §10 q3).
- **Speed:** our prompt is ~2,700 input tokens per design, prefilled on CPU. Scoring runs a `linux/amd64` image on an Apple M5 Pro, so probably emulated. Latency there is unmeasured. Latency isn't disqualifying, but a minute per task adds up.
- **Quality:** the smallest hosted model we tested (gemini-3.5-flash-lite) reached only τ 0.83, and a 4B local model is smaller still. That's likely nearer the rules than the big models.

## 5. Options and the call

| | Cost in the image | Expected quality | Risk |
|---|---|---|---|
| A. Better rules | none | fixes the known misses; plateaus on real ads | tuning to the cases we've seen |
| **B. Small model trained on LLM labels** | a weights file, milliseconds, numpy only | close to the teacher where features capture the signal | needs a data/label/training pipeline |
| C. Local 3–4B LLM | +2–3 GB, seconds to minutes per task | unproven, probably below the hosted models | image size, emulated CPU speed |

**Recommendation: B on top of A, decided by a bake-off** on 100 held-out real ads (challenge2 + challenge3):
- Labels come from a teacher (gemini-4-argon). The held-out answer key is voted by three other models and reviewed by a teammate.
- B ships if it beats A by ≥ 0.03 τ, with a paired bootstrap 95% CI excluding 0.
- C replaces the winner only if it beats it by ≥ 0.05 τ, runs in ≤ 30 s per ad on the Mac, and adds ≤ 3 GB to the image.
- Any winner must keep the official samples' labels exactly right.

## 6. Risks and open questions

- **Label bias:** the trained model copies one teacher. The answer key leaves the teacher out so B gets no free advantage. The synthetic and dev designs are Claude-written, so they're secondary evidence only.
- **Small held-out set:** 100 ads give wide confidence intervals; that's why the thresholds are in τ with CIs, not raw wins.
- **Running the friend's converter on Windows:** it creates a symlink (Windows needs Developer Mode for that). The fallback is to get Angelo's built folders.
- **Organiser questions still open:** whether network is allowed during scoring, the image size limit, and the reference hardware.
- **Scope:** layout is probably the bigger quality lever (the friend's 70–17), but this work is node 1 only.

Sources: [Best CPU-only local LLMs in 2026](https://www.popularai.org/p/best-cpu-only-local-llm-2026), [cyberagent/crello on Hugging Face](https://huggingface.co/datasets/cyberagent/crello/commit/136add81db43f7dcc600267098ab2815f9452dcc), the friend's [research artifact](https://claude.ai/artifact/9bZLwhuod195ZncaMfXVAQ) and branch `angelo/distill`.
