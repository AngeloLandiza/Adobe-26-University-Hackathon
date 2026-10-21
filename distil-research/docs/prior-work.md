# Prior work: what research, products, judges, other teams and design practice say about resizing ad layouts

Researched 2026-10-21. Scope: about 25 sources read in some depth (papers, product docs, four public repos from other CHAMELEON teams), chosen for relevance to *our* problem. That problem: a fixed set of elements, moved and resized onto a new canvas, then judged by a vision-language model against the original. Every source has a link. Where I only read an abstract or a secondary summary, I say so. No code was copied. The other teams' repos have no licence, which means all rights are reserved, so only their ideas are described here.

## Terms used in this document

- **Retargeting** means adapting an existing design to a new size or aspect ratio while keeping its content. Example: a 1080x1080 post becomes a 728x90 leaderboard.
- **Layout generation** means producing element boxes, either from nothing or from a few conditions such as "these element types" or "this background image". Most ML layout papers do this, not retargeting.
- **Conditional layout task names** used in papers:
  - **C→S+P** means "given the element categories, predict sizes and positions".
  - **C+S→P** means "given categories and sizes, predict positions only". Retargeting is closest to these.
- **Content-aware layout** means the layout is placed on a given background image and avoids its important regions. Example: text kept off a face.
- **Saliency** (or **importance**) map: a per-pixel estimate of where people look.
- **Underlay** is a shape that sits behind text to make the text readable. Example: the pill behind a CTA label.
- **VLM** stands for vision-language model, a model that reads images and text (GPT-4o, Claude, Gemini, Qwen-VL). A **VLM judge** is such a model asked to score or compare rendered designs.
- **Position bias** happens when a pairwise judge's verdict depends on which image is shown first.
- **Modular type scale** is a set of font sizes in which each step is a fixed ratio times the previous one. Example: 1.25 gives 16, 20, 25, 31 px.
- **Widow / orphan**: a single short word left alone on the last line of a paragraph. In ads, people use either term for it.
- **Relation graph** is a list of pairwise facts about a design. Examples: "the CTA is smaller than the headline", "the logo is above the headline", "these two texts share a left edge".

## Bottom line

The strongest pattern in the literature and in other teams' work is that a good resize keeps the source's *relations* (which element is bigger, which is above or left of which, which ones share an edge) and changes only the *geometry*. Four sources do this: ReLayout (2026), Neural Design Network, the 2014 O'Donovan retargeting model, and the "source-relative hierarchy" rule in another CHAMELEON team's engine. Our remaining flaws (CTA oversized vs the headline, legal text blown up, mixed alignment) are almost all **broken relations from the source**, not lack of space. Our bisection engine also currently *permits* the main one by design:
- non-headline text may reach 0.9x the headline (`HIERARCHY_CAP`) and 1.3x its source ratio to the headline (`RATIO_SLACK`);
- the CTA's readability floor (11 px) is 79% of the headline's (14 px), so on strips, where the headline sits at its floor, a CTA pill reads as big as or bigger than the headline.

Ranked by expected gain for our VLM-judged A/B, with effort:

| Rank | Change | Expected gain | Effort |
|---|---|---|---|
| 1 | **Hierarchy lock**: clamp every subordinate text, and the CTA pill's area, to the source's size ratio vs the headline, with absolute caps from design practice (CTA label ≤ 0.6x headline, legal ≤ 0.45x headline and ≤ its source ratio) | Biggest. It hits the judge's #1 complaint (hierarchy) and the CTA and legal flaws directly | Small, about half a day |
| 2 | **Relation-preservation term in the grader/picker**: share of the source's size-order and 3x3 relative-position relations kept, used as a filter or tie-breaker | Medium. A faithfulness signal the grader lacks today. The tournament showed the picker is the bottleneck once candidates are diverse | About 1 day |
| 3 | **Alignment snap post-pass**: snap near-aligned edges and centre axes within a column to one shared axis; accept only if gates pass; never change sizes | Medium-small. Targets "mixed alignment". Unlike the failed S3 repair, it never enlarges anything | About 1 day |
| 4 | **Near-aspect source-fit**: when target and source aspect are close (ratio of aspects within about 1.35x), prefer a uniform fit of the original with re-centred margins | Medium on those cells, none elsewhere. The teacher already preferred the plain shrink 34–3 where it was picked | Small |
| 5 | **Source-proportional gaps**: distribute leftover space between blocks in the proportions the source designer used, instead of evenly or by value | Small to medium on "cramped next to empty". Risky because FLEX lost | About 1 day |
| 6 | **Judge hygiene**: ask every decisive pair in both orders and count disagreement as a tie (today 20% are swapped) | No engine gain, but it protects decisions from position bias, which TASTE measured as large | Small, at about 1.8x judge cost on decided cells |

Learned layout models (LayoutDM, RALF, LayoutPrompter, AesthetiQ) are real options but not worth it for stage one. They need Crello training, and they solve generation rather than faithful retargeting. Our per-ad retrieval already captures their main win, which is reusing designer layouts.

---

## 1. Academic prior work on retargeting and layout generation

### 1a. Work that actually retargets a given design to a new size

**O'Donovan, Agarwala, Hertzmann, "Learning Layouts for Single-Page Graphic Designs", IEEE TVCG 2014.** https://www.dgp.toronto.edu/~donovan/layout/ (DOI 10.1109/TVCG.2014.48)
- **Key idea:** an energy function built from design principles. An energy function is a cost to minimise, with terms for alignment, importance-proportional size, balance, white space, overlap and similar principles (I read the project page, not the full PDF, so the exact list is unconfirmed). Its weights are learned from a few example layouts by "nonlinear inverse optimization", meaning weights are fitted so that the real layouts come out as optimal.
- **Retargeting:** applied to new sizes (98 retargets in the supplement). The model performed "as well as, or better than, novice designers" in Mechanical Turk tests.
- **Actionable for us:** this is the classical blueprint for "a few hand-weighted terms plus search", which our grader already is. The lesson that carries over is that **importance determines size**: an element's area should track its importance rank. Our bisection value already uses priority. What it lacks is the converse constraint, *a lower-ranked element may not out-mass a higher-ranked one* (change #1).

**Li, Yang, Zhang et al., "Attribute-Conditioned Layout GAN for Automatic Graphic Design", IEEE TVCG 2020 (Adobe Research co-authors).** DOI 10.1109/TVCG.2020.2999335
- **Key idea:** a layout generative adversarial network (GAN), a generator trained against a critic network, conditioned on per-element attributes such as area, aspect ratio and reading order. They demonstrate "retargeting existing layouts to new sizes while preserving elements' reading-orders", keeping element area ratios and logo aspect.
- **Actionable for us:** the attributes it chose to preserve when retargeting are reading order, relative area and aspect. That is exactly the relation set for change #2.

**Jacobs, Li, Schrier, Bargeron, Salesin, "Adaptive Grid-Based Document Layout", SIGGRAPH 2003 (Microsoft Research; Wilmot Li later at Adobe).** https://research.adobe.com/publication/adaptive-grid-based-document-layout
- **Key idea:** a style is a *set of grid templates*, each valid for a range of page sizes. Each template holds one-way constraints, formulas computed from other variables, such as "figure width = 2 columns + gutter". At display time the engine picks the best-fitting template and solves its constraints.
- **Actionable for us:** this is the ancestor of our archetype/grid strategies. Our S4 grid lost on strips, so treat it as already tried. The transferable idea is to choose the template per aspect band and then solve sizes by constraint, rather than one universal algorithm. Bisection already does a version of this implicitly.

**Dayama et al., "Interactive Layout Transfer", IUI 2021.** DOI 10.1145/3397481.3450652
**Warner, Zhou, Hartmann, "Interactively Optimizing Layout Transfer for Vector Graphics" (VLT), arXiv 2023.** https://arxiv.org/abs/2309.11635
- **Key idea:** take the layout of design B and apply it to design A's elements.
  - VLT first extracts **layout rules** from each design: containment, relative order, vertical/horizontal alignment, overlap, margin offset and same width/height. It then matches elements and transforms A while keeping as many rules as possible.
  - Its planned objective rewards the number of elements following each rule, plus non-overlap, plus fewer unique property values (consistency).
  - VLT states that copying positions and sizes alone "cannot handle canvas size or aspect-ratio changes".
- **Actionable for us:** "fewer unique property values" is a cheap, judge-relevant consistency measure: few distinct left edges, few distinct font sizes, few distinct gap sizes. It is a natural extra grader term and pairs with change #3.

**Lin et al., "ReLayout: Versatile and Structure-Preserving Design Layout Editing via Relation-Aware Design Reconstruction", arXiv 2602.01046, Feb 2026 (Microsoft Research and XJTU).** https://arxiv.org/html/2602.01046v1
- **Key idea:** a multimodal LLM (Llama-3.1-8B backbone) edits layouts (move, resize, add, delete) while preserving a **relation graph**.
  - **Size relation** between two elements: area ratio below 1−α means smaller, within ±α means equal, above 1+α means larger.
  - **Position relation:** the 3x3 cell of another element's extended box, or of the canvas, that the element's centre falls in.
  - Structure preservation is scored as **Size Rel** (share of size relations kept) and **Pos Rel** (share of position relations kept).
  - It does not change canvas size. Code is not released; the paper is CC BY-NC-ND.
- **Actionable for us:** these two metrics are trivially computable on our JSON and are exactly a "faithful to the original" proxy. Change #2 is to add `size_rel` and `pos_rel` (source vs output) to the grader and gate. A CTA that becomes "larger" than the headline when it was "smaller" in the source is a size-relation violation, which is our top flaw.

**Lee et al., "Neural Design Network", ECCV 2020** (graph of size and position relations as layout constraints). This is the earlier origin of ReLayout's relation types. I did not open it this session; the citation is from memory, so check it before quoting.

**Graphic-Design-Bench (Deganutti et al., arXiv 2604.04192, Apr 2026).** https://arxiv.org/html/2604.04192v2
- **Key idea:** 50 design tasks built on LICA layered templates, including a "Multi-Aspect Ratio Adaptation" task (9:16 → 1:1, only 13 layouts).
- Image-generation models were tested:
  - GPT-Image-1.5 kept text (TextAcc 1.0) but kept only 23% of core assets.
  - Gemini-3.1-Flash-Image kept assets (1.0) but dropped some text (0.92).
  - Conclusion: "no single system consistently preserves all design constraints."
- Its pairwise MLLM judge, "M-Judge", compares against the ground truth on aesthetics, clarity, usability, creativity and consistency. Code: https://github.com/purvanshi/lica-bench (licence NOASSERTION, so do not reuse).
- **Actionable for us:** confirms that generative re-rendering is not competitive on faithfulness, and the rules forbid it anyway. Structured relayout like ours is the right family. Their judge criteria are worth mirroring in our teacher prompt as a check: we already cover clarity and consistency, and "usability" maps to CTA findability.

### 1b. Layout generators that could be conditioned on our elements (ML)

| Work | Key idea (2–3 sentences) | Offline, small? | Code / licence |
|---|---|---|---|
| LayoutGAN, Li et al., ICLR 2019, https://arxiv.org/abs/1901.06767 | First GAN over element boxes, with a "wireframe" renderer so a critic judges alignment and overlap visually. | Small, but trained per dataset. | – |
| LayoutTransformer, Gupta et al., ICCV 2021, https://arxiv.org/abs/2006.14615 | Autoregressive transformer that writes boxes one by one and can complete partial layouts. | Small. | – |
| LayoutDM, Inoue et al. (CyberAgent), CVPR 2023, https://arxiv.org/abs/2303.08137 | Discrete diffusion over box tokens. A single model handles C→S+P, C+S→P, completion and refinement by masking what is given. Trained on Rico, PubLayNet and Crello. | Yes, a few tens of M parameters, CPU-feasible. | https://github.com/CyberAgentAILab/layout-dm (Apache-2.0) |
| LayoutFormer++, Jiang et al., CVPR 2023, https://arxiv.org/abs/2208.08037 | Serialises user constraints (types, sizes, relations) into the sequence and restricts decoding so the constraints hold. | Small. | microsoft/LayoutGeneration (MIT) |
| Kikuchi et al., "Constrained Graphic Layout Generation via Latent Optimization", ACM MM 2021, https://arxiv.org/abs/2108.00871 | Generate with a pretrained GAN, then optimise the latent code so that alignment, overlap and user relations hold ("beautification"). | Small. | – |
| LACE, ICLR 2024, https://arxiv.org/abs/2402.04754 | Continuous diffusion with differentiable **global alignment** and **pairwise overlap** losses, applied during training *and as a post-processing step*. | Small. | https://github.com/puar-playground/LACE (MIT) |
| PosterLayout / DS-GAN, Hsu et al., CVPR 2023, https://arxiv.org/abs/2303.15937 | Content-aware poster layouts on a background image. Defines the standard metric suite (below). | Small. | GitHub repo has no licence |
| LayoutPrompter, Lin et al., NeurIPS 2023, https://arxiv.org/abs/2311.06495 | Prompts an LLM with retrieved exemplar layouts (dynamic exemplar selection), serialised as text, then **ranks** outputs by alignment/overlap metrics. No training. | Needs a large LLM, so not offline. | microsoft/LayoutGeneration (MIT) |
| RALF, Horita et al. (CyberAgent), CVPR 2024, https://arxiv.org/abs/2311.13602 | Retrieves nearest-neighbour layouts for the input and feeds them to the generator. Retrieval alone cut FID for weaker baselines (e.g. CGL-GAN 34.5 → 13.3). | Medium. | https://github.com/CyberAgentAILab/RALF (Apache-2.0) |
| LayoutNUWA, ICLR 2024, https://arxiv.org/abs/2309.09506 | Treats layout as HTML code that an LLM completes. | Needs a 7B model. | ProjectNUWA/LayoutNUWA (MIT) |
| Desigen, Weng et al., CVPR 2024, https://arxiv.org/abs/2403.09093 | Generates background and layout together, with attention reduction so the background leaves room for text. | Diffusion model; not offline. | – |
| COLE (Jia et al., https://arxiv.org/abs/2311.16974) / OpenCOLE (Inoue et al., CVPRW 2024, https://arxiv.org/abs/2406.08232) | Text-to-design pipelines (LLM plan → image → layout → typography), evaluated with GPT-4V on 5 dimensions. | No. | OpenCOLE: CyberAgentAILab/OpenCOLE (Apache-2.0) |
| AesthetiQ, Patnaik et al. (Adobe MDSR), CVPR 2025, https://arxiv.org/abs/2503.00591 | A 1–8B multimodal LLM predicts layouts and is tuned by DPO (direct preference optimisation, training from "A beats B" pairs) on pairs judged by a VLM (VILA-7B) from *rendered* layouts. They switched to pairwise because "MLLMs often assign similar scores across a range of layouts". | 1B variant is borderline. | – |
| LayoutRectifier, Shen, Shamir, Igarashi, Pacific Graphics 2025, https://arxiv.org/abs/2508.11177 | Training-free post-processing in two stages: (1) **discrete search that snaps elements to a design grid**; (2) continuous optimisation with a box-containment function that removes unwanted overlaps and encourages intended containment, staying close to the input. | Yes, pure optimisation. | Project page only |

**Actionable for us:**
- **RALF and LayoutPrompter are retrieval plus ranking**, which is what our `retr` variant already does with teacher floor plans. Their success supports keeping retrieval; nothing new to try there.
- **LACE and LayoutRectifier show that alignment and overlap can be fixed by a post-pass that stays close to the input.** Our S3 repair lost because it also pushed sizes up (the rulebook rewards big). Change #3 is the narrow version: snap only, with no size changes.
- LayoutDM (Apache-2.0, small) is the one learned model that could legally run offline. But it needs Crello training, and it does not know our sizes or fonts. Park it.

### 1c. Metrics these papers use (what is measurable on our JSON)

From PosterLayout (CVPR 2023) and reused by LayoutPrompter, RALF, PosterO and others:
- **Overlay:** mean overlap (IoU) between non-underlay element pairs. Lower is better.
- **Alignment:** for each element, the distance from its left, centre and right (and top, middle and bottom) lines to the nearest such line of any other element. Lower is better. AesthetiQ's version maps distance d to exp(1−d), normalised.
- **Underlay effectiveness:**
  - *loose:* how much a text sits on its underlay;
  - *strict:* 1 only if the text is fully inside the underlay.
- **Utility:** share of the canvas's non-salient area that is used.
- **Occlusion:** how much salient image area is covered by elements.
- **Readability:** average image gradient under plain text (busy background = less readable).
- **Validity:** share of elements that are not degenerate (too small).

ReLayout adds **Size Rel / Pos Rel** (relation preservation, above). VLT and design-system practice suggest a **consistency** count (number of distinct edges, sizes and gaps).

**Actionable for us:** our gate and grader already cover overlay, contrast and readability. Missing or weak:
- **strict underlay** (CTA label fully inside its pill);
- **Size Rel / Pos Rel** vs the source;
- the **distinct-edges count**.

All three are cheap.

---

## 2. Products: how commercial resize tools decide layouts

Short version: **no product publishes its algorithm.** What is documented is element-level rules.

**Figma constraints and auto layout** (product behaviour; documentation pages and community explanations):
- Each layer has horizontal and vertical constraints: Left, Right, Left & Right (stretch), Center, Scale.
- Inside auto-layout frames, children use Hug (shrink to content), Fixed, or Fill (take the remaining space), and the frame stacks them with fixed gaps and padding.
- Constraints do not apply inside auto layout.
- **Actionable for us:** this is the industry's mental model of a resize: **anchor to an edge, keep fixed padding, let one flexible element absorb the leftover space.** It suggests that the hero or the background, not the copy, should absorb spare space, which is consistent with our "cramped next to empty" complaints.

**Bannersnack (Creatopy's sibling) Smart Resize**, https://www.bannersnack.com/blog/smart-resize-guide (help centre: https://help.bannersnack.com/en/articles/12026114):
- Each element's *bounding box* decides how it scales and where it lands, so text boxes should be trimmed to the content.
- **Anchored elements stay in place and scale proportionally.**
- **Grouped items move and scale together.**
- **Element order is preserved: top-left items stay first, bottom-right last.**
- It claims to "preserve alignments, spacing, and visual hierarchy".
- **Actionable for us:** two cheap rules.
  - (a) Measure text by its *ink* (the measured lines) rather than its source box. We already do this.
  - (b) Reading order is preserved as a hard rule. Bisection keeps source reading order, which is good. The rule is worth adding to the gate as a Pos-Rel check (change #2).

**Celtra** (marketing posts: https://celtra.com/blog/supercharging-creative-production-with-advanced-design-automation/):
- "Auto-Layout", "Auto-Sizes", "Automated Text Treatment" and auto-cropping are built from a **minimum set of master layouts**: the tool works out the fewest master designs needed to cover all sizes.
- A G2 reviewer says the auto-layout and text resizing work but templates are complex to set up.
- **Actionable for us:** the industry answer to strips vs squares is *a few masters per aspect family*, then per-size fitting. That is what our archetype and retrieval work does, so it is already covered.

**Google responsive display ads** (https://developers.google.com/google-ads/api/docs/responsive-display-ads/overview):
- Google assembles ads from separate assets (headlines, long headline, description, images, square and 4:1 logos) and picks combinations with ML.
- Placements can render as text-only or native.
- Blog summaries of guidance: 320x50 should carry about a logo plus four words; 728x90 one line of copy.
- **Actionable for us:** on the smallest strips, real systems **drop** description and badges entirely. We may not drop elements (the same ids must stay), so the equivalent is demoting them to fine print in a minor zone, which bisection does.

**Meta Advantage+ creative** (third-party summaries only, e.g. https://www.jonloomer.com/advantage-plus-creative-enhancements/amp/): uses generative "image expansion" to reach 9:16, "adapt to placement" and "text improvements". These are generative, so they are forbidden for us. Meta's own story safe zones are already in `format-specs.md`.

**Canva Magic Resize / Magic Switch** (https://www.canva.com/help/resize/) and **Adobe Express Resize** (help page returned 403; Express documents Resize with generative Expand for images): neither publishes how elements are rearranged. Canva's help points users to manual element resizing afterwards.

**Adobe XD responsive resize** and Adobe patent US 11,320,975 (https://patents.google.com/patent/US11320975), "Automatically generating and applying GUI resize-constraints based on design semantics":
- Infers per-element constraints (pin left, pin right, fixed width and so on) from the element's role and its relative position to its neighbours and container, using heuristics.
- **Actionable for us:** this is Adobe's own framing. *Infer each element's anchor from the source* (near the left edge → pin left; centred → centre) and keep it. A cheap use: when a source element is edge-anchored (logo top-left, CTA bottom-right), bisection's in-cell alignment should follow that anchor. Today it uses the dominant text alignment.

---

## 3. VLM/LLM judges of design quality

**Haraguchi et al. (CyberAgent), "Can GPTs Evaluate Graphic Design Based on Design Principles?", SIGGRAPH Asia 2024.** https://arxiv.org/abs/2410.08885
- GPT-4o rated 700 banners (100 originals and 600 perturbed) on alignment, overlap and white space, 1–10. Correlation with 60 human raters was *better than* O'Donovan's heuristic metrics.
- **GPT-4o could not see small perturbations.** Heuristics beat it on small misalignments and white-space changes; it caught up on medium and large ones.
- Their prompt pattern: role, "grade seriously", anchored scale, JSON output, a "both" option for ties, and averaging several runs.
- **Actionable for us:** judges punish *visible* misalignment, not 2-pixel misalignment. An alignment snap (change #3) only pays off where offsets are large enough to see, roughly ≥ 1–2% of the canvas. Our rubric already follows their prompt pattern.

**TASTE (Zhu et al., arXiv 2605.20731, 2026)**, designer-annotated preferences on 9 criteria. https://arxiv.org/html/2605.20731
- Open VLM judges agreed with the designer majority only 0.51–0.59 of the time; chance is 0.50.
- The paper's position-bias rate, which measures how far a verdict depends on which image comes first, ranged from 0.44 to 0.87 across judges. The secondary summaries I read disagree on the exact definition, so check the appendix before quoting the numbers.
- Abstract criteria such as visual hierarchy and colour harmony had the most position bias.
- They recommend asking in both orders and scoring order-inconsistent verdicts as ties.
- **Actionable for us:** our harness swaps only 20% of pairs (`swap_share=0.2` in `ab_engine.py`). For the final ship decision on a variant, run the swapped order for every decided cell and treat disagreement as a tie (change #6). Our teacher is a frontier model, not one of the open VLMs measured, so its bias is probably lower, but it is unmeasured.

**"Fooling the LVLM Judges" (EMNLP 2025).** https://arxiv.org/abs/2505.15249
- Score inflation from image-side tricks: overlaid instruction text, bounding-box highlights, brightness and gamma changes, black padding. For example, GPT-4o rose up to +80% with box highlights on some domains.
- Not tested on graphic design.
- **Actionable for us:** nothing to exploit; any such trick would be a fidelity violation. It is a reminder that the official judge can be swayed by *overall brightness and contrast*. Don't let backgrounds get darker or muddier in our resize, for example by over-cropping a light area of the background image.

**AesthetiQ (Adobe MDSR, CVPR 2025)**, above.
- The judge prompt criteria were aesthetics, clarity, usability, creativity and consistency.
- VILA-7B agreed with humans 78.8% and GPT-4o 88.6% (method for the latter not given).
- **Pointwise scores clustered, so they used pairwise judging.** This matches our experience (mean ship-readiness 2.3/5 with little spread).

**BannerAgency (Sony, arXiv 2503.11060, 2025).** https://arxiv.org/abs/2503.11060
- Multi-agent banner generation at 13 IAB sizes, with an MLLM critic.
- GPT-4o vs human ratings on six criteria (audience fit, **logo placement**, **CTA effectiveness**, copy, brand identity, aesthetics) gave Pearson r = 0.85–0.96 (aesthetics r = 0.89).
- **Actionable for us:** logo placement and CTA effectiveness are rated as their *own criteria* by ad-specific judges. So "logo tiny" and "CTA wrong size" are probably scored on their own by the official judge too, not just folded into "intentional".

**DesignProbe (arXiv 2404.14801, 2024)** and its follow-up "Can MLLMs understand graphic design?" (Microsoft Research; best model GPT-4.1 at 65.5% on 1,600 items): models are good at high-level semantics, weak at fine layout and typography perception. Graphic-Design-Bench says the same: "the gap widens sharply as tasks demand precision". **PosterIQ (arXiv 2603.24078, 2026)** found commercial models to be "insensitive automatic raters" of design quality (abstract only).

**UICrit (Duan et al., UIST 2024).** https://arxiv.org/abs/2407.08850 3,059 designer critiques of 983 mobile UIs. Few-shot examples plus visual prompting (boxes drawn on the image) improved LLM critique quality by 55%. **Actionable for us:** if we ever let the teacher point at flaws, ask for a bounding box per flaw. That would make the critique corpus machine-checkable (flaw → element).

**What VLM judges reward and penalise, synthesised.** These are inferences from the papers above plus our 1,955 critiques; no paper tested size or centring bias directly.
- **They reward:**
  - a single clear focal point (hierarchy);
  - visibly shared alignment;
  - text that is big enough to read at a glance;
  - logos and CTAs that are present and noticeable;
  - fidelity to the source's look.
- **They penalise:**
  - visible collisions and text on busy image areas;
  - tiny text;
  - competing focal points;
  - large empty areas next to crowded ones.
- **What they cannot see:** sub-2% misalignments and small spacing inconsistencies.
- **Known bias:** position bias in pairwise asks. Our rulebook-picker test also showed the opposite of a "bigger is better" bias: our harsh teacher penalised layouts in which *everything* got big.

---

## 4. Other teams and public code

Searched GitHub (via `gh api search/repositories`) for chameleon/hackathon/resize/retargeting terms. I found four public CHAMELEON entries, all created in the last month. None has a licence, so all rights are reserved: ideas only, no code.

**fernando-ace/Adobe-Hackathon-Chameleon-Challenge** (https://github.com/fernando-ace/Adobe-Hackathon-Chameleon-Challenge). This is the most developed one; I read its README, research notes, failure taxonomy and engineering walkthrough.
- **Pipeline:**
  - "Design DNA" analysis covers roles, groups, anchors, reading order, alignment and a relationship graph (proximity, alignment, containment, same row or column, group, source overlaps).
  - Families: source-fit, stack, split, mirrored split, banner, very-wide banded. That makes 12–18 candidates, with the top 2 repaired.
- **Ideas worth noting:**
  - **Source-relative hierarchy:** "Existing hierarchy ratios are treated as source constraints; candidates are penalized when they materially worsen them". This removed 108 false flags in their stress tests.
  - **CTA and badge groups are capped at 1.25x their source scale on taller formats "to avoid oversized color slabs"** (exactly our "CTA oversized" flaw).
  - **Source-fit with a gradual adaptation penalty:** for small aspect changes the uniform fit of the original stays competitive, and the penalty grows as the aspects diverge.
  - Overlaps and bleed already in the source don't count as new collisions unless they get worse.
  - A date or time in bold text is not a CTA.
  - They document that the 728x90 Level-3 sample still forces tiny supporting text (headline about 16.6 px, legal about 7.9 px).
- **Also:** an Adobe Express add-on prototype for the finale.

**amanadhav/chameleon** (https://github.com/amanadhav/chameleon). README only.
- Roles inferred from geometry. Text measured with the real fonts and renderer-identical wrapping.
- Candidates: stack, stack_split, two_column, banner, where banner uses a "priority waterfall that keeps the logo, headline, hero and CTA at full size and shrinks the least important pieces first".
- **"stack ... redistributes leftover whitespace in the proportions the designer used."**
- Scoring: absolute readable type size, hero present, canvas use, no *new* collisions, and the headline must stay the largest text.

**rsingh135/chameleon** (https://github.com/rsingh135/chameleon). This is mostly the starter plus an `engine/eval` proxy-metric package (contract, geometry, text, contrast, "text busyness"). Its constants: minimum text 1.2% of the short side, floored at 10 px; margin 3% of the short side; alignment snap 1% of canvas. These are comparable to ours.

**4rd2/ChameleonChallenge** is the starter only.

No public code was found for "smart resize JSON", "layout retargeting" or an Express resize add-on beyond fernando-ace's prototype.

**Actionable for us:** two independent teams converged on **"don't make hierarchy worse than the source"** (one capped CTA growth at 1.25x the source and penalised worsened hierarchy ratios; the other required the headline to stay the largest text). The other converged idea is **"a source-fit candidate for small aspect changes"**. Both are changes #1 and #4.

---

## 5. Typographic and proportion rules from design practice

| Flaw we see | Practice rule | Source | Suggested engine number |
|---|---|---|---|
| CTA oversized vs headline | Sizes follow a modular scale. Each hierarchy level is one or more steps below the one above. Common ratios are 1.2 (minor third), 1.25 (major third), 1.333 (perfect fourth) and 1.618 (golden). A CTA label is usually set at body or "label" size: Material 3 uses Label-Large 14 vs Headline-Large 32 and Display 45–57, so the button label is roughly 0.25–0.45x the headline. Ad guides: the CTA must be distinct "without eclipsing the main message". | Tim Brown, "More Meaningful Typography", A List Apart 2011, https://alistapart.com/article/more-meaningful-typography/ ; Material 3 type scale, https://m3.material.io/styles/typography/type-scale-tokens (from memory, not fetched this session) ; https://www.pmg.com/blog/how-to-create-killer-banner-ads | CTA label ≤ min(source ratio, 0.6) x headline size. Pill area ≤ about 0.5x the headline's ink area unless the source had a bigger ratio. On strips where both sit at their floors, shrink the pill's padding before letting it out-mass the headline. |
| Legal text blown up | Fine print is the smallest step on the scale. It stays legible but visibly subordinate, and is never larger relative to the headline than in the source. | Same modular-scale logic; our teacher rubric | Legal ≤ min(source ratio, 0.45) x headline, and never above its source ratio, but never below the absolute floor (7 px x scale). If the floor and the ratio cap conflict, keep the floor and shrink nothing else for it. |
| Logo tiny | The logo should be visible but not compete with the headline and CTA. Ad guides place it at an edge or corner. Real designer data (our Crello stats): logo area median about 0.6–2.2% of the canvas depending on size. | https://www.pmg.com/blog/how-to-create-killer-banner-ads ; `docs/real-ad-layouts.md` | Already in bisection v2 (`LOGO_MIN_LONG` = sqrt(area) ≥ 6.5% of the long side). Check it against the real median of 0.6–2.2% of canvas area. |
| Edge crowding | Consistent safe margins on all sides. Platform safe zones (Meta stories: 14% top and bottom). Real ads: median edge margins of 3–6% of the short side. | `docs/format-specs.md`, `docs/real-ad-layouts.md` | Already v2 `V2_MARGINS`. |
| Mixed alignment | One alignment axis per column (flush-left *or* centred), shared edges, few distinct x-positions. | VLT rules; Haraguchi (only visible offsets count); LayoutRectifier grid snap | Snap pass: within a stacked column, if edges differ by more than 0 and less than 4% of the column width, snap to the dominant axis. |
| Awkward wraps / widows | Avoid a lone short word on the last line; balance line lengths (ragged right shouldn't zig-zag); line length about 45–90 characters for body (much shorter in ads). | Butterick, *Practical Typography*, https://practicaltypography.com/widow-and-orphan-control.html and https://practicaltypography.com/line-length.html (from memory, not fetched this session) | Already v2 "no widow". Add: headline last line at least about 40% of the longest line's width. |
| Orphaned decorative fragments | Gestalt proximity: related things sit together; decoration belongs to the thing it frames. | General design principle; Bannersnack "grouped items move and scale together" | Already v2 "decorations kept whole / follow their unit". |
| What to drop when space is tight | Observed order in real systems and guides: description/body → badges/secondary → legal (kept only if mandatory, as fine print) → hero image → CTA → logo → headline (never). Google: 320x50 carries about a logo plus four words. | Google RDA guidance via https://www.walkertek.com/google-responsive-display-ad-guidelines (secondary) ; amanadhav's "priority waterfall" | We may not delete ids. "Drop" here means a minor-zone at the floor size, which bisection already does in this order. No change needed beyond checking that badges go before the CTA. |

Also from practice: whitespace is designed, not leftover. Gaps between groups should be larger than gaps within a group (proximity), and the same kind of gap should repeat. Today our bisection gutters are uniform (`GUTTER`), so a header group and a footer group get the same spacing as two lines of one group.

---

## Concrete, testable engine changes (env-switched, for the A/B harness)

Each item names the switch, what it does, the flaw it targets, and its overlap with things already tried (see `optimization-program.md`). Run each vs `bisect` (or `bisect2` once that is judged) on holdout, then once on the challenge set.

1. **`CHAMELEON_HIER_LOCK=1`: hierarchy lock.** Targets CTA oversized and legal blown up; hierarchy is the judge's top complaint.
   - In `text_cap`: `RATIO_SLACK` 1.3 → 1.0, `HIERARCHY_CAP` 0.9 → 0.75.
   - Role caps vs the headline: CTA 0.6, subheadline 0.75, badge 0.5, legal 0.45, each still `max(cap, floor)`.
   - In rigid and pill placement: cap the CTA unit's scale at 1.25x its source scale relative to the headline's, as the other team did.
   - Overlap: none. This is the *opposite* direction from the rulebook picker (which made things bigger and lost 0–3) and from FLEX.
2. **`CHAMELEON_RELPRESERVE=1`: relation preservation in picking.** Targets faithfulness, reading order, and "same design".
   - Compute ReLayout-style Size Rel (area order with ±20% tolerance) and Pos Rel (3x3 relative cell and reading order) between source and candidate over the main units (headline, CTA, logo, hero, subheadline).
   - Add `w·(1 − rel)` to the grader penalty, or drop candidates below 0.6 of the best candidate's score.
   - Overlap: the E1 relation features were a *learned* pairwise model (dropped). This is a hand-set faithfulness term, not learned.
3. **`CHAMELEON_ALIGN_SNAP=1`: alignment snap pass.** Targets mixed alignment.
   - After placement, for units stacked in one column, snap left, centre or right edges that differ by 1–4% of the column width to the dominant axis.
   - Positions only, never sizes; re-run gates; keep only if gates pass.
   - Overlap: S3 rule-repair lost, but S3 also enlarged text. v2 "shared alignment" may already cover most of this, so run only if `bisect2` still shows alignment complaints.
4. **`CHAMELEON_NEAR_FIT=1`: near-aspect source-fit.** Targets cells such as 1080x1080 → 1080x1350 and 1920x1080 → 1280x720.
   - If max(a_s/a_t, a_t/a_s) ≤ 1.35, where a is width/height, add a candidate that is the uniform fit of the source (letterbox the background to fill), with the content group re-centred.
   - Let the grader prefer it unless it fails a gate.
   - Overlap: this is queue item 4, "fallback policy", which was never run. The teacher preferred the shrink 34–3 where it was picked.
5. **`CHAMELEON_SRC_GAPS=1`: grouped, source-proportional gaps.** Targets cramped next to empty.
   - Replace the uniform `GUTTER` with a two-level gap: within-group gap g, between-group gap about 2g.
   - Scale g by the source's own median gap ratio.
   - Overlap: FLEX (content-aware slot sizing) lost 32–47, but it changed slot sizes. This only changes the spacing between already-sized blocks. Expect a smaller effect.
6. **`CHAMELEON_ANCHOR=1`: inherit source anchors** (Adobe patent / Figma constraints idea). Targets logo placement and balance.
   - Inside its bisection cell, a unit that touched a canvas edge or corner in the source (within 8% of it) is aligned toward the same edge or corner.
   - Overlap: v2 shared alignment (it would override shared alignment for edge-anchored logos and CTAs only).
7. **Harness, no switch: `swap_share=1.0` for the final ship decision**, with order-inconsistent verdicts counted as ties (TASTE / MT-Bench). It costs about 1.8x the judge calls on decided cells, so use it only for the final confirmation of a winner.

Not recommended now: learned layout models (LayoutDM, RALF, LayoutPrompter, AesthetiQ). Reasons:
- they solve generation, not faithful retargeting;
- they need training on Crello;
- our retrieval variant already captures the retrieval benefit;
- combining strategies showed the picker, not candidate diversity, is the bottleneck.

Revisit LayoutDM (Apache-2.0, small) only if a distilled local model becomes the finale goal.

## Source list (all links)

**Retargeting and layout**
- O'Donovan et al. 2014: https://www.dgp.toronto.edu/~donovan/layout/
- Li et al. 2020 (attribute-conditioned LayoutGAN): https://doi.org/10.1109/tvcg.2020.2999335
- Jacobs et al. 2003: https://research.adobe.com/publication/adaptive-grid-based-document-layout
- Dayama et al. 2021: https://doi.org/10.1145/3397481.3450652
- VLT 2023: https://arxiv.org/abs/2309.11635
- ReLayout 2026: https://arxiv.org/abs/2602.01046
- Graphic-Design-Bench 2026: https://arxiv.org/abs/2604.04192
- LayoutGAN: https://arxiv.org/abs/1901.06767
- LayoutTransformer: https://arxiv.org/abs/2006.14615
- LayoutDM: https://arxiv.org/abs/2303.08137
- LayoutFormer++: https://arxiv.org/abs/2208.08037
- Kikuchi 2021: https://arxiv.org/abs/2108.00871
- LACE: https://arxiv.org/abs/2402.04754
- PosterLayout: https://arxiv.org/abs/2303.15937
- LayoutPrompter: https://arxiv.org/abs/2311.06495
- RALF: https://arxiv.org/abs/2311.13602
- LayoutNUWA: https://arxiv.org/abs/2309.09506
- Desigen: https://arxiv.org/abs/2403.09093
- COLE: https://arxiv.org/abs/2311.16974
- OpenCOLE: https://arxiv.org/abs/2406.08232
- AesthetiQ: https://arxiv.org/abs/2503.00591
- LayoutRectifier: https://arxiv.org/abs/2508.11177
- Visual importance: Bylinskii et al. UIST 2017, https://doi.org/10.1145/3126594.3126653 ; Fosco et al. UIST 2020, https://doi.org/10.1145/3379337.3415825

**Judges**
- Haraguchi et al. 2024: https://arxiv.org/abs/2410.08885
- TASTE 2026: https://arxiv.org/abs/2605.20731
- Fooling LVLM judges 2025: https://arxiv.org/abs/2505.15249
- BannerAgency 2025: https://arxiv.org/abs/2503.11060
- DesignProbe 2024: https://arxiv.org/abs/2404.14801
- PosterIQ 2026: https://arxiv.org/abs/2603.24078
- UICrit 2024: https://arxiv.org/abs/2407.08850

**Products**
- Bannersnack Smart Resize: https://www.bannersnack.com/blog/smart-resize-guide
- Celtra: https://celtra.com/blog/supercharging-creative-production-with-advanced-design-automation/
- Google RDA: https://developers.google.com/google-ads/api/docs/responsive-display-ads/overview
- Adobe patent US 11,320,975: https://patents.google.com/patent/US11320975
- Canva resize help: https://www.canva.com/help/resize/
- Meta Advantage+ (secondary): https://www.jonloomer.com/advantage-plus-creative-enhancements/amp/

**Other teams**
- https://github.com/fernando-ace/Adobe-Hackathon-Chameleon-Challenge
- https://github.com/amanadhav/chameleon
- https://github.com/rsingh135/chameleon
- https://github.com/4rd2/ChameleonChallenge

**Design practice**
- Tim Brown: https://alistapart.com/article/more-meaningful-typography/
- Material 3 type scale: https://m3.material.io/styles/typography/type-scale-tokens
- Butterick: https://practicaltypography.com/
- PMG banner guide: https://www.pmg.com/blog/how-to-create-killer-banner-ads
