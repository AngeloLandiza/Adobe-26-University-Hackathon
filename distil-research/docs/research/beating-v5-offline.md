# Beating v5 Offline: A Ranked Experiment Plan for the CHAMELEON Resize Engine

**Recommendation: stop asking any model to output final pixel coordinates.** Use a split I call "structure from a model, geometry from a solver". A model or v5 decides the discrete plan: which region each element goes in, reading order, what is dropped or demoted, and which edge each group anchors to. A small constraint/least-squares solver then computes exact boxes, with alignment, equal gaps, hierarchy and readability floors as hard or strongly weighted constraints. Every remaining v5 flaw on your critique list (lopsided whitespace, dead bands, groups drifting, edge crowding) is a geometry-distribution failure that solvers handle directly. The precision loss you saw with Q4 GGUF goes away once the model only emits a handful of discrete tokens.

## TL;DR
- **Rank 1: a constraint-solver "geometry pass" behind v5 and every other candidate source.** Use OR-Tools CP-SAT or a linear program with equal-gap, shared-axis, group-rigidity and hierarchy constraints. It is the cheapest and most likely clear win. Prior work (GRIDS MILP, ORCSolver, LayoutRectifier) shows layouts of this size solve at interactive speed on CPU: LayoutRectifier reports 0.65 s for 22 elements. The main risk is repeating your failed flexbox sizer, so the solver must preserve source relations as constraints, not just cut dead space.
- **Rank 2: distil the Sonnet/Opus "designer" into a tiny *plan* predictor, not a coordinate generator.** Use a 0.5–1.7B LLM at Q8, or even a gradient-boosted per-element classifier. It predicts region, order, keep/drop and anchor, and the rank-1 solver produces the coordinates. This targets most of the 13–1 LLM ceiling without the Q4 precision problem, at an estimated 1–5 s on CPU (unmeasured).
- **Rank 3: a ~13M-parameter flow-matching layout model (LayoutFlow, MIT licence) used as a diverse candidate generator.** Condition it on the source layout plus target aspect, train it on your LLM layouts plus augmented Crello, then pass its outputs through the solver, the gates and the picker. It works in continuous coordinates and runs in milliseconds (LayoutFlow reports 1.75 ms per layout). The risk is data: 2,659 paired examples is small, so it needs retrieval-augmentation and synthetic pairs.

## Key Findings

**1. Optimization-based layout has a long, directly relevant track record, including retargeting.**
- **O'Donovan, Agarwala and Hertzmann (Adobe/Toronto, IEEE TVCG 2014).** An energy model built from design principles: alignment, importance, segmentation and balance. Its weights were learned by Nonlinear Inverse Optimization from a small set of examples, and it was applied explicitly to "retargeting designs to new sizes." Crowdsourced evaluation found it performed "as well as, or better than, novice designers."\[1\]\[2\] This is the closest published precedent to your task.
- **GRIDS (Dayama et al., CHI 2020).** A MILP that guarantees packing, alignment, grouping and preferential placement.\[3\] The authors report it solves "within few seconds using commodity hardware" and handles 10+ elements, whereas a black-box optimizer it compares against "throttled at problem sizes of 10 or more elements." Repo: aalto-ui/GRIDS, MIT licence, but it depends on Gurobi (commercial), so you would port the formulation to CP-SAT or HiGHS.\[3\]\[4\]\[5\]\[6\]
- **ORCSolver (Jiang et al., CHI 2020).** It adds OR-constraints so one specification can "turn horizontal arrangements into vertical ones," and drops optional elements as space shrinks. It runs at "near-interactive rates" using branch-and-bound, then Cassowary or OSQP once the OR-choices are fixed.\[7\]\[8\] This is the exact pattern you need for square → 728×90: discrete choice first, continuous solve second.
- **LayoutRectifier (Shen, Shamir and Igarashi, 2025).** It alternates a discrete grid "search-and-snap" with continuous Adam updates on an overlap/containment energy.\[9\] It reports 0.22 s for 7 elements, 0.65 s for 22 elements and 0.71 s for a 12-element magazine layout. Its simulated-annealing baseline took about 30 s and left residual misalignment.\[10\]
  - In a 50-worker-per-pair MTurk study, its rectified layouts were preferred for visual appeal 71–94% of the time over learned refiners and LayoutPrompter.\[10\]
  - No public code was found, so you would re-implement it from the paper.
- **CLG-LO (Kikuchi et al., MM 2021).** It optimizes in a GAN's latent space: 13.6 s mean with Adam and 1.45 s with CMA-ES.\[11\] Treat this as evidence that latent and black-box search is slower than direct geometric solving. Its code (ktrk115/const_layout) is AGPL-3.0, which matters for the requirement that weights and code be public.\[12\]\[13\]

**2. Small layout-specific models are genuinely small and fast, but they were built for generation, not retargeting.**
- **LayoutFlow (ECCV 2024, MIT).** About 12.7M parameters; Table 8 reports 1.75 ms per layout at 50 Euler steps versus 16.6 ms for LayoutDM and 1,600 ms for the 86M-parameter LayoutDiffusion.\[14\]\[15\]\[16\]
  - Quality plateaus at around 40 steps, and 8 function evaluations already bring FID to about 13.\[14\]
  - The paper itself is inconsistent: about 15M parameters in the text versus 12.7M in the table, and 100 steps versus 50. Hardware is unstated.\[14\]
- **LayoutDM (CVPR 2023, Apache-2.0).** 12.4M parameters.\[17\]\[18\] Note that the repo was reportedly archived in July 2026.\[19\]
- **RALF (CVPR 2024 Oral, Apache-2.0).** 42.8M parameters total, but the layout decoder is only 6.59M; 0.305 s total on PKU. Retrieval augmentation let a #Dim=256 decoder beat a #Dim=768 non-retrieval baseline.\[20\]\[21\]\[22\]\[23\] This is the strongest published evidence that retrieval substitutes for data when data is scarce, which is your situation.
- **LACE (ICLR 2024, MIT).** Continuous diffusion with differentiable alignment and overlap losses, plus a constraint-optimization post-process. With post-processing, alignment drops from 0.043 to 0.010 on PubLayNet with no FID penalty.\[24\]\[25\]\[26\] That supports "continuous model + geometric post-pass".
- **What this means for you:** continuous-coordinate models like LayoutFlow and LACE avoid the token-precision issue entirely, and all of these models are CPU-trivial at this size. None was trained for source→target retargeting, though; you would train on your own pairs.

**3. LLM numeric precision: the evidence favours discretizing the plan, not the coordinates.**
- LayoutNUWA keeps coordinates to one decimal and uses the LLM's native number tokens. In its own ablation, structured "code" output beats raw numeric tuning on FID, but the paper does not isolate quantization.\[27\]\[28\]
- llama.cpp's own `quantize --help` table, quoted in GitHub Discussion #2094, lists "Q8_0 : 6.70G, +0.0004 ppl @ 7B - very large, extremely low quality loss" and "Q4_K_M : 3.80G, +0.0535 ppl @ 7B - medium, balanced quality - *recommended*" (absolute perplexity increases on a 7B model). These figures measure general text perplexity, not coordinate accuracy. Rababah et al., "The Illusion of Equivalency" (arXiv 2607.08734, July 2026), testing 8-bit down to 2-bit, warn that "perplexity is not a reliable proxy for preserved decisions under quantization."
- **Unverified inference:** going from Q4 to Q8 will likely shrink but not eliminate your coordinate drift, and it will slow CPU inference because Q8 roughly doubles memory traffic. The more robust fix is to have the model emit only discrete decisions.

**4. What VLM judges can and cannot see.**
- Haraguchi et al. (2024, CyberAgent) found GPT-4o evaluations of alignment, overlap and whitespace "have a reasonably good correlation with human annotation," but that GPTs "cannot distinguish small details," whereas vector-based heuristics can.\[29\]
- **Implication 1:** micro-polish (a few pixels of alignment) will be largely invisible to the judge.
- **Implication 2:** structural choices are what flip verdicts: what is dropped, where the hero sits, whether bands are dead.
- **Implication 3:** your learned-picker "headroom" being noise is consistent with this. The picker was optimizing differences below the judge's resolution.
- Position bias in pairwise LLM judging is real and concentrated on close pairs (Shi et al., IJCNLP 2025).\[30\]\[31\] Your both-orders protocol is correct; keep it and count order-inconsistent verdicts as ties.
- **Uni-Layout (Lu et al., CASIA & JD.COM, ACM MM 2025)** introduced Layout-HF100k, which the paper calls "the first large-scale human feedback dataset with 100,000 expertly annotated layouts", plus a human-mimicking layout evaluator. The authors state "Our code is publicly available at https://github.com/JD-GenX/Uni-Layout"; its licence is unverified. **AesEval-Bench (An et al., arXiv 2603.01083, March 2026)** spans "four dimensions, twelve indicators, and three fully quantifiable tasks" (indicators include balance, layering, whitespace, alignment, hierarchy and legibility), but its code and data are only promised for release at github.com/arctanxarc/AesEval-Bench. Both are candidate sources for a local critic; their size for offline CPU use is unverified.

**5. Data.**
- **Crello (cyberagent/crello, Hugging Face):** CDLA-Permissive-2.0, but users "must agree to the VistaCreate License Agreements," and the original files are not redistributed. The Hugging Face dataset card lists 23.3k rows: train 19.5k, validation 1.85k, test 1.97k.
- It matches the organizers' renderer (cr-renderer) and schema family, so it is the best augmentation source. It has no paired multi-size versions of the same design, so retargeting pairs must be synthesized.

## Details: The Top 5 Approaches

### #1: Solver geometry pass ("v5-structure → CP-SAT/LP geometry") — Effort 3–4 days · Risk medium · CPU: estimated 0.1–1 s per candidate
- **Key sources:**
  - GRIDS (arXiv 2001.02921; aalto-ui/GRIDS, MIT, needs Gurobi)\[6\]\[32\]
  - ORCSolver (arXiv 2002.09925)\[33\]
  - LayoutRectifier (arXiv 2508.11177; no code)\[10\]\[34\]
  - O'Donovan et al. 2014 (dgp.toronto.edu/~donovan/layout)\[2\]
  - OR-Tools CP-SAT (Apache-2.0)
  - The CP-SAT Primer: it packed a dense rectangle instance "in less than 0.3s", but warns that the `no_overlap_2d` propagator is "poor" on hard instances, according to the OR-Tools maintainer.\[35\]\[36\]
- **Formulation.** Take the region tree v5 already produces from bisection and anchors. Fix the topology: left-of/above relations, group membership and region assignment. Then solve a linear program (or a CP-SAT program on a 4 px integer grid) over x, y, w, h and font size.
  - **Hard constraints:**
    - in-canvas with a safe margin
    - image aspect ratio fixed (w = r·h)
    - text box ≥ measured text extent at the chosen font size (use piecewise-linear font-size levels)
    - hierarchy: headline font ≥ k·subhead ≥ k·CTA, i.e. your hierarchy lock
    - group rigidity: a container and its contents move as one affine unit, i.e. your layer preservation
    - no overlap between non-layered siblings
  - **Soft objective (linear/L1):**
    - variance of gaps between consecutive siblings, which directly kills lopsided whitespace and dead bands
    - deviation from shared alignment axes inherited from the source, which kills group drift
    - distance from the source anchor, as in anchored retargeting
    - fill ratio of the hero element
    - edge-margin equality, which kills edge crowding
    - a penalty when text sits on a busy image (precompute an image-busyness mask)
- **Why it is new versus what failed:**
  - The flexbox sizer only *removed* dead space. This approach *redistributes* it under preserved relations and the hierarchy lock.
  - Relation-preservation failed as a *scoring* term. Here relations are *constraints* inside a solver, so the layout is constructed to satisfy them rather than selected after the fact.
  - The principle: weighted scoring of fixed candidates cannot create the layout you want; a solver can.
- **Plugging in.** It is a post-pass applied to every candidate: v5 bisection, v5 anchored, the near-fit layout, and later the outputs of #2 and #3. Gates are re-run after solving. The existing picker chooses as before.
- **Weights.** Start hand-set with a few orders of magnitude between tiers: hard > hierarchy > gaps/axes > source fidelity > fill. Do not learn them from pairwise judgments yet; your picker experience shows that overfits the noise. O'Donovan's inverse optimization learned weights from *good examples*, not preferences.\[37\] The analogous move is to fit weights so the solver reproduces Sonnet layouts given Sonnet's topology. That is a regression on 2,659 examples, not a noisy preference fit.
- **Minimal one-day experiment:**
  - Implement the LP with scipy `linprog`/HiGHS or OR-Tools on wide formats only: 728×90, 970×250, 320×50, 1600×400. Those are your weakest formats and the most constrained.
  - Constraints: equal gaps, shared vertical-centre axis, group rigidity, margins.
  - Compare v5 against v5+solver on 40–50 Crello ads in both orders.
  - Success bar: at least 27 of 40 decisive pairs, which is p≈0.04 by a two-sided binomial test (my calculation).
- **Failure modes:**
  - Infeasibility on extreme ratios. Mitigate with ORCSolver-style OR-choices (drop or demote optional elements), with infeasibility triggering a fallback to v5.
  - Equal gaps can look mechanical. Allow group-internal gaps smaller than inter-group gaps; that is proximity grouping.
  - CP-SAT `no_overlap_2d` can be slow.\[36\] Fixing the topology first turns no-overlap into linear precedence constraints, so prefer the LP.

### #2: Distilled *plan* model + solver — Effort 4–6 days · Risk medium · CPU: estimated 1–5 s (0.5–1.7B at Q8) or under 50 ms (tree model), both unmeasured
- **Key sources:**
  - LayoutPrompter (NeurIPS 2023): retrieval of exemplars plus serialization plus a ranker, training-free.\[38\]
  - LayoutNUWA (ICLR 2024): structured code output beats raw numeric output.\[28\]
  - ORCSolver: discrete choice → continuous solve.
  - Qwen small models: you already have the training pipeline.
- **Design.** From your 2,659 Sonnet layouts, extract per-element *plan labels* automatically:
  - target region cell on a coarse grid (e.g. 3×3, or 1×5 for strips)
  - reading-order index
  - keep / shrink / drop
  - horizontal and vertical anchor (left/centre/right, top/middle/bottom)
  - relative-scale bucket
  - group-together flags
  
  Train either (a) Qwen 0.5–1.7B with LoRA to emit about 20 short tokens per element under a JSON-schema grammar (llama.cpp GBNF), or (b) a LightGBM per-element classifier on hand features: type, area share, text length, font size rank, colour contrast, position, target aspect bucket. The solver from #1 produces exact geometry from the plan.
- **Why this should work.** Your evidence says the Opus/Sonnet ceiling is a *structure* advantage, and that the 4B model fails at *precision*. This keeps the first and removes the second. Discrete labels are also far more learnable from 2.6k examples than continuous boxes. Extend the dataset cheaply by asking Opus for plans on more Crello ads: plans are about 10× shorter than full layouts.
- **Overlap.** Per-ad retrieval of LLM floor plans "adds nothing on top of bisection". That retrieval returned *whole plans from other ads*. This approach predicts a plan *for this ad*, conditioned on its elements, and the solver realizes it. Archetype clustering failed because text got too small; here readability floors are hard solver constraints.
- **Minimal one-day experiment (oracle test first):**
  - Feed *Sonnet's own plans* (extracted labels) into the #1 solver.
  - Compare against v5 and against raw Sonnet coordinates on 40 ads.
  - If oracle-plan + solver does not beat v5 clearly, no plan model will. Kill or redesign before training.
  - If it ties raw Sonnet, train the plan model next.
- **Failure modes:**
  - Label extraction from free-form Sonnet boxes is ambiguous (overlaps, layered decorations).
  - The plan model collapses to the majority archetype; check prediction entropy per target format.

### #3: LayoutFlow-style conditional generator as a candidate source — Effort 6–9 days · Risk high · CPU: milliseconds per sample (paper reports 1.75 ms; hardware unstated)
- **Key sources:**
  - LayoutFlow (arXiv 2403.18187; JulianGuerreiro/LayoutFlow, MIT)\[16\]\[39\]
  - LACE (arXiv 2402.04754; puar-playground/LACE, MIT)\[24\]\[40\]
  - RALF (CyberAgentAILab/RALF, Apache-2.0)\[22\]
  - LayoutDM (Apache-2.0)\[17\]
- **Design.**
  - Tokens are elements, with features: type, source box normalized, text length, font size, group id, target aspect.
  - The model is a 4-layer, 512-dimension transformer velocity field.\[14\] Training target: the Sonnet/Opus target boxes, normalized to the target canvas.
  - Add LACE-style alignment and overlap losses at low noise levels.\[24\]\[25\]
  - Add RALF-style retrieval: the k nearest source designs' target layouts as extra tokens.\[23\]
  - Sample 16–64 candidates per request, pass them through the solver (#1), then gates, then the picker.
- **Augmentation:**
  - Self-retargeting pairs from Crello: scale and jitter the source, flip horizontally when no text-direction asymmetry, permute z-order of non-overlapping siblings.
  - Treat each Sonnet layout as a new *source* for a different target (cycle pairs).
  - Element dropout to simulate variable counts.
- **Minimal one-day experiment.** It is not feasible in one day end-to-end. The one-day probe is to train on the 2,659 pairs for a few GPU-hours, generate 16 samples per ad and solver-rectify them. Then measure only the *oracle-best-of-16* against v5, with Claude picking the best sample. If even the best sample does not beat v5, stop.
- **Failure modes:**
  - Too little data, so mode collapse onto "centered stack".
  - Generated layouts are diverse but not faithful to the source ("unmistakably the same design").
  - Picker noise wastes the diversity. You have already seen this.

### #4: Evaluator matching and a cheap local critic — Effort 2–3 days · Risk low · CPU: under 0.2 s
- **Key sources:**
  - Haraguchi et al. 2024 (arXiv 2410.08885)\[41\]
  - Shi et al., position bias (arXiv 2406.07791)\[42\]
  - Uni-Layout / Layout-HF100k (arXiv 2508.02374; JD-GenX/Uni-Layout, licence unverified)\[43\]
  - AesEval-Bench (arXiv 2603.01083)\[44\]
- **Actions:**
  - (a) Keep both-orders judging and count order-flips as ties.
  - (b) Shift engineering toward *visible* structural decisions, since GPT-class judges miss small details.\[29\]
  - (c) Make the picker a *veto* model, not a ranker. Train a tiny classifier only on cases with a large judged gap (e.g. ship-readiness Δ ≥ 1.5) to flag visible defects: text over a busy image, dead band larger than X% of the canvas, hero element not the largest, logo below a size floor. That matches the Shi et al. finding that information lives in far pairs.\[30\]
  - (d) Prompt the judge-proxy with the official four criteria verbatim. You already do this.
- **Overlap.** The learned pairwise picker and the learned grader weights both failed. This is different: a binary defect detector trained only on unambiguous pairs, used as a gate.
- **Failure mode:** too few far-gap pairs to train on. Mitigate by synthesizing defects (shift text onto an image, shrink the logo).

### #5: Make the distilled LLM viable (only if #2 needs an LLM) — Effort 2–3 days · Risk medium · CPU: 5–25 s, unmeasured
- **Actions:**
  - Q8_0 or Q6_K instead of Q4_K_M. llama.cpp's own quantize table lists +0.0004 perplexity at 7B for Q8_0 versus +0.0535 for Q4_K_M.
  - Grammar-constrained decoding (llama.cpp GBNF / JSON schema) to remove invalid outputs.
  - Integer 0–1000 or coarse grid coordinates on the *target* canvas.
  - Relative offsets inside groups.
  - Greedy decoding.
- **Reasoning.** Outputting fewer, coarser numbers matters more than bit-width. That is why #2 subsumes this.
- **Minimal experiment.** Re-quantize the existing Qwen3-4B LoRA to Q8_0, run the same 22-ad A/B (6–16 at Q4), and log seconds per layout. If it moves to about 11–11 at under 40 s, quantization was the main culprit; otherwise the model is.
- **Failure mode:** Q8 is about 2× slower on memory-bound CPUs,\[45\] and the model-size cap is unknown.

## Overlap with What You Have Tried vs Genuinely New

| Proposal | Closest prior attempt | What's new |
|---|---|---|
| #1 Solver geometry | Flexbox sizer (lost), relation scoring (neutral) | Relations, hierarchy and groups are *constraints*; whitespace is redistributed, not deleted |
| #2 Plan model | LLM floor-plan retrieval (no gain), archetypes (text too small), 4B distill (precision) | Per-ad discrete plan + solver; readability floors are hard constraints |
| #3 Flow model | None (no small non-LLM generator tried) | Continuous coordinates, multi-sample, solver-rectified |
| #4 Critic | Learned picker, learned weights | Veto-only, trained on far-gap pairs and synthetic defects |
| #5 LLM quantization | Q4 GGUF (6–16) | Q8 + grammar + coarse grid, ideally plan-only |

## Recommendations (Order of Execution)
1. **Day 1:** run the #1 LP solver on wide formats only, and A/B it against v5. In parallel, run the #2 *oracle-plan* test (Sonnet plans → solver). Together they tell you whether "plan + solver" can reach the LLM ceiling.
2. **Days 2–4:** extend the solver to all formats, add ORCSolver-style OR-choices (drop/demote) for extreme ratios, and fit tier weights by inverse optimization against Sonnet layouts.
3. **Days 4–8:** if the oracle passes, train the plan model. Try LightGBM first (fast, no size-cap risk), then Qwen-0.5–1.7B at Q8 with a grammar.
4. **Background:** use #4's veto critic as a gate. Start #3 only after #1 and #2 have landed.
5. **Licence hygiene:**
   - Avoid AGPL code (const_layout) in the shipped image.\[13\]
   - Gurobi is not usable; use OR-Tools or HiGHS.\[3\]
   - Crello-derived training requires accepting the VistaCreate terms.\[46\]

## Caveats
- **No source measures resize quality on your task.** All gains are hypotheses to be tested by A/B; I have not invented expected win counts.
- **Latency figures come from the papers' own, often unstated, hardware.** LayoutRectifier's 0.22–0.71 s is for its Adam-based method, not an LP.\[10\] CP-SAT times for your exact formulation are unmeasured.
- **LayoutFlow's paper is internally inconsistent** on parameter count (about 15M vs 12.7M) and steps (100 vs 50).\[14\]
- **Quantization perplexity figures come from secondary blog sources**, measure text perplexity rather than coordinates, and should be treated as directional.
- **Your judge is Claude, not the hidden organizer VLM.** Findings about GPT-4o's insensitivity to small details are suggestive, not guaranteed to transfer.

## Sources

1. [Peter O'Donovan](https://www.dgp.toronto.edu/~donovan/)
2. [Learning Layouts for Single-Page Graphic Designs](https://www.dgp.toronto.edu/~donovan/layout/)
3. [GRIDS: Interactive Layout Design with Integer Programming](https://userinterfaces.aalto.fi/grids/)
4. [GRIDS: Interactive Layout Design with Integer Programming](https://research.aalto.fi/files/40720569/CHI2020_Dayama_GRIDS.pdf)
5. [GRIDS: Interactive Layout Design with Integer Programming](https://arxiv.org/pdf/2001.02921)
6. [GitHub - aalto-ui/GRIDS: Implementation of the MILP-based grid-layout generation algorithm](https://github.com/aalto-ui/GRIDS)
7. [ORCSolver: An Efficient Solver for Adaptive GUI Layout with OR-Constraints](https://dl.acm.org/doi/fullHtml/10.1145/3313831.3376610)
8. [An Efficient Solver for Adaptive GUI Layout with OR-Constraints](https://purehost.bath.ac.uk/ws/files/203367736/ORCSolver_CHI2020.pdf)
9. [Content-aware generative modeling of graphic design layouts](https://www.researchgate.net/publication/334434774_Content-aware_generative_modeling_of_graphic_design_layouts)
10. <https://arxiv.org/pdf/2508.11177>
11. <https://arxiv.org/pdf/2108.00871>
12. [Constrained Graphic Layout Generation via Latent Optimization Kotaro Kikuchi](https://dl.acm.org/doi/pdf/10.1145/3474085.3475497)
13. [GitHub - ktrk115/const\_layout: Official implementation of the MM'21 paper "Constrained Graphic Layout Generation via Latent Optimization" (LayoutGAN++, CLG-LO, and Layout evaluation)](https://github.com/ktrk115/const_layout)
14. [LayoutFlow: Flow Matching for Layout Generation](https://arxiv.org/pdf/2403.18187)
15. [LayoutFlow: Flow Matching for Layout Generation](https://mlanthology.org/eccv/2024/guerreiro2024eccv-layoutflow/)
16. [GitHub - JulianGuerreiro/LayoutFlow: LayoutFlow: Flow Matching for Layout Generation \[Andrade Guerreiro et al., ECCV 2024\] · GitHub](https://github.com/JulianGuerreiro/LayoutFlow)
17. [GitHub - CyberAgentAILab/layout-dm: \[CVPR 2023\] LayoutDM: Discrete Diffusion Model for Controllable Layout Generation · GitHub](https://github.com/CyberAgentAILab/layout-dm)
18. <https://arxiv.org/pdf/2303.08137>
19. [Actions · CyberAgentAILab/layout-dm · GitHub](https://github.com/CyberAgentAILab/layout-dm/actions)
20. [Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation](https://arxiv.org/html/2311.13602v4.pdf)
21. [\[2311.13602\] Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation](https://ar5iv.labs.arxiv.org/html/2311.13602)
22. [GitHub - CyberAgentAILab/RALF: \[CVPR 2024 Oral\] Official repository for RALF: Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation · GitHub](https://github.com/CyberAgentAILab/RALF)
23. [Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation](https://arxiv.org/pdf/2311.13602)
24. [Towards Aligned Layout Generation via Diffusion Model with Aesthetic Constraints](https://arxiv.org/pdf/2402.04754)
25. [Towards Aligned Layout Generation via Diffusion Model with Aesthetic Constraints \[Quick Review\]](https://liner.com/review/towards-aligned-layout-generation-via-diffusion-model-with-aesthetic-constraints)
26. [LACE/LICENSE at main · puar-playground/LACE · GitHub](https://github.com/puar-playground/LACE/blob/main/LICENSE)
27. [\*Preprint LAYOUTNUWA: REVEALING THE HIDDEN LAYOUT](https://arxiv.org/pdf/2309.09506v2)
28. [\[2309.09506\] LayoutNUWA: Revealing the Hidden Layout Expertise of Large Language Models](https://ar5iv.labs.arxiv.org/html/2309.09506)
29. [Can GPTs Evaluate Graphic Design Based on Design Principles?](https://dl.acm.org/doi/10.1145/3681758.3698010)
30. [Pair Difficulty Matters: Rethinking Pairwise LLM-as-a-JudgeEvaluation and Consistency](https://arxiv.org/html/2609.37577v1)
31. [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge - ACL Anthology](https://aclanthology.org/2025.ijcnlp-long.18/)
32. [GRIDS: Interactive Layout Design with Integer Programming](https://kashyaptodi.com/grids/)
33. [\[2002.09925\] ORCSolver: An Efficient Solver for Adaptive GUI Layout with OR-Constraints](https://arxiv.org/abs/2002.09925)
34. [Computer Science Aug 2025](https://arxiv.org/list/cs/2025-08?skip=5725&show=50)
35. [Advanced Modeling - The CP-SAT Primer - Using and Understanding Google OR-Tools' CP-SAT Solver](https://d-krupke.github.io/cpsat-primer/advanced_modelling.html)
36. [Performance of packing problems · google/or-tools · Discussion #3177 · GitHub](https://github.com/google/or-tools/discussions/3177)
37. [Figure 7 from Learning Layouts for Single-PageGraphic Designs](https://www.semanticscholar.org/paper/Learning-Layouts-for-Single-PageGraphic-Designs-O'Donovan-Agarwala/0042e77533923466961fcb459ac5e8b950d451e2/figure/6)
38. [\[2311.06495\] LayoutPrompter: Awaken the Design Ability of Large Language Models](https://arxiv.org/abs/2311.06495)
39. [LayoutFlow: Flow Matching for Layout Generation (arXiv:2403.18187)](https://paperswithcode.co/paper/2403.18187)
40. [GitHub - puar-playground/LACE: Continuous diffusion for layout generation](https://github.com/puar-playground/LACE)
41. [Can GPTs Evaluate Graphic Design Based on Design Principles? Daichi Haraguchi](https://arxiv.org/pdf/2410.08885)
42. [\[2406.07791\] Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://arxiv.org/abs/2406.07791)
43. [Uni-Layout: Integrating Human Feedback in Unified Layout Generation and Evaluation](https://arxiv.org/pdf/2508.02374)
44. [Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](https://arxiv.org/html/2603.01083v1)
45. [2026 LLM Quantization: Q4\_K\_M vs Q8\_0 vs FP16](https://www.kunalganglani.com/blog/llm-quantization-levels-q4-q8-fp16)
46. [cyberagent/crello · Datasets at Hugging Face](https://huggingface.co/datasets/cyberagent/crello)
