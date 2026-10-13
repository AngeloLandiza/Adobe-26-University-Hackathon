# Node 2: Resize engine

**Job:** given the source design, the target canvas and the `PriorityPlan`, generate many valid candidate layouts. Important units get space first; minor units get what's left. Node 2 doesn't pick a winner. That's node 3's job.

**Input:** `source`, `target_canvas`, `PriorityPlan`.
**Output:** `list[Candidate]` ([README §4.2](README.md#42-candidate-node-2--node-3)). The **safe fallback** candidate is always included and always valid.
**Design rule:** build valid by construction. Gate failures in node 3 should mean a bug or a genuinely impossible fit, not routine sloppiness.

```mermaid
flowchart LR
    A["aspect class<br/>of target"] --> T["templates<br/>for that class"]
    T --> V["parameter grid<br/>(mirror, margin, split, cutoff)"]
    V --> L["layers:<br/>background · decoration · content"]
    L --> S["assign units to slots<br/>in rank order"]
    S --> Z["size slots<br/>(flex-style)"]
    Z --> F["fit each unit<br/>image · text · button"]
    F --> N["normalise<br/>+ invariants"]
    N --> C["Candidate"]
```

---

## 1. Aspect classes

`r = target width / target height`. The class decides which templates run. Thresholds are tunable.

| Class | r | Real formats | Samples |
|---|---|---|---|
| `very_tall` | < 0.55 | story 1080×1920 (0.56, borderline), wide skyscraper 160×600 (0.27) | |
| `tall` | 0.55 – 0.85 | portrait 1080×1350 (0.8) | Level 1 target |
| `square` | 0.85 – 1.25 | post 1080×1080, medium rectangle 300×250 (1.2) | |
| `landscape` | 1.25 – 2.2 | YouTube thumbnail 1280×720 (1.78) | |
| `wide` | 2.2 – 5 | Facebook cover 820×312 (2.6), billboard 970×250 (3.9), LinkedIn 1584×396 (4.0) | Level 2 target |
| `strip` | > 5 | leaderboard 728×90 (8.1) | Level 3 target |

The **proportional reflow** template (§3.6) runs for every class. It wins when the target shape is close to the source shape, and the grader sinks it when it isn't.

## 2. Layers

Each candidate is built in three layers. `z_index` is never changed, so the painting order is always the source's.

1. **Background** (`background_unit_ids`): `x=0, y=0, width=W, height=H`. `fit: cover` crops as allowed. Shape backgrounds are treated the same way.
2. **Decoration** (`decoration_unit_ids`):
   - uniform scale by `s = min(W/Ws, H/Hs)`, so circles stay circles;
   - placed at the same relative offset from the source's nearest corner or edge, offset scaled by `s`;
   - clamped fully inside the canvas until open question 1 (bleed) is answered.

   Overlapping content is fine. It's intentional layering, and node 3 doesn't count it as a collision.
3. **Content**: every ranked unit, laid out by a template.

## 3. Templates

A template is a small **slot tree**: rows and columns of slots, where each slot lists the roles it accepts (in preference order), its alignment, and whether it can hold a *cluster* (several units of the same role, e.g. three badges in a row). **Every template has a `minor` zone** where units that didn't get a proper slot end up.

The sketches below show intent, not exact proportions.

### 3.1 `strip` (e.g. 728×90)

```
┌────────────────────────────────────────────────────────────────────────┐
│ [logo] │ [hero] │ HEADLINE (1–3 lines)                     │ [  CTA  ] │
│        │        │ minor: badges · subheadline · legal       │           │
└────────────────────────────────────────────────────────────────────────┘
```

Variants: hero left or right of the text; logo next to the CTA instead of first; minor line on or off (off means minor units go to the free-space pass, §4.3).

### 3.2 `wide` (e.g. 1600×400)

```
┌──────────────────────────────────────────────────────────────────┐
│ (badge)                                                          │
│   [ hero  ]    logo                                              │
│   [ image ]    HEADLINE                            [   CTA   ]   │
│   [       ]    subheadline                                       │
│                                                minor / legal     │
└──────────────────────────────────────────────────────────────────┘
```

Variants: mirrored; CTA in its own right-hand column, or under the text stack.

### 3.3 `landscape` (e.g. 1280×720)

```
┌──────────────────────────────────────┐
│ logo                         (badge) │
│                                      │
│  HEADLINE               [   hero   ] │
│  subheadline            [   image  ] │
│  [ CTA ]                [          ] │
│ minor / legal                        │
└──────────────────────────────────────┘
```

### 3.4 `square` / `tall` / `very_tall`: vertical stack

```
┌─────────────────────┐
│ (badge)        logo │
│                     │
│     [  hero   ]     │
│     [  image  ]     │
│                     │
│      HEADLINE       │
│     subheadline     │
│      [ CTA ]        │
│ minor / legal       │
└─────────────────────┘
```

Variants: hero above or below the text; `very_tall` adds more space between groups instead of growing the images past their cap.

### 3.5 Shared slot rules

- `cta` always gets a dedicated slot in every template. It's never sent to the minor zone.
- `logo` slots are small and aspect-locked; a logo never grows past ~1.5× its source scale.
- `legal` prefers a full-width single line along the bottom or another edge.
- Units of the same role with adjacent ranks (the three badges in Level 3) are **clustered** into one slot as a row, or a stack in tall formats.

### 3.6 Proportional reflow (every class)

Keep the source arrangement:

1. scale every content unit uniformly by `s = min(W/Ws, H/Hs)` around its own centre;
2. map unit centres proportionally (`cx' = cx · W/Ws`, `cy' = cy · H/Hs`);
3. fix any overlaps this creates by nudging along the axis that got more room.

This is the best answer for Level 1 (1080² → 1080×1350): same composition, with the extra height spread between logo, headline and CTA.

## 4. Assigning units to slots

1. Walk content units in rank order. Each takes the first open slot that accepts its role.
2. **Cutoff `k`** (a grid parameter): only the top `k` units may take regular slots. Everything ranked below `k` goes to the minor zone. Low `k` gives big, clean top units; high `k` shows more.
3. **Minor zone:** units are laid out in a row (or stack) in rank order and shrunk together until they fit, but never below their readability floor (§5.2).
4. **Free-space pass:** whatever still doesn't fit is placed at floor size in the largest empty rectangle that doesn't cover a higher-ranked unit. It's marked in `degraded_unit_ids`, so node 3 penalises it in proportion to its rank.
5. **Never hide.** No zero sizes, no `opacity` changes, no tucking under other elements. If a unit genuinely can't be placed at floor size, the candidate is dropped. The safe fallback still exists.

## 5. Sizing

### 5.1 Slot boxes (flex-style)

Each container shares its main axis between its children the way CSS flexbox does:

- each child has a **min**, a **preferred** and a **max** size, derived from its content (below);
- leftover space is shared out by **priority weight** `w = 1 / rank^α` (α ≈ 0.7, tunable);
- margins and gutters come from the grid (`margin` × the canvas's short side).

| Content | Preferred size along a row (columns are symmetric) |
|---|---|
| image unit | cross size × the source box's aspect ratio |
| text unit | width at which the text fits in ≤ its role's preferred line count at its target font |
| button / badge unit | label width at the target font + the source's padding ratio |

### 5.2 Text fitting

Text must fit exactly the way the official renderer will wrap it.

- **Measurement:** `engine/pipeline/textmeasure.py` calls `renderer.converter.wrap.char_widths` / `wrap_text` with the brand-kit `FontManager`. It normalises weight and style exactly like `renderer/converter/convert.py` does: `_normalize_weight` and `_weight_style_to_font_style` are mirrored, not imported, because they're private.
- **Rendered height** = `lines × font_size × line_height`. **Before relying on this, verify against cr-renderer**: check how it places the first line within the box and what happens on vertical overflow (whether it clips or spills). Do it with a quick render test on a known string.
- **Search:** binary-search `font_size` in `[floor(role), cap]` (0.5 px steps) for the largest size where:
  - the wrapped text fits the slot's height,
  - no single word is wider than the slot (the renderer never breaks words),
  - `max_lines` is respected when set.
- **cap** = source `font_size` × `max_text_growth` (default 1.5), so text never balloons.
- **floor(role):** tunable readability floors. Starting values, to be tuned on renders: headline 14 px, cta 11, subheadline 10, badge 9, legal 7. Scaled up on large canvases by `max(1, min(W, H) / 600)`.
- **Keep the size ordering.** If the source's headline font was larger than its subheadline font, the output keeps that order (lower-ranked text is clamped to ≤ 0.85× the text above it).
- **Tight box:** after fitting, the text box height shrinks to the measured block and is aligned in its slot. Width stays at the slot width so `text_align` still means something.
- `content`, `font_family`, `font_weight`, `font_style`, `line_height`, `text_align`, `style_runs` are **never changed**. `letter_spacing` (px) scales with the font-size ratio.

### 5.3 Images

The box is the largest box **with the source box's aspect ratio** that fits the slot, aligned per the slot. This keeps the ±5% aspect gate true by construction. A non-background `fit: cover` image may take the slot's own shape, since it's expected to crop.

### 5.4 Buttons, badges and other groups

- **Rigid uniform scale** of the whole unit around its bbox: `k = min(slot_w / unit_w, slot_h / unit_h)`. Member positions and sizes scale by `k`, and so do `corner_radius` and `stroke_width`.
- **Labels:** start at source `font_size × k`, then run the text-fit check inside the label's scaled box; shrink if needed.
- **Rotation** is preserved. Member offsets are scaled about the unit's origin, and the in-canvas check uses each element's **rotated** bounds (the Level 3 ribbon is at −12°).
- Optional, behind a flag: let a pill shrink in width only when its label still fits. This saves space in `strip` layouts.

## 6. Parameter grid

| Parameter | Values (initial) | Applies to |
|---|---|---|
| `mirror` | false, true | all except stacks |
| `margin` | 0.04, 0.07 (× short side) | all |
| `split` | 0.35, 0.45, 0.55 (image share of the main axis) | strip, wide, landscape |
| `cutoff k` | all, 4, 3 | strip, wide |
| `hero_first` | true, false | stacks |

At most a few dozen candidates per template, ~50–100 per task. Geometry scoring is cheap, so that's fine; `max_candidates` caps it if needed. Enumerate in a fixed order so `candidate_id`s and tie-breaks are deterministic.

## 7. Normalisation (every candidate)

1. Start from a **deep copy of each source element**. Overwrite only `x, y, width, height`, plus `font_size` and `letter_spacing` for text and `corner_radius` and `stroke_width` for shapes. Everything else is kept untouched, including unknown extra keys like `_note`.
2. `canvas = {"width": W, "height": H}` as ints.
3. Round geometry to 2 decimals.
4. Check every element's rotated bounding box is inside `[0, W] × [0, H]`. If it isn't, **drop the candidate**. Never silently squash the element to make it fit.
5. Same element ids, same count, same order as the source.

## 8. Safe fallback candidate

Always generated first, and guaranteed to pass every gate:

1. Background cover-fills the canvas.
2. Everything else scales uniformly by `s = min(W/Ws, H/Hs)` around the source centre, then is centred on the target. Text `font_size × s` (minimum 1) wraps identically, because glyph widths scale linearly with size.
3. Anything still outside (e.g. bleeding decorations) is shrunk or moved inside.

It's ugly on extreme targets (Level 3 comes out at ~7% scale) but always valid. It's the last resort, not a contender.

## 9. Worked examples (what a good result should look like)

**Level 1, 1080×1080 → 1080×1350 (`tall`).** Proportional reflow likely wins: logo stays top-centre; headline and CTA keep their column but spread over the extra 270 px; the background re-covers. The stack template is the runner-up.

**Level 2, 1080×1080 → 1600×400 (`wide`).**

```
┌──────────────────────────────────────────────────────────────────────────┐
│ (NEW FLAVOR)                                              ○ deco         │
│   ▮▮▮▮    ☀ SOLSTICE                                                     │
│   ▮▮▮▮    Chase the Light                         ( Shop the Summer Line )│
│   ▮▮▮▮    Botanical sparkling infusions, crafted for golden hour.        │
│ ○ deco                                                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Level 3, 1080×1350 → 728×90 (`strip`, cutoff 4).** Logo, bottle, headline (about 3 lines at ~11–12 px) and the CTA pill read well. Badges, subhead, ribbon, legal and the missing-asset watermark are present but small. In 728×90 there's no layout where all 17 elements are readable, so the aim is for the top 3–4 to read well and the rest to stay present and tidy. See open question 2.

## 10. Testing

- **Property tests** over every source × every `dev/targets.json` size, plus random aspect ratios:
  - every emitted candidate passes node 3's gates;
  - the safe fallback passes on every input.
- **Text-fit agreement:** for a handful of fitted texts, render and check that the inked pixel bounds sit inside the box. This guards against our measurement drifting from cr-renderer.
- **Unit tests:**
  - slot assignment (rank order, clustering, cutoff, the CTA always placed);
  - flex resolution;
  - rotated bounds;
  - normalisation copying every field.
- **Visual review:** `dev/run_matrix.py` contact sheets, every candidate or just the winners.
