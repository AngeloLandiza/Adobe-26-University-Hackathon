# challenge3: 50 more real ads from Crello (EVALUATION ONLY, unseen)

**What it is:** a second fresh set of 50 real ads (Crello / VistaCreate ad templates), converted into our design format the same way as challenge2 (see [CHALLENGE2.md](CHALLENGE2.md) for the converter and the quality gate). challenge2 has now been used to choose between engine variants, and crello_dev is a development set. challenge3 is an untouched set that confirms a final choice on ads nothing was tuned on.

**Rules:**

- The licence rule is the same as for challenge2. **Nothing that contains Crello content is committed.** That covers the designs (`cr3_*.json`), their rasters (`assets/cr3_*.png`), the Crello previews and the review sheet, all in `corpus/challenge3.nosync/`, which is gitignored. A/B judgements on it (`ab/*chl3*/judgments*.jsonl`) are gitignored too. This file quotes only aggregate counts.
- **Keep it unseen.** Don't tune on it, don't look at per-ad results, and run it only to confirm a variant that was already chosen. The picker study (`picker_study.py`) refuses it. So far it has had one smoke test of 5 cells under the default engine: 0 errors and 0 hard-gate failures. Those winners were deleted.

## How to build and use it

From `distil-research/chameleon/` (needs challenge2's scan, `challenge2.nosync/scan.jsonl`, and the Crello shards):

```
python -m dev.distill.make_challenge3                 # deterministic (--seed 0), ~1 min, one process
python -m dev.distill.ab_engine run --name base_chl3 --split challenge3 --workers 3
```

- `load_corpus("challenge3")` / `load_challenge3()` return these ads. No other split ever does.
- Each record has `split: "challenge3"`, `origin: "crello"` and `brand: "crello"`.
- The rasters resolve through `merged_assets_dir(*CHALLENGE3_FOLDERS)`. The harness merges them only for `--split challenge3`.
- 626 cells: 50 ads × 13 targets, minus the targets within 2% of an ad's own aspect.

## Method

1. **Pool.** Start from the 327 templates that passed challenge2's quality gate. Drop the 110 ads already used (50 in challenge2, 60 in crello_dev). Then drop every template that shares a **design family** with any of them. A family is a shared text of 8 or more characters (Crello stores the same design in several formats). That leaves 177 templates, or 154 distinct families.
2. **Selection.** The same as challenge2's (`make_real_challenge.select`):
   - a quota per source aspect class: square 10, landscape 10, tall 10, wide 9, very_tall 6, strip 5;
   - half text-only within a class where the class has enough;
   - canvas sizes spread round-robin;
   - one template per family;
   - ordered by a seeded hash that differs from the other two splits.
3. **Top-up.** The remaining pool has **no very_tall or strip family at all**: challenge2 and crello_dev took every one that passed the gate. Those 11 slots are filled round-robin across the other classes, taking a text-only ad while the set is under half text-only.
4. **Write.** The picks are converted again and re-checked against the gate (all pass). Each Crello preview and our render are drawn side by side on `challenge3.nosync/sheet.png`. A look at the first 12 ads on that sheet matches challenge2's conversions: same layout, colours and images, with only our two typefaces swapped in.

## Counts

| | count |
|---|---|
| templates that passed the gate (challenge2's scan) | 327 |
| ads excluded (challenge2 50 + crello_dev 60) | 110 |
| pool after the family exclusion | 177 templates, 154 families |
| selected | **50** |

What the pool allowed, in families, and what was selected:

| source aspect class | pool: photo | pool: text-only | selected: photo | selected: text-only |
|---|---|---|---|---|
| square | 39 | 10 | 5 | 8 |
| landscape | 38 | 3 | 10 | 3 |
| tall | 15 | 12 | 5 | 8 |
| wide | 32 | 5 | 7 | 4 |
| very_tall | 0 | 0 | 0 | 0 |
| strip | 0 | 0 | 0 | 0 |
| **all** | **124** | **30** | **27** | **23** |

- **Text-only share: 23 of 50 (46%).** That is close to half. Landscape is the only class short of text-only ads: it has 3, and all 3 were taken.
- **Bias compared with challenge2:** there are no very_tall or strip sources. Sources that tall or that wide are tested only through challenge2. The targets still cover every aspect class.
- **Sizes covered (24):**
  - 851x315 ×5;
  - 1080x1080, 1920x1080 and 300x250 ×4 each;
  - 1080x1920, 1200x628, 360x504, 540x810 and 940x788 ×3 each;
  - 1200x600, 1296x1728 and 336x280 ×2 each;
  - 1024x512, 1190x1683, 1200x480, 1280x720, 1500x500, 1590x400, 2560x1440, 3000x1055, 537x240, 560x315, 600x200 and 735x1102 ×1 each.
- **Elements:** 3 to 27 per ad (median 7), 457 in all: 230 raster images, 153 texts and 74 shapes.
