"""Write challenge3: 50 more REAL Crello ads, a second fresh EVALUATION set
(like challenge2: never tuned on, never looked at per ad).

    cd distil-research/chameleon && python -m dev.distill.make_challenge3 [--n 50] [--seed 0]

Same source, converter and quality gate as challenge2
(make_real_challenge.convert + make_challenge.quality_issues), from the
templates that passed the gate in challenge2's scan (challenge2.nosync/
scan.jsonl; run make_real_challenge first). Excluded: every design family
used by challenge2 AND by crello_dev, i.e. no pick shares a text of 8+
characters (the scan's `sig` hashes) with any ad of either split; at most one
pick per family here too (make_crello_dev.py excludes challenge2 the same way).

Selection is challenge2's (make_real_challenge.select): CLASS_QUOTA per source
aspect class, half text-only where the class has enough, sizes round-robin.
Classes the remaining pool can't fill (after challenge2 and crello_dev it has
no very_tall or strip family left) are topped up round-robin over the other
classes, taking a text-only ad while the set is under half text-only.

LICENCE RULE (as challenge2): Crello content is never committed. Everything
goes under training-data/corpus/challenge3.nosync/ (gitignored): the designs
(cr3_*.json), their rasters (assets/cr3_*.png), the Crello previews, the
review sheet and summary.json (aggregates only, what CHALLENGE3.md quotes).
"""

import argparse
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from dev.distill import make_real_challenge as mrc
from dev.distill.assets import CHALLENGE3_ASSETS_DIR, CHALLENGE3_FOLDERS, merged_assets_dir
from dev.distill.paths import CHALLENGE2_DIR, CHALLENGE3_DIR, CRELLO_DEV_DIR

PREFIX = "cr3"
SPLIT = "challenge3"
SEED_TAG = "chl3"  # rank salt, so the order differs from challenge2's and crello_dev's
PREVIEW_DIR = CHALLENGE3_DIR / "previews"
SHEET = CHALLENGE3_DIR / "sheet.png"
SUMMARY_FILE = CHALLENGE3_DIR / "summary.json"


def used_ids(*folders_and_patterns: tuple[Path, str]) -> set[str]:
    """The crello_ids of the ads already in the given (folder, glob) splits."""
    return {json.loads(p.read_text())["crello_id"] for folder, pat in folders_and_patterns
            for p in Path(folder).glob(pat)}


def banned_families(scan_rows: list[dict], ids: set[str]) -> set[str]:
    """Every text signature (design family) of the ads in `ids`."""
    return {s for r in scan_rows if r["crello_id"] in ids for s in r["sig"]}


def candidates(scan_rows: list[dict], ids: set[str]) -> list[dict]:
    """Gate-passing templates that share no family with `ids` (the ads of
    challenge2 and crello_dev). One-per-family is left to select()."""
    banned = banned_families(scan_rows, ids)
    return [r for r in scan_rows if r["kept"] and r["crello_id"] not in ids and not banned & set(r["sig"])]


def select(rows: list[dict], n: int, seed: int) -> list[dict]:
    """challenge2's stratified pick, then a round-robin top-up over classes
    when a class's pool is short (text-only first while under half)."""
    tag = f"{SEED_TAG}:{seed}"
    picked = mrc.select(rows, n, tag)
    if len(picked) >= n:
        return picked[:n]
    fams = set().union(*(set(r["sig"]) for r in picked)) if picked else set()
    rest = []
    for r in sorted(rows, key=lambda r: mrc._rank(r["crello_id"], tag)):
        if r in picked or fams & set(r["sig"]):
            continue
        fams |= set(r["sig"])
        rest.append(r)
    by_cls = {c: [r for r in rest if r["aspect_class"] == c] for c in mrc.CLASS_QUOTA}
    while len(picked) < n and any(by_cls.values()):
        for c in mrc.CLASS_QUOTA:
            if len(picked) >= n or not by_cls[c]:
                continue
            want_txt = sum(not r["photo"] for r in picked) < len(picked) / 2
            pool = [r for r in by_cls[c] if r["photo"] != want_txt] or by_cls[c]
            r = pool[0]
            by_cls[c].remove(r)
            picked.append(r)
    return picked


def task_id_for(k: int, r: dict) -> str:
    return f"{PREFIX}_{k:02d}_{r['W']}x{r['H']}_{'img' if r['photo'] else 'txt'}"


def write_selected(picked: list[dict]) -> list[dict]:
    """Convert the picks for real (rasters into challenge3.nosync/assets/),
    re-check each against the gate, write them and the review sheet."""
    from dev.distill.make_challenge import quality_issues
    from dev.distill.make_corpus import measurer

    F = mrc.features()
    CHALLENGE3_DIR.mkdir(parents=True, exist_ok=True)
    if CHALLENGE3_ASSETS_DIR.exists():
        shutil.rmtree(CHALLENGE3_ASSETS_DIR)
    CHALLENGE3_ASSETS_DIR.mkdir(parents=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for p in list(CHALLENGE3_DIR.glob(f"{PREFIX}_*.json")) + list(PREVIEW_DIR.glob("*.png")):
        p.unlink()
    order = sorted(picked, key=lambda r: (list(mrc.CLASS_QUOTA).index(r["aspect_class"]), not r["photo"],
                                          r["W"] / r["H"], r["crello_id"]))
    ids = {r["crello_id"]: task_id_for(k + 1, r) for k, r in enumerate(order)}
    by_group = defaultdict(set)
    for r in order:
        by_group[(r["shard"], r["group"])].add(r["crello_id"])
    records = []
    for (shard, group), want in sorted(by_group.items()):
        for row in mrc.read_group(str(mrc.SHARDS_DIR / shard), group, want):
            tid = ids[row["id"]]
            rec = mrc.convert(row, F, tid, CHALLENGE3_ASSETS_DIR, measurer())
            rec.update(split=SPLIT, notes=rec["notes"].replace("Crello ad template", "Crello ad template (challenge3)"))
            Image.open(io.BytesIO(row["preview"]["bytes"])).convert("RGB").save(PREVIEW_DIR / f"{tid}.png")
            records.append(rec)
    records.sort(key=lambda r: r["task_id"])
    with merged_assets_dir(*CHALLENGE3_FOLDERS) as assets:
        for rec in records:
            issues = quality_issues(rec, assets)
            assert not issues, (rec["task_id"], issues)
            (CHALLENGE3_DIR / f"{rec['task_id']}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        saved = mrc.PREVIEW_DIR
        mrc.PREVIEW_DIR = PREVIEW_DIR  # review_sheet reads the previews from there
        try:
            mrc.review_sheet(records, assets, path=SHEET)
        finally:
            mrc.PREVIEW_DIR = saved
    return records


def summarise(cands: list[dict], picked: list[dict], records: list[dict], excluded: int) -> dict:
    """Aggregate numbers only (no Crello content)."""
    fams, seen = [], set()
    for r in cands:
        if not seen & set(r["sig"]):
            seen |= set(r["sig"])
            fams.append(r)
    cls = lambda rs, key: dict(sorted(Counter(key(r) for r in rs).items()))
    els = [len(r["source"]["elements"]) for r in records]
    types = Counter(e["type"] for r in records for e in r["source"]["elements"])
    return {"source": "cyberagent/crello rev 5.1.0, gate-passing ad templates of challenge2's scan, "
                      "challenge2's and crello_dev's design families excluded",
            "excluded_ads": excluded, "pool_templates": len(cands), "pool_families": len(fams),
            "pool_families_by_class_photo": cls(fams, lambda r: f"{r['aspect_class']}/"
                                                               f"{'photo' if r['photo'] else 'text-only'}"),
            "selected": len(records),
            "selected_by_class": cls(records, lambda r: r["features"][0]),
            "selected_by_class_photo": cls(records, lambda r: f"{r['features'][0]}/{r['features'][1]}"),
            "selected_by_photo": cls(records, lambda r: r["features"][1]),
            "selected_sizes": cls(records, lambda r: f"{r['source']['canvas']['width']}x{r['source']['canvas']['height']}"),
            "selected_elements": {"min": min(els, default=0), "median": sorted(els)[len(els) // 2] if els else 0,
                                  "max": max(els, default=0), "by_type": dict(types)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    scan_file = CHALLENGE2_DIR / "scan.jsonl"
    if not scan_file.exists():
        print("no challenge2 scan: run python -m dev.distill.make_real_challenge first")
        return 1
    scan_rows = [json.loads(x) for x in scan_file.read_text().splitlines() if x.strip()]
    ids = used_ids((CHALLENGE2_DIR, "cr2_*.json"), (CRELLO_DEV_DIR, "crd_*.json"))
    cands = candidates(scan_rows, ids)
    picked = select(cands, args.n, args.seed)
    records = write_selected(picked)
    summary = summarise(cands, picked, records, len(ids))
    SUMMARY_FILE.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {len(records)} ads to {CHALLENGE3_DIR}; review sheet {SHEET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
