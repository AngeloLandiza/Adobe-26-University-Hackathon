"""Write crello_dev: ~60 more REAL Crello ads, for DEVELOPMENT (tuning and
looking at per-ad results is allowed here, unlike challenge2).

    cd distil-research/chameleon && python -m dev.distill.make_crello_dev [--n 60] [--workers 3] [--seed 1]

Same source and converter as challenge2 (make_real_challenge.convert and its
quality gate), from the templates that passed the gate in challenge2's scan
(challenge2.nosync/scan.jsonl; run make_real_challenge first), excluding
every design family challenge2 uses: no pick shares a text of 8+ characters
(the scan's `sig` hashes) with any challenge2 ad, and at most one pick per
family here too.

Selection prefers layered ads, since layer handling is what this split is
for: each candidate is converted once and its layers profiled
(layout/layer_groups.profile). "Layered" = at least one container group (copy
on a card, band, label, photo...) or an image in the full-bleed background
layer. Per source aspect class (challenge2's quotas scaled to n), about
LAYERED_SHARE are layered and the rest simple, so the split doesn't overfit
to layered ads.

LICENCE RULE (as challenge2): Crello content is never committed. Everything
goes under training-data/corpus/crello_dev.nosync/ (gitignored): the designs
(crd_*.json), their rasters (assets/crd_*.png), Crello previews, profile.jsonl
(per candidate: layer counts, no content) and summary.json (aggregates only).
"""

import argparse
import hashlib
import io
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from dev.distill import make_real_challenge as mrc
from dev.distill.assets import CRELLO_DEV_ASSETS_DIR, CRELLO_DEV_FOLDERS, merged_assets_dir
from dev.distill.paths import CHALLENGE2_DIR, CRELLO_DEV_DIR

PREFIX = "crd"
SPLIT = "crello_dev"
LAYERED_SHARE = 2 / 3
PREVIEW_DIR = CRELLO_DEV_DIR / "previews"
PROFILE_FILE = CRELLO_DEV_DIR / "profile.jsonl"
SUMMARY_FILE = CRELLO_DEV_DIR / "summary.json"


def _rank(crello_id: str, seed: int) -> str:
    return hashlib.sha1(f"dev:{seed}:{crello_id}".encode()).hexdigest()


def challenge2_families(scan_rows: list[dict], chl2_ids: set[str]) -> set[str]:
    """Every text signature of a challenge2 ad (its family)."""
    return {s for r in scan_rows if r["crello_id"] in chl2_ids for s in r["sig"]}


def pool(scan_rows: list[dict], chl2_ids: set[str], seed: int) -> list[dict]:
    """Gate-passing templates outside challenge2's families, one per family (seeded)."""
    banned = challenge2_families(scan_rows, chl2_ids)
    seen: set[str] = set()
    out = []
    for r in sorted((r for r in scan_rows if r["kept"]), key=lambda r: _rank(r["crello_id"], seed)):
        if r["crello_id"] in chl2_ids or banned & set(r["sig"]) or seen & set(r["sig"]):
            continue
        seen |= set(r["sig"])
        out.append(r)
    return out


def is_layered(prof: dict) -> bool:
    return prof["groups"] > 0 or prof["background_image"]


def _profile_group(job: tuple[str, int, list[str]]) -> list[dict]:
    """Convert the wanted templates of one row group into a scratch folder and profile their layers."""
    from dev.distill.make_corpus import measurer
    from engine.pipeline.layout.layer_groups import profile
    from engine.pipeline.priority import heuristic_plan

    shard, group, ids = job
    F = mrc.features()
    out = []
    with mrc.scratch_assets() as folder:
        for row in mrc.read_group(shard, group, set(ids)):
            try:
                rec = mrc.convert(row, F, "crdscan", folder, measurer())
                prof = profile(rec["source"], heuristic_plan(rec["source"], folder))
                out.append({"crello_id": row["id"], **prof})
            except Exception as exc:  # converted fine in the scan; skip if not now
                out.append({"crello_id": row["id"], "error": f"{type(exc).__name__}"})
            for p in folder.glob("*"):
                p.unlink()
    return out


def profile_pool(rows: list[dict], workers: int) -> dict[str, dict]:
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["shard"], r["group"])].append(r["crello_id"])
    jobs = [(str(mrc.SHARDS_DIR / s), g, ids) for (s, g), ids in sorted(by_group.items())]
    out = {}
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 3))) as ex:
        for part in ex.map(_profile_group, jobs):
            for p in part:
                out[p["crello_id"]] = p
    return out


def select(rows: list[dict], profiles: dict[str, dict], n: int, seed: int) -> list[dict]:
    """Per aspect class (challenge2's quota scaled to n), LAYERED_SHARE layered
    and the rest simple; a class short of one kind is topped up with the other,
    and a class short of ads overall is topped up from the biggest pools."""
    ok = [r for r in rows if "error" not in profiles.get(r["crello_id"], {"error": 1})]
    total = sum(mrc.CLASS_QUOTA.values())
    quota = {c: max(1, round(q * n / total)) for c, q in mrc.CLASS_QUOTA.items()}
    picked, rest = [], []
    for cls, q in quota.items():
        pools = {lay: sorted((r for r in ok if r["aspect_class"] == cls and is_layered(profiles[r["crello_id"]]) == lay),
                             key=lambda r: _rank(r["crello_id"], seed)) for lay in (True, False)}
        want_l = min(round(q * LAYERED_SHARE), len(pools[True]))
        want_s = min(q - want_l, len(pools[False]))
        want_l = min(q - want_s, len(pools[True]))
        picked += pools[True][:want_l] + pools[False][:want_s]
        rest += pools[True][want_l:] + pools[False][want_s:]
    rest.sort(key=lambda r: _rank(r["crello_id"], seed))
    picked += rest[:max(0, n - len(picked))]
    return picked


def task_id_for(k: int, r: dict) -> str:
    return f"{PREFIX}_{k:02d}_{r['W']}x{r['H']}_{'img' if r['photo'] else 'txt'}"


def write_selected(picked: list[dict]) -> list[dict]:
    from dev.distill.make_challenge import quality_issues
    from dev.distill.make_corpus import measurer

    F = mrc.features()
    CRELLO_DEV_DIR.mkdir(parents=True, exist_ok=True)
    if CRELLO_DEV_ASSETS_DIR.exists():
        shutil.rmtree(CRELLO_DEV_ASSETS_DIR)
    CRELLO_DEV_ASSETS_DIR.mkdir(parents=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for p in list(CRELLO_DEV_DIR.glob(f"{PREFIX}_*.json")) + list(PREVIEW_DIR.glob("*.png")):
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
            rec = mrc.convert(row, F, tid, CRELLO_DEV_ASSETS_DIR, measurer())
            rec.update(split=SPLIT, notes=rec["notes"].replace("Crello ad template", "Crello ad template (dev)"))
            Image.open(io.BytesIO(row["preview"]["bytes"])).convert("RGB").save(PREVIEW_DIR / f"{tid}.png")
            records.append(rec)
    records.sort(key=lambda r: r["task_id"])
    with merged_assets_dir(*CRELLO_DEV_FOLDERS) as assets:
        for rec in records:
            issues = quality_issues(rec, assets)
            assert not issues, (rec["task_id"], issues)
            (CRELLO_DEV_DIR / f"{rec['task_id']}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    return records


def summarise(pool_rows: list[dict], profiles: dict, picked: list[dict], records: list[dict]) -> dict:
    """Aggregate numbers only (no Crello content)."""
    prof = [profiles[r["crello_id"]] for r in picked]
    cls = lambda rs, key: dict(sorted(Counter(key(r) for r in rs).items()))
    return {"source": "cyberagent/crello rev 5.1.0, gate-passing ad templates of challenge2's scan, "
                      "challenge2's design families excluded",
            "pool_families": len(pool_rows), "selected": len(records),
            "selected_by_class": cls(records, lambda r: r["features"][0]),
            "selected_by_photo": cls(records, lambda r: r["features"][1]),
            "layered": sum(is_layered(p) for p in prof), "simple": sum(not is_layered(p) for p in prof),
            "with_groups": sum(p["groups"] > 0 for p in prof),
            "with_background_image": sum(p["background_image"] for p in prof),
            "with_two_or_more_background_layers": sum(p["background_layers"] >= 2 for p in prof),
            "groups": sum(p["groups"] for p in prof), "nested_groups": sum(p["nested_groups"] for p in prof),
            "stretch_groups": sum(p["stretch_groups"] for p in prof),
            "fixed_aspect_groups": sum(p["fixed_groups"] for p in prof),
            "image_containers": sum(p["image_containers"] for p in prof),
            "units_held": sum(p["units_held"] for p in prof),
            "content_units": sum(p["content_units"] for p in prof),
            "pool_layered_share": round(sum(is_layered(p) for p in profiles.values() if "error" not in p)
                                        / max(1, sum("error" not in p for p in profiles.values())), 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    scan_file = CHALLENGE2_DIR / "scan.jsonl"
    if not scan_file.exists():
        print("no challenge2 scan: run python -m dev.distill.make_real_challenge first")
        return 1
    scan_rows = [json.loads(x) for x in scan_file.read_text().splitlines() if x.strip()]
    chl2_ids = {json.loads(p.read_text())["crello_id"] for p in CHALLENGE2_DIR.glob("cr2_*.json")}
    rows = pool(scan_rows, chl2_ids, args.seed)
    print(f"[crello_dev] {len(rows)} families outside challenge2's {len(chl2_ids)} ads; profiling layers")
    profiles = profile_pool(rows, args.workers)
    CRELLO_DEV_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text("".join(json.dumps(profiles[r["crello_id"]]) + "\n" for r in rows if r["crello_id"] in profiles))
    picked = select(rows, profiles, args.n, args.seed)
    records = write_selected(picked)
    summary = summarise(rows, profiles, picked, records)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {len(records)} ads to {CRELLO_DEV_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
