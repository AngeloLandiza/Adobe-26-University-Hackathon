"""Calibrate and check CHAMELEON_NEAR_FIT_GUARD (grading/score.py).

The near-fit rule hands near-aspect cells to the uniform fit (the safe
fallback) unless a re-layout leads the grader by more than 0.10. The guard
undoes that when the fit makes the headline (and CTA) clearly smaller than
the best re-layout's and leaves much of the new canvas empty.

    # train: features of every teacher-judged fallback-vs-layout pair on a near-aspect cell
    python -m dev.distill.nearfit_guard train
    # check (not tuning): the guard on the cells of ab/bisect2__vs__b2_nearfit (holdout)
    python -m dev.distill.nearfit_guard check
    # how many near-fit picks the guard flips (v3L engine) on a split
    python -m dev.distill.nearfit_guard flips --split train|holdout|crello_dev|challenge2

CPU only, one process (the cells are few).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from dev.distill.paths import DATA_ROOT

NEAR = 1.35
TRAIN_JUDGMENTS = DATA_ROOT / "judgments" / "judgments.jsonl"
CHECK_PAIR = DATA_ROOT / "ab" / "bisect2__vs__b2_nearfit" / "judgments.jsonl"
V3L = {"CHAMELEON_STRATEGIES": "bisection", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_HIER_LOCK": "1",
       "CHAMELEON_NEAR_FIT": "1", "CHAMELEON_LAYERS": "1"}
B2_NEARFIT = {"CHAMELEON_STRATEGIES": "bisection", "CHAMELEON_BISECT_V2": "1", "CHAMELEON_NEAR_FIT": "1"}


def _aspect(src_canvas: dict, target: dict) -> float:
    a = src_canvas["width"] / src_canvas["height"]
    b = target["width"] / target["height"]
    return max(a / b, b / a)


def _setup(env: dict, folders: tuple = ()):
    import os
    from dev.distill import ab_engine
    for k in list(os.environ):
        if k.startswith("CHAMELEON_") and k not in ("CHAMELEON_ASSETS_DIR",):
            del os.environ[k]
    ab_engine._worker_init(env, folders)


def _candidates(source: dict, target: dict):
    from engine.pipeline.layout.fallback import safe_fallback
    from engine.pipeline.layout.generate import generate_candidates
    from engine.pipeline.priority import plan as priority_plan
    from engine.resize import _get_measurer
    m = _get_measurer()
    plan = priority_plan(source)
    return plan, m, [safe_fallback(source, target, plan, m)] + generate_candidates(source, target, plan, m)


def _fmt(f: dict) -> str:
    r = lambda v: "  -  " if v is None else f"{v:.2f}"
    return (f"head {r(f['head_ratio'])} cta {r(f['cta_ratio'])} empty src {f['empty_src']:.2f} "
            f"fb {f['empty_fb']:.2f} layout {f['empty_best']:.2f} (fb-layout {f['empty_fb'] - f['empty_best']:+.2f})")


def train() -> list[dict]:
    """Every train fallback-vs-layout judgment on a near-aspect cell: the guard's features, the verdict."""
    from dev.distill import ab_engine
    from dev.distill.corpus import load_corpus
    from engine.pipeline.grading.score import guard_blocks, near_fit_features
    _setup({})  # the pool was built by the plain engine
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    srcs = {d["task_id"]: d["source"] for d in load_corpus("train")}
    rows = []
    for line in TRAIN_JUDGMENTS.read_text().splitlines():
        r = json.loads(line)
        if r["split"] != "train" or "fallback" not in (r["a"], r["b"]) or r["winner"] not in ("a", "b"):
            continue
        tid, tname = r["cell"].split("@")
        t = {"width": targets[tname]["width"], "height": targets[tname]["height"]}
        if _aspect(srcs[tid]["canvas"], t) > NEAR:
            continue
        other = r["b"] if r["a"] == "fallback" else r["a"]
        plan, m, cands = _candidates(srcs[tid], t)
        by = {c.candidate_id: c for c in cands}
        if other not in by:
            print(f"{r['cell']}: {other} not rebuilt; skipped")
            continue
        f = near_fit_features(srcs[tid], plan, m, by["fallback"].design, by[other].design)
        fb_won = (r["winner"] == "a") == (r["a"] == "fallback")
        rows.append({"cell": r["cell"], "other": other, "fb_won": fb_won, "conf": r["confidence"], **f,
                     "guard": guard_blocks(f)})
    for x in sorted(rows, key=lambda x: (x["fb_won"], x["head_ratio"] or 9)):
        print(f"{'FB ' if x['fb_won'] else 'LAY'} c{x['conf']} {x['cell'][:44]:44s} {_fmt(x)} guard={int(x['guard'])}")
    agree = sum(x["guard"] != x["fb_won"] for x in rows)
    print(f"train: {len(rows)} judged near pairs, fallback won {sum(x['fb_won'] for x in rows)}; "
          f"guard fires on {sum(x['guard'] for x in rows)}; guard decision agrees with the teacher on {agree}/{len(rows)}")
    return rows


def check() -> list[dict]:
    """bisect2 vs b2_nearfit (holdout, 15 changed cells): would the guard have
    kept bisect2's re-layout, and did the teacher prefer it?"""
    from dev.distill import ab_engine
    from dev.distill.corpus import load_corpus
    from engine.pipeline.grading.score import guard_blocks, near_fit_features, pick_best
    import os
    _setup(B2_NEARFIT)
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    srcs = {d["task_id"]: d["source"] for d in load_corpus("holdout")}
    rows = []
    for line in CHECK_PAIR.read_text().splitlines():
        r = json.loads(line)
        tid, tname = r["cell"].split("@")
        t = {"width": targets[tname]["width"], "height": targets[tname]["height"]}
        plan, m, cands = _candidates(srcs[tid], t)
        os.environ.pop("CHAMELEON_NEAR_FIT", None)
        best, _ = pick_best(srcs[tid], t, plan, cands, m)
        os.environ["CHAMELEON_NEAR_FIT"] = "1"
        if best.is_fallback:
            continue
        f = near_fit_features(srcs[tid], plan, m, cands[0].design, best.design)
        fb_won = None if r["winner"] not in ("a", "b") else (r["winner"] == "a") == (r["a"] == "b2_nearfit")
        rows.append({"cell": r["cell"], "fb_won": fb_won, "conf": r.get("confidence"), **f, "guard": guard_blocks(f)})
    for x in sorted(rows, key=lambda x: (str(x["fb_won"]), x["head_ratio"] or 9)):
        tag = {True: "FB ", False: "LAY", None: "TIE"}[x["fb_won"]]
        print(f"{tag} c{x['conf']} {x['cell'][:44]:44s} {_fmt(x)} guard={int(x['guard'])}")
    decided = [x for x in rows if x["fb_won"] is not None]
    print(f"check: {len(rows)} cells; teacher preferred the fit in {sum(x['fb_won'] for x in decided)} of "
          f"{len(decided)} decided; guard fires on {sum(x['guard'] for x in rows)} "
          f"({sum(x['guard'] and x['fb_won'] for x in decided)} of them against the teacher's preference for the fit); "
          f"agreement {sum(x['guard'] != x['fb_won'] for x in decided)}/{len(decided)}")
    return rows


def flips(split: str) -> Counter:
    """v3L on every near-aspect cell of `split`: how often near-fit picks the
    fallback over a re-layout, and how often the guard undoes it."""
    from dev.distill import ab_engine
    from dev.distill.layer_flaws import cells_for, folders_for
    from engine.pipeline.grading.score import guard_blocks, near_fit_features, pick_best
    import os
    _setup(V3L, folders_for(split))
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    cells, designs = cells_for(split)
    out = Counter()
    for c in cells:
        src = designs[c["task_id"]]["source"]
        t = {"width": targets[c["target"]]["width"], "height": targets[c["target"]]["height"]}
        if _aspect(src["canvas"], t) > NEAR:
            continue
        out["near cells"] += 1
        plan, m, cands = _candidates(src, t)
        win, _ = pick_best(src, t, plan, cands, m)
        os.environ.pop("CHAMELEON_NEAR_FIT", None)
        best, _ = pick_best(src, t, plan, cands, m)
        os.environ["CHAMELEON_NEAR_FIT"] = "1"
        if not win.is_fallback or best.is_fallback:
            continue
        out["near-fit picks the fallback over a re-layout"] += 1
        f = near_fit_features(src, plan, m, cands[0].design, best.design)
        if guard_blocks(f):
            out["guard flips"] += 1
            print(f"flip {c['cell']}: {_fmt(f)} -> {best.candidate_id}")
    print(f"{split}: {dict(out)}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("train")
    sub.add_parser("check")
    f = sub.add_parser("flips")
    f.add_argument("--split", required=True)
    args = ap.parse_args(argv)
    {"train": train, "check": check}.get(args.step, lambda: flips(args.split))()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
