"""Fit Sonnet's proportions (engine/pipeline/layout/targets.py) from its
layouts of train ads, CPU only, no LLM calls.

Data: training-data/llm-distill/train_v2.jsonl (2,659 Sonnet layouts in compact
form {"e": [[id, x, y, w, h, (font)], ...]} for train-split ads and the extra
augmented train ads, dev/distill/extra_ads.py). Roles come from the engine's
own heuristic plan (same assets dir as the A/B runs), so the runtime features
match exactly.

Model per target: ridge regression on standardized features (targets.features:
target / source aspect, area and short-side scale, headline length and size,
element counts, hero share, the same quantity measured on the source, the
role's text length, plus pairwise products), in the target's own scale or in
log scale, whichever cross-validates better; lambda from a small grid. CV is
5-fold grouped by PARENT ad (an ad and all its augmented variants are in one
fold), stricter than grouping by task id. Targets with CV R^2 < MIN_R2 are
dropped. Also reports the fitted model on the Sonnet holdout layouts
(qc.nosync/gen_holdout.jsonl: holdout ads, never used for fitting).

    python -m dev.distill.fit_targets fit [--write]     # CV table (+ holdout), writes sonnet_targets.json
    python -m dev.distill.fit_targets fit --exclude-fold 0 --out <path>   # a tuning model without one fold

Gitignored inputs; the written coefficients are a few KB of numbers (no content).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dev.distill import ab_engine as ab

TRAIN = ab.DATA_ROOT / "llm-distill" / "train_v2.jsonl"
HOLDOUT = ab.DATA_ROOT / "llm-distill" / "qc.nosync" / "gen_holdout.jsonl"
MIN_R2 = 0.3
LAMBDAS = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
FOLDS = 5
NAMES = ("font_headline", "font_subheadline", "font_other_text", "font_cta", "font_legal", "font_badge",
         "cta_h", "cta_w", "logo", "hero", "content_w", "content_h", "margin", "gap")


def parent(task_id: str) -> str:
    return re.sub(r"_v\d+$", "", task_id)


def sonnet_elements(source: dict, answer_rows: list) -> dict:
    """Sonnet's compact rows merged onto the source elements (type etc. from the source)."""
    src = {e["id"]: e for e in source["elements"]}
    out = {}
    for r in answer_rows:
        i = str(r[0])
        if i not in src:
            continue
        e = dict(src[i], x=float(r[1]), y=float(r[2]), width=float(r[3]), height=float(r[4]))
        if src[i]["type"] == "text":
            e["font_size"] = float(r[5]) if len(r) > 5 and r[5] else None
        out[i] = e
    return out


def _setup_env():
    os.environ["CHAMELEON_LLM"] = "heuristic"
    os.environ.setdefault("CHAMELEON_ASSETS_DIR", str(ab._assets()))


def load_rows(path: Path, designs: dict, targets: dict, raw: bool = False) -> list[dict]:
    """One record per usable Sonnet layout: task, group, target, source stats, Sonnet's measured values.
    Measured on the layout as the engine draws it (llm_layout.execute: images kept in proportion inside
    Sonnet's box, text fitted into its box), which is what the judge saw; `raw`: Sonnet's numbers as written."""
    from dev.distill.llm_layout import _compact_to_elements, execute
    from engine.pipeline.layout import targets as T
    from engine.pipeline.priority import heuristic_plan
    from engine.resize import _get_measurer
    m = _get_measurer()
    plans, stats, out = {}, {}, []
    for line in open(path):
        r = json.loads(line)
        if not r.get("answer") or r.get("ok") is False:
            continue
        tid, tn = r["cell"].split("@")
        d = designs.get(tid)
        if d is None or tn not in targets:
            continue
        src = d["source"]
        if tid not in plans:
            plans[tid] = heuristic_plan(src)
            stats[tid] = T.source_stats(src, plans[tid])
        try:
            rows = json.loads(r["answer"])["e"]
        except Exception:
            continue
        t = targets[tn]
        if raw:
            els = sonnet_elements(src, rows)
        else:
            try:
                d_ = execute(src, {"width": t["width"], "height": t["height"]}, plans[tid], m, _compact_to_elements(rows))
            except Exception:
                continue
            els = {e["id"]: e for e in d_["elements"]}
        y = T.measure(els, plans[tid], t["width"], t["height"])
        out.append({"cell": r["cell"], "task_id": tid, "group": parent(tid), "W": t["width"], "H": t["height"],
                    "st": stats[tid], "y": y})
    return out


def design_maps(include_holdout: bool = False) -> tuple[dict, dict]:
    from dev.distill.corpus import load_corpus
    from dev.distill.extra_ads import load_extra
    designs = {d["task_id"]: d for d in load_corpus("train") + load_extra()}
    if include_holdout:
        designs.update({d["task_id"]: d for d in load_corpus("holdout")})
    targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
    return designs, targets


def matrix(recs: list[dict], name: str):
    from engine.pipeline.layout import targets as T
    X, y, g, keys = [], [], [], None
    for r in recs:
        v = r["y"].get(name)
        if v is None or not math.isfinite(v):
            continue
        f = T.features(r["st"], r["W"], r["H"], name)
        if f is None:
            continue
        if keys is None:
            keys = sorted(f)
        X.append([f[k] for k in keys])
        y.append(v)
        g.append(r["group"])
    return np.array(X, dtype=float), np.array(y, dtype=float), g, keys


def _ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    b = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - b))
    return mu, sd, w, b


def _pred(model, X):
    mu, sd, w, b = model
    return ((X - mu) / sd) @ w + b


def _fwd(y, transform):
    return np.log(np.maximum(y, 1e-4)) if transform == "log" else y


def _inv(z, transform):
    return np.exp(z) if transform == "log" else z


def _folds(groups: list[str], seed: int = 0) -> np.ndarray:
    ads = sorted(set(groups))
    random.Random(seed).shuffle(ads)
    fold = {a: i % FOLDS for i, a in enumerate(ads)}
    return np.array([fold[a] for a in groups])


def cv(X, y, groups, transform, lam):
    F = _folds(groups)
    pred = np.zeros_like(y)
    for f in range(FOLDS):
        tr, te = F != f, F == f
        if not te.any() or tr.sum() < 20:
            continue
        m = _ridge(X[tr], _fwd(y[tr], transform), lam)
        pred[te] = _inv(_pred(m, X[te]), transform)
    return pred


def _r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def fit(recs: list[dict], hold: list[dict] | None = None, log=print) -> dict:
    out = {"targets": {}, "report": {}}
    for name in NAMES:
        X, y, g, keys = matrix(recs, name)
        if len(y) < 50 or len(set(g)) < FOLDS:
            log(f"{name:18s} too few examples ({len(y)})")
            continue
        best = None
        for transform in ("id", "log"):
            if transform == "log" and (y <= 0).any():
                continue
            for lam in LAMBDAS:
                p = cv(X, y, g, transform, lam)
                r2 = _r2(y, p)
                if best is None or r2 > best[0]:
                    best = (r2, transform, lam, p)
        r2, transform, lam, p = best
        mae = float(np.abs(y - p).mean())
        # the source value scaled naively (the same share on the new canvas) as a reference
        rep = {"n": int(len(y)), "ads": len(set(g)), "mean": float(y.mean()), "cv_r2": round(r2, 3), "cv_mae": round(mae, 4),
               "transform": transform, "lam": lam}
        m = _ridge(X, _fwd(y, transform), lam)
        if hold:
            Xh, yh, _, kh = matrix(hold, name)
            if len(yh) and kh == keys:
                ph = _inv(_pred(m, Xh), transform)
                rep["hold_n"] = int(len(yh))
                rep["hold_r2"] = round(_r2(yh, ph), 3)
                rep["hold_mae"] = round(float(np.abs(yh - ph).mean()), 4)
        out["report"][name] = rep
        line = (f"{name:18s} n={rep['n']:5d} ads={rep['ads']:3d} mean={rep['mean']:.3f} CV R2={r2:5.2f} "
                f"MAE={mae:.4f} [{transform}, lam {lam}]")
        if "hold_r2" in rep:
            line += f"  holdout n={rep['hold_n']} R2={rep['hold_r2']:5.2f} MAE={rep['hold_mae']:.4f}"
        log(line + ("" if r2 >= MIN_R2 else "  -> dropped"))
        if r2 < MIN_R2:
            continue
        mu, sd, w, b = m
        lo, hi = float(np.quantile(y, 0.005)), float(np.quantile(y, 0.995))
        def r6(a):
            return [float(f"{float(v):.6g}") for v in a]
        out["targets"][name] = {"features": keys, "mean": r6(mu), "std": r6(sd), "w": r6(w),
                                "lo": r6(X.min(0)), "hi": r6(X.max(0)), "b": float(f"{float(b):.6g}"), "transform": transform, "range": [round(lo, 5), round(hi, 5)],
                                "cv_r2": round(r2, 3), "cv_mae": round(mae, 4)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dev.distill.fit_targets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--write", action="store_true", help="write engine/pipeline/layout/sonnet_targets.json")
    f.add_argument("--out", default=None, help="write the model here instead")
    f.add_argument("--exclude-fold", type=int, default=None, help="fit without this parent-ad fold (tuning model)")
    f.add_argument("--raw", action="store_true", help="Sonnet's numbers as written, not as the engine draws them")
    args = ap.parse_args(argv)
    _setup_env()
    from engine.pipeline.layout import targets as T
    designs, targets = design_maps(include_holdout=True)
    recs = load_rows(TRAIN, designs, targets, args.raw)
    hold = load_rows(HOLDOUT, designs, targets, args.raw)
    if args.exclude_fold is not None:
        F = _folds([r["group"] for r in recs])
        held = sorted({r["group"] for r, k in zip(recs, F) if k == args.exclude_fold})
        recs = [r for r, k in zip(recs, F) if k != args.exclude_fold]
        print(f"excluding fold {args.exclude_fold}: {held}")
    print(f"{len(recs)} train layouts, {len({r['group'] for r in recs})} parent ads; {len(hold)} holdout layouts")
    model = fit(recs, hold)
    model["meta"] = {"data": "train_v2.jsonl (Sonnet compact layouts, train + extra train ads)",
                     "measured": "raw" if args.raw else "executed", "folds": FOLDS,
                     "group": "parent ad", "min_r2": MIN_R2, "exclude_fold": args.exclude_fold}
    path = Path(args.out) if args.out else (T.MODEL_PATH if args.write else None)
    if path is not None:
        path.write_text(json.dumps(model, separators=(",", ":")))
        print(f"wrote {path} ({path.stat().st_size} bytes, {len(model['targets'])} targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
