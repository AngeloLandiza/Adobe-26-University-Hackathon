"""Detectors for the space-and-grouping flaws the teacher names in v4's layouts,
and the harness that validates them against critiques (CPU only, no judge).

The teacher's critiques of the anchored layouts (ab/v3L__vs__v3L_anch and the
other v3L/v4 pairs) keep naming three things:

- **lopsided whitespace / dead bands**: "the stack sits too high, with about
  30px above the headline and about 75px of empty band below the button";
  "the whole right half is dead space";
- **bands stop spanning**: "the dark header bar no longer spans the canvas";
- **related pieces drift apart**: "'Ships late June' has drifted far below
  the CTA, cut off from its button".

Each detector measures one of them on a finished design against its source:

- margin_imbalance: the content block's two margins on an axis are lopsided
  (the larger more than IMB_RATIO x the smaller and the difference more than
  IMB_SHARE of the axis), and the source wasn't lopsided that way on that axis.
  Content = units that are not background or decoration, minus any unit that
  spans BAND_SPAN of that axis (a band, a full-height photo).
- empty_band: the largest rectangle no content ink touches is at least
  EMPTY_RECT of the canvas and EMPTY_GROWTH of it more than the source's.
  (A first version measured the largest strip across the whole canvas; it
  did not separate the holdout critiques, 14% vs 13%, because a photo at the
  side keeps any full-width strip from being empty.)
- band_broken: a shape or image (not background) that spanned the source
  edge to edge on an axis (BAND_SPAN of it, or touching both edges) no longer
  spans the target on that axis.
- group_split: two content units that were close in the source (stacked or
  side by side with a gap at most PAIR_GAP x the smaller one's size along
  that axis, overlapping across it) are now more than SPLIT_RATIO x their
  scaled source gap apart (and at least SPLIT_SLACK of the short side more).

    # rebuild every design the teacher critiqued in the v3L/v4 pairs (engine re-run, <= 3 processes)
    python -m dev.distill.balance_flaws rebuild-judged --name bal_judged
    # detector vs critique table
    python -m dev.distill.balance_flaws validate --name bal_judged
    # detector rates on layer_flaws/eval_anchored caches (before/after)
    python -m dev.distill.balance_flaws rates --name bal_tr_v4 --name bal_tr_b
    # source | cache 1 | cache 2 for cells whose designs differ (Crello: write into a .nosync folder)
    python -m dev.distill.balance_flaws sheet --name bal_tr_v4 --name bal_tr_b --out F.png

Caches are written under training-data/ab/_flaws.nosync/ (gitignored: the
challenge2 and crello_dev designs are Crello-derived).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dev.distill.bisect_flaws import AB_ROOT, CACHE_DIR, ink_box, plan_from_dict, plan_to_dict

LAYER_ROLES = ("background", "decoration")
_PHOTO_ROLES = ("hero_image", "secondary_image")

# -- thresholds, from the critiques' own words and design reasoning (not fitted) --
IMB_RATIO = 2.0      # one margin more than 2x the other ...
IMB_SHARE = 0.06     # ... and more than 6% of the axis apart ("30px above, 75px below" on a 768 render)
IMB_SOURCE = 0.5     # the source was lopsided the same way by at least this x the target's imbalance: deliberate
BAND_SPAN = 0.95     # spans 95% of an axis: a band (or a full-height photo)
EDGE_TOL = 0.01      # share of the side: this close to an edge touches it
EMPTY_RECT = 0.25    # the largest empty rectangle a quarter of the canvas or more ...
EMPTY_GROWTH = 0.05  # ... and 5 points of the canvas more than the source's
PAIR_GAP = 1.0       # a source pair counts as close when its gap is at most the smaller one's size along the axis ...
PAIR_GAP_SIDE = 0.08  # ... and at most 8% of that source side
PAIR_ACROSS = 0.5    # ... and they overlap across the axis by half the narrower one (same column / row)
BAND_ELONG = 2.0     # a band is at least 2x longer along the axis it spans than across it
SPLIT_RATIO = 1.5    # now farther apart than 1.5x the gap scaled with the pair ...
SPLIT_SLACK = 0.03   # ... and at least 3% of the short side more than that scaled gap
FAMILIES = ("margin_imbalance", "empty_band", "band_broken", "group_split")


# ============================================================ helpers

def _box(el: dict):
    from engine.pipeline.geometry import rotated_aabb
    return rotated_aabb(el)


def _union(boxes):
    from engine.pipeline.geometry import union_box
    return union_box(boxes)


def _span(b, ax):
    return (b[0], b[0] + b[2]) if ax == "x" else (b[1], b[1] + b[3])


def _content(plan):
    return [u for u in plan.ranked_units() if u.role not in LAYER_ROLES and u.element_ids]


def unit_boxes(design: dict, plan, m) -> dict[str, tuple]:
    els = {e["id"]: e for e in design["elements"]}
    return {u.unit_id: _union([ink_box(els[i], m) for i in u.element_ids]) for u in _content(plan)
            if all(i in els for i in u.element_ids)}


def spans(b, ax: str, side: float) -> bool:
    """Spans BAND_SPAN of the axis, or touches both of its edges."""
    lo, hi = _span(b, ax)
    tol = EDGE_TOL * side
    return (hi - lo) >= BAND_SPAN * side or (lo <= tol and hi >= side - tol)


def margins(boxes: list, ax: str, side: float) -> tuple[float, float] | None:
    """(start margin, end margin) of the content block on an axis; boxes that
    span the axis don't count (a band, a full-height photo)."""
    use = [b for b in boxes if not spans(b, ax, side)]
    if not use:
        return None
    lo = min(_span(b, ax)[0] for b in use)
    hi = max(_span(b, ax)[1] for b in use)
    return max(0.0, lo), max(0.0, side - hi)


def lopsided(mg, side: float) -> tuple[bool, float, int]:
    """(lopsided?, signed imbalance as a share of the side (+: more room at the end), heavier side)."""
    if mg is None:
        return False, 0.0, 0
    a, b = mg
    imb = (b - a) / side
    big, small = max(a, b), min(a, b)
    return big > IMB_RATIO * max(small, 1.0) and abs(imb) > IMB_SHARE, imb, (1 if imb > 0 else -1)


def empty_strip(boxes: list, ax: str, side: float) -> tuple[float, str]:
    """Largest strip across the canvas, perpendicular to `ax`, that no box
    touches: (share of the side, 'edge' | 'inner')."""
    iv = sorted(_span(b, ax) for b in boxes)
    if not iv:
        return 1.0, "edge"
    best, where = max(0.0, iv[0][0]) / side, "edge"
    reach = iv[0][1]
    for lo, hi in iv[1:]:
        if lo > reach and (lo - reach) / side > best:
            best, where = (lo - reach) / side, "inner"
        reach = max(reach, hi)
    if (side - reach) / side > best:
        best, where = (side - reach) / side, "edge"
    return max(0.0, best), where


def _gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return math.hypot(dx, dy)


def close_pairs(boxes: dict, W: float, H: float) -> list[tuple[str, str, str, float]]:
    """Source pairs (a, b, axis, gap) that sit close: stacked along `axis` with a
    gap at most PAIR_GAP x the smaller one's size along it (and PAIR_GAP_SIDE of
    the side), overlapping across it by PAIR_ACROSS of the narrower one."""
    from engine.pipeline.geometry import overlap_area
    ids = sorted(boxes)
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            A, B = boxes[a], boxes[b]
            if overlap_area(A, B) > 0:
                continue
            for ax, side in (("y", H), ("x", W)):
                cx = "x" if ax == "y" else "y"
                (a0, a1), (b0, b1) = _span(A, cx), _span(B, cx)
                across = min(a1, b1) - max(a0, b0)
                if across < PAIR_ACROSS * min(a1 - a0, b1 - b0):
                    continue
                (p0, p1), (q0, q1) = _span(A, ax), _span(B, ax)
                gap = max(q0 - p1, p0 - q1)
                if gap < 0:
                    continue
                if gap <= PAIR_GAP * min(p1 - p0, q1 - q0) and gap <= PAIR_GAP_SIDE * side:
                    out.append((a, b, ax, gap))
    return out


def unit_scale(src_els: dict, els: dict, unit, sb, tb) -> float:
    """How much a unit grew: its texts' font ratio, else sqrt of the box area ratio."""
    fr = [els[i]["font_size"] / src_els[i]["font_size"] for i in unit.element_ids
          if src_els[i]["type"] == "text" and src_els[i].get("font_size")]
    if fr:
        return max(fr)
    a0, a1 = sb[2] * sb[3], tb[2] * tb[3]
    return math.sqrt(a1 / a0) if a0 > 0 and a1 > 0 else 1.0


# ============================================================ detectors

def detect(source: dict, design: dict, plan, m) -> dict[str, dict]:
    SW, SH = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    W, H = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    src_els = {e["id"]: e for e in source["elements"]}
    els = {e["id"]: e for e in design["elements"]}
    sbox = unit_boxes(source, plan, m)
    tbox = unit_boxes(design, plan, m)
    common = [u for u in _content(plan) if u.unit_id in sbox and u.unit_id in tbox]
    out: dict[str, dict] = {}

    # -- margin imbalance
    worst, axes = 0.0, []
    for ax, side, sside in (("x", W, SW), ("y", H, SH)):
        lt, it, dt = lopsided(margins([tbox[u.unit_id] for u in common], ax, side), side)
        ls, is_, ds = lopsided(margins([sbox[u.unit_id] for u in common], ax, sside), sside)
        deliberate = (is_ != 0 and (is_ > 0) == (it > 0) and abs(is_) >= IMB_SOURCE * abs(it))
        if lt and not deliberate:
            axes.append(ax)
        worst = max(worst, abs(it) if not deliberate else 0.0)
    out["margin_imbalance"] = {"fire": bool(axes), "axes": axes, "worst": round(worst, 3)}

    # -- the largest empty area, against the source's (and, for reference, the largest empty strip)
    from engine.pipeline.layout.strategies.bisection import largest_empty_share
    t_rect = largest_empty_share([tbox[u.unit_id] for u in common], W, H)
    s_rect = largest_empty_share([sbox[u.unit_id] for u in common], SW, SH)
    detail = {}
    for ax, side, sside in (("x", W, SW), ("y", H, SH)):
        t, where = empty_strip([tbox[u.unit_id] for u in common], ax, side)
        s, _ = empty_strip([sbox[u.unit_id] for u in common], ax, sside)
        detail[ax] = (round(t, 3), round(s, 3), where)
    out["empty_band"] = {"fire": t_rect >= EMPTY_RECT and t_rect >= s_rect + EMPTY_GROWTH,
                         "share": round(t_rect, 3), "source": round(s_rect, 3), "strips": detail}

    # -- source-spanning bands that no longer span
    broken = []
    bg = {i for u in plan.units if u.role == "background" for i in u.element_ids}
    for e0 in source["elements"]:
        if e0["type"] not in ("shape", "image") or e0["id"] in bg or e0["id"] not in els:
            continue
        b0, b1 = _box(e0), _box(els[e0["id"]])
        if spans(b0, "x", SW) and spans(b0, "y", SH):
            continue  # a full-bleed layer, not a band
        for ax, sside, side in (("x", SW, W), ("y", SH, H)):
            along, across = (b0[2], b0[3]) if ax == "x" else (b0[3], b0[2])
            if spans(b0, ax, sside) and along >= BAND_ELONG * across and not spans(b1, ax, side):
                broken.append(f"{e0['id']}:{ax}")
    out["band_broken"] = {"fire": bool(broken), "ids": broken}

    # -- close pairs that drifted apart
    by_id = {u.unit_id: u for u in common}
    split = []
    short = min(W, H)
    for a, b, ax, g in close_pairs({k: v for k, v in sbox.items() if k in by_id}, SW, SH):
        if by_id[a].role in _PHOTO_ROLES or by_id[b].role in _PHOTO_ROLES:
            continue  # text, buttons, badges and logos: a photo moving off its caption is a re-layout
        ka = unit_scale(src_els, els, by_id[a], sbox[a], tbox[a])
        kb = unit_scale(src_els, els, by_id[b], sbox[b], tbox[b])
        want = g * math.sqrt(ka * kb)
        now = _gap(tbox[a], tbox[b])
        if now > SPLIT_RATIO * want and now > want + SPLIT_SLACK * short:
            split.append(f"{by_id[a].role}~{by_id[b].role}")
    out["group_split"] = {"fire": bool(split), "pairs": split}
    return out


def detect_all(source: dict, design: dict, plan, m) -> dict[str, dict]:
    """These families plus bisect_flaws' geometry families."""
    from dev.distill.bisect_flaws import detect as bisect_detect
    out = dict(bisect_detect(source, design, plan, m))
    out.update(detect(source, design, plan, m))
    return out


# ============================================================ critiques

_DEAD = r"dead|empty|void|unused|wasted|blank|bare"
_SIDES = r"(top|bottom|left|right|upper|lower)"
_PART = r"(half|third|quarter|band|strip|side|area|portion|corner|region|zone)"
MENTIONS = {
    "margin_imbalance": re.compile(
        r"lopsided|unbalanced|imbalanc|off-cent(re|er)|top-heavy|bottom-heavy|\bsits? (too |very |well )?(high|low)\b|"
        r"too (high|low)\b|(gap|space|margin) above[^.]{0,40}below|"
        rf"(crowds|hugs|clings to|bunched (at|in|into)|clustered (at|in|into)|huddled (at|in|into)|crammed (in|into|at)|"
        rf"pushed (up|down|to|into)|piled (in|into)|sits? in) the {_SIDES}|"
        rf"{_SIDES} {_PART}s?( of the (canvas|frame|banner|layout))? (is |are |sits |stays |remains |reads as |left )?"
        rf"(almost |mostly |completely |entirely |largely )?({_DEAD})|"
        rf"({_DEAD}) {_SIDES} {_PART}|leaves? the {_SIDES}[\w ]{{0,25}}({_DEAD})", re.I),
    "empty_band": re.compile(
        rf"({_DEAD}) (band|strip|space|area|zone|gap|half|third|block|field|stretch|expanse|region|patch|canvas|bands|areas)|"
        rf"(band|strip|gap|half|third|area) of ({_DEAD})|({_DEAD})(?: \w+)? (band|space|gap|area)", re.I),
    "band_broken": re.compile(
        r"(bar|band|strip|banner|header|footer|panel|ribbon|stripe|rule|block|overlay|tint)\b[^.]{0,60}?"
        r"(no longer spans|doesn't span|does not span|doesn't reach|does not reach|stops (short|awkwardly|at)|inset|"
        r"fills only the (left|right|top|bottom)|short of the (left|right|top|bottom|canvas|edge)|midpoint)", re.I),
    "group_split": re.compile(
        r"drift|apart from|cut off from|detached|disconnected|far (below|above|from)|separated from|split from|"
        r"torn away|away from (its|the|their)|no longer (sits|grouped|with|next)|isolated from|stranded (below|above|far)|"
        r"(big|large|huge|wide|dead) gap between|loose|scattered|fragment", re.I),
}
JUDGED_PAIRS = ("v3L__vs__v3L_anch", "v3L__vs__v4p", "v3L_chl2__vs__v3L_anch_chl2", "v3L_chl2__vs__v4p_chl2",
                "base_chl2__vs__v3L_chl2")
_SIDE = re.compile(r"(?:^|(?<=[.!?]\s))(A|B)(?=[:\s'’,])")
_BOTH = re.compile(r"^(both|in both|neither)\b", re.I)


def side_paragraphs(reason: str) -> dict[str, str]:
    """{"a": text about A, "b": text about B}: sentences are given to the side
    named last at a sentence start; sentences opening 'Both' / 'In both' go to both."""
    out = {"a": [], "b": []}
    cur = None
    for s in re.split(r"(?<=[.!?])\s+", reason):
        hit = _SIDE.match(s)
        if hit:
            cur = hit.group(1).lower()
        if _BOTH.match(s):
            out["a"].append(s)
            out["b"].append(s)
        elif cur:
            out[cur].append(s)
    return {k: " ".join(v) for k, v in out.items()}


def judged_texts(pairs=JUDGED_PAIRS) -> dict[tuple[str, str], dict]:
    """(variant, cell) -> {"split", "task_id", "target", "texts": [...]} over every judged row."""
    out: dict[tuple[str, str], dict] = {}
    for pair in pairs:
        path = AB_ROOT / pair / "judgments.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            paras = side_paragraphs(row["reason"])
            tid, tname = row["cell"].split("@")
            for side in ("a", "b"):
                key = (row[side], row["cell"])
                c = out.setdefault(key, {"split": row.get("split"), "task_id": tid, "target": tname, "texts": []})
                if paras[side]:
                    c["texts"].append(paras[side])
    return out


def mentions(text: str) -> dict[str, bool]:
    return {k: bool(r.search(text)) for k, r in MENTIONS.items()}


# ============================================================ rebuild (engine re-run)

_W = {}


def _init(env: dict, folders: tuple):
    import os
    from dev.distill import ab_engine
    for k in [k for k in os.environ if k.startswith("CHAMELEON_")]:
        del os.environ[k]
    ab_engine._worker_init(env, folders)


def _one(job: dict) -> dict:
    from dev.distill import ab_engine
    try:
        t0 = time.perf_counter()
        design, plan, best, _ = ab_engine._resize(job["source"], job["canvas"])
        return {"key": job["key"], "cell": job["cell"], "variant": job["variant"], "task_id": job["task_id"],
                "target": job["target"], "candidate_id": best.candidate_id, "is_fallback": bool(best.is_fallback),
                "design_hash": ab_engine.design_hash(design), "seconds": round(time.perf_counter() - t0, 3),
                "design": design, "plan": plan_to_dict(plan)}
    except Exception as exc:
        return {"key": job["key"], "error": f"{type(exc).__name__}: {exc}"}


def rebuild_judged(name: str, workers: int = 3, log=print) -> Path:
    """Re-run the engine for every (variant, cell) the teacher critiqued, with
    that variant's switches; rows keep the design and whether its hash matches
    what was judged (winners.jsonl)."""
    from dev.distill import ab_engine
    from dev.distill.corpus import load_corpus
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    judged = judged_texts()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.jsonl"
    done = {json.loads(x)["key"] for x in path.read_text().splitlines() if x.strip()} if path.exists() else set()
    designs = {}
    for split in sorted({c["split"] for c in judged.values()}):
        designs.update({d["task_id"]: d for d in load_corpus(split)})
    groups: dict[tuple, list] = defaultdict(list)
    for (variant, cell), c in sorted(judged.items()):
        key = f"{variant}|{cell}"
        if key in done or c["task_id"] not in designs:
            continue
        env = json.loads((AB_ROOT / variant / "variant.json").read_text())["env"]
        t = targets[c["target"]]
        groups[(json.dumps(env, sort_keys=True), c["split"])].append(
            {"key": key, "cell": cell, "variant": variant, "task_id": c["task_id"], "target": c["target"],
             "source": designs[c["task_id"]]["source"], "canvas": {"width": t["width"], "height": t["height"]}})
    with path.open("a") as f:
        for (env, split), jobs in groups.items():
            log(f"[balance] {len(jobs)} cells, split {split}, env {env}")
            with ProcessPoolExecutor(max_workers=max(1, min(workers, 3)), initializer=_init,
                                     initargs=(json.loads(env), ab_engine.folders_for(split))) as ex:
                for row in ex.map(_one, jobs, chunksize=4):
                    if "error" in row:
                        log(f"[balance] {row['key']}: {row['error']}")
                        continue
                    f.write(json.dumps(row) + "\n")
    return path


def load_cache(name: str) -> dict[str, dict]:
    path = CACHE_DIR / f"{name}.jsonl"
    rows = (json.loads(x) for x in path.read_text().splitlines() if x.strip())
    return {r.get("key", r["cell"]): r for r in rows}


def _sources() -> dict:
    from dev.distill.corpus import load_corpus
    out = {}
    for split in ("train", "holdout", "challenge2", "crello_dev", "samples"):
        try:
            out.update({d["task_id"]: d["source"] for d in load_corpus(split)})
        except Exception:
            pass
    for p in Path("input").glob("*.json"):
        out.setdefault(p.stem, json.loads(p.read_text())["source"])
    return out


def _detect_rows(rows: dict) -> dict[str, dict]:
    from engine.pipeline.textmeasure import TextMeasurer
    m = TextMeasurer()
    srcs = _sources()
    return {k: detect_all(srcs[r["task_id"]], r["design"], plan_from_dict(r["plan"]), m) for k, r in rows.items()}


# ============================================================ reports

def validate(name: str = "bal_judged", families=FAMILIES + ("edge", "floating")) -> list[dict]:
    """Per family: fire rate where the critique names the flaw vs where it doesn't
    (only designs whose rebuilt hash matches the judged one)."""
    from dev.distill import ab_engine
    from dev.distill.bisect_flaws import MENTIONS as BMENTIONS
    judged = judged_texts()
    rows = load_cache(name)
    winners = {}
    for v in {r["variant"] for r in rows.values()}:
        winners[v] = ab_engine.load_winners(v)
    ok = {k: r for k, r in rows.items() if winners[r["variant"]].get(r["cell"], {}).get("design_hash") == r["design_hash"]}
    det = _detect_rows(ok)
    table = []
    print(f"rebuilt {len(rows)} judged designs; {len(ok)} match the judged design hash")
    print("| family | critiques naming it | detector fires there | other critiques | fires there | gap |")
    print("|---|---|---|---|---|---|")
    for fam in families:
        pos, neg = [], []
        for k in det:
            v, c = k.split("|", 1)
            text = " ".join(judged[(v, c)]["texts"])
            named = MENTIONS[fam].search(text) if fam in MENTIONS else BMENTIONS[fam](text)
            (pos if named else neg).append(k)
        a = sum(det[k][fam]["fire"] for k in pos) / len(pos) if pos else None
        b = sum(det[k][fam]["fire"] for k in neg) / len(neg) if neg else None
        table.append({"family": fam, "mentioned": len(pos), "fire_if_mentioned": a, "not_mentioned": len(neg),
                      "fire_if_not": b})
        f = lambda x: "n/a" if x is None else f"{100 * x:.0f}%"
        gap = None if a is None or b is None else a - b
        print(f"| {fam} | {len(pos)} | {f(a)} | {len(neg)} | {f(b)} | {'n/a' if gap is None else f'{100 * gap:+.0f} pts'} |")
    return table


def rates(names: list[str], families=FAMILIES + ("edge", "floating", "mixed_align", "text_small", "decor_orphan")) -> dict:
    out = {}
    for name in names:
        rows = load_cache(name)
        det = _detect_rows(rows)
        n = len(rows) or 1
        out[name] = {"cells": len(rows), "rates": {k: sum(d[k]["fire"] for d in det.values()) / n for k in families},
                     "fallback": sum(bool(r.get("is_fallback")) for r in rows.values()) / n,
                     "gate_failures": sum(bool(r.get("failed_gates")) for r in rows.values()),
                     "seconds": sum(r.get("seconds", 0) for r in rows.values()) / n}
    print("| family | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in families:
        print(f"| {k} | " + " | ".join(f"{100 * out[nm]['rates'][k]:.0f}%" for nm in names) + " |")
    for k in ("cells", "fallback", "gate_failures", "seconds"):
        print(f"| {k} | " + " | ".join(f"{out[nm][k]:.3f}" if isinstance(out[nm][k], float) else str(out[nm][k])
                                     for nm in names) + " |")
    return out


def contact_sheet(names: list[str], out: Path, cells: list[str] | None = None, n: int = 12, seed: int = 0,
                  tile: int = 300, split: str | None = None) -> Path:
    """source | <name 1> | <name 2> ... for `cells` (default: n seeded cells whose
    designs differ); each tile labelled with its candidate and the families that fire."""
    import random
    from PIL import Image, ImageDraw
    from dev.distill import ab_engine
    from dev.distill.assets import merged_assets_dir
    from engine.pipeline.render import render_to_image

    caches = [load_cache(nm) for nm in names]
    common = sorted(set.intersection(*(set(c) for c in caches)))
    if cells is None:
        diff = [c for c in common if len({cache[c]["design_hash"] for cache in caches}) > 1]
        cells = sorted(random.Random(seed).sample(diff, min(n, len(diff))))
    if split is None:
        meta = CACHE_DIR / f"{names[0]}.meta.json"
        split = json.loads(meta.read_text())["split"] if meta.exists() else "train"
    srcs = _sources()
    dets = [_detect_rows({c: cache[c] for c in cells}) for cache in caches]
    label_h = 30
    rows = []
    folders = () if split == "samples" else ab_engine.folders_for(split)
    with merged_assets_dir(*folders) as assets:
        for c in cells:
            tid = caches[0][c]["task_id"]
            imgs = [("source", render_to_image(srcs[tid], assets_dir=assets, max_size=tile))]
            for nm, cache, det in zip(names, caches, dets):
                fired = ",".join(k for k in FAMILIES + ("edge",) if det[c][k]["fire"])
                imgs.append((f"{nm}: {cache[c]['candidate_id'][:36]}\n{fired}",
                             render_to_image(cache[c]["design"], assets_dir=assets, max_size=tile)))
            w = sum(im.width for _, im in imgs) + 10 * len(imgs)
            h = max(im.height for _, im in imgs) + label_h + 16
            row = Image.new("RGB", (max(w, 400), h), (225, 225, 225))
            d = ImageDraw.Draw(row)
            d.text((4, 2), c, fill="black")
            x = 0
            for lab, im in imgs:
                d.text((x + 4, 14), lab, fill=(40, 40, 160))
                row.paste(im.convert("RGB"), (x, label_h + 12))
                x += im.width + 10
            rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 6 for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("rebuild-judged")
    r.add_argument("--name", default="bal_judged")
    r.add_argument("--workers", type=int, default=3)
    v = sub.add_parser("validate")
    v.add_argument("--name", default="bal_judged")
    s = sub.add_parser("rates")
    s.add_argument("--name", action="append", required=True)
    c = sub.add_parser("sheet")
    c.add_argument("--name", action="append", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--cell", action="append", default=None)
    c.add_argument("--n", type=int, default=12)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--tile", type=int, default=300)
    c.add_argument("--split", default=None)
    args = ap.parse_args(argv)
    if args.step == "rebuild-judged":
        print(rebuild_judged(args.name, args.workers))
    elif args.step == "validate":
        validate(args.name)
    elif args.step == "rates":
        rates(args.name)
    else:
        print(contact_sheet(args.name, args.out, args.cell, args.n, args.seed, args.tile, args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
