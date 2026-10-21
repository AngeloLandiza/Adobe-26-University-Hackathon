"""Geometry detectors for the flaws the teacher keeps naming in bisection layouts.

Each detector looks at one finished design (plus the plan's roles and the
source) and says whether a flaw family is present, e.g. "the CTA label is at
least 0.75x the headline's size" or "a decorative circle that framed the
product in the source no longer touches it". They are measurements, not
fixes: they are only kept if they agree with what the teacher wrote.

    # rebuild designs (engine re-run, CPU only, <= 4 processes); writes a cache under training-data/ab/_flaws.nosync/
    python -m dev.distill.bisect_flaws rebuild --name bisect --env CHAMELEON_STRATEGIES=bisection --split judged
    python -m dev.distill.bisect_flaws rebuild --name train_v1 --env CHAMELEON_STRATEGIES=bisection --split train --cells 60
    # detector vs critique table for the judged bisection cells (holdout + challenge)
    python -m dev.distill.bisect_flaws validate --name bisect
    # detector rates for a cache (before/after comparisons on train)
    python -m dev.distill.bisect_flaws rates --name train_v1 --name train_v2
    # contact sheet: original | cache 1 | cache 2 for cells whose designs differ
    python -m dev.distill.bisect_flaws sheet --name train_v1 --name train_v2 --out sheet.png

"Judged" cells are the 160 cells of base__vs__bisect and base_chl__vs__bisect_chl;
the teacher's paragraph about the bisection side of each row is matched with
keyword rules (MENTIONS) to say whether the critique names the flaw.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dev.distill.paths import DATA_ROOT

AB_ROOT = DATA_ROOT / "ab"
CACHE_DIR = AB_ROOT / "_flaws.nosync"
JUDGED = (("base__vs__bisect", "bisect", "holdout"), ("base_chl__vs__bisect_chl", "bisect_chl", "challenge"))
LAYER_ROLES = ("background", "decoration")

# -- thresholds. Set from design reasoning; logo, edge, legal, alignment and text
# size were then adjusted once against the 158 critiques (what an ad looks like
# at the judge's 768 px render), see docs/optimization-program.md. ---------------
CTA_FONT_RATIO = 0.75      # CTA label at least this x the headline's size: competes with it
CTA_AREA_RATIO = 0.6       # ... or the button's box at least this x the headline's ink area
LEGAL_LINES = 4            # legal wrapped onto this many lines or more: a column, not fine print
LOGO_SMALL = 0.06          # logo's sqrt(area) under this share of the canvas's LONG side: a speck as seen
LOGO_BIG = 0.035           # logo area over this share of the canvas ...
LOGO_BIG_HEAD = 1.0        # ... or over this x the headline's ink area: competes
SAFE_EDGE = 0.05           # content ink closer than this share of the short side to an edge
ALIGN_TOL = 0.02           # share of the canvas width: a 3+ column whose edges and centres all spread wider
TEXT_SMALL = 0.011         # smallest non-legal text under this share of the long side (8.4 px at 768)
SQUASH = 0.10              # aspect change of a shape/image over 10%
WIDOW_SHARE = 0.4          # last line one word, under this share of the longest line
FLOAT_GAP = 0.22           # nearest neighbour farther than this share of sqrt(canvas area)


# ============================================================ geometry helpers

def _box(el: dict):
    from engine.pipeline.geometry import rotated_aabb
    return rotated_aabb(el)


def ink_box(el: dict, m) -> tuple:
    """Where a text's glyphs are (widest line, per text_align); other elements: their bounds."""
    if el["type"] != "text" or el.get("rotation") or m is None:
        return _box(el)
    widths = m.line_widths(el, el["font_size"], el["width"]) or [0.0]
    w = min(max(widths), el["width"])
    align = el.get("text_align") or "left"
    x = el["x"] + (el["width"] - w) / 2 if align == "center" else el["x"] + el["width"] - w if align == "right" else el["x"]
    return (x, el["y"], w, min(el["height"], m.block_height(el, el["font_size"], el["width"])))


def _union(boxes):
    from engine.pipeline.geometry import union_box
    return union_box(boxes)


def _gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return math.hypot(dx, dy)


def _overlap(a, b) -> float:
    from engine.pipeline.geometry import overlap_area
    return overlap_area(a, b)


def plan_from_dict(units: list[dict]):
    from engine.pipeline.types import PriorityPlan, Unit
    return PriorityPlan("heuristic", [Unit(u["unit_id"], u["element_ids"], u["kind"], u.get("role"), u.get("rank"))
                                      for u in units])


def plan_to_dict(plan) -> list[dict]:
    return [{"unit_id": u.unit_id, "element_ids": list(u.element_ids), "kind": u.kind, "role": u.role, "rank": u.rank}
            for u in plan.units]


# ============================================================ detectors

def _content_units(plan):
    return [u for u in plan.ranked_units() if u.role not in LAYER_ROLES]


def _texts(els, unit):
    return [els[i] for i in unit.element_ids if els[i]["type"] == "text"]


def detect(source: dict, design: dict, plan, m) -> dict[str, dict]:
    """{family: {"fire": bool, ...measurements}} for one design."""
    els = {e["id"]: e for e in design["elements"]}
    src = {e["id"]: e for e in source["elements"]}
    W, H = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    short = min(W, H)
    units = _content_units(plan)
    ubox = {u.unit_id: _union([ink_box(els[i], m) for i in u.element_ids]) for u in units}
    by_role = defaultdict(list)
    for u in units:
        by_role[u.role].append(u)
    out: dict[str, dict] = {}

    head = by_role["headline"][0] if by_role["headline"] else None
    head_font = max((t["font_size"] for t in _texts(els, head)), default=None) if head else None
    head_area = ubox[head.unit_id][2] * ubox[head.unit_id][3] if head else None

    # -- CTA competes with the headline
    ratio = area_ratio = None
    for u in by_role["cta"]:
        f = max((t["font_size"] for t in _texts(els, u)), default=None)
        b = _union([_box(els[i]) for i in u.element_ids])
        if f and head_font:
            ratio = max(ratio or 0.0, f / head_font)
        if head_area:
            area_ratio = max(area_ratio or 0.0, b[2] * b[3] / head_area)
    out["cta_big"] = {"fire": bool((ratio or 0) >= CTA_FONT_RATIO or (area_ratio or 0) >= CTA_AREA_RATIO),
                      "font_ratio": ratio, "area_ratio": area_ratio}

    # -- legal blown up: as big as body copy, or a block of real area
    legal_font, legal_lines, body = None, 0, []
    for u in units:
        ts = _texts(els, u)
        if not ts:
            continue
        if u.role == "legal":
            legal_font = max(legal_font or 0.0, max(t["font_size"] for t in ts))
            if m is not None:
                legal_lines = max(legal_lines, max(m.line_count(t, t["font_size"], t["width"]) for t in ts))
        elif u.role in ("subheadline", "other_text"):
            body.append(min(t["font_size"] for t in ts))
    legal_rel = legal_font / min(body) if legal_font and body else None
    out["legal_big"] = {"fire": bool(legal_font and ((legal_rel or 0) >= 1.0 or legal_lines >= LEGAL_LINES)),
                        "font_vs_body": legal_rel, "lines": legal_lines}

    # -- logo a speck / logo a hero
    small = big = False
    logo_share = None
    for u in by_role["logo"]:
        b = ubox[u.unit_id]
        share = math.sqrt(b[2] * b[3]) / max(W, H)
        logo_share = share if logo_share is None else min(logo_share, share)
        small |= share < LOGO_SMALL
        big |= b[2] * b[3] / (W * H) > LOGO_BIG or bool(head_area and b[2] * b[3] > LOGO_BIG_HEAD * head_area)
    out["logo_small"] = {"fire": small, "share": logo_share}
    out["logo_big"] = {"fire": big, "share": logo_share}

    # -- content ink too close to the canvas edge
    dmin = math.inf
    for u in units:
        x, y, w, h = ubox[u.unit_id]
        dmin = min(dmin, x, y, W - x - w, H - y - h)
    out["edge"] = {"fire": dmin < SAFE_EDGE * short, "min_edge": dmin / short if math.isfinite(dmin) else None}

    # -- the smallest non-legal text, as seen (relative to the long side)
    sizes = [t["font_size"] for u in units if u.role != "legal" for t in _texts(els, u)]
    small_text = min(sizes) / max(W, H) if sizes else None
    out["text_small"] = {"fire": bool(small_text is not None and small_text < TEXT_SMALL), "min_share": small_text}

    # -- mixed alignment inside a vertical stack
    out["mixed_align"] = _mixed_alignment([ubox[u.unit_id] for u in units], W)

    # -- decorative pieces: orphaned from what they framed, or cropped/squashed
    out["decor_orphan"], out["squash"], out["decor_edge"] = _decor(source, design, plan, src, els, m, W, H)

    # -- widow word in a multi-line headline (and any text, for reference)
    out["widow"] = _widow(els, [u for u in units if u.role == "headline"], m)
    out["widow_any"] = _widow(els, [u for u in units if u.role not in ("legal",)], m)

    # -- a unit floating far from everything else
    far, worst = [], 0.0
    norm = math.sqrt(W * H)
    for u in units:
        others = [ubox[v.unit_id] for v in units if v is not u]
        if not others:
            continue
        g = min(_gap(ubox[u.unit_id], o) for o in others) / norm
        worst = max(worst, g)
        if g > FLOAT_GAP:
            far.append(u.role)
    out["floating"] = {"fire": bool(far), "worst_gap": worst, "roles": far}
    return out


def _mixed_alignment(boxes: list, W: float) -> dict:
    """Stacks: boxes one above the other whose x-ranges overlap by at least
    half the narrower one (bisection's `_stacks`/`columns` rule: side-by-side
    members are alternatives, one per row). A column of 3+ mixes alignment
    when its left edges, centres and right edges are all spread wider than
    ALIGN_TOL x W, for every way of reading the stack as a column."""
    from engine.pipeline.layout.strategies.bisection import _stacks, columns
    worst, stacks, fire = 0.0, 0, False
    spread = lambda vals: (max(vals) - min(vals)) / W
    for comp in _stacks(boxes):
        cols = [c for c in columns(boxes, comp) if len(c) >= 3]
        if not cols:
            continue
        stacks += 1
        best = min(min(spread([boxes[k][0] for k in c]), spread([boxes[k][0] + boxes[k][2] / 2 for k in c]),
                       spread([boxes[k][0] + boxes[k][2] for k in c])) for c in cols)
        worst = max(worst, best)
        fire |= best > ALIGN_TOL
    return {"fire": fire, "spread": worst, "stacks": stacks}


def _decor(source, design, plan, src, els, m, W, H):
    """Orphan: a decoration that in the source overlapped content units (by at
    least 30% of the smaller box) touches none of them any more. Decor edge
    (reported, not kept): it now sits on the canvas edge though in the source
    it was clear of it. Squash: a shape or
    image (not a background, not a cover image) whose aspect changed > SQUASH."""
    SW, SH = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    content = _content_units(plan)
    orphans, squashed, edged = [], [], []
    for u in plan.units:
        for eid in u.element_ids:
            e0, e1 = src[eid], els[eid]
            if u.role == "background" or (e0["type"] == "image" and e0.get("fit") == "cover"):
                continue
            round_or_image = e0["type"] == "image" or e0.get("shape_type") in ("ellipse", "circle")
            if (round_or_image and u.kind != "container_label" and e0["height"] > 0 and e1["height"] > 0
                    and e0["width"] > 0):
                a0, a1 = e0["width"] / e0["height"], e1["width"] / e1["height"]
                if abs(math.log(a1 / a0)) > math.log(1 + SQUASH):
                    squashed.append(eid)
            if u.role != "decoration":
                continue
            b0, b1 = _box(e0), _box(e1)
            framed, kept = 0, 0
            for v in content:
                c0 = _union([_box(src[i]) for i in v.element_ids])
                c1 = _union([_box(els[i]) for i in v.element_ids])
                small = min(b0[2] * b0[3], c0[2] * c0[3])
                if small > 0 and _overlap(b0, c0) >= 0.3 * small:
                    framed += 1
                    kept += _overlap(b1, c1) > 0
            if framed and not kept:
                orphans.append(eid)
            else:
                tol = 1.0
                on_edge = lambda b, w, h: b[0] <= tol or b[1] <= tol or b[0] + b[2] >= w - tol or b[1] + b[3] >= h - tol
                if on_edge(b1, W, H) and not on_edge(b0, SW, SH) and b1[2] * b1[3] < 0.5 * W * H:
                    edged.append(eid)
    return ({"fire": bool(orphans), "ids": orphans}, {"fire": bool(squashed), "ids": squashed},
            {"fire": bool(edged), "ids": edged})


def _widow(els, units, m) -> dict:
    hits = []
    for u in units:
        for t in _texts(els, u):
            if m is None:
                continue
            wrapped = m.wrap(t, t["font_size"], t["width"])
            lines = [ln for ln in wrapped.split("\n")]
            if len(lines) < 2 or len(lines[-1].split()) != 1:
                continue
            widths = m.line_widths(t, t["font_size"], t["width"])
            if widths and widths[-1] < WIDOW_SHARE * max(widths):
                hits.append(lines[-1].strip())
    return {"fire": bool(hits), "words": hits}


# ============================================================ critique mentions

_B_START = re.compile(r"(?:^|(?<=[.!?]\s))B(?=[:\s'’,])")


def bisect_paragraph(row: dict) -> str | None:
    """The part of the teacher's reason about the bisection side."""
    side = "a" if row["a"].startswith("bisect") else "b"
    r = row["reason"]
    hit = _B_START.search(r)
    if not hit or hit.start() == 0:
        return None
    return r[:hit.start()] if side == "a" else r[hit.start():]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+|,\s+(?:and|but|while)\s+", text) if s]


def _both(a: str, b: str):
    ra, rb = re.compile(a, re.I), re.compile(b, re.I)
    return lambda text: any(ra.search(s) and rb.search(s) for s in _sentences(text))


def _any(a: str):
    r = re.compile(a, re.I)
    return lambda text: bool(r.search(text))


_CTA = r"\bCTA|button|call to action|\bpill"
_DECOR = r"circle|shape|blob|ring|confetti|pattern|stripe|oval|disc|sun\b|decor|ellipse|arc\b|squiggle|shapes"
MENTIONS = {
    "cta_big": _both(_CTA, r"oversized|over-sized|too (big|large)|as (big|large) as|compet|outweigh|huge|blown|overpower|dominat|bigger than"),
    "legal_big": _both(r"legal|fine print|disclaimer|terms", r"blown|oversized|large|big|compet|dense|heavy|\w+-line|column|block"),
    "logo_small": _both(r"logo|wordmark|brand mark", r"speck|tiny|small|illegible|invisible|shrunk|unreadable|sliver|barely|can't be read|cannot be read|too little"),
    "logo_big": _both(r"logo|wordmark", r"blown|oversized|huge|too (big|large)|compet|dominat|outweigh|hero|so large"),
    "edge": _both(r"edge|corner|border|margin", r"jam|crowd|touch|press|hug|crammed|cramped|squeez|tight|hard against|flush|right to|runs to|no \w+ margin|little \w* ?margin|against"),
    "mixed_align": _any(r"align|line up|lined up|lines up|off-axis|different axes|common axis|shared edge|mixes"),
    "decor_orphan": _both(_DECOR, r"cut off|cropped|crop|fragment|sliver|stray|clipped|bleed|apart from|away from|separated|pulled out|outside|parked|orphan|lost|stranded|floats"),
    "squash": _any(r"squash|\boval\b|stretch|distort|warp|elongat"),
    "widow": _any(r"orphan(ed)? ['‘\"“]|orphaned \w+'|single word|one-word|alone on (the|its)|on its own line|widow|['’] sits alone|single-word"),
    "text_small": _any(r"tiny|illegible|unreadable|too small|small to read|can't be read|near-illegible"),
    "floating": _any(r"float|strand|adrift|drift|isolated|on its own|alone|cut off from|torn away|far from|far below|far above|disconnected|scattered|detached"),
}
# The coordinator's broader tallies, for reference only (not matched to a detector).
EXTRA = {
    "dead_space": _any(r"dead|empty|void|unused|sparse"),
}
# (critique family, detector) pairs to validate
DETECTOR_FOR = [("cta_big", "cta_big"), ("legal_big", "legal_big"), ("logo_small", "logo_small"),
                ("logo_big", "logo_big"), ("edge", "edge"), ("mixed_align", "mixed_align"),
                ("decor_orphan", "decor_orphan"), ("decor_orphan", "decor_edge"), ("squash", "squash"),
                ("widow", "widow"), ("floating", "floating"), ("text_small", "text_small")]


def judged_mentions() -> dict[str, dict]:
    """cell -> {"split", "variant", "mentions": {family: bool}, "text"} over every judged row about that cell."""
    out: dict[str, dict] = {}
    for pair, variant, split in JUDGED:
        for line in (AB_ROOT / pair / "judgments.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            para = bisect_paragraph(row)
            if para is None:
                continue
            c = out.setdefault(row["cell"], {"split": split, "variant": variant, "texts": []})
            c["texts"].append(para)
    for c in out.values():
        text = " ".join(c["texts"])
        c["mentions"] = {k: f(text) for k, f in {**MENTIONS, **EXTRA}.items()}
    return out


# ============================================================ rebuild (engine re-run)

def _rebuild_init(env: dict):
    from dev.distill import ab_engine
    ab_engine._worker_init(env)


def _rebuild_one(job: dict) -> dict:
    from dev.distill import ab_engine
    try:
        source, target = job["source"], job["target"]
        t0 = time.perf_counter()
        design, plan, best, _ = ab_engine._resize(source, {"width": target["width"], "height": target["height"]})
        return {"cell": job["cell"], "task_id": job["task_id"], "target": job["target_name"],
                "candidate_id": best.candidate_id, "design_hash": ab_engine.design_hash(design),
                "seconds": round(time.perf_counter() - t0, 3), "design": design, "plan": plan_to_dict(plan)}
    except Exception as exc:
        return {"cell": job["cell"], "error": f"{type(exc).__name__}: {exc}"}


def cells_for(split: str, n: int | None = None, seed: int = 0) -> tuple[list[dict], dict]:
    """(cells, designs by task_id). split: judged | train | holdout | challenge."""
    from dev.distill import ab_engine
    from dev.distill.corpus import load_corpus
    targets = json.loads(ab_engine.TARGETS_PATH.read_text())
    if split == "judged":
        judged = judged_mentions()
        designs = {d["task_id"]: d for d in load_corpus("holdout") + load_corpus("challenge")}
        cells = []
        for cell, info in sorted(judged.items()):
            tid, tname = cell.split("@")
            cells.append({"cell": cell, "task_id": tid, "target": tname, "split": info["split"]})
        return cells, designs
    if split == "challenge":
        chl = load_corpus("challenge")
        return ab_engine.challenge_cells(chl, targets, n, seed), {d["task_id"]: d for d in chl}
    cells = ab_engine.pool_cells(ab_engine.POOL_DIR, split, n, seed)
    return cells, {d["task_id"]: d for d in load_corpus(split)}


def rebuild(name: str, env: dict, split: str, n: int | None = None, seed: int = 0, workers: int = 4,
            log=print) -> Path:
    from dev.distill import ab_engine
    targets = {t["name"]: t for t in json.loads(ab_engine.TARGETS_PATH.read_text())}
    cells, designs = cells_for(split, n, seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.jsonl"
    done = {json.loads(x)["cell"] for x in path.read_text().splitlines() if x.strip()} if path.exists() else set()
    jobs = [{"cell": c["cell"], "task_id": c["task_id"], "target_name": c["target"],
             "source": designs[c["task_id"]]["source"], "target": targets[c["target"]]}
            for c in cells if c["cell"] not in done and c["task_id"] in designs and c["target"] in targets]
    log(f"[flaws] {name}: {len(jobs)} cells to run ({len(done)} cached)")
    with ProcessPoolExecutor(max_workers=min(workers, 4), initializer=_rebuild_init, initargs=(env,)) as ex, \
            path.open("a") as f:
        for row in ex.map(_rebuild_one, jobs):
            if "error" in row:
                log(f"[flaws] {row['cell']}: {row['error']}")
                continue
            f.write(json.dumps(row) + "\n")
    meta = CACHE_DIR / f"{name}.meta.json"
    meta.write_text(json.dumps({"name": name, "env": env, "split": split, "cells": n, "seed": seed}, indent=2))
    return path


def load_cache(name: str) -> dict[str, dict]:
    path = CACHE_DIR / f"{name}.jsonl"
    return {r["cell"]: r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def _sources(cells) -> dict:
    from dev.distill.corpus import load_corpus
    return {d["task_id"]: d["source"] for d in load_corpus(None, include_challenge=True)}


def detect_cache(rows: dict[str, dict], measurer=None) -> dict[str, dict]:
    from engine.pipeline.textmeasure import TextMeasurer
    m = measurer or TextMeasurer()
    srcs = _sources(rows)
    return {c: detect(srcs[r["task_id"]], r["design"], plan_from_dict(r["plan"]), m) for c, r in rows.items()}


# ============================================================ reports

def validate(name: str = "bisect") -> list[dict]:
    """Per family: fire rate where the critique names the flaw vs where it doesn't."""
    from dev.distill import ab_engine
    judged = judged_mentions()
    rows = load_cache(name)
    winners = {}
    for _, variant, _ in JUDGED:
        winners.update(ab_engine.load_winners(variant))
    match = sum(1 for c, r in rows.items() if winners.get(c, {}).get("design_hash") == r["design_hash"])
    det = detect_cache({c: r for c, r in rows.items() if c in judged})
    table = []
    for fam, dkey in DETECTOR_FOR:
        pos = [c for c in det if judged[c]["mentions"][fam]]
        neg = [c for c in det if not judged[c]["mentions"][fam]]
        fp = sum(det[c][dkey]["fire"] for c in pos)
        fn = sum(det[c][dkey]["fire"] for c in neg)
        table.append({"family": fam if fam == dkey else f"{fam} ({dkey})", "mentioned": len(pos), "fire_if_mentioned": fp / len(pos) if pos else None,
                      "not_mentioned": len(neg), "fire_if_not": fn / len(neg) if neg else None})
    print(f"rebuilt {len(rows)} cells; design_hash matches winners.jsonl on {match}")
    print("| family | critiques naming it | detector fires there | other critiques | detector fires there | gap |")
    print("|---|---|---|---|---|---|")
    for t in table:
        a, b = t["fire_if_mentioned"], t["fire_if_not"]
        gap = (a - b) if a is not None and b is not None else None
        f = lambda x: "n/a" if x is None else f"{100 * x:.0f}%"
        print(f"| {t['family']} | {t['mentioned']} | {f(a)} | {t['not_mentioned']} | {f(b)} | "
              f"{'n/a' if gap is None else f'{100 * gap:+.0f} pts'} |")
    return table


def rates(names: list[str]) -> dict[str, dict]:
    out = {}
    for name in names:
        rows = load_cache(name)
        det = detect_cache(rows)
        n = len(det)
        fam = {k: sum(d[k]["fire"] for d in det.values()) / n for k in next(iter(det.values()))} if n else {}
        bis = sum(r["candidate_id"].startswith("bisection") for r in rows.values()) / n if n else 0
        secs = sum(r["seconds"] for r in rows.values()) / n if n else 0
        out[name] = {"cells": n, "bisection_pick_share": bis, "mean_seconds": secs, "rates": fam}
    fams = sorted({k for o in out.values() for k in o["rates"]})
    print("| family | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in fams:
        print(f"| {k} | " + " | ".join(f"{100 * out[n]['rates'].get(k, 0):.0f}%" for n in names) + " |")
    for k in ("cells", "bisection_pick_share", "mean_seconds"):
        print(f"| {k} | " + " | ".join(f"{out[n][k]:.2f}" if isinstance(out[n][k], float) else str(out[n][k])
                                     for n in names) + " |")
    return out


def contact_sheet(names: list[str], out: Path, cells: list[str] | None = None, n: int = 16, seed: int = 0,
                  tile: int = 300) -> Path:
    """original | <name 1> | <name 2> ... for `cells` (default: n cells, seeded, whose
    designs differ between the first two caches). Each row is labelled with
    the cell and each tile with its candidate id and the detectors that fire."""
    import random
    from PIL import Image, ImageDraw
    from dev.distill.assets import merged_assets_dir
    from engine.pipeline.render import render_to_image

    caches = [load_cache(nm) for nm in names]
    common = sorted(set.intersection(*(set(c) for c in caches)))
    if cells is None:
        diff = [c for c in common if len({caches[i][c]["design_hash"] for i in range(len(caches))}) > 1]
        cells = sorted(random.Random(seed).sample(diff, min(n, len(diff))))
    dets = [detect_cache({c: cache[c] for c in cells}) for cache in caches]
    srcs = _sources(cells)
    label_h = 30
    rows = []
    with merged_assets_dir() as assets:
        for c in cells:
            tid = caches[0][c]["task_id"]
            imgs = [("original", render_to_image(srcs[tid], assets_dir=assets, max_size=tile))]
            for nm, cache, det in zip(names, caches, dets):
                fired = ",".join(k for k, v in det[c].items() if v["fire"] and k not in ("widow_any", "floating", "cta_big", "logo_big", "decor_edge"))
                imgs.append((f"{nm}: {cache[c]['candidate_id'][:34]}\n{fired}",
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
    r = sub.add_parser("rebuild")
    r.add_argument("--name", required=True)
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--split", default="judged", choices=["judged", "train", "holdout", "challenge"])
    r.add_argument("--cells", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=4)
    v = sub.add_parser("validate")
    v.add_argument("--name", default="bisect")
    s = sub.add_parser("rates")
    s.add_argument("--name", action="append", required=True)
    c = sub.add_parser("sheet")
    c.add_argument("--name", action="append", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--cell", action="append", default=None)
    c.add_argument("--n", type=int, default=16)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--tile", type=int, default=300)
    args = ap.parse_args(argv)
    if args.step == "sheet":
        print(contact_sheet(args.name, args.out, args.cell, args.n, args.seed, args.tile))
        return 0
    if args.step == "rebuild":
        from dev.distill.ab_engine import parse_env
        print(rebuild(args.name, parse_env(args.env), args.split, args.cells, args.seed, args.workers))
    elif args.step == "validate":
        validate(args.name)
    else:
        rates(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
