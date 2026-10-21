"""How fragile is the engine's text wrapping? (the CHAMELEON_TEXT_SLACK study)

The official converter (renderer/converter/wrap.py) word-wraps greedily at
the box width with skia glyph widths, and the engine sizes many text boxes to
exactly their widest line. Any tiny difference at scoring time (rounding, a
different skia build, font rasterisation on the reference machine) then
re-wraps the box, and a CTA label that re-wraps spills its second line out of
its pill. This tool runs the engine on a set of cells and counts, on each
winning design:

- `fragile_1px`: text elements whose official line count changes when the
  box is 1 px narrower;
- `fragile_slack`: text elements whose wrap changes when the box is
  `slack_px(width)` narrower (max(1.5 px, 1.5% of the width): the margin
  CHAMELEON_TEXT_SLACK guarantees);
- pill labels (a text in a container_label unit, or a text lying inside a
  small shape of its own group): `pill_multiline` (more than one line),
  `pill_over_src` (more lines than in the source), `pill_outside` (ink
  outside the pill's box), `pill_fragile_1px`;
- `short_boxes`: text boxes shorter than the text's drawn block.

    python -m dev.distill.text_robustness run --name v4 --split samples --env K=V ... [--cells N] [--workers 3]
    python -m dev.distill.text_robustness report --names base,v4,v4ts [--split samples]
    python -m dev.distill.text_robustness sheet --base v4 --name v4ts --split samples --out F.png [--cell C ...]

`run` writes training-data/text-robustness.nosync/<name>_<split>/rows.jsonl
(one row per cell: winner id, design, analysis; resumable). `report` prints
the counts per run and, against the first name, how many designs and picks
changed and the mean font-size change of the text elements.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dev.distill.paths import DATA_ROOT

from engine.pipeline.layout.text_slack import TOL, _box, pill_pairs, slack_px  # noqa: F401  (the pass's own definitions)

ROOT = DATA_ROOT / "text-robustness.nosync"


# ---------------------------------------------------------------- analysis

def ink_box(el: dict, m) -> tuple[float, float, float, float]:
    """Where the text is drawn: widest line placed by text_align, the block height."""
    widths = m.line_widths(el, el["font_size"], el["width"]) or [0.0]
    w = max(widths)
    align = el.get("text_align") or "left"
    x = el["x"] + (el["width"] - w) / 2 if align == "center" else \
        el["x"] + el["width"] - w if align == "right" else el["x"]
    return (x, el["y"], w, m.block_height(el, el["font_size"], el["width"]))


def analyze(source: dict, design: dict, plan, m) -> dict:
    src = {e["id"]: e for e in source["elements"]}
    els = {e["id"]: e for e in design["elements"]}
    pills = pill_pairs(source, plan)
    out = {"texts": 0, "fragile_1px": [], "fragile_slack": [], "short_boxes": [], "pills": 0,
           "pill_multiline": [], "pill_over_src": [], "pill_outside": [], "pill_fragile_1px": []}
    for el in design["elements"]:
        if el["type"] != "text" or not str(el.get("content", "")).strip():
            continue
        out["texts"] += 1
        f, w = el["font_size"], el["width"]
        wrap = m.wrap(el, f, w)
        lines = wrap.count("\n") + 1
        frag1 = m.line_count(el, f, w - 1) != lines
        if frag1:
            out["fragile_1px"].append(el["id"])
        if m.wrap(el, f, w - slack_px(w)) != wrap:
            out["fragile_slack"].append(el["id"])
        if el["height"] < m.block_height(el, f, w) - 0.01:
            out["short_boxes"].append(el["id"])
        sid = pills.get(el["id"])
        if sid is None or sid not in els:
            continue
        out["pills"] += 1
        s0 = src[el["id"]]
        src_lines = m.line_count(s0, s0["font_size"], s0["width"])
        if lines > 1:
            out["pill_multiline"].append(el["id"])
        if lines > src_lines:
            out["pill_over_src"].append(el["id"])
        ink, shape = ink_box(el, m), _box(els[sid])
        if not (ink[0] >= shape[0] - TOL and ink[1] >= shape[1] - TOL and ink[0] + ink[2] <= shape[0] + shape[2] + TOL
                and ink[1] + ink[3] <= shape[1] + shape[3] + TOL):
            out["pill_outside"].append(el["id"])
        if frag1:
            out["pill_fragile_1px"].append(el["id"])
    return out


# ---------------------------------------------------------------- run

def run_dir(name: str, split: str, root: Path = ROOT) -> Path:
    return Path(root) / f"{name}_{split}"


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def run_cell(job: dict) -> dict:
    from engine.resize import _get_measurer, resize_with_report
    try:
        source, target = job["design"]["source"], job["target"]
        t0 = time.perf_counter()
        design, plan, best, _ = resize_with_report(source, target)
        seconds = time.perf_counter() - t0
        return {"cell": job["cell"]["cell"], "task_id": job["cell"]["task_id"], "target": job["cell"]["target"],
                "candidate_id": best.candidate_id, "seconds": round(seconds, 3),
                "analysis": analyze(source, design, plan, _get_measurer()), "design": design}
    except Exception as exc:  # one bad cell must not sink the run
        return {"cell": job["cell"]["cell"], "error": f"{type(exc).__name__}: {exc}"}


def run(name: str, split: str, env: dict, n: int | None = None, seed: int = 0, workers: int = 3,
        root: Path = ROOT, log=print) -> int:
    from dev.distill import ab_engine as ab
    cells, designs, folders = ab.split_cells(split, n, seed)
    targets = {t["name"]: {"width": t["width"], "height": t["height"]} for t in json.loads(ab.TARGETS_PATH.read_text())}
    out_dir = run_dir(name, split, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "variant.json").write_text(json.dumps({"name": name, "split": split, "env": env, "cells": n}, indent=2))
    path = out_dir / "rows.jsonl"
    done = {r["cell"] for r in _rows(path)}
    jobs = [{"cell": c, "design": designs[c["task_id"]], "target": targets[c["target"]]}
            for c in cells if c["cell"] not in done and c["task_id"] in designs]
    added = 0
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 3)), initializer=ab._worker_init,
                             initargs=(env, folders)) as ex, path.open("a") as f:
        for row in ex.map(run_cell, jobs):
            if "error" in row:
                log(f"[robust] {row['cell']}: {row['error']}")
                continue
            f.write(json.dumps(row) + "\n")
            added += 1
    log(f"[robust] {name}/{split}: {added} new cells ({len(done) + added} total) -> {path}")
    return added


# ---------------------------------------------------------------- report

COUNTS = ("fragile_1px", "fragile_slack", "short_boxes", "pill_multiline", "pill_over_src", "pill_outside",
          "pill_fragile_1px")


def summarize(rows: list[dict]) -> dict:
    s = {"cells": len(rows), "texts": sum(r["analysis"]["texts"] for r in rows),
         "pills": sum(r["analysis"]["pills"] for r in rows)}
    for k in COUNTS:
        s[k] = sum(len(r["analysis"][k]) for r in rows)
    return s


def change_kinds(a: dict, b: dict, m) -> dict:
    """Per element changed between two designs of the same pick: a text box
    whose ink stays put (only its box moved: invisible), a text drawn
    differently (font size, wrap or ink position), or another element (a
    pill widened)."""
    ea = {e["id"]: e for e in a["elements"]}
    out = {"box_only": 0, "text_visible": 0, "other": 0}
    for e in b["elements"]:
        o = ea.get(e["id"])
        if o is None or o == e:
            continue
        if e["type"] != "text":
            out["other"] += 1
            continue
        same = (o["font_size"] == e["font_size"] and m.wrap(o, o["font_size"], o["width"]) == m.wrap(e, e["font_size"], e["width"])
                and all(abs(p - q) <= 0.02 for p, q in zip(ink_box(o, m), ink_box(e, m))))
        out["box_only" if same else "text_visible"] += 1
    return out


def compare(base: list[dict], other: list[dict]) -> dict:
    """Cells in both: designs changed, picks (candidate ids) changed, and, on
    cells with the same pick, the font-size change of text elements (mean
    over all their texts and over the resized ones)."""
    b = {r["cell"]: r for r in base}
    common = [r for r in other if r["cell"] in b]
    from engine.pipeline.textmeasure import TextMeasurer
    m = TextMeasurer()
    changed = picks = 0
    ratios, moved = [], []
    kinds = {"box_only": 0, "text_visible": 0, "other": 0}
    for r in common:
        a = b[r["cell"]]
        if json.dumps(a["design"], sort_keys=True) != json.dumps(r["design"], sort_keys=True):
            changed += 1
        if a["candidate_id"] != r["candidate_id"]:
            picks += 1
            continue  # font change of the same pick only: another candidate's sizes say nothing about the pass
        for k, v in change_kinds(a["design"], r["design"], m).items():
            kinds[k] += v
        fa = {e["id"]: e["font_size"] for e in a["design"]["elements"] if e["type"] == "text"}
        for e in r["design"]["elements"]:
            if e["type"] == "text" and fa.get(e["id"]):
                q = e["font_size"] / fa[e["id"]] - 1
                ratios.append(q)
                if abs(q) > 1e-9:
                    moved.append(q)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"common": len(common), "designs_changed": changed, "picks_changed": picks,
            "font_mean_pct": 100 * mean(ratios), "font_changed_texts": len(moved),
            "font_mean_changed_pct": 100 * mean(moved), "font_min_pct": 100 * min(moved, default=0.0), **kinds}


def report(names: list[str], split: str, root: Path = ROOT) -> str:
    runs = {n: _rows(run_dir(n, split, root) / "rows.jsonl") for n in names}
    lines = [f"split {split}", "| run | cells | texts | fragile 1px | fragile slack | short boxes | pills | "
             "pill >1 line | pill > src lines | pill outside | pill fragile 1px |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    pct = lambda k, t: f"{k} ({100 * k / t:.1f}%)" if t else "0"
    for n, rows in runs.items():
        s = summarize(rows)
        lines.append(f"| {n} | {s['cells']} | {s['texts']} | {pct(s['fragile_1px'], s['texts'])} | "
                     f"{pct(s['fragile_slack'], s['texts'])} | {s['short_boxes']} | {s['pills']} | "
                     f"{s['pill_multiline']} | {s['pill_over_src']} | {s['pill_outside']} | {s['pill_fragile_1px']} |")
    base = names[0]
    for n in names[1:]:
        c = compare(runs[base], runs[n])
        lines.append(f"{n} vs {base}: {c['designs_changed']}/{c['common']} designs changed, "
                     f"{c['picks_changed']} picks changed; same-pick cells: font size mean {c['font_mean_pct']:+.2f}% over all texts, "
                     f"{c['font_changed_texts']} texts resized (mean {c['font_mean_changed_pct']:+.2f}%, "
                     f"min {c['font_min_pct']:+.2f}%); elements changed: {c['box_only']} text boxes with the ink in place, "
                     f"{c['text_visible']} texts drawn differently, {c['other']} other (pills)")
    return "\n".join(lines)


# ---------------------------------------------------------------- contact sheet

def sheet(base: str, other: str, split: str, out: Path, n: int = 12, must: tuple[str, ...] = (),
          root: Path = ROOT, size: int = 360) -> list[str]:
    """source | base | other for `must` cells, then the changed cells with the
    most texts drawn differently, up to n. Text boxes are outlined (base red,
    other green); each tile says whether the two renders differ in pixels."""
    from PIL import Image, ImageChops, ImageDraw

    from dev.distill import ab_engine as ab
    from dev.distill.assets import merged_assets_dir
    from dev.distill.corpus import load_corpus
    from engine.pipeline.render import render_to_image
    from engine.pipeline.textmeasure import TextMeasurer

    m = TextMeasurer()
    ra = {r["cell"]: r for r in _rows(run_dir(base, split, root) / "rows.jsonl")}
    rb = {r["cell"]: r for r in _rows(run_dir(other, split, root) / "rows.jsonl")}
    ads = {d["task_id"]: d for d in load_corpus(split)}
    changed = []
    for c, r in rb.items():
        a = ra.get(c)
        if a is None or a["design"] == r["design"]:
            continue
        k = change_kinds(a["design"], r["design"], m)
        changed.append((-(k["text_visible"] + k["other"]), -k["box_only"], c))
    cells = [c for c in must if c in rb] + [c for *_, c in sorted(changed) if c not in must]
    cells = cells[:max(n, len(must))]

    def tile(design: dict, label: str, assets, outline: str | None) -> Image.Image:
        img = render_to_image(design, assets_dir=assets, max_size=size)
        k = img.width / design["canvas"]["width"]
        if outline:
            d = ImageDraw.Draw(img)
            for e in design["elements"]:
                if e["type"] == "text":
                    d.rectangle([e["x"] * k, e["y"] * k, (e["x"] + e["width"]) * k, (e["y"] + e["height"]) * k],
                                outline=outline)
        out = Image.new("RGB", (size, size + 16), "white")
        out.paste(img, ((size - img.width) // 2, 16 + (size - img.height) // 2))
        ImageDraw.Draw(out).text((3, 2), label[:58], fill="black")
        return out

    rows_img = []
    with merged_assets_dir(*ab.folders_for(split)) as assets:
        for c in cells:
            a, b = ra[c], rb[c]
            src = ads[a["task_id"]]["source"]
            ia = render_to_image(a["design"], assets_dir=assets, max_size=size)
            ib = render_to_image(b["design"], assets_dir=assets, max_size=size)
            diff = ImageChops.difference(ia, ib).convert("L").point(lambda v: 255 if v > 24 else 0)
            px = diff.histogram()[255]
            k = change_kinds(a["design"], b["design"], m)
            note = f"{px} px differ; {k['box_only']} box, {k['text_visible']} text, {k['other']} pill"
            rows_img.append([tile(src, c, assets, None), tile(a["design"], f"{base}  (red: text boxes)", assets, "red"),
                             tile(b["design"], f"{other}: {note}", assets, "lime")])
    sheet_img = Image.new("RGB", (3 * size, len(rows_img) * (size + 16)), (230, 230, 230))
    for i, row in enumerate(rows_img):
        for j, t in enumerate(row):
            sheet_img.paste(t, (j * size, i * (size + 16)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet_img.save(out)
    return cells


def main(argv=None) -> int:
    from dev.distill import ab_engine as ab
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--split", default="samples")
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--cells", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("report")
    p.add_argument("--names", required=True)
    p.add_argument("--split", default="samples")
    sh = sub.add_parser("sheet")
    sh.add_argument("--base", required=True)
    sh.add_argument("--name", required=True)
    sh.add_argument("--split", default="samples")
    sh.add_argument("--out", required=True)
    sh.add_argument("--n", type=int, default=12)
    sh.add_argument("--cell", action="append", default=[], help="always include this cell (repeatable)")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.name, args.split, ab.parse_env(args.env), args.cells, args.seed, args.workers)
    elif args.cmd == "sheet":
        print("\n".join(sheet(args.base, args.name, args.split, Path(args.out), args.n, tuple(args.cell))))
    else:
        print(report([n.strip() for n in args.names.split(",") if n.strip()], args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
