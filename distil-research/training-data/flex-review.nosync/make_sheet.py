"""Before/after contact sheet for flex. Run from distil-research/chameleon:
    python <this> OUT_DIR [template-filter]
Per sample x target: column 1 = best pick flag off, 2 = best pick flag on,
3 = the flag-on winner's candidate id built with the flag off, 4 = same id flag on."""

import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.getcwd())
from engine import resize as R  # noqa: E402
from engine.pipeline.layout.generate import generate_candidates  # noqa: E402
from engine.pipeline.render import render_to_image  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = {t["name"]: t for t in json.loads(Path("dev/targets.json").read_text())}
NAMES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["story", "instagram_post", "medium_rectangle", "leaderboard"]
TILE = 300


def run(task, target, flag):
    if flag:
        os.environ["CHAMELEON_FLEX"] = "1"
    else:
        os.environ.pop("CHAMELEON_FLEX", None)
    design, plan, best, reports = R.resize_with_report(task["source"], target)
    cands = {c.candidate_id: c for c in generate_candidates(task["source"], target, plan, R._get_measurer())}
    os.environ.pop("CHAMELEON_FLEX", None)
    return best, cands, {r.candidate_id: r.total for r in reports}


def tile(design, label):
    img = render_to_image(design, max_size=TILE)
    img.thumbnail((TILE, TILE))
    out = Image.new("RGB", (TILE, TILE + 28), "white")
    out.paste(img, ((TILE - img.width) // 2, 28))
    d = ImageDraw.Draw(out)
    for i, line in enumerate([label[:52], label[52:104]]):
        d.text((3, 2 + 12 * i), line, fill="black")
    return out


rows = []
for path in sorted(Path("input").glob("*.json")):
    task = json.loads(path.read_text())
    for name in NAMES:
        t = TARGETS[name]
        target = {"width": t["width"], "height": t["height"]}
        b_off, c_off, s_off = run(task, target, False)
        b_on, c_on, s_on = run(task, target, True)
        lvl = path.stem.split("_")[2]
        tiles = [tile(b_off.design, f"OFF best {lvl} {name} {s_off.get(b_off.candidate_id, 0):.3f} {b_off.candidate_id}"),
                 tile(b_on.design, f"ON best {s_on.get(b_on.candidate_id, 0):.3f} {b_on.candidate_id}")]
        cid = b_on.candidate_id if b_on.candidate_id in c_off and b_on.candidate_id != "reflow" else None
        if cid is None:
            cid = next((k for k in c_on if k != "reflow" and k in c_off), None)
        if cid:
            tiles += [tile(c_off[cid].design, f"OFF same id {cid}"), tile(c_on[cid].design, f"ON same id {cid}")]
        row = Image.new("RGB", (4 * (TILE + 6), TILE + 34), (200, 200, 200))
        for i, im in enumerate(tiles):
            row.paste(im, (i * (TILE + 6) + 3, 3))
        rows.append((f"{lvl}_{name}", row))
        print(lvl, name, "off", b_off.candidate_id, round(s_off.get(b_off.candidate_id, 0), 3),
              "on", b_on.candidate_id, round(s_on.get(b_on.candidate_id, 0), 3), flush=True)

for i in range(0, len(rows), 4):
    chunk = rows[i:i + 4]
    sheet = Image.new("RGB", (chunk[0][1].width, sum(r.height for _, r in chunk)), (120, 120, 120))
    y = 0
    for _, r in chunk:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(OUT / f"sheet_{i // 4}.png")
for name, r in rows:
    r.save(OUT / f"row_{name}.png")

# Banners at full size: per sample, off best / on best / off same id / on same id, stacked.
BANNERS = [n for n in ("leaderboard", "billboard", "linkedin_banner", "mobile_banner") if n in TARGETS]
for name in BANNERS:
    t = TARGETS[name]
    target = {"width": t["width"], "height": t["height"]}
    strips = []
    for path in sorted(Path("input").glob("*.json")):
        task = json.loads(path.read_text())
        b_off, c_off, s_off = run(task, target, False)
        b_on, c_on, s_on = run(task, target, True)
        cid = b_on.candidate_id if b_on.candidate_id in c_off else next((k for k in c_on if k in c_off and k != "reflow"), None)
        items = [("OFF best " + b_off.candidate_id, b_off.design), ("ON best " + b_on.candidate_id, b_on.design)]
        if cid:
            items += [("OFF same " + cid, c_off[cid].design), ("ON same " + cid, c_on[cid].design)]
        for label, design in items:
            img = render_to_image(design, max_size=800)
            strip = Image.new("RGB", (max(img.width, 400), img.height + 14), "white")
            ImageDraw.Draw(strip).text((3, 1), f"{path.stem.split('_')[2]} {label}", fill="black")
            strip.paste(img, (0, 14))
            strips.append(strip)
    sheet = Image.new("RGB", (max(s.width for s in strips), sum(s.height + 4 for s in strips)), (120, 120, 120))
    y = 0
    for s in strips:
        sheet.paste(s, (0, y))
        y += s.height + 4
    sheet.save(OUT / f"banner_{name}.png")
