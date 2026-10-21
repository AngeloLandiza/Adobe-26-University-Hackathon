"""Run JSON inputs through two engine configurations and show them side by side:
original | variant A | variant B, plus each output JSON. For eyeballing a change.

    python -m dev.distill.compare_variants --out DIR --task input/x.json [--target 728x90] ...
        [--a "base"] [--b "CHAMELEON_STRATEGIES=bisection CHAMELEON_RETRIEVAL=1 CHAMELEON_PICKER=rules"]

Each --task may be followed by --target WxH (several allowed); without one the
task's own target_canvas is used. Images resolve through the merged brand-kit +
synthetic folder (CHAMELEON_ASSETS_DIR), like the A/B harness.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _run_one(task_path: str, target: dict, env_spec: str, out_json: Path, assets: str) -> None:
    env = {**os.environ, "CHAMELEON_LLM": "heuristic", "CHAMELEON_ASSETS_DIR": assets}
    for kv in env_spec.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    code = ("import json,sys; from engine.resize import resize_with_report; t=json.load(open(sys.argv[1]));"
            "s=t.get('source', t); d,_,b,_=resize_with_report(s, json.loads(sys.argv[2]));"
            "json.dump({'design': d, 'candidate_id': b.candidate_id}, open(sys.argv[3], 'w'))")
    subprocess.run([sys.executable, "-c", code, task_path, json.dumps(target), str(out_json)],
                   env=env, check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--a", default="", help="env for variant A (space separated KEY=VAL); empty = today's engine")
    ap.add_argument("--b", default="CHAMELEON_STRATEGIES=bisection CHAMELEON_RETRIEVAL=1 CHAMELEON_PICKER=rules")
    ap.add_argument("--task", action="append", default=[])
    ap.add_argument("--target", action="append", default=[], help="WxH, applies to the preceding --task")
    args, rest = ap.parse_known_args()
    # pair each --task with the --target values that follow it on the command line
    jobs, argv = [], sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--task":
            jobs.append([argv[i + 1], []])
        elif a == "--target" and jobs:
            w, h = argv[i + 1].lower().split("x")
            jobs[-1][1].append({"width": int(w), "height": int(h)})
    from dev.distill.assets import merged_assets_dir
    from engine.pipeline.render import render_to_image
    args.out.mkdir(parents=True, exist_ok=True)
    font, small = ImageFont.load_default(size=15), ImageFont.load_default(size=12)
    rows = []
    with merged_assets_dir() as assets:
        for task_path, targets in jobs:
            task = json.loads(Path(task_path).read_text())
            source = task.get("source", task)
            for target in targets or [task["target_canvas"]]:
                name = f"{Path(task_path).stem}@{target['width']}x{target['height']}"
                panels = [("original", render_to_image(source, assets_dir=assets, max_size=600))]
                for label, spec in (("A: " + (args.a or "today's engine"), args.a), ("B: combined", args.b)):
                    out_json = args.out / f"{name}.{'a' if label.startswith('A') else 'b'}.json"
                    _run_one(task_path, target, spec, out_json, str(assets))
                    res = json.loads(out_json.read_text())
                    panels.append((f"{label[:2]} {res['candidate_id'][:46]}",
                                   render_to_image(res["design"], assets_dir=assets, max_size=600)))
                for _, im in panels:
                    im.thumbnail((380, 330))
                W = sum(im.width for _, im in panels) + 48
                H = max(im.height for _, im in panels) + 46
                row = Image.new("RGB", (W, H), "white")
                d = ImageDraw.Draw(row)
                d.text((8, 4), name, fill="black", font=font)
                x = 8
                for lbl, im in panels:
                    d.text((x, 24), lbl, fill=(90, 90, 90), font=small)
                    row.paste(im, (x, 42))
                    x += im.width + 16
                rows.append(row)
                print("done", name, flush=True)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 8 for r in rows)), (225, 225, 225))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 8
    sheet.save(args.out / "sheet.png")
    print(args.out / "sheet.png")


if __name__ == "__main__":
    main()
