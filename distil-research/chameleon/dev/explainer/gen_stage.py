"""Run one pipeline stage over the demo examples x sizes; dump designs + real renders.
usage: gen_stage.py <stage> <outdir>   (env for the stage is set by the caller)"""
import base64, io, json, os, sys
from pathlib import Path
from dev.distill import ab_engine as ab
os.environ["CHAMELEON_LLM"] = "heuristic"
os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
from dev.distill.corpus import load_corpus
from engine.resize import resize_with_report
from engine.pipeline.render import render_to_image

stage, out = sys.argv[1], Path(sys.argv[2])
EX = json.loads((out / "examples.json").read_text())
SIZES = json.loads((out / "sizes.json").read_text())
designs = {d["task_id"]: d for d in load_corpus(None) + load_corpus("samples")}

def jpeg(img, q=72, max_side=520):
    img = img.convert("RGB"); img.thumbnail((max_side, max_side))
    b = io.BytesIO(); img.save(b, "JPEG", quality=q, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

res = {}
for ex in EX:
    src = designs[ex["task_id"]]["source"]
    for sz in SIZES:
        if stage == "original":
            continue
        d, plan, best, _ = resize_with_report(src, {"width": sz["w"], "height": sz["h"]})
        res[f'{ex["id"]}@{sz["id"]}'] = {"design": d, "cand": best.candidate_id, "fallback": bool(best.is_fallback),
                                          "png": jpeg(render_to_image(d, assets_dir=ab._assets(), max_size=1100))}
    if stage == "original":
        res[ex["id"]] = {"design": src, "png": jpeg(render_to_image(src, assets_dir=ab._assets(), max_size=1100))}
(out / f"stage_{stage}.json").write_text(json.dumps(res))
print(stage, len(res))
