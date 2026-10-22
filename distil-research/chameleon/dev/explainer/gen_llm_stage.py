"""Demo stage from LLM layouts: teacher (Sonnet, lean) or student outputs file. Gate failures fall back to v5.
usage: gen_llm_stage.py teacher <O>            -> stage_teacher.json
       gen_llm_stage.py student <O> <outputs>  -> stage_student.json"""
import base64, io, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dev.distill import ab_engine as ab
os.environ["CHAMELEON_LLM"] = "heuristic"; os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
from dev.distill.corpus import load_corpus
from dev.distill.llm_layout import GEN_SCHEMA, _compact_to_elements, _gen_prompt, _parse_json, execute
from dev.distill.teacher import ClaudeCodeTeacher
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.priority import plan as priority_plan
from engine.pipeline.render import render_to_image
from engine.resize import _get_measurer

mode, O = sys.argv[1], Path(sys.argv[2])
EX = json.loads((O / "examples.json").read_text()); SZ = json.loads((O / "sizes.json").read_text())
designs = {d["task_id"]: d for d in load_corpus(None) + load_corpus("samples")}
v5 = json.loads((O / "stage_v5.json").read_text())
m = _get_measurer()

def jpeg(img, q=72, max_side=520):
    img = img.convert("RGB"); img.thumbnail((max_side, max_side))
    b = io.BytesIO(); img.save(b, "JPEG", quality=q, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

plans = {}
if mode == "teacher":
    t = ClaudeCodeTeacher("claude-sonnet-5-5", effort="low")
    def ask(ex):
        tl = [{"name": s["id"], "width": s["w"], "height": s["h"]} for s in SZ]
        return ex["id"], t.ask(_gen_prompt(designs[ex["task_id"]]["source"], tl), GEN_SCHEMA)
    with ThreadPoolExecutor(4) as pool:
        for xid, ans in pool.map(ask, EX):
            for l in ans.get("layouts", []):
                plans[f'{xid}@{l["t"]}'] = _compact_to_elements(l["e"])
    print("teacher cost $", round(sum(c["cost_usd"] for c in t.calls), 2))
else:
    for r in map(json.loads, open(sys.argv[3])):
        try:
            a = _parse_json(r["output"]); plans[r["cell"]] = _compact_to_elements(a["e"]) if "e" in a else a
        except Exception:
            pass
res, fb = {}, 0
for ex in EX:
    src = designs[ex["task_id"]]["source"]; plan = priority_plan(src)
    for s in SZ:
        k = f'{ex["id"]}@{s["id"]}'; tgt = {"width": s["w"], "height": s["h"]}
        try:
            d = execute(src, tgt, plan, m, plans[k]); failed = check_gates(src, d, tgt, plan, m)
        except Exception as e:
            d, failed = None, [type(e).__name__]
        if failed:
            fb += 1; res[k] = {**v5[k], "cand": "fallback-v5 (" + ",".join(failed) + ")", "fallback": True}
        else:
            res[k] = {"design": d, "cand": mode, "fallback": False, "png": jpeg(render_to_image(d, assets_dir=ab._assets(), max_size=1100))}
(O / f"stage_{mode}.json").write_text(json.dumps(res))
print(mode, len(res), "cells,", fb, "fell back to v5")
