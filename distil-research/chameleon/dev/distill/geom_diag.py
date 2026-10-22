import json, os, statistics as st
os.environ.update({"CHAMELEON_STRATEGIES":"bisection,anchored","CHAMELEON_BISECT_V2":"1","CHAMELEON_HIER_LOCK":"1","CHAMELEON_NEAR_FIT":"1","CHAMELEON_LAYERS":"1","CHAMELEON_TEXT_SLACK":"1","CHAMELEON_LLM":"heuristic"})
from dev.distill import ab_engine as ab
os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
from dev.distill.corpus import load_corpus
from dev.distill.llm_layout import execute, _compact_to_elements
from dev.distill import oracle_plan as op
from engine.pipeline.priority import plan as priority_plan
from engine.resize import resize_with_report, _get_measurer
m = _get_measurer()
designs = {d["task_id"]: d for d in load_corpus(None)}
targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
rows = [json.loads(l) for l in open(ab.DATA_ROOT / "llm-distill/qc.nosync/gen_holdout.jsonl")]
rows = [r for r in rows if r["ok"]]

def metrics(src, d, W, H, roles):
    els = d["elements"]; S = min(W, H); A = W * H
    texts = [e for e in els if e["type"] == "text"]
    head = [e for e in texts if roles.get(e["id"]) == "headline"]
    hf = max((e["font_size"] for e in head), default=max((e["font_size"] for e in texts), default=0))
    cta = [e for e in els if roles.get(e["id"]) == "cta"]
    imgs = [e for e in els if e["type"] == "image" and roles.get(e["id"]) not in ("background", "decoration") and e["width"] * e["height"] < 0.9 * A]
    content = [e for e in els if roles.get(e["id"]) not in ("background", "decoration") and e["width"] * e["height"] < 0.9 * A]
    x0 = min(e["x"] for e in content); y0 = min(e["y"] for e in content); x1 = max(e["x"] + e["width"] for e in content); y1 = max(e["y"] + e["height"] for e in content)
    return {"headline_font/S": hf / S, "hero_area": max((e["width"] * e["height"] for e in imgs), default=0) / A,
            "text_area": sum(e["width"] * e["height"] for e in texts) / A, "bbox_fill": (x1 - x0) * (y1 - y0) / A,
            "margin_min/S": min(x0, y0, W - x1, H - y1) / S, "margin_max/S": max(x0, y0, W - x1, H - y1) / S,
            "cta_h/S": max((e["height"] for e in cta), default=0) / S}

out = {"sonnet": [], "plan+solver": [], "v5": []}
for r in rows:
    src = designs[r["task_id"]]["source"]; t = targets[r["target"]]; W, H = t["width"], t["height"]; tgt = {"width": W, "height": H}
    plan = priority_plan(src); roles = {eid: u.role for u in plan.ranked_units() for eid in u.element_ids}
    try:
        son = execute(src, tgt, plan, m, _compact_to_elements(json.loads(r["answer"])["e"]))
        P = op.extract(src, tgt, son, plan, m); sol, _ = op.realize(src, tgt, plan, P, m)
        v5 = resize_with_report(src, tgt)[0]
    except Exception as e:
        continue
    if sol is None: continue
    for k, d in [("sonnet", son), ("plan+solver", sol), ("v5", v5)]:
        out[k].append(metrics(src, d, W, H, roles))
n = len(out["sonnet"]); print("cells", n)
keys = list(out["sonnet"][0])
print(f'{"metric":18s} {"sonnet":>9s} {"plan+sol":>9s} {"v5":>9s}   plan+sol/sonnet (median ratio)')
for k in keys:
    s = [x[k] for x in out["sonnet"]]; p = [x[k] for x in out["plan+solver"]]; v = [x[k] for x in out["v5"]]
    ratios = [b / a for a, b in zip(s, p) if a > 1e-6]
    print(f"{k:18s} {st.median(s):9.3f} {st.median(p):9.3f} {st.median(v):9.3f}   {st.median(ratios) if ratios else float('nan'):.2f}")
