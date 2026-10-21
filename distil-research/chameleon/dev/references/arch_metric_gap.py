import json, os, sys
from collections import defaultdict
os.environ.update({"CHAMELEON_STRATEGIES": "bisection", "CHAMELEON_LLM": "heuristic", "CHAMELEON_EXTRA_TEMPLATES": sys.argv[1]})
from dev.distill import ab_engine as ab
os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
from dev.distill.corpus import load_corpus
from engine.resize import resize_with_report
from engine.pipeline.grading.metrics import WEIGHTS
targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
designs = {d["task_id"]: d for d in load_corpus("holdout")}
gap = defaultdict(list)
for c in ab.pool_cells(ab.POOL_DIR, "holdout")[::3]:
    t = targets[c["target"]]
    _, _, best, reps = resize_with_report(designs[c["task_id"]]["source"], {"width": t["width"], "height": t["height"]})
    win = next(r for r in reps if r.candidate_id == best.candidate_id)
    arch = [r for r in reps if "arch_" in r.candidate_id and r.passed_gates]
    if not arch or "arch_" in win.candidate_id: continue
    a = max(arch, key=lambda r: r.total)
    for k, w in WEIGHTS.items():
        gap[k].append(w * (win.metrics[k] - a.metrics[k]))
print("weighted metric gap, winner minus best archetype (positive = archetype loses on it), n =", len(gap[next(iter(gap))]))
for k, v in sorted(gap.items(), key=lambda kv: -sum(kv[1])):
    print(f"  {k:28s} weight {WEIGHTS[k]:+.2f}  mean gap {sum(v)/len(v):+.3f}")
