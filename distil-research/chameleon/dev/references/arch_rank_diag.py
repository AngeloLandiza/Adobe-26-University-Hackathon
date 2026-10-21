import json, os, sys
from collections import Counter
os.environ.update({"CHAMELEON_STRATEGIES": "bisection", "CHAMELEON_LLM": "heuristic",
    "CHAMELEON_EXTRA_TEMPLATES": sys.argv[1]})
from dev.distill import ab_engine as ab
os.environ["CHAMELEON_ASSETS_DIR"] = str(ab._assets())
from dev.distill.corpus import load_corpus
from engine.resize import resize_with_report
targets = {t["name"]: t for t in json.loads(ab.TARGETS_PATH.read_text())}
designs = {d["task_id"]: d for d in load_corpus("holdout")}
cells = ab.pool_cells(ab.POOL_DIR, "holdout")
st = Counter(); ranks = []; gates = Counter()
for c in cells[::3]:
    t = targets[c["target"]]
    _, _, best, reps = resize_with_report(designs[c["task_id"]]["source"], {"width": t["width"], "height": t["height"]})
    arch = [r for r in reps if "arch_" in r.candidate_id]
    st["cells"] += 1; st["cells_with_arch"] += bool(arch)
    st["arch_cands"] += len(arch); st["arch_pass"] += sum(r.passed_gates for r in arch)
    for r in arch:
        if not r.passed_gates: gates.update(r.failed_gates)
    passing = sorted([r for r in reps if r.passed_gates], key=lambda r: -r.total)
    ar = [i for i, r in enumerate(passing) if "arch_" in r.candidate_id]
    if ar: ranks.append((ar[0] + 1, len(passing)))
print(dict(st)); print("gate fails:", gates.most_common(6))
print("best arch rank / passing (first 25):", ranks[:25])
print("median best arch rank:", sorted(r for r, _ in ranks)[len(ranks)//2] if ranks else None)
