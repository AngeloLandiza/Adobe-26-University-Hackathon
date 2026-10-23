import json, os, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, os.getcwd())

def cell(args):
    path, t, flag = args
    if flag: os.environ["CHAMELEON_FLEX"] = "1"
    else: os.environ.pop("CHAMELEON_FLEX", None)
    from engine import resize as R
    from tests.test_flex import proxy_metrics
    task = json.loads(Path(path).read_text())
    tg = {"width": t["width"], "height": t["height"]}
    design, plan, best, reports = R.resize_with_report(task["source"], tg)
    score = next((r.total for r in reports if r.candidate_id == best.candidate_id), 0.0)
    return flag, path, t["name"], best.candidate_id, score, proxy_metrics(design, plan, tg)

if __name__ == "__main__":
    T = json.loads(Path("dev/targets.json").read_text())
    jobs = [(str(p), t, f) for f in (0, 1) for p in sorted(Path("input").glob("*.json")) for t in T]
    with Pool(4) as pool:
        rows = pool.map(cell, jobs)
    out = {}
    for f in (0, 1):
        rs = [r for r in rows if r[0] == f]
        out[f] = {k: sum(r[5][k] for r in rs) / len(rs) for k in rs[0][5]}
        out[f]["grader_total"] = sum(r[4] for r in rs) / len(rs)
        out[f]["changed_pick"] = sum(1 for a, b in zip([r for r in rows if r[0] == 0], rs) if a[3] != b[3])
    print(json.dumps(out, indent=1))
    Path(sys.argv[1]).write_text(json.dumps(rows, default=str, indent=1))
