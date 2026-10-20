"""Pilot for the rubric: a person and the teacher judge the same pairs, then we compare.

    python -m dev.distill.pilot make [--n 25]    # sheets + a blank answer CSV in training-data/judgments/pilot/
    python -m dev.distill.pilot judge [--model claude-opus-5-5] [--pairs-per-call 4]
                                                 # the teacher judges exactly those pairs (each also swapped)
    python -m dev.distill.pilot score            # compare human_answers.csv with the teacher's judgments

The teacher's pilot answers go to pilot/teacher_judgments.jsonl, never to the main judgments file.

The person sees only ORIGINAL | A | B sheets, never the teacher's answers.
"""

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from dev.distill.judge import judge_pairs, position_bias, swap_agreement
from dev.distill.paths import DATA_ROOT, POOL_DIR, RENDERS_SUBDIR

PILOT_DIR = DATA_ROOT / "judgments" / "pilot"
MIN_AGREE_SHARE = 18 / 25
BIAS_RANGE = (0.35, 0.65)
MIN_SWAP_AGREEMENT = 0.8
SHEET_HEIGHT = 420
LABEL_HEIGHT = 28
PER_CELL = 2
TEACHER_FILE = "teacher_judgments.jsonl"


def pick_pilot_pairs(pool_rows, n: int = 25, seed: int = 0) -> list[tuple[dict, dict]]:
    """n pairs from train cells only, spread round-robin over the cells (at most PER_CELL each)."""
    from dev.distill.pool import sample_pairs

    train = [r for r in pool_rows if r["split"] == "train"]
    by_cell: dict[str, list] = {}
    for pair in sample_pairs(train, per_cell=PER_CELL, seed=seed):
        by_cell.setdefault(pair[0]["cell"], []).append(pair)
    order = sorted(by_cell)
    random.Random(seed).shuffle(order)
    picked: list[tuple[dict, dict]] = []
    for rank in range(PER_CELL):
        for cell in order:
            if len(picked) < n and rank < len(by_cell[cell]):
                picked.append(by_cell[cell][rank])
    return picked


def judge_pilot(out_dir: Path, pool_rows, teacher, pool_dir: Path, pairs_per_call: int = 4, seed: int = 0) -> list[dict]:
    """Have the teacher judge exactly the pairs in pairs.json, every one also re-asked swapped."""
    out_dir = Path(out_dir)
    by_key = {(r["cell"], r["candidate_id"]): r for r in pool_rows}
    meta = json.loads((out_dir / "pairs.json").read_text())
    pairs = [(by_key[m["cell"], m["a"]], by_key[m["cell"], m["b"]]) for m in meta.values()]
    return judge_pairs(pairs, teacher, lambda t: Path(pool_dir) / RENDERS_SUBDIR / t / "source.png",
                       out_dir / TEACHER_FILE, pairs_per_call, 1.0, seed, pool_dir)


def _panel(path: Path, label: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((max(1, round(img.width * SHEET_HEIGHT / img.height)), SHEET_HEIGHT))
    panel = Image.new("RGB", (img.width, SHEET_HEIGHT + LABEL_HEIGHT), "white")
    panel.paste(img, (0, LABEL_HEIGHT))
    ImageDraw.Draw(panel).text((6, 8), label, fill="black")
    return panel


def make_pilot(pairs, out_dir: Path, pool_dir: Path, seed: int = 0) -> None:
    """Write pairs.json, one labelled sheet per pair and a blank human_answers.csv."""
    out_dir = Path(out_dir)
    (out_dir / "sheets").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    meta = {}
    for i, (x, y) in enumerate(pairs, 1):
        a, b = (x, y) if rng.random() < 0.5 else (y, x)
        pid = f"p{i:02d}"
        meta[pid] = {"cell": a["cell"], "task_id": a["task_id"], "a": a["candidate_id"], "b": b["candidate_id"]}
        panels = [_panel(pool_dir / RENDERS_SUBDIR / a["task_id"] / "source.png", "ORIGINAL"),
                  _panel(pool_dir / a["png"], "A"), _panel(pool_dir / b["png"], "B")]
        sheet = Image.new("RGB", (sum(p.width for p in panels) + 10 * (len(panels) - 1), panels[0].height), "gray")
        x0 = 0
        for p in panels:
            sheet.paste(p, (x0, 0))
            x0 += p.width + 10
        sheet.save(out_dir / "sheets" / f"{pid}.png")
    (out_dir / "pairs.json").write_text(json.dumps(meta, indent=1))
    with (out_dir / "human_answers.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "winner"])
        w.writerows([pid, ""] for pid in meta)


def _winner_id(row: dict) -> str:
    return {"a": row["a"], "b": row["b"]}.get(row["winner"], "tie")


def score_pilot(out_dir: Path, judgments: list[dict], teacher_name: str) -> dict:
    """Compare the person's answers with the teacher's; write RESULTS.md with pass/fail."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "pairs.json").read_text())
    human = {r["pair_id"]: (r.get("winner") or "").strip().lower()  # a short or blank row is "not judged"
             for r in csv.DictReader((out_dir / "human_answers.csv").open()) if r.get("pair_id")}
    mine = [j for j in judgments if j["teacher"] == teacher_name]
    teacher_pick = {(j["cell"], frozenset((j["a"], j["b"]))): _winner_id(j) for j in mine if not j["swapped"]}
    n = agree = 0
    for pid, m in meta.items():
        pick = teacher_pick.get((m["cell"], frozenset((m["a"], m["b"]))))
        if pick is None or human.get(pid) not in ("a", "b", "tie"):
            continue
        n += 1
        agree += pick == _winner_id({"a": m["a"], "b": m["b"], "winner": human[pid]})
    result = {"n": n, "agree": agree, "position_bias": position_bias(mine), "swap_agreement": swap_agreement(mine)}

    bias_ok = BIAS_RANGE[0] <= result["position_bias"] <= BIAS_RANGE[1]
    checks = [(f"agreement >= {MIN_AGREE_SHARE:.0%} of pairs", f"{agree} / {n}", n > 0 and agree / n >= MIN_AGREE_SHARE),
              (f"position bias in [{BIAS_RANGE[0]}, {BIAS_RANGE[1]}]", f"{result['position_bias']:.2f}", bias_ok),
              (f"swap agreement >= {MIN_SWAP_AGREEMENT}", f"{result['swap_agreement']:.2f}",
               result["swap_agreement"] >= MIN_SWAP_AGREEMENT)]
    lines = [f"# Rubric pilot: {teacher_name} vs the human", "", "| check | value | result |", "|---|---|---|"]
    lines += [f"| {c} | {v} | {'PASS' if ok else 'FAIL'} |" for c, v, ok in checks]
    lines += ["", "Rubric accepted." if all(ok for *_, ok in checks) else "Rubric NOT accepted: revise rubric.md and repeat."]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--n", type=int, default=25)
    mk.add_argument("--seed", type=int, default=0)
    jd = sub.add_parser("judge")
    jd.add_argument("--model", default="claude-opus-5-5")
    jd.add_argument("--pairs-per-call", type=int, default=4)
    jd.add_argument("--seed", type=int, default=0)
    sc = sub.add_parser("score")
    sc.add_argument("--teacher", default=None, help="teacher name (default: the one with most judgments)")
    args = ap.parse_args()

    pool_rows = lambda: [json.loads(x) for x in (POOL_DIR / "pool.jsonl").read_text().splitlines() if x.strip()]
    if args.cmd == "make":
        pairs = pick_pilot_pairs(pool_rows(), args.n, args.seed)
        make_pilot(pairs, PILOT_DIR, POOL_DIR, args.seed)
        print(f"wrote {len(pairs)} pairs over {len({a['cell'] for a, _ in pairs})} cells to {PILOT_DIR}; "
              "fill in human_answers.csv (a / b / tie)")
    elif args.cmd == "judge":
        from dev.distill.teacher import ClaudeCodeTeacher
        added = judge_pilot(PILOT_DIR, pool_rows(), ClaudeCodeTeacher(args.model), POOL_DIR, args.pairs_per_call, args.seed)
        print(f"judged {len(added)} new answers -> {PILOT_DIR / TEACHER_FILE}")
    else:
        judgments = [json.loads(x) for x in (PILOT_DIR / TEACHER_FILE).read_text().splitlines() if x.strip()]
        name = args.teacher or max({j["teacher"] for j in judgments},
                                   key=lambda t: sum(j["teacher"] == t for j in judgments))
        print(score_pilot(PILOT_DIR, judgments, name))


if __name__ == "__main__":
    main()
