"""Pairwise judge: a teacher model picks the better of two resizes of the same ad.

    python -m dev.distill.judge [--per-cell 6] [--pairs-per-call 4] [--swap-share 0.2]
                                [--split train|holdout|all] [--limit-calls N]
                                [--model claude-opus-5-5] [--seed 0]

Each call shows the original ad plus a few A/B pairs from one cell, and the
official rubric. A and B are shuffled by seed, and a share of pairs is re-asked
in the opposite order so position bias can be measured. Every answered batch is
appended to the output file straight away, and a rerun skips what is done.
"""

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

from dev.distill.paths import JUDGMENTS_FILE, POOL_DIR
from dev.distill.teacher import TeacherUnavailable

RUBRIC = (Path(__file__).parent / "rubric.md").read_text()


def _key(cell: str, x: str, y: str, swapped: bool) -> tuple:
    return cell, frozenset((x, y)), swapped


def _done(out: Path, teacher_name: str) -> set:
    if not out.exists():
        return set()
    rows = (json.loads(x) for x in out.read_text().splitlines() if x.strip())
    return {_key(r["cell"], r["a"], r["b"], r["swapped"]) for r in rows if r["teacher"] == teacher_name}


def _schema(n: int) -> dict:
    verdict = {"type": "object", "required": ["pair", "winner", "confidence", "reason"],
               "properties": {"pair": {"type": "integer"}, "winner": {"enum": ["a", "b", "tie"]},
                              "confidence": {"enum": [1, 2, 3]}, "reason": {"type": "string"}}}
    return {"type": "object", "required": ["verdicts"],
            "properties": {"verdicts": {"type": "array", "items": verdict, "minItems": n, "maxItems": n}}}


def _prompt(n: int) -> str:
    labels = ["Image 1: the ORIGINAL design."]
    for i in range(n):
        labels.append(f"Image {2 + 2 * i}: pair {i + 1}, candidate A. Image {3 + 2 * i}: pair {i + 1}, candidate B.")
    return (f"{RUBRIC}\n\n{' '.join(labels)}\n\nFor each pair, say which candidate (a or b) is the better "
            f"resize, or tie if you truly cannot separate them. Confidence: 1 = a guess, 2 = fairly sure, "
            f"3 = obvious. Give one short reason. Answer with one verdict per pair, numbered 1 to {n}. "
            f"Read all of the listed image files, in parallel in one step, before you answer.")


def _validator(n: int) -> Callable[[dict], None]:
    """Raise if the answer is not exactly one well-formed verdict per pair 1..n."""
    def check(answer: dict) -> None:
        verdicts = answer.get("verdicts") if isinstance(answer, dict) else None
        if not isinstance(verdicts, list):
            raise ValueError("no verdicts list")
        seen = []
        for v in verdicts:
            if not isinstance(v, dict):
                raise ValueError(f"verdict is not an object: {v!r}")
            if type(v.get("pair")) is not int:
                raise ValueError("pair is not an integer")
            if v.get("winner") not in ("a", "b", "tie"):
                raise ValueError(f"bad winner {v.get('winner')!r}")
            if type(v.get("confidence")) is not int or v["confidence"] not in (1, 2, 3):
                raise ValueError(f"bad confidence {v.get('confidence')!r}")
            if not isinstance(v.get("reason"), str):
                raise ValueError("reason is not a string")
            seen.append(v["pair"])
        if sorted(seen) != list(range(1, n + 1)):
            raise ValueError(f"pair numbers {sorted(seen)} do not match 1..{n}")
    return check


def _plan(pairs, swap_share: float, seed: int) -> list[tuple]:
    """(first, second, swapped) items: shuffled originals, then the swapped re-asks."""
    rng = random.Random(seed)
    shown = [(a, b) if rng.random() < 0.5 else (b, a) for a, b in pairs]
    n_swap = round(len(shown) * swap_share)
    picked = sorted(rng.sample(range(len(shown)), n_swap))
    return [(a, b, False) for a, b in shown] + [(shown[i][1], shown[i][0], True) for i in picked]


def judge_pairs(pairs, teacher, source_png: Callable[[str], Path], out: Path = JUDGMENTS_FILE,
                pairs_per_call: int = 4, swap_share: float = 0.2, seed: int = 0,
                pool_dir: Path = POOL_DIR, limit_calls: int | None = None) -> list[dict]:
    """Judge pairs batch by batch (one cell per call); return the rows added to `out`."""
    out = Path(out)
    done = _done(out, teacher.name)
    todo = [it for it in _plan(pairs, swap_share, seed)
            if _key(it[0]["cell"], it[0]["candidate_id"], it[1]["candidate_id"], it[2]) not in done]
    batches = []
    for swapped in (False, True):  # all originals first, so a call never holds both orders of a pair
        by_cell: dict[str, list] = {}
        for it in todo:
            if it[2] is swapped:
                by_cell.setdefault(it[0]["cell"], []).append(it)
        batches += [items[i:i + pairs_per_call] for items in by_cell.values()
                    for i in range(0, len(items), pairs_per_call)]

    added: list[dict] = []
    calls = 0
    for batch in batches:
        if limit_calls is not None and calls >= limit_calls:
            break
        calls += 1
        sources = [Path(source_png(batch[0][0]["task_id"]))]
        for a, b, _ in batch:
            sources += [pool_dir / a["png"], pool_dir / b["png"]]
        with tempfile.TemporaryDirectory(prefix="judge-") as tmp:
            images = []  # neutral names: the file name must not hint at which candidate is the fallback
            for i, src in enumerate(sources, 1):
                dest = Path(tmp) / f"image_{i}.png"
                shutil.copyfile(src, dest)
                images.append(dest)
            try:
                check = _validator(len(batch))
                answer = teacher.ask(_prompt(len(batch)), _schema(len(batch)), images, validate=check)
                check(answer)  # belt and braces: never write an unchecked answer
            except TeacherUnavailable as e:
                print(f"[judge] {e}; stopping, progress kept", file=sys.stderr)
                break
            except Exception as e:  # the finished batches are already on disk; a rerun resumes
                print(f"[judge] teacher gave up on a batch ({type(e).__name__}: {e}); skipping it", file=sys.stderr)
                continue
        verdicts = answer["verdicts"]
        rows = []
        for v in sorted(verdicts, key=lambda v: v["pair"]):
            a, b, swapped = batch[v["pair"] - 1]
            rows.append({"cell": a["cell"], "task_id": a["task_id"], "split": a["split"],
                         "a": a["candidate_id"], "b": b["candidate_id"], "winner": v["winner"],
                         "confidence": v["confidence"], "reason": v["reason"],
                         "teacher": teacher.name, "swapped": swapped})
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        added += rows
    return added


def position_bias(judgments) -> float:
    """Share of non-tie answers where the first-shown candidate won (0.5 = no bias)."""
    decided = [j for j in judgments if j["winner"] != "tie"]
    return sum(j["winner"] == "a" for j in decided) / len(decided) if decided else float("nan")


def _winner_id(j: dict) -> str:
    return {"a": j["a"], "b": j["b"]}.get(j["winner"], "tie")


def swap_agreement(judgments) -> float:
    """Share of pairs where the swapped re-ask picked the same candidate (or both said tie)."""
    first, again = {}, {}
    for j in judgments:
        (again if j["swapped"] else first)[(j["cell"], frozenset((j["a"], j["b"])))] = _winner_id(j)
    both = [k for k in again if k in first]
    return sum(first[k] == again[k] for k in both) / len(both) if both else float("nan")


def main() -> None:
    from dev.distill.pool import sample_pairs
    from dev.distill.teacher import ClaudeCodeTeacher

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-cell", type=int, default=6)
    ap.add_argument("--pairs-per-call", type=int, default=4)
    ap.add_argument("--swap-share", type=float, default=0.2)
    ap.add_argument("--split", choices=["train", "holdout", "all"], default="all")
    ap.add_argument("--limit-calls", type=int, default=None)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(x) for x in (POOL_DIR / "pool.jsonl").read_text().splitlines() if x.strip()]
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    pairs = sample_pairs(rows, per_cell=args.per_cell, seed=args.seed)
    teacher = ClaudeCodeTeacher(args.model)
    added = judge_pairs(pairs, teacher, lambda t: POOL_DIR / "renders" / t / "source.png", JUDGMENTS_FILE,
                        args.pairs_per_call, args.swap_share, args.seed, POOL_DIR, args.limit_calls)
    mine = [json.loads(x) for x in JUDGMENTS_FILE.read_text().splitlines() if x.strip()]
    mine = [j for j in mine if j["teacher"] == teacher.name]
    print(f"judged {len(added)} new of {len(pairs)} pairs ({len(mine)} total by {teacher.name})")
    print(f"position bias {position_bias(mine):.2f}  swap agreement {swap_agreement(mine):.2f}")


if __name__ == "__main__":
    main()
