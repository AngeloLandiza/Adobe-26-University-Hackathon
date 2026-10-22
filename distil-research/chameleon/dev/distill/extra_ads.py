"""More TRAIN-only source ads for distillation data: further augmented variants
of the drafted training ads (augment.py's operations, new seeds), written outside
the corpus so no experiment split changes.

    python -m dev.distill.extra_ads [--per-ad 13]   # slots beyond the corpus's 3

Writes training-data/llm-distill/extra_ads/<task_id>.json (corpus record format).
"""

import argparse
import json

from dev.distill import augment
from dev.distill import make_corpus as mc
from dev.distill.corpus import load_corpus
from dev.distill.paths import DATA_ROOT

OUT = DATA_ROOT / "llm-distill" / "extra_ads"


def load_extra() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-ad", type=int, default=13)
    a = p.parse_args()
    existing = {d["task_id"] for d in load_corpus(None)}
    parents = [r for r in mc.drafted() if r["origin"] == "claude-drafted" and r.get("split") == "train"]
    augment.VARIANTS_PER_AD = a.per_ad
    records, dropped = augment.generate(parents, log=lambda m: None)
    new = [r for r in records if r["task_id"] not in existing and r.get("split") == "train"]
    OUT.mkdir(parents=True, exist_ok=True)
    for r in new:
        (OUT / f"{r['task_id']}.json").write_text(json.dumps(r))
    print(f"{len(parents)} train parents -> {len(records)} variants, {len(new)} new written to {OUT} ({len(dropped)} dropped attempts)")


if __name__ == "__main__":
    main()
