"""The ad corpus: source designs the teacher loop resizes and judges."""

import json
from pathlib import Path

from dev.distill.paths import CORPUS_DIR


def load_corpus(split: str | None = None, directory: Path = CORPUS_DIR) -> list[dict]:
    """Every design file in `directory`, sorted by task_id; only `split`
    ("train" or "holdout") when given."""
    designs = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
    designs.sort(key=lambda d: d["task_id"])
    return [d for d in designs if split is None or d["split"] == split]
