"""The ad corpus: source designs the teacher loop resizes and judges.

The challenge ads (training-data/corpus/challenge/, split "challenge") live in
their own folder and are evaluation only: `load_corpus()` never returns them
unless asked (`include_challenge=True`, or `split="challenge"`)."""

import json
from pathlib import Path

from dev.distill.paths import CHALLENGE_DIR, CORPUS_DIR


def _load_dir(directory: Path) -> list[dict]:
    designs = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("*.json"))]
    designs.sort(key=lambda d: d["task_id"])
    return designs


def load_corpus(split: str | None = None, directory: Path = CORPUS_DIR, include_challenge: bool = False,
                challenge_dir: Path = CHALLENGE_DIR) -> list[dict]:
    """Every design file in `directory`, sorted by task_id; only `split`
    ("train" or "holdout") when given. `include_challenge` adds the challenge
    ads; `split="challenge"` returns only them."""
    designs = _load_dir(directory)
    if include_challenge or split == "challenge":
        designs = sorted(designs + load_challenge(challenge_dir), key=lambda d: d["task_id"])
    return [d for d in designs if split is None or d["split"] == split]


def load_challenge(directory: Path = CHALLENGE_DIR) -> list[dict]:
    """The challenge ads (evaluation only), sorted by task_id."""
    return _load_dir(directory) if Path(directory).is_dir() else []
