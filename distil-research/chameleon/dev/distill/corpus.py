"""The ad corpus: source designs the teacher loop resizes and judges.

The challenge ads (training-data/corpus/challenge/, split "challenge") live in
their own folder and are evaluation only: `load_corpus()` never returns them
unless asked (`include_challenge=True`, or `split="challenge"`).

The real-ad challenge (training-data/corpus/challenge2.nosync/, split
"challenge2", Crello-derived and gitignored) is evaluation only too and is
returned only by `split="challenge2"` or `load_challenge2()`; its images
resolve with `merged_assets_dir(*CHALLENGE2_FOLDERS)` (dev/distill/assets.py).

The development Crello ads (training-data/corpus/crello_dev.nosync/, split
"crello_dev", make_crello_dev.py; no design family shared with challenge2)
are returned only by `split="crello_dev"` or `load_crello_dev()`; their
images resolve with `merged_assets_dir(*CRELLO_DEV_FOLDERS)`.

The second real-ad evaluation set (training-data/corpus/challenge3.nosync/,
split "challenge3", make_challenge3.py; no design family shared with
challenge2 or crello_dev) is evaluation only and returned only by
`split="challenge3"` or `load_challenge3()`; its images resolve with
`merged_assets_dir(*CHALLENGE3_FOLDERS)`.

`split="samples"` returns the 3 official sample ads (input/*.json, in the
corpus as the train ads SAMPLE_TASKS), relabelled split "samples"."""

import json
from pathlib import Path

from dev.distill.paths import CHALLENGE2_DIR, CHALLENGE3_DIR, CHALLENGE_DIR, CORPUS_DIR, CRELLO_DEV_DIR

# The official samples (input/chameleon_sample_level{1,2,3}_*.json) as they sit in the corpus.
SAMPLE_TASKS = ("warmup_01", "core_01", "advanced_01")


def _load_dir(directory: Path, pattern: str = "*.json") -> list[dict]:
    designs = [json.loads(p.read_text()) for p in sorted(Path(directory).glob(pattern))]
    designs.sort(key=lambda d: d["task_id"])
    return designs


def load_corpus(split: str | None = None, directory: Path = CORPUS_DIR, include_challenge: bool = False,
                challenge_dir: Path = CHALLENGE_DIR, challenge2_dir: Path = CHALLENGE2_DIR,
                crello_dev_dir: Path = CRELLO_DEV_DIR, challenge3_dir: Path = CHALLENGE3_DIR) -> list[dict]:
    """Every design file in `directory`, sorted by task_id; only `split`
    ("train" or "holdout") when given. `include_challenge` adds the challenge
    ads; `split="challenge"` returns only them; `split="challenge2"` returns
    only the real (Crello) evaluation ads; `split="crello_dev"` only the
    real development ads; `split="challenge3"` only the second real
    evaluation set; `split="samples"` the 3 official samples (copies
    labelled split "samples")."""
    if split == "challenge2":
        return [d for d in load_challenge2(challenge2_dir) if d["split"] == "challenge2"]
    if split == "crello_dev":
        return [d for d in load_crello_dev(crello_dev_dir) if d["split"] == "crello_dev"]
    if split == "challenge3":
        return [d for d in load_challenge3(challenge3_dir) if d["split"] == "challenge3"]
    if split == "samples":
        return load_samples(directory)
    designs = _load_dir(directory)
    if include_challenge or split == "challenge":
        designs = sorted(designs + load_challenge(challenge_dir), key=lambda d: d["task_id"])
    return [d for d in designs if split is None or d["split"] == split]


def load_challenge(directory: Path = CHALLENGE_DIR) -> list[dict]:
    """The challenge ads (evaluation only), sorted by task_id."""
    return _load_dir(directory) if Path(directory).is_dir() else []


def load_challenge2(directory: Path = CHALLENGE2_DIR) -> list[dict]:
    """The real-ad challenge (Crello conversions, evaluation only), sorted by
    task_id; empty when the folder hasn't been built (make_real_challenge).
    Only the design files (cr2_*.json): the folder also holds summary.json."""
    return _load_dir(directory, "cr2_*.json") if Path(directory).is_dir() else []


def load_crello_dev(directory: Path = CRELLO_DEV_DIR) -> list[dict]:
    """The development Crello ads, sorted by task_id; empty when the folder
    hasn't been built (make_crello_dev). Only the design files (crd_*.json)."""
    return _load_dir(directory, "crd_*.json") if Path(directory).is_dir() else []


def load_challenge3(directory: Path = CHALLENGE3_DIR) -> list[dict]:
    """The second real-ad evaluation set (Crello conversions, evaluation only),
    sorted by task_id; empty when the folder hasn't been built
    (make_challenge3). Only the design files (cr3_*.json)."""
    return _load_dir(directory, "cr3_*.json") if Path(directory).is_dir() else []


def load_samples(directory: Path = CORPUS_DIR) -> list[dict]:
    """The 3 official sample ads (SAMPLE_TASKS) from the corpus, as copies
    labelled split "samples" (their corpus split stays "train")."""
    designs = [d for d in _load_dir(directory) if d["task_id"] in SAMPLE_TASKS]
    return [{**d, "split": "samples"} for d in designs]
