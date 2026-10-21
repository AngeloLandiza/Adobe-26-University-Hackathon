"""Where the distillation data lives: distil-research/training-data/, one
folder per dataset (each has a README saying what it is and what it feeds)."""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[3] / "training-data"
CORPUS_DIR = DATA_ROOT / "corpus" / "designs"
# Evaluation-only ads (dev/distill/make_challenge.py): never used for tuning or selection.
CHALLENGE_DIR = DATA_ROOT / "corpus" / "challenge"
# Real ads converted from the Crello dataset (dev/distill/make_real_challenge.py), split "challenge2".
# Crello-derived content: the folder ends in ".nosync", so it is gitignored and never committed.
CHALLENGE2_DIR = DATA_ROOT / "corpus" / "challenge2.nosync"
# More real Crello ads, for DEVELOPMENT (dev/distill/make_crello_dev.py), split "crello_dev": no design
# family shared with challenge2. Crello-derived too, so gitignored (".nosync").
CRELLO_DEV_DIR = DATA_ROOT / "corpus" / "crello_dev.nosync"
POOL_DIR = DATA_ROOT / "candidate-pool"
JUDGMENTS_FILE = DATA_ROOT / "judgments" / "judgments.jsonl"
# Regenerable bulk output goes in folders ending ".nosync": iCloud Drive skips
# those, so thousands of PNG writes in a synced ~/Documents create no conflict copies.
RENDERS_SUBDIR = "renders.nosync"
TEACHER_CACHE = DATA_ROOT / "teacher-cache" / "cache.nosync"
