"""Where the distillation data lives: distil-research/training-data/, one
folder per dataset (each has a README saying what it is and what it feeds)."""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[3] / "training-data"
CORPUS_DIR = DATA_ROOT / "corpus" / "designs"
POOL_DIR = DATA_ROOT / "candidate-pool"
JUDGMENTS_FILE = DATA_ROOT / "judgments" / "judgments.jsonl"
# Regenerable bulk output goes in folders ending ".nosync": iCloud Drive skips
# those, so thousands of PNG writes in a synced ~/Documents create no conflict copies.
RENDERS_SUBDIR = "renders.nosync"
TEACHER_CACHE = DATA_ROOT / "teacher-cache" / "cache.nosync"
