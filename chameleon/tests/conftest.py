"""Shared fixtures. Tests run from chameleon/ (python -m pytest), so paths are
relative to it, matching engine/main.py's conventions."""

import json
from pathlib import Path

import pytest

INPUT_DIR = Path("input")
TARGETS_PATH = Path("dev/targets.json")


def _load(stem: str) -> dict:
    return json.loads((INPUT_DIR / f"{stem}.json").read_text())


def element(task: dict, element_id: str) -> dict:
    """Return the source element with `element_id` from a task dict."""
    return next(el for el in task["source"]["elements"] if el["id"] == element_id)


@pytest.fixture
def l1() -> dict:
    return _load("chameleon_sample_level1_warmup")


@pytest.fixture
def l2() -> dict:
    return _load("chameleon_sample_level2_core")


@pytest.fixture
def l3() -> dict:
    return _load("chameleon_sample_level3_advanced")


@pytest.fixture
def targets() -> list[dict]:
    return json.loads(TARGETS_PATH.read_text())
