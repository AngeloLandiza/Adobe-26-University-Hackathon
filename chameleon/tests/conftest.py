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


def assert_valid(source: dict, design: dict, target: dict) -> None:
    """The structural rules every output must meet (same ids in order, in canvas, image aspect, font >= 1)."""
    from engine.pipeline.geometry import inside_canvas, rotated_aabb

    assert design["canvas"] == {"width": target["width"], "height": target["height"]}
    assert [e["id"] for e in design["elements"]] == [e["id"] for e in source["elements"]]
    src = {e["id"]: e for e in source["elements"]}
    for el in design["elements"]:
        assert inside_canvas(rotated_aabb(el), target["width"], target["height"]), el["id"]
        if el["type"] == "image" and el.get("fit", "contain") != "cover":
            s = src[el["id"]]
            assert (el["width"] / el["height"]) / (s["width"] / s["height"]) == pytest.approx(1, abs=0.02), el["id"]
        if el["type"] == "text":
            assert el["font_size"] >= 1


@pytest.fixture(scope="session")
def measurer():
    # Imported here so a broken textmeasure module can't stop unrelated test files from collecting.
    from engine.pipeline.textmeasure import TextMeasurer

    return TextMeasurer()
