"""The ad corpus the distillation loop learns from (training-data/corpus/).

Checks that there are enough ads, that every ad only uses the brand kit's
images and fonts (except the ads that test a missing asset on purpose), and
that the engine can plan and render every one of them."""

from pathlib import Path

import pytest

from dev.distill.corpus import load_corpus
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.render import render_to_image

ASSETS_DIR = Path("brand_kit/assets")
FONTS_DIR = Path("brand_kit/fonts")
BRAND_FAMILIES = {"Solstice Serif", "Solstice Sans"}

CORPUS = load_corpus()


def _font_file(el: dict) -> Path:
    """The brand-kit file a text element resolves to, e.g. SolsticeSans-BoldItalic.ttf."""
    bold = el.get("font_weight", "regular") == "bold"
    italic = el.get("font_style", "normal") == "italic"
    suffix = ("Bold" if bold else "") + ("Italic" if italic else "")
    return FONTS_DIR / f"{el['font_family'].replace(' ', '')}-{suffix or 'Regular'}.ttf"


def test_corpus_size():
    ids = [d["task_id"] for d in CORPUS]
    assert len(CORPUS) >= 30
    assert len(load_corpus("holdout")) >= 8
    assert len(load_corpus("train")) + len(load_corpus("holdout")) == len(CORPUS)
    assert len(ids) == len(set(ids))


def test_corpus_records_are_well_formed():
    for d in CORPUS:
        assert d["split"] in ("train", "holdout"), d["task_id"]
        assert d["origin"] in ("sample", "hand", "claude-drafted", "imported"), d["task_id"]
        assert isinstance(d["expects_missing_asset"], bool), d["task_id"]
        ids = [el["id"] for el in d["source"]["elements"]]
        assert len(ids) == len(set(ids)), d["task_id"]
        assert 3 <= len(ids) <= 20, d["task_id"]
    assert sum(d["origin"] == "sample" for d in CORPUS) == 3


def test_assets_and_fonts_exist():
    for d in CORPUS:
        missing = []
        for el in d["source"]["elements"]:
            if el["type"] == "image" and not (ASSETS_DIR / el["asset_id"]).is_file():
                missing.append(el["asset_id"])
            if el["type"] == "text":
                assert el["font_family"] in BRAND_FAMILIES, (d["task_id"], el["id"])
                assert _font_file(el).is_file(), (d["task_id"], el["id"], _font_file(el))
        assert bool(missing) == d["expects_missing_asset"], (d["task_id"], missing)


@pytest.mark.parametrize("design", CORPUS, ids=[d["task_id"] for d in CORPUS])
def test_designs_plan_and_render(design):
    source = design["source"]
    plan = heuristic_plan(source)
    assert {eid for u in plan.units for eid in u.element_ids} == {el["id"] for el in source["elements"]}
    img = render_to_image(source, max_size=256)
    assert max(img.size) <= 256
