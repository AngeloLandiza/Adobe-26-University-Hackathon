"""The ad corpus the distillation loop learns from (training-data/corpus/).

Checks that there are enough ads of each origin and brand, that augmented
variants never leak a holdout ad into train, that every ad only uses the brand
kit's fonts and images from the brand kit or the synthetic research assets
(except the ads that test a missing asset on purpose), that generated ads pass
the generators' own layout checks, and that the engine can plan and render
every one of them with the merged assets folder."""

from pathlib import Path

import pytest

from dev.distill.assets import asset_exists, merged_assets_dir
from dev.distill.corpus import load_corpus
from engine.pipeline.priority import heuristic_plan
from engine.pipeline.render import render_to_image

FONTS_DIR = Path("brand_kit/fonts")
BRAND_FAMILIES = {"Solstice Serif", "Solstice Sans"}
ORIGINS = ("sample", "hand", "claude-drafted", "imported", "augmented", "offbrand")

CORPUS = load_corpus()
BY_ID = {d["task_id"]: d for d in CORPUS}


@pytest.fixture(scope="module")
def assets():
    with merged_assets_dir() as path:
        yield path


def _font_file(el: dict) -> Path:
    """The brand-kit file a text element resolves to, e.g. SolsticeSans-BoldItalic.ttf."""
    bold = el.get("font_weight", "regular") == "bold"
    italic = el.get("font_style", "normal") == "italic"
    suffix = ("Bold" if bold else "") + ("Italic" if italic else "")
    return FONTS_DIR / f"{el['font_family'].replace(' ', '')}-{suffix or 'Regular'}.ttf"


def _count(**match) -> int:
    return sum(all(d.get(k) == v for k, v in match.items()) for d in CORPUS)


def test_corpus_size():
    ids = [d["task_id"] for d in CORPUS]
    assert len(ids) == len(set(ids))
    assert len(load_corpus("train")) + len(load_corpus("holdout")) == len(CORPUS)
    assert _count(origin="claude-drafted") >= 30
    assert _count(origin="augmented") >= 80
    assert _count(origin="offbrand") >= 25
    assert _count(split="holdout", brand="solstice") >= 8
    assert _count(split="holdout", brand="offbrand") >= 6
    assert _count(origin="offbrand", split="holdout") >= 6


def test_corpus_records_are_well_formed():
    for d in CORPUS:
        assert d["split"] in ("train", "holdout"), d["task_id"]
        assert d["origin"] in ORIGINS, d["task_id"]
        assert d["brand"] in ("solstice", "offbrand"), d["task_id"]
        assert isinstance(d["expects_missing_asset"], bool), d["task_id"]
        ids = [el["id"] for el in d["source"]["elements"]]
        assert len(ids) == len(set(ids)), d["task_id"]
        assert 3 <= len(ids) <= 20, d["task_id"]
    assert sum(d["origin"] == "sample" for d in CORPUS) == 3
    assert all(d["brand"] == "solstice" for d in CORPUS if d["origin"] in ("sample", "claude-drafted"))
    assert all(d["brand"] == "offbrand" for d in CORPUS if d["origin"] == "offbrand")


def test_augmented_variants_inherit_their_parents_split():
    """A variant of a holdout ad is holdout: it is the same design underneath."""
    for d in CORPUS:
        if d["origin"] != "augmented":
            assert "parent" not in d, d["task_id"]
            continue
        parent = BY_ID.get(d["parent"])
        assert parent is not None and parent["origin"] == "claude-drafted", d["task_id"]
        assert d["split"] == parent["split"], d["task_id"]
        assert isinstance(d["augmentations"], list) and 2 <= len(d["augmentations"]) <= 4, d["task_id"]


def test_offbrand_ads_use_only_synthetic_images():
    for d in CORPUS:
        if d["origin"] == "offbrand":
            images = [el["asset_id"] for el in d["source"]["elements"] if el["type"] == "image"]
            assert all(a.startswith("syn_") for a in images), d["task_id"]


def test_assets_and_fonts_exist():
    for d in CORPUS:
        missing = []
        for el in d["source"]["elements"]:
            if el["type"] == "image" and not asset_exists(el["asset_id"]):
                missing.append(el["asset_id"])
            if el["type"] == "text":
                assert el["font_family"] in BRAND_FAMILIES, (d["task_id"], el["id"])
                assert _font_file(el).is_file(), (d["task_id"], el["id"], _font_file(el))
        assert bool(missing) == d["expects_missing_asset"], (d["task_id"], missing)


def test_generated_ads_pass_the_layout_checks():
    """Text fits its box (TextMeasurer), nothing off-canvas but decoration,
    no accidental overlaps, images keep their aspect ratio."""
    from dev.distill.make_corpus import check

    for d in CORPUS:
        if d["origin"] in ("augmented", "offbrand"):
            assert check(d, strict=True) == [], d["task_id"]


@pytest.mark.parametrize("design", CORPUS, ids=[d["task_id"] for d in CORPUS])
def test_designs_plan_and_render(design, assets):
    source = design["source"]
    plan = heuristic_plan(source, assets)
    assert {eid for u in plan.units for eid in u.element_ids} == {el["id"] for el in source["elements"]}
    flagged = any("missing_asset" in u.flags for u in plan.units)
    assert flagged == design["expects_missing_asset"], design["task_id"]
    img = render_to_image(source, assets_dir=assets, max_size=256)
    assert max(img.size) <= 256


def test_quality_gate_catches_broken_layouts(assets):
    """The generators' checks really reject what the quality bar forbids:
    text moved onto a busy image, a stretched image, text too big for its box."""
    import copy

    from dev.distill.make_corpus import check, pixel_check

    good = BY_ID["o03_q_story_cold_brew"]
    assert check(good, strict=True) == [] and pixel_check(good, assets) == []
    els = lambda rec: {e["id"]: e for e in rec["source"]["elements"]}

    on_busy = copy.deepcopy(good)
    els(on_busy)["headline"]["y"] = 40  # off the calm panel, onto the terrazzo
    assert any("busy" in w for w in pixel_check(on_busy, assets))

    stretched = copy.deepcopy(good)
    els(stretched)["product"]["width"] *= 1.5
    assert any("aspect" in w for w in check(stretched, strict=True))

    crammed = copy.deepcopy(good)
    els(crammed)["headline"]["height"] = 40
    assert any("does not fit" in w for w in check(crammed, strict=True))
