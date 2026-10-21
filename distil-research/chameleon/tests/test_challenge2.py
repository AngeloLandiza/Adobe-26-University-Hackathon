"""challenge2: real ads converted from Crello (dev/distill/make_real_challenge.py).

The converted ads are Crello-derived and gitignored (corpus/challenge2.nosync/),
so the checks on them skip on a checkout that hasn't built them; the loader,
harness plumbing and converter helpers are tested on synthetic data.
"""

import json

import pytest
from PIL import Image

from dev.distill import ab_engine as ab
from dev.distill import make_real_challenge as mrc
from dev.distill.assets import BRAND_ASSETS_DIR, CHALLENGE2_FOLDERS, asset_exists, merged_assets_dir
from dev.distill.corpus import load_challenge2, load_corpus


def _write(directory, rec):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{rec['task_id']}.json").write_text(json.dumps(rec))


def test_challenge2_split_is_separate(tmp_path):
    main, chl, chl2 = tmp_path / "designs", tmp_path / "challenge", tmp_path / "challenge2.nosync"
    _write(main, {"task_id": "a", "split": "train"})
    _write(chl, {"task_id": "z", "split": "challenge"})
    _write(chl2, {"task_id": "cr2_01", "split": "challenge2"})
    (chl2 / "summary.json").write_text("{}")
    kw = dict(directory=main, challenge_dir=chl, challenge2_dir=chl2)
    assert [d["task_id"] for d in load_corpus(**kw)] == ["a"]
    assert [d["task_id"] for d in load_corpus(include_challenge=True, **kw)] == ["a", "z"]  # unchanged
    assert [d["task_id"] for d in load_corpus("challenge", **kw)] == ["z"]
    assert [d["task_id"] for d in load_corpus("challenge2", **kw)] == ["cr2_01"]
    assert load_challenge2(tmp_path / "missing") == []


def test_harness_renders_challenge2_sources_with_their_own_images(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("RGB", (40, 20), (255, 0, 0)).save(assets / "cr2_test_bg.png")
    design = {"task_id": "cr2_t", "split": "challenge2", "brand": "crello",
              "source": {"canvas": {"width": 200, "height": 100},
                         "elements": [{"id": "bg", "type": "image", "x": 0, "y": 0, "width": 200, "height": 100,
                                       "z_index": 0, "asset_id": "cr2_test_bg.png", "fit": "cover"}]}}
    cells = ab.challenge_cells([design], [{"name": "sq", "width": 100, "height": 100},
                                          {"name": "same", "width": 400, "height": 200}])
    assert [c["cell"] for c in cells] == ["cr2_t@sq"] and cells[0]["brand"] == "crello"
    assert ab.write_sources([design], tmp_path / "ab", folders=(BRAND_ASSETS_DIR, assets)) == 1
    px = Image.open(ab.source_png(tmp_path / "ab", "cr2_t")).convert("RGB").getpixel((100, 50))
    assert px[0] > 200 and px[1] < 60  # the asset, not the grey missing-asset placeholder
    assert ab.family("crello") == "crello"


def test_converter_helpers():
    assert mrc._rgba("rgba(255, 236, 91, 1)") == "rgba(255,236,91,1)"
    assert mrc._rgba("rgba(10, 20, 30, 0.5)", 0.5) == "rgba(10,20,30,0.25)"
    assert mrc._angle(359.8) == 0.0 and mrc._angle(270) == -90.0 and mrc._angle(12.5) == 12.5
    assert mrc.font_for("Playfair Display", 0) == {"font_family": "Solstice Serif"}
    assert mrc.font_for("Bebas Neue", 0)["font_weight"] == "bold"
    assert mrc.font_for("Pacifico", 0)["font_style"] == "italic"
    content = "BBQ GOODS"
    runs = mrc._runs(["g"] * 3 + ["w"] * 6, "w", "color", content)
    assert runs == [{"start": 0, "end": 3, "color": "g"}]

    rect = Image.new("RGBA", (50, 30), (10, 120, 200, 255))
    assert mrc.classify_svg(rect) == ("rectangle", (10, 120, 200))
    ell = Image.new("RGBA", (60, 40), (0, 0, 0, 0))
    from PIL import ImageDraw
    ImageDraw.Draw(ell).ellipse((0, 0, 59, 39), fill=(250, 20, 20, 255))
    assert mrc.classify_svg(ell)[0] == "ellipse"
    stripes = Image.new("RGBA", (200, 100), (0, 0, 0, 255))
    for x in range(0, 200, 8):
        ImageDraw.Draw(stripes).rectangle((x, 0, x + 3, 99), fill=(30, 40, 120, 255))
    assert mrc.classify_svg(stripes) is None  # texture stays a raster

    im, box = mrc.crop_to_canvas(Image.new("RGBA", (100, 100)), (-50, 0, 100, 100), 200, 200)
    assert box == (0.0, 0.0, 50.0, 100.0) and im.size == (50, 100)


CHL2 = load_challenge2()


@pytest.mark.skipif(not CHL2, reason="challenge2 not built (python -m dev.distill.make_real_challenge)")
def test_challenge2_ads_are_labelled_resolvable_and_pass_the_gate():
    from dev.distill.make_challenge import quality_issues

    assert 40 <= len(CHL2) <= 60
    with merged_assets_dir(*CHALLENGE2_FOLDERS) as assets:
        for d in CHL2:
            assert d["split"] == "challenge2" and d["origin"] == "crello" and d["brand"] == "crello", d["task_id"]
            assert d["task_id"].startswith("cr2_")
            assert any(el["type"] == "text" for el in d["source"]["elements"]), d["task_id"]
            for el in d["source"]["elements"]:
                if el["type"] == "image":
                    assert el["asset_id"].startswith("cr2_") and asset_exists(el["asset_id"]), (d["task_id"], el["id"])
            assert quality_issues(d, assets) == [], d["task_id"]
