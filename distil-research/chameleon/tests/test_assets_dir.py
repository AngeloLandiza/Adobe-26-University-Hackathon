"""CHAMELEON_ASSETS_DIR: one setting for where the engine looks for images
(planning, pixel scoring, rendering). Unset = brand_kit/assets, as before."""

import json
from pathlib import Path

from engine.pipeline.assets import default_assets_dir
from engine.pipeline.priority import heuristic_plan


def _ad_with_image(asset_id):
    src = json.loads(Path("input/chameleon_sample_level1_warmup.json").read_text())["source"]
    src["elements"].append({"id": "extra_img", "type": "image", "asset_id": asset_id, "x": 10, "y": 10,
                            "width": 100, "height": 100, "rotation": 0, "opacity": 1, "z_index": 50, "fit": "contain"})
    return src


def test_default_is_brand_kit(monkeypatch):
    monkeypatch.delenv("CHAMELEON_ASSETS_DIR", raising=False)
    assert default_assets_dir() == Path("brand_kit/assets")


def test_env_redirects_planning(monkeypatch, tmp_path):
    (tmp_path / "only_here.png").write_bytes(Path("brand_kit/assets/solstice_bottle.png").read_bytes())
    src = _ad_with_image("only_here.png")
    monkeypatch.delenv("CHAMELEON_ASSETS_DIR", raising=False)
    missing = [u for u in heuristic_plan(src).units if "missing_asset" in u.flags]
    assert any("extra_img" in u.element_ids for u in missing)
    monkeypatch.setenv("CHAMELEON_ASSETS_DIR", str(tmp_path))
    found = [u for u in heuristic_plan(src).units if "missing_asset" in u.flags]
    assert not any("extra_img" in u.element_ids for u in found)
