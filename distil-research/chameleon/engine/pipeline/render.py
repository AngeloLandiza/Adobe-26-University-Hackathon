"""Render a design with the official renderer and hand back the pixels."""

import tempfile
from pathlib import Path

from PIL import Image

from engine.pipeline.assets import default_assets_dir
from renderer.converter.convert import render_design_to_png


def render_to_image(design: dict, assets_dir: str | Path | None = None, max_size: int = 1024) -> Image.Image:
    """RGB render of `design`. The renderer only downscales, so the pixel
    scale is `img.width / design["canvas"]["width"]` (1.0 for small canvases)."""
    assets_dir = default_assets_dir() if assets_dir is None else assets_dir
    with tempfile.TemporaryDirectory() as tmp:
        png_path = Path(tmp) / "render.png"
        render_design_to_png(design, assets_dir, png_path, max_size=max_size)
        # convert() returns a loaded copy, so the file is closed before the
        # temp dir is removed (Windows won't delete an open file).
        with Image.open(png_path) as img:
            return img.convert("RGB")
