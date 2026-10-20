"""Where the engine looks for image files: $CHAMELEON_ASSETS_DIR if set, else
brand_kit/assets (the scored run's folder). One place, so planning, pixel
scoring and rendering always agree on which images exist."""

import os
from pathlib import Path

DEFAULT = Path("brand_kit/assets")


def default_assets_dir() -> Path:
    return Path(os.environ.get("CHAMELEON_ASSETS_DIR") or DEFAULT)
