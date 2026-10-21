"""One assets folder for a run: the brand kit's images plus the code-drawn
research stand-ins in training-data/corpus/assets/ (all named `syn_*`).

The engine and renderer take a single `assets_dir`, so corpus ads that use a
synthetic image need both folders visible at once. `merged_assets_dir()`
builds a temporary folder of symlinks (copies where symlinks aren't allowed)
and removes it afterwards. The renderer looks for fonts in a `fonts/` folder
next to the assets folder, so the brand fonts are linked in beside it:

    with merged_assets_dir() as assets:
        render_to_image(design, assets_dir=assets)
"""

import contextlib
import shutil
import tempfile
from pathlib import Path

from dev.distill.paths import DATA_ROOT

BRAND_ASSETS_DIR = Path(__file__).resolve().parents[2] / "brand_kit" / "assets"
BRAND_FONTS_DIR = BRAND_ASSETS_DIR.parent / "fonts"
SYNTH_ASSETS_DIR = DATA_ROOT / "corpus" / "assets"
# Per-element rasters of the Crello ads (split "challenge2", all named `cr2_*`). Gitignored
# (.nosync); merged in only when asked: merged_assets_dir(*CHALLENGE2_FOLDERS).
CHALLENGE2_ASSETS_DIR = DATA_ROOT / "corpus" / "challenge2.nosync" / "assets"
CHALLENGE2_FOLDERS = (BRAND_ASSETS_DIR, SYNTH_ASSETS_DIR, CHALLENGE2_ASSETS_DIR)
LOOKUP_FOLDERS = CHALLENGE2_FOLDERS  # where asset_exists/asset_file look (cr2_ names never clash)


def asset_file(asset_id: str) -> Path | None:
    """The file `asset_id` names in the brand kit, the synthetic assets or the
    Crello (challenge2) rasters, or None."""
    return next((d / asset_id for d in LOOKUP_FOLDERS if (d / asset_id).is_file()), None)


def asset_exists(asset_id: str) -> bool:
    """True if `asset_id` is a file in one of the asset folders."""
    return asset_file(asset_id) is not None


@contextlib.contextmanager
def merged_assets_dir(*folders: Path):
    """Yield a temp folder holding every image of `folders` (default: the
    brand kit, then the synthetic assets). Two files with the same name are an
    error: an ad must never silently get a different image than it names."""
    folders = folders or (BRAND_ASSETS_DIR, SYNTH_ASSETS_DIR)
    seen: dict[str, Path] = {}
    for folder in folders:
        for path in sorted(Path(folder).glob("*")):
            if not path.is_file() or path.suffix.lower() == ".md":
                continue
            if path.name in seen:
                raise ValueError(f"asset name collision: {path.name} in {seen[path.name].parent} and {folder}")
            seen[path.name] = path
    with tempfile.TemporaryDirectory(prefix="chameleon-assets-") as tmp:
        out = Path(tmp) / "assets"
        out.mkdir()
        for name, path in seen.items():
            _link(path, out / name)
        _link(BRAND_FONTS_DIR, Path(tmp) / "fonts")
        yield out


def _link(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    except OSError:  # e.g. Windows without symlink rights
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
