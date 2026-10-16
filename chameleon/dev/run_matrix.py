"""Run every source design against every target size and tile the results
into one contact sheet per source: the team's main way to look at quality.

    python -m dev.run_matrix [--out output/matrix] [--inputs input/a.json ...] [--targets dev/targets.json]
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from engine.pipeline.render import render_to_image
from engine.resize import resize_with_report

TILE = 360          # longest side of each render on the sheet, px
LABEL_H = 34        # label strip above each render
COLUMNS = 5


def _tile(render: Image.Image, lines: list[str]) -> Image.Image:
    tile = Image.new("RGB", (max(render.width, 200), render.height + LABEL_H), "white")
    draw = ImageDraw.Draw(tile)
    for i, line in enumerate(lines):
        draw.text((4, 3 + 15 * i), line, fill="black")
    tile.paste(render, (0, LABEL_H))
    return tile


def _sheet(tiles: list[Image.Image]) -> Image.Image:
    cell_w = max(t.width for t in tiles) + 10
    cell_h = max(t.height for t in tiles) + 10
    columns = min(COLUMNS, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (225, 225, 225))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * cell_w + 5, (i // columns) * cell_h + 5))
    return sheet


def run_matrix(inputs: list[Path], targets: list[dict], out_dir: Path) -> dict:
    """Writes out_dir/<stem>/<target>.png, out_dir/<stem>_sheet.png and
    out_dir/winners.json ({stem: {target: candidate_id}}); returns the winners."""
    out_dir.mkdir(parents=True, exist_ok=True)
    winners: dict[str, dict[str, str]] = {}
    for path in inputs:
        stem = path.stem
        source = json.loads(path.read_text())["source"]
        (out_dir / stem).mkdir(exist_ok=True)
        winners[stem], tiles = {}, []
        for target in targets:
            canvas = {"width": target["width"], "height": target["height"]}
            design, plan, best, _ = resize_with_report(source, canvas)
            winners[stem][target["name"]] = best.candidate_id
            render = render_to_image(design, max_size=max(target["width"], target["height"]))
            render.save(out_dir / stem / f"{target['name']}.png")
            render.thumbnail((TILE, TILE))
            tiles.append(_tile(render, [f"{target['name']} {target['width']}x{target['height']}  [{plan.source}]",
                                        best.candidate_id]))
        _sheet(tiles).save(out_dir / f"{stem}_sheet.png")
        print(f"[matrix] {stem}: {len(targets)} targets -> {out_dir / f'{stem}_sheet.png'}")
    (out_dir / "winners.json").write_text(json.dumps(winners, indent=2), encoding="utf-8")
    return winners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("output/matrix"))
    parser.add_argument("--inputs", type=Path, nargs="*", default=sorted(Path("input").glob("*.json")))
    parser.add_argument("--targets", type=Path, default=Path("dev/targets.json"))
    args = parser.parse_args()
    run_matrix(args.inputs, json.loads(args.targets.read_text()), args.out)


if __name__ == "__main__":
    main()
