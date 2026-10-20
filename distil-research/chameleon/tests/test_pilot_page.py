"""The person's pilot page: every pair, its three images and the rubric."""

import json

from PIL import Image

from dev.distill.judge import RUBRIC
from dev.distill.paths import RENDERS_SUBDIR
from dev.distill.pilot_page import write_page


def test_page_has_every_pair_images_and_rubric(tmp_path):
    pool, pilot = tmp_path / "pool", tmp_path / "pilot"
    pilot.mkdir()
    rows, pairs = [], {}
    for i in range(3):
        cell, task = f"t{i}@leaderboard", f"t{i}"
        for cid in ("x", "y"):
            png = f"{RENDERS_SUBDIR}/{task}/leaderboard/{cid}.png"
            (pool / png).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 4)).save(pool / png)
            rows.append({"cell": cell, "task_id": task, "target": "leaderboard", "candidate_id": cid, "png": png})
        Image.new("RGB", (4, 4)).save(pool / RENDERS_SUBDIR / task / "source.png")
        pairs[f"p{i + 1:02d}"] = {"cell": cell, "task_id": task, "a": "x", "b": "y"}
    (pool / "pool.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (pilot / "pairs.json").write_text(json.dumps(pairs))

    page = write_page(pilot, pool).read_text()
    assert page.count('class="pair"') == 3
    assert page.count("data:image/png;base64,") == 9
    for pid in pairs:
        assert page.count(f'name="{pid}"') == 3
    assert RUBRIC.splitlines()[0] in page
