import json
from pathlib import Path
from dev.run_matrix import run_matrix


def test_run_matrix_smoke(tmp_path):
    inp = [Path("input/chameleon_sample_level1_warmup.json")]
    tg = [{"name": "leaderboard", "width": 728, "height": 90}, {"name": "story", "width": 1080, "height": 1920}]
    winners = run_matrix(inp, tg, tmp_path)
    assert set(winners["chameleon_sample_level1_warmup"]) == {"leaderboard", "story"}
    assert (tmp_path / "chameleon_sample_level1_warmup_sheet.png").exists()
    assert json.loads((tmp_path / "winners.json").read_text()) == winners
    # beyond the brief: one render per target, and the winners are real candidate ids
    assert {p.name for p in (tmp_path / "chameleon_sample_level1_warmup").iterdir()} == {"leaderboard.png", "story.png"}
    ids = winners["chameleon_sample_level1_warmup"].values()
    assert all(i == "fallback" or i == "reflow" or "/" in i for i in ids)
