import json
import random

import pytest

from dev.distill import pool
from dev.distill.paths import RENDERS_SUBDIR
from engine.pipeline.layout.aspect import aspect_class

METRIC_KEYS = ["hierarchy", "legibility", "overlap", "degradation", "faithfulness", "space_use", "alignment",
               "spacing", "balance", "text_contrast", "busyness", "placeholder"]


def _rows(cell_count=3, per_cell=12, seed=1):
    """Fake pool rows (no rendering), so the sampling tests are fast."""
    rng = random.Random(seed)
    rows = []
    for c in range(cell_count):
        for i in range(per_cell):
            rows.append({"cell": f"ad{c}@story", "task_id": f"ad{c}", "split": "train", "target": "story",
                         "candidate_id": f"c{i:02d}", "png": f"renders/ad{c}/story/{i}.png", "is_fallback": i == 0,
                         "metrics": {k: rng.random() for k in METRIC_KEYS}})
    return rows


def _ids(pairs):
    return [(a["cell"], a["candidate_id"], b["candidate_id"]) for a, b in pairs]


def test_pool_rows_have_metrics_and_pngs(l1, targets, tmp_path):
    design = {"task_id": l1["task_id"], "split": "train", "source": l1["source"]}
    target = next(t for t in targets if t["name"] == "leaderboard")
    rows = pool.build_pool([design], [target], tmp_path)
    assert rows
    on_disk = [json.loads(line) for line in (tmp_path / "pool.jsonl").read_text().splitlines()]
    assert on_disk == rows
    assert (tmp_path / RENDERS_SUBDIR / l1["task_id"] / "source.png").exists()
    for row in rows:
        assert row["cell"] == f"{l1['task_id']}@leaderboard"
        assert (row["task_id"], row["split"], row["target"]) == (l1["task_id"], "train", "leaderboard")
        assert len(row["metrics"]) == 12
        assert (tmp_path / row["png"]).exists()
    assert len({r["candidate_id"] for r in rows}) == len(rows)
    assert sum(r["is_fallback"] for r in rows) <= 1


def test_sample_pairs_deterministic():
    rows = _rows()
    assert _ids(pool.sample_pairs(rows, seed=3)) == _ids(pool.sample_pairs(rows, seed=3))
    assert _ids(pool.sample_pairs(rows, seed=3)) != _ids(pool.sample_pairs(rows, seed=4))


def test_pairs_within_cell_no_self_pairs():
    pairs = pool.sample_pairs(_rows())
    assert pairs
    for a, b in pairs:
        assert a["cell"] == b["cell"]
        assert a["candidate_id"] != b["candidate_id"]
    keys = [(a["cell"], frozenset((a["candidate_id"], b["candidate_id"]))) for a, b in pairs]
    assert len(keys) == len(set(keys))  # never the same pair twice


def test_per_cell_cap():
    rows = _rows(cell_count=3, per_cell=12)
    pairs = pool.sample_pairs(rows, per_cell=6)
    for cell in {r["cell"] for r in rows}:
        assert sum(a["cell"] == cell for a, _ in pairs) == 6
    # a cell with only 3 candidates has just 3 distinct pairs
    small = pool.sample_pairs(_rows(cell_count=1, per_cell=3), per_cell=6)
    assert len(small) == 3
    assert pool.sample_pairs(_rows(cell_count=1, per_cell=1)) == []


def test_choose_targets_covers_classes_and_is_deterministic(targets):
    ids = [f"ad{i}" for i in range(10)]
    chosen = pool.choose_targets(ids, targets, per_ad=6, seed=0)
    assert chosen == pool.choose_targets(ids, targets, per_ad=6, seed=0)
    assert chosen != pool.choose_targets(ids, targets, per_ad=6, seed=1)
    for ts in chosen.values():
        assert len(ts) == 6 and len({t["name"] for t in ts}) == 6
    covered = {aspect_class(t["width"], t["height"]) for ts in chosen.values() for t in ts}
    assert covered == {aspect_class(t["width"], t["height"]) for t in targets}


def test_choose_targets_skips_near_source_aspect(targets):
    chosen = pool.choose_targets(["a"], targets, per_ad=12, seed=0, source_aspects={"a": 1.0})
    names = {t["name"] for t in chosen["a"]}
    assert "instagram_post" not in names  # 1080x1080 is the source's own shape
    assert len(names) == 12


def test_pair_mix_half_close_contests_half_top_vs_rest():
    rows = _rows(cell_count=2, per_cell=12)
    pairs = pool.sample_pairs(rows, per_cell=6)
    for cell in {r["cell"] for r in rows}:
        ranked = sorted((r for r in rows if r["cell"] == cell),
                        key=lambda r: (-pool._total(r["metrics"]), r["candidate_id"]))
        top = {r["candidate_id"] for r in ranked[:pool.TOP_N]}
        in_top = [sum(x["candidate_id"] in top for x in pair) for pair in pairs if pair[0]["cell"] == cell]
        assert sorted(in_top) == [1, 1, 1, 2, 2, 2]  # 3 close contests, 3 top-vs-outsider


def test_pair_mix_without_outsiders_is_all_close_contests():
    pairs = pool.sample_pairs(_rows(cell_count=1, per_cell=pool.TOP_N), per_cell=6)
    assert len(pairs) == 6


def test_sample_pairs_ignores_row_order():
    rows = _rows()
    assert _ids(pool.sample_pairs(rows, seed=2)) == _ids(pool.sample_pairs(rows[::-1], seed=2))


def _l1_design(l1):
    return {"task_id": l1["task_id"], "split": "train", "source": l1["source"]}


def test_build_pool_collapses_duplicate_designs(l1, targets, tmp_path, monkeypatch):
    from engine.pipeline.layout import generate
    target = next(t for t in targets if t["name"] == "leaderboard")
    baseline = pool.build_pool([_l1_design(l1)], [target], tmp_path / "one")
    real = generate.generate_candidates
    monkeypatch.setattr(generate, "generate_candidates", lambda *a, **k: list(real(*a, **k)) * 3)
    tripled = pool.build_pool([_l1_design(l1)], [target], tmp_path / "three")
    assert [r["candidate_id"] for r in tripled] == [r["candidate_id"] for r in baseline]


def test_build_pool_drops_candidates_failing_the_gates(l1, targets, tmp_path, monkeypatch):
    from engine.pipeline.grading import gates
    target = next(t for t in targets if t["name"] == "leaderboard")
    baseline = pool.build_pool([_l1_design(l1)], [target], tmp_path / "all")
    assert len(baseline) >= 2
    real = gates.check_gates
    seen = []

    def reject_first(source, design, *a, **k):
        seen.append(design)
        return ["rejected"] if len(seen) == 1 else real(source, design, *a, **k)
    monkeypatch.setattr(gates, "check_gates", reject_first)
    fewer = pool.build_pool([_l1_design(l1)], [target], tmp_path / "gated")
    assert len(fewer) == len(baseline) - 1


def test_synthetic_asset_resolves_through_merged_dir(l1, targets, tmp_path):
    """An ad that uses a code-drawn `syn_` image (training-data/corpus/assets/)
    plans without the missing-asset flag and lands in pool rows with its brand."""
    from dev.distill.assets import SYNTH_ASSETS_DIR, merged_assets_dir
    from engine.pipeline.priority import heuristic_plan

    syn = sorted(p.name for p in SYNTH_ASSETS_DIR.glob("syn_*.png"))
    assert syn, "run python -m dev.distill.synth_images first"
    source = json.loads(json.dumps(l1["source"]))
    source["elements"].append({"id": "syn_mark", "type": "image", "x": 20, "y": 20, "width": 100, "height": 100,
                               "z_index": 9, "asset_id": syn[0], "fit": "contain"})
    with merged_assets_dir() as assets:
        assert (assets / syn[0]).is_file() and (assets / "solstice_bottle.png").is_file()
        plan = heuristic_plan(source, assets)
    assert not any("missing_asset" in u.flags for u in plan.units)
    design = {"task_id": "syn_probe", "split": "holdout", "brand": "offbrand", "source": source}
    target = next(t for t in targets if t["name"] == "leaderboard")
    rows = pool.build_pool([design], [target], tmp_path)
    assert rows and all(r["brand"] == "offbrand" for r in rows)
    assert all(r["metrics"]["placeholder"] == 1.0 for r in rows)


def test_merged_assets_dir_rejects_name_collisions(tmp_path):
    from dev.distill.assets import merged_assets_dir

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "x.png").write_bytes(b"1")
    (b / "x.png").write_bytes(b"2")
    with pytest.raises(ValueError):
        with merged_assets_dir(a, b):
            pass
