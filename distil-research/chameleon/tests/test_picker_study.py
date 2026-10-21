"""Picker study (dev/distill/picker_study.py): choice of the K, the teacher's
answer checks and mapping, the report math, and a fake-teacher run. One
engine run on an official sample (CPU, no teacher)."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from dev.distill import picker_study as ps


def _cand(cid, template, fallback=False):
    return NS(candidate_id=cid, template=template, is_fallback=fallback, design={}, degraded_unit_ids=[])


def _rep(cid, total, rendered=True, passed=True):
    return NS(candidate_id=cid, total=total, rendered=rendered, passed_gates=passed, metrics={})


def test_choose_pick_first_family_cap_and_fallback():
    cands = [_cand("fb", "fallback", True), _cand("b1", "bisection"), _cand("b2", "bisection"),
             _cand("b3", "bisection"), _cand("r1", "row"), _cand("r2", "row"), _cand("s1", "stack"),
             _cand("x", "row")]
    reps = [_rep("fb", 0.5, rendered=False), _rep("b1", 0.95), _rep("b2", 0.94), _rep("b3", 0.93),
            _rep("r1", 0.90), _rep("r2", 0.99, rendered=False), _rep("s1", 0.80), _rep("x", 0.99, passed=False)]
    chosen = [cands[i].candidate_id for i in ps.choose(cands, reps, cands[2], k=4)]
    # pick first; b1 (bisection #2); b3 capped; rendered r1 before unrendered r2; fallback replaces the last
    assert chosen == ["b2", "b1", "r1", "fb"]
    assert "x" not in chosen  # failed the gates
    # without a passing fallback the 4th slot goes to the next by total
    reps[0].passed_gates = False
    assert [cands[i].candidate_id for i in ps.choose(cands, reps, cands[2], k=4)] == ["b2", "b1", "r1", "s1"]
    # the near-fit fallback as the pick
    reps[0].passed_gates = True
    assert [cands[i].candidate_id for i in ps.choose(cands, reps, cands[0], k=3)] == ["fb", "b1", "b2"]


def test_validator_and_schema():
    check = ps.validator(3)
    good = {"ranking": [2, 1, 3], "candidates": [{"candidate": i, "score": 3, "reason": "r"} for i in (1, 2, 3)]}
    check(good)
    for bad in ({"ranking": [1, 1, 3], "candidates": good["candidates"]},
                {"ranking": [1, 2, 3], "candidates": good["candidates"][:2]},
                {"ranking": [1, 2, 3], "candidates": [{"candidate": 1, "score": 6, "reason": "r"}] + good["candidates"][1:]}):
        with pytest.raises(ValueError):
            check(bad)
    s = ps.schema(3)
    assert s["properties"]["ranking"]["minItems"] == 3 and "1 to 3" in ps.prompt(3)


def _prepared_row(cell="t@leader"):
    cands = [{"idx": j, "candidate_id": cid, "family": fam, "template": fam, "is_fallback": fam == "fallback",
              "is_pick": j == 0, "grader_total": t, "rendered_by_grader": True, "png": f"r/{j}.png",
              "features": {"relation": 1.0}}
             for j, (cid, fam, t) in enumerate([("b1", "bisection", 0.9), ("r1", "row", 0.8), ("fb", "fallback", 0.7)])]
    return {"cell": cell, "task_id": "t", "split": "holdout", "brand": "solstice", "target": "leader",
            "grader_pick": "b1", "engine_final": "b1", "source_png": "r/src.png", "candidates": cands,
            "shown_order": [2, 0, 1]}


def test_ranking_row_maps_labels_back():
    row = _prepared_row()
    # labels: 1 -> fb (idx 2), 2 -> b1 (idx 0), 3 -> r1 (idx 1); teacher likes r1 best, then b1, then fb
    answer = {"ranking": [3, 2, 1], "candidates": [{"candidate": 1, "score": 1, "reason": "tiny"},
                                                   {"candidate": 2, "score": 3, "reason": "ok"},
                                                   {"candidate": 3, "score": 4, "reason": "good"}]}
    out = ps.ranking_row(row, answer, "fake")
    assert out["ranking"] == [1, 0, 2] and out["consistent"]
    by = {c["candidate_id"]: c for c in out["candidates"]}
    assert (by["r1"]["teacher_rank"], by["r1"]["teacher_score"], by["r1"]["reason"]) == (1, 4, "good")
    assert (by["fb"]["label"], by["fb"]["teacher_rank"]) == (1, 3) and by["b1"]["features"] == {"relation": 1.0}
    rep = ps.compute_report([out])
    assert rep["agree"] == 0 and rep["headroom"] == 1 and rep["pick_mean_rank"] == 2
    assert rep["families"]["row"]["teacher_top"] == 1 and rep["fallback_shown"] == 1 and rep["fallback_top"] == 0
    assert ps.kendall_tau([0, 1, 2], [0, 1, 2]) == 1 and ps.kendall_tau([0, 1, 2], [2, 1, 0]) == -1


class FakeTeacher:
    name = "fake"

    def __init__(self):
        self.n = 0

    def ask(self, prompt, schema, images, validate=None):
        self.n += 1
        k = schema["properties"]["ranking"]["maxItems"]
        assert len(images) == k + 1 and all(Path(p).exists() for p in images)
        ans = {"ranking": list(range(1, k + 1)),
               "candidates": [{"candidate": i, "score": 5 - i if i < 5 else 1, "reason": "r"} for i in range(1, k + 1)]}
        validate(ans)
        return ans


def test_rank_resumes_shards_and_reports(tmp_path):
    name = "st"
    sdir = tmp_path / name
    (sdir / "r").mkdir(parents=True)
    for f in ("src", "0", "1", "2"):
        (sdir / "r" / f"{f}.png").write_bytes(b"png")
    (sdir / "study.json").write_text(json.dumps({"name": name, "split": "holdout", "env": {}, "k": 3, "seed": 0,
                                                 "max_per_family": 2}))
    rows = [_prepared_row("a@x"), _prepared_row("b@x"), {"cell": "c@x", "skip": "only 1"}]
    (sdir / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    t = FakeTeacher()
    got = ps.rank(name, t, root=tmp_path, shard=(0, 2), log=lambda *_: None)
    got += ps.rank(name, t, root=tmp_path, shard=(1, 2), log=lambda *_: None)
    assert len(got) == 2 and t.n == 2
    assert ps.rank(name, t, root=tmp_path, log=lambda *_: None) == [] and t.n == 2  # resumable across shards
    ps.judge_mod.merge_shards(sdir / "rankings.jsonl")
    assert len(ps.load_rankings(name, tmp_path)) == 2
    text = ps.write_report(name, tmp_path).read_text()
    assert "Cells ranked by the teacher: 2" in text and "1 skipped" in text


def test_crello_rows_go_to_nosync_and_eval_splits_are_refused(tmp_path):
    assert ps.data_dir("x", tmp_path, "crello_dev").name == "private.nosync"
    assert ps.data_dir("x", tmp_path, "holdout") == tmp_path / "x"
    for split in ("challenge", "challenge2", "challenge3"):
        with pytest.raises(ValueError):
            ps.prepare("x", split, root=tmp_path)


def test_prepare_one_sample_cell_end_to_end(tmp_path):
    from dev.distill import ab_engine
    cells, designs, folders = ab_engine.split_cells("samples")
    cells = [c for c in cells if c["cell"] == "warmup_01@leaderboard"]
    rows = ps.prepare("e2e", "samples", root=tmp_path, cells=cells, designs=designs, folders=folders,
                      workers=1, log=lambda *_: None)
    assert len(rows) == 1 and "skip" not in rows[0], rows
    row = rows[0]
    assert 2 <= len(row["candidates"]) <= ps.K and sorted(row["shown_order"]) == list(range(len(row["candidates"])))
    assert sum(c["is_pick"] for c in row["candidates"]) >= 1 and row["candidates"][0]["is_pick"]
    fams = [c["family"] for c in row["candidates"]]
    assert all(fams.count(f) <= ps.MAX_PER_FAMILY for f in fams)
    for c in row["candidates"]:
        assert (tmp_path / "e2e" / c["png"]).stat().st_size > 0
        assert "relation" in c["features"] and "flaws" in c["features"] and c["features"]["metrics"]
    assert (tmp_path / "e2e" / row["source_png"]).exists()
    import os
    assert "CHAMELEON_STRATEGIES" not in os.environ  # restored
