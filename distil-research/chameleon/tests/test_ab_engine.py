"""A/B harness with a fake teacher and the 3 sample inputs: no claude, no network."""

import json
import os
from pathlib import Path

import pytest

from dev.distill import ab_engine as ab
from dev.distill.paths import RENDERS_SUBDIR

TARGET = {"name": "leaderboard", "width": 728, "height": 90}
STEMS = ["chameleon_sample_level1_warmup", "chameleon_sample_level2_core", "chameleon_sample_level3_advanced"]


@pytest.fixture
def world(tmp_path):
    designs, pool_rows = {}, []
    for i, stem in enumerate(STEMS):
        task = json.loads((Path("input") / f"{stem}.json").read_text())
        tid = f"t{i}"
        split = "holdout" if i < 2 else "train"
        brand = "solstice" if i == 0 else "other"
        designs[tid] = {"task_id": tid, "split": split, "brand": brand, "source": task["source"]}
        pool_rows.append({"cell": f"{tid}@leaderboard", "task_id": tid, "split": split, "brand": brand,
                          "target": "leaderboard"})
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool_rows))
    for tid in designs:
        (pool / RENDERS_SUBDIR / tid).mkdir(parents=True)
        (pool / RENDERS_SUBDIR / tid / "source.png").write_bytes(b"src")
    return {"designs": designs, "targets": {"leaderboard": TARGET}, "pool": pool, "root": tmp_path / "ab"}


def _fake_resize(monkeypatch, calls=None):
    """Variant X (env AB_X=1) returns a differently-sized-source design on t1 only."""
    real = ab._resize

    def fake(source, canvas):
        if calls is not None:
            calls.append(os.environ.get("AB_X"))
        design, plan, best, reports = real(source, canvas)
        if os.environ.get("AB_X") == "1" and source is not None and source.get("canvas", {}).get("width") == _W1[0]:
            design = json.loads(json.dumps(design))
            design["canvas"]["width"] += 0  # unchanged geometry...
            design["_variant"] = "x"  # ...but a different design
        return design, plan, best, reports
    monkeypatch.setattr(ab, "_resize", fake)


_W1 = [None]


def _cells(world, split="holdout"):
    return ab.pool_cells(world["pool"], split)


def _run(world, name, env, **kw):
    return ab.run_variant(name, env, _cells(world), world["designs"], world["targets"], root=world["root"],
                          workers=1, log=lambda *_: None, **kw)


def test_pool_cells_split_and_sample(world):
    assert [c["task_id"] for c in _cells(world)] == ["t0", "t1"]
    assert len(ab.pool_cells(world["pool"], "all")) == 3
    assert len(ab.pool_cells(world["pool"], "all", n=3)) == 3


def test_run_records_winners_and_sets_env(world, monkeypatch):
    calls = []
    _W1[0] = world["designs"]["t1"]["source"]["canvas"]["width"]
    _fake_resize(monkeypatch, calls)
    rows = _run(world, "x", {"AB_X": "1"})
    assert len(rows) == 2 and calls == ["1", "1"]
    assert "AB_X" not in os.environ  # restored
    for r in rows:
        assert {"cell", "task_id", "split", "brand", "target", "variant", "candidate_id", "is_fallback",
                "design_hash", "png", "seconds"} <= set(r)
        assert (world["root"] / "x" / r["png"]).stat().st_size > 0
    assert len(ab.load_winners("x", world["root"])) == 2


def test_resume_skips_done_cells(world, monkeypatch):
    calls = []
    _fake_resize(monkeypatch, calls)
    _run(world, "x", {})
    assert len(calls) == 2
    assert _run(world, "x", {}) == []
    assert len(calls) == 2
    (world["root"] / "x" / "renders.nosync" / "t0" / "leaderboard.png").unlink()  # lost render is redone
    assert len(_run(world, "x", {})) == 1


def test_shards_and_merge(world, monkeypatch):
    _fake_resize(monkeypatch)
    for k in range(2):
        _run(world, "x", {}, shard=(k, 2))
    assert len(ab.load_winners("x", world["root"])) == 2  # visible before merge
    assert ab.merge_winners("x", world["root"]) == 2
    assert len((world["root"] / "x" / "winners.jsonl").read_text().splitlines()) == 2


class Fake:
    name = "fake"

    def __init__(self, winner="b", confidence=3):
        self.winner, self.confidence, self.n = winner, confidence, 0

    def ask(self, prompt, schema, images, validate=None):
        self.n += 1
        k = schema["properties"]["verdicts"]["maxItems"]
        return {"verdicts": [{"pair": i + 1, "winner": self.winner, "confidence": self.confidence,
                              "score_a": 2, "score_b": 4, "reason": "r"} for i in range(k)]}


def _two_variants(world, monkeypatch):
    _W1[0] = world["designs"]["t1"]["source"]["canvas"]["width"]
    _fake_resize(monkeypatch)
    _run(world, "base", {})
    _run(world, "flex", {"AB_X": "1"})


def test_judge_only_changed_cells(world, monkeypatch):
    _two_variants(world, monkeypatch)
    pairs, compared, same = ab.changed_pairs("base", "flex", world["root"])
    changed = {a["cell"] for a, _ in pairs}
    assert compared == 2 and same + len(changed) == 2 and changed <= {"t0@leaderboard", "t1@leaderboard"}
    assert changed  # the env-sensitive variant really changed something
    teacher = Fake()
    rows = ab.judge_variants("base", "flex", teacher, root=world["root"], pool_dir=world["pool"])
    assert rows and {r["cell"] for r in rows} == changed
    assert all(r["split"] == "holdout" and r["task_id"] for r in rows)
    assert (world["root"] / "base__vs__flex" / "judgments.jsonl").exists()
    n = teacher.n
    assert ab.judge_variants("base", "flex", teacher, root=world["root"], pool_dir=world["pool"]) == []
    assert teacher.n == n


def test_identical_variants_need_no_judging(world, monkeypatch):
    _fake_resize(monkeypatch)
    _run(world, "p", {})
    _run(world, "q", {})
    teacher = Fake()
    assert ab.judge_variants("p", "q", teacher, root=world["root"], pool_dir=world["pool"]) == []
    assert teacher.n == 0
    rep = ab.compute_report("p", "q", world["root"])
    assert (rep["compared"], rep["identical"], rep["changed"]) == (2, 2, 0)
    assert "NO VERDICT" in ab.verdict_line(rep)


def test_report_math(tmp_path):
    root = tmp_path
    def w(name, cells):
        d = root / name
        d.mkdir(parents=True)
        (d / "winners.jsonl").write_text("".join(json.dumps(
            {"cell": c, "task_id": c, "split": "holdout", "brand": br, "target": "t", "variant": name,
             "candidate_id": "x", "is_fallback": False, "design_hash": h, "png": "p.png", "seconds": s}) + "\n"
            for c, br, h, s in cells))
    w("A", [("c1", "solstice", "h1", 1.0), ("c2", "solstice", "h2", 1.0), ("c3", "off", "h3", 2.0), ("c4", "off", "same", 3.0)])
    w("B", [("c1", "solstice", "g1", 2.0), ("c2", "solstice", "g2", 2.0), ("c3", "off", "g3", 4.0), ("c4", "off", "same", 3.0)])
    def j(cell, win, conf, sa=2, sb=4, swapped=False):
        return {"cell": cell, "task_id": cell, "split": "holdout", "a": "A", "b": "B", "winner": win,
                "confidence": conf, "score_a": sa, "score_b": sb, "reason": "", "teacher": "f",
                "rubric_version": ab.judge_mod.RUBRIC_VERSION, "swapped": swapped}
    d = root / "A__vs__B"
    d.mkdir()
    (d / "judgments.jsonl").write_text("".join(json.dumps(x) + "\n" for x in [
        j("c1", "b", 3), j("c2", "b", 2), j("c3", "a", 1)]))  # c3 low confidence -> unsure
    r = ab.compute_report("A", "B", root)
    assert (r["compared"], r["identical"], r["changed"], r["judged"]) == (4, 1, 3, 3)
    assert r["wins"] == {"a": 0, "b": 2, "unsure": 1}
    assert r["by_brand"] == {"off": {"a": 0, "b": 0, "unsure": 1}, "solstice": {"a": 0, "b": 2, "unsure": 0}}
    assert r["mean_score"] == {"A": 2.0, "B": 4.0}
    assert r["mean_seconds"] == {"A": 1.75, "B": 2.75}
    path = ab.write_report("A", "B", root)
    text = path.read_text()
    assert text.splitlines()[2].startswith("**Verdict: B is")
    assert "Identical winners: 1" in text


def test_cell_verdict_votes():
    j = lambda w, s: {"cell": "c", "a": "A", "b": "B", "winner": w, "confidence": 3, "swapped": s}
    assert ab.cell_verdicts([j("a", False), j("b", True)], "A", "B") == {"c": "unsure"}
    assert ab.cell_verdicts([j("a", False), j("a", True)], "A", "B") == {"c": "a"}


def test_parse_env():
    assert ab.parse_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(ValueError):
        ab.parse_env(["bad"])


def test_sample_pairs_caps_and_keeps_every_brand():
    from dev.distill.ab_engine import sample_pairs
    def pair(i, brand):
        side = {"cell": f"c{i}", "brand": brand}
        return (side, side)
    pairs = [pair(i, "solstice") for i in range(80)] + [pair(100 + i, "offbrand") for i in range(20)]
    got = sample_pairs(pairs, 10, seed=0)
    assert len(got) == 10
    assert {p[0]["brand"] for p in got} == {"solstice", "offbrand"}
    assert got == sample_pairs(pairs, 10, seed=0)
    assert sample_pairs(pairs, None, seed=0) == pairs and sample_pairs(pairs, 500, seed=0) == pairs


def test_samples_split_is_the_3_samples_x_every_target(tmp_path):
    from dev.distill.corpus import SAMPLE_TASKS, load_corpus
    smp = load_corpus("samples")
    assert sorted(d["task_id"] for d in smp) == sorted(SAMPLE_TASKS)
    assert all(d["split"] == "samples" for d in smp)
    assert {d["task_id"]: d["split"] for d in load_corpus("train")}["warmup_01"] == "train"  # corpus unchanged
    by_stem = {f"chameleon_sample_level{i}": t for i, t in ((1, "warmup_01"), (2, "core_01"), (3, "advanced_01"))}
    for path in Path("input").glob("*.json"):
        tid = next(t for s, t in by_stem.items() if path.stem.startswith(s))
        assert next(d for d in smp if d["task_id"] == tid)["source"] == json.loads(path.read_text())["source"]
    cells, designs, folders = ab.split_cells("samples")
    n_targets = len(json.loads(ab.TARGETS_PATH.read_text()))
    assert len(cells) == 3 * n_targets and folders == () and set(designs) == set(SAMPLE_TASKS)
    assert "warmup_01@instagram_post" in {c["cell"] for c in cells}  # near-source targets kept
    assert {c["split"] for c in cells} == {"samples"} and "samples" in ab.SOURCE_SPLITS
    assert "samples" not in ab.EVAL_SPLITS  # other tools' eval-split handling unchanged


def test_judge_uses_source_renders_for_samples(world, monkeypatch, tmp_path):
    """samples cells take the judge's original from _sources.nosync, not the pool."""
    _W1[0] = world["designs"]["t1"]["source"]["canvas"]["width"]
    _fake_resize(monkeypatch)
    cells = [dict(c, split="samples") for c in _cells(world)]
    for name, env in (("sa", {}), ("sb", {"AB_X": "1"})):
        ab.run_variant(name, env, cells, world["designs"], world["targets"], root=world["root"], workers=1,
                       log=lambda *_: None)
    for tid in ("t0", "t1"):
        p = ab.source_png(world["root"], tid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"smp-src")
    teacher = Fake()
    rows = ab.judge_variants("sa", "sb", teacher, root=world["root"], pool_dir=tmp_path / "no_pool")
    assert rows and all(r["split"] == "samples" for r in rows)
