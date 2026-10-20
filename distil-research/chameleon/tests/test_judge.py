"""The pairwise judge, with a fake teacher: no test calls claude or an API."""

import json
import math
from pathlib import Path

import pytest

from dev.distill.judge import RUBRIC, RUBRIC_VERSION, judge_pairs, pick_cells, position_bias, swap_agreement


def _row(cell, cid, task="t1", split="train"):
    return {"cell": cell, "task_id": task, "split": split, "target": "300x250", "candidate_id": cid,
            "png": f"renders/{task}/{cid}.png", "is_fallback": False, "metrics": {}}


def _pairs(n, cell="c1"):
    return [(_row(cell, f"a{i}"), _row(cell, f"b{i}")) for i in range(n)]


class FakeTeacher:
    """Always picks the first-shown image; records every call."""
    name = "fake"

    def __init__(self, fail_on_call=None, winner="a"):
        self.calls = []
        self.fail_on_call = fail_on_call
        self.winner = winner

    def ask(self, prompt, schema, images, validate=None):
        # contents are read now: the judge deletes its staged copies after the call
        self.calls.append({"prompt": prompt, "schema": schema, "images": list(images),
                           "contents": [p.read_text() for p in images]})
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("rate limited")
        n = schema["properties"]["verdicts"]["maxItems"]
        return {"verdicts": [{"pair": i + 1, "winner": self.winner, "confidence": 2, "score_a": 3, "score_b": 2, "reason": "r"}
                             for i in range(n)]}


def _src(task_id):
    return Path(f"/pool/renders/{task_id}/source.png")


def _touch(pool_dir, pairs):
    """Make every png the pairs name; each file's text is its own relative path."""
    for a, b in pairs:
        for r in (a, b):
            (pool_dir / r["png"]).parent.mkdir(parents=True, exist_ok=True)
            (pool_dir / r["png"]).write_text(r["png"])


def _run(tmp_path, pairs, teacher, **kw):
    kw.setdefault("swap_share", 0.0)
    _touch(tmp_path, pairs)
    src = tmp_path / "renders" / "t1" / "source.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("SOURCE")
    kw.setdefault("source_png", lambda task_id: src)
    return judge_pairs(pairs, teacher, kw.pop("source_png"), tmp_path / "j.jsonl", pool_dir=tmp_path, **kw)


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_prompt_shows_original_plus_eight_images_and_rubric(tmp_path):
    t = FakeTeacher()
    _run(tmp_path, _pairs(4), t, pairs_per_call=4)
    assert len(t.calls) == 1 and len(t.calls[0]["images"]) == 9
    assert "No unwanted collisions" in t.calls[0]["prompt"]


def test_image_order_neutral_names_and_full_rubric(tmp_path):
    t = FakeTeacher()
    pairs = _pairs(2)
    rows = _run(tmp_path, pairs, t, pairs_per_call=4)
    call = t.calls[0]
    assert [p.name for p in call["images"]] == [f"image_{i}.png" for i in range(1, 6)]
    expected = ["SOURCE"]
    for a, b in [(r["a"], r["b"]) for r in rows]:  # rows come back in pair order: original, then A, B of pair 1, then pair 2
        expected += [f"renders/t1/{a}.png", f"renders/t1/{b}.png"]
    assert call["contents"] == expected
    assert RUBRIC in call["prompt"]
    assert "Read all of the listed image files" in call["prompt"]


def test_call_never_holds_both_orders_of_a_pair(tmp_path):
    t = FakeTeacher()
    _run(tmp_path, _pairs(4), t, pairs_per_call=8, swap_share=1.0)
    assert len(t.calls) == 2  # originals together, swapped re-asks together

    def shown(call):
        c = call["contents"]
        return {(c[i], c[i + 1]) for i in range(1, len(c), 2)}
    first, second = shown(t.calls[0]), shown(t.calls[1])
    assert len(first) == len(second) == 4
    assert second == {(y, x) for x, y in first} and not first & second


BAD_ANSWERS = [
    {"verdicts": [{"pair": 1, "winner": "a", "confidence": 2}]},                       # no reason
    {"verdicts": [{"pair": 1, "winner": "a", "reason": "r"}]},                         # no confidence
    {"verdicts": ["a"]},                                                               # not a dict
    {"verdicts": [{"pair": 1, "winner": "a", "confidence": 5, "score_a": 3, "score_b": 2, "reason": "r"}]},        # confidence out of range
    {"verdicts": [{"pair": 1, "winner": "x", "confidence": 1, "score_a": 3, "score_b": 2, "reason": "r"}]},        # bad winner
    {"verdicts": [{"pair": 1, "winner": "a", "confidence": 1, "score_a": 3, "score_b": 2, "reason": "r"}] * 2},    # pair 1 twice
    {"verdicts": [{"pair": 1, "winner": "a", "confidence": 1, "score_a": 3, "score_b": 2, "reason": 7}]},          # reason not text
    {"verdicts": [{"pair": 1, "winner": "a", "confidence": 1, "score_b": 2, "reason": "r"}]},          # no score_a
    {"verdicts": [{"pair": 1, "winner": "a", "confidence": 1, "score_a": 6, "score_b": 2, "reason": "r"}]},  # score out of range
    {"verdicts": None},
    ["not", "a", "dict"],
]
GOOD = {"verdicts": [{"pair": 1, "winner": "a", "confidence": 2, "score_a": 3, "score_b": 2, "reason": "r"}]}


class Scripted(FakeTeacher):
    """Gives a scripted answer per call; optionally runs the validate hook like the real teacher."""
    def __init__(self, answers, use_validate):
        super().__init__()
        self.answers, self.use_validate = answers, use_validate

    def ask(self, prompt, schema, images, validate=None):
        self.calls.append(1)
        answer = self.answers[len(self.calls) - 1]
        if self.use_validate and validate:
            validate(answer)
        return answer


@pytest.mark.parametrize("use_validate", [True, False])
@pytest.mark.parametrize("bad", BAD_ANSWERS)
def test_bad_verdicts_are_skipped_without_crashing(tmp_path, bad, use_validate):
    t = Scripted([bad, GOOD], use_validate)
    rows = _run(tmp_path, _pairs(2), t, pairs_per_call=1)
    assert len(t.calls) == 2  # the bad batch is skipped and the next one still runs
    assert len(rows) == 1 and len(_lines(tmp_path / "j.jsonl")) == 1


def test_resume_skips_judged_pairs(tmp_path):
    t = FakeTeacher()
    _run(tmp_path, _pairs(4), t)
    assert len(t.calls) == 1
    t2 = FakeTeacher()
    new = _run(tmp_path, _pairs(6), t2)
    assert len(new) == 2 and len(_lines(tmp_path / "j.jsonl")) == 6
    assert len(t2.calls) == 1 and len(t2.calls[0]["images"]) == 1 + 2 * 2
    other = FakeTeacher()
    other.name = "other"
    _run(tmp_path, _pairs(4), other)
    assert len(other.calls) == 1  # a different teacher is not skipped


def test_order_shuffled_and_recorded(tmp_path):
    rows = _run(tmp_path, _pairs(40), FakeTeacher(), seed=3)
    firsts = [r["a"].startswith("a") for r in rows]
    assert 0 < sum(firsts) < 40
    for r in rows:
        assert r["a"][1:] == r["b"][1:] and r["swapped"] is False
        assert r["task_id"] == "t1" and r["split"] == "train" and r["teacher"] == "fake"
    (tmp_path / "x").mkdir()
    again = _run(tmp_path / "x", _pairs(40), FakeTeacher(), seed=3)
    assert [r["a"] for r in again] == [r["a"] for r in rows]


def test_swap_share_reasks_swapped(tmp_path):
    rows = _run(tmp_path, _pairs(10), FakeTeacher(), swap_share=0.2)
    swapped = [r for r in rows if r["swapped"]]
    assert len(swapped) == 2
    firsts = {(r["a"], r["b"]) for r in rows if not r["swapped"]}
    for r in swapped:
        assert (r["b"], r["a"]) in firsts


def test_position_bias_and_swap_agreement():
    j = [{"cell": "c", "a": "x", "b": "y", "winner": "a", "swapped": False},
         {"cell": "c", "a": "y", "b": "x", "winner": "b", "swapped": True},   # x both times: agree
         {"cell": "c", "a": "p", "b": "q", "winner": "a", "swapped": False},
         {"cell": "c", "a": "q", "b": "p", "winner": "a", "swapped": True},   # p then q: disagree
         {"cell": "c", "a": "m", "b": "n", "winner": "tie", "swapped": False}]
    assert position_bias(j) == pytest.approx(3 / 4)
    assert swap_agreement(j) == pytest.approx(0.5)
    assert math.isnan(position_bias([{"winner": "tie"}]))


def test_teacher_failure_keeps_progress(tmp_path):
    t = FakeTeacher(fail_on_call=2)
    rows = _run(tmp_path, _pairs(8), t, pairs_per_call=4)
    assert len(rows) == 4 and len(_lines(tmp_path / "j.jsonl")) == 4
    assert len(t.calls) == 2
    t2 = FakeTeacher()
    _run(tmp_path, _pairs(8), t2, pairs_per_call=4)
    assert len(t2.calls) == 1 and len(_lines(tmp_path / "j.jsonl")) == 8


def test_mismatched_answer_skips_batch(tmp_path):
    class Bad(FakeTeacher):
        def ask(self, prompt, schema, images, validate=None):
            return {"verdicts": [{"pair": 9, "winner": "a", "confidence": 1, "score_a": 3, "score_b": 2, "reason": ""}] * 4}
    assert _run(tmp_path, _pairs(4), Bad()) == []


def _cell_rows(n_cells, split, brand, prefix):
    return [{**_row(f"{prefix}{i}@t", "c0", task=f"{prefix}{i}", split=split), "brand": brand} for i in range(n_cells)]


def test_pick_cells_is_stratified_seeded_and_capped():
    rows = (_cell_rows(60, "train", "solstice", "ts") + _cell_rows(20, "train", "offbrand", "to")
            + _cell_rows(15, "holdout", "solstice", "hs") + _cell_rows(5, "holdout", "offbrand", "ho"))
    cells = pick_cells(rows, 20, seed=0)
    assert len(cells) == 20
    groups = {}
    for r in rows:
        if r["cell"] in cells:
            groups[(r["split"], r["brand"])] = groups.get((r["split"], r["brand"]), 0) + 1
    assert groups == {("train", "solstice"): 12, ("train", "offbrand"): 4,
                      ("holdout", "solstice"): 3, ("holdout", "offbrand"): 1}
    assert pick_cells(rows, 20, seed=0) == cells
    assert pick_cells(rows, 20, seed=1) != cells
    assert pick_cells(rows, 500, seed=0) == {r["cell"] for r in rows}


def test_pick_cells_keeps_every_group_represented():
    rows = _cell_rows(97, "train", "solstice", "ts") + _cell_rows(3, "holdout", "offbrand", "ho")
    cells = pick_cells(rows, 10, seed=0)
    assert any(c.startswith("ho") for c in cells)


def test_rows_carry_scores_and_rubric_version_and_resume_ignores_other_versions(tmp_path):
    pairs = _pairs(2)
    (tmp_path / "j.jsonl").write_text(json.dumps({"cell": "c1", "a": "a0", "b": "b0", "swapped": False, "teacher": FakeTeacher.name,
                               "rubric_version": "older000"}) + "\n")
    rows = _run(tmp_path, pairs, FakeTeacher())
    assert len(rows) == 2  # the row judged under another rubric does not count as done
    assert all(r["rubric_version"] == RUBRIC_VERSION and r["score_a"] == 3 and r["score_b"] == 2 for r in rows)


def test_shards_split_cells_disjointly_and_cover_everything(tmp_path):
    pairs = [p for c in range(6) for p in _pairs(2, cell=f"c{c}")]
    _touch(tmp_path, pairs)
    seen = []
    for k in range(3):
        rows = judge_pairs(pairs, FakeTeacher(), lambda t: _src_file(tmp_path), tmp_path / f"s{k}.jsonl",
                           swap_share=0.0, pool_dir=tmp_path, shard=(k, 3))
        seen.append({r["cell"] for r in rows})
    assert set().union(*seen) == {f"c{c}" for c in range(6)}
    assert sum(len(s) for s in seen) == 6  # no cell judged by two shards


def test_shard_skips_pairs_already_in_the_main_file(tmp_path):
    pairs = _pairs(2)
    _touch(tmp_path, pairs)
    main_file = tmp_path / "main.jsonl"
    first = judge_pairs(pairs[:1], FakeTeacher(), lambda t: _src_file(tmp_path), main_file, swap_share=0.0, pool_dir=tmp_path)
    rows = judge_pairs(pairs, FakeTeacher(), lambda t: _src_file(tmp_path), tmp_path / "shard.jsonl", swap_share=0.0,
                       pool_dir=tmp_path, shard=(0, 1), done_files=[main_file])
    assert len(first) == 1 and len(rows) == 1


def test_merge_shards_appends_new_rows_once_and_removes_shards(tmp_path):
    from dev.distill.judge import merge_shards
    main_file = tmp_path / "judgments.jsonl"
    row = {"cell": "c1", "a": "x", "b": "y", "swapped": False, "teacher": "t", "rubric_version": "v"}
    main_file.write_text(json.dumps(row) + "\n")
    (tmp_path / "judgments.shard0of2.jsonl").write_text(json.dumps(row) + "\n" + json.dumps({**row, "a": "z"}) + "\n")
    (tmp_path / "judgments.shard1of2.jsonl").write_text(json.dumps({**row, "cell": "c2"}) + "\n")
    assert merge_shards(main_file) == 2
    assert len(_lines(main_file)) == 3
    assert not list(tmp_path.glob("judgments.shard*"))


def _src_file(tmp_path):
    src = tmp_path / "renders" / "t1" / "source.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("SOURCE")
    return src
