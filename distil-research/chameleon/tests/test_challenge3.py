"""challenge3: a second fresh real-ad evaluation set (dev/distill/make_challenge3.py).

The ads are Crello-derived and gitignored (corpus/challenge3.nosync/), so the
checks on them skip where they haven't been built; the loader, the harness
plumbing and the selection are tested on synthetic rows.
"""

import json

import pytest

from dev.distill import ab_engine as ab
from dev.distill import make_challenge3 as mc3
from dev.distill.assets import CHALLENGE3_ASSETS_DIR, CHALLENGE3_FOLDERS, LOOKUP_FOLDERS
from dev.distill.corpus import load_challenge2, load_challenge3, load_corpus, load_crello_dev


def _write(directory, rec):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{rec['task_id']}.json").write_text(json.dumps(rec))


def test_challenge3_split_is_separate(tmp_path):
    main, chl3 = tmp_path / "designs", tmp_path / "challenge3.nosync"
    _write(main, {"task_id": "a", "split": "train"})
    _write(chl3, {"task_id": "cr3_01", "split": "challenge3"})
    (chl3 / "summary.json").write_text("{}")
    kw = dict(directory=main, challenge_dir=tmp_path / "c", challenge2_dir=tmp_path / "c2",
              crello_dev_dir=tmp_path / "cd", challenge3_dir=chl3)
    assert [d["task_id"] for d in load_corpus(**kw)] == ["a"]
    assert [d["task_id"] for d in load_corpus(include_challenge=True, **kw)] == ["a"]
    assert [d["task_id"] for d in load_corpus("challenge3", **kw)] == ["cr3_01"]
    assert load_corpus("challenge2", **kw) == [] and load_challenge3(tmp_path / "missing") == []


def test_harness_knows_challenge3():
    assert "challenge3" in ab.EVAL_SPLITS and ab.folders_for("challenge3") == CHALLENGE3_FOLDERS
    assert CHALLENGE3_ASSETS_DIR in LOOKUP_FOLDERS
    with pytest.raises(SystemExit):  # the CLI accepts the split (argparse exits 0 on --help)
        ab.main(["run", "--split", "challenge3", "--help"])


def test_candidates_exclude_used_families():
    rows = [
        {"crello_id": "c2", "kept": True, "sig": ["f1"]},                 # in challenge2
        {"crello_id": "cd", "kept": True, "sig": ["f2"]},                 # in crello_dev
        {"crello_id": "sib2", "kept": True, "sig": ["f1", "f9"]},         # challenge2's family
        {"crello_id": "sibd", "kept": True, "sig": ["f2"]},               # crello_dev's family
        {"crello_id": "ok", "kept": True, "sig": ["f3"]},
        {"crello_id": "bad", "kept": False, "sig": ["f4"]},
    ]
    got = {r["crello_id"] for r in mc3.candidates(rows, {"c2", "cd"})}
    assert got == {"ok"}


def _row(cid, cls, photo, sig):
    return {"crello_id": cid, "aspect_class": cls, "photo": photo, "W": 100, "H": 100, "sig": [sig], "kept": True}


def test_select_tops_up_missing_classes_toward_half_text_only():
    rows = []
    for cls in ("square", "landscape", "tall", "wide"):  # no very_tall / strip, like the real pool
        rows += [_row(f"{cls}p{k}", cls, True, f"{cls}p{k}") for k in range(20)]
        rows += [_row(f"{cls}t{k}", cls, False, f"{cls}t{k}") for k in range(6)]
    picked = mc3.select(rows, 50, seed=0)
    ids = [r["crello_id"] for r in picked]
    assert len(picked) == 50 and len(set(ids)) == 50
    txt = sum(not r["photo"] for r in picked)
    assert 20 <= txt <= 26
    assert {r["aspect_class"] for r in picked} == {"square", "landscape", "tall", "wide"}
    rows.append(_row("dup", "square", True, "squarep0"))  # same family as a pick: never both
    assert sum(r["sig"] == ["squarep0"] for r in mc3.select(rows, 50, seed=0)) <= 1
    assert mc3.select(rows, 50, seed=0) == mc3.select(rows, 50, seed=0)  # deterministic


CHL3 = load_challenge3()


@pytest.mark.skipif(not CHL3, reason="challenge3 not built (python -m dev.distill.make_challenge3)")
def test_challenge3_ads_are_labelled_resolvable_and_outside_other_splits():
    from dev.distill.assets import asset_exists

    assert len(CHL3) == 50
    used = {d["crello_id"] for d in load_challenge2() + load_crello_dev()}
    for d in CHL3:
        assert d["split"] == "challenge3" and d["origin"] == "crello" and d["brand"] == "crello"
        assert d["task_id"].startswith("cr3_") and d["crello_id"] not in used
        assert any(el["type"] == "text" for el in d["source"]["elements"]), d["task_id"]
        for el in d["source"]["elements"]:
            if el["type"] == "image":
                assert el["asset_id"].startswith("cr3_") and asset_exists(el["asset_id"]), (d["task_id"], el["id"])
    scan = mc3.CHALLENGE2_DIR / "scan.jsonl"
    if scan.exists():
        rows = [json.loads(x) for x in scan.read_text().splitlines() if x.strip()]
        banned = mc3.banned_families(rows, used)
        sig = {r["crello_id"]: set(r["sig"]) for r in rows}
        assert not any(sig[d["crello_id"]] & banned for d in CHL3)
