"""The generalisation yardstick: the challenge corpus (training-data/corpus/challenge/),
the challenge path through the A/B harness, and the worst-slice report.

Fake teacher only: no claude, no network."""

import json
from pathlib import Path

import pytest

from dev.distill import ab_engine as ab
from dev.distill.assets import asset_exists, merged_assets_dir
from dev.distill.corpus import load_challenge, load_corpus
from dev.distill.paths import CHALLENGE_DIR, RENDERS_SUBDIR

TARGETS = json.loads(Path("dev/targets.json").read_text())
CHALLENGE = load_challenge()
FONTS_DIR = Path("brand_kit/fonts")


# ---------------------------------------------------------------- loader

def _write(folder: Path, rec: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{rec['task_id']}.json").write_text(json.dumps(rec))


def test_loader_keeps_challenge_out_by_default(tmp_path):
    main, chl = tmp_path / "designs", tmp_path / "challenge"
    _write(main, {"task_id": "a", "split": "train"})
    _write(main, {"task_id": "b", "split": "holdout"})
    _write(chl, {"task_id": "z", "split": "challenge"})
    assert [d["task_id"] for d in load_corpus(directory=main)] == ["a", "b"]
    assert [d["task_id"] for d in load_corpus("holdout", directory=main)] == ["b"]
    both = load_corpus(directory=main, include_challenge=True, challenge_dir=chl)
    assert [d["task_id"] for d in both] == ["a", "b", "z"]
    assert [d["task_id"] for d in load_corpus("challenge", directory=main, challenge_dir=chl)] == ["z"]
    assert [d["task_id"] for d in load_challenge(chl)] == ["z"]


def test_real_corpus_loader_unchanged():
    ids = {d["task_id"] for d in load_corpus()}
    assert ids and not ids & {d["task_id"] for d in CHALLENGE}
    assert all(d["split"] in ("train", "holdout") for d in load_corpus())


# ---------------------------------------------------------------- the challenge ads

def test_challenge_counts_and_labels():
    assert 36 <= len(CHALLENGE) <= 48
    ids = [d["task_id"] for d in CHALLENGE]
    assert len(ids) == len(set(ids))
    for d in CHALLENGE:
        assert d["split"] == "challenge" and d["origin"] == "challenge" and d["brand"] == "challenge", d["task_id"]
        assert d["challenge_brand"] and d["palette"] in ("light", "dark", "saturated", "monochrome"), d["task_id"]
        assert d["notes"] and d["features"], d["task_id"]
        el_ids = [el["id"] for el in d["source"]["elements"]]
        assert len(el_ids) == len(set(el_ids)), d["task_id"]
    brands = {d["challenge_brand"] for d in CHALLENGE}
    assert 4 <= len(brands) <= 5
    assert {d["palette"] for d in CHALLENGE} == {"light", "dark", "saturated", "monochrome"}
    assert len(list(CHALLENGE_DIR.glob("README.md"))) == 1


def test_challenge_ads_differ_from_the_corpus():
    corpus = load_corpus()
    sizes = {(d["source"]["canvas"]["width"], d["source"]["canvas"]["height"]) for d in CHALLENGE}
    for want in [(600, 900), (1200, 300), (970, 250), (160, 600), (1500, 500)]:
        assert want in sizes, want
    corpus_sizes = {(d["source"]["canvas"]["width"], d["source"]["canvas"]["height"]) for d in corpus}
    assert len(sizes - corpus_sizes) >= 8  # mostly sizes the corpus never had
    images = {el["asset_id"] for d in CHALLENGE for el in d["source"]["elements"] if el["type"] == "image"}
    assert images and all(a.startswith("chl_") for a in images)
    assert len(images) >= 15
    counts = [len(d["source"]["elements"]) for d in CHALLENGE]
    assert min(counts) == 1 and max(counts) >= 20
    feats = {f for d in CHALLENGE for f in d["features"]}
    for f in ["two-products", "quote-led", "price-led", "event-date", "list-led", "logo-centrepiece",
              "one-element", "20-plus-elements", "long-legal", "rotated-sticker", "full-bleed-panel"]:
        assert f in feats, f


def test_challenge_assets_and_fonts_exist():
    for d in CHALLENGE:
        assert not d["expects_missing_asset"], d["task_id"]
        for el in d["source"]["elements"]:
            if el["type"] == "image":
                assert asset_exists(el["asset_id"]), (d["task_id"], el["asset_id"])
            if el["type"] == "text":
                assert el["font_family"] in ("Solstice Serif", "Solstice Sans"), (d["task_id"], el["id"])


def test_challenge_ads_pass_the_quality_gate():
    from dev.distill.make_challenge import quality_issues

    with merged_assets_dir() as assets:
        for d in CHALLENGE:
            assert quality_issues(d, assets) == [], d["task_id"]


@pytest.mark.parametrize("design", CHALLENGE, ids=[d["task_id"] for d in CHALLENGE])
def test_challenge_ads_plan_and_render(design):
    from engine.pipeline.priority import heuristic_plan
    from engine.pipeline.render import render_to_image

    with merged_assets_dir() as assets:
        plan = heuristic_plan(design["source"], assets)
        assert {e for u in plan.units for e in u.element_ids} == {el["id"] for el in design["source"]["elements"]}
        assert max(render_to_image(design["source"], assets_dir=assets, max_size=200).size) <= 200


# ---------------------------------------------------------------- A/B harness: challenge cells

def _design(tid, w, h, brand="challenge", split="challenge"):
    src = json.loads(Path("input/chameleon_sample_level1_warmup.json").read_text())["source"]
    src = json.loads(json.dumps(src))
    src["canvas"] = {"width": w, "height": h}
    return {"task_id": tid, "split": split, "brand": brand, "source": src}


def test_challenge_cells_cover_every_target_except_near_source():
    designs = [_design("sq", 1000, 1000), _design("lb", 728, 90)]
    cells = ab.challenge_cells(designs, TARGETS)
    by_task = {}
    for c in cells:
        assert c["cell"] == f"{c['task_id']}@{c['target']}"
        assert c["split"] == "challenge" and c["brand"] == "challenge"
        by_task.setdefault(c["task_id"], set()).add(c["target"])
    names = {t["name"] for t in TARGETS}
    assert by_task["sq"] == names - {"instagram_post"}
    assert by_task["lb"] == names - {"leaderboard"}
    assert cells == sorted(cells, key=lambda c: c["cell"])
    assert len(ab.challenge_cells(designs, TARGETS, n=5)) == 5


def test_write_sources_renders_each_challenge_ad_once(tmp_path):
    designs = [_design("sq", 1080, 1080)]
    assert ab.write_sources(designs, tmp_path) == 1
    png = tmp_path / ab.SOURCES_SUBDIR / "sq" / "source.png"
    assert png.stat().st_size > 0
    assert ab.write_sources(designs, tmp_path) == 0  # resumable


class PeekTeacher:
    """Fake teacher that remembers the bytes of the 'original' image it was shown."""
    name = "fake"

    def __init__(self):
        self.originals = []

    def ask(self, prompt, schema, images, validate=None):
        self.originals.append(Path(images[0]).read_bytes())
        k = schema["properties"]["verdicts"]["maxItems"]
        return {"verdicts": [{"pair": i + 1, "winner": "b", "confidence": 3, "score_a": 2, "score_b": 4,
                              "reason": "r"} for i in range(k)]}


def _winners(root, name, rows):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "winners.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        (d / r["png"]).parent.mkdir(parents=True, exist_ok=True)
        (d / r["png"]).write_bytes(b"png-" + name.encode())


def _row(cell, target, brand, split, h, name):
    tid = cell.split("@")[0]
    return {"cell": cell, "task_id": tid, "split": split, "brand": brand, "target": target, "variant": name,
            "candidate_id": "x", "is_fallback": False, "design_hash": h, "png": f"{RENDERS_SUBDIR}/{tid}/{target}.png",
            "seconds": 1.0}


def test_judge_shows_the_challenge_source_render(tmp_path):
    root, pool = tmp_path / "ab", tmp_path / "pool"
    for name, h in (("A", "h1"), ("B", "h2")):
        _winners(root, name, [_row("chl@story", "story", "challenge", "challenge", h, name),
                              _row("hold@story", "story", "solstice", "holdout", h, name)])
    (root / ab.SOURCES_SUBDIR / "chl").mkdir(parents=True)
    (root / ab.SOURCES_SUBDIR / "chl" / "source.png").write_bytes(b"challenge-source")
    (pool / RENDERS_SUBDIR / "hold").mkdir(parents=True)
    (pool / RENDERS_SUBDIR / "hold" / "source.png").write_bytes(b"pool-source")
    teacher = PeekTeacher()
    rows = ab.judge_variants("A", "B", teacher, root=root, pool_dir=pool)
    assert {r["cell"] for r in rows} == {"chl@story", "hold@story"}
    assert set(teacher.originals) == {b"challenge-source", b"pool-source"}


# ---------------------------------------------------------------- worst-slice report

def _judg(cell, win):
    return {"cell": cell, "task_id": cell.split("@")[0], "split": "x", "a": "A", "b": "B", "winner": win,
            "confidence": 3, "score_a": 3, "score_b": 3, "reason": "", "teacher": "f",
            "rubric_version": ab.judge_mod.RUBRIC_VERSION, "swapped": False}


def _slice_world(tmp_path):
    """12 changed cells: solstice holdout on story (tall) wins B 6-0; challenge on
    leaderboard (strip) goes 2-4 to A. One identical cell."""
    root = tmp_path
    rows_a, rows_b, js = [], [], []
    for i in range(6):
        c = f"s{i}@story"
        rows_a.append(_row(c, "story", "solstice", "holdout", f"a{i}", "A"))
        rows_b.append(_row(c, "story", "solstice", "holdout", f"b{i}", "B"))
        js.append(_judg(c, "b"))
    for i in range(6):
        c = f"c{i}@leaderboard"
        rows_a.append(_row(c, "leaderboard", "challenge", "challenge", f"ca{i}", "A"))
        rows_b.append(_row(c, "leaderboard", "challenge", "challenge", f"cb{i}", "B"))
        js.append(_judg(c, "b" if i < 2 else "a"))
    rows_a.append(_row("o0@story", "story", "offbrand", "holdout", "same", "A"))
    rows_b.append(_row("o0@story", "story", "offbrand", "holdout", "same", "B"))
    _winners(root, "A", rows_a)
    _winners(root, "B", rows_b)
    (root / "A__vs__B").mkdir()
    (root / "A__vs__B" / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in js))
    return root


def test_report_slices_and_generalisation_score(tmp_path):
    root = _slice_world(tmp_path)
    r = ab.compute_report("A", "B", root)
    assert r["wins"] == {"a": 4, "b": 8, "unsure": 0}
    assert r["by_aspect"] == {"strip": {"a": 4, "b": 2, "unsure": 0}, "tall": {"a": 0, "b": 6, "unsure": 0}}
    assert r["by_family"] == {"challenge": {"a": 4, "b": 2, "unsure": 0}, "solstice": {"a": 0, "b": 6, "unsure": 0}}
    g = r["generalisation"]
    assert g["score"] == pytest.approx(2 / 6)
    assert g["worst"] in ("aspect:strip", "family:challenge")
    assert g["overall"] == pytest.approx(8 / 12)
    assert g["slices"] == 4
    text = ab.write_report("A", "B", root).read_text()
    for needle in ["By target aspect class", "By brand family", "Generalisation score", "0.33", "0.67",
                   "| strip | 4 | 2 |", "| challenge | 4 | 2 |", "Identical winners: 1"]:
        assert needle in text, needle


def test_generalisation_ignores_small_slices(tmp_path):
    root = _slice_world(tmp_path)
    js_path = root / "A__vs__B" / "judgments.jsonl"
    keep = [j for j in map(json.loads, js_path.read_text().splitlines()) if not j["cell"].startswith("c")
            or j["cell"] in ("c0@leaderboard", "c2@leaderboard", "c3@leaderboard")]
    js_path.write_text("".join(json.dumps(j) + "\n" for j in keep))
    g = ab.compute_report("A", "B", root)["generalisation"]
    assert g["score"] == pytest.approx(1.0) and g["slices"] == 2  # the 3-cell challenge slices don't count
    for j in keep:
        j["winner"] = "tie"
    js_path.write_text("".join(json.dumps(j) + "\n" for j in keep))
    g = ab.compute_report("A", "B", root)["generalisation"]
    assert g["score"] is None and g["overall"] is None
    assert "n/a" in ab.write_report("A", "B", root).read_text()
