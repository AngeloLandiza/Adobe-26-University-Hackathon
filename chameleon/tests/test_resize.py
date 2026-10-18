import json
import pytest
from pathlib import Path
from engine import resize as R
from engine.pipeline.geometry import inside_canvas, rotated_aabb
from engine.pipeline.grading.gates import check_gates
from engine.pipeline.priority import heuristic_plan
from tests.conftest import assert_valid


@pytest.mark.parametrize("path", sorted(Path("input").glob("*.json")), ids=lambda p: p.stem)
def test_resize_outputs_pass_gates(path, measurer):
    task = json.loads(path.read_text()); src, tgt = task["source"], task["target_canvas"]
    design, plan, best, reports = R.resize_with_report(src, tgt)
    assert check_gates(src, design, tgt, plan, measurer) == []


def test_resize_survives_node_failures(monkeypatch, l2):
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(R, "generate_candidates", boom); monkeypatch.setattr(R, "pick_best", boom)
    assert_valid(l2["source"], R.resize(l2["source"], l2["target_canvas"]), l2["target_canvas"])


def test_resize_text_only_and_image_only():
    text_only = {"canvas": {"width": 1080, "height": 1080}, "elements": [
        {"id": "t", "type": "text", "content": "Hello world", "font_family": "Solstice Serif", "font_size": 60,
         "color": "rgba(0,0,0,1)", "x": 100, "y": 400, "width": 880, "height": 100}]}
    image_only = {"canvas": {"width": 1080, "height": 1080}, "elements": [
        {"id": "i", "type": "image", "asset_id": "solstice_bottle.png", "x": 290, "y": 290, "width": 500, "height": 500}]}
    for src in (text_only, image_only):
        for tgt in ({"width": 728, "height": 90}, {"width": 1080, "height": 1920}):
            assert_valid(src, R.resize(src, tgt), tgt)


def test_resize_is_deterministic(l3):
    a = json.dumps(R.resize(l3["source"], l3["target_canvas"]))
    assert a == json.dumps(R.resize(l3["source"], l3["target_canvas"]))


def test_debug_dump_only_when_enabled(monkeypatch, tmp_path, l1):
    # beyond the brief: dump into a temp dir and check what's written, not just that something is
    monkeypatch.setattr(R, "DEBUG_DIR", tmp_path)
    monkeypatch.delenv("CHAMELEON_DEBUG", raising=False)
    R.resize(l1["source"], l1["target_canvas"])
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setenv("CHAMELEON_DEBUG", "1")
    R.resize(l1["source"], l1["target_canvas"])
    (folder,) = tmp_path.iterdir()
    assert {p.name for p in folder.iterdir()} == {"plan.json", "reports.json", "contact_sheet.png"}
    totals = [r["total"] for r in json.loads((folder / "reports.json").read_text())]
    assert totals == sorted(totals, reverse=True)
    assert json.loads((folder / "plan.json").read_text())["source"] == "heuristic"


# --- beyond the brief ---

def test_last_resort_when_everything_else_fails(monkeypatch, l3):
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(R, "resize_with_report", boom); monkeypatch.setattr(R, "safe_fallback", boom)
    src, tgt = l3["source"], l3["target_canvas"]
    design = R.resize(src, tgt)
    assert design["canvas"] == tgt and [e["id"] for e in design["elements"]] == [e["id"] for e in src["elements"]]
    for el in design["elements"]:
        assert inside_canvas(rotated_aabb(el), tgt["width"], tgt["height"]), el["id"]
        if el["type"] == "text":
            assert el["font_size"] >= 1


def test_last_resort_keeps_image_proportions(monkeypatch, l2):
    # its whole job is never to be disqualified: one uniform scale, no stretched logo or product
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(R, "resize_with_report", boom); monkeypatch.setattr(R, "safe_fallback", boom)
    src, tgt = l2["source"], l2["target_canvas"]
    design = R.resize(src, tgt)
    assert_valid(src, design, tgt)
    s = min(tgt["width"] / 1080, tgt["height"] / 1080)  # the whole design fits: the smaller scale
    logo = next(e for e in design["elements"] if e["id"] == "logo")
    assert (logo["width"], logo["height"]) == (pytest.approx(320 * s), pytest.approx(110 * s))


def test_priority_line_on_stderr(capsys, l1):
    R.resize_with_report(l1["source"], l1["target_canvas"])
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert line == "[priority] source=heuristic provider=none model=- latency=0.00s cost=$0.0000 error=-"


def test_resize_returns_the_winner(l2):
    design, plan, best, reports = R.resize_with_report(l2["source"], l2["target_canvas"])
    assert design is best.design and not best.is_fallback and len(reports) >= 2
