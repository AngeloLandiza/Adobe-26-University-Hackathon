import copy
import json
import pytest
from pathlib import Path
from engine.pipeline.grading.gates import GATES, check_gates
from engine.pipeline.layout.fallback import safe_fallback
from engine.pipeline.layout.generate import generate_candidates
from engine.pipeline.priority import heuristic_plan

SAMPLES = sorted(Path("input").glob("*.json"))
TARGETS = json.loads(Path("dev/targets.json").read_text())


def base(l2):
    src, tgt = l2["source"], l2["target_canvas"]
    plan = heuristic_plan(src)
    return src, tgt, plan, safe_fallback(src, tgt, plan).design


def el(design, id): return next(e for e in design["elements"] if e["id"] == id)


MUTATIONS = {
    "G1_ids": lambda d: d["elements"].pop(),
    "G2_schema": lambda d: el(d, "headline").pop("color"),
    "G3_canvas": lambda d: d["canvas"].update(width=999),
    "G4_in_canvas": lambda d: el(d, "logo").update(x=-50),
    "G5_aspect": lambda d: el(d, "product").update(width=el(d, "product")["width"] * 1.5),
    "G6_frozen": lambda d: el(d, "headline").update(content="Changed"),
    "G7_text_fit": lambda d: el(d, "subhead").update(height=1),
    "G8_groups": lambda d: el(d, "cta_label").update(x=el(d, "cta_bg")["x"] + el(d, "cta_bg")["width"]),
    # product (z 3) shrunk to a 2x2 square fully inside the logo image (z 4)
    "G9_hidden": lambda d: el(d, "product").update(x=el(d, "logo")["x"] + 1, y=el(d, "logo")["y"] + 1, width=2, height=2),
}


@pytest.mark.parametrize("gate", list(MUTATIONS))
def test_each_gate_catches_its_mutation(gate, l2, measurer):
    src, tgt, plan, design = base(l2)
    bad = copy.deepcopy(design); MUTATIONS[gate](bad)
    assert gate in check_gates(src, bad, tgt, plan, measurer)


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_fallback_passes_all_gates(path, target, measurer):
    src = json.loads(path.read_text())["source"]; plan = heuristic_plan(src)
    assert check_gates(src, safe_fallback(src, target, plan).design, target, plan, measurer) == []


# --- beyond the brief ---

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["name"])
def test_every_generated_candidate_passes_all_gates(path, target, measurer):
    # node 2 builds valid by construction; a gate failure here is a node 2 bug
    src = json.loads(path.read_text())["source"]; plan = heuristic_plan(src)
    for c in generate_candidates(src, target, plan, measurer):
        assert check_gates(src, c.design, target, plan, measurer) == [], c.candidate_id


def test_clean_design_passes_and_failures_come_in_gate_order(l2, measurer):
    src, tgt, plan, design = base(l2)
    assert check_gates(src, design, tgt, plan, measurer) == []
    bad = copy.deepcopy(design)
    MUTATIONS["G9_hidden"](bad); MUTATIONS["G3_canvas"](bad); MUTATIONS["G6_frozen"](bad)
    failed = check_gates(src, bad, tgt, plan, measurer)
    assert failed == sorted(failed, key=GATES.index) and {"G3_canvas", "G6_frozen", "G9_hidden"} <= set(failed)


def test_duplicate_id_fails_g1(l2, measurer):
    src, tgt, plan, design = base(l2)
    bad = copy.deepcopy(design); bad["elements"][-1] = dict(bad["elements"][-1], id=bad["elements"][0]["id"])
    assert "G1_ids" in check_gates(src, bad, tgt, plan, measurer)


@pytest.mark.parametrize("field,value", [("x", float("nan")), ("width", 0), ("font_size", 0.5)])
def test_g2_rejects_bad_numbers(field, value, l2, measurer):
    src, tgt, plan, design = base(l2)
    bad = copy.deepcopy(design); el(bad, "headline")[field] = value
    assert "G2_schema" in check_gates(src, bad, tgt, plan, measurer)


def test_g5_lets_cover_images_crop(l2, measurer):
    src, tgt, plan, design = base(l2)
    stretched = copy.deepcopy(design); el(stretched, "bg").update(width=el(stretched, "bg")["width"] / 2)
    assert "G5_aspect" not in check_gates(src, stretched, tgt, plan, measurer)


def test_g6_catches_an_added_key(l2, measurer):
    src, tgt, plan, design = base(l2)
    bad = copy.deepcopy(design); el(bad, "logo")["filter"] = "blur"  # a key the source element doesn't have
    assert "G6_frozen" in check_gates(src, bad, tgt, plan, measurer)


def square(id, x, y, size, z, fill="rgba(0,0,0,1)", **extra):
    return {"id": id, "type": "shape", "shape_type": "rectangle", "fill_color": fill,
            "x": x, "y": y, "width": size, "height": size, "z_index": z, **extra}


def button(id, x, y, size, z, fill="rgba(0,0,0,1)", **extra):
    """A shape with a label on it: a button, so never a decoration (G9 ignores decoration covers)."""
    return [square(id, x, y, size, z, fill, group_id=f"{id}_g", **extra),
            {"id": f"{id}_t", "type": "text", "content": "Go", "font_family": "Solstice Sans", "font_size": 10,
             "color": "rgba(255,255,255,1)", "x": x + 5, "y": y + 5, "width": size - 10, "height": 14,
             "z_index": z + 1, "group_id": f"{id}_g"}]


def hidden(elements, measurer):
    design = {"canvas": {"width": 200, "height": 200}, "elements": elements}
    plan = heuristic_plan(design)
    return "G9_hidden" in check_gates(design, design, {"width": 200, "height": 200}, plan, measurer)


def test_g9_equal_z_later_element_paints_on_top(measurer):
    # cr-renderer sorts by z_index with a stable sort, so at equal z the later element covers the earlier
    assert hidden([square("small", 50, 50, 20, 2), *button("big", 40, 40, 60, 2)], measurer)
    assert not hidden([*button("big", 40, 40, 60, 2), square("small", 50, 50, 20, 2)], measurer)


def test_g9_ignores_see_through_and_decoration_covers(measurer):
    assert not hidden([square("small", 50, 50, 20, 1), *button("veil", 40, 40, 60, 2, fill="rgba(0,0,0,0.5)")], measurer)
    assert not hidden([square("small", 50, 50, 20, 1), *button("faded", 40, 40, 60, 2, opacity=0.4)], measurer)
    assert not hidden([square("small", 50, 50, 20, 1), square("deco", 40, 40, 60, 2)], measurer)
