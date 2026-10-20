"""Hand-written relation features (dev/attention/relations.py), on small synthetic designs."""

import copy
import math
import random

import pytest

from dev.attention.relations import FEATURE_NAMES, relation_features
from engine.pipeline.layout.place import floor_px
from engine.pipeline.types import PriorityPlan, Unit
from tests.designs import design, image, shape, text


def _plan(*specs) -> PriorityPlan:
    """specs: (unit_id, element_ids, role); content units are ranked in order given."""
    units, rank = [], 1
    for uid, ids, role in specs:
        layered = role in ("background", "decoration")
        units.append(Unit(uid, list(ids), "single" if len(ids) == 1 else "group", role=role,
                          rank=None if layered else rank))
        rank += 0 if layered else 1
    return PriorityPlan("heuristic", units)


PLAN = _plan(("bg", ["bg"], "background"), ("headline", ["headline"], "headline"),
             ("hero", ["hero"], "hero_image"), ("cta", ["cta"], "cta"), ("logo", ["logo"], "logo"),
             ("b1", ["b1"], "badge"), ("b2", ["b2"], "badge"), ("sub", ["sub"], "subheadline"),
             ("legal", ["legal"], "legal"))


def _ad(width=1000, height=1000, *, headline=(100, 60, 800, 120), hero=(250, 220, 500, 400),
        cta=(380, 680, 240, 70), logo=(420, 900, 160, 60), b1=(700, 230, 80, 80), b2=(790, 230, 80, 80),
        sub=(150, 790, 700, 50), legal=(200, 960, 600, 20), headline_size=80, legal_size=12):
    return design(
        shape("bg", 0, 0, width, height, z_index=0),
        text("headline", "Chase the light", *headline, headline_size, z_index=2),
        image("hero", "solstice_bottle.png", *hero, z_index=1),
        text("cta", "Shop now", *cta, 30, z_index=2),
        image("logo", "solstice_logo.png", *logo, z_index=2),
        text("b1", "NEW", *b1, 20, z_index=3),
        text("b2", "50%", *b2, 20, z_index=3),
        text("sub", "Botanical sparkling infusions", *sub, 30, z_index=2),
        text("legal", "Please enjoy responsibly.", *legal, legal_size, z_index=2),
        width=width, height=height)


def _scaled(d: dict, s: float, ox: float = 0, oy: float = 0, width=None, height=None) -> dict:
    out = copy.deepcopy(d)
    out["canvas"] = {"width": width or d["canvas"]["width"] * s, "height": height or d["canvas"]["height"] * s}
    for el in out["elements"]:
        el["x"], el["y"] = el["x"] * s + ox, el["y"] * s + oy
        el["width"], el["height"] = el["width"] * s, el["height"] * s
        if el["type"] == "text":
            el["font_size"] *= s
    return out


def test_all_features_present_finite_and_bounded():
    src = _ad()
    f = relation_features(src, src, PLAN)
    assert list(f) == list(FEATURE_NAMES)
    assert 20 <= len(FEATURE_NAMES) <= 40
    for k, v in f.items():
        assert isinstance(v, float) and math.isfinite(v), k
        assert -1e-9 <= v <= 1 + 1e-9, (k, v)


def test_logo_next_to_cta_beats_logo_in_far_corner():
    src = _ad()
    near = relation_features(src, _ad(logo=(640, 685, 160, 60)), PLAN)
    far = relation_features(src, _ad(logo=(10, 10, 80, 30)), PLAN)
    assert near["logo_cta_gap"] < far["logo_cta_gap"]
    assert near["logo_cta_gap"] == pytest.approx(20 / math.hypot(1000, 1000), abs=1e-6)


def test_headline_above_hero_and_cta_below_headline():
    src = _ad()
    good = relation_features(src, src, PLAN)
    flipped = relation_features(src, _ad(headline=(100, 820, 800, 120), cta=(380, 20, 240, 70)), PLAN)
    assert good["headline_above_hero"] > 0.5 > flipped["headline_above_hero"]
    assert good["cta_below_headline"] > 0.5 > flipped["cta_below_headline"]


def test_badges_attached_and_in_a_row():
    src = _ad()
    together = relation_features(src, src, PLAN)
    scattered = relation_features(src, _ad(b1=(20, 400, 80, 80), b2=(900, 900, 80, 80)), PLAN)
    assert together["badge_hero_gap"] == pytest.approx(0.0)  # b1 sits on the hero
    assert together["badge_hero_gap"] < scattered["badge_hero_gap"]
    assert together["badge_spread"] < scattered["badge_spread"]


def test_reading_order_agreement():
    src = _ad()
    assert relation_features(src, src, PLAN)["reading_order"] == pytest.approx(1.0)
    # mirror every unit top-to-bottom: the order reverses
    flipped = copy.deepcopy(src)
    for el in flipped["elements"]:
        if el["id"] != "bg":
            el["y"] = 1000 - el["y"] - el["height"]
    assert relation_features(src, flipped, PLAN)["reading_order"] < 0.2


def test_uniform_shrink_signature():
    src = _ad()
    shrunk = _scaled(src, 0.25, ox=40, width=330, height=250)  # letterboxed uniform shrink
    reflowed = _ad(300, 250, headline=(10, 10, 140, 40), hero=(160, 10, 130, 230), cta=(10, 180, 120, 30),
                   logo=(10, 220, 60, 20), b1=(165, 15, 30, 30), b2=(200, 15, 30, 30), sub=(10, 60, 140, 30),
                   legal=(10, 235, 140, 10), headline_size=30, legal_size=7)
    assert relation_features(src, shrunk, PLAN)["uniform_scale"] > 0.95
    assert relation_features(src, reflowed, PLAN)["uniform_scale"] < 0.5


def test_size_relations():
    src = _ad()
    f = relation_features(src, src, PLAN)
    assert f["headline_body_ratio"] == pytest.approx(80 / (80 + 30))  # r / (1 + r), r = 80 / 30
    small_head = relation_features(src, _ad(headline_size=20), PLAN)
    assert small_head["headline_body_ratio"] < 0.5 < f["headline_body_ratio"]
    assert 0 < f["cta_headline_size"] < 0.5  # the CTA box is smaller than the headline box
    assert f["hero_area_share"] == pytest.approx(0.2)
    assert f["logo_area_share"] == pytest.approx(160 * 60 / 1e6)


def test_floors_legal_and_below_floor_share():
    src = _ad()
    ok = relation_features(src, src, PLAN)
    assert ok["below_floor_share"] == pytest.approx(0.0)
    tiny = relation_features(src, _ad(headline_size=8, legal_size=3.5), PLAN)
    floor = floor_px("legal", 1000, 1000)  # 7 px, scaled up on a 1000 px canvas
    assert tiny["legal_floor_ratio"] == pytest.approx(3.5 / floor / 2)  # halved so 2x the floor maps to 1
    assert tiny["below_floor_share"] == pytest.approx(2 / 6)  # headline + legal of 6 text units


def test_empty_space_and_edge_margin():
    src = _ad()
    packed = relation_features(src, src, PLAN)
    corner = _ad(headline=(0, 0, 100, 30), hero=(0, 30, 100, 60), cta=(0, 90, 50, 20), logo=(50, 90, 50, 20),
                 b1=(0, 110, 20, 20), b2=(20, 110, 20, 20), sub=(0, 130, 100, 20), legal=(0, 150, 100, 10))
    sparse = relation_features(src, corner, PLAN)
    assert sparse["empty_rect_share"] > 0.7 > packed["empty_rect_share"]
    assert sparse["min_edge_margin"] == pytest.approx(0.0)
    assert packed["min_edge_margin"] == pytest.approx(20 / 1000 / 0.1)  # the legal line ends 20 px from the bottom


def test_text_on_image_overlap():
    src = _ad()
    clear = relation_features(src, src, PLAN)
    on_photo = relation_features(src, _ad(sub=(300, 400, 400, 50)), PLAN)
    assert on_photo["text_image_overlap"] > clear["text_image_overlap"]
    assert on_photo["text_image_overlap_gain"] > 0 and clear["text_image_overlap_gain"] == pytest.approx(0)


def test_alignment_and_rhythm():
    plan = _plan(("a", ["a"], "headline"), ("b", ["b"], "subheadline"), ("c", ["c"], "other_text"))
    even = design(text("a", "A", 100, 100, 300, 50, 40), text("b", "B", 100, 200, 300, 50, 20),
                  text("c", "C", 100, 300, 300, 50, 20))
    ragged = design(text("a", "A", 100, 100, 300, 50, 40), text("b", "B", 237, 160, 300, 50, 20),
                    text("c", "C", 513, 500, 300, 50, 20))
    fe, fr = relation_features(even, even, plan), relation_features(even, ragged, plan)
    assert fe["nn_axis_share"] == pytest.approx(1.0) and fr["nn_axis_share"] < 1.0
    assert fe["rhythm_cv"] == pytest.approx(0.0) and fr["rhythm_cv"] > 0.2


def test_neighbours_kept_together():
    src = _ad()
    assert relation_features(src, src, PLAN)["neighbour_kept"] == pytest.approx(1.0)
    apart = relation_features(src, _ad(b1=(20, 500, 80, 80), b2=(900, 20, 80, 80), cta=(20, 20, 240, 70)), PLAN)
    assert apart["neighbour_kept"] < 1.0


def test_missing_roles_get_neutral_defaults_and_flags():
    plan = _plan(("headline", ["headline"], "headline"), ("sub", ["sub"], "subheadline"))
    d = design(text("headline", "Hi", 100, 100, 600, 100, 60), text("sub", "there", 100, 300, 600, 50, 20))
    f = relation_features(d, d, plan)
    for role in ("logo", "cta", "hero", "badge", "legal"):
        assert f[f"has_{role}"] == 0.0
    assert f["has_headline"] == 1.0
    assert f["logo_cta_gap"] == 0.5 and f["badge_spread"] == 0.0 and f["legal_floor_ratio"] == 0.5
    assert f["headline_above_hero"] == 0.5 and f["hero_area_share"] == 0.0


def test_element_order_does_not_matter():
    src = _ad()
    moved = _ad(logo=(640, 685, 160, 60), sub=(300, 400, 400, 50))
    shuffled = copy.deepcopy(moved)
    random.Random(3).shuffle(shuffled["elements"])
    assert relation_features(src, moved, PLAN) == relation_features(src, shuffled, PLAN)


def test_real_sample_with_heuristic_plan(l2, targets, measurer):
    from engine.pipeline.layout.generate import generate_candidates
    from engine.pipeline.priority import heuristic_plan

    source = l2["source"]
    plan = heuristic_plan(source)
    target = next(t for t in targets if t["name"] == "leaderboard")
    cand = generate_candidates(source, {"width": target["width"], "height": target["height"]}, plan, measurer)[0]
    f = relation_features(source, cand.design, plan)
    assert all(0 <= v <= 1 for v in f.values())


# --- dev/attention/features_dataset.py: rebuilding judged candidates and featurising them ---

def test_judged_cells_collects_both_sides_once():
    from dev.attention.features_dataset import judged_cells

    rows = [{"cell": "t1@x", "a": "p", "b": "q"}, {"cell": "t1@x", "a": "q", "b": "r"},
            {"cell": "t2@y", "a": "s", "b": "fallback"}]
    assert judged_cells(rows) == {"t1@x": {"p", "q", "r"}, "t2@y": {"s", "fallback"}}


def test_features_for_cell_matches_candidates_by_id(l2, targets, measurer):
    from dev.attention.features_dataset import features_for_cell
    from engine.pipeline.grading.metrics import geometry_metrics
    from engine.pipeline.layout.fallback import safe_fallback
    from engine.pipeline.layout.generate import generate_candidates
    from engine.pipeline.priority import heuristic_plan

    source = l2["source"]
    record = {"task_id": "sample_l2", "source": source}
    target = next(t for t in targets if t["name"] == "leaderboard")
    canvas = {"width": target["width"], "height": target["height"]}
    plan = heuristic_plan(source, "brand_kit/assets")
    cands = [safe_fallback(source, canvas, plan, measurer)] + generate_candidates(source, canvas, plan, measurer)
    picked = {cands[0].candidate_id: cands[0], cands[2].candidate_id: cands[2]}
    expected = {cid: geometry_metrics(source, c, plan) for cid, c in picked.items()}

    rows, stats = features_for_cell(record, target, set(picked) | {"no/such/candidate"}, "brand_kit/assets",
                                    measurer, expected_metrics=expected)
    assert sorted(r["candidate_id"] for r in rows) == sorted(picked)
    for r in rows:
        assert r["cell"] == "sample_l2@leaderboard"
        assert r["features"] == relation_features(source, picked[r["candidate_id"]].design, plan)
    assert stats["missing"] == ["no/such/candidate"]
    assert stats["max_metric_diff"] == pytest.approx(0.0, abs=1e-9)
