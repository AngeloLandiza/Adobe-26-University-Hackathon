import pytest
from engine.pipeline.types import LLMCallStats
from engine.pipeline.priority import plan
from engine.pipeline.priority.providers import LLMResult, ProviderError


class Fake:
    name, model = "fake", "m"
    def __init__(self, data=None, exc=None): self.data, self.exc, self.calls = data, exc, 0
    def complete(self, system, user, schema):
        self.calls += 1
        if self.exc: raise self.exc
        return LLMResult(self.data, LLMCallStats("fake", "m", latency_s=1.5, input_tokens=900, output_tokens=100, cost_usd=0.01))


L2_GOOD = {"summary": "s", "roles": [
    {"unit_id": u, "role": r} for u, r in [("bg", "background"), ("deco_circle_1", "decoration"), ("deco_circle_2", "decoration"),
    ("logo", "logo"), ("product", "hero_image"), ("headline", "headline"), ("subhead", "subheadline"), ("badge", "badge"), ("cta", "cta")]],
    "ranking": ["headline", "product", "cta", "logo", "subhead", "badge"]}


def test_good_llm_output_is_used(l2):
    p = plan(l2["source"], provider=Fake(L2_GOOD))
    assert p.source == "llm" and [u.unit_id for u in p.ranked_units()][:3] == ["headline", "product", "cta"]


def heur(task):
    from engine.pipeline.priority.units import build_units
    from engine.pipeline.priority.features import unit_features
    from engine.pipeline.priority.heuristic import heuristic_roles, heuristic_ranking
    units = build_units(task["source"]); feats = unit_features(task["source"], units)
    roles = heuristic_roles(feats)
    return units, roles, heuristic_ranking(feats, roles)


def test_repair_fixes_duplicates_and_gaps(l2):
    from engine.pipeline.priority.validate import repair
    units, hr, hk = heur(l2)
    _, ranking, frac = repair(dict(L2_GOOD, ranking=["headline", "product", "cta", "logo", "subhead", "subhead"]), units, hr, hk)
    assert ranking == ["headline", "product", "cta", "logo", "subhead", "badge"] and frac == pytest.approx(2 / 9)
    _, ranking, frac = repair(dict(L2_GOOD, ranking=["headline", "product", "cta", "deco_circle_1", "logo", "subhead", "badge"]), units, hr, hk)
    assert "deco_circle_1" not in ranking and frac == pytest.approx(1 / 9)


def test_too_broken_output_falls_back(l2):
    p = plan(l2["source"], provider=Fake({"summary": "", "roles": [], "ranking": []}))
    assert p.source == "heuristic" and p.stats.error == "too_many_repairs"


def test_provider_error_falls_back(l2):
    p = plan(l2["source"], provider=Fake(exc=ProviderError("refusal")))
    assert p.source == "heuristic" and "refusal" in p.stats.error


def test_llm_cannot_relabel_background(l2):
    data = dict(L2_GOOD, roles=[{"unit_id": "bg", "role": "hero_image"}] + L2_GOOD["roles"][1:])
    assert plan(l2["source"], provider=Fake(data)).unit_for_element("bg").role == "background"


def test_missing_asset_forced_last(l3):
    from engine.pipeline.priority import _assemble
    units, roles, ranking = heur(l3)
    ranking = ["holiday_watermark"] + [u for u in ranking if u != "holiday_watermark"]
    assert _assemble(units, roles, ranking, "llm", None).ranked_units()[-1].unit_id == "holiday_watermark"


def test_auto_without_keys_uses_heuristic(monkeypatch, l2):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "CHAMELEON_LLM"):
        monkeypatch.delenv(k, raising=False)
    assert plan(l2["source"]).source == "heuristic"


def test_cache_hit_skips_provider(tmp_path, monkeypatch, l2):
    monkeypatch.setenv("CHAMELEON_LLM_CACHE", str(tmp_path))
    fake = Fake(L2_GOOD)
    plan(l2["source"], provider=fake); p = plan(l2["source"], provider=fake)
    assert fake.calls == 1 and p.stats.cached


# --- beyond the brief ---

def test_llm_may_skip_background_roles(l2):
    # the prompt lists background units by id only, so leaving them out is not a repair
    data = dict(L2_GOOD, roles=[r for r in L2_GOOD["roles"] if r["unit_id"] != "bg"])
    p = plan(l2["source"], provider=Fake(data))
    assert p.source == "llm" and p.stats.repaired_fraction == 0 and p.unit_for_element("bg").role == "background"


def test_llm_cannot_make_a_unit_background(l2):
    # node 2 cover-fills background units; a decoration circle must not become one
    data = dict(L2_GOOD, roles=[dict(r, role="background") if r["unit_id"] == "deco_circle_1" else r for r in L2_GOOD["roles"]])
    p = plan(l2["source"], provider=Fake(data))
    assert p.unit_for_element("deco_circle_1").role == "decoration" and p.stats.repaired_fraction == pytest.approx(1 / 9)


def test_repair_keeps_first_role_and_drops_unknown_ids(l2):
    from engine.pipeline.priority.validate import repair
    units, hr, hk = heur(l2)
    raw = dict(L2_GOOD, roles=L2_GOOD["roles"] + [{"unit_id": "logo", "role": "badge"}, {"unit_id": "ghost", "role": "logo"}],
               ranking=L2_GOOD["ranking"] + ["ghost"])
    roles, ranking, frac = repair(raw, units, hr, hk)
    assert roles["logo"] == "logo" and "ghost" not in roles and "ghost" not in ranking and frac == pytest.approx(1 / 9)


def test_unknown_provider_name_falls_back(l2):
    p = plan(l2["source"], provider="gpt")
    assert p.source == "heuristic" and "gpt" in p.stats.error


def test_provider_error_records_who_failed(l2):
    p = plan(l2["source"], provider=Fake(exc=TimeoutError("read timed out")))
    assert (p.stats.provider, p.stats.model) == ("fake", "m") and "timed out" in p.stats.error and p.stats.latency_s >= 0


def test_llm_stats_carry_through(l2):
    p = plan(l2["source"], provider=Fake(L2_GOOD))
    assert (p.stats.input_tokens, p.stats.output_tokens, p.stats.cost_usd, p.stats.cached) == (900, 100, 0.01, False)


def test_cache_hit_costs_nothing_and_key_includes_model(tmp_path, monkeypatch, l2):
    monkeypatch.setenv("CHAMELEON_LLM_CACHE", str(tmp_path))
    fake = Fake(L2_GOOD)
    plan(l2["source"], provider=fake); hit = plan(l2["source"], provider=fake)
    assert hit.stats.cost_usd == 0 and hit.stats.latency_s == 0 and hit.stats.input_tokens == 900
    other = Fake(L2_GOOD); other.model = "m2"
    plan(l2["source"], provider=other)
    assert other.calls == 1


def test_plan_never_raises_on_a_malformed_design():
    src = {"canvas": {"width": 100, "height": 100}, "elements": [{"id": "a", "type": "text"}, {"id": "b", "type": "image"}]}
    p = plan(src, provider="heuristic")
    assert sorted(e for u in p.units for e in u.element_ids) == ["a", "b"] and p.stats.error


def test_missing_units_are_inserted_where_the_heuristic_puts_them(l2):
    # heuristic order: headline, product, cta, logo, subhead, badge
    from engine.pipeline.priority.validate import repair
    units, hr, hk = heur(l2)
    _, ranking, frac = repair(dict(L2_GOOD, ranking=["headline", "cta", "subhead", "badge"]), units, hr, hk)
    assert ranking == ["headline", "product", "cta", "logo", "subhead", "badge"] and frac == pytest.approx(2 / 9)


def test_llm_cannot_make_copy_or_images_decoration(l2):
    # node 2 parks decorations in corners at source scale; the headline would shrink to a few px
    data = dict(L2_GOOD, roles=[dict(r, role="decoration") if r["unit_id"] in ("headline", "product") else r
                                for r in L2_GOOD["roles"]])
    p = plan(l2["source"], provider=Fake(data))
    assert p.unit_for_element("headline").role == "headline" and p.unit_for_element("product").role == "hero_image"
    assert p.stats.repaired_fraction == pytest.approx(2 / 9)
