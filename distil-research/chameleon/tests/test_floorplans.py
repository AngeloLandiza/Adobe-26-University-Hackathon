"""B2 floor plans: propose -> build -> judge -> cluster (+ evaluate), with fake
teachers only: no test calls claude or an API."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dev.distill import floorplans as fp
from dev.distill.judge import RUBRIC_VERSION
from engine.pipeline.layout import extra_templates

ENV = "CHAMELEON_EXTRA_TEMPLATES"
METRIC_KEYS = ["hierarchy", "legibility", "overlap", "degradation", "faithfulness", "space_use", "alignment",
               "spacing", "balance", "text_contrast", "busyness", "placeholder"]
TARGETS = {t["name"]: t for t in json.loads(Path("dev/targets.json").read_text())}


def slot(name, box):
    return {"name": name, "box": box}


TALL_SLOTS = [slot("logo", [0, 0, 1, 0.07]), slot("hero", [0, 0.08, 1, 0.42]), slot("text", [0, 0.52, 1, 0.26]),
              slot("cta", [0, 0.8, 1, 0.1]), slot("minor", [0, 0.92, 1, 0.08])]
OVERLAPPING = [slot("text", [0, 0, 1, 0.6]), slot("hero", [0, 0.3, 1, 0.5]), slot("minor", [0, 0.9, 1, 0.1])]
NO_MINOR = [slot("text", [0, 0, 1, 0.5]), slot("hero", [0, 0.5, 1, 0.5])]


@pytest.fixture(autouse=True)
def _no_extra_file(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(extra_templates, "EXTRA_PATH", tmp_path / "absent_default.json")


def _metrics(v):
    return {k: v for k in METRIC_KEYS}


def _pool_row(task, target, cid, value, split="train", brand="solstice", fallback=False):
    return {"cell": f"{task}@{target}", "task_id": task, "split": split, "brand": brand, "target": target,
            "candidate_id": cid, "png": f"renders.nosync/{task}/{target}/{cid.replace('/', '_')}.png",
            "is_fallback": fallback, "metrics": _metrics(value)}


# ---------------------------------------------------------------- winners and cells

def test_current_winner_is_best_by_default_grader_and_skips_fallback():
    rows = [_pool_row("ad", "story", "fallback", 0.99, fallback=True), _pool_row("ad", "story", "stack/b", 0.5),
            _pool_row("ad", "story", "stack/a", 0.5), _pool_row("ad", "story", "stack/c", 0.4),
            _pool_row("lone", "story", "fallback", 0.3, fallback=True)]
    winners = fp.cell_winners(rows)
    assert winners["ad@story"]["candidate_id"] == "stack/a"  # tie broken by candidate_id; fallback never wins
    assert winners["lone@story"]["candidate_id"] == "fallback"  # ...unless it is all there is


def test_propose_cells_are_train_only_and_cover_every_aspect_class():
    rows = []
    for i in range(30):
        for name in TARGETS:
            for split in ("train", "holdout"):
                rows.append(_pool_row(f"{split}{i}", name, "stack/a", 0.5, split=split,
                                      brand="solstice" if i % 2 else "offbrand"))
    cells = fp.pick_propose_cells(rows, 40, seed=0, targets=TARGETS)
    assert cells == fp.pick_propose_cells(rows, 40, seed=0, targets=TARGETS)
    assert cells != fp.pick_propose_cells(rows, 40, seed=1, targets=TARGETS)
    assert 40 <= len(cells) <= 41 and all(c.startswith("train") for c in cells)
    classes = {fp.aspect_of(c.split("@")[1], TARGETS) for c in cells}
    assert classes == {"very_tall", "tall", "square", "landscape", "wide", "strip"}


# ---------------------------------------------------------------- propose

class PlanTeacher:
    """Answers every propose call with the scripted plans; records the calls."""
    name = "fake-designer"

    def __init__(self, plans=None, fail=False):
        self.plans = plans if plans is not None else [
            {"idea": "hero top", "margins": [0.04, 0.07], "slots": TALL_SLOTS},
            {"idea": "overlap", "margins": [0.05], "slots": OVERLAPPING},
            {"idea": "no minor", "margins": [0.05], "slots": NO_MINOR}]
        self.calls, self.fail = [], fail

    def ask(self, prompt, schema, images=(), validate=None):
        self.calls.append({"prompt": prompt, "schema": schema, "images": [Path(p) for p in images]})
        if self.fail:
            raise RuntimeError("rate limited")
        answer = {"plans": json.loads(json.dumps(self.plans))}
        if validate:
            validate(answer)
        return answer


def _context(cell):
    task, target = cell.split("@")
    t = TARGETS[target]
    return {"cell": cell, "task_id": task, "split": "train", "target": target, "width": t["width"],
            "height": t["height"], "aspect_class": fp.aspect_of(target, TARGETS),
            "units": [{"rank": 1, "role": "headline", "kind": "single", "elements": [{"type": "text", "content": "Big sale"}]}],
            "layers": 1, "source_png": Path(f"/pool/{task}/source.png"), "winner_png": Path(f"/pool/{task}/{target}/0.png"),
            "winner_id": "stack/a"}


def test_propose_writes_valid_plans_skips_invalid(tmp_path, capsys):
    out = tmp_path / "proposals.jsonl"
    t = PlanTeacher()
    rows = fp.propose(["ad1@instagram_portrait"], _context, t, out)
    assert len(t.calls) == 1
    call = t.calls[0]
    assert call["images"] == [Path("/pool/ad1/source.png"), Path("/pool/ad1/instagram_portrait/0.png")]
    assert "1080" in call["prompt"] and "1350" in call["prompt"] and "tall" in call["prompt"]
    assert "Big sale" in call["prompt"] and "headline" in call["prompt"]
    assert "No unwanted collisions" in call["prompt"]  # the harsh rubric is shown
    assert call["schema"]["properties"]["plans"]["minItems"] == 3
    assert len(rows) == 1  # the overlapping and the minor-less plans are skipped
    err = capsys.readouterr().err
    assert err.count("[floorplans] skipping") == 2 and "overlap" in err and "minor" in err
    on_disk = [json.loads(x) for x in out.read_text().splitlines()]
    assert on_disk == rows
    row = rows[0]
    assert row["cell"] == "ad1@instagram_portrait" and row["task_id"] == "ad1" and row["split"] == "train"
    assert row["target"] == "instagram_portrait" and row["aspect_class"] == "tall"
    assert row["teacher"] == "fake-designer" and row["rubric_version"] == RUBRIC_VERSION
    plan = row["plan"]
    assert plan["name"] == row["proposal_id"] and plan["name"].startswith("p") and len(plan["name"]) == 9
    assert plan["classes"] == ["tall"] and plan["margins"] == [0.04, 0.07] and plan["slots"] == TALL_SLOTS
    extra_templates._parse_plan(plan, {})  # engine-loadable as is


def test_propose_names_are_unique_and_duplicates_skipped(tmp_path):
    same = {"idea": "x", "margins": [0.04], "slots": TALL_SLOTS}
    t = PlanTeacher([same, dict(same, idea="again"), dict(same, margins=[0.06])])
    rows = fp.propose(["ad1@instagram_portrait", "ad2@instagram_portrait"], _context, t, tmp_path / "p.jsonl")
    names = [r["proposal_id"] for r in rows]
    assert len(rows) == 4 and len(set(names)) == 4  # one duplicate per cell dropped; same plan in two cells differs


def test_propose_resumes_and_respects_limit(tmp_path):
    out = tmp_path / "p.jsonl"
    cells = ["ad1@story", "ad2@story", "ad3@story"]
    t0 = PlanTeacher()
    assert fp.propose(cells, _context, t0, out, limit_calls=0) == [] and t0.calls == [] and not out.exists()
    t1 = PlanTeacher()
    fp.propose(cells, _context, t1, out, limit_calls=2)
    assert len(t1.calls) == 2
    t2 = PlanTeacher()
    fp.propose(cells, _context, t2, out)
    assert len(t2.calls) == 1  # only the cell not yet done
    other = PlanTeacher()
    other.name = "other"
    fp.propose(cells, _context, other, out)
    assert len(other.calls) == 3  # another teacher's rows are not this teacher's work


def test_propose_survives_teacher_failure(tmp_path, capsys):
    rows = fp.propose(["ad1@story"], _context, PlanTeacher(fail=True), tmp_path / "p.jsonl")
    assert rows == [] and "[floorplans]" in capsys.readouterr().err


def test_shards_partition_cells_and_merge(tmp_path):
    cells = [f"ad{i}@story" for i in range(12)]
    main = tmp_path / "proposals.jsonl"
    seen = []
    for k in range(3):
        t = PlanTeacher()
        fp.propose(fp.in_shard(cells, (k, 3)), _context, t, fp.shard_file(main, k, 3), done_files=[main])
        seen += [c["prompt"] for c in t.calls]
    assert len(seen) == 12
    assert fp.merge_shards(main) == 12 * 1 and len(main.read_text().splitlines()) == 12
    assert not list(tmp_path.glob("*shard*"))


def test_cell_context_lists_units_from_heuristic_plan(l1, tmp_path):
    rows = [_pool_row(l1["task_id"], "story", "stack/a", 0.5), _pool_row(l1["task_id"], "story", "stack/b", 0.6)]
    designs = {l1["task_id"]: {"task_id": l1["task_id"], "split": "train", "source": l1["source"]}}
    ctx = fp.cell_context(f"{l1['task_id']}@story", fp.cell_winners(rows), designs, TARGETS, Path("brand_kit/assets"),
                          pool_dir=tmp_path)
    assert (ctx["width"], ctx["height"], ctx["aspect_class"]) == (1080, 1920, "tall")  # 0.5625 >= 0.55: tall
    assert ctx["winner_id"] == "stack/b" and ctx["winner_png"] == tmp_path / rows[1]["png"]
    assert ctx["source_png"] == tmp_path / "renders.nosync" / l1["task_id"] / "source.png"
    ranks = [u["rank"] for u in ctx["units"]]
    assert ranks == sorted(ranks) and ranks[0] == 1
    assert {u["role"] for u in ctx["units"]} & {"headline", "cta", "logo", "hero_image"}
    texts = [e["content"] for u in ctx["units"] for e in u["elements"] if e["type"] == "text"]
    assert texts and all(isinstance(x, str) for x in texts)


# ---------------------------------------------------------------- build

def _proposal(slots=TALL_SLOTS, name="pabc12345", task="lvl1", target="instagram_portrait", margins=(0.04, 0.07)):
    return {"proposal_id": name, "cell": f"{task}@{target}", "task_id": task, "split": "train", "target": target,
            "aspect_class": fp.aspect_of(target, TARGETS), "teacher": "fake", "rubric_version": RUBRIC_VERSION,
            "plan": {"name": name, "classes": [fp.aspect_of(target, TARGETS)], "margins": list(margins), "slots": slots}}


def test_build_fills_the_plan_with_the_real_engine(l1, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "/some/user/setting.json")
    designs = {"lvl1": {"task_id": "lvl1", "split": "train", "source": l1["source"]}}
    out = tmp_path / "builds.jsonl"
    rows = fp.build([_proposal()], designs, TARGETS, Path("brand_kit/assets"), out, fp_dir=tmp_path)
    assert os.environ[ENV] == "/some/user/setting.json"  # restored after each plan
    assert len(rows) == 1 and rows == [json.loads(x) for x in out.read_text().splitlines()]
    row = rows[0]
    assert row["built"] is True and row["reason_if_not"] is None
    assert row["proposal_id"] == "pabc12345" and row["cell"] == "lvl1@instagram_portrait"
    assert row["candidate_id"] in ("pabc12345/margin=0.04", "pabc12345/margin=0.07")
    assert set(row["metrics"]) == set(METRIC_KEYS) and row["total"] == pytest.approx(fp.score(row["metrics"]))
    assert row["n_passed"] >= 1 and (tmp_path / row["png"]).exists()
    assert row["png"].startswith("renders.nosync/lvl1/instagram_portrait/")
    again = fp.build([_proposal()], designs, TARGETS, Path("brand_kit/assets"), out, fp_dir=tmp_path)
    assert again == []  # resumable: already built


def test_build_records_a_plan_that_builds_nothing(l1, tmp_path, monkeypatch):
    designs = {"lvl1": {"task_id": "lvl1", "split": "train", "source": l1["source"]}}
    monkeypatch.setattr(fp, "_template_candidate", lambda *a, **k: None)
    rows = fp.build([_proposal()], designs, TARGETS, Path("brand_kit/assets"), tmp_path / "b.jsonl", fp_dir=tmp_path)
    assert rows[0]["built"] is False and rows[0]["candidate_id"] is None and rows[0]["png"] is None
    assert "no candidate" in rows[0]["reason_if_not"]


def test_build_records_gate_failures(l1, tmp_path, monkeypatch):
    designs = {"lvl1": {"task_id": "lvl1", "split": "train", "source": l1["source"]}}
    monkeypatch.setattr(fp, "check_gates", lambda *a, **k: ["G7_text_fit"])
    rows = fp.build([_proposal()], designs, TARGETS, Path("brand_kit/assets"), tmp_path / "b.jsonl", fp_dir=tmp_path)
    assert rows[0]["built"] is False and "G7_text_fit" in rows[0]["reason_if_not"] and rows[0]["n_candidates"] == 2


# ---------------------------------------------------------------- judge

class JudgeTeacher:
    """Judge fake: whichever image is not a pool render (the proposal) wins, with the given confidence."""
    name = "fake-judge"

    def __init__(self, confidence=3):
        self.calls, self.confidence = [], confidence

    def ask(self, prompt, schema, images, validate=None):
        self.calls.append([Path(p).read_bytes() for p in images])
        n = schema["properties"]["verdicts"]["maxItems"]
        verdicts = []
        for i in range(n):
            winner = "b" if self.calls[-1][1 + 2 * i].startswith(b"POOL") else "a"
            verdicts.append({"pair": i + 1, "winner": winner, "confidence": self.confidence, "score_a": 3,
                             "score_b": 2, "reason": "r"})
        return {"verdicts": verdicts}


def _judge_fixture(tmp_path, n_cells=2):
    pool_dir, fp_dir = tmp_path / "pool", tmp_path / "fp"
    pool_rows, builds = [], []
    for c in range(n_cells):
        task = f"ad{c}"
        for cid, v in (("stack/a", 0.4), ("stack/b", 0.6)):
            r = _pool_row(task, "story", cid, v)
            pool_rows.append(r)
            (pool_dir / r["png"]).parent.mkdir(parents=True, exist_ok=True)
            (pool_dir / r["png"]).write_text(f"POOL:{cid}")
        src = pool_dir / "renders.nosync" / task / "source.png"
        src.write_text("SOURCE")
        for k, built in ((1, True), (2, True), (3, False)):
            pid = f"p{c}{k}000000"
            png = f"renders.nosync/{task}/story/{pid}.png" if built else None
            if built:
                (fp_dir / png).parent.mkdir(parents=True, exist_ok=True)
                (fp_dir / png).write_text(f"FP:{pid}")
            builds.append({"proposal_id": pid, "cell": f"{task}@story", "task_id": task, "split": "train",
                           "target": "story", "candidate_id": f"{pid}/margin=0.04" if built else None,
                           "metrics": _metrics(0.5) if built else None, "total": 0.5 if built else None,
                           "png": png, "built": built, "reason_if_not": None if built else "no candidate built"})
    return pool_dir, fp_dir, pool_rows, builds


def test_judge_pairs_each_built_proposal_against_the_cell_winner(tmp_path):
    pool_dir, fp_dir, pool_rows, builds = _judge_fixture(tmp_path)
    pairs = fp.judge_pairs_for(builds, fp.cell_winners(pool_rows), pool_dir, fp_dir)
    assert len(pairs) == 4  # 2 cells x 2 built proposals; the unbuilt one is not judged
    for winner, proposal in pairs:
        assert winner["candidate_id"] == "stack/b"
        assert proposal["candidate_id"].startswith("p")
        assert Path(winner["png"]).is_absolute() and Path(proposal["png"]).is_absolute()
    t = JudgeTeacher()
    out = fp_dir / "judgments.jsonl"
    rows = fp.judge(pairs, t, pool_dir, out, swap_share=0.0)
    assert len(t.calls) == 2 and all(call[0] == b"SOURCE" for call in t.calls)  # one call per cell, original first
    assert len(rows) == 4 and all(r["teacher"] == "fake-judge" for r in rows)
    assert all({r["a"], r["b"]} & {"stack/b"} for r in rows)
    assert fp.judge(pairs, JudgeTeacher(), pool_dir, out, swap_share=0.0) == []  # resumes


# ---------------------------------------------------------------- cluster

def test_vectorise_plan_is_slot_ordered():
    plan = {"slots": [slot("text", [0.1, 0.2, 0.3, 0.4]), slot("minor", [0, 0.9, 1, 0.1])]}
    v = fp.vectorise(plan)
    assert v.shape == (30,)
    assert list(v[5:10]) == [1, 0.1, 0.2, 0.3, 0.4]  # text is the second slot name
    assert list(v[:5]) == [0, 0, 0, 0, 0]  # no hero
    assert list(v[25:30]) == [1, 0, 0.9, 1, 0.1]


def test_kmedoids_and_silhouette_find_three_blobs():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 0], [0, 10]])
    X = np.concatenate([c + rng.normal(0, 0.3, (8, 2)) for c in centres])
    D = fp.pairwise(X)
    k, medoids, labels, sil = fp.choose_k(D)
    assert k == 3 and len(medoids) == 3 and sil[3] > sil[2] and sil[3] > sil[4]
    assert sorted(len(set(labels[i * 8:(i + 1) * 8])) for i in range(3)) == [1, 1, 1]
    assert set(labels[m] for m in medoids) == {0, 1, 2}
    assert fp.choose_k(D) == (k, medoids, labels, sil)  # deterministic


def test_choose_k_small_groups():
    assert fp.choose_k(fp.pairwise(np.zeros((1, 2))))[0] == 1
    k, medoids, labels, _ = fp.choose_k(fp.pairwise(np.array([[0.0, 0], [1, 0]])))
    assert k == 1 and medoids == [0] and list(labels) == [0, 0]
    k, medoids, _, _ = fp.choose_k(fp.pairwise(np.zeros((5, 2))))  # all identical: one cluster
    assert k == 1 and medoids == [0]


def _cluster_inputs(tmp_path):
    proposals, builds, judgments = [], [], []
    shapes = {"top": [slot("hero", [0, 0, 1, 0.5]), slot("text", [0, 0.52, 1, 0.3]), slot("minor", [0, 0.9, 1, 0.1])],
              "bottom": [slot("text", [0, 0, 1, 0.3]), slot("hero", [0, 0.32, 1, 0.55]), slot("minor", [0, 0.9, 1, 0.1])]}
    i = 0
    for shape, slots in shapes.items():
        for j in range(5):
            jitter = [dict(s, box=[s["box"][0], s["box"][1], s["box"][2], round(s["box"][3] - 0.01 * j, 3)]) for s in slots]
            pid = f"p{i:08d}"
            cell = f"ad{i}@half_page"  # 300x600: very_tall
            proposals.append(dict(_proposal(jitter, name=pid, task=f"ad{i}", target="half_page")))
            builds.append({"proposal_id": pid, "cell": cell, "candidate_id": f"{pid}/margin=0.04", "built": True})
            outcome = "win" if j < 4 else ("loss" if shape == "top" else "weak")
            winner = {"win": "b", "weak": "b", "loss": "a"}[outcome]
            judgments.append({"cell": cell, "task_id": f"ad{i}", "split": "train", "a": "stack/x",
                              "b": f"{pid}/margin=0.04", "winner": winner, "confidence": 1 if outcome == "weak" else 2,
                              "score_a": 2, "score_b": 3, "reason": "r", "teacher": "t",
                              "rubric_version": RUBRIC_VERSION, "swapped": False})
            i += 1
    # a swapped re-ask that disagrees cancels a win
    judgments.append(dict(judgments[0], a=judgments[0]["b"], b="stack/x", winner="b", swapped=True))
    return proposals, builds, judgments


def test_cluster_writes_medoids_and_candidate_file(tmp_path):
    proposals, builds, judgments = _cluster_inputs(tmp_path)
    shipped = tmp_path / "shipped.json"
    shipped.write_text(json.dumps({"floor_plans": [{"name": "text_first", "classes": ["tall"], "slots": TALL_SLOTS}]}))
    clusters_path, candidate_path = tmp_path / "clusters.json", tmp_path / "candidate.json"
    report = fp.cluster(proposals, builds, judgments, clusters_path, candidate_path, shipped)
    very_tall = report["classes"]["very_tall"]
    assert very_tall["n_judged"] == 10 and very_tall["n_wins"] == 7  # 8 clear wins, one cancelled by its swap
    assert very_tall["k"] == 2
    members = sorted(m["members"] for m in very_tall["medoids"])
    assert members == [3, 4]
    rates = sorted(m["win_rate"] for m in very_tall["medoids"])
    assert rates == [pytest.approx(3 / 5), pytest.approx(4 / 5)]
    assert json.loads(clusters_path.read_text()) == report
    cand = json.loads(candidate_path.read_text())
    names = [p["name"] for p in cand["floor_plans"]]
    assert names[-1] == "text_first" and len(names) == 3
    assert all(p["classes"] == ["very_tall"] for p in cand["floor_plans"][:2])
    assert len(extra_templates._parse(candidate_path.read_text())) == 3  # every plan loads in the engine
    medoid_sources = {m["proposal_id"] for m in very_tall["medoids"]}
    assert medoid_sources <= {p["proposal_id"] for p in proposals}


# ---------------------------------------------------------------- evaluate

def test_evaluate_builds_holdout_pairs_for_new_plans(l1, tmp_path):
    pool_dir, fp_dir = tmp_path / "pool", tmp_path / "fp"
    pool_rows = [_pool_row("lvl1", "instagram_portrait", "stack/a", 0.5, split="holdout"),
                 _pool_row("lvl1", "half_page", "stack/a", 0.5, split="holdout"),
                 _pool_row("tr", "instagram_portrait", "stack/a", 0.5, split="train")]
    for r in pool_rows:
        (pool_dir / r["png"]).parent.mkdir(parents=True, exist_ok=True)
        (pool_dir / r["png"]).write_text("POOL")
    (pool_dir / "renders.nosync" / "lvl1" / "source.png").write_text("SOURCE")
    shipped = tmp_path / "shipped.json"
    shipped.write_text(json.dumps({"floor_plans": [{"name": "text_first", "classes": ["tall"], "slots": TALL_SLOTS}]}))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"floor_plans": [
        {"name": "b2_tall_1", "classes": ["tall"], "margins": [0.04], "slots": TALL_SLOTS},
        {"name": "text_first", "classes": ["tall"], "slots": TALL_SLOTS}]}))
    designs = {"lvl1": {"task_id": "lvl1", "split": "holdout", "source": l1["source"]}}
    builds_out = fp_dir / "eval-builds.jsonl"
    builds = fp.evaluate_build(candidate, shipped, pool_rows, designs, TARGETS, Path("brand_kit/assets"),
                               builds_out, fp_dir=fp_dir)
    # only holdout cells whose class a NEW plan covers (half_page is very_tall; text_first already in the pool)
    assert [(b["cell"], b["plan"]) for b in builds] == [("lvl1@instagram_portrait", "b2_tall_1")]
    assert builds[0]["built"] and builds[0]["candidate_id"] == "b2_tall_1/margin=0.04"
    assert "engine_picks_new" in builds[0]
    pairs = fp.evaluate_pairs(builds, fp.cell_winners(pool_rows), pool_dir, fp_dir)
    assert len(pairs) == 1 and pairs[0][0]["candidate_id"] == "stack/a"
    t = JudgeTeacher()
    rows = fp.judge(pairs, t, pool_dir, fp_dir / "eval-judgments.jsonl", swap_share=0.0)
    assert len(rows) == 1
    summary = fp.evaluate_summary(builds, rows)
    assert summary["b2_tall_1"]["judged"] == 1 and summary["b2_tall_1"]["wins"] == 1


# ---------------------------------------------------------------- extras: sharded judging, rubric filter, CLI

def test_judge_shards_cover_every_pair_once(tmp_path):
    pool_dir, fp_dir, pool_rows, builds = _judge_fixture(tmp_path, n_cells=6)
    pairs = fp.judge_pairs_for(builds, fp.cell_winners(pool_rows), pool_dir, fp_dir)
    main = fp_dir / "judgments.jsonl"
    for k in range(4):
        fp.judge(pairs, JudgeTeacher(), pool_dir, fp.shard_file(main, k, 4), swap_share=0.0, shard=(k, 4),
                 done_files=[main])
    assert fp.merge_shards(main) == len(pairs) == 12


def test_cluster_ignores_other_rubric_and_weak_wins(tmp_path):
    proposals, builds, judgments = _cluster_inputs(tmp_path)
    old = [dict(j, rubric_version="00000000") for j in judgments]
    report = fp.cluster(proposals, builds, old, tmp_path / "c.json", tmp_path / "x.json", tmp_path / "none.json")
    assert report["classes"]["very_tall"]["n_judged"] == 0 and report["classes"]["very_tall"]["medoids"] == []
    assert json.loads((tmp_path / "x.json").read_text())["floor_plans"] == []


def test_evaluate_summary_counts_engine_picks():
    builds = [{"cell": "h@story", "plan": "b2_tall_1", "candidate_id": "b2_tall_1/margin=0.04", "built": True,
               "engine_picks_new": True},
              {"cell": "g@story", "plan": "b2_tall_1", "candidate_id": None, "built": False, "engine_picks_new": False}]
    j = {"cell": "h@story", "a": "stack/a", "b": "b2_tall_1/margin=0.04", "winner": "b", "confidence": 3,
         "rubric_version": RUBRIC_VERSION, "swapped": False}
    s = fp.evaluate_summary(builds, [j])["b2_tall_1"]
    assert (s["cells"], s["built"], s["judged"], s["wins"], s["engine_picks_new"], s["engine_picks_new_and_wins"]) == \
        (2, 1, 1, 1, 1, 1)


def test_cli_merge_and_class_filter(tmp_path, monkeypatch, capsys):
    main = tmp_path / "proposals.jsonl"
    monkeypatch.setattr(fp, "PROPOSALS_FILE", main)
    fp.shard_file(main, 0, 2).write_text(json.dumps({"cell": "a@story"}) + "\n")
    assert fp.main(["propose", "--merge"]) == 0 and len(main.read_text().splitlines()) == 1
    rows = [_pool_row(f"t{i}", name, "stack/a", 0.5) for i in range(10) for name in TARGETS]
    cells = fp.pick_propose_cells(rows, 12, seed=0, targets=TARGETS, classes=["tall", "very_tall"])
    assert len(cells) == 12 and {fp.aspect_of(c.split("@")[1], TARGETS) for c in cells} == {"tall", "very_tall"}
