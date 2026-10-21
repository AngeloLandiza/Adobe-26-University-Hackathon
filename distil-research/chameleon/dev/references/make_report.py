"""Build docs/real-ad-layouts.md from the three aggregate JSONs (Crello, engine base, engine bisect). Stdlib only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "training-data" / "references"
SIZES = {t["name"]: (t["width"], t["height"]) for t in json.loads((Path(__file__).resolve().parents[1] / "targets.json").read_text())}
ACC = json.loads((Path(__file__).with_name("labelling_accuracy.json")).read_text())
real = json.loads((REF / "crello_layout_stats.json").read_text())
eng = {n: json.loads((REF / f"engine_layout_stats_{n}.json").read_text()) for n in ("base", "bisect")}
ROLES = ("headline", "body", "cta", "logo", "product")


def g(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


def f(v, n=2):
    return "-" if v is None else f"{v:.{n}f}"


def metrics(t, role):
    r = g(t, "roles", role)
    if not r or "cx" not in r:
        return {}
    m = {"cx": g(r, "cx", "median"), "cy": g(r, "cy", "median"), "area": g(r, "area_share", "median"),
         "font": g(r, "font_short_side_share", "median")}
    m["iqr"] = {"cx": (g(r, "cx", "q3") - g(r, "cx", "q1")), "cy": (g(r, "cy", "q3") - g(r, "cy", "q1")),
                "area": (g(r, "area_share", "q3") - g(r, "area_share", "q1")),
                "font": (g(r, "font_short_side_share", "q3") - g(r, "font_short_side_share", "q1")) if m["font"] is not None else None}
    return m


def sketch(t, W=48, H=16):
    """ASCII sketch of the median layout: each role drawn at its median centre with its median area (square-ish)."""
    grid = [[" "] * W for _ in range(H)]
    ch = {"headline": "H", "body": "b", "cta": "C", "logo": "L", "product": "P"}
    for role in ("product", "body", "headline", "cta", "logo"):
        r = g(t, "roles", role)
        if not r or r.get("presence", 0) < 0.25 or "cx" not in r:
            continue
        cx, cy, a = r["cx"]["median"], r["cy"]["median"], r["area_share"]["median"]
        w = max(1, min(W, round((a ** 0.5) * W * (1.6 if role in ("headline", "body") else 1.0))))
        h = max(1, min(H, round((a ** 0.5) * H * (0.7 if role in ("headline", "body") else 1.0))))
        x0 = min(max(0, round(cx * W - w / 2)), W - w); y0 = min(max(0, round(cy * H - h / 2)), H - h)
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                grid[y][x] = ch[role]
    return "\n".join("|" + "".join(r) + "|" for r in grid)


def deviations(var):
    """Scaled deviation per (target, role, metric): |engine - real| / max(real IQR / 1.35, floor). Strict-pool targets only."""
    out = []
    floors = {"cx": 0.04, "cy": 0.04, "area": 0.02, "font": 0.01}
    for tn, rt in real["targets"].items():
        if rt["pool_kind"] != "strict" or tn not in eng[var]["targets"]:
            continue
        et = eng[var]["targets"][tn]
        for role in ROLES:
            mr, me = metrics(rt, role), metrics(et, role)
            if not mr or not me or g(rt, "roles", role, "presence") < 0.3 or g(et, "roles", role, "presence") < 0.3:
                continue
            for k in ("cx", "cy", "area", "font"):
                if mr.get(k) is None or me.get(k) is None:
                    continue
                sig = max(mr["iqr"][k] / 1.35, floors[k])
                out.append((abs(me[k] - mr[k]) / sig, tn, role, k, mr[k], me[k]))
        for k, lab in (("empty_space_share", "empty space"),):
            a, b = g(rt, k, "median"), g(et, k, "median")
            if a is not None and b is not None:
                out.append((abs(a - b) / max((g(rt, k, "q3") - g(rt, k, "q1")) / 1.35, 0.03), tn, "layout", lab, a, b))
        a, b = g(rt, "edge_margin_short_side", "min_any_edge", "median"), g(et, "edge_margin_short_side", "min_any_edge", "median")
        if a is not None and b is not None:
            q = g(rt, "edge_margin_short_side", "min_any_edge")
            out.append((abs(a - b) / max((q["q3"] - q["q1"]) / 1.35, 0.01), tn, "layout", "min edge margin", a, b))
    return sorted(out, reverse=True)


L = []
L.append("# Real designer ad layouts vs our engine\n")
L.append("What real designers do at each of our 13 target sizes, measured on one shard of the public Crello dataset, "
         "and where today's engine (`base`) and bisection (`bisect`) differ. Only aggregates are committed; the raw shard "
         "stays in `training-data/references/crello.nosync/` (gitignored). Numbers: `training-data/references/crello_layout_stats.json`, "
         "`engine_layout_stats_base.json`, `engine_layout_stats_bisect.json`. Scripts: `chameleon/dev/references/`.\n")
L.append("## Method\n")
L.append(f"- Data: `cyberagent/crello` revision 5.1.0, `data/test-00000-of-00004.parquet` ({real['n_designs_in_shard']} templates; "
         f"{real['n_ad_format_designs']} in ad/social/promo formats after excluding resumes, invitations, cards, certificates, menus, logos, presentations, calendars and similar).")
L.append("- Pool per target: ad-format templates whose canvas aspect is within 5% of the target. If fewer than 20 match, the pool is the 20 ad-format templates "
         "nearest in aspect, marked `nearest-aspect` below. **For the very wide targets (leaderboard, mobile_banner, large_leaderboard, billboard, linkedin_banner) the real pool is NOT the same shape as the target**; treat those rows as a rough guide only.")
L.append("- Roles are labelled by simple rules (`dev/references/crello_roles.py`): background (full-bleed), headline (largest non-CTA text), CTA (short text inside a small button shape, or starting with an action verb), "
         "logo (small graphic near an edge, or short small text near an edge), product (largest remaining image of at least 5% of the canvas), body (other text), decoration (everything else).")
L.append("- Positions are the element box centre as a fraction of the canvas; area share is the clipped box over canvas area; font size is a share of the canvas short side (headline: largest, others: median); "
         "edge margin is the gap between all non-background content (union box) and each canvas edge, as a share of the short side; empty space is the share of a 64x64 grid not covered by any non-background element box "
         "(boxes, not glyph pixels, so it under-states emptiness). Reading order: `reads before` = above by more than 0.05, or on the same row and left.")
L.append("- Engine side: holdout cells of `ab/base` and `ab/bisect` (192 cells each) regenerated with the same code and environment as `ab_engine.py` (heuristic planner, no teacher), "
         "using the engine's own plan roles (hero_image->product, subheadline/other_text/legal/badge->body). The engine's roles are its own judgement, not my heuristic, so role-level gaps include labelling differences.\n")
L.append("## Labelling accuracy (honest)\n")
L.append(ACC["text"] + "\n")
L.append("## Matches per target\n")
L.append("| target | size | pool | strict matches | used | aspect range used | top formats |\n|---|---|---|---|---|---|---|")
for tn, (w, h) in SIZES.items():
    t = real["targets"][tn]
    fm = ", ".join(f"{k} {v}" for k, v in list(t["formats_used"].items())[:3])
    L.append(f"| {tn} | {w}x{h} | {t['pool_kind']} | {t['n_matches_strict']} | {t['n_used']} | {t.get('aspect_range', '-')} | {fm} |")
L.append("\n## Per-size summaries (real designers)\n")
for tn, (w, h) in SIZES.items():
    t = real["targets"][tn]
    L.append(f"### {tn} ({w}x{h}) - {t['pool_kind']} pool, n={t['n_used']}\n")
    L.append("| role | presence | centre x (median, IQR) | centre y (median, IQR) | area share | font / short side |\n|---|---|---|---|---|---|")
    for role in ROLES + ("decoration", "background"):
        r = t["roles"][role]
        if "cx" not in r:
            L.append(f"| {role} | {r['presence']:.2f} | - | - | - | - |")
            continue
        q = lambda d: f"{d['median']:.2f} ({d['q1']:.2f}-{d['q3']:.2f})"
        fs = q(r["font_short_side_share"]) if "font_short_side_share" in r else "-"
        L.append(f"| {role} | {r['presence']:.2f} | {q(r['cx'])} | {q(r['cy'])} | {r['area_share']['median']:.3f} | {fs} |")
    m = t["edge_margin_short_side"]
    L.append(f"\nEdge margin (share of short side, median): min-any-edge {f(g(m,'min_any_edge','median'))}; left {f(g(m,'left','median'))}, top {f(g(m,'top','median'))}, right {f(g(m,'right','median'))}, bottom {f(g(m,'bottom','median'))}. "
             f"Empty space {f(g(t,'empty_space_share','median'))}. Reading order: " +
             "; ".join(f"{k.replace('_', ' ')}: above {v['a_above_b']:.2f}, left {v['a_left_of_b']:.2f}, reads-before {v['a_reads_before_b']:.2f} (n={v['n']})" for k, v in t["order"].items()) + ".\n")
    L.append("Typical layout (H headline, b body, C CTA, L logo, P product; roles present in under 25% of templates omitted):\n")
    L.append("```\n" + sketch(t) + "\n```\n")
L.append("## Our engine vs real practice (holdout cells)\n")
L.append("Median centre / area / font by role. `real` = Crello, `base` = today's engine, `bisect` = bisection. Only targets with a strict aspect pool are ranked in the deviation lists; "
         "a role is compared only if present in at least 30% of both sides' layouts.\n")
for tn in SIZES:
    rt = real["targets"][tn]
    if tn not in eng["base"]["targets"]:
        continue
    L.append(f"### {tn} ({rt['pool_kind']}; engine n={eng['base']['targets'][tn]['n_layouts']})\n")
    L.append("| role | presence real/base/bisect | cx real/base/bisect | cy real/base/bisect | area real/base/bisect | font real/base/bisect |\n|---|---|---|---|---|---|")
    for role in ROLES:
        ms = [metrics(x["targets"][tn], role) for x in (real, eng["base"], eng["bisect"])]
        ps = [g(x["targets"][tn], "roles", role, "presence") for x in (real, eng["base"], eng["bisect"])]
        cell = lambda k: " / ".join(f(m.get(k), 3 if k in ("area", "font") else 2) for m in ms)
        L.append(f"| {role} | {' / '.join(f(p) for p in ps)} | {cell('cx')} | {cell('cy')} | {cell('area')} | {cell('font')} |")
    xs = [x["targets"][tn] for x in (real, eng["base"], eng["bisect"])]
    L.append(f"\nMin edge margin (median) real/base/bisect: " + " / ".join(f(g(x, 'edge_margin_short_side', 'min_any_edge', 'median')) for x in xs) +
             "; empty space: " + " / ".join(f(g(x, 'empty_space_share', 'median')) for x in xs) +
             "; headline reads before CTA: " + " / ".join(f(g(x, 'order', 'headline_vs_cta', 'a_reads_before_b')) for x in xs) + ".\n")
L.append("## Biggest deviations from real layouts\n")
for var in ("base", "bisect"):
    L.append(f"### {var}: top 12 (scaled gap = |engine - real| / (real IQR/1.35, floored))\n")
    L.append("| scaled gap | target | role | metric | real | engine |\n|---|---|---|---|---|---|")
    for s, tn, role, k, a, b in deviations(var)[:12]:
        L.append(f"| {s:.1f} | {tn} | {role} | {k} | {a:.3f} | {b:.3f} |")
    L.append("")
(ROOT / "docs" / "real-ad-layouts.md").write_text("\n".join(L) + "\n")
print("ok")
