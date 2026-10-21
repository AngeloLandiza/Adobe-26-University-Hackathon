"""Role-labelled layout -> aggregate statistics (shared by the Crello and engine sides).

A layout is {"W": int, "H": int, "items": [{"role", "x0","y0","x1","y1", "font" (text only, px)}]}
in canvas pixels. Roles: background, headline, body, cta, logo, product, decoration.
Only aggregate numbers leave this module.
"""
import numpy as np

ROLES = ("headline", "body", "cta", "logo", "product", "decoration", "background")
TEXT_ROLES = ("headline", "body", "cta")
GRID = 64


def clip(it, W, H):
    return max(0.0, it["x0"]), max(0.0, it["y0"]), min(float(W), it["x1"]), min(float(H), it["y1"])


def _q(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    a, m, b = np.percentile(v, [25, 50, 75])
    return {"median": round(float(m), 4), "q1": round(float(a), 4), "q3": round(float(b), 4)}


def content_mask(layout):
    """GRID x GRID occupancy of every non-background item (decorations included)."""
    W, H = layout["W"], layout["H"]
    m = np.zeros((GRID, GRID), bool)
    for it in layout["items"]:
        if it["role"] == "background":
            continue
        x0, y0, x1, y1 = clip(it, W, H)
        if x1 <= x0 or y1 <= y0:
            continue
        m[int(y0 / H * GRID):max(int(np.ceil(y1 / H * GRID)), int(y0 / H * GRID) + 1),
          int(x0 / W * GRID):max(int(np.ceil(x1 / W * GRID)), int(x0 / W * GRID) + 1)] = True
    return m


def per_layout(layout):
    """One record: per role first item geometry + layout-level numbers."""
    W, H = layout["W"], layout["H"]
    short = min(W, H)
    rec = {"roles": {}}
    for role in ROLES:
        its = [it for it in layout["items"] if it["role"] == role]
        if not its:
            continue
        it = max(its, key=lambda i: (i["x1"] - i["x0"]) * (i["y1"] - i["y0"]))
        x0, y0, x1, y1 = clip(it, W, H)
        if x1 <= x0 or y1 <= y0:
            continue
        r = {"cx": (x0 + x1) / 2 / W, "cy": (y0 + y1) / 2 / H, "area": sum(
            max(0.0, clip(i, W, H)[2] - clip(i, W, H)[0]) * max(0.0, clip(i, W, H)[3] - clip(i, W, H)[1])
            for i in its) / (W * H), "n": len(its)}
        if role in TEXT_ROLES and it.get("font"):
            fs = [i["font"] for i in its if i.get("font")]
            r["font"] = float(np.max(fs)) / short if role == "headline" else float(np.median(fs)) / short
        rec["roles"][role] = r
    content = [clip(i, W, H) for i in layout["items"] if i["role"] != "background"]
    content = [c for c in content if c[2] > c[0] and c[3] > c[1]]
    if content:
        x0 = min(c[0] for c in content); y0 = min(c[1] for c in content)
        x1 = max(c[2] for c in content); y1 = max(c[3] for c in content)
        rec["margins"] = {"left": x0 / short, "top": y0 / short, "right": (W - x1) / short, "bottom": (H - y1) / short}
        rec["margin_min"] = min(rec["margins"].values())
        rec["empty"] = 1.0 - float(content_mask(layout).mean())
    return rec


def _order(recs, a, b):
    """Share of layouts with both roles where a precedes b: above (cy smaller by > 0.03),
    left (cx smaller by > 0.03), and 'reading order' = above, or on the same row (|dy| <= 0.05) and left."""
    pairs = [(r["roles"][a], r["roles"][b]) for r in recs if a in r["roles"] and b in r["roles"]]
    if not pairs:
        return None
    n = len(pairs)
    above = sum(p[0]["cy"] < p[1]["cy"] - 0.03 for p in pairs) / n
    left = sum(p[0]["cx"] < p[1]["cx"] - 0.03 for p in pairs) / n
    read = sum((p[0]["cy"] < p[1]["cy"] - 0.05) or (abs(p[0]["cy"] - p[1]["cy"]) <= 0.05 and p[0]["cx"] < p[1]["cx"])
               for p in pairs) / n
    return {"n": n, "a_above_b": round(above, 3), "a_left_of_b": round(left, 3), "a_reads_before_b": round(read, 3)}


def aggregate(layouts):
    recs = [per_layout(l) for l in layouts]
    out = {"n_layouts": len(recs), "roles": {}, "order": {}}
    for role in ROLES:
        rs = [r["roles"][role] for r in recs if role in r["roles"]]
        d = {"presence": round(len(rs) / max(1, len(recs)), 3)}
        if rs:
            d.update(cx=_q([r["cx"] for r in rs]), cy=_q([r["cy"] for r in rs]), area_share=_q([r["area"] for r in rs]))
            f = [r["font"] for r in rs if "font" in r]
            if f:
                d["font_short_side_share"] = _q(f)
        out["roles"][role] = d
    mg = [r["margins"] for r in recs if "margins" in r]
    out["edge_margin_short_side"] = {
        "min_any_edge": _q([r["margin_min"] for r in recs if "margin_min" in r]),
        **{k: _q([m[k] for m in mg]) for k in ("left", "top", "right", "bottom")}}
    out["empty_space_share"] = _q([r["empty"] for r in recs if "empty" in r])
    for a, b, name in (("headline", "cta", "headline_vs_cta"), ("headline", "product", "headline_vs_product"),
                       ("product", "cta", "product_vs_cta"), ("logo", "headline", "logo_vs_headline")):
        o = _order(recs, a, b)
        if o:
            out["order"][name] = o
    return out
