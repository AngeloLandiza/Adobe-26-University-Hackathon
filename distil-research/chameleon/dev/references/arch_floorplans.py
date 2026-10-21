"""Step 4 (plain python): archetype medoids -> engine floor-plan JSON (training-data/references/archetype_floorplans.json).
Medoid role boxes (canvas fractions) -> slots hero/text/logo/cta/minor as content-box fractions. Deviations from the raw medoid,
all forced by the floor-plan format (slots may not overlap, text and minor are mandatory):
  - headline (+ body when it touches the headline) -> 'text'; a separate body -> 'minor'; no body -> 'minor' is a free band.
  - text slot is at least 35% x 14% of the content box (real headline boxes are often tiny).
  - when the hero covers >= 25% of the content box it has priority and the text slot is moved to the largest free band outside it.
  - overlapping slots: priority text > minor > cta > logo > hero; the loser is cut to its largest non-overlapping rectangle
    (hero dropped if under 30% of its size survives).  Real ads often put text ON the hero; that overlay cannot be expressed.
Geometry only; no template content.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parents[2] / "training-data" / "references"
TARGETS = {t["name"]: t for t in json.loads((HERE.parent / "targets.json").read_text())}
M = 0.055  # margin share of the short side assumed when converting canvas -> content-box fractions
PRIORITY = ["text", "minor", "cta", "logo", "hero"]


def aclass(r):
    return "very_tall" if r < .55 else "tall" if r < .85 else "square" if r < 1.25 else "landscape" if r < 2.2 else "wide" if r <= 5 else "strip"


def rect(b):  # (cx,cy,w,h) -> (x0,y0,x1,y1)
    return (b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2)


def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def area(a):
    return max(0, a[2] - a[0]) * max(0, a[3] - a[1])


def cut(loser, win):
    """largest rectangle inside `loser` not overlapping `win`."""
    cands = [(loser[0], loser[1], loser[2], min(loser[3], win[1])), (loser[0], max(loser[1], win[3]), loser[2], loser[3]),
             (loser[0], loser[1], min(loser[2], win[0]), loser[3]), (max(loser[0], win[2]), loser[1], loser[2], loser[3])]
    return max(cands, key=area)


def plan_for(name, boxes, size):
    t = TARGETS[size]
    W, H = t["width"], t["height"]
    m = M * min(W, H)
    cw, ch = W - 2 * m, H - 2 * m
    def to_content(r):
        x0, y0, x1, y1 = (max(0, (r[0] * W - m) / cw), max(0, (r[1] * H - m) / ch), min(1, (r[2] * W - m) / cw), min(1, (r[3] * H - m) / ch))
        return (x0, y0, x1, y1)
    R = {k: to_content(rect(v)) for k, v in boxes.items()}
    slots = {}
    if "headline" not in R and "body" in R:
        R["headline"] = R.pop("body")
    if "headline" not in R:
        return None, "no headline in medoid"
    text = R["headline"]
    if "body" in R:
        b, h = R["body"], R["headline"]
        gap = max(b[1] - h[3], h[1] - b[3], 0)
        if inter(b, h) > 0 or gap < 0.06:
            text = (min(b[0], h[0]), min(b[1], h[1]), max(b[2], h[2]), max(b[3], h[3]))
        else:
            slots["minor"] = b
    # minimum text size
    tw, th = text[2] - text[0], text[3] - text[1]
    cx, cy = (text[0] + text[2]) / 2, (text[1] + text[3]) / 2
    tw, th = max(tw, 0.35), max(th, 0.14)
    slots["text"] = (max(0, min(cx - tw / 2, 1 - tw)), max(0, min(cy - th / 2, 1 - th)), max(0, min(cx - tw / 2, 1 - tw)) + tw, max(0, min(cy - th / 2, 1 - th)) + th)
    for src, dst in (("product", "hero"), ("cta", "cta"), ("logo", "logo")):
        if src in R:
            slots[dst] = R[src]
    notes = []
    if "hero" in slots and area(slots["hero"]) >= 0.25:
        # reserve a text band on the side where the medoid's text sat, then trim the hero to the rest
        horiz = W / H >= 1.2
        left_side = (slots["text"][0] + slots["text"][2]) / 2 < 0.5
        top_side = (slots["text"][1] + slots["text"][3]) / 2 < 0.5
        if horiz:
            band = (0, 0, 0.38, 1) if left_side else (0.62, 0, 1, 1)
        else:
            band = (0, 0, 1, 0.30) if top_side else (0, 0.70, 1, 1)
        slots["text"] = band
        slots["hero"] = cut(slots["hero"], band)
        notes.append(f"text band reserved ({'left' if left_side else 'right'} column)" if horiz else f"text band reserved ({'top' if top_side else 'bottom'})")
    order = [k for k in PRIORITY if k in slots]
    if "hero" in slots and area(slots["hero"]) >= 0.25:  # a big hero is the archetype's point: text yields, not the hero
        order = ["hero"] + [k for k in order if k != "hero"]
        slots.pop("minor", None) if False else None
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if b in slots and inter(slots[a], slots[b]) > 0.005 * min(area(slots[a]), area(slots[b])):
                before = area(slots[b])
                if a == "hero" and b == "text":  # relocate text to the largest free band beside/above/below the hero
                    slots[b] = cut((0, 0, 1, 1), slots[a])
                    notes.append("text moved off the hero")
                    continue
                slots[b] = cut(slots[b], slots[a])
                keep = area(slots[b]) / max(before, 1e-9)
                if b == "hero" and keep < 0.30 or area(slots[b]) < 0.002:
                    notes.append(f"{b} dropped (overlapped {a})")
                    del slots[b]
                else:
                    notes.append(f"{b} cut to {keep*100:.0f}% (overlapped {a})")
    if "minor" not in slots:  # free band: largest strip between the content and the edge, else carve from the bottom of text
        used = list(slots.values())
        ub, ut, ul, ur = max(u[3] for u in used), min(u[1] for u in used), min(u[0] for u in used), max(u[2] for u in used)
        CAP = 0.2  # the minor zone is a small flow area, not a big empty band
        bands = [(0, ub, 1, min(1, ub + CAP)), (0, max(0, ut - CAP), 1, ut), (0, 0, ul, 1) if ul <= CAP else (ul - CAP, 0, ul, 1),
                 (ur, 0, min(1, ur + CAP), 1)]
        best = max(bands, key=area)
        if min(best[2] - best[0], best[3] - best[1]) >= 0.05 and area(best) >= 0.02:
            slots["minor"] = best
        else:
            tx = slots["text"]
            h = tx[3] - tx[1]
            slots["minor"] = (tx[0], tx[3] - min(0.3 * h, 0.08), tx[2], tx[3])
            slots["text"] = (tx[0], tx[1], tx[2], slots["minor"][1])
            notes.append("minor carved from text bottom")
    out = []
    for k in ("hero", "text", "logo", "cta", "minor"):
        if k in slots:
            x0, y0, x1, y1 = slots[k]
            out.append({"name": k, "box": [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]})
    return {"name": name, "classes": [aclass(W / H)], "margins": [0.04, 0.07], "slots": out}, "; ".join(notes)


def main():
    data = json.loads((REF / "archetypes.nosync" / "archetypes.json").read_text())
    plans, log = [], []
    for pool in data["pools"]:
        key = pool["key"]
        parts = key.split(":")
        label = parts[1] if parts[0] == "size" else (parts[2] if len(parts) > 2 else parts[1])
        for i, a in enumerate(pool["archetypes"]):
            plan, note = plan_for(f"arch_{label}_{i + 1}", {k: tuple(v) for k, v in a["medoid_boxes"].items()}, pool["sizes"][0])
            if plan is None:
                log.append((f"arch_{label}_{i + 1}", note)); continue
            plan["_source"] = {"pool": key, "sizes": pool["sizes"], "share": round(a["share"], 3), "n": a["n"], "notes": note}
            plans.append(plan)
            log.append((plan["name"], note))
    # validate with the engine's own parser if importable (project python)
    try:
        sys.path.insert(0, str(HERE.parent))
        from engine.pipeline.layout.extra_templates import _parse_plan
        seen = {}
        for p in plans:
            q = {k: v for k, v in p.items() if k != "_source"}
            seen[p["name"]] = _parse_plan(q, seen)
        print("all", len(plans), "plans valid per _parse_plan")
    except ImportError:
        print("engine not importable here; validated later")
    clean = [{k: v for k, v in p.items() if k != "_source"} for p in plans]
    (REF / "archetype_floorplans.json").write_text(json.dumps({"floor_plans": clean}, indent=1))
    (REF / "archetypes.nosync" / "floorplan_sources.json").write_text(json.dumps({p["name"]: p["_source"] for p in plans}, indent=1))
    for n, note in log:
        print(n, note)


if __name__ == "__main__":
    main()
