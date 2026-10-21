"""Heuristic role labels for a Crello (v5.1.0) template. Pixel coordinates, no rendering.

Types: 0 SvgElement 1 TextElement 2 ImageElement 3 ColoredBackground 4 SvgMaskElement.

Rules (applied in this order; thresholds are shares of the canvas area A):
 1 skip: opacity < 0.05, zero size, or box entirely off-canvas. Boxes are clipped to the canvas.
 2 background: ColoredBackground; or any non-text element whose clipped box covers >= 85% of A.
 3 text with an empty string is skipped. Other text elements are candidates:
   - url-ish (www., .com, @, http) -> body.
   - CTA: short text (<= 4 words, <= 28 chars) AND (it sits inside a "button" container, i.e. a
     non-text non-background element with 1.1x..14x the text area, < 20% of A, containing the text centre)
     OR its first word is an action verb (shop, buy, get, learn, ...).
   - headline: of the non-CTA texts, the one with the largest font_size (ties: larger area), if >= 1 text.
   - logo (text): <= 2 words, <= 18 chars, no digit/URL, centre within 15% of an edge, font <= 0.7x headline font.
   - everything else -> body.
 4 product: the largest remaining ImageElement/SvgMaskElement with area >= 5% of A.
 5 logo (graphic): a remaining Image/Svg/SvgMask < 6% of A, aspect (long/short) <= 6, box within 14% of an edge band
   (centre in the outer 18% band of any side). The container of a CTA is not a logo.
 6 decoration: everything else (including CTA button containers: the CTA item box is the container when present).
"""
ACTION = {"shop", "buy", "get", "learn", "order", "book", "call", "visit", "join", "sign", "register", "download",
          "subscribe", "try", "discover", "contact", "apply", "click", "swipe", "start", "see", "find", "explore",
          "read", "watch", "view", "reserve", "donate", "claim", "grab", "enroll", "enrol", "play", "save", "tap",
          "listen", "support", "follow", "send", "buy", "purchase", "request", "browse", "check", "go", "shop"}
URLISH = ("www.", ".com", "@", "http", ".net", ".org", ".co")


def _words(t):
    return t.replace("\n", " ").split()


def label(row, feats):
    """row: dict of python lists for one design. Returns a layout dict (see layout_stats) + 'idx' per item."""
    W, H = float(row["canvas_width"]), float(row["canvas_height"])
    A = W * H
    tnames = feats["type"]["feature"]["names"]
    items = []
    for i in range(row["length"]):
        t = tnames[row["type"][i]]
        if (row["opacity"][i] or 0) < 0.05:
            continue
        x0, y0 = row["left"][i], row["top"][i]
        x1, y1 = x0 + row["width"][i], y0 + row["height"][i]
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        items.append({"i": i, "t": t, "x0": cx0, "y0": cy0, "x1": cx1, "y1": cy1, "area": (cx1 - cx0) * (cy1 - cy0) / A,
                      "text": (row["text"][i] or "") if t == "TextElement" else "",
                      "font": float(row["font_size"][i] or 0) if t == "TextElement" else 0.0, "role": None})
    for it in items:
        if it["t"] == "ColoredBackground" or (it["t"] != "TextElement" and it["area"] >= 0.85):
            it["role"] = "background"
    texts = [it for it in items if it["t"] == "TextElement" and it["text"].strip() and it["role"] is None]
    for it in items:
        if it["t"] == "TextElement" and not it["text"].strip():
            it["role"] = "decoration"

    def container(tx):
        cxm, cym = (tx["x0"] + tx["x1"]) / 2, (tx["y0"] + tx["y1"]) / 2
        best = None
        for c in items:
            if c["t"] == "TextElement" or c["role"] == "background" or c["role"] == "cta_box":
                continue
            if not (c["x0"] <= cxm <= c["x1"] and c["y0"] <= cym <= c["y1"]):
                continue
            if 1.1 * tx["area"] <= c["area"] <= 14 * tx["area"] and c["area"] < 0.2:
                if best is None or c["area"] < best["area"]:
                    best = c
        return best

    ctas = []
    for tx in texts:
        w = _words(tx["text"])
        if not w or len(w) > 4 or len(tx["text"].strip()) > 28:
            continue
        low = tx["text"].lower()
        if any(u in low for u in URLISH):
            continue
        box = container(tx)
        verb = w[0].lower().strip(".,!:;-") in ACTION
        if box is not None or verb:
            tx["cta_box"] = box
            ctas.append(tx)
    # at most the best two CTAs; prefer the ones with a button container
    ctas.sort(key=lambda t: (t.get("cta_box") is None, -t["font"]))
    for tx in ctas[:2]:
        tx["role"] = "cta"
        if tx.get("cta_box") is not None:
            tx["cta_box"]["role"] = "cta_box"
            tx["x0"], tx["y0"] = min(tx["x0"], tx["cta_box"]["x0"]), min(tx["y0"], tx["cta_box"]["y0"])
            tx["x1"], tx["y1"] = max(tx["x1"], tx["cta_box"]["x1"]), max(tx["y1"], tx["cta_box"]["y1"])
    rest = [t for t in texts if t["role"] is None]
    head = max(rest, key=lambda t: (t["font"], t["area"])) if rest else None
    if head:
        head["role"] = "headline"
    for tx in rest:
        if tx["role"]:
            continue
        w = _words(tx["text"]); low = tx["text"].lower()
        cx, cy = (tx["x0"] + tx["x1"]) / 2 / W, (tx["y0"] + tx["y1"]) / 2 / H
        edge = min(cx, 1 - cx, cy, 1 - cy) < 0.15
        if (len(w) <= 2 and len(tx["text"].strip()) <= 18 and not any(c.isdigit() for c in low)
                and not any(u in low for u in URLISH) and edge and head and tx["font"] <= 0.7 * head["font"]):
            tx["role"] = "logo"
        else:
            tx["role"] = "body"
    pics = sorted((it for it in items if it["role"] is None and it["t"] in ("ImageElement", "SvgMaskElement")
                   and it["area"] >= 0.05), key=lambda t: -t["area"])
    if pics:
        pics[0]["role"] = "product"
    for it in items:
        if it["role"] is None and it["t"] != "TextElement" and it["area"] < 0.06:
            w, h = it["x1"] - it["x0"], it["y1"] - it["y0"]
            cx, cy = (it["x0"] + it["x1"]) / 2 / W, (it["y0"] + it["y1"]) / 2 / H
            if max(w, h) / max(1e-6, min(w, h)) <= 6 and min(cx, 1 - cx, cy, 1 - cy) < 0.18 and it["t"] != "ColoredBackground":
                it["role"] = "logo"
    for it in items:
        if it["role"] is None:
            it["role"] = "decoration"
    items = [it for it in items if it["role"] != "cta_box"]  # folded into its CTA's box
    # a layout should have at most one logo: keep the topmost-largest, demote the rest
    logos = [it for it in items if it["role"] == "logo"]
    if len(logos) > 1:
        keep = max(logos, key=lambda t: t["area"])
        for it in logos:
            if it is not keep:
                it["role"] = "decoration"
    return {"W": W, "H": H, "items": items}
