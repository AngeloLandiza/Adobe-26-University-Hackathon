import base64, io, json, sys
from pathlib import Path
from PIL import Image
O = Path(sys.argv[1]); ASSETS = Path(sys.argv[2]); FONTS = Path(sys.argv[3])
STAGES = ["base", "bisect", "v2", "v3", "v3L"]
EX = json.loads((O / "examples.json").read_text()); SIZES = json.loads((O / "sizes.json").read_text())
orig = json.loads((O / "stage_original.json").read_text())
st = {s: json.loads((O / f"stage_{s}.json").read_text()) for s in STAGES}
GEOM = ("x", "y", "width", "height")
def clean(e): return {k: v for k, v in e.items() if not k.startswith("_")}
def r(v): return round(v, 1) if isinstance(v, float) else v
assets, examples = set(), []
for ex in EX:
    src = orig[ex["id"]]["design"]
    els = [clean(e) for e in src["elements"]]
    for e in els:
        if e["type"] == "image": assets.add(e["asset_id"])
    byid = {e["id"]: e for e in els}
    outs = {}
    for sz in SIZES:
        k = f'{ex["id"]}@{sz["id"]}'; outs[sz["id"]] = {}
        for s in STAGES:
            o = st[s][k]; diff = {}
            for e in o["design"]["elements"]:
                e = clean(e); b = byid[e["id"]]
                diff[e["id"]] = {kk: r(v) for kk, v in e.items() if kk in GEOM or b.get(kk) != v}
                for kk in ("id", "type"): diff[e["id"]].pop(kk, None)
            outs[sz["id"]][s] = {"c": o["cand"], "f": o["fallback"], "g": diff, "png": o["png"]}
    examples.append({**ex, "canvas": src["canvas"], "els": els, "png": orig[ex["id"]]["png"], "outs": outs})
amap = {}
for a in sorted(assets):
    p = ASSETS / a
    if not p.exists(): print("missing asset", a); continue
    im = Image.open(p); im.thumbnail((640, 640))
    b = io.BytesIO(); im.save(b, "WEBP", quality=80); amap[a] = "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()
fonts = {}
for f in ["SolsticeSans-Regular", "SolsticeSans-Bold", "SolsticeSerif-Regular", "SolsticeSerif-Bold", "SolsticeSerif-Italic"]:
    fonts[f] = base64.b64encode((FONTS / f"{f}.ttf").read_bytes()).decode()
data = {"sizes": SIZES, "examples": examples, "assets": amap}
(O / "data.json").write_text(json.dumps(data, separators=(",", ":")))
(O / "fonts.json").write_text(json.dumps(fonts))
print("data", round(len((O / "data.json").read_text()) / 1e6, 2), "MB; fonts", round(sum(map(len, fonts.values())) / 1e6, 2), "MB; assets", len(amap))
