"""Crello shard -> per-target aggregate layout statistics (and a role-labelling review sheet).

    <scratch venv>/bin/python -I -m ... no: run as a script (pyarrow/pandas/PIL needed, not the project venv):
    python crello_stats.py stats            # writes training-data/references/crello_layout_stats.json
    python crello_stats.py review [seed]    # contact sheets into crello.nosync/review/
Raw rows are only read locally; the JSON output holds aggregates only.
"""
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pyarrow.parquet as pq
from crello_roles import label
from layout_stats import aggregate

ROOT = HERE.parents[2]  # distil-research
REF = ROOT / "training-data" / "references"
SHARD = REF / "crello.nosync" / "data" / "test-00000-of-00004.parquet"
TARGETS = json.loads((HERE.parent / "targets.json").read_text())
# Ad / social-post / promo formats. Everything else (resume, invitation, calendar, card, certificate, menu, logo,
# presentation, zoom background, mind map, invoice, label, ticket, book cover, ...) is excluded.
AD_FORMATS = {"Instagram Story", "Instagram", "Facebook", "Facebook cover", "Twitter", "Facebook AD", "Instagram AD",
              "Tumblr", "Image", "Pinterest", "Medium Rectangle", "Large Rectangle", "Skyscraper", "Leaderboard",
              "Youtube Thumbnail", "Youtube", "FB event cover", "Email header", "VK Universal Post",
              "VK Post with Button", "VK Community Cover", "LinkedIn Cover", "Web Banner", "Graphic", "Poster",
              "Poster US", "Flayer", "IGTV Cover", "Twitch Profile Banner", "Title"}
TOL = 0.05
MIN_POOL = 20
COLS = ["id", "length", "format", "canvas_width", "canvas_height", "type", "left", "top", "width", "height",
        "opacity", "text", "font_size"]


def feats():
    return json.loads(pq.ParquetFile(SHARD).schema_arrow.metadata[b"huggingface"])["info"]["features"]


def shards():
    """every complete test-*.parquet shard present (partial downloads end in .part and are ignored)."""
    return sorted(SHARD.parent.glob("test-*-of-*.parquet"))


def iter_rows(extra=()):
    for sh in shards():
        f = pq.ParquetFile(sh)
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=COLS + list(extra))
            for r in t.to_pylist():
                yield r


def pools(meta):
    """{target: {"strict": [idx], "pool": [idx], "pool_kind": ..}} over ad-format designs; meta = list of (w,h,fmt)."""
    out = {}
    for t in TARGETS:
        ta = t["width"] / t["height"]
        strict = [i for i, (w, h, f) in enumerate(meta) if abs((w / h) / ta - 1) <= TOL]
        if len(strict) >= MIN_POOL:
            out[t["name"]] = {"idx": strict, "kind": "strict", "n_strict": len(strict)}
        else:
            near = sorted(range(len(meta)), key=lambda i: abs(math.log((meta[i][0] / meta[i][1]) / ta)))[:MIN_POOL]
            dev = max(abs((meta[i][0] / meta[i][1]) / ta - 1) for i in near)
            out[t["name"]] = {"idx": near, "kind": "nearest-aspect", "n_strict": len(strict),
                              "max_aspect_deviation": round(dev, 3),
                              "aspect_range": [round(min(meta[i][0] / meta[i][1] for i in near), 3),
                                               round(max(meta[i][0] / meta[i][1] for i in near), 3)]}
    return out


def load_ad_layouts():
    F = feats()
    fmt_names = F["format"]["names"]
    layouts, meta = [], []
    n_all = 0
    for r in iter_rows():
        n_all += 1
        f = fmt_names[r["format"]]
        if f not in AD_FORMATS or r["canvas_width"] < 50 or r["canvas_height"] < 50:
            continue
        lay = label(r, F)
        lay["format"], lay["id"] = f, r["id"]
        layouts.append(lay)
        meta.append((r["canvas_width"], r["canvas_height"], f))
    return layouts, meta, n_all


def main_stats():
    layouts, meta, n_all = load_ad_layouts()
    P = pools(meta)
    out = {"source": f"cyberagent/crello revision 5.1.0, {len(shards())} of 4 test shards (data/test-0000N-of-00004.parquet)",
           "n_designs_in_shard": n_all, "n_ad_format_designs": len(layouts),
           "role_labelling": "heuristic, see dev/references/crello_roles.py; accuracy in docs/real-ad-layouts.md",
           "aspect_tolerance": TOL, "min_pool": MIN_POOL, "targets": {}}
    for t in TARGETS:
        p = P[t["name"]]
        stats = aggregate([layouts[i] for i in p["idx"]])
        fm = {}
        for i in p["idx"]:
            fm[layouts[i]["format"]] = fm.get(layouts[i]["format"], 0) + 1
        out["targets"][t["name"]] = {"size": [t["width"], t["height"]], "pool_kind": p["kind"],
                                     "n_matches_strict": p["n_strict"], "n_used": len(p["idx"]),
                                     **{k: p[k] for k in ("max_aspect_deviation", "aspect_range") if k in p},
                                     "formats_used": dict(sorted(fm.items(), key=lambda kv: -kv[1])), **stats}
    (REF / "crello_layout_stats.json").write_text(json.dumps(out, indent=1))
    for t in TARGETS:
        o = out["targets"][t["name"]]
        print(t["name"], o["pool_kind"], "strict", o["n_matches_strict"], "used", o["n_used"], o.get("aspect_range", ""))


def main_review(seed=0):
    from PIL import Image, ImageDraw
    F = feats()
    fmt_names = F["format"]["names"]
    rows = []
    for r in iter_rows(extra=["preview"]):
        if fmt_names[r["format"]] in AD_FORMATS:
            rows.append(r)
    random.Random(seed).shuffle(rows)
    rows = rows[:30]
    colours = {"headline": (230, 20, 20), "body": (240, 150, 0), "cta": (0, 170, 0), "logo": (0, 90, 255),
               "product": (200, 0, 200), "decoration": (130, 130, 130), "background": (0, 0, 0)}
    out = REF / "crello.nosync" / "review"
    out.mkdir(exist_ok=True)
    CELL = 420
    for s in range(3):
        sheet = Image.new("RGB", (5 * CELL, 2 * CELL + 40), "white")
        d0 = ImageDraw.Draw(sheet)
        d0.text((6, 4), "red=headline orange=body green=CTA blue=logo magenta=product grey=decoration (background not drawn)", fill="black")
        for k, r in enumerate(rows[s * 10:(s + 1) * 10]):
            lay = label(r, F)
            im = Image.open(__import__("io").BytesIO(r["preview"]["bytes"])).convert("RGB")
            sc = (CELL - 8) / max(lay["W"], lay["H"])
            im = im.resize((max(1, int(lay["W"] * sc)), max(1, int(lay["H"] * sc))))
            d = ImageDraw.Draw(im)
            for it in lay["items"]:
                if it["role"] == "background":
                    continue
                c = colours[it["role"]]
                w = 3 if it["role"] != "decoration" else 1
                d.rectangle([it["x0"] * sc, it["y0"] * sc, it["x1"] * sc, it["y1"] * sc], outline=c, width=w)
                if it["role"] not in ("decoration",):
                    d.text((it["x0"] * sc + 3, it["y0"] * sc + 2), it["role"][:4], fill=c)
            sheet.paste(im, ((k % 5) * CELL + 4, 40 + (k // 5) * CELL + 4))
            d0.text(((k % 5) * CELL + 6, 40 + (k // 5) * CELL + CELL - 14), f"#{s * 10 + k + 1}", fill="black")
        sheet.save(out / f"sheet{s + 1}.png")
    print("wrote", out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    (main_review(int(sys.argv[2]) if len(sys.argv) > 2 else 0) if cmd == "review" else main_stats())
