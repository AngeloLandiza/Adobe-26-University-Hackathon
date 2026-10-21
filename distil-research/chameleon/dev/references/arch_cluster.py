"""Steps 1-4 (scratch venv with numpy/scikit-learn/matplotlib): vectorise role-labelled Crello layouts, cluster per size,
map engine/bisection layouts onto the real cloud, draw plots + box diagrams, write floor plans.
Inputs : training-data/references/archetypes.nosync/{crello_layouts,engine_base,engine_bisect}.json
Outputs: same folder (plots, diagrams, archetypes.json) + training-data/references/archetype_floorplans.json
Run    : python -I dev/references/arch_cluster.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
REF = HERE.parents[2] / "training-data" / "references"
OUT = REF / "archetypes.nosync"
TARGETS = json.loads((HERE.parent / "targets.json").read_text())
ROLES = ("headline", "body", "product", "cta", "logo")
WEIGHT = {"headline": 1.0, "body": 1.0, "product": 1.0, "cta": 0.4, "logo": 0.4}  # CTA/logo labels are ~25-30% precise
COL = {"headline": "#d62728", "body": "#ff9900", "product": "#b020b0", "cta": "#2ca02c", "logo": "#1f5fff"}
MIN_STRICT, MIN_SHARE, KMAX, SEED = 40, 0.05, 6, 0


def aclass(r):
    return "very_tall" if r < .55 else "tall" if r < .85 else "square" if r < 1.25 else "landscape" if r < 2.2 else "wide" if r <= 5 else "strip"


def boxes(layout):
    """role -> (cx, cy, w, h) canvas fractions. headline/product/cta/logo: largest item; body: union of body items."""
    W, H = layout["W"], layout["H"]
    out = {}
    for role in ROLES:
        its = [i for i in layout["items"] if i["role"] == role]
        its = [(max(0, i["x0"]), max(0, i["y0"]), min(W, i["x1"]), min(H, i["y1"])) for i in its]
        its = [b for b in its if b[2] > b[0] and b[3] > b[1]]
        if not its:
            continue
        if role == "body":
            b = (min(i[0] for i in its), min(i[1] for i in its), max(i[2] for i in its), max(i[3] for i in its))
        else:
            b = max(its, key=lambda i: (i[2] - i[0]) * (i[3] - i[1]))
        out[role] = ((b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H)
    return out


def vec(layout):
    bx = boxes(layout)
    v = []
    for role in ROLES:
        w = WEIGHT[role]
        v += [w * 0.5] + ([w * x for x in bx[role]] if role in bx else [0, 0, 0, 0])
        # presence bit scaled 0.5 so a missing role costs about as much as a mid-size box shift
    return np.array(v)


def pool_for(size, layouts, pools, ratio):
    """strict pool when >= 40 else the aspect-class pool (nearest 100 by aspect)."""
    p = pools[size]
    if p["kind"] == "strict" and len(p["idx"]) >= MIN_STRICT:
        return ("size:" + size), list(p["idx"]), "strict size pool"
    cls = aclass(ratio)
    idx = [i for i, l in enumerate(layouts) if aclass(l["W"] / l["H"]) == cls]
    idx.sort(key=lambda i: abs(math.log(layouts[i]["W"] / layouts[i]["H"] / ratio)))
    return ("class:" + cls + ("" if cls not in ("tall",) else ":" + size)), idx[:100], f"aspect-class pool ({cls}, nearest {min(100, len(idx))} of {len(idx)})"


def cluster(X):
    n = len(X)
    kmax = max(2, min(KMAX, n // 5))
    best = None
    scan = {}
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=20, random_state=SEED).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        s = float(silhouette_score(X, km.labels_))
        scan[k] = s
        if best is None or s > best[0]:
            best = (s, k, km)
    s, k, km = best
    # tie-break toward finer structure: k=2 is mostly 'hero present or not'; take k=3 when it is within 0.05 of the best silhouette (n >= 40)
    if n >= 40 and k == 2 and 3 in scan and scan[3] >= s - 0.05:
        km = KMeans(3, n_init=20, random_state=SEED).fit(X)
        s, k = scan[3], 3
    bic = {}
    for kk in range(2, kmax + 1):
        g = GaussianMixture(kk, covariance_type="diag", n_init=3, random_state=SEED, reg_covar=1e-3).fit(X)
        bic[kk] = float(g.bic(X))
    kg = min(bic, key=bic.get)
    gm = GaussianMixture(kg, covariance_type="diag", n_init=3, random_state=SEED, reg_covar=1e-3).fit(X)
    return {"k": k, "sil": s, "scan": scan, "labels": km.labels_, "centers": km.cluster_centers_,
            "kg": kg, "bic": bic, "glabels": gm.predict(X)}


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def describe(bx, ratio):
    """plain-words description of a role-box dict."""
    parts = []
    def pos(cx, cy):
        h = "left" if cx < .38 else "right" if cx > .62 else "centre"
        v = "top" if cy < .36 else "bottom" if cy > .64 else "middle"
        return f"{v}-{h}".replace("middle-centre", "centre")
    for role, name in (("product", "hero image"), ("headline", "headline"), ("body", "body text"), ("cta", "CTA?"), ("logo", "logo?")):
        if role in bx:
            cx, cy, w, h = bx[role]
            parts.append(f"{name} {pos(cx, cy)} ({w*h*100:.0f}% of canvas)")
    return "; ".join(parts) if parts else "no labelled roles"


def draw_boxes(ax, bx, title, ratio, small=False):
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="#f4f4f4", ec="k", lw=1))
    for role in ("product", "body", "headline", "cta", "logo"):
        if role in bx:
            cx, cy, w, h = bx[role]
            ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=COL[role], alpha=.45, ec=COL[role], lw=1.5))
            ax.text(cx, cy, role[:4], ha="center", va="center", fontsize=6 if small else 8, color="k")
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_box_aspect(1 / ratio)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=7 if small else 9)


def main():
    data = json.loads((OUT / "crello_layouts.json").read_text())
    layouts, pools = data["layouts"], data["pools"]
    eng = {n: json.loads((OUT / f"engine_{n}.json").read_text()) for n in ("base", "bisect")}
    tsize = {t["name"]: t for t in TARGETS}
    res, plans, pool_cache = {}, [], {}
    for t in TARGETS:
        size = t["name"]
        ratio = t["width"] / t["height"]
        key, idx, kind = pool_for(size, layouts, pools, ratio)
        if key in pool_cache:  # same pool already clustered for another size of the class
            res[size] = dict(pool_cache[key], sizes=pool_cache[key]["sizes"])
            pool_cache[key]["sizes"].append(size)
            continue
        L = [layouts[i] for i in idx]
        X = np.array([vec(l) for l in L])
        C = cluster(X)
        labels = C["labels"]
        n = len(L)
        arche = []
        for c in range(C["k"]):
            mem = np.where(labels == c)[0]
            if len(mem) / n < MIN_SHARE:
                continue
            cen = X[mem].mean(0)
            d = np.linalg.norm(X[mem] - cen, axis=1)
            med = mem[int(np.argmin(np.linalg.norm(X[mem][:, None] - X[mem][None], axis=2).sum(1)))]
            # mean role presence / box of members
            pres = {r: float(np.mean([r in boxes(L[m]) for m in mem])) for r in ROLES}
            mean_box = {}
            for r in ROLES:
                bs = [boxes(L[m])[r] for m in mem if r in boxes(L[m])]
                if bs and pres[r] >= .4:
                    mean_box[r] = tuple(float(x) for x in np.median(bs, 0))
            arche.append({"cluster": int(c), "n": int(len(mem)), "share": len(mem) / n, "medoid": int(med),
                          "medoid_id": L[med]["id"], "medoid_boxes": boxes(L[med]), "median_boxes": mean_box,
                          "presence": pres, "spread": float(d.mean()), "centroid": cen.tolist()})
        arche.sort(key=lambda a: -a["share"])
        pca = PCA(2, random_state=SEED).fit(X)
        entry = {"key": key, "kind": kind, "n": n, "ratio": ratio, "X": X, "labels": labels, "L": L, "C": C,
                 "arche": arche, "pca": pca, "sizes": [size]}
        pool_cache[key] = entry
        res[size] = entry
    # ---- distances of engine layouts to real cloud
    summary, rows = {}, []
    for key, e in pool_cache.items():
        X = e["X"]
        cents = np.array([a["centroid"] for a in e["arche"]])
        e["cents"] = cents
        d_real_arch = np.array([np.linalg.norm(cents - x, axis=1).min() for x in X])
        D = np.linalg.norm(X[:, None] - X[None], axis=2); np.fill_diagonal(D, 1e9)
        d_real_nn = D.min(1)
        e["d_real_arch"], e["d_real_nn"] = d_real_arch, d_real_nn
        e["eng"] = {}
        for n, lst in eng.items():
            for size in e["sizes"]:
                V = np.array([vec(l) for l in lst if l["target"] == size])
                if len(V) == 0:
                    continue
                da = np.array([np.linalg.norm(cents - x, axis=1).min() for x in V])
                dn = np.array([np.linalg.norm(X - x, axis=1).min() for x in V])
                e["eng"].setdefault(n, {})[size] = {"V": V, "d_arch": da, "d_nn": dn}
        for size in e["sizes"]:
            row = {"size": size, "pool": e["kind"], "n_pool": e["n"], "real_med_to_arch": float(np.median(d_real_arch)),
                   "real_p90_to_arch": float(np.percentile(d_real_arch, 90)), "real_med_nn": float(np.median(d_real_nn))}
            for n in ("base", "bisect"):
                g = e["eng"].get(n, {}).get(size)
                if g is not None:
                    row[n] = {"n": len(g["V"]), "med_to_arch": float(np.median(g["d_arch"])), "med_nn": float(np.median(g["d_nn"])),
                              "frac_beyond_real_p90": float(np.mean(g["d_arch"] > np.percentile(d_real_arch, 90))),
                              # share of the engine layouts whose nearest real layout is farther than the median real layout's nearest-real distance
                              "frac_nn_beyond_real_p90nn": float(np.mean(g["d_nn"] > np.percentile(d_real_nn, 90))),
                              "med_to_arch_core": float(np.median([np.linalg.norm(cents[:, :15] - x[:15], axis=1).min() for x in g["V"]])),
                              "hero_area_med_when_present": float(np.median([bb["product"][2]*bb["product"][3] for bb in (boxes(l) for l in eng[n] if l["target"] == size) if "product" in bb] or [0])),
                              "hero_present": float(np.mean(["product" in boxes(l) for l in eng[n] if l["target"] == size]))}
            row["real_med_to_arch_core"] = float(np.median([np.linalg.norm(cents[:, :15] - x[:15], axis=1).min() for x in X]))
            row["real_hero_area_med_when_present"] = float(np.median([bb["product"][2]*bb["product"][3] for bb in (boxes(l) for l in e["L"]) if "product" in bb] or [0]))
            row["real_hero_present"] = float(np.mean(["product" in boxes(l) for l in e["L"]]))
            rows.append(row)
    # ---- plots
    OUT.mkdir(exist_ok=True)
    for key, e in pool_cache.items():
        size0 = e["sizes"][0]
        tag = key.replace(":", "_")
        pca, X = e["pca"], e["X"]
        Z = pca.transform(X)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        cmap = plt.get_cmap("tab10")
        keep = {a["cluster"]: i for i, a in enumerate(e["arche"])}
        for c in range(e["C"]["k"]):
            m = e["C"]["labels"] == c
            lab = (f"A{keep[c]+1} ({e['arche'][keep[c]]['share']*100:.0f}%)" if c in keep else "dropped (<5%)")
            ax.scatter(Z[m, 0], Z[m, 1], s=28, color=cmap(keep[c]) if c in keep else "#bbb", label=lab, alpha=.8)
        for i, a in enumerate(e["arche"]):
            zm = pca.transform(np.array(a["centroid"])[None])[0]
            ax.scatter(*zm, marker="*", s=260, color=cmap(i), ec="k")
        for n, mk, col in (("base", "s", "k"), ("bisect", "^", "red")):
            for size in e["sizes"]:
                g = e["eng"].get(n, {}).get(size)
                if g is not None:
                    zz = pca.transform(g["V"])
                    ax.scatter(zz[:, 0], zz[:, 1], marker=mk, s=40, fc="none", ec=col, lw=1.2, label=f"engine {n} {size}" if len(e["sizes"]) > 1 else f"engine {n}")
        ax.set_title(f"{key}  n={e['n']}  k={len(e['arche'])} (sil {e['C']['sil']:.2f})\nPCA of [present,cx,cy,w,h] x headline/body/product/cta/logo ({pca.explained_variance_ratio_.sum()*100:.0f}% var)", fontsize=9)
        ax.legend(fontsize=6, loc="best"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT / f"pca_{tag}.png", dpi=110); plt.close(fig)
        # archetype diagram: median boxes + medoid boxes
        k = len(e["arche"])
        fig, axs = plt.subplots(2, k, figsize=(2.6 * k, 2.6 * 2 / max(.5, min(1.6, e["ratio"])) ** 0 + .6), squeeze=False)
        for i, a in enumerate(e["arche"]):
            draw_boxes(axs[0][i], a["medoid_boxes"], f"A{i+1} medoid ({a['share']*100:.0f}%, n={a['n']})", e["ratio"], True)
            draw_boxes(axs[1][i], a["median_boxes"], f"A{i+1} member median", e["ratio"], True)
        fig.suptitle(f"{key} (aspect {e['ratio']:.2f}); red=headline orange=body purple=hero green=CTA? blue=logo?", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / f"boxes_{tag}.png", dpi=110); plt.close(fig)
    # ---- floor plans from medoids
    out = []
    for key, e in pool_cache.items():
        out.append({"key": key, "sizes": e["sizes"], "kind": e["kind"], "n": e["n"], "k": len(e["arche"]), "silhouette": e["C"]["sil"],
                    "silhouette_scan": e["C"]["scan"], "gmm_k_bic": e["C"]["kg"], "ari_kmeans_gmm": ari(e["C"]["labels"], e["C"]["glabels"]),
                    "archetypes": [{k: v for k, v in a.items() if k != "centroid"} | {"words": describe(a["median_boxes"], e["ratio"]), "medoid_words": describe(a["medoid_boxes"], e["ratio"])} for a in e["arche"]],
                    "ratio": e["ratio"]})
    (OUT / "archetypes.json").write_text(json.dumps({"pools": out, "distance_rows": rows}, indent=1, default=float))
    for o in out:
        print(o["key"], o["sizes"], o["kind"], "n", o["n"], "k", o["k"], f"sil {o['silhouette']:.2f}", "gmm k", o["gmm_k_bic"], f"ARI {o['ari_kmeans_gmm']:.2f}")
        for i, a in enumerate(o["archetypes"]):
            print(f"   A{i+1} {a['share']*100:.0f}% n={a['n']} spread={a['spread']:.2f}: {a['words']}")
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
