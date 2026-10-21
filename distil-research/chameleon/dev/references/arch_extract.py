"""Step 1a (scratch venv with pyarrow): dump role-labelled Crello layouts + per-target pools to archetypes.nosync/crello_layouts.json.
Raw-derived (boxes only), stays in .nosync."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crello_stats as cs

OUT = cs.REF / "archetypes.nosync"
OUT.mkdir(exist_ok=True)
layouts, meta, _ = cs.load_ad_layouts()
P = cs.pools(meta)
slim = [{"id": l["id"], "format": l["format"], "W": l["W"], "H": l["H"],
         "items": [{k: it[k] for k in ("role", "x0", "y0", "x1", "y1")} for it in l["items"]]} for l in layouts]
(OUT / "crello_layouts.json").write_text(json.dumps({"layouts": slim, "pools": {k: v for k, v in P.items()}}))
print(len(slim), {k: (v["kind"], len(v["idx"])) for k, v in P.items()})
