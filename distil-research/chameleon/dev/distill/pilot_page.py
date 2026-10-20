"""A self-contained HTML page for the person's half of the pilot: each pair
shown at its real size next to the original, A / B / tie buttons, the rubric
on top, and a button that downloads human_answers.csv. Images are embedded,
so the page works offline; it lives in a *.nosync folder and is not committed.

    python -m dev.distill.pilot_page        # then open the printed path
"""

import base64
import html
import json
from pathlib import Path

from dev.distill.judge import RUBRIC
from dev.distill.paths import DATA_ROOT, POOL_DIR, RENDERS_SUBDIR

PILOT_DIR = DATA_ROOT / "judgments" / "pilot"


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def write_page(pilot_dir: Path = PILOT_DIR, pool_dir: Path = POOL_DIR) -> Path:
    pairs = json.loads((pilot_dir / "pairs.json").read_text())
    rows = {}
    for line in (pool_dir / "pool.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[(r["cell"], r["candidate_id"])] = r
    blocks = []
    for pid, p in pairs.items():
        src = pool_dir / RENDERS_SUBDIR / p["task_id"] / "source.png"
        a, b = rows[(p["cell"], p["a"])], rows[(p["cell"], p["b"])]
        target = html.escape(a["target"])
        options = "".join(
            f'<label><input type="radio" name="{pid}" value="{v}"> {t}</label>'
            for v, t in (("a", "A is better"), ("b", "B is better"), ("tie", "Tie")))
        blocks.append(f"""
<section class="pair" id="{pid}">
  <h2>{pid} <small>resized to {target}</small></h2>
  <div class="row">
    <figure class="orig"><figcaption>Original</figcaption><img src="{_img(src)}" alt="original"></figure>
    <figure><figcaption>A</figcaption><img src="{_img(pool_dir / a['png'])}" alt="A"></figure>
    <figure><figcaption>B</figcaption><img src="{_img(pool_dir / b['png'])}" alt="B"></figure>
  </div>
  <div class="choice">{options}</div>
</section>""")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pilot Judging</title>
<style>
:root {{ --bg:#f6f5f2; --fg:#1d1b18; --muted:#6b665e; --card:#fff; --line:#ddd8cf; --accent:#c4501b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17161a; --fg:#ecebe8; --muted:#a19c94; --card:#222127; --line:#38363d; --accent:#f08a4b; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1400px; padding:16px; }}
header {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1; }}
pre {{ white-space:pre-wrap; background:var(--card); border:1px solid var(--line); padding:12px; font:13px/1.45 system-ui; }}
.pair {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin:16px 0; }}
.pair h2 {{ margin:0 0 8px; font-size:16px; }} small {{ color:var(--muted); font-weight:normal; }}
.row {{ display:flex; gap:12px; flex-wrap:wrap; align-items:flex-start; }}
figure {{ margin:0; flex:1 1 300px; min-width:0; }} figure.orig {{ flex:0 1 220px; }}
figcaption {{ font-weight:600; color:var(--muted); }}
img {{ max-width:100%; max-height:70vh; border:1px solid var(--line); display:block; }}
.choice {{ display:flex; gap:20px; margin-top:10px; flex-wrap:wrap; }}
.choice label {{ cursor:pointer; }}
button {{ background:var(--accent); color:#fff; border:0; border-radius:6px; padding:8px 14px; font-size:14px; cursor:pointer; }}
textarea {{ width:100%; height:90px; background:var(--card); color:var(--fg); border:1px solid var(--line); }}
</style></head><body>
<header><strong>Pilot judging</strong> · <span id="count">0</span>/{len(pairs)} answered ·
<button id="dl">Download human_answers.csv</button></header>
<p>For each pair: which resized version (A or B) would you ship? Judge blind, using only the criteria below. Save the CSV over
<code>training-data/judgments/pilot/human_answers.csv</code>, then tell Claude to score the pilot.</p>
<pre>{html.escape(RUBRIC)}</pre>
{''.join(blocks)}
<h2>Your answers (CSV)</h2><textarea id="csv" readonly></textarea>
<script>
const ids = {json.dumps(list(pairs))};
const KEY = "pilot-answers";
function load() {{ try {{ return JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ return {{}}; }} }}
function save(a) {{ try {{ localStorage.setItem(KEY, JSON.stringify(a)); }} catch (e) {{}} }}
function csv(a) {{ return "pair_id,winner\\n" + ids.map(i => i + "," + (a[i] || "")).join("\\n") + "\\n"; }}
function refresh() {{
  const a = {{}};
  ids.forEach(i => {{ const c = document.querySelector('input[name="' + i + '"]:checked'); if (c) a[i] = c.value; }});
  save(a); document.getElementById("csv").value = csv(a);
  document.getElementById("count").textContent = Object.keys(a).length;
}}
const saved = load();
ids.forEach(i => {{ if (saved[i]) {{ const r = document.querySelector('input[name="' + i + '"][value="' + saved[i] + '"]'); if (r) r.checked = true; }} }});
document.addEventListener("change", refresh); refresh();
document.getElementById("dl").onclick = () => {{
  const blob = new Blob([document.getElementById("csv").value], {{type: "text/csv"}});
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "human_answers.csv"; link.click();
}};
</script></body></html>"""
    out = pilot_dir / "page.nosync" / "judge.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


if __name__ == "__main__":
    print(write_page())
