"""Keep the live dashboard (distil-research/dashboard/, on Vercel) up to date:
every few seconds, build a status snapshot, upload it to the project's Vercel
Blob store as a new blob, and delete the older ones. The dashboard's
/api/status returns the newest. Read-only on the pipeline itself.

    python -m dev.distill.publish_status [--every 15] [--once]

Needs BLOB_READ_WRITE_TOKEN in distil-research/dashboard/.env.local
(`vercel env pull .env.local` from the dashboard folder). The token is never
printed.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dev.distill.paths import DATA_ROOT
from dev.distill.status import snapshot

DASHBOARD = DATA_ROOT.parent / "dashboard"
HISTORY = DASHBOARD / ".data.nosync" / "history.json"
MAX_POINTS = 2000
KEEP = 2  # newest snapshots kept in the store


def _token() -> str:
    env = DASHBOARD / ".env.local"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("BLOB_READ_WRITE_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit(f"BLOB_READ_WRITE_TOKEN not found in {env}; run `vercel env pull .env.local` in {DASHBOARD}")


def _history(point: list) -> list:
    points = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
    if not points or points[-1][1:] != point[1:] or point[0] - points[-1][0] >= 60:
        points.append(point)  # keep a point when something changed, and at least once a minute
    points = points[-MAX_POINTS:]
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(points))
    return points


def publish_once(token: str, uploaded: list[str]) -> str:
    snap = snapshot()
    snap["history"] = _history([round(snap["generated_at"]), snap["judging"]["done"], snap["judging"]["workers"]])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(snap, f)
    try:
        out = subprocess.run(["vercel", "blob", "put", f.name, "--pathname", "status/snapshot.json",
                              "--add-random-suffix", "true", "--access", "public", "--content-type", "application/json",
                              "--rw-token", token, "--no-color"],
                             capture_output=True, text=True, cwd=DASHBOARD, timeout=120)
    finally:
        Path(f.name).unlink(missing_ok=True)
    found = re.findall(r"https://\S+\.blob\.vercel-storage\.com/\S+", out.stdout + out.stderr)
    if out.returncode or not found:
        raise RuntimeError(f"blob put failed (exit {out.returncode}): {(out.stderr or out.stdout)[-300:]}")
    uploaded.append(found[0].rstrip(".,)"))
    while len(uploaded) > KEEP:
        old = uploaded.pop(0)
        subprocess.run(["vercel", "blob", "del", old, "--rw-token", token, "--no-color"],
                       capture_output=True, text=True, cwd=DASHBOARD, timeout=120)
    j = snap["judging"]
    return f"judged {j['done']}/{j['expected']}, workers {j['workers']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--every", type=float, default=15.0, help="seconds between snapshots")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    token, uploaded = _token(), []
    while True:
        started = time.time()
        try:
            print(time.strftime("%H:%M:%S"), publish_once(token, uploaded), flush=True)
        except Exception as e:  # keep publishing through transient failures
            print(time.strftime("%H:%M:%S"), f"publish failed: {e}", file=sys.stderr, flush=True)
        if args.once:
            break
        time.sleep(max(1.0, args.every - (time.time() - started)))


if __name__ == "__main__":
    main()
