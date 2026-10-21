"""Local progress dashboard: serves distil-research/dashboard/index.html and a
live /api/status built from the training data (dev.distill.status), on
localhost only. Nothing is uploaded anywhere.

    python -m dev.distill.serve_dashboard [--port 8765] [--every 15]
    then open http://127.0.0.1:8765
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dev.distill.paths import DATA_ROOT
from dev.distill.status import snapshot

DASHBOARD = DATA_ROOT.parent / "dashboard"
HISTORY = DASHBOARD / ".data.nosync" / "history.json"
MAX_POINTS = 2000

_latest: dict = {"body": b'{"error": "starting"}'}


def _history(point: list) -> list:
    points = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
    if not points or points[-1][1:] != point[1:] or point[0] - points[-1][0] >= 60:
        points.append(point)  # a point when something changed, and at least once a minute
    points = points[-MAX_POINTS:]
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(points))
    return points


def refresh_forever(every: float) -> None:
    while True:
        try:
            snap = snapshot()
            snap["history"] = _history([round(snap["generated_at"]), snap["judging"]["done"], snap["judging"]["workers"]])
            _latest["body"] = json.dumps(snap).encode()
        except Exception as e:  # keep serving the last good snapshot
            print(f"[dashboard] snapshot failed: {e}", flush=True)
        time.sleep(every)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            body, ctype = _latest["body"], "application/json"
        elif path in ("/", "/index.html"):
            body, ctype = (DASHBOARD / "index.html").read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--every", type=float, default=15.0, help="seconds between status snapshots")
    args = ap.parse_args()
    threading.Thread(target=refresh_forever, args=(args.every,), daemon=True).start()
    print(f"dashboard: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
