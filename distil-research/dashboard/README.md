# Distillation dashboard (local)

A local page showing the research pipeline's progress:
- data collected (ads, candidate layouts);
- active jobs and their workers;
- every teacher run (main judging, B2 floor plans, retrieval, A/B engine comparisons);
- coverage by split and brand;
- ship-readiness scores and judge reliability;
- training results, experiment logs and reports, the latest critiques, and recent commits.

**Run it** (localhost only; nothing is uploaded anywhere):

```bash
cd distil-research/chameleon
../../chameleon/.venv/bin/python -m dev.distill.serve_dashboard     # then open http://127.0.0.1:8765
```

- `serve_dashboard` serves `index.html` and a live `/api/status`.
- It rebuilds the status snapshot every 15 s from the files in `training-data/` (`dev/distill/status.py`, read-only).
- The progress history is kept in `.data.nosync/history.json` (gitignored).
- The page polls every 15 s. Leave the server running during long jobs.

(It was briefly hosted on Vercel; that deployment was removed on 2026-10-21 at the team's request.)
