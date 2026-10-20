# Distillation dashboard

**Live:** https://chameleon-distill-dashboard.vercel.app. It is public, and refreshes every 15 s.

It shows the research pipeline's progress:
- data collected (ads, candidate layouts);
- teacher judging (progress over time, per worker, coverage by split and brand, ship-readiness scores, judge reliability);
- training results (learned vs hand-set grader accuracy, weights);
- the latest critiques and recent commits.

## How it stays live without redeploying

```
your laptop                                    Vercel
dev/distill/status.py  ── snapshot every 15 s ─►  Blob store "chameleon-distill-status" (status/*.json, new blob each time)
dev/distill/publish_status.py  (uploads, deletes old)        │
                                                             ▼
                              dashboard page ◄── polls ── /api/status  (newest snapshot, never cached)
```

- **`index.html`:** the page. Plain HTML and SVG with no build step; light/dark themes, hover tooltips, and a table view under each chart.
- **`api/status.js`:** a Vercel function. It lists the Blob store and returns the newest snapshot with `Cache-Control: no-store`.
- **The publisher** runs on whichever laptop runs the pipeline. It only *reads* the training data:
  ```bash
  cd distil-research/chameleon
  ../../chameleon/.venv/bin/python -m dev.distill.publish_status --every 15
  ```
  It needs `BLOB_READ_WRITE_TOKEN` in `dashboard/.env.local`. Get it with `vercel env pull .env.local` from this folder (after `vercel link --project chameleon-distill-dashboard`). That file is gitignored. **Never commit it.**
- **Redeploy the page** only when `index.html` or `api/` changes: `vercel deploy --prod` from this folder.

The "updated N s ago" dot turns amber when the publisher hasn't sent anything for 90 s, for example when the laptop is asleep.
