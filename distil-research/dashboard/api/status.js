// Returns the newest status snapshot the local publisher uploaded to Vercel Blob.
// Each snapshot is a new blob (random suffix), so nothing is ever served stale;
// this function itself is never cached.
import { list } from "@vercel/blob";

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  try {
    const { blobs } = await list({ prefix: "status/", limit: 100 });
    if (!blobs.length) return res.status(404).json({ error: "no snapshot published yet" });
    const newest = blobs.reduce((a, b) => (new Date(a.uploadedAt) > new Date(b.uploadedAt) ? a : b));
    const r = await fetch(newest.url, { cache: "no-store" });
    res.setHeader("Content-Type", "application/json");
    return res.status(200).send(await r.text());
  } catch (err) {
    return res.status(500).json({ error: String(err) });
  }
}
