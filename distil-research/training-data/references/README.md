# references/

Reference material about ad formats, separate from the training sets.

## format_specs.json
What: per-size (13 targets) cited conventions and safe zones from official guidance (Meta, Google, YouTube, LinkedIn). Only verified fields; `safe_zones` entries are `{edge, share, source}`, `conventions` are `{rule, source}`. No `min_text_px`: no checked source states one. Facebook cover has no verified conventions.
How: hand-researched from web help pages on 2026-10-21; human-readable version in `docs/format-specs.md` (includes unverified items).
Feeds: possible layout rules/priors, chiefly the Story/Reels safe bands (top 14%, bottom 14-35%, sides 6%).

## crello_layout_stats.json (+ engine_layout_stats_base.json, engine_layout_stats_bisect.json)
What: aggregate statistics of where real designer-made ad templates put things at each of our 13 target sizes (role presence, centre x/y median and IQR, area share, font size as a share of the short side, edge margins, empty space, reading order), and the same numbers for today's engine and bisection on the holdout cells. Aggregates only: no template content, text or images.
How: one shard (`cyberagent/crello` revision 5.1.0, `data/test-00000-of-00004.parquet`, 366 MB, 493 templates) downloaded to `references/crello.nosync/` (gitignored, never committed). Filtered to ad/social/promo formats, pooled by aspect (within 5%, else the 20 nearest-aspect templates), roles labelled by heuristics (`chameleon/dev/references/crello_roles.py`; CTA and logo labels are weak, see accuracy in `docs/real-ad-layouts.md`). Rebuild: `crello_stats.py stats` (scratch venv with pyarrow), `python -m dev.references.engine_stats base|bisect` (project venv), `dev/references/make_report.py`.
Feeds: `docs/real-ad-layouts.md`, a possible layout prior or a sanity check for grader/engine rules.
Licence: Crello is CDLA-Permissive-2.0; the underlying VistaCreate template terms are unverified, so the raw data stays local and only aggregates are committed. Cite CyberAgent/Crello (Yamaguchi, CanvasVAE, ICCV 2021).
