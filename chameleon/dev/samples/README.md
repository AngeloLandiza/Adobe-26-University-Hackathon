# Dev-only source designs

Extra source designs for benchmarking node 1 (priority: roles + ranking). They live
here, not in `input/`, so `python -m engine.main`, `dev.run_matrix` and the test
suite keep using only the three official samples.

Each file has the same shape as `input/*.json` (`source` + `target_canvas`) and a
golden label file with the same stem in `dev/golden/`. The goldens carry a
`review` flag and per-unit `notes`; **they are Claude-drafted team judgements
and need a teammate's review before they count.**

| File | What's different on purpose |
|---|---|
| `dev_sample_04_flat_no_groups` | no `group_id` at all; units come from implicit labels; a big panel behind the copy (control) |
| `dev_sample_05_type_only` | no images: shape background, text wordmark instead of the logo image, CTA verb "find" |
| `dev_sample_06_rebate_fine_print` | text-heavy: three numbered steps, three legal lines, one over 160 characters |
| `dev_sample_07_rotated_sticker` | rotated "SAVE 20%" sticker (a badge, not the CTA) and a loose rotated callout |
| `dev_sample_08_ghost_word` | the biggest font is a faint background word, not the headline |
| `dev_sample_09_plain_text_cta` | plain-text CTA with no button, bigger than the subhead |
| `dev_sample_10_wide_lineup` | wide 1584x396 source; three bottles grouped as one hero (control) |

These were written by someone who had read the heuristic's rules, so they lean
toward the cases those rules don't cover. Treat them as a stress test, not a
random sample of real ads.

Run the benchmark over old + new designs (from `chameleon/`):

    .venv/Scripts/python -m dev.bench_priority --runs 2 --samples input/*.json dev/samples/*.json
