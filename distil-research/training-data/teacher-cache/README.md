# teacher-cache: raw teacher answers (gitignored)

**What it is:** every successful teacher call's answer, stored under a hash of (teacher, prompt, schema, image bytes). Written by `chameleon/dev/distill/teacher.py`.

**Why:** re-running a step (after a crash, a code change downstream, or on another day) re-uses the answers instead of spending Claude Code usage again. Failures are never cached.

**Not committed:** it's large and fully reproducible from the commands that produced `judgments/` etc. Delete the folder's contents to force fresh answers.

**Where exactly:** `teacher-cache/cache.nosync/`. The `.nosync` suffix keeps iCloud Drive from syncing it, because the repo lives in `~/Documents`.
