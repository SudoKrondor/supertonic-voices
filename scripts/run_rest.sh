#!/usr/bin/env bash
# What is left after the English batch: ERIC and GEORGE from a male preset (their first fits came
# out female), then the eleven non-English voices from clips spoken in their own language.
#
#     bash scripts/run_rest.sh            # detached; progress in out/batch/rest.log
#
# NOT safe to rerun whole: stage 1 always refits ERIC and GEORGE. To resume, run only
# fit_nonenglish.py, which skips voices already refitted and reuses finished clips.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p ../out/batch
setsid nohup bash -c '
    echo "=== $(date +%H:%M) refitting ERIC and GEORGE from a male preset"
    ../.venv/bin/python -u refit_same_gender.py ERIC GEORGE
    echo "=== $(date +%H:%M) non-English voices, each in its own language"
    ../.venv/bin/python -u fit_nonenglish.py
    echo "=== $(date +%H:%M) all done; next: .venv/bin/python scripts/pack_release.py"
' >> ../out/batch/rest.log 2>&1 < /dev/null &
echo "started (log: out/batch/rest.log)"
