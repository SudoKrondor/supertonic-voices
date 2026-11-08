#!/usr/bin/env bash
# Start or resume the overnight batch fit of the whole Kokoro cast (ADAM first) from an English essay.
# Safe to rerun at any time: voices that are already fitted are skipped, and a voice that was
# interrupted mid-way is redone from the start. Runs detached, so the terminal can be closed.
#
#     bash scripts/resume_batch.sh ESSAY.txt
set -euo pipefail
ESSAY="$(realpath "${1:?usage: resume_batch.sh ESSAY.txt}")"
cd "$(dirname "$0")"

if pgrep -f "python -u batch_fit" >/dev/null; then
    echo "batch is already running:"; tail -3 ../out/batch/batch.log; exit 0
fi

ORDER="ADAM $(../.venv/bin/python -c 'from batch_fit import cast; print(" ".join(v.name for v in cast() if v.name != "ADAM"))')"
mkdir -p ../out/batch
setsid nohup ../.venv/bin/python -u batch_fit.py "$ESSAY" $ORDER \
    >> ../out/batch/batch.log 2>&1 < /dev/null &
echo "batch started (log: out/batch/batch.log)"
