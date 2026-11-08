#!/usr/bin/env bash
# Elite Intel release: pack out/release/voice.bin from the SupertonicVoices enum.
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python scripts/pack_release.py
