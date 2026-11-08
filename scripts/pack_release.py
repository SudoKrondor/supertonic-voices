"""Pack the voice.bin that matches Elite Intel's SupertonicVoices enum, id for id.

The enum is the source of truth: speaker id N in voice.bin must be the voice the enum gives sid N.
Ids 0-9 (F1-M5) come from the original shipped voice.bin; every other entry comes from the fitted
JSON with the same name (out/step5/<NAME>/<NAME>.json). An id the enum skips gets a copy of F1 as a
filler, so the ids after it don't shift.

    .venv/bin/python scripts/pack_release.py

Writes out/release/voice.bin. Nothing in the app is touched; copy it there yourself.
"""

import re
import sys
from pathlib import Path

import numpy as np

import voicebin
from check_int8 import load_voice_json
from common import OUT

ENUM = Path.home() / "projects/EliteIntel/app/src/main/java/elite/intel/ai/mouth/supertonic/SupertonicVoices.java"
ORIGINAL = OUT / "voice_bin_backup" / "voice.bin.original"
PRESETS = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]


def enum_voices() -> dict[int, str]:
    """sid -> enum name, for the entries that are not commented out."""
    pat = re.compile(r'^\s+([A-Z_0-9]+)\((\d+),\s*(?:true|false),')
    voices = {}
    for line in ENUM.read_text().splitlines():
        if m := pat.match(line):
            sid, name = int(m[2]), m[1]
            assert sid not in voices, f"sid {sid} used by both {voices[sid]} and {name}"
            voices[sid] = name
    return voices


def main() -> None:
    voices = enum_voices()
    ttl0, dp0 = voicebin.read(ORIGINAL)
    for sid, name in enumerate(PRESETS):
        assert voices.get(sid) == name, f"enum sid {sid} should be {name}, is {voices.get(sid)}"

    missing = [n for s, n in voices.items() if s >= len(PRESETS) and not (OUT / "step5" / n / f"{n}.json").exists()]
    if missing:
        sys.exit(f"not fitted yet: {', '.join(missing)}")

    ttl, dp, gaps = [], [], []
    for sid in range(max(voices) + 1):
        name = voices.get(sid)
        if name is None:
            gaps.append(sid)
            t, d = ttl0[0:1], dp0[0:1]
        elif sid < len(PRESETS):
            t, d = ttl0[sid:sid + 1], dp0[sid:sid + 1]
        else:
            t, d = load_voice_json(OUT / "step5" / name / f"{name}.json")
        ttl.append(t)
        dp.append(d)

    out = OUT / "release"
    out.mkdir(parents=True, exist_ok=True)
    voicebin.write(out / "voice.bin", np.concatenate(ttl), np.concatenate(dp))
    n = voicebin.read(out / "voice.bin")[0].shape[0]
    print(f"wrote {out / 'voice.bin'}: {n} speakers (ids 0-{n - 1}), {len(voices)} named in the enum")
    if gaps:
        print(f"ids not in the enum, filled with a copy of F1: {gaps}")


if __name__ == "__main__":
    main()
