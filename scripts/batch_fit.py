"""Fit every voice in Elite Intel's Kokoro cast, then pack them into one voice.bin.

Resumable: a voice whose clips.tsv exists is not regenerated, and one whose fitted JSON exists
is not refitted, so rerunning after an interruption picks up where it stopped. Clip generation
(CPU) for the next voice runs while the current one fits (GPU).

    .venv/bin/python scripts/batch_fit.py ESSAY.txt [VOICE ...]     # default: the whole enabled cast, in its order

Writes out/batch/voice.bin (F1-M5 as ids 0-9, then the fitted voices), out/batch/voices.tsv
(id, voice, start preset, similarity before/after) and out/batch/enum.txt with suggested
SupertonicVoices entries. The app's own files are not touched.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import voicebin
from check_int8 import APP_MODEL, load_voice_json
from common import OUT, ROOT

KOKORO_ENUM = Path.home() / "projects/EliteIntel/app/src/main/java/elite/intel/ai/mouth/kokoro/KokoroVoices.java"
PY = str(ROOT / ".venv" / "bin" / "python")
SCRIPTS = ROOT / "scripts"
ITERS = 400
FIRST_SID = 10


def cast() -> list[tuple[str, int, bool, str, str]]:
    """Enabled (not commented-out) entries: (name, sid, male, display name, description)."""
    pat = re.compile(r'^\s+([A-Z_]+)\((\d+),\s*(true|false),\s*"([^"]*)",\s*"([^"]*)"\)')
    return [(m[1], int(m[2]), m[3] == "true", m[4], m[5])
            for line in KOKORO_ENUM.read_text().splitlines() if (m := pat.match(line))]


def clips_ready(voice: str) -> bool:
    return (ROOT / "data" / "kokoro" / voice / "clips.tsv").exists()


def fitted(voice: str) -> Path:
    return OUT / "step5" / voice / f"{voice}.json"


def start_clips(voice: str, sid: int, essay: str) -> subprocess.Popen | None:
    if clips_ready(voice):
        return None
    log = (OUT / "kokoro" / f"make_{voice}.log").open("w")
    return subprocess.Popen([PY, "make_kokoro_clips.py", voice, str(sid), essay], cwd=SCRIPTS, stdout=log, stderr=log)


def summary(voice: str) -> tuple[str, float, float]:
    """(start preset, held-out similarity of start, of fitted) from the fit's log."""
    log = (OUT / "step5" / voice / "run.log").read_text()
    start = re.search(r"starting from (\w+);", log)[1]
    rows = re.findall(r"^\s+\d{4}\s+start ([\d.]+)\s+fitted ([\d.]+)", log, re.M)
    s = np.array(rows, dtype=float).mean(0) if rows else (float("nan"), float("nan"))
    return start, float(s[0]), float(s[1])


def main(essay: str, *only: str) -> None:
    by_name = {v[0]: v for v in cast()}
    voices = [by_name[v] for v in only] if only else list(by_name.values())   # names given: in that order
    print(f"{len(voices)} voices: {', '.join(v[0] for v in voices)}", flush=True)
    (OUT / "kokoro").mkdir(parents=True, exist_ok=True)

    pending = start_clips(voices[0][0], voices[0][1], essay) if voices else None
    for i, (voice, sid, *_rest) in enumerate(voices):
        t0 = time.time()
        if pending:
            pending.wait()
        if not clips_ready(voice):
            print(f"[{voice}] clip generation failed, see out/kokoro/make_{voice}.log", flush=True)
            pending = start_clips(*voices[i + 1][:2], essay) if i + 1 < len(voices) else None
            continue
        # Next voice's clips on the CPU while this one fits on the GPU.
        pending = start_clips(*voices[i + 1][:2], essay) if i + 1 < len(voices) else None
        if fitted(voice).exists():
            print(f"[{voice}] already fitted, skipping", flush=True)
            continue
        out = OUT / "step5" / voice
        out.mkdir(parents=True, exist_ok=True)
        with (out / "run.log").open("w") as log:
            code = subprocess.run([PY, "step5_fit_kokoro.py", voice, str(ITERS)], cwd=SCRIPTS,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        if code or not fitted(voice).exists():
            print(f"[{voice}] fit failed (exit {code}), see {out / 'run.log'}", flush=True)
            continue
        start, before, after = summary(voice)
        print(f"[{voice}] done in {(time.time() - t0) / 60:.0f} min: from {start}, "
              f"similarity {before:.2f} -> {after:.2f}", flush=True)

    pack([v for v in cast() if fitted(v[0]).exists()])


def pack(done) -> None:
    """voice.bin with the shipped presets first, then every fitted voice. BELLA keeps id 10 if present."""
    done = sorted(done, key=lambda v: (v[0] != "BELLA", v[1]))
    ttl, dp = voicebin.read(APP_MODEL / "voice.bin")
    rows, enum = ["id\tvoice\tstart\tsim_start\tsim_fitted"], []
    for k, (voice, _, male, display, desc) in enumerate(done):
        t, d = load_voice_json(fitted(voice))
        ttl, dp = np.concatenate([ttl, t]), np.concatenate([dp, d])
        start, before, after = summary(voice)
        rows.append(f"{FIRST_SID + k}\t{voice}\t{start}\t{before:.2f}\t{after:.2f}")
        enum.append(f'    K_{voice}({FIRST_SID + k}, {str(male).lower()}, "{display}", "{desc} (from Kokoro)"),')
    out = OUT / "batch"
    out.mkdir(parents=True, exist_ok=True)
    voicebin.write(out / "voice.bin", ttl, dp)
    (out / "voices.tsv").write_text("\n".join(rows) + "\n")
    (out / "enum.txt").write_text("\n".join(enum) + "\n")
    print(f"packed {len(done)} fitted voices (ids {FIRST_SID}-{FIRST_SID + len(done) - 1}) "
          f"into {out / 'voice.bin'}; see voices.tsv and enum.txt", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
