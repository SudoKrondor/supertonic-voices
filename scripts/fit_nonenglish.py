"""Refit the non-English voices from clips spoken in their own language.

The first pass fitted every voice from Kokoro reading the English essay, which mangles a Spanish
or Hindi voice: Kokoro applies English pronunciation rules, and the fit copies the result
(ES_DORA was unusable). Each voice here is redone from data/essays/<lang>.txt, read by Kokoro
with that language's phonemizer and encoded for Supertonic in the same language.

    .venv/bin/python scripts/fit_nonenglish.py [VOICE ...]      # default: all eleven

The old English-fitted clips and voice are moved aside to data/kokoro/<VOICE>_en and
out/step5/<VOICE>_en. Resumable: a voice whose clips are already in the right language is not
regenerated, and one already refitted is skipped.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

from batch_fit import ITERS, PY, SCRIPTS, cast, fitted
from common import OUT, ROOT
from step5_fit_kokoro import clip_lang

ESSAYS = ROOT / "data" / "essays"
LANG_BY_PREFIX = {"ES_": "es", "FR_": "fr", "HI_": "hi", "IT_": "it", "PT_": "pt"}
# Kept as fitted from the English essay: the commander listened to every sample and was happy with
# it, so there is nothing for a Spanish refit to improve. Naming a voice on the command line still
# refits it. (ES_DORA, fitted the same way, was unusable - the damage is per voice, not per language.)
KEEP_ENGLISH_FIT = {"ES_ALEX"}


def lang_of(voice: str) -> str | None:
    return next((l for p, l in LANG_BY_PREFIX.items() if voice.startswith(p)), None)


def move_aside(voice: str) -> None:
    """Keep the English-fitted attempt for comparison instead of deleting it."""
    for live, kept in [(ROOT / "data" / "kokoro" / voice, ROOT / "data" / "kokoro" / f"{voice}_en"),
                       (OUT / "step5" / voice, OUT / "step5" / f"{voice}_en")]:
        if live.exists():
            shutil.rmtree(kept, ignore_errors=True)
            shutil.move(live, kept)


def main(*only: str) -> None:
    by_name = {v[0]: v for v in cast()}
    voices = list(only) or [v for v in by_name if lang_of(v) and v not in KEEP_ENGLISH_FIT]
    for voice in voices:
        lang, sid = lang_of(voice), by_name[voice][1]
        if not lang:
            print(f"[{voice}] not a non-English voice, skipping", flush=True)
            continue
        if fitted(voice).exists() and clip_lang(voice) == lang:
            print(f"[{voice}] already refitted in {lang}, skipping", flush=True)
            continue
        t0 = time.time()
        if clip_lang(voice) != lang:
            move_aside(voice)
            print(f"[{voice}] making {lang} clips", flush=True)
            log = (OUT / "kokoro" / f"make_{voice}.log").open("w")
            code = subprocess.run([PY, "make_kokoro_clips.py", voice, str(sid),
                                   str(ESSAYS / f"{lang}.txt"), lang],
                                  cwd=SCRIPTS, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                print(f"[{voice}] clip generation failed, see out/kokoro/make_{voice}.log", flush=True)
                continue
        out = OUT / "step5" / voice
        out.mkdir(parents=True, exist_ok=True)
        with (out / "run.log").open("w") as log:
            code = subprocess.run([PY, "step5_fit_kokoro.py", voice, str(ITERS)], cwd=SCRIPTS,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        if code or not fitted(voice).exists():
            print(f"[{voice}] fit failed (exit {code}), see {out / 'run.log'}", flush=True)
            continue
        print(f"[{voice}] done in {(time.time() - t0) / 60:.0f} min ({lang})", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
