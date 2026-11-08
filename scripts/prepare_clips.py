"""Turn your own recordings into training clips for step5_fit_kokoro.py.

SRC is a folder in one of two layouts:
  * pairs:     any.wav + any.txt with the same name, the .txt holding exactly what is said
  * LJSpeech:  metadata.csv with lines "id|text" (or "id|raw text|normalised text", the last is used),
               and the audio in SRC/wavs/<id>.wav or SRC/<id>.wav

    .venv/bin/python scripts/prepare_clips.py VOICE SRC --gender m|f [--lang en] [--force]

Audio can be WAV, FLAC or OGG at any sample rate; it is mixed down to mono. Clips must be single
utterances of 3-18 seconds (longer or shorter ones are skipped), one speaker, no music or effects. Aim for
30-50 minutes in total; the Kokoro voices were fitted on about 40. The text must match the audio word for
word, in the language given with --lang (a Supertonic code such as en, es, fr, de, it, pt, ko, ja).

Writes data/kokoro/<VOICE>/wavs/NNNN.wav, clips.tsv (file, split, seconds, text; every 10th clip is held
out), lang.txt and gender.txt.

Only use recordings of a voice you have the right to copy: your own, a synthetic voice whose license
allows it, or a person who has agreed to it.
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from cast import write_gender
from config import CLIPS

MIN_S, MAX_S = 3.0, 18.0       # what make_kokoro_clips.py keeps
HELDOUT_EVERY = 10
AUDIO = (".wav", ".flac", ".ogg")


def find_audio(base: Path) -> Path | None:
    return next((p for ext in AUDIO if (p := base.with_suffix(ext)).exists()), None)


def pairs(src: Path) -> list[tuple[Path, str]]:
    """(audio, text) in a stable order."""
    meta = src / "metadata.csv"
    out = []
    if meta.exists():
        for line in meta.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split("|")
            if len(fields) < 2:
                raise SystemExit(f"{meta}: expected id|text, got {line!r}")
            audio = find_audio(src / "wavs" / fields[0]) or find_audio(src / fields[0])
            if audio is None:
                print(f"  skip {fields[0]}: no audio file")
                continue
            out.append((audio, fields[-1]))
    else:
        for txt in sorted(src.glob("*.txt")):
            audio = find_audio(txt)
            if audio is None:
                print(f"  skip {txt.name}: no audio file with the same name")
                continue
            out.append((audio, txt.read_text(encoding="utf-8")))
    if not out:
        raise SystemExit(f"no clips found in {src}: need metadata.csv, or .wav/.flac/.ogg files with matching .txt")
    return [(a, re.sub(r"\s+", " ", t).strip()) for a, t in out]


def main() -> None:
    ap = argparse.ArgumentParser(description="Make training clips from your own recordings.")
    ap.add_argument("voice", help="a name for the voice, e.g. NARRATOR (used for folder and file names)")
    ap.add_argument("src", type=Path, help="folder with the recordings and their text")
    ap.add_argument("--gender", required=True, choices=["m", "f"], help="picks which presets the fit starts from")
    ap.add_argument("--lang", default="en", help="Supertonic language code of the recordings (default en)")
    ap.add_argument("--force", action="store_true", help="replace clips already made for this voice")
    a = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.voice):
        raise SystemExit("VOICE may only use letters, digits, _ and -")
    out = CLIPS / a.voice
    if (out / "clips.tsv").exists() and not a.force:
        raise SystemExit(f"{out} already has clips; pass --force to replace them")
    shutil.rmtree(out / "wavs", ignore_errors=True)
    (out / "wavs").mkdir(parents=True, exist_ok=True)

    rows, total, n = [], 0.0, 0
    for audio, text in pairs(a.src):
        if not text:
            print(f"  skip {audio.name}: empty text")
            continue
        wav, sr = sf.read(audio, dtype="float32", always_2d=True)
        wav = wav.mean(1)
        secs = len(wav) / sr
        if not MIN_S <= secs <= MAX_S:
            print(f"  skip {audio.name}: {secs:.1f}s (keep clips between {MIN_S:.0f} and {MAX_S:.0f}s)")
            continue
        n += 1
        name = f"{n:04d}.wav"
        sf.write(out / "wavs" / name, np.clip(wav, -1, 1), sr)
        split = "heldout" if n % HELDOUT_EVERY == 0 else "train"
        rows.append(f"{name}\t{split}\t{secs:.2f}\t{text}")
        total += secs

    if not rows:
        raise SystemExit("no usable clips")
    (out / "clips.tsv").write_text("file\tsplit\tseconds\ttext\n" + "\n".join(rows) + "\n", encoding="utf-8")
    (out / "lang.txt").write_text(a.lang)
    write_gender(a.voice, a.gender)
    held = sum(r.split("\t")[1] == "heldout" for r in rows)
    print(f"{len(rows)} clips ({held} held out), {total / 60:.1f} min -> {out / 'clips.tsv'}")
    if held == 0:
        print("  WARNING: no held-out clips (need at least 10); the fit will have nothing to evaluate on")
    if total < 20 * 60:
        print("  WARNING: under 20 minutes of audio; the fit may not capture the voice well")


if __name__ == "__main__":
    main()
