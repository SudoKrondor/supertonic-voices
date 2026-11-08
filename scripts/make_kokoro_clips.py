"""Make Kokoro training clips: essay text -> 5-15 s WAVs + clips.tsv with the exact text of each.

Uses the same Kokoro model and settings Elite Intel ships (kokoro-multi-lang-v1_0 via
sherpa-onnx 1.13.8, the language's espeak code, speed 1.0), so the clips sound like the app. The version
matters: sherpa-onnx 2.x refuses this model without a lexicon file. One difference: the app
speaks clause by clause, but a training clip must be whole sentences with their exact text.

    .venv/bin/python scripts/make_kokoro_clips.py VOICE SID ESSAY.txt [LANG]

Needs the Kokoro model at config.KOKORO_DIR (or KOKORO_MODEL_DIR). SID is the Kokoro speaker id; the
cast's ids are in voices/kokoro_cast.tsv.

Writes data/kokoro/<VOICE>/wavs/NNNN.wav and data/kokoro/<VOICE>/clips.tsv
(file, split, seconds, text); split is "train" or "heldout" (every 10th clip). LANG is a
Supertonic language code (en, es, fr, it, pt, hi; default en) and is saved as lang.txt.
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

from config import CLIPS, KOKORO_DIR, ROOT
WORDS_PER_SECOND = 2.6          # rough Kokoro pace, only used to group sentences
MIN_S, MAX_S = 5.0, 15.0
HELDOUT_EVERY = 10
# Supertonic's language code -> the espeak-ng code Kokoro phonemizes with, as the app maps them.
# Reading a language with English rules mangles it, so a non-English voice needs its own text.
ESPEAK = {"en": "en-us", "es": "es", "fr": "fr", "it": "it", "pt": "pt-br", "hi": "hi"}


def sentences(essay: str) -> list[str]:
    out = []
    for line in essay.splitlines():
        line = line.strip()
        # Headings and titles: short lines with no sentence-ending punctuation.
        if not line or (len(line.split()) < 12 and not re.search(r"[.!?।]$", line)):
            continue
        line = re.sub(r"(?<=\w)\s{2,}(?=\w)", ", ", line)   # stripped dashes left double spaces
        line = re.sub(r"\s+", " ", line)
        # Devanagari ends a sentence with a danda and has no capitals, so the next character only
        # has to not be a lowercase Latin one (which would mean an abbreviation, not a new sentence).
        out += [s for s in re.split(r"(?<=[.!?।])\s+(?![a-z])", line) if s]
    return out


def split_long(sentence: str) -> list[str]:
    """Split a sentence too long for one clip at clause boundaries."""
    if len(sentence.split()) / WORDS_PER_SECOND <= MAX_S:
        return [sentence]
    parts, cur = [], ""
    for piece in re.split(r"(?<=[,;:])\s+", sentence):
        if cur and len((cur + " " + piece).split()) / WORDS_PER_SECOND > MAX_S * 0.8:
            parts.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    return parts + [cur] if cur else parts


def chunks(sents: list[str]) -> list[str]:
    """Group consecutive sentences into clips of roughly MIN_S..MAX_S seconds."""
    out, cur = [], ""
    for s in (p for s in sents for p in split_long(s)):
        nxt = f"{cur} {s}".strip()
        if cur and len(nxt.split()) / WORDS_PER_SECOND > MAX_S * 0.85:
            out.append(cur)
            nxt = s
        cur = nxt
        if len(cur.split()) / WORDS_PER_SECOND >= MIN_S * 1.4:
            out.append(cur)
            cur = ""
    return out + [cur] if cur else out


def main(voice: str, sid: str, essay: str, lang: str = "en") -> None:
    if not (KOKORO_DIR / "model.onnx").exists():
        raise SystemExit(f"Kokoro model not found in {KOKORO_DIR}; set KOKORO_MODEL_DIR")
    out = CLIPS / voice
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    texts = chunks(sentences(Path(essay).read_text(encoding="utf-8")))
    # The fit has to encode the same text in the same language, so record it beside the clips.
    (out / "lang.txt").write_text(lang)

    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(KOKORO_DIR / "model.onnx"), voices=str(KOKORO_DIR / "voices.bin"),
                tokens=str(KOKORO_DIR / "tokens.txt"), data_dir=str(KOKORO_DIR / "espeak-ng-data"),
                lang=ESPEAK[lang]),
            num_threads=16, provider="cpu"),
        max_num_sentences=1))   # as the app sets it; sherpa-onnx's Kokoro speaks the whole text either way

    rows, total, t0 = [], 0.0, time.time()
    for i, text in enumerate(texts, 1):
        audio = tts.generate(text, sid=int(sid), speed=1.0)
        wav = np.asarray(audio.samples, dtype=np.float32)
        secs = len(wav) / audio.sample_rate
        if not 3.0 <= secs <= 18.0:
            print(f"  skip {i}: {secs:.1f}s")
            continue
        name = f"{i:04d}.wav"
        sf.write(out / "wavs" / name, wav, audio.sample_rate)
        split = "heldout" if i % HELDOUT_EVERY == 0 else "train"
        rows.append(f"{name}\t{split}\t{secs:.2f}\t{text}")
        total += secs
        if i % 25 == 0:
            print(f"  {i}/{len(texts)}  {total / 60:.1f} min of audio  ({time.time() - t0:.0f}s)", flush=True)

    (out / "clips.tsv").write_text("file\tsplit\tseconds\ttext\n" + "\n".join(rows) + "\n", encoding="utf-8")
    lens = [float(r.split("\t")[2]) for r in rows]
    print(f"{len(rows)} clips, {total / 60:.1f} min, {min(lens):.1f}-{max(lens):.1f}s each "
          f"(sample rate {audio.sample_rate} Hz) -> {(out / 'clips.tsv').relative_to(ROOT)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
