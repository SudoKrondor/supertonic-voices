# supertonic-voices

Make new voices for [Supertonic 3](https://huggingface.co/supertone-oss-archive/supertonic-3) TTS and pack
them into a `voice.bin` that sherpa-onnx's Supertonic engine can load.

Supertonic ships with 10 preset speakers (F1-F5, M1-M5). This project gives it a new speaker that
matches a target voice. You supply 30-50 minutes of the target speaking, with the exact text of each
clip. The fitter starts from the closest preset and tunes the voice style until timbre and pitch match
the recordings. The new voice works in every language Supertonic supports, including languages the
target never spoke.

It was built for [Elite Intel](#elite-intel), to move its old Kokoro voice cast to Supertonic. Nothing in
the core pipeline depends on Elite Intel or Kokoro.

## How it works

```
recordings + text  ──prepare_clips.py──▶     data/kokoro/<VOICE>/       (or make_kokoro_clips.py for Kokoro)
                   ──step5_fit_kokoro.py──▶  out/step5/<VOICE>/<VOICE>.json
                   ──pack_voices.py──▶       voice.bin                  (sids 0-9 presets, 10+ yours)
```

1. **Clips** are 3-18 second WAVs, each with its exact text. Every 10th clip is held out for evaluation.
2. **The fit** (`step5_fit_kokoro.py`) loads Supertonic as PyTorch (converted from the official ONNX
   graphs), so gradients can flow back to the voice style:
   - It starts from the closest preset **of the voice's gender**.
   - It tunes the timbre vector on speaker similarity (ECAPA), pitch and long-term spectrum together.
   - It fits the pacing vector to the clip lengths.
   - Pitch movement is kept at least as lively as the starting preset's.
3. **Packing** puts the 10 upstream presets at ids 0-9 and your voices after them.

You don't need to run `step1` to `step4c`. They are the go/no-go experiments that showed the method
works: ONNX baseline, PyTorch parity, gradient checks, and recovering a known preset from its audio.
The recipe they settled on lives in `fitting.py`.

## Requirements

- Linux and an NVIDIA GPU with CUDA. A fit takes about 50 minutes per voice on an RTX 3090
  (400 iterations).
- Python 3.13
- git-lfs, to download the model

## Setup

```bash
git clone https://github.com/SudoKrondor/supertonic-voices.git
cd supertonic-voices

# Supertonic 3 full-precision model, about 800 MB
git lfs install
git clone https://huggingface.co/supertone-oss-archive/supertonic-3 model

python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

You also need the **upstream 10-speaker `voice.bin`** from sherpa-onnx's
`sherpa-onnx-supertonic-3-tts-int8-2026-05-11` release. Ids 0-9 of every packed file are copied from it.
Save it here:

```
out/voice_bin_backup/voice.bin.original
```

To keep it somewhere else, set `UPSTREAM_VOICE_BIN=/path/to/voice.bin`.

## Make a voice from your own recordings

### 1. Prepare the clips

Put the recordings in a folder, in either of these layouts:

```
recordings/                       recordings/
  line01.wav  line01.txt            metadata.csv      # id|text   (LJSpeech style)
  line02.wav  line02.txt            wavs/0001.wav ...
```

Then run:

```bash
.venv/bin/python scripts/prepare_clips.py NARRATOR recordings/ --gender f --lang en
```

`NARRATOR` is any name you choose; folders and files are named after it.

What makes a good fit:
- **The text matches the audio word for word.** The fit reads the text to know what is being said.
- **One speaker, clean audio.** No music, effects or background noise.
- **Single utterances of 3-18 s each, 30-50 minutes in total.** Shorter and longer clips are skipped.
- **The right `--gender`.** The fit starts only from presets of that gender. If it starts from the
  other gender, it keeps that gender and the result sounds wrong.
- **The language the recordings are actually in.** Pass it as a Supertonic code with `--lang`
  (en, es, fr, de, it, pt, ko, ja, …). The text is read with that language's rules.

The clips go to `data/kokoro/NARRATOR/`. The folder is called `kokoro` for historical reasons, but it
holds any voice's clips.

### 2. Fit

```bash
.venv/bin/python scripts/step5_fit_kokoro.py NARRATOR          # [ITERS] [OUT_DIR] [--gender m|f]
```

This writes `out/step5/NARRATOR/NARRATOR.json` and `eval/` WAVs to listen to:
- For each held-out clip, three files: `NNNN_target` (your recording), `NNNN_start` (the nearest
  preset) and `NNNN_fitted`.
- `en_*`, `it_*`, `ru_*` and `uk_*`: test sentences in other languages, with the start and fitted voice.

**Let your ears decide.** The similarity score printed at the end is a rough guide only. It has rated
good fits low and missed bad ones (wrong gender, mangled pronunciation).

### 3. Pack

```bash
.venv/bin/python scripts/pack_voices.py out/my_voice.bin NARRATOR [MORE ...]
```

`out/my_voice.bin` has the 10 presets at sids 0-9 and your voices from sid 10, in the order you named
them. Pass `--no-presets` to start your voices at sid 0. Besides voice names, you can pass any voice
JSON in Supertonic's format, such as `model/voice_styles/F1.json`.

To use the file, replace `voice.bin` in a sherpa-onnx Supertonic 3 model folder with it, and choose the
voice by its speaker id (`sid`).

### 4. Optional: hear it in the int8 engine

The fit renders with the full-precision model. `check_int8.py` renders with sherpa-onnx's int8 release
(10 steps, speed 0.85, the settings Elite Intel uses) and scores each render against your clips:

```bash
SUPERTONIC_INT8_DIR=/path/to/sherpa-onnx-supertonic-3-tts-int8-2026-05-11 \
    .venv/bin/python scripts/check_int8.py NARRATOR
```

## Use it responsibly

This tool copies a voice. Only fit voices you have the right to copy: your own, a synthetic voice whose
license allows it, or a person who has agreed to it. Supertonic 3 is released under the BigScience
**Open RAIL-M** license (`model/LICENSE`). Its use restrictions, including those against impersonation
and deception, also apply to the voices you make with it.

## Elite Intel

Elite Intel's Supertonic `voice.bin` was built here. Sids 0-9 are the presets, and sids 10 and up are its
pre-V1.2 Kokoro cast, fitted voice by voice. **Contributors already have that file in the Elite Intel
repo.** You only need this section to add or refit voices.

### Kokoro clips

The Kokoro cast (name, Kokoro speaker id, gender) is saved in `voices/kokoro_cast.tsv`, because Elite Intel
no longer contains Kokoro (it was removed in commit `da9503a2c`). To make Kokoro clips, you need the
`kokoro-multi-lang-v1_0` model from before that commit and `sherpa-onnx==1.13.8`. Point
`KOKORO_MODEL_DIR` at the model.

```bash
.venv/bin/python scripts/make_kokoro_clips.py BELLA 2 essay.txt [LANG]   # VOICE  KOKORO_SID  TEXT
.venv/bin/python scripts/step5_fit_kokoro.py BELLA                       # gender comes from the cast
```

The commands for the whole cast:

```bash
.venv/bin/python scripts/batch_fit.py essay.txt [VOICE ...]   # resumable; bash scripts/resume_batch.sh essay.txt runs it detached
.venv/bin/python scripts/fetch_essays.py                      # public-domain es/fr/it/pt/hi text
.venv/bin/python scripts/fit_nonenglish.py [VOICE ...]        # refit ES_*, FR_*, HI_*, IT_*, PT_* in their own language
```

Fit a non-English voice from clips in its own language. Kokoro reads an English text with English
rules, and the fit copies the mispronunciation. `ES_ALEX` is the exception: it was kept as fitted from
English.

### Release

```bash
./package.sh            # = .venv/bin/python scripts/pack_release.py
```

This reads the `SupertonicVoices` enum in the Elite Intel checkout and writes `out/release/voice.bin`,
id for id. **The enum is the source of truth.** Speaker N must be the voice the enum gives sid N, so
never reorder or reuse sids. An id the enum skips gets a copy of F1. The script never writes into the
app. To ship the file, copy it over `voice.bin` in
`distribution/tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/`.

To add a voice, give it the next free sid in the enum, then make its clips and fit it. Listen to the
eval WAVs, then run `./package.sh` and copy the file into the app.

## Configuration

`scripts/config.py` holds every path. Each one can be overridden with an environment variable:

| Variable | Default | Needed for |
|---|---|---|
| `SUPERTONIC_MODEL_DIR` | `model/` | everything |
| `UPSTREAM_VOICE_BIN` | `out/voice_bin_backup/voice.bin.original` | packing |
| `SUPERTONIC_INT8_DIR` | `$ELITE_INTEL_DIR/distribution/tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11` | `check_int8.py` |
| `KOKORO_MODEL_DIR` | `$ELITE_INTEL_DIR/distribution/tts/kokoro-multi-lang-v1_0` | Kokoro clips |
| `ELITE_INTEL_DIR` | `~/projects/EliteIntel` | the paths above, `pack_release.py` |
| `SUPERTONIC_ENUM` | `$ELITE_INTEL_DIR/app/.../SupertonicVoices.java` | `pack_release.py` |

## Layout

```
scripts/
  config.py            every path, overridable
  common.py            model config, test sentences
  cast.py              voice gender/language, the Kokoro cast
  torch_model.py       Supertonic ONNX graphs -> PyTorch (onnx2torch)
  speaker.py           ECAPA speaker embedding (SpeechBrain)
  acoustics.py         differentiable pitch and spectrum
  fitting.py           the fitting recipe (Fitter)
  voicebin.py          read/write voice.bin
  prepare_clips.py     your recordings -> clips
  step5_fit_kokoro.py  fit one voice
  pack_voices.py       pack voices into a voice.bin
  check_int8.py        render in the int8 engine
  make_kokoro_clips.py, batch_fit.py, fit_nonenglish.py, refit_same_gender.py,
  fetch_essays.py, pack_release.py, *.sh                     Elite Intel / Kokoro
  step1-4c_*.py, diag_voice_space.py                         go/no-go experiments
voices/kokoro_cast.tsv the Kokoro cast Elite Intel's voices were fitted to
model/                 Supertonic 3 (cloned, not committed)
data/, out/            clips, fits, eval audio, packed files (generated, not committed)
```

## voice.bin format

The file is little-endian. It starts with six int64 header values, `[N, 50, 256, N, 8, 16]`, followed by
the N timbre styles (float32, N×50×256) and then the N pacing styles (float32, N×8×16). Speaker id *i* is
the *i*-th entry. See `voicebin.py`.
