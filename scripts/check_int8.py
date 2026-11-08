"""Does a fitted voice survive Elite Intel's setup? Render it the way the app does.

The app runs the int8 Supertonic model through sherpa-onnx 1.13.8 with 10 flow steps and
speed 0.85; the fitting used the full-precision model at 5 steps and speed 1.05. This builds a
test model folder (links to the app's int8 files + a voice.bin with the fitted voices appended
after F1-M5), renders with the app's settings, and scores each render against the Kokoro voice.
The app's own files are only read.

    .venv/bin/python scripts/check_int8.py VOICE [VOICE ...]
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf
import torch
import torchaudio.functional as AF

import voicebin
from common import OUT, SAMPLE_RATE
from speaker import SpeakerEncoder
from step4_recover_f2 import HELD_OUT
from step5_fit_kokoro import N_EVAL_CLIPS, load_clips

APP_MODEL = Path.home() / "projects/EliteIntel/distribution/tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
APP_STEPS = 10      # SupertonicTTS.NUM_STEPS
APP_SPEED = 0.85    # SupertonicTTS.SPEED_MULTIPLIER at the commander's default speed 1.0
FIRST_NEW_SID = 10


def build_model_dir(out: Path, voices: list[str]) -> Path:
    model = out / "model"
    model.mkdir(parents=True, exist_ok=True)
    for f in APP_MODEL.iterdir():
        link = model / f.name
        if f.name != "voice.bin" and not link.exists():
            link.symlink_to(f)
    ttl, dp = voicebin.read(APP_MODEL / "voice.bin")
    for v in voices:
        t, d = load_voice_json(OUT / "step5" / v / f"{v}.json")
        ttl, dp = np.concatenate([ttl, t]), np.concatenate([dp, d])
    voicebin.write(model / "voice.bin", ttl, dp)
    return model


def load_voice_json(path: Path):
    import json
    d = json.loads(path.read_text())
    return (np.array(d["style_ttl"]["data"], dtype=np.float32).reshape(d["style_ttl"]["dims"]),
            np.array(d["style_dp"]["data"], dtype=np.float32).reshape(d["style_dp"]["dims"]))


def app_tts(model: Path) -> sherpa_onnx.OfflineTts:
    m = lambda n: str(model / n)
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor=m("duration_predictor.int8.onnx"), text_encoder=m("text_encoder.int8.onnx"),
                vector_estimator=m("vector_estimator.int8.onnx"), vocoder=m("vocoder.int8.onnx"),
                tts_json=m("tts.json"), unicode_indexer=m("unicode_indexer.bin"), voice_style=m("voice.bin")),
            num_threads=8, provider="cpu"),
        max_num_sentences=1))


def speak(tts, text: str, sid: int, lang: str) -> tuple[np.ndarray, int]:
    cfg = sherpa_onnx.GenerationConfig()
    cfg.sid, cfg.speed, cfg.num_steps, cfg.extra = sid, APP_SPEED, APP_STEPS, {"lang": lang}
    audio = tts.generate(text, cfg)
    return np.asarray(audio.samples, dtype=np.float32), audio.sample_rate


def main(*voices: str) -> None:
    out = OUT / "int8_check"
    (out / "eval").mkdir(parents=True, exist_ok=True)
    tts = app_tts(build_model_dir(out, list(voices)))
    enc = SpeakerEncoder()

    def emb(wav: np.ndarray, sr: int) -> torch.Tensor:
        w = torch.from_numpy(wav).cuda()[None]
        return enc(AF.resample(w, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else w)[0]

    for k, voice in enumerate(voices):
        sid = FIRST_NEW_SID + k
        root, clips = load_clips(voice)
        with torch.no_grad():
            centroid = torch.nn.functional.normalize(
                torch.stack([emb(w.numpy(), sr) for _, _, w, sr in clips["train"]]).mean(0), dim=-1)
            print(f"{voice} as app speaker id {sid}; similarity to Kokoro {voice}:")
            print(f"  {'clip':<6}{'F1 in app':>10}{'fitted, fp32 5-step':>21}{'fitted in app':>15}")
            scores = []
            for i, (name, text, _, _) in enumerate(clips["heldout"][:N_EVAL_CLIPS]):
                stem = Path(name).stem
                shutil.copy(root / "wavs" / name, out / "eval" / f"{voice}_{stem}_kokoro.wav")
                row = []
                for label, s in [("F1_app", 0), (f"{voice}_app", sid)]:
                    wav, sr = speak(tts, text, s, "en")
                    sf.write(out / "eval" / f"{voice}_{stem}_{label}.wav", wav, sr)
                    row.append(float(emb(wav, sr) @ centroid))
                fp32, sr = sf.read(OUT / "step5" / voice / "eval" / f"{stem}_fitted.wav", dtype="float32")
                row.insert(1, float(emb(fp32, sr) @ centroid))
                scores.append(row)
                print(f"  {stem:<6}{row[0]:>10.2f}{row[1]:>21.2f}{row[2]:>15.2f}")
            m = np.mean(scores, 0)
            print(f"  {'mean':<6}{m[0]:>10.2f}{m[1]:>21.2f}{m[2]:>15.2f}")
            for key, text in HELD_OUT.items():
                for label, s in [("F1_app", 0), (f"{voice}_app", sid)]:
                    wav, sr = speak(tts, text, s, key[:2])
                    sf.write(out / "eval" / f"{voice}_{key}_{label}.wav", wav, sr)
    print(f"wavs in {out / 'eval'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
