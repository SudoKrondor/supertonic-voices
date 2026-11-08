"""Shared paths and helpers for the go/no-go experiments."""

import json
from pathlib import Path

import numpy as np

from config import CLIPS, MODEL, OUT, ROOT

ONNX = MODEL / "onnx"
VOICES = MODEL / "voice_styles"

CFG = json.loads((ONNX / "tts.json").read_text())
SAMPLE_RATE = CFG["ae"]["sample_rate"]                      # 44100
CHUNK = CFG["ae"]["base_chunk_size"] * CFG["ttl"]["chunk_compress_factor"]  # 3072 samples per latent frame
LATENT_CH = CFG["ttl"]["latent_dim"] * CFG["ttl"]["chunk_compress_factor"]  # 144

# Test sentences: English for the sanity check, plus the languages Kokoro can't do well.
SENTENCES = {
    "en": "Docking request granted. Proceed to landing pad seven, commander.",
    "it": "Richiesta di attracco accettata. Procedi alla piattaforma sette, comandante.",
    "ru": "Запрос на стыковку одобрен. Следуйте к посадочной площадке семь, командир.",
    "uk": "Запит на стикування схвалено. Прямуйте до посадкового майданчика сім, командире.",
}


def load_voice(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (style_ttl [1,50,256], style_dp [1,8,16]) for a preset like 'F1'."""
    d = json.loads((VOICES / f"{name}.json").read_text())
    ttl = np.array(d["style_ttl"]["data"], dtype=np.float32).reshape(d["style_ttl"]["dims"])
    dp = np.array(d["style_dp"]["data"], dtype=np.float32).reshape(d["style_dp"]["dims"])
    return ttl, dp


def text_processor():
    # Reuse the official SDK's text normalisation so our inputs match theirs exactly.
    from supertonic.core import UnicodeProcessor
    return UnicodeProcessor(str(ONNX / "unicode_indexer.json"))


def noise_for(duration_s: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic starting noise and latent mask for one utterance (batch of 1)."""
    wav_len = int(duration_s * SAMPLE_RATE)
    n_frames = (wav_len + CHUNK - 1) // CHUNK
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((1, LATENT_CH, n_frames)).astype(np.float32)
    mask = np.ones((1, 1, n_frames), dtype=np.float32)
    return noise, mask
