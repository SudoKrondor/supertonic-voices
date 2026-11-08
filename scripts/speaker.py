"""Differentiable speaker-identity embeddings (SpeechBrain ECAPA-TDNN, VoxCeleb).

Two clips of the same person give embeddings with high cosine similarity, regardless of
what is said or how long the clips are. Gradients flow through it back to the audio.
"""

import torch
import torchaudio.functional as AF
from speechbrain.inference.speaker import EncoderClassifier

from common import ROOT, SAMPLE_RATE

ECAPA_RATE = 16000


class SpeakerEncoder:
    def __init__(self, device: str = "cuda:0"):
        self.clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(ROOT / "pretrained" / "spkrec-ecapa-voxceleb"),
            run_opts={"device": device},
        )
        self.clf.eval()
        for p in self.clf.mods.parameters():
            p.requires_grad_(False)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T] at Supertonic's 44.1 kHz -> unit-length embeddings [B, 192]."""
        wav = AF.resample(wav, SAMPLE_RATE, ECAPA_RATE)
        lens = torch.ones(wav.shape[0], device=wav.device)
        feats = self.clf.mods.compute_features(wav)
        feats = self.clf.mods.mean_var_norm(feats, lens)
        emb = self.clf.mods.embedding_model(feats, lens).squeeze(1)
        return torch.nn.functional.normalize(emb, dim=-1)
