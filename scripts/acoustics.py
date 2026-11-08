"""Differentiable measurements of what ears notice first: pitch and spectral colour.

Both are averages over the whole clip, so like the speaker embedding they don't care
what is being said or how long the clip is.
"""

import torch
import torchaudio
import torchaudio.functional as AF
import torchcrepe

from common import SAMPLE_RATE

CREPE_RATE = torchcrepe.SAMPLE_RATE      # 16 kHz
CREPE_WINDOW = torchcrepe.WINDOW_SIZE    # 1024 samples
CREPE_HOP = CREPE_RATE // 50             # 20 ms: plenty for clip averages, half the cost of CREPE's 10 ms
FMIN, FMAX = 50.0, 600.0                 # speaking-voice range
VOICED = 0.5                             # CREPE confidence above which a frame counts as voiced


class Acoustics:
    def __init__(self, device: str = "cuda"):
        torchcrepe.load.model(device, "full")
        self.crepe = torchcrepe.infer.model
        for p in self.crepe.parameters():
            p.requires_grad_(False)
        cents = torchcrepe.CENTS_PER_BIN * torch.arange(torchcrepe.PITCH_BINS, device=device) + 1997.3794084376191
        self.cents = cents
        hz = 10 * 2 ** (cents / 1200)
        self.in_range = (hz >= FMIN) & (hz <= FMAX)
        self.mel = torchaudio.transforms.MelSpectrogram(
            SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=80, f_min=50, f_max=12000).to(device)

    def pitch_frames(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wav [1, T] at 44.1 kHz -> (pitch in semitones re 10 Hz per frame, voiced mask), differentiable in pitch."""
        x = AF.resample(wav, SAMPLE_RATE, CREPE_RATE)
        x = torch.nn.functional.pad(x, (CREPE_WINDOW // 2, CREPE_WINDOW // 2))
        frames = x.unfold(1, CREPE_WINDOW, CREPE_HOP).reshape(-1, CREPE_WINDOW)
        frames = frames - frames.mean(1, keepdim=True)
        frames = frames / frames.std(1, keepdim=True).clamp_min(1e-10)
        probs = self.crepe(frames) * self.in_range
        # torchcrepe's weighted_argmax decoder, vectorised: average of the 9 bins around the peak.
        peak = probs.argmax(1, keepdim=True)
        idx = torch.arange(probs.shape[1], device=probs.device)[None]
        window = probs * ((idx - peak).abs() <= 4)
        cents = (window * self.cents).sum(1) / window.sum(1).clamp_min(1e-8)
        voiced = probs.max(1).values.detach() > VOICED
        return cents / 100, voiced

    def pitch_stats(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and spread of pitch over voiced frames, in semitones."""
        mean, spread, _ = self.pitch_summary(wav)
        return mean, spread

    def pitch_summary(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean and spread of pitch (semitones), plus movement: mean |change| between consecutive voiced
        frames (semitones per 20 ms). Movement is what tracked "lively vs machine-like" by ear in step 4c."""
        st, voiced = self.pitch_frames(wav)
        both = voiced[1:] & voiced[:-1]
        movement = (st[1:] - st[:-1]).abs()[both].mean()
        st = st[voiced]
        return st.mean(), st.std(), movement

    def ltas(self, wav: torch.Tensor) -> torch.Tensor:
        """Long-term average log-mel spectrum over non-silent frames [80]: the voice's 'EQ curve' and level."""
        logmel = (self.mel(wav[0]) + 1e-6).log()           # [80, frames]
        energy = logmel.mean(0)
        speech = energy.detach() > energy.detach().max() - 8  # within ~35 dB of the loudest frame
        return logmel[:, speech].mean(1)


def hz(semitones: float) -> float:
    return 10 * 2 ** (semitones / 12)
