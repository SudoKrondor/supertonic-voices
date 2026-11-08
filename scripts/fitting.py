"""Fit a Supertonic voice style to a set of target clips (text + audio).

The recipe that came out of the go/no-go steps:
  1. Measure every preset against the target clips and start from the closest one.
     (Gradient search over preset blends was too noisy to find it; step 4b.)
  2. Fine-tune the timbre vector on speaker similarity + pitch + spectrum together.
     Speaker similarity alone gets fooled; step 4.
  3. Fit the pacing vector to the clip lengths.

Targets are plain clips, so this works the same for Supertonic-rendered test clips and for
Kokoro recordings.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio.functional as AF

import torch_model
from acoustics import Acoustics, hz
from common import CHUNK, LATENT_CH, SAMPLE_RATE, load_voice, text_processor
from speaker import SpeakerEncoder
from step1_onnx_baseline import SPEED

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

PRESETS = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
TERMS = ["speaker", "pitch", "spectrum"]
LIVELY_SCALE = 0.05   # semitones per 20 ms; roughly the flat-F4 to lively-F2 gap heard in step 4c


def frames_for(seconds: float) -> int:
    return (int(seconds * SAMPLE_RATE) + CHUNK - 1) // CHUNK


@dataclass
class Target:
    """What we match: per-clip text and length, plus the voice summarised over all clips."""
    encoded: list          # (text_ids, text_mask) per clip
    seconds: torch.Tensor  # clip lengths
    frames: list[int]
    emb: torch.Tensor      # speaker centroid
    pitch: torch.Tensor    # mean pitch, semitones
    spread: torch.Tensor   # pitch spread, semitones
    movement: torch.Tensor # pitch movement, semitones per 20 ms
    ltas: torch.Tensor     # long-term average spectrum


class Fitter:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.m = torch_model.load(device)
        self.enc = SpeakerEncoder()
        self.ac = Acoustics(device)
        self.tp = text_processor()
        self.scale = None
        # None: copy the target's pitch range too. A number: don't copy how lively the target is;
        # instead keep pitch movement at least this high (for flat-sounding targets like Kokoro).
        self.liveliness = None

    def t(self, a) -> torch.Tensor:
        return torch.from_numpy(np.asarray(a)).to(self.device)

    def encode(self, text: str, lang: str = "en"):
        ids, tmask = self.tp([text], lang)
        return self.t(ids), self.t(tmask)

    def preset(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(map(self.t, load_voice(name)))

    def duration(self, encoded, dp) -> torch.Tensor:
        return torch.cat([self.m["duration_predictor"](ids, dp, tm) for ids, tm in encoded]) / SPEED

    def render(self, ids, tmask, ttl, n_frames: int, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn((1, LATENT_CH, n_frames), generator=g, device=self.device)
        lmask = torch.ones((1, 1, n_frames), device=self.device)
        return self.m["vocoder"](torch_model.synthesize_latent(self.m, ids, tmask, ttl, noise, lmask))

    def speak(self, text: str, lang: str, ttl, dp, seed: int = 0) -> torch.Tensor:
        """Ordinary synthesis with a style, duration from its own pacing vector."""
        ids, tm = self.encode(text, lang)
        return self.render(ids, tm, ttl, frames_for(float(self.duration([(ids, tm)], dp)[0])), seed)

    def features(self, wav: torch.Tensor) -> dict:
        mean, spread, movement = self.ac.pitch_summary(wav)
        return {"emb": self.enc(wav)[0], "pitch": mean, "spread": spread, "movement": movement,
                "ltas": self.ac.ltas(wav)}

    def target_from_clips(self, clips: list[tuple[str, str, torch.Tensor, int]]) -> Target:
        """clips: (text, lang, wav [T], sample_rate)."""
        encoded, seconds, feats = [], [], []
        with torch.no_grad():
            for text, lang, wav, sr in clips:
                wav = wav.to(self.device, torch.float32)[None]
                if sr != SAMPLE_RATE:
                    wav = AF.resample(wav, sr, SAMPLE_RATE)
                encoded.append(self.encode(text, lang))
                seconds.append(wav.shape[1] / SAMPLE_RATE)
                feats.append(self.features(wav))
        return Target(
            encoded=encoded, seconds=self.t(np.array(seconds, dtype=np.float32)),
            frames=[frames_for(s) for s in seconds],
            emb=torch.nn.functional.normalize(torch.stack([f["emb"] for f in feats]).mean(0), dim=-1),
            pitch=torch.stack([f["pitch"] for f in feats]).mean(),
            spread=torch.stack([f["spread"] for f in feats]).mean(),
            movement=torch.stack([f["movement"] for f in feats]).mean(),
            ltas=torch.stack([f["ltas"] for f in feats]).mean(0),
        )

    def terms(self, f: dict, tg: Target) -> dict:
        """Raw distance from the target voice per aspect (speaker: 1-cos; pitch: semitones; spectrum: log-mel)."""
        pitch = (f["pitch"] - tg.pitch).abs()
        if self.liveliness is None:
            pitch = pitch + 0.5 * (f["spread"] - tg.spread).abs()
        tr = {"speaker": 1 - f["emb"] @ tg.emb, "pitch": pitch, "spectrum": (f["ltas"] - tg.ltas).abs().mean()}
        if self.liveliness is not None:
            tr["flatness"] = torch.relu(self.liveliness - f["movement"])
        return tr

    def loss(self, f: dict, tg: Target) -> torch.Tensor:
        tr = self.terms(f, tg)
        loss = sum(tr[k] / self.scale[k] for k in TERMS)
        if "flatness" in tr:
            loss = loss + tr["flatness"] / LIVELY_SCALE
        return loss

    def measure(self, ttl, tg: Target, seed0: int = 500, n: int | None = None) -> dict:
        """Average distance of a timbre style from the target, rendered on the target's own sentences."""
        idx = range(len(tg.encoded) if n is None else min(n, len(tg.encoded)))
        with torch.no_grad():
            fs = [self.features(self.render(*tg.encoded[i], ttl, tg.frames[i], seed=seed0 + i)) for i in idx]
            per = [self.terms(f, tg) for f in fs]
            res = {k: float(torch.stack([p[k] for p in per]).mean()) for k in TERMS}
            res["sim"] = 1 - res["speaker"]
            res["hz"] = hz(float(torch.stack([f["pitch"] for f in fs]).mean()))
            res["movement"] = float(torch.stack([f["movement"] for f in fs]).mean())
            if self.scale:
                res["loss"] = sum(res[k] / self.scale[k] for k in TERMS)
        return res

    def rank_presets(self, tg: Target, exclude: tuple[str, ...] = (), n: int = 8) -> list[tuple[str, dict]]:
        """Score each preset against the target. Also sets the loss scales: each term is divided by
        its average over the presets, so a 'typical difference between voices' counts as 1."""
        scores = {p: self.measure(self.preset(p)[0], tg, n=n) for p in PRESETS if p not in exclude}
        self.scale = {k: float(np.mean([s[k] for s in scores.values()])) for k in TERMS}
        for s in scores.values():
            s["loss"] = sum(s[k] / self.scale[k] for k in TERMS)
        return sorted(scores.items(), key=lambda kv: kv[1]["loss"])

    def fine_tune(self, tg: Target, ttl0, dp0, iters: int = 400, lr_ttl: float = 5e-4,
                  lr_dp: float = 1e-2, batch: int = 4, log_every: int = 10):
        raw_ttl = ttl0.clone().requires_grad_(True)
        raw_dp = dp0.clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [raw_ttl], "lr": lr_ttl}, {"params": [raw_dp], "lr": lr_dp}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        rng = np.random.default_rng(0)
        print(f"{'iter':>5} {'loss':>6} {'sim':>6} {'pitch':>7} {'spectrum':>9} {'movement':>9} {'dur err':>8}  s/iter")
        t0 = time.time()
        for it in range(1, iters + 1):
            opt.zero_grad()
            log = []
            for k in rng.choice(len(tg.encoded), min(batch, len(tg.encoded)), replace=False):
                # Fresh noise each time so the voice works for any noise, not just one draw.
                wav = self.render(*tg.encoded[k], torch_model.unit_rows(raw_ttl), tg.frames[k],
                                  seed=10_000 + it * batch + len(log))
                f = self.features(wav)
                loss = self.loss(f, tg)
                (loss / batch).backward()
                log.append((loss.item(), (f["emb"] @ tg.emb).item(), f["pitch"].item(),
                            self.terms(f, tg)["spectrum"].item(), f["movement"].item()))
            dur = self.duration(tg.encoded, torch_model.unit_rows(raw_dp))
            ((dur.log() - tg.seconds.log()) ** 2).mean().backward()
            opt.step()
            sched.step()
            if it % log_every == 0 or it == 1:
                l = np.mean(log, 0)
                rel_dur = ((dur - tg.seconds).abs() / tg.seconds).mean().item()
                print(f"{it:>5} {l[0]:>6.3f} {l[1]:>6.3f} {hz(l[2]):>5.0f}Hz {l[3]:>9.3f} {l[4]:>9.3f} {rel_dur:>7.1%}  "
                      f"{(time.time() - t0) / it:.2f}", flush=True)
        return torch_model.unit_rows(raw_ttl).detach(), torch_model.unit_rows(raw_dp).detach()
