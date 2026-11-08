"""Diagnostic: is the timbre space smooth enough for gradient fitting to cross from male to female?

Renders the same sentences along a straight path from START to TARGET in style space, plus
every preset on its own and an equal blend of the female presets, and measures pitch and
speaker similarity to TARGET for each. A smooth rise in pitch along the path means gradients
can get there; a sudden jump means they can't.

    .venv/bin/python scripts/diag_voice_space.py [START] [TARGET]
"""

import sys

import numpy as np
import soundfile as sf
import torch

import torch_model
from acoustics import Acoustics, hz
from common import LATENT_CH, OUT, SAMPLE_RATE, load_voice, text_processor
from speaker import SpeakerEncoder
from step1_onnx_baseline import SPEED
from step4_recover_f2 import TRAIN, frames_for
from step4b_blend_then_fit import PRESETS

DEVICE = "cuda"
SENTENCES = TRAIN[:6]


def main(start: str = "M1", target: str = "F2") -> None:
    out = OUT / "diag"
    out.mkdir(parents=True, exist_ok=True)
    m = torch_model.load(DEVICE)
    enc = SpeakerEncoder()
    ac = Acoustics(DEVICE)
    tp = text_processor()
    t = lambda a: torch.from_numpy(np.asarray(a)).to(DEVICE)
    encoded = [tuple(map(t, tp([s], "en"))) for s in SENTENCES]
    styles = {p: t(load_voice(p)[0]) for p in PRESETS}
    _, dp_t = load_voice(target)
    dp_t = t(dp_t)

    def measure(ttl, wav_name=None):
        sims, pitches = [], []
        for i, (ids, tm) in enumerate(encoded):
            n = frames_for(float(m["duration_predictor"](ids, dp_t, tm)[0]) / SPEED)
            g = torch.Generator(device=DEVICE).manual_seed(i)
            noise = torch.randn((1, LATENT_CH, n), generator=g, device=DEVICE)
            wav = m["vocoder"](torch_model.synthesize_latent(m, ids, tm, ttl, noise, torch.ones((1, 1, n), device=DEVICE)))
            if wav_name and i == 0:
                sf.write(out / f"{wav_name}.wav", wav[0].cpu().numpy(), SAMPLE_RATE)
            sims.append(enc(wav)[0])
            pitches.append(ac.pitch_stats(wav)[0].item())
        return torch.stack(sims), float(np.mean(pitches))

    with torch.no_grad():
        target_emb = torch.nn.functional.normalize(measure(styles[target])[0].mean(0), dim=-1)
        sim = lambda e: float((e @ target_emb).mean())

        print(f"path {start} -> {target} (alpha = fraction of the way):")
        for a in np.linspace(0, 1, 11):
            ttl = torch_model.unit_rows((1 - a) * styles[start] + a * styles[target])
            e, p = measure(ttl, f"path_{start}to{target}_{a:.1f}")
            print(f"  alpha {a:.1f}   pitch {hz(p):>4.0f} Hz   sim {sim(e):.3f}   {'#' * int(hz(p) / 5)}")

        print("\neach preset alone:")
        for name, ttl in styles.items():
            e, p = measure(ttl, f"preset_{name}")
            print(f"  {name}   pitch {hz(p):>4.0f} Hz   sim to {target} {sim(e):.3f}")

        females = [p for p in PRESETS if p.startswith("F") and p != target]
        e, p = measure(torch_model.unit_rows(torch.stack([styles[f] for f in females]).mean(0)), "blend_females")
        print(f"\nequal blend of {'+'.join(females)}   pitch {hz(p):>4.0f} Hz   sim to {target} {sim(e):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
