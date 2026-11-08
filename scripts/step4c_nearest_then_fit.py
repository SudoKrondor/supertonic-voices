"""Go/no-go step 4c: the realistic case. Target F2, with F2 itself not available as a preset.

A dress rehearsal for Kokoro: F2's clips are rendered and handed to the fitter as plain
(text, audio) recordings. The fitter picks the closest remaining preset by measurement,
then fine-tunes timbre and pacing from it.

    .venv/bin/python scripts/step4c_nearest_then_fit.py [TARGET] [ITERS]
"""

import sys

import soundfile as sf
import torch

from common import OUT, SAMPLE_RATE
from fitting import Fitter, frames_for
from step4_recover_f2 import HELD_OUT, TRAIN, save_voice


def main(target: str = "F2", iters: int = 400) -> None:
    iters = int(iters)
    out = OUT / "step4c"
    (out / "eval").mkdir(parents=True, exist_ok=True)
    fit = Fitter()
    ttl_t, dp_t = fit.preset(target)

    # The "recordings".
    with torch.no_grad():
        clips = []
        for i, text in enumerate(TRAIN):
            wav = fit.speak(text, "en", ttl_t, dp_t, seed=i)
            clips.append((text, "en", wav[0], SAMPLE_RATE))
    tg = fit.target_from_clips(clips)

    print(f"presets ranked by distance to the {target} clips ({target} excluded):")
    print(f"  {'':<4}{'loss':>6} {'sim':>6} {'pitch':>7} {'spectrum':>9}")
    ranking = fit.rank_presets(tg, exclude=(target,))
    for name, s in ranking:
        print(f"  {name:<4}{s['loss']:>6.2f} {s['sim']:>6.3f} {s['hz']:>5.0f}Hz {s['spectrum']:>9.3f}")
    ceiling = fit.measure(ttl_t, tg)
    print(f"  {target} itself, fresh noise (ceiling): loss {ceiling['loss']:.2f}  sim {ceiling['sim']:.3f}  "
          f"{ceiling['hz']:.0f}Hz  spectrum {ceiling['spectrum']:.3f}")

    start = ranking[0][0]
    ttl_s, dp_s = fit.preset(start)
    print(f"\nstarting from {start}; fine-tuning {iters} iterations")
    ttl_f, dp_f = fit.fine_tune(tg, ttl_s, dp_s, iters=iters)

    name = f"fit_{target}_from_{start}"
    save_voice(out / f"{name}.json", ttl_f, dp_f, f"fitted to {target} clips, starting from {start}")
    print(f"\nsaved {(out / f'{name}.json').relative_to(OUT.parent)}")

    print(f"\non training sentences, fresh noise:{'loss':>7} {'sim':>6} {'pitch':>7} {'spectrum':>9}")
    for label, ttl in [(f"{start} (start)", ttl_s), ("fitted", ttl_f), (f"{target} (target)", ttl_t)]:
        e = fit.measure(ttl, tg)
        print(f"  {label:<33}{e['loss']:>6.2f} {e['sim']:>6.3f} {e['hz']:>5.0f}Hz {e['spectrum']:>9.3f}")

    print(f"\nheld-out sentences: sim / pitch (wavs in out/step4c/eval/):")
    with torch.no_grad():
        for i, (key, text) in enumerate(HELD_OUT.items()):
            row = []
            for label, ttl, dp in [("start", ttl_s, dp_s), ("fitted", ttl_f, dp_f), ("target", ttl_t, dp_t)]:
                wav = fit.speak(text, key[:2], ttl, dp, seed=777 + i)
                sf.write(out / "eval" / f"{key}_{label}.wav", wav[0].cpu().numpy(), SAMPLE_RATE)
                f = fit.features(wav)
                row.append(f"{label} {(f['emb'] @ tg.emb).item():.2f}/{2 ** (f['pitch'].item() / 12) * 10:.0f}Hz")
            print(f"  {key:<4} " + "   ".join(row))


if __name__ == "__main__":
    main(*sys.argv[1:])
