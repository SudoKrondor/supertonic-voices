"""Go/no-go step 1: run the official full-precision ONNX graphs with a preset voice.

Writes out/step1/<voice>_<lang>.wav to listen to, and out/step1/reference.npz with every
intermediate tensor, so step 2 can check the PyTorch conversion against it exactly.

    .venv/bin/python scripts/step1_onnx_baseline.py [VOICE]
"""

import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

from common import ONNX, OUT, SAMPLE_RATE, SENTENCES, load_voice, noise_for, text_processor

TOTAL_STEPS = 5   # SDK default
SPEED = 1.05      # SDK default


def main(voice: str = "F1") -> None:
    out = OUT / "step1"
    out.mkdir(parents=True, exist_ok=True)

    sess = {n: ort.InferenceSession(str(ONNX / f"{n}.onnx"), providers=["CPUExecutionProvider"])
            for n in ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]}
    tp = text_processor()
    ttl, dp = load_voice(voice)

    ref = {"style_ttl": ttl, "style_dp": dp}
    for i, (lang, text) in enumerate(SENTENCES.items()):
        ids, tmask = tp([text], lang)
        dur, = sess["duration_predictor"].run(None, {"text_ids": ids, "style_dp": dp, "text_mask": tmask})
        dur = dur / SPEED
        emb, = sess["text_encoder"].run(None, {"text_ids": ids, "style_ttl": ttl, "text_mask": tmask})

        x, lmask = noise_for(float(dur[0]), seed=1000 + i)
        noise = x.copy()
        total = np.array([TOTAL_STEPS], dtype=np.float32)
        for step in range(TOTAL_STEPS):
            x, = sess["vector_estimator"].run(None, {
                "noisy_latent": x, "text_emb": emb, "style_ttl": ttl,
                "text_mask": tmask, "latent_mask": lmask,
                "current_step": np.array([step], dtype=np.float32), "total_step": total,
            })
        wav, = sess["vocoder"].run(None, {"latent": x})

        path = out / f"{voice}_{lang}.wav"
        sf.write(path, wav[0], SAMPLE_RATE)
        print(f"{lang}: predicted {float(dur[0]):.2f}s, {x.shape[2]} latent frames -> {path.relative_to(OUT.parent)}")

        ref.update({f"{lang}/text_ids": ids, f"{lang}/text_mask": tmask, f"{lang}/duration": dur,
                    f"{lang}/text_emb": emb, f"{lang}/noise": noise, f"{lang}/latent_mask": lmask,
                    f"{lang}/latent": x, f"{lang}/wav": wav})

    np.savez(out / "reference.npz", voice=voice, **ref)
    print(f"saved {out.relative_to(OUT.parent)}/reference.npz")


if __name__ == "__main__":
    main(*sys.argv[1:])
