"""Go/no-go step 2: check the PyTorch conversion matches the ONNX output from step 1.

Two checks per sentence:
  * per graph: feed each PyTorch graph the exact inputs ONNX saw, compare outputs
  * end to end: run the whole chain in PyTorch from the same noise, compare the audio

Writes out/step2/<voice>_<lang>_torch.wav to listen to next to the step-1 files.

    .venv/bin/python scripts/step2_torch_parity.py
"""

import numpy as np
import soundfile as sf
import torch

import torch_model
from common import OUT, SAMPLE_RATE, SENTENCES
from step1_onnx_baseline import SPEED, TOTAL_STEPS

# TF32 matmuls on Ampere trade precision for speed; we want a strict float32 comparison.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = "cuda"


def report(label: str, ours: torch.Tensor, ref: np.ndarray) -> float:
    ours = ours.detach().cpu().numpy()
    if ours.shape != ref.shape:
        print(f"  {label:<22} SHAPE MISMATCH {ours.shape} vs {ref.shape}")
        return float("inf")
    err = np.abs(ours - ref).max()
    rel = err / (np.abs(ref).max() + 1e-12)
    print(f"  {label:<22} max abs diff {err:.2e}  (relative {rel:.2e})")
    return float(rel)


def synthesize(m, ids, tmask, ttl, dp, noise, lmask):
    """The full Supertonic pipeline in PyTorch. Same steps as the SDK's Supertonic.__call__."""
    dur = m["duration_predictor"](ids, dp, tmask) / SPEED
    emb = m["text_encoder"](ids, ttl, tmask)
    x = noise
    total = torch.tensor([TOTAL_STEPS], dtype=torch.float32, device=DEVICE)
    for step in range(TOTAL_STEPS):
        cur = torch.tensor([step], dtype=torch.float32, device=DEVICE)
        x = m["vector_estimator"](x, emb, ttl, lmask, tmask, cur, total)
    wav = m["vocoder"](x)
    return dur, emb, x, wav


def main() -> None:
    ref = np.load(OUT / "step1" / "reference.npz")
    voice = str(ref["voice"])
    out = OUT / "step2"
    out.mkdir(parents=True, exist_ok=True)
    m = torch_model.load(DEVICE)
    t = lambda a: torch.from_numpy(np.asarray(a)).to(DEVICE)
    ttl, dp = t(ref["style_ttl"]), t(ref["style_dp"])

    worst = 0.0
    with torch.no_grad():
        for lang in SENTENCES:
            r = lambda k: ref[f"{lang}/{k}"]
            ids, tmask, lmask = t(r("text_ids")), t(r("text_mask")), t(r("latent_mask"))
            print(f"{lang}:")

            # Per graph, each fed the ONNX inputs.
            worst = max(worst, report("duration_predictor", m["duration_predictor"](ids, dp, tmask) / SPEED, r("duration")))
            worst = max(worst, report("text_encoder", m["text_encoder"](ids, ttl, tmask), r("text_emb")))
            worst = max(worst, report("vocoder", m["vocoder"](t(r("latent"))), r("wav")))

            # End to end from the same noise: errors compound through the 5-step loop.
            dur, emb, x, wav = synthesize(m, ids, tmask, ttl, dp, t(r("noise")), lmask)
            worst = max(worst, report("end-to-end latent", x, r("latent")))
            worst = max(worst, report("end-to-end wav", wav, r("wav")))
            sf.write(out / f"{voice}_{lang}_torch.wav", wav[0].cpu().numpy(), SAMPLE_RATE)

    # 1e-3 relative is far below anything audible; float32 reordering alone gives ~1e-6.
    print(f"\nworst relative diff {worst:.2e}: {'PASS' if worst < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
