"""Go/no-go step 3: do gradients reach the style vectors, and are they correct?

  1. Backprop a simple loss on the final audio to style_ttl, split into its two routes
     (text encoder, and the vector estimator across all flow-matching steps).
  2. Show the gradient survives back through every one of the steps.
  3. Backprop the predicted duration to style_dp (the pacing vector).
  4. Finite-difference check: nudge the style along a random direction for real and
     compare the change in loss with what the gradient predicted. A converter that
     silently cuts the autograd graph would give a gradient that exists but is wrong.

    .venv/bin/python scripts/step3_gradients.py
"""

import numpy as np
import torch

import torch_model
from common import OUT
from step1_onnx_baseline import SPEED, TOTAL_STEPS

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = "cuda"
LANG = "en"


def main() -> None:
    ref = np.load(OUT / "step1" / "reference.npz")
    m = torch_model.load(DEVICE)
    t = lambda a: torch.from_numpy(np.asarray(a)).to(DEVICE)
    r = lambda k: t(ref[f"{LANG}/{k}"])
    ids, tmask, lmask, noise = r("text_ids"), r("text_mask"), r("latent_mask"), r("noise")
    ttl0, dp0 = t(ref["style_ttl"]), t(ref["style_dp"])
    total = torch.tensor([TOTAL_STEPS], dtype=torch.float32, device=DEVICE)

    def audio_loss(ttl_te, ttl_ve, keep_steps=None):
        """A stand-in loss: energy of the generated audio. Any smooth function of it will do."""
        emb = m["text_encoder"](ids, ttl_te, tmask)
        x = noise
        for step in range(TOTAL_STEPS):
            cur = torch.tensor([step], dtype=torch.float32, device=DEVICE)
            x = m["vector_estimator"](x, emb, ttl_ve, lmask, tmask, cur, total)
            if keep_steps is not None:
                x.retain_grad()
                keep_steps.append(x)
        wav = m["vocoder"](x)
        return (wav ** 2).mean()

    ok = True

    # 1 + 2: gradient to style_ttl through both routes, and through every step.
    ttl_te = ttl0.clone().requires_grad_(True)
    ttl_ve = ttl0.clone().requires_grad_(True)
    steps = []
    loss = audio_loss(ttl_te, ttl_ve, steps)
    loss.backward()
    print("style_ttl (timbre):")
    for name, g in [("via text encoder", ttl_te.grad), ("via vector estimator", ttl_ve.grad)]:
        good = g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        ok &= bool(good)
        print(f"  {name:<22} |grad| = {g.norm():.3e}  nonzero {int((g != 0).sum())}/{g.numel()}  {'ok' if good else 'MISSING'}")
    print("  gradient at each flow-matching step's output (step 4 is the final latent):")
    for i, x in enumerate(steps):
        good = x.grad is not None and x.grad.abs().sum() > 0
        ok &= bool(good)
        print(f"    step {i}: |grad| = {x.grad.norm():.3e}  {'ok' if good else 'MISSING'}")

    # 3: gradient from predicted duration to style_dp.
    dp = dp0.clone().requires_grad_(True)
    dur = m["duration_predictor"](ids, dp, tmask) / SPEED
    dur.sum().backward()
    good = torch.isfinite(dp.grad).all() and dp.grad.abs().sum() > 0
    ok &= bool(good)
    print(f"style_dp (pacing):\n  d(duration)/d(style_dp) |grad| = {dp.grad.norm():.3e}  {'ok' if good else 'MISSING'}")

    # 4: finite-difference check along one random direction, for each vector.
    print("finite-difference check (predicted vs actual change in loss):")
    g_ttl = ttl_te.grad + ttl_ve.grad
    torch.manual_seed(0)
    with torch.no_grad():
        for name, x0, g, f in [
            ("style_ttl", ttl0, g_ttl, lambda s: audio_loss(s, s)),
            ("style_dp", dp0, dp.grad, lambda s: (m["duration_predictor"](ids, s, tmask) / SPEED).sum()),
        ]:
            v = torch.randn_like(x0)
            v *= x0.norm() / v.norm()        # direction the size of the vector itself
            predicted = (g * v).sum().item()
            for eps in [1e-2, 3e-3, 1e-3]:
                actual = ((f(x0 + eps * v) - f(x0 - eps * v)) / (2 * eps)).item()
                rel = abs(actual - predicted) / (abs(predicted) + 1e-12)
                print(f"  {name:<10} eps={eps:.0e}  predicted {predicted:+.4e}  actual {actual:+.4e}  rel err {rel:.1e}")
            ok &= rel < 5e-2

    print(f"\n{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
