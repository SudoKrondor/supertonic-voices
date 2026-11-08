"""Go/no-go step 4b: step 4 again, with two fixes after step 4 fooled the speaker model.

Step 4 raised the ECAPA score while moving the voice lower and louder, away from F2.
  1. The loss also matches pitch (mean and spread) and the long-term average spectrum,
     which are what ears notice first. Each term is scaled so it starts at 1.
  2. Stage A fits only a blend of the presets, which always stays a natural voice.
     The target preset is excluded from the blend so it can't just pick it.
     Stage B then fine-tunes the full timbre vector from that blend.

Pacing (style_dp) is fitted to the target's clip lengths as in step 4.

    .venv/bin/python scripts/step4b_blend_then_fit.py [START] [TARGET] [BLEND_ITERS] [FIT_ITERS]
"""

import json
import sys
import time

import numpy as np
import soundfile as sf
import torch

import torch_model
from acoustics import Acoustics, hz
from common import LATENT_CH, OUT, SAMPLE_RATE, load_voice, text_processor
from speaker import SpeakerEncoder
from step1_onnx_baseline import SPEED
from step4_recover_f2 import BATCH, HELD_OUT, LR_DP, TRAIN, frames_for, save_voice

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = "cuda"
PRESETS = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]

LR_BLEND = 5e-2    # on blend logits
LR_FIT = 3e-4      # on the timbre vector; smaller than step 4, we start close and refine


def main(start: str = "M1", target: str = "F2", blend_iters: int = 150, fit_iters: int = 300) -> None:
    blend_iters, fit_iters = int(blend_iters), int(fit_iters)
    out = OUT / "step4b"
    (out / "eval").mkdir(parents=True, exist_ok=True)
    m = torch_model.load(DEVICE)
    enc = SpeakerEncoder()
    ac = Acoustics(DEVICE)
    tp = text_processor()
    t = lambda a: torch.from_numpy(np.asarray(a)).to(DEVICE)

    def encode(text, lang="en"):
        ids, tmask = tp([text], lang)
        return t(ids), t(tmask)

    def durations(encoded, dp):
        return torch.cat([m["duration_predictor"](ids, dp, tmask) for ids, tmask in encoded]) / SPEED

    def render(ids, tmask, ttl, n_frames, seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        noise = torch.randn((1, LATENT_CH, n_frames), generator=g, device=DEVICE)
        lmask = torch.ones((1, 1, n_frames), device=DEVICE)
        return m["vocoder"](torch_model.synthesize_latent(m, ids, tmask, ttl, noise, lmask))

    def features(wav):
        mean, spread = ac.pitch_stats(wav)
        return {"emb": enc(wav)[0], "pitch": mean, "spread": spread, "ltas": ac.ltas(wav)}

    pool = [p for p in PRESETS if p != target]
    pool_ttl = torch.cat([t(load_voice(p)[0]) for p in pool])   # [9, 50, 256]
    pool_dp = torch.cat([t(load_voice(p)[1]) for p in pool])    # [9, 8, 16]
    ttl_s, dp_s = map(t, load_voice(start))
    ttl_t, dp_t = map(t, load_voice(target))
    train = [encode(x) for x in TRAIN]

    # The "recordings": target audio per training sentence, summarised into what we match.
    with torch.no_grad():
        tgt_dur = durations(train, dp_t)
        tgt_frames = [frames_for(float(d)) for d in tgt_dur]
        tf = [features(render(ids, tm, ttl_t, n, seed=i)) for i, ((ids, tm), n) in enumerate(zip(train, tgt_frames))]
        target_feat = {
            "emb": torch.nn.functional.normalize(torch.stack([f["emb"] for f in tf]).mean(0), dim=-1),
            "pitch": torch.stack([f["pitch"] for f in tf]).mean(),
            "spread": torch.stack([f["spread"] for f in tf]).mean(),
            "ltas": torch.stack([f["ltas"] for f in tf]).mean(0),
        }

    def terms(f):
        """Raw distance from the target voice, per aspect."""
        return {
            "speaker": 1 - f["emb"] @ target_feat["emb"],
            "pitch": (f["pitch"] - target_feat["pitch"]).abs() + 0.5 * (f["spread"] - target_feat["spread"]).abs(),
            "spectrum": (f["ltas"] - target_feat["ltas"]).abs().mean(),
        }

    def evaluate(ttl, seed0):
        with torch.no_grad():
            fs = [features(render(ids, tm, ttl, n, seed=seed0 + i)) for i, ((ids, tm), n) in enumerate(zip(train, tgt_frames))]
            per = [terms(f) for f in fs]
            return {k: float(torch.stack([p[k] for p in per]).mean()) for k in per[0]} | {
                "sim": float(torch.stack([f["emb"] @ target_feat["emb"] for f in fs]).mean()),
                "hz": hz(float(torch.stack([f["pitch"] for f in fs]).mean()))}

    floor, ceiling = evaluate(ttl_s, 500), evaluate(ttl_t, 500)
    scale = {k: floor[k] for k in ["speaker", "pitch", "spectrum"]}  # each term starts at ~1 for the start voice
    print(f"{'':<28}{'sim':>6} {'pitch':>7} {'spectrum':>9}")
    print(f"{start + ' as-is (floor)':<28}{floor['sim']:>6.3f} {floor['hz']:>5.0f}Hz {floor['spectrum']:>9.3f}")
    print(f"{target + ' new noise (ceiling)':<28}{ceiling['sim']:>6.3f} {ceiling['hz']:>5.0f}Hz {ceiling['spectrum']:>9.3f}")
    print(f"pool for blending: {', '.join(pool)}")

    logits = torch.zeros(len(pool), device=DEVICE, requires_grad=True)
    raw_ttl = None
    raw_dp = dp_s.clone().requires_grad_(True)
    rng = np.random.default_rng(0)

    def blended(w):
        return torch_model.unit_rows((w[:, None, None] * pool_ttl).sum(0, keepdim=True))

    def run_stage(name, params_lr, iters, ttl_fn):
        opt = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in params_lr] + [{"params": [raw_dp], "lr": LR_DP}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        print(f"\n{name}\n{'iter':>5} {'loss':>6} {'sim':>6} {'pitch':>7} {'spectrum':>9} {'dur err':>8}  s/iter")
        t0 = time.time()
        for it in range(1, iters + 1):
            opt.zero_grad()
            log = []
            for k in rng.choice(len(TRAIN), BATCH, replace=False):
                ids, tm = train[k]
                f = features(render(ids, tm, ttl_fn(), tgt_frames[k], seed=10_000 + it * BATCH + len(log)))
                tr = terms(f)
                loss = sum(tr[k] / scale[k] for k in tr)
                (loss / BATCH).backward()
                log.append((loss.item(), (f["emb"] @ target_feat["emb"]).item(), f["pitch"].item(), tr["spectrum"].item()))
            dur = durations(train, torch_model.unit_rows(raw_dp))
            ((dur.log() - tgt_dur.log()) ** 2).mean().backward()
            opt.step()
            sched.step()
            if it % 10 == 0 or it == 1:
                l = np.mean(log, 0)
                rel_dur = float(((dur - tgt_dur).abs() / tgt_dur).mean())
                print(f"{it:>5} {l[0]:>6.3f} {l[1]:>6.3f} {hz(l[2]):>5.0f}Hz {l[3]:>9.3f} {rel_dur:>7.1%}  {(time.time() - t0) / it:.2f}", flush=True)

    # Stage A: only the blend weights move.
    run_stage("stage A: blend of presets", [(logits, LR_BLEND)], blend_iters, lambda: blended(logits.softmax(0)))
    w = logits.softmax(0).detach()
    ttl_blend = blended(w).detach()
    weights = {p: round(float(x), 4) for p, x in sorted(zip(pool, w), key=lambda pw: -float(pw[1]))}
    print("blend weights:", ", ".join(f"{p} {x:.0%}" for p, x in weights.items() if x >= 0.01))

    # Stage B: the full timbre vector moves, starting from the blend.
    raw_ttl = ttl_blend.clone().requires_grad_(True)
    run_stage("stage B: fine-tune from blend", [(raw_ttl, LR_FIT)], fit_iters, lambda: torch_model.unit_rows(raw_ttl))
    ttl_fit = torch_model.unit_rows(raw_ttl).detach()
    dp_fit = torch_model.unit_rows(raw_dp).detach()

    name = f"{start}to{target}"
    save_voice(out / f"{name}_blend.json", ttl_blend, dp_fit, f"blend of presets fitted to {target} audio")
    save_voice(out / f"{name}.json", ttl_fit, dp_fit, f"blend + fine-tune fitted to {target} audio")
    (out / f"{name}_blend_weights.json").write_text(json.dumps(weights, indent=2))

    print(f"\non training sentences, fresh noise:{'sim':>8} {'pitch':>7} {'spectrum':>9}")
    for label, ttl in [(f"{start} (start)", ttl_s), ("blend", ttl_blend), ("fitted", ttl_fit), (f"{target} (target)", ttl_t)]:
        e = evaluate(ttl, 500)
        print(f"  {label:<33}{e['sim']:>6.3f} {e['hz']:>5.0f}Hz {e['spectrum']:>9.3f}")

    print("\nheld-out sentences: sim / pitch (wavs in out/step4b/eval/):")
    with torch.no_grad():
        for i, (key, text) in enumerate(HELD_OUT.items()):
            ids, tm = encode(text, key[:2])
            row = []
            for label, s_ttl, s_dp in [("start", ttl_s, dp_s), ("blend", ttl_blend, dp_fit),
                                       ("fitted", ttl_fit, dp_fit), ("target", ttl_t, dp_t)]:
                d = float(m["duration_predictor"](ids, s_dp, tm)[0]) / SPEED
                wav = render(ids, tm, s_ttl, frames_for(d), seed=777 + i)
                sf.write(out / "eval" / f"{key}_{label}.wav", wav[0].cpu().numpy(), SAMPLE_RATE)
                f = features(wav)
                row.append(f"{label} {float(f['emb'] @ target_feat['emb']):.2f}/{hz(f['pitch'].item()):.0f}Hz")
            print(f"  {key:<4} " + "   ".join(row))


if __name__ == "__main__":
    main(*sys.argv[1:])
