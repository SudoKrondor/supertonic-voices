"""Go/no-go step 4: starting from preset M1, can we fit our way to F2 from F2's audio alone?

F2's audio stands in for Kokoro clips: we generate it with Supertonic, so the right answer
is known. We never look at F2's style vectors while fitting, only at the end to report.

  timbre (style_ttl): push up ECAPA "same speaker?" similarity to F2's clips
  pacing (style_dp):  match predicted durations to F2's clip lengths

Training uses English only, like the Kokoro data will. The held-out sentences, including
Italian, Russian and Ukrainian, are rendered for listening with M1 (start), the fitted
voice, and F2 (target) from the same noise.

    .venv/bin/python scripts/step4_recover_f2.py [START] [TARGET] [ITERS]
"""

import json
import sys
import time

import numpy as np
import soundfile as sf
import torch

import torch_model
from common import CHUNK, LATENT_CH, OUT, SAMPLE_RATE, SENTENCES, load_voice, text_processor
from speaker import SpeakerEncoder
from step1_onnx_baseline import SPEED

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = "cuda"

TRAIN = [
    "The station is quiet tonight, and the traffic control lights are blinking slowly.",
    "How many jumps until we reach the next system?",
    "Fuel reserves are at forty two percent.",
    "I would not recommend landing on that planet without a heat sink.",
    "Shields are holding.",
    "Your bounty has been paid. Twelve thousand five hundred credits have been added to your account.",
    "Did you remember to restock the limpets before we left?",
    "Scanning complete. Three unknown signals detected on the far side of the moon.",
    "Warning: hull integrity is dropping fast.",
    "We have arrived at the outpost, but the market is closed until morning.",
    "Would you like me to plot a route to the nearest material trader?",
    "The cargo hold contains sixteen tons of palladium and four tons of gold.",
    "Frame shift drive charging. Stand by.",
    "That pilot has been following us for the last three jumps.",
    "It is a beautiful view from here, isn't it?",
    "Landing gear deployed.",
    "According to the logs, the previous owner never once serviced the power plant.",
    "Mission accepted. The client wants the data delivered within forty eight hours.",
    "Why would anyone build a station this far from the core systems?",
    "I have updated your navigation panel with the new waypoints.",
    "Temperature is rising. I suggest we move away from the star.",
    "One hundred and twenty light years remain on the current route.",
    "Please confirm that you want to sell all of the cargo.",
    "Interdiction detected! Submit or evade, commander.",
]
HELD_OUT = {
    "en0": "The research team found traces of an ancient structure buried beneath the ice.",
    "en1": "Can you tell me what time the next shuttle leaves for the orbital platform?",
    **SENTENCES,  # en, it, ru, uk from step 1
}

BATCH = 4          # sentences per gradient step
LR_TTL = 1e-3      # Adam moves each element ~lr per step: ~1.6% of the vector, the size step 3 says is safe
LR_DP = 1e-2


def frames_for(seconds: float) -> int:
    return (int(seconds * SAMPLE_RATE) + CHUNK - 1) // CHUNK


def save_voice(path, ttl: torch.Tensor, dp: torch.Tensor, source: str) -> None:
    ttl, dp = ttl.detach().cpu().numpy(), dp.detach().cpu().numpy()
    path.write_text(json.dumps({
        "style_ttl": {"data": ttl.tolist(), "dims": list(ttl.shape), "type": "float32"},
        "style_dp": {"data": dp.tolist(), "dims": list(dp.shape), "type": "float32"},
        "metadata": {"source_file": source, "fitted_by": "supertonic-voices step4",
                     "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
    }))


def main(start: str = "M1", target: str = "F2", iters: int = 300) -> None:
    iters = int(iters)
    out = OUT / "step4"
    (out / "eval").mkdir(parents=True, exist_ok=True)
    m = torch_model.load(DEVICE)
    enc = SpeakerEncoder()
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

    ttl_s, dp_s = map(t, load_voice(start))
    ttl_t, dp_t = map(t, load_voice(target))
    train = [encode(x) for x in TRAIN]

    # The "recordings": target-voice audio for each training sentence, plus its length.
    with torch.no_grad():
        tgt_dur = durations(train, dp_t)
        tgt_frames = [frames_for(float(d)) for d in tgt_dur]
        tgt_emb = torch.cat([enc(render(ids, tm, ttl_t, n, seed=i)) for i, ((ids, tm), n) in enumerate(zip(train, tgt_frames))])
        centroid = torch.nn.functional.normalize(tgt_emb.mean(0), dim=-1)

        def similarity(ttl, seed0):
            """Mean ECAPA similarity to the target voice over all training sentences, fresh noise."""
            e = torch.cat([enc(render(ids, tm, ttl, n, seed=seed0 + i)) for i, ((ids, tm), n) in enumerate(zip(train, tgt_frames))])
            return float((e @ centroid).mean())

        print(f"speaker similarity to {target} (1.0 = identical):")
        print(f"  {start} as-is (floor):                 {similarity(ttl_s, 500):.3f}")
        print(f"  {target} itself, new noise (ceiling):  {similarity(ttl_t, 500):.3f}")

    raw_ttl = ttl_s.clone().requires_grad_(True)
    raw_dp = dp_s.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [raw_ttl], "lr": LR_TTL}, {"params": [raw_dp], "lr": LR_DP}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    rng = np.random.default_rng(0)

    print(f"\n{'iter':>5} {'sim':>6} {'dur err':>8} {'|ttl-start|':>12} {'|ttl-target|':>13} {'|dp-target|':>12}  s/iter")
    t0 = time.time()
    for it in range(1, iters + 1):
        opt.zero_grad()
        sims = []
        for k in rng.choice(len(TRAIN), BATCH, replace=False):
            ids, tm = train[k]
            # Fresh noise each time so the voice works for any noise, not just one draw.
            wav = render(ids, tm, torch_model.unit_rows(raw_ttl), tgt_frames[k], seed=10_000 + it * BATCH + len(sims))
            sim = (enc(wav) @ centroid).squeeze()
            (-sim / BATCH).backward()
            sims.append(sim.item())

        dur = durations(train, torch_model.unit_rows(raw_dp))
        dur_loss = ((dur.log() - tgt_dur.log()) ** 2).mean()
        dur_loss.backward()
        opt.step()
        sched.step()

        if it % 10 == 0 or it == 1:
            with torch.no_grad():
                ttl, dp = torch_model.unit_rows(raw_ttl), torch_model.unit_rows(raw_dp)
                rel_dur = float(((dur - tgt_dur).abs() / tgt_dur).mean())
                print(f"{it:>5} {np.mean(sims):>6.3f} {rel_dur:>7.1%} {float((ttl - ttl_s).norm()):>12.3f} "
                      f"{float((ttl - ttl_t).norm()):>13.3f} {float((dp - dp_t).norm()):>12.3f}  {(time.time() - t0) / it:.2f}", flush=True)

    ttl, dp = torch_model.unit_rows(raw_ttl).detach(), torch_model.unit_rows(raw_dp).detach()
    name = f"{start}to{target}"
    save_voice(out / f"{name}.json", ttl, dp, f"fitted from {start} to {target} audio")
    print(f"\nsaved {(out / f'{name}.json').relative_to(OUT.parent)}  "
          f"(start->target distance was {float((ttl_s - ttl_t).norm()):.3f} ttl, {float((dp_s - dp_t).norm()):.3f} dp)")

    with torch.no_grad():
        print(f"speaker similarity to {target}, fitted, on training sentences: {similarity(ttl, 500):.3f}")
        print("\nheld-out sentences (similarity to target centroid; wavs in out/step4/eval/):")
        for i, (key, text) in enumerate(HELD_OUT.items()):
            lang = key[:2]
            ids, tm = encode(text, lang)
            row = []
            for label, s_ttl, s_dp in [("start", ttl_s, dp_s), ("fitted", ttl, dp), ("target", ttl_t, dp_t)]:
                d = float(m["duration_predictor"](ids, s_dp, tm)[0]) / SPEED
                wav = render(ids, tm, s_ttl, frames_for(d), seed=777 + i)
                sf.write(out / "eval" / f"{key}_{label}.wav", wav[0].cpu().numpy(), SAMPLE_RATE)
                row.append(f"{label} {float(enc(wav) @ centroid):.3f} ({d:.2f}s)")
            print(f"  {key:<4} " + "   ".join(row))


if __name__ == "__main__":
    main(*sys.argv[1:])
