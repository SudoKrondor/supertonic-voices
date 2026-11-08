"""Step 5: fit a Supertonic voice to a Kokoro voice from clips made by make_kokoro_clips.py.

Timbre (speaker similarity, average pitch, spectrum) is fitted to Kokoro. Liveliness is not:
Kokoro sounds more machine-like than Supertonic, so instead of copying its pitch range we keep
pitch movement at least as high as the starting preset's. (BELLA was fitted with an earlier rule,
"the liveliest preset of the same gender", 0.42 against F1's 0.35; she ended at 0.44 either way.)

    .venv/bin/python scripts/step5_fit_kokoro.py VOICE [ITERS] [OUT_DIR]

Reads data/kokoro/<VOICE>/clips.tsv. Writes <OUT_DIR>/<VOICE>.json (default out/step5/<VOICE>)
and eval/ WAVs to listen to:
  NNNN_kokoro / NNNN_start / NNNN_fitted   held-out essay clips: the original, nearest preset, fitted
  <lang>_start / <lang>_fitted             test sentences in EN, IT, RU, UK (Kokoro has no reference)
"""

import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch

from common import OUT, ROOT, SAMPLE_RATE
from fitting import PRESETS, Fitter
from step4_recover_f2 import HELD_OUT, save_voice

N_EVAL_CLIPS = 6


def clip_lang(voice: str) -> str:
    """The language the clips were spoken in; clips made before lang.txt existed are English."""
    path = ROOT / "data" / "kokoro" / voice / "lang.txt"
    return path.read_text().strip() if path.exists() else "en"


def load_clips(voice: str):
    root = ROOT / "data" / "kokoro" / voice
    rows = [line.split("\t") for line in (root / "clips.tsv").read_text(encoding="utf-8").splitlines()[1:] if line]
    clips = {"train": [], "heldout": []}
    for name, split, _, text in rows:
        wav, sr = sf.read(root / "wavs" / name, dtype="float32")
        clips[split].append((name, text, torch.from_numpy(wav), sr))
    return root, clips


def main(voice: str, iters: int = 400, out_dir: str | None = None) -> None:
    iters = int(iters)
    out = Path(out_dir) if out_dir else OUT / "step5" / voice
    (out / "eval").mkdir(parents=True, exist_ok=True)
    root, clips = load_clips(voice)
    lang = clip_lang(voice)
    print(f"{voice}: {len(clips['train'])} training clips, {len(clips['heldout'])} held out, language {lang}")

    fit = Fitter()
    tg = fit.target_from_clips([(text, lang, wav, sr) for _, text, wav, sr in clips["train"]])

    print(f"\npresets ranked by distance to {voice}:")
    print(f"  {'':<4}{'loss':>6} {'sim':>6} {'pitch':>7} {'spectrum':>9} {'movement':>9}")
    # Start only from presets of the voice's own gender: the loss alone picked F5 for ERIC and
    # GEORGE, and both came out female. (Imported here: batch_fit imports this module.)
    from batch_fit import cast
    male = {v[0]: v[2] for v in cast()}.get(voice)
    other = () if male is None else tuple(p for p in PRESETS if p.startswith("F" if male else "M"))
    ranking = fit.rank_presets(tg, exclude=other)
    for name, s in ranking:
        print(f"  {name:<4}{s['loss']:>6.2f} {s['sim']:>6.3f} {s['hz']:>5.0f}Hz {s['spectrum']:>9.3f} {s['movement']:>9.3f}")
    start, start_scores = ranking[0]
    # Never flatter than where we start. (Not "the liveliest preset": on deep male voices the pitch
    # tracker wobbles, which inflates "movement" - M5 measures 1.04 against 0.32-0.42 for the females.)
    fit.liveliness = start_scores["movement"]
    print(f"  {voice} (Kokoro) pitch movement {tg.movement.item():.3f}; "
          f"keeping fitted voice at >= {fit.liveliness:.3f} ({start}, the starting preset)")

    ttl_s, dp_s = fit.preset(start)
    print(f"\nstarting from {start}; fine-tuning {iters} iterations")
    ttl_f, dp_f = fit.fine_tune(tg, ttl_s, dp_s, iters=iters)
    save_voice(out / f"{voice}.json", ttl_f, dp_f, f"fitted to Kokoro {voice}, starting from {start}")
    print(f"\nsaved {out / f'{voice}.json'}")

    print(f"\non training clips, fresh noise:{'sim':>10} {'pitch':>7} {'spectrum':>9} {'movement':>9}")
    for label, ttl in [(f"{start} (start)", ttl_s), ("fitted", ttl_f)]:
        e = fit.measure(ttl, tg, n=24)
        print(f"  {label:<30}{e['sim']:>6.3f} {e['hz']:>5.0f}Hz {e['spectrum']:>9.3f} {e['movement']:>9.3f}")

    print(f"\nheld-out clips: sim to {voice} (wavs in {out / 'eval'}):")
    with torch.no_grad():
        for i, (name, text, _, _) in enumerate(clips["heldout"][:N_EVAL_CLIPS]):
            stem = Path(name).stem
            shutil.copy(root / "wavs" / name, out / "eval" / f"{stem}_kokoro.wav")
            row = []
            for label, ttl, dp in [("start", ttl_s, dp_s), ("fitted", ttl_f, dp_f)]:
                wav = fit.speak(text, lang, ttl, dp, seed=777 + i)
                sf.write(out / "eval" / f"{stem}_{label}.wav", wav[0].cpu().numpy(), SAMPLE_RATE)
                row.append(f"{label} {(fit.features(wav)['emb'] @ tg.emb).item():.2f}")
            print(f"  {stem}  " + "   ".join(row))
        for i, (key, text) in enumerate(HELD_OUT.items()):
            for label, ttl, dp in [("start", ttl_s, dp_s), ("fitted", ttl_f, dp_f)]:
                wav = fit.speak(text, key[:2], ttl, dp, seed=900 + i)
                sf.write(out / "eval" / f"{key}_{label}.wav", wav[0].cpu().numpy(), SAMPLE_RATE)
    print("done")


if __name__ == "__main__":
    main(*sys.argv[1:])
