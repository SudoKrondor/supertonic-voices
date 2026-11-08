"""Pack fitted voices into a voice.bin for sherpa-onnx's Supertonic 3 engine.

Ids 0-9 are the 10 upstream presets (F1-F5, M1-M5), unchanged, so anything that already uses them keeps
working. The voices you name follow as ids 10, 11, ... in the order given.

    .venv/bin/python scripts/pack_voices.py OUT.bin VOICE [VOICE ...]
    .venv/bin/python scripts/pack_voices.py OUT.bin --no-presets VOICE [VOICE ...]     # your voices from id 0

A VOICE is a name fitted by step5_fit_kokoro.py (out/step5/<VOICE>/<VOICE>.json) or a path to any voice
JSON in Supertonic's format, such as model/voice_styles/F1.json. The presets come from the upstream
voice.bin (config.UPSTREAM_VOICE_BIN, or UPSTREAM_VOICE_BIN).

To use it, put OUT.bin in the int8 model folder as voice.bin and pick the voice with its speaker id (sid).
"""

import argparse
from pathlib import Path

import numpy as np

import voicebin
from config import OUT, UPSTREAM_VOICE_BIN


def voice_json(voice: str) -> Path:
    path = Path(voice)
    if path.suffix == ".json":
        if not path.exists():
            raise SystemExit(f"{path} not found")
        return path
    path = OUT / "step5" / voice / f"{voice}.json"
    if not path.exists():
        raise SystemExit(f"{voice} is not fitted yet ({path} not found)")
    return path


def pack(out: Path, voices: list[str], presets: bool = True) -> list[tuple[int, str]]:
    """Write out; return (speaker id, voice) for every speaker in it."""
    ttl, dp, ids = [], [], []
    if presets:
        t, d = voicebin.upstream_presets(UPSTREAM_VOICE_BIN)
        ttl.append(t)
        dp.append(d)
        ids += list(enumerate(voicebin.PRESETS))
    for v in voices:
        t, d = voicebin.load_voice_json(voice_json(v))
        ttl.append(t)
        dp.append(d)
        ids.append((len(ids), Path(v).stem))
    out.parent.mkdir(parents=True, exist_ok=True)
    voicebin.write(out, np.concatenate(ttl), np.concatenate(dp))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack fitted voices into a Supertonic voice.bin.")
    ap.add_argument("out", type=Path, help="the voice.bin to write")
    ap.add_argument("voices", nargs="+", help="fitted voice names or voice JSON paths, in speaker-id order")
    ap.add_argument("--no-presets", action="store_true", help="leave out the 10 upstream presets")
    a = ap.parse_args()
    ids = pack(a.out, a.voices, presets=not a.no_presets)
    print(f"wrote {a.out}: {len(ids)} speakers")
    for sid, name in ids:
        print(f"  sid {sid:>3}  {name}")


if __name__ == "__main__":
    main()
