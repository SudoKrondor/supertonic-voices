"""Every path outside scripts/, in one place. Each one can be overridden with an environment variable.

Only the Elite Intel and Kokoro paths are optional: fitting a voice from your own clips and packing it
need nothing but this repo, the Supertonic model in model/ and the upstream voice.bin.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    return Path(os.environ[env]).expanduser() if os.environ.get(env) else default


MODEL = _path("SUPERTONIC_MODEL_DIR", ROOT / "model")          # git clone of supertone-oss-archive/supertonic-3
CLIPS = ROOT / "data" / "kokoro"                                # one folder of training clips per voice
OUT = ROOT / "out"

# The 10-speaker voice.bin that sherpa-onnx's Supertonic 3 int8 release ships. Ids 0-9 of every packed file
# are copied from it, unchanged.
UPSTREAM_VOICE_BIN = _path("UPSTREAM_VOICE_BIN", OUT / "voice_bin_backup" / "voice.bin.original")

# The fitting voice list for the Kokoro cast (name, Kokoro speaker id, gender), kept here because Elite Intel
# no longer has Kokoro.
KOKORO_CAST = ROOT / "voices" / "kokoro_cast.tsv"

# Optional, Elite Intel only.
ELITE_INTEL = _path("ELITE_INTEL_DIR", Path.home() / "projects" / "EliteIntel")
SUPERTONIC_ENUM = _path("SUPERTONIC_ENUM", ELITE_INTEL / "app/src/main/java/elite/intel/ai/mouth/supertonic/SupertonicVoices.java")
# The int8 model sherpa-onnx runs in the app; check_int8.py renders with it.
INT8_MODEL = _path("SUPERTONIC_INT8_DIR", ELITE_INTEL / "distribution/tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11")
# Kokoro, only needed to make Kokoro clips (removed from Elite Intel in da9503a2c; restore it from there).
KOKORO_DIR = _path("KOKORO_MODEL_DIR", ELITE_INTEL / "distribution/tts/kokoro-multi-lang-v1_0")
