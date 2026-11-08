"""Read and write sherpa-onnx's Supertonic voice.bin, the packed voice file Elite Intel loads.

Layout (little-endian): six int64 header values [N, 50, 256, N, 8, 16], then all N timbre
styles (float32, N x 50 x 256), then all N pacing styles (float32, N x 8 x 16).
Speaker id i is the i-th voice; the shipped file holds F1-F5, M1-M5 as ids 0-9.
"""

import json
from pathlib import Path

import numpy as np


def read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).read_bytes()
    n, a, b, n2, c, d = np.frombuffer(raw[:48], dtype="<i8")
    assert n == n2 and (a, b, c, d) == (50, 256, 8, 16), f"unexpected voice.bin header {raw[:48].hex()}"
    data = np.frombuffer(raw[48:], dtype="<f4")
    assert data.size == n * (a * b + c * d), "voice.bin size does not match its header"
    ttl = data[: n * a * b].reshape(n, a, b)
    dp = data[n * a * b:].reshape(n, c, d)
    return ttl.copy(), dp.copy()


def write(path: Path, ttl: np.ndarray, dp: np.ndarray) -> None:
    n = ttl.shape[0]
    assert ttl.shape == (n, 50, 256) and dp.shape == (n, 8, 16)
    header = np.array([n, 50, 256, n, 8, 16], dtype="<i8")
    Path(path).write_bytes(header.tobytes() + ttl.astype("<f4").tobytes() + dp.astype("<f4").tobytes())


PRESETS = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]


def upstream_presets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The 10 upstream speakers (F1-F5, M1-M5 as ids 0-9) from a voice.bin.

    Any voice.bin this project packs keeps them unchanged as ids 0-9, so a packed file works here too.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found: save the voice.bin from sherpa-onnx's Supertonic 3 int8 release "
                         f"there, or point UPSTREAM_VOICE_BIN at it")
    ttl, dp = read(path)
    assert ttl.shape[0] >= len(PRESETS), f"{path} has only {ttl.shape[0]} speakers, expected at least 10"
    return ttl[:len(PRESETS)], dp[:len(PRESETS)]


def load_voice_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A fitted voice (or a Supertonic preset JSON) as (style_ttl [1,50,256], style_dp [1,8,16])."""
    d = json.loads(Path(path).read_text())
    return (np.array(d["style_ttl"]["data"], dtype=np.float32).reshape(d["style_ttl"]["dims"]),
            np.array(d["style_dp"]["data"], dtype=np.float32).reshape(d["style_dp"]["dims"]))
