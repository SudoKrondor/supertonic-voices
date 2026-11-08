"""Read and write sherpa-onnx's Supertonic voice.bin, the packed voice file Elite Intel loads.

Layout (little-endian): six int64 header values [N, 50, 256, N, 8, 16], then all N timbre
styles (float32, N x 50 x 256), then all N pacing styles (float32, N x 8 x 16).
Speaker id i is the i-th voice; the shipped file holds F1-F5, M1-M5 as ids 0-9.
"""

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
