"""Who a voice is: its gender and language, and the Kokoro cast that Elite Intel's voices were fitted to.

Gender decides which presets a fit may start from (see step5_fit_kokoro.py); language decides how the clip
text is read. Both are stored beside a voice's clips as gender.txt and lang.txt.
"""

from typing import NamedTuple

from config import CLIPS, KOKORO_CAST


class KokoroVoice(NamedTuple):
    name: str
    sid: int            # Kokoro speaker id
    male: bool
    display: str
    description: str


def kokoro_cast() -> list[KokoroVoice]:
    """The voices of Elite Intel's pre-V1.2 Kokoro cast, in its order."""
    rows = [line.split("\t") for line in KOKORO_CAST.read_text(encoding="utf-8").splitlines()[1:] if line]
    return [KokoroVoice(n, int(s), g == "m", d, desc) for n, s, g, d, desc in rows]


def clip_lang(voice: str) -> str:
    """The language the clips were spoken in; clips made before lang.txt existed are English."""
    path = CLIPS / voice / "lang.txt"
    return path.read_text().strip() if path.exists() else "en"


def gender(voice: str) -> str | None:
    """'m', 'f', or None if unknown: from the clips' gender.txt, else the Kokoro cast."""
    path = CLIPS / voice / "gender.txt"
    if path.exists():
        g = path.read_text().strip().lower()
        assert g in ("m", "f"), f"{path} should say m or f, says {g!r}"
        return g
    known = {v.name: v.male for v in kokoro_cast()}
    return None if voice not in known else ("m" if known[voice] else "f")


def write_gender(voice: str, g: str) -> None:
    g = g.lower()
    if g not in ("m", "f"):
        raise SystemExit(f"gender must be m or f, not {g!r}")
    (CLIPS / voice).mkdir(parents=True, exist_ok=True)
    (CLIPS / voice / "gender.txt").write_text(g)
