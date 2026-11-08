"""Refit voices that were fitted before step 5 kept to the voice's own gender.

ERIC and GEORGE (male) started from F5 and came out female. Step 5 now starts only from presets
of the same gender; this reruns it for each voice given, moving the old fit to
out/step5/<VOICE>_mixed_gender first.

    .venv/bin/python scripts/refit_same_gender.py VOICE [VOICE ...]
"""

import shutil
import sys

import step5_fit_kokoro
from common import OUT


def main(*voices: str) -> None:
    for voice in voices:
        out = OUT / "step5" / voice
        old = OUT / "step5" / f"{voice}_mixed_gender"
        if out.exists() and not old.exists():
            shutil.move(out, old)
        step5_fit_kokoro.main(voice)


if __name__ == "__main__":
    main(*sys.argv[1:])
