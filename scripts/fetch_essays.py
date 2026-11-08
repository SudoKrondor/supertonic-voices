"""Fetch public-domain reading material for the non-English voices.

The English essay cannot train a Spanish, French, Italian, Portuguese or Hindi voice: Kokoro
reads it with English rules, so the clips are mispronounced and the fit copies that (ES_DORA).
Each of those voices needs ~40 minutes of natural speech in its own language instead.

Sources, both free to use: Project Gutenberg prose for Spanish, French and Italian, and
Wikipedia's featured articles for Portuguese and Hindi (Gutenberg has no Hindi, and its
Portuguese is pre-1911 spelling - "elle", "annos", "theatro" - which would be mispronounced).

    .venv/bin/python scripts/fetch_essays.py [LANG ...]      # default: all five

Writes data/essays/<lang>.txt, plain prose, one paragraph per line, ready for
make_kokoro_clips.py. Downloads land in data/essays/raw/ and are only read as text.
"""

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from common import ROOT

OUT = ROOT / "data" / "essays"
RAW = OUT / "raw"
UA = {"User-Agent": "supertonic-voice-fitting/1.0 (research; contact via github.com/stone-alex)"}
WORDS_PER_SECOND = 2.6          # same rough pace make_kokoro_clips.py assumes
TARGET_WORDS = 8000             # ~50 min of speech, a little over what BELLA was fitted on

# Project Gutenberg ids: prose, modern enough orthography to be read correctly.
GUTENBERG = {
    "es": [25640, 55514],       # Blasco Ibanez, Los argonautas; Pardo Bazan, Cuentos de amor
    "fr": [62215],              # Leroux, Le Fantome de l'Opera
    "it": [52484, 61885],       # Collodi, Pinocchio; De Amicis, Ricordi d'infanzia e di scuola
}
# Wikipedia featured-article categories ("featured article" in the language itself).
FEATURED = {
    "pt": "Categoria:!Artigos destacados",
    "hi": "श्रेणी:निर्वाचित लेख",
}


def get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def gutenberg(book: int) -> str:
    """The book's text, without Gutenberg's own header and footer."""
    path = RAW / f"pg{book}.txt"
    if not path.exists():
        path.write_bytes(get(f"https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt"))
    text = path.read_text(encoding="utf-8", errors="replace")
    body = re.split(r"\*\*\* ?START OF TH[EI]S? PROJECT GUTENBERG.*?\*\*\*", text)[-1]
    return re.split(r"\*\*\* ?END OF TH[EI]S? PROJECT GUTENBERG", body)[0]


def api(lang: str, **params) -> dict:
    params = {"format": "json", "action": "query", "formatversion": "2", **params}
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    return json.loads(get(url))


def featured(lang: str, category: str) -> str:
    """Plain text of featured articles, longest first, until there is enough of it."""
    path = RAW / f"wiki_{lang}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    titles = [p["title"] for p in api(lang, list="categorymembers", cmtitle=category,
                                     cmlimit=60, cmnamespace=0)["query"]["categorymembers"]]
    parts, words = [], 0
    for i in range(0, len(titles), 6):
        pages = api(lang, prop="extracts", explaintext=1, titles="|".join(titles[i:i + 6]))
        for p in pages["query"]["pages"]:
            text = p.get("extract", "")
            parts.append(text)
            words += len(text.split())
        if words > TARGET_WORDS * 3:        # cleaning throws a lot away
            break
    text = "\n\n".join(parts)
    path.write_text(text, encoding="utf-8")
    return text


def clean(text: str, latin_ok: bool) -> list[str]:
    """Readable prose paragraphs: no headings, lists, notes, or lines full of names and digits."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        para = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", para)        # footnotes, asides, glosses
        para = re.sub(r"[_*]|\s{2,}", " ", para).strip(" =")
        if len(para.split()) < 25 or not re.search(r"[.!?।]\s*$", para):
            continue                                            # headings, captions, fragments
        if len(re.findall(r"\d", para)) / len(para) > 0.02:     # dates, tables, statistics
            continue
        if not latin_ok and len(re.findall(r"[A-Za-z]", para)) / len(para) > 0.05:
            continue                                            # Hindi: paragraphs full of English
        out.append(para)
    return out


def main(*langs: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for lang in langs or [*GUTENBERG, *FEATURED]:
        if lang in GUTENBERG:
            raw = "\n\n".join(gutenberg(b) for b in GUTENBERG[lang])
        else:
            raw = featured(lang, FEATURED[lang])
        paras, kept, words = clean(raw, latin_ok=lang != "hi"), [], 0
        for para in paras:
            kept.append(para)
            words += len(para.split())
            if words >= TARGET_WORDS:
                break
        path = OUT / f"{lang}.txt"
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"{lang}: {len(kept)} paragraphs, {words} words, about "
              f"{words / WORDS_PER_SECOND / 60:.0f} min of speech -> {path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
