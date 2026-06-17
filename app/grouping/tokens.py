"""Text → match tokens.

How OCR text and VLM hint lines are cut into the tokens the aligner overlaps to bind a cell to
its hint line, plus ``_token_score`` — the (script-agnostic) fuzzy/exact match of one token
against a hint's token set, which absorbs OCR garble.

Two segmentation passes, run on the same normalized text:

  * **spaced scripts** — Latin letters and digits, in runs (a word per run). This pass is
    byte-for-byte the historical tokenizer.
  * **scriptio continua** — Han / Kana / Hangul have no word spaces, so each such character is its
    own token. This is what lets a pure-CJK OCR line bind to its CJK hint line instead of falling
    through as a leftover.

Zero-regression boundary for non-CJK (incl. all Latin-script) source: the CJK pass only ever emits
characters in ``_CJK`` — code points that no character's NFKD decomposition introduces — so on any
text without CJK source characters the pass is empty and ``_tokens`` returns exactly the spaced
pass, i.e. the old ``re.findall(r"[a-z0-9]+", …)`` result. The Latin pass regex is left untouched
precisely so this holds even for inputs whose NFKD form lands in another alphabet (the micro sign
``µ`` → Greek ``μ``, ``Ω`` → Greek ``Ω``): those still produce no token, exactly as before. Other
spaced non-Latin scripts (Cyrillic, Greek, …) are intentionally NOT added here — Greek would break
that guarantee via ``µ``/``Ω`` — so they are left for a separate, deliberately-scoped pass.
``tests/test_tokens.py`` locks the equality against the old tokenizer over a Latin corpus.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

# Fuzzy token-match bounds: a token must be at least this long before a substring/ratio match
# counts (so short tokens cannot collide by chance), and similarity must reach this ratio.
_FUZZY_MIN_LEN = 4
_FUZZY_RATIO = 0.8

# Scriptio-continua characters (no word spaces) — each is its own token. Ranges cover Hiragana +
# Katakana, the common CJK ideograph blocks, and Hangul syllables; CJK punctuation (。、 …) is
# deliberately excluded so it is dropped, like ASCII punctuation. None of these code points is the
# NFKD image of any other character, which is what keeps the spaced pass unchanged for non-CJK text.
_CJK = (
    "぀-ヿ"  # Hiragana + Katakana
    "㐀-䶿"  # CJK Unified Ideographs Extension A
    "一-鿿"  # CJK Unified Ideographs
    "豈-﫿"  # CJK Compatibility Ideographs
    "가-힣"  # Hangul syllables
)

_SPACED = re.compile(r"[a-z0-9]+")
_CJK_CHAR = re.compile(f"[{_CJK}]")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _tokens(text: str) -> list[str]:
    """Match tokens for one text: spaced-script words (Latin + digits) plus one token per CJK
    character. For any text without CJK source characters this is exactly the historical
    ``[a-z0-9]+`` token list (see the module docstring for the zero-regression argument). Token
    order is irrelevant downstream — every consumer overlaps against a hint *set* or counts."""
    normalized = _normalize(text)
    return _SPACED.findall(normalized) + _CJK_CHAR.findall(normalized)


def _token_score(token: str, hint_set: set[str]) -> float:
    """1.0 exact, else fuzzy (slightly lower, so exact wins a tie) for OCR garble: the cell
    must still bind to its clean VLM line when OCR splits a word ("Kaar thouder" vs
    "Kaarthouder") or drops/adds a character ("AHNEDAARDBEI" vs "AHNEDAARBEI") — otherwise
    the cell becomes a leftover, the per-unit fallback translates the garbled text in
    isolation, and the good structured translation of the VLM line is orphaned. Fuzzy =
    substring or high character similarity, both only for tokens long enough that they
    cannot collide by chance; below exact so "Kaart" still binds its own line, not
    "Kaarthouder"."""
    if token in hint_set:
        return 1.0
    if len(token) < _FUZZY_MIN_LEN:
        return 0.0
    for hint_token in hint_set:
        if len(hint_token) < _FUZZY_MIN_LEN:
            continue
        if token in hint_token or hint_token in token:
            return 0.9
        shorter, longer = sorted((len(token), len(hint_token)))
        if 2 * shorter / (shorter + longer) < _FUZZY_RATIO:  # ratio can't reach the bar
            continue
        if difflib.SequenceMatcher(None, token, hint_token).ratio() >= _FUZZY_RATIO:
            return 0.9
    return 0.0
