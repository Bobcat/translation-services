"""Small, self-contained grouping heuristics.

Pure predicates the aligner (and the translator) consult — each a single text or geometry rule
with a magic threshold or two, collected here so the rules read as a list instead of being
scattered through the alignment algorithm. No alignment state: every function takes plain
cell/text/box data, so this module has no dependency on ``align``.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from statistics import median
from typing import Any


# Fuzzy token-match bounds: a token must be at least this long before a substring/ratio match
# counts (so short tokens cannot collide by chance), and similarity must reach this ratio.
_FUZZY_MIN_LEN = 4
_FUZZY_RATIO = 0.8

# Below this fraction of the line's text height a same-unit member is treated as an icon/badge
# label rather than body text (see _is_icon_fragment).
_ICON_HEIGHT_RATIO = 0.6

# A whole-cell URL (scheme, ``www.`` or a trailing domain suffix) is not translatable.
_URL_SUFFIX = re.compile(r"\.(com|nl|org|net|io|de|fr|co|eu)\b")
# A money/amount token, optionally followed by a single UPPERCASE tax-class letter (a
# receipt's "1,69 B", "4,99 B", "€ 8,50", "-2,00"): the trailing letter slipped past the "no
# alpha" rule below, so the price leaked into the translation and was re-drawn over the
# original. Uppercase-only keeps lowercase measurements like "25 m" translatable as before.
_PRICE_TAX = re.compile(r"^[€$£]?\s*[-+]?\s*\d[\d.,]*\s*[A-Z]?$")


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", stripped)


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


def _is_nontranslatable(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if "://" in lowered or lowered.startswith("www.") or _URL_SUFFIX.search(lowered):
        return True
    if _PRICE_TAX.match(stripped):
        return True
    return not any(char.isalpha() for char in stripped)


def _is_continuation(previous_cell: dict[str, Any] | None, cell: dict[str, Any]) -> bool:
    """A wrapped continuation line starts at ~the same left margin as, and directly
    below, its element's previous line. A price in the right column or the next row of
    a receipt does not qualify, so genuinely repeated rows ("BONUS ...") still bind by
    position."""
    if previous_cell is None:
        return False
    prev, this = previous_cell.get("bbox") or {}, cell.get("bbox") or {}
    height = float(this.get("height") or 0.0) or float(prev.get("height") or 0.0)
    if height <= 0:
        return False
    drop = float(this.get("top") or 0.0) - float(prev.get("top") or 0.0)
    indent = abs(float(this.get("left") or 0.0) - float(prev.get("left") or 0.0))
    return 0 < drop <= 2.5 * height and indent <= 2.0 * height


def _near(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Two boxes are adjacent enough to be one wrapped element: x-ranges overlap and they are
    vertically close (stacked onto the next row), or y-ranges overlap and they are horizontally
    close (split across one line). A far-off box (an embedded image, a far column) is neither."""
    height = max(a[3] - a[1], b[3] - b[1], 1.0)
    x_overlap = min(a[2], b[2]) > max(a[0], b[0])
    y_overlap = min(a[3], b[3]) > max(a[1], b[1])
    y_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    x_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    stacked = x_overlap and y_gap <= 1.5 * height       # wrapped onto the next line
    same_line = y_overlap and x_gap <= 2.0 * height     # split across one line ("... nooit.")
    return stacked or same_line


def _is_icon_fragment(index: int, group_indices: list[int], cells: list[dict[str, Any]]) -> bool:
    """A group member is an icon-label fragment when its glyphs are much shorter than the line's
    other members (``_ICON_HEIGHT_RATIO``) and every token it carries is already covered by those
    others (it adds no new word). Height alone would be too eager; redundancy alone would drop a
    legitimate small repeat — together they pinpoint a badge label, regardless of where it sits."""
    others = [j for j in group_indices if j != index]
    if not others:
        return False
    tokens = _tokens(str(cells[index].get("text") or ""))
    if not tokens:
        return False
    height = float((cells[index].get("bbox") or {}).get("height") or 0.0)
    others_median = median([float((cells[j].get("bbox") or {}).get("height") or 0.0) for j in others])
    if height <= 0 or others_median <= 0 or height >= _ICON_HEIGHT_RATIO * others_median:
        return False
    other_tokens = {t for j in others for t in _tokens(str(cells[j].get("text") or ""))}
    return all(_token_score(token, other_tokens) > 0 for token in tokens)
