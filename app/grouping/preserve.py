"""The preserve policy: pure predicates deciding which text keeps its original pixels.

Each is a single text or geometry rule with a magic threshold or two, collected here so the
policy reads as a list instead of being scattered through the alignment algorithm. Consulted
by align (member translate flags, symbolic-label units, icon fragments), by the translator
(skip routing) and by the table renderer — no alignment state, plain cell/text/box data only.
This is what the ``preserve_heuristic_text`` request flag switches.
"""
from __future__ import annotations

import re
from statistics import median
from typing import Any

from app.grouping.tokens import _token_score
from app.grouping.tokens import _tokens
from app.grouping.units import _cell_box
from app.grouping.units import _near

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

# Two adjacent letters anywhere = at least one real (non-CJK) word candidate.
_LETTER_PAIR = re.compile(r"[^\W\d_]{2}", re.UNICODE)
_CJK_START, _CJK_END = "一", "鿿"


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


def _is_symbolic_label(text: str) -> bool:
    """UPPERCASE letters, but never two adjacent, or no letters at all: rating
    codes ('A-', 'A"'), column glyphs, symbol runs with a stray capital. Labels,
    not language. Any lowercase letter keeps the text out (the "25 m"
    measurement convention _PRICE_TAX follows), as does a CJK/Kana/Hangul char
    (a complete word). Consulted at UNIT level only — a whole unit of such
    tokens is a label row the translator can only hallucinate on (measured:
    the batch answered 'A " A- A-' with another line's sentence); a single
    such member inside a prose unit rides along as before."""
    stripped = str(text or "").strip()
    if not stripped:
        return True
    if any(_CJK_START <= char <= _CJK_END or "぀" <= char <= "ヿ" or "가" <= char <= "힯"
           for char in stripped):
        return False
    if any(char.islower() for char in stripped):
        return False
    return not _LETTER_PAIR.search(stripped)


def _is_icon_fragment(
    index: int,
    group_indices: list[int],
    cells: list[dict[str, Any]],
    allow_detached: bool = True,
) -> bool:
    """A group member is an icon/logo fragment when every token it carries is already covered by
    the line's other members (it adds no new word) AND it is set apart from them — either its
    glyphs are much shorter than theirs (``_ICON_HEIGHT_RATIO``, a tiny badge label sitting on the
    line, e.g. a "postnl" logo next to "Bezorging door PostNL") or it is a spatial outlier adjacent
    to none of them (a logo elsewhere in the image, e.g. the "NIKE" on a shoe pulled into a body
    line that names the product). Redundancy alone would drop a legitimate small repeat; height
    alone misses a logo larger than the text; the set-apart test marks it as not part of the line.

    ``allow_detached`` gates the spatial-outlier branch: the caller turns it off for ``|`` field
    rows, whose fields are meant to sit apart, so a far-left quantity is not mistaken for a logo."""
    others = [j for j in group_indices if j != index]
    if not others:
        return False
    tokens = _tokens(str(cells[index].get("text") or ""))
    if not tokens:
        return False
    other_tokens = {t for j in others for t in _tokens(str(cells[j].get("text") or ""))}
    if not all(_token_score(token, other_tokens) > 0 for token in tokens):
        return False  # carries a new word -> real content, keep it
    height = float((cells[index].get("bbox") or {}).get("height") or 0.0)
    others_median = median([float((cells[j].get("bbox") or {}).get("height") or 0.0) for j in others])
    if 0.0 < height < _ICON_HEIGHT_RATIO * others_median:
        return True
    if not allow_detached:
        return False
    box = _cell_box(cells[index])
    others_boxes = [_cell_box(cells[j]) for j in others]
    # A redundant cell sitting entirely ABOVE the others' top is a stray from the element above,
    # not a wrapped line of this one — text never wraps above its first line (the last line, by
    # contrast, legitimately sits below the rest, so we do NOT mirror this downward). ``_near`` can
    # still clip one edge of such a cell (a banner caption ~60px under the banner above it, sharing
    # its x-range), so this band test catches what adjacency misses.
    if box[3] <= min(b[1] for b in others_boxes):
        return True
    return not any(_near(box, b) for b in others_boxes)
