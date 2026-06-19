"""Small, self-contained grouping heuristics.

Pure predicates the aligner (and the translator) consult — each a single text or geometry rule
with a magic threshold or two, collected here so the rules read as a list instead of being
scattered through the alignment algorithm. No alignment state: every function takes plain
cell/text/box data, so this module has no dependency on ``align``.
"""
from __future__ import annotations

import re
from statistics import median
from typing import Any

from app.grouping.tokens import _token_score
from app.grouping.tokens import _tokens


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


def _cell_box(cell: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = cell.get("bbox") or {}
    left = float(bbox.get("left") or 0.0)
    top = float(bbox.get("top") or 0.0)
    return left, top, left + float(bbox.get("width") or 0.0), top + float(bbox.get("height") or 0.0)


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
