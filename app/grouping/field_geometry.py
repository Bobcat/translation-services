"""Geometry-derived field boundaries: spot where the VLM missed a rule-3/4 column ``|``.

The VLM's ``|`` is a SEMANTIC field marker, and most boundaries sit at normal spacing (the
quantity column right next to a product name) — geometry cannot and need not reproduce those.
But a SPATIALLY separated column — a label and a far-right amount, two touching columns OCR read
with a clear gap — is a boundary the VLM intermittently omits, and that gap is geometrically
obvious. Calibration against the VLM's own ``|`` (its rule-3/4 output) showed: prose word gaps
sit ~0.2 char-widths, a real column gap ~2+; the cleanest reference is the line's own smallest
gap (a measured word space in that exact font) when the label is multi-cell, else the glyph scale
(char width), so a WIDE header font does not split its own ``JOUW | VOORDEEL``.

Observe-only for now: this computes the adjusted hint lines for inspection in the workbench and
does NOT feed translation. Only a single-line unit with >= 2 cells is a column candidate; a
multi-line (wrapped) unit is prose and is skipped.
"""
from __future__ import annotations

import re
from statistics import median
from typing import Any

from app.grouping.units import TranslationUnit
from app.grouping.units import UnitMember

# A gap counts as a column boundary when it clears this multiple of the reference space.
_MIN_GAP_RATIO = 1.8     # reference = the line's smallest word gap (>= 3 cells; font-agnostic)
_CHAR_WIDTH_RATIO = 2.0  # reference = average glyph width (2 cells; no measured word gap available)
# Members are on one physical line when their vertical centres spread less than this of the height.
_SINGLE_LINE_SPREAD = 0.6
# A section number set in its own cell ("2", "3.1") a wide space before its title. The separation
# is typographic, not a column: number and title are ONE heading. Left to the gap rule it lands on
# a knife edge — the 2-cell threshold scales with the average glyph width, so on one page
# "2 Background" stays whole while "3 Model Architecture", with the same 27px gap, splits purely
# because its longer title lowers that average. The split half then renders the title as a lone
# field and leaves the digit's ORIGINAL pixels standing beside it, 3-4px off the re-drawn
# baseline; the unsplit half draws the number inline and collapses the gap to a word space. Same
# page, two different artifacts. A heading's leading enumerator therefore never opens a column.
_HEADING_LEVELS = {"title", "header"}
_LEADING_ENUMERATOR = re.compile(r"^\(?\[?[A-Za-z]?\d+(?:[.\-]\d+)*[.):\]]?$")


def geometry_adjusted_hints(
    units: list[TranslationUnit], hint_units: list[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return ``(adjusted_hint_units, changes)``. ``adjusted_hint_units`` is parallel to
    ``hint_units`` with a ``|`` injected on the lines where geometry found a column the VLM
    missed; unchanged lines are copied verbatim. ``changes`` lists the touched lines (raw +
    adjusted + the detected columns) for inspection."""
    adjusted = list(hint_units)
    changes: list[dict[str, Any]] = []
    for unit in units:
        index = unit.hint_index
        if index is None or not (0 <= index < len(hint_units)):
            continue
        line = hint_units[index]
        if "|" in line:  # the VLM already marked this row's fields — leave it
            continue
        columns = _column_split(unit.members, level=unit.level)
        if columns is None:
            continue
        injected = _inject_pipes(line, columns)
        column_texts = [" ".join(m.text for m in col).strip() for col in columns]
        adjusted[index] = injected if injected is not None else " | ".join(column_texts)
        changes.append({
            "hint_index": index,
            "raw": line,
            "adjusted": adjusted[index],
            "columns": column_texts,
            "mapped_into_vlm_line": injected is not None,
        })
    return adjusted, changes


def _column_split(
    members: list[UnitMember], *, level: str | None = None
) -> list[list[UnitMember]] | None:
    """Group a unit's members into >= 2 spatial columns, or None when it is not a column row.
    Only a single physical line is considered (a multi-line unit is wrapped prose). On a heading
    a leading enumerator is part of the heading, never a column of its own (see
    ``_LEADING_ENUMERATOR``); on body text it still can be (a receipt's quantity column)."""
    placed = [m for m in members if m.bbox]
    if len(placed) < 2 or not _is_single_line(placed):
        return None
    # A column `|` only changes the outcome when there is text to translate on a side: it lets the
    # translation keep a number in place and reflow the label in its own column. An all-numeric row
    # (two temperatures "15° 15", a quantity and a price) has nothing to translate, so splitting it
    # is pointless — and would surface a meaningless "15° | 15".
    if not any(m.translate for m in placed):
        return None
    ordered = sorted(placed, key=lambda m: m.bbox["left"])
    gaps = [
        ordered[i + 1].bbox["left"] - (ordered[i].bbox["left"] + ordered[i].bbox["width"])
        for i in range(len(ordered) - 1)
    ]
    positive = [g for g in gaps if g > 0]
    if not positive:
        return None
    if len(ordered) >= 3:
        reference = min(positive)
        threshold = _MIN_GAP_RATIO * reference
    else:
        chars = sum(len(re.sub(r"[^a-z0-9]", "", m.text.lower())) for m in ordered) or 1
        char_width = sum(m.bbox["width"] for m in ordered) / chars
        threshold = _CHAR_WIDTH_RATIO * char_width
    heading_lead = (
        str(level or "") in _HEADING_LEVELS
        and bool(_LEADING_ENUMERATOR.match(str(ordered[0].text or "").strip()))
    )
    columns: list[list[UnitMember]] = [[ordered[0]]]
    for index, (member, gap) in enumerate(zip(ordered[1:], gaps)):
        if gap >= threshold and not (heading_lead and index == 0):
            columns.append([member])
        else:
            columns[-1].append(member)
    return columns if len(columns) >= 2 else None


def _is_single_line(members: list[UnitMember]) -> bool:
    centres = [m.bbox["top"] + m.bbox["height"] / 2 for m in members]
    height = median(m.bbox["height"] for m in members) or 1.0
    return (max(centres) - min(centres)) <= _SINGLE_LINE_SPREAD * height


def _inject_pipes(line: str, columns: list[list[UnitMember]]) -> str | None:
    """Insert ``|`` into the VLM hint line at each column boundary. Anchor on the LEFT column's
    last member (its longer, more reliable text — the right column may start with an OCR symbol
    the VLM read differently, e.g. ``×x8001`` vs ``xx8001``) and cut at the gap that follows it.
    Returns None when an anchor cannot be located (then the caller shows the cell-derived split)."""
    out = line
    for index in range(len(columns) - 1, 0, -1):  # right to left so earlier cuts stay valid
        anchor = (columns[index - 1][-1].text or "").strip()
        cut = _gap_after(out, anchor)
        if cut is None:
            return None
        out = out[:cut] + " |" + out[cut:]
    return out


def _gap_after(line: str, anchor: str) -> int | None:
    """Index of the whitespace gap just after ``anchor`` (skipping trailing punctuation), where
    the column ``|`` belongs. None when the anchor is not found or nothing follows it."""
    end = _anchor_end(line, anchor)
    if end is None:
        return None
    i = end
    while i < len(line) and not line[i].isspace():  # step over a trailing '.', ':' etc.
        i += 1
    return i if i < len(line) else None


def _anchor_end(line: str, anchor: str) -> int | None:
    line_norm, line_map = _alnum_map(line)
    anchor_norm, _ = _alnum_map(anchor)
    if not anchor_norm:
        return None
    at = line_norm.find(anchor_norm)
    if at < 0:
        return None
    return line_map[at + len(anchor_norm) - 1] + 1  # original index past the anchor's last glyph


def _alnum_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for i, char in enumerate(text):
        if char.isalnum():
            chars.append(char.lower())
            positions.append(i)
    return "".join(chars), positions
