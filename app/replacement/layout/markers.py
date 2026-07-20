"""Bullets and enumerators: detect the marker, redraw or keep it, inset the text."""
from __future__ import annotations

from typing import Any
import re
import numpy as np
from statistics import median
from PIL import Image
from app.replacement.geometry import _ANGLE_DEADZONE_DEG
from app.replacement.layout.sweep import _ink_runs
from app.replacement.text.fit import EM_SPACE


# An ALPHANUMERIC enumerate marker at the START of a cell: "1."/"2)"/"(a)"/"A."/"ii.", a dotted
# multi-level section number "3.4.1"/"A.1.2" (two or more dot-joined segments — OCR merges those
# into the title's cell on ToC/outline rows, so without the redraw the erase swallows the number
# and the translation drops it), or a bracket citation marker "[1]"/"[23]" (a bibliography's
# item numbers — the erase wiped them while nothing redrew them, and every reference lost its
# number). OCR reads the digit/letter reliably, so we redraw it as text on
# the cell. A GLYPH bullet ("•"/"*"/"-"/"◊") is deliberately NOT matched here: glyphs route to the
# ink-scan path that keeps the original glyph in place, which renders the SAME glyph uniformly
# whether or not OCR happened to read it on a given line (mixed OCR recognition across a bullet
# list otherwise splits identical bullets over two paths). The trailing ``(?=\s)`` keeps a price
# ("1.69") or a word from matching; a single-dot decimal never matches (that IS a price), so a
# plain two-level "2.3 Title" row stays off this path — only "x.y."-and-deeper forms qualify.
_ENUMERATE_MARKER = re.compile(
    r"^\s*(\([A-Za-z0-9]{1,3}\)"
    r"|\[[A-Za-z0-9]{1,3}\]"
    r"|(?:[A-Za-z0-9]{1,3}\.){2,}[A-Za-z0-9]{0,3}[.)]?"
    r"|[A-Za-z0-9]{1,3}[.)])(?=\s)"
)


def _cell_marker(unit: dict[str, Any]) -> str | None:
    """The alphanumeric enumerate marker at the start of the cell, else ``None`` (no marker, or a glyph
    bullet that the ink-scan path handles). The VLM's captured marker counts only when it both leads the
    source AND is itself an enumerate form — otherwise we fall back to the pattern OCR put there."""
    source = str(unit.get("source_text") or "")
    bullet_marker = str(unit.get("bullet_marker") or "")
    if bullet_marker and source.lstrip().startswith(bullet_marker) and _ENUMERATE_MARKER.match(f"{bullet_marker} "):
        return bullet_marker
    match = _ENUMERATE_MARKER.match(source)
    return match.group(1) if match else None

def _prepend_marker(units: list[dict[str, Any]], marker: str) -> list[dict[str, Any]]:
    """A shallow copy of ``units`` with ``marker`` prepended to the first translatable line when the
    translation dropped it (idempotent), so the redrawn line keeps its "1."/"(a)" at the cell's place."""
    out = list(units)
    for index, unit in enumerate(out):
        text = str(unit.get("translated_text") or "").strip()
        if len(text) <= 1:
            continue
        if not text.lstrip().startswith(marker):
            copy = dict(unit)
            copy["translated_text"] = f"{marker} {text}"
            out[index] = copy
        break
    return out


# A numbered BADGE or leader arrow the VLM transcribes into a heading's hint line ("2 Tile
# Tabs", "4 → Go To Arrow") while the badge graphic itself stays intact in the image: the
# unit's own members never printed the token, so rendering the translation would print it
# AGAIN next to the badge. Strip such leading tokens from header/title units when NO member
# carries them. Self-limiting twice over: a printed enumerator is a member (stays), and a
# printed number OCR missed is not erased either, so stripping stays visually right. The
# header/title gate keeps body prose out (a translator may legitimately digitise a written
# number there).
_LEAD_BADGE_TOKEN = re.compile(r"^\s*([0-9]{1,2}[.)]?|[→←↑↓])(?:\s+|$)")


def _strip_unprinted_lead(translated: str, unit: dict[str, Any]) -> str:
    """``translated`` without leading badge-number/arrow tokens its members never printed."""
    if unit.get("level") not in ("header", "title") or unit.get("bullet"):
        return translated
    member_text = " ".join(str(m.get("text") or "") for m in (unit.get("members") or []))
    member_tokens = set(re.findall(r"[0-9]{1,4}|[→←↑↓]", member_text))
    out = translated
    while True:
        match = _LEAD_BADGE_TOKEN.match(out)
        if not match:
            return out
        token = match.group(1).rstrip(".)")
        remainder = out[match.end():]
        if not remainder.strip() or token in member_tokens:
            return out
        out = remainder


# Footnote-style lead marker ("*Equal contributions.", "†note") — a symbol of this class
# directly before a word. The hint parse can eat a leading "*" as markdown noise (and the
# translator may drop the symbol), after which nothing re-draws it while the erase wipes the
# printed one: the footnote loses its mark. The OCR member text is the PRINT evidence — when
# it leads with such a marker and the translation lost every leading symbol, the printed
# marker is re-added. No-op when the translation still carries a marker of its own (also a
# VLM-lookalike variant: the class covers the suit/star swaps).
_LEAD_MARKER_CHARS = "*†‡§¶⋆✦◊⋄♠♣♥♦♡♢"
# Print side: the marker must sit DIRECTLY on the word (footnote typography, "*Equal") — a
# detached "* word" is a bullet, which has its own paths.
_PRINTED_LEAD_MARKER = re.compile(rf"^([{re.escape(_LEAD_MARKER_CHARS)}])(?=\w)")


def _restore_printed_lead_marker(translated: str, unit: dict[str, Any]) -> str:
    """``translated`` with the unit's PRINTED lead marker re-added when the translation lost it
    (see _PRINTED_LEAD_MARKER). The marker comes from the first member's OCR text — print
    evidence, not the hint — and is attached the way the print attaches it (no space)."""
    members = unit.get("members") or []
    first_text = str(members[0].get("text") or "").lstrip() if members else ""
    printed = _PRINTED_LEAD_MARKER.match(first_text)
    if not printed:
        return translated
    out = str(translated or "")
    # The translation kept a marker of its own (attached or spaced, incl. a lookalike variant).
    if not out.strip() or out.lstrip()[0] in _LEAD_MARKER_CHARS:
        return out
    return printed.group(1) + out


# A section number set in its OWN cell ("2", "3.1", "A.2") ahead of the heading it numbers. The
# translator drops it about as often as it keeps it — it reads as a stray token, not as prose —
# and once the cell is erased with the rest of the line a dropped number is simply gone from the
# page. Re-add it from the print, like the footnote marker above, but spaced: it numbers the
# heading, it is not attached to its first word.
# Bare and unambiguous: "2", "3.1", "A.2", "4." — an opening bracket must be closed, so an OCR
# fragment like "(60" (the head of "(60 reviews)") can never pass for a section number.
_NUMBER_CELL = re.compile(r"^(?:\d+|[A-Za-z]\.?\d+)(?:[.\-]\d+)*[.:)]?$")
# Only a HEADING is numbered this way. On body text a leading number cell is a quantity, a year
# or a table figure, and the column machinery owns it (see grouping/field_geometry).
_NUMBERED_LEVELS = {"title", "header"}


def _heading_number(unit: dict[str, Any]) -> str | None:
    """The section number this heading prints in its own leading cell, or ``None``. A unit that
    is only a number (a page number, a lone table figure) has no heading to number, so it never
    qualifies."""
    if str(unit.get("level") or "") not in _NUMBERED_LEVELS:
        return None
    members = unit.get("members") or []
    if len(members) < 2:
        return None
    number = str(members[0].get("text") or "").strip()
    return number if _NUMBER_CELL.match(number) else None


# Channel distance at which the number's ground is a DIFFERENT surface from the text it numbers,
# not the same paper sampled twice. A step badge measures far beyond it (white digit on a
# saturated disc vs black text on white); scanner noise and paper tint stay well under.
_BADGE_GROUND_DELTA = 60


def _heading_number_is_badge(base_px: Any, unit: dict[str, Any]) -> bool:
    """True when the heading's leading number sits on its OWN ground — a step badge (a white
    digit in a filled disc), not a number typeset on the same surface as its title.

    A badge is artwork: erasing it fills the disc with its own colour and the re-drawn digit
    lands as body-coloured text on top, which destroys the design. Print numbering (a paper's
    "3 Model Architecture") shares the page ground and re-draws cleanly, so only the pixels can
    tell the two apart.
    """
    if _heading_number(unit) is None:
        return False
    members = unit.get("members") or []
    grounds = [_border_colour(base_px, m.get("bbox") or {}) for m in members[:2]]
    if any(g is None for g in grounds):
        return False
    return max(abs(int(a) - int(b)) for a, b in zip(*grounds)) >= _BADGE_GROUND_DELTA


def _border_colour(base_px: Any, bbox: dict[str, Any]) -> tuple[int, int, int] | None:
    """Median colour of the ring just outside ``bbox`` — the ground the cell is printed on."""
    height, width = base_px.shape[:2]
    x0, y0 = int(bbox.get("left", 0)), int(bbox.get("top", 0))
    x1 = x0 + int(bbox.get("width") or 0)
    y1 = y0 + int(bbox.get("height") or 0)
    pad = 3
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    patch = base_px[y0:y1, x0:x1]
    ring = np.concatenate([
        patch[0].reshape(-1, patch.shape[-1]), patch[-1].reshape(-1, patch.shape[-1]),
        patch[:, 0].reshape(-1, patch.shape[-1]), patch[:, -1].reshape(-1, patch.shape[-1]),
    ])
    return tuple(int(v) for v in np.median(ring, axis=0)[:3])


def _strip_heading_number(translated: str, unit: dict[str, Any]) -> str:
    """``translated`` without the leading section number — for a badge, whose printed digit
    stays in the image and would otherwise render a second time."""
    number = _heading_number(unit)
    if not number:
        return translated
    out = str(translated or "")
    match = re.match(rf"^\s*{re.escape(number)}\s*", out)
    return out[match.end():] if match else out


def _widen_heading_number_gap(translated: str, unit: dict[str, Any]) -> str:
    """``translated`` with an EM space between the section number and the heading it numbers.

    Print sets that separator one em wide; joined into the line as an ordinary space it renders
    four times tighter (measured on a paper: 28px in the source, 7px re-drawn), which reads as
    the number having drifted into its title."""
    number = _heading_number(unit)
    if not number:
        return translated
    out = str(translated or "")
    match = re.match(rf"^(\s*{re.escape(number)})(\s+)(?=\S)", out)
    if not match:
        return out
    return out[: match.end(1)] + EM_SPACE + out[match.end(2):]


def _restore_printed_lead_number(translated: str, unit: dict[str, Any]) -> str:
    """``translated`` with the unit's printed leading section NUMBER re-added when the
    translation lost it. Only on a heading whose number is a cell of its own with more text
    after it, and only when the number appears nowhere in the translation — so a page number, a
    table figure and a translation that moved the number are all left alone."""
    number = _heading_number(unit)
    if not number:
        return translated
    out = str(translated or "").strip()
    if not out or re.search(rf"(?<!\d){re.escape(number.rstrip('.:)'))}(?!\d)", out):
        return str(translated or "")
    return f"{number}{EM_SPACE}{out}"


# A glyph marker ("•"/"*"/"-"/"◊"...) that may lead the translated text. The ink-scan path keeps the
# ORIGINAL glyph in the image, so a glyph still in the text would render twice — strip one leading glyph
# (plus its space) before the inset. Alphanumeric markers take the redraw path (_prepend_marker) instead.
_LEADING_GLYPH = re.compile(r"^\s*[•·∙●○◦‣⁃*–—-]\s+")

def _strip_leading_glyph(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A shallow copy of ``units`` with a single leading glyph marker removed from the first translatable
    line, so the ink-scan path (which keeps the original glyph in place) does not render it twice."""
    out = list(units)
    for index, unit in enumerate(out):
        text = str(unit.get("translated_text") or "").strip()
        if len(text) <= 1:
            continue
        stripped = _LEADING_GLYPH.sub("", text, count=1)
        if stripped != text:
            copy = dict(unit)
            copy["translated_text"] = stripped
            out[index] = copy
        break
    return out

def _bullet_geometry(base: Image.Image, frame: tuple, angle: float) -> tuple[float, float] | None:
    """For a bullet line, return (text_start_x, bullet_y_center) — where the text starts (past
    the leading bullet glyph and its gap) and the bullet glyph's vertical centre. None when no
    clear glyph+gap is found near the line's left edge (or the line is tilted, where the
    axis-aligned scan is unreliable).
    Scans the line's vertical band from a margin LEFT of the plane edge, because the OCR cell
    box's left wanders relative to the fixed bullet (sometimes landing right of it). The original
    bullet stays in the image; the caller starts the erase/anchor at the text and centres the
    re-rendered text on the bullet. Triggered only when the VLM flagged the unit as a bullet
    item, so a stray short first word can't be mistaken for a bullet."""
    if abs(angle) > _ANGLE_DEADZONE_DEG:
        return None
    _, _, xmin, xmax, ymin, ymax = frame
    line_h = max(1, int(round(ymax - ymin)))
    x0 = max(0, int(round(xmin - 1.5 * line_h)))           # the bullet may sit left of the box
    x1 = int(round(xmin + 0.6 * (xmax - xmin)))
    y0, y1 = int(round(ymin)), int(round(ymax))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    arr = np.asarray(base.crop((x0, y0, x1, y1)).convert("L")).astype(int)
    bg = int(np.median(arr))
    mask = np.abs(arr - bg) > 60
    ink = mask.any(axis=0)                                 # columns holding a high-contrast pixel
    runs = _ink_runs(ink)
    # Find the bullet: the first glyph-sized run that is followed by a clear gap and then the
    # text. A bullet is small in INK HEIGHT, not necessarily narrow: a dot/circle/diamond is
    # narrow, a dash is wide but flat. The width cap alone rejected a dash or not depending on
    # a few px of quad-height wobble (the 0.4x threshold sat right on a dash's width), so wide
    # runs are accepted too when their ink rows span only a thin band — which still rejects
    # adjacent layout ink (a coloured panel/book edge next to the column is line-TALL); the
    # VLM flag guarantees a real bullet is present.
    min_width = max(2.0, 0.06 * line_h)  # a 1px anti-alias speck is not a bullet
    for i in range(len(runs) - 1):
        width = runs[i][1] - runs[i][0] + 1
        gap = runs[i + 1][0] - runs[i][1] - 1
        if width < min_width or gap < 0.12 * line_h:
            continue
        rows = np.where(mask[:, runs[i][0]:runs[i][1] + 1].any(axis=1))[0]
        row_span = (rows.max() - rows.min() + 1) if len(rows) else 0
        # A bullet glyph is SMALL in a mix a letter never is: compact in both dimensions (a
        # dot/square/diamond at ~half the line height or less) or wide-but-flat (a dash). A
        # letter keeps one dimension at ~0.6x+ — so BOTH rules cap the ink height: without the
        # compact rule's height cap a narrow CAP-HEIGHT letter ("I"/"l"/"1", ~0.1 x 0.8) counted
        # as a dot, anchored the erase after itself, and survived as a stray "|" before the
        # re-rendered line.
        compact = width <= 0.55 * line_h and row_span <= 0.55 * line_h
        dash_like = width <= 0.9 * line_h and row_span <= 0.35 * line_h
        if compact or dash_like:
            text_start = float(x0 + runs[i + 1][0])
            # A real bullet sits AT the line's left edge: its glyph (<=0.9x line height) plus gap
            # never puts the text start more than ~1.2x line height past the OCR box's left. A
            # match deeper into the line is a narrow letter/digit inside the TEXT (the VLM flags
            # ToC/numbered rows as bullets with no glyph present) — anchoring there would leave
            # the words left of it standing and squeeze the translation into the remainder. Any
            # later run sits deeper still, so give up rather than scan on.
            if text_start > xmin + 1.5 * line_h:
                return None
            bullet_y = y0 + (rows.min() + rows.max()) / 2.0 if len(rows) else (y0 + y1) / 2.0
            return text_start, float(bullet_y)
    return None
