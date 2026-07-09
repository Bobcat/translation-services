"""Bullets and enumerators: detect the marker, redraw or keep it, inset the text."""
from __future__ import annotations

from typing import Any
import re
import numpy as np
from statistics import median
from PIL import Image
from app.replacement.geometry import _ANGLE_DEADZONE_DEG
from app.replacement.layout.sweep import _ink_runs


# An ALPHANUMERIC enumerate marker at the START of a cell: "1."/"2)"/"(a)"/"A."/"ii.". OCR reads the
# digit/letter reliably, so we redraw it as text on the cell. A GLYPH bullet ("•"/"*"/"-"/"◊") is
# deliberately NOT matched here: glyphs route to the ink-scan path that keeps the original glyph in
# place, which renders the SAME glyph uniformly whether or not OCR happened to read it on a given line
# (mixed OCR recognition across a bullet list otherwise splits identical bullets over two paths). The
# trailing ``(?=\s)`` keeps a price ("1.69") or a word from matching.
_ENUMERATE_MARKER = re.compile(
    r"^\s*(\([A-Za-z0-9]{1,3}\)|[A-Za-z0-9]{1,3}[.)])(?=\s)"
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
    clear glyph+gap is found (or the line is tilted, where the axis-aligned scan is unreliable).
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
        dot_like = width <= 0.4 * line_h
        dash_like = width <= 0.9 * line_h and row_span <= 0.35 * line_h
        if dot_like or dash_like:
            bullet_y = y0 + (rows.min() + rows.max()) / 2.0 if len(rows) else (y0 + y1) / 2.0
            return float(x0 + runs[i + 1][0]), float(bullet_y)
    return None
