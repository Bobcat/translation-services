"""Text-layer cells for born-digital pages (phase 1, design doc §6/§8).

Extracts PyMuPDF text lines as the pipeline's ``cells`` — the same dict shape
OCR produces (id, text, bbox, confidence, polygon), in render-pixel space at
the analysis dpi — plus exact style metadata OCR cannot give (size_pt, font
family, weight, italic, color). Align and translation consume cells
source-agnostically; the style fields ride along for the cohort-consistent
sizing of slice B and are inert until then.

Protection policy (deliberately conservative for phase 1): a line containing a
SUBSTANTIAL protected span — math fonts, symbol/icon fonts, private-use or
unmapped glyphs, longer than a marker glyph — is dropped whole, as is any
non-horizontal line. Dropped lines keep their original pixels: the bitmap
backend only erases where units place text, so formulas, icons and rotated
margin text survive untouched. The cost is that prose with inline math stays
untranslated (it reads as unchanged in the benchmark); that limit is named in
the page summary via the drop counters.

A SHORT protected span (<= 2 chars: a bullet "•", an arrow, a dingbat) is a
list/decoration marker riding inline on a prose line; it is stripped from the
cell text and bbox instead of dropping the line — marker rendering is the
existing bullet machinery's job (VLM hint -> unit.bullet), not the cell's.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import re
from typing import Any

import pymupdf

# Two kinds of protected fonts, with different line policies. MATH fonts
# (Computer Modern & friends, AMS, anything *Math*) always drop the whole
# line: formulas shatter into tiny spans, and a half-stripped formula line is
# garbage for the translator AND erases pixels it cannot redraw. SYMBOL fonts
# (bullets, dingbats) may ride inline as ≤2-char markers on prose lines and
# are then stripped. Matched against the base font name with the subset
# prefix (``ABCDEF+``) stripped.
_MATH_FONT_RE = re.compile(r"^(CM(?!R)[A-Z0-9]*|CMR\d|MSAM|MSBM|.*Math)", re.IGNORECASE)
_SYMBOL_FONT_RE = re.compile(r"^(Symbol|ZapfDingbats|Wingdings|Webdings)", re.IGNORECASE)
_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")
_BOLD_NAME_RE = re.compile(r"bold|black|heavy|semibold|demibold", re.IGNORECASE)

_FLAG_ITALIC = 1 << 1
_FLAG_BOLD = 1 << 4

# Private-use-area ranges plus the replacement char (an unmapped CID).
_PUA_RANGES = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))


@dataclass(frozen=True)
class PageCells:
    cells: list[dict[str, Any]]
    dropped_protected_lines: int = 0
    dropped_rotated_lines: int = 0
    stripped_marker_spans: int = 0

# A protected span at most this long (stripped) is an inline marker, not content.
_MARKER_MAX_CHARS = 2


@dataclass
class _LineStats:
    protected: int = 0
    rotated: int = 0
    stripped_markers: int = 0
    cells: list[dict[str, Any]] = field(default_factory=list)


class PageTextExtractor:
    """Holds one document open and extracts per-page text-layer cells in
    render-pixel space at the given dpi. Use as a context manager, like
    ``PageRasterizer`` — the two run side by side over the same document."""

    def __init__(self, path: Path, *, dpi: int) -> None:
        self._path = Path(path)
        self._scale = float(dpi) / 72.0
        self._doc: pymupdf.Document | None = None

    def __enter__(self) -> "PageTextExtractor":
        self._doc = pymupdf.open(str(self._path))
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def cells_for_page(self, page_index: int) -> PageCells:
        if self._doc is None:
            raise RuntimeError("PageTextExtractor used outside its context")
        stats = _LineStats()
        # rawdict, not dict: the per-glyph boxes are what make the cell bbox
        # ink-honest. A line's leading space can carry a glyph box that starts
        # at the LIST MARKER's x (a bullet followed by a tab-wide space), so a
        # span-level bbox silently swallows the marker — the erase pass then
        # clips a glyph it does not redraw. Non-space glyph union avoids that.
        page_dict = self._doc[int(page_index)].get_text("rawdict")
        for block in page_dict.get("blocks") or []:
            if block.get("type") != 0:
                continue
            for line in block.get("lines") or []:
                self._append_line(line, stats)
        return PageCells(
            cells=stats.cells,
            dropped_protected_lines=stats.protected,
            dropped_rotated_lines=stats.rotated,
            stripped_marker_spans=stats.stripped_markers,
        )

    def _append_line(self, line: dict[str, Any], stats: _LineStats) -> None:
        spans = [span for span in (line.get("spans") or []) if _span_text(span).strip()]
        if not spans:
            return
        direction = line.get("dir") or (1.0, 0.0)
        if abs(float(direction[0])) < 0.99:  # rotated/vertical: keep original pixels
            stats.rotated += 1
            return
        kinds = [(span, _span_protection(span)) for span in spans]
        if any(
            kind == "math"
            or (kind == "symbol" and len(_span_text(span).strip()) > _MARKER_MAX_CHARS)
            for span, kind in kinds
        ):
            stats.protected += 1  # substantial protected content (a formula): keep pixels
            return
        markers = [span for span, kind in kinds if kind == "symbol"]
        marker_text = ""
        if markers:  # inline markers only (bullets, dingbats): strip, keep the prose
            stats.stripped_markers += len(markers)
            marker_text = " ".join(_span_text(span).strip() for span in markers).strip()
            marker_ids = {id(span) for span in markers}
            spans = [span for span in spans if id(span) not in marker_ids]
            if not spans:
                return
        text = " ".join(_span_text(span).strip() for span in spans)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return
        bbox = _glyph_bbox(spans) or [float(v) for v in (line.get("bbox") or (0, 0, 0, 0))]
        left, top = bbox[0] * self._scale, bbox[1] * self._scale
        width = max(1.0, (bbox[2] - bbox[0]) * self._scale)
        height = max(1.0, (bbox[3] - bbox[1]) * self._scale)
        # Dominant span by GLYPH AREA (chars x size^2), not by char count: a KPI
        # line "$865 million" is a large amount plus a small unit-word, and the
        # small word has more characters — the size that visually carries the
        # line is the big one.
        dominant = max(spans, key=lambda span: len(_span_text(span).strip()) * float(span.get("size") or 0.0) ** 2)
        font_name = _SUBSET_PREFIX_RE.sub("", str(dominant.get("font") or ""))
        flags = int(dominant.get("flags") or 0)
        bold = bool(flags & _FLAG_BOLD) or bool(_BOLD_NAME_RE.search(font_name))
        cell: dict[str, Any] = {
            "id": len(stats.cells) + 1,
            "text": text,
            "bbox": {
                "left": round(left, 2),
                "top": round(top, 2),
                "width": round(width, 2),
                "height": round(height, 2),
            },
            "confidence": 1.0,
            "polygon": [
                {"x": round(left, 2), "y": round(top, 2)},
                {"x": round(left + width, 2), "y": round(top, 2)},
                {"x": round(left + width, 2), "y": round(top + height, 2)},
                {"x": round(left, 2), "y": round(top + height, 2)},
            ],
            # Ground-truth style from the PDF. size_px (the em size in image
            # pixels at the analysis dpi) drives rendering; the rest rides
            # along for the remaining slice-B steps (weight, color, italic).
            "source": "pdf_text_layer",
            "size_pt": round(float(dominant.get("size") or 0.0), 2),
            "size_px": round(float(dominant.get("size") or 0.0) * self._scale, 2),
            "font_name": font_name,
            "font_weight": 700 if bold else 400,
            "italic": bool(flags & _FLAG_ITALIC),
            "color": f"#{int(dominant.get('color') or 0):06x}",
        }
        if marker_text:
            # The line carried a stripped list/decoration marker: ground truth
            # for the unit's bullet flag, stronger than any hint label.
            cell["marker"] = marker_text
        stats.cells.append(cell)


def _span_text(span: dict[str, Any]) -> str:
    """rawdict spans carry chars, not a text field."""
    chars = span.get("chars")
    if chars is not None:
        return "".join(str(ch.get("c") or "") for ch in chars)
    return str(span.get("text") or "")


def _glyph_bbox(spans: list[dict[str, Any]]) -> list[float] | None:
    """Union of the NON-SPACE glyph boxes of the kept spans: the box of the ink
    the cell actually owns. Whitespace glyphs are excluded because a leading
    space's glyph box can reach back to a stripped marker's position, and the
    erase pass must never cover a glyph it will not redraw."""
    boxes = [
        ch.get("bbox")
        for span in spans
        for ch in (span.get("chars") or [])
        if str(ch.get("c") or "").strip() and ch.get("bbox")
    ]
    if not boxes:
        boxes = [span.get("bbox") for span in spans if span.get("bbox")]
    if not boxes:
        return None
    return [
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    ]


def _span_protection(span: dict[str, Any]) -> str | None:
    """"math" (always drop the line), "symbol" (marker-strippable), or None."""
    font_name = _SUBSET_PREFIX_RE.sub("", str(span.get("font") or ""))
    if _MATH_FONT_RE.match(font_name):
        return "math"
    if _SYMBOL_FONT_RE.match(font_name):
        return "symbol"
    text = str(span.get("text") or "")
    suspect = sum(1 for ch in text if _is_unmappable(ch))
    if suspect > 0 and suspect >= max(1, len(text.strip()) // 2):
        return "symbol"  # PUA/unmapped glyphs behave like icon markers
    return None


def _is_protected_span(span: dict[str, Any]) -> bool:
    """Kept for the classifier unit tests: any protection kind counts."""
    return _span_protection(span) is not None


def _is_unmappable(ch: str) -> bool:
    code = ord(ch)
    if ch == "�":
        return True
    return any(low <= code <= high for low, high in _PUA_RANGES)
