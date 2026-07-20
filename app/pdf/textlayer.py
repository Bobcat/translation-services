"""Text-layer cells for born-digital pages (phase 1, design doc §6/§8).

Extracts PyMuPDF text lines as the pipeline's ``cells`` — the same dict shape
OCR produces (id, text, bbox, confidence, polygon), in render-pixel space at
the analysis dpi — plus exact style metadata OCR cannot give (size_pt, font
family, weight, italic, color). Align and translation consume cells
source-agnostically; the style fields ride along for the cohort-consistent
sizing of slice B and are inert until then.

Protection policy: a formula-dominated line, a line with substantial symbol/
icon/unmapped content, and any non-horizontal line are dropped whole; dropped
lines keep their original pixels (the bitmap backend only erases where units
place text). A prose line whose math is a clear minority becomes a cell with
inline-math ISLANDS (islands design doc): each math run turns into a ⟦Mn⟧
placeholder in the text plus a recorded pixel bbox, rides through translation
as an opaque token, and is transplanted back as source pixels by the render.

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
# always drop the whole line: formulas shatter into tiny spans, and a
# half-stripped formula line is garbage for the translator AND erases pixels
# it cannot redraw. Only the genuinely mathematical faces count — CMMI (math
# italic), CMSY/CMBSY (symbols), CMEX (extensions), the AMS MSAM/MSBM sets,
# and OpenType *Math* fonts. The Computer Modern TEXT faces (CMR roman, CMBX
# bold, CMTI italic, CMTT typewriter, CMSS sans, CMSL slanted, CMCSC caps)
# are a document's body type in classic LaTeX and must extract as prose — a
# pure-CM document yielded 0 cells under the old CM* blanket (islands design
# doc, phase 1). A CMR digit run INSIDE a formula still drops with its line:
# the trigger is the true-math span beside it. SYMBOL fonts (bullets,
# dingbats) may ride inline as ≤2-char markers on prose lines and are then
# stripped. Matched against the base font name with the subset prefix
# (``ABCDEF+``) stripped.
_MATH_FONT_RE = re.compile(r"^(CMMI|CMSY|CMBSY|CMEX|MSAM|MSBM|.*Math)", re.IGNORECASE)
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
    island_seq: int = 0  # page-unique ⟦Mn⟧ numbering (units merge cells, ids must not collide)
    cells: list[dict[str, Any]] = field(default_factory=list)


# Inline-math islands (islands design doc, phase 2). A line whose math is a clear
# MINORITY becomes a cell: each maximal run of math spans (plus absorbed adjacent
# letterless spans — the CMR "= 8" beside a CMMI "h") is replaced by a ⟦Mn⟧
# placeholder in the text and recorded as an island with its pixel bbox; the render
# transplants the source pixels there. Above the share cap, or without enough prose
# words around it, the line is formula-dominated and keeps today's whole-line drop:
# a display equation must stay source pixels entirely.
_ISLAND_MAX_MATH_SHARE = 0.35
_ISLAND_MIN_PROSE_WORDS = 3
_ISLAND_SCRIPT_SIZE_RATIO = 0.85  # spans this much smaller than the line = sub/superscript
_ISLAND_TOKEN_RE = re.compile(r"⟦M(\d+)⟧")

# The Computer Modern TEXT faces collapse to one family for the dominance test:
# in a pure-CM document the body (CMR), headings (CMBX) and theorem italic (CMTI)
# are one typographic voice, and an island line's prose in any of them counts as
# body prose.
_CM_TEXT_RE = re.compile(r"^CM(R|BX|TI|TT|SS|SL|CSC)", re.IGNORECASE)


def _text_family(span: dict[str, Any]) -> str:
    """Normalized family key for the dominance test: subset prefix stripped, CM
    text faces collapsed, otherwise the leading alphabetic run of the base name
    (``NimbusRomNo9L-Regu``/``-Medi`` -> ``NIMBUSROMNO``)."""
    name = _SUBSET_PREFIX_RE.sub("", str(span.get("font") or ""))
    if _CM_TEXT_RE.match(name):
        return "CM-TEXT"
    match = re.match(r"[A-Za-z]+", name)
    return (match.group(0) if match else name).upper()


def _dominant_text_family(lines: list[dict[str, Any]]) -> str | None:
    """The page's body face: the normalized text family carrying the most non-space
    characters, math/symbol faces excluded. ``None`` on pages without text spans."""
    counts: dict[str, int] = {}
    for line in lines:
        for span in line.get("spans") or []:
            if _span_protection(span) is not None:
                continue
            chars = len(_span_text(span).replace(" ", ""))
            if chars:
                key = _text_family(span)
                counts[key] = counts.get(key, 0) + chars
    if not counts:
        return None
    return max(counts, key=counts.get)


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
        lines = [
            line
            for block in page_dict.get("blocks") or []
            if block.get("type") == 0
            for line in block.get("lines") or []
        ]
        # The page's dominant TEXT family (by char volume, math faces excluded)
        # anchors the formula-vs-prose call on island lines: a display equation
        # sets its operator names in the math ecosystem's roman, not in the
        # page's body face.
        dominant_family = _dominant_text_family(lines)
        for line in lines:
            self._append_line(line, stats, dominant_family=dominant_family)
        return PageCells(
            cells=stats.cells,
            dropped_protected_lines=stats.protected,
            dropped_rotated_lines=stats.rotated,
            stripped_marker_spans=stats.stripped_markers,
        )

    def _append_line(
        self, line: dict[str, Any], stats: _LineStats, *, dominant_family: str | None = None
    ) -> None:
        spans = [span for span in (line.get("spans") or []) if _span_text(span).strip()]
        if not spans:
            return
        direction = line.get("dir") or (1.0, 0.0)
        if abs(float(direction[0])) < 0.99:  # rotated/vertical: keep original pixels
            stats.rotated += 1
            return
        kinds = [(span, _span_protection(span)) for span in spans]
        if any(
            kind == "symbol" and len(_span_text(span).strip()) > _MARKER_MAX_CHARS
            for span, kind in kinds
        ):
            stats.protected += 1  # substantial symbol content (an icon row): keep pixels
            return
        island_runs = _island_runs(spans, kinds, dominant_family=dominant_family)
        if island_runs is _FORMULA_LINE:
            stats.protected += 1  # formula-dominated line: keep pixels whole
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
        island_span_ids = {id(span) for run in island_runs for span in run}
        prose_spans = [span for span in spans if id(span) not in island_span_ids]
        islands: list[dict[str, Any]] = []
        parts: list[str] = []
        run_starts = {id(run[0]): run for run in island_runs}
        for span in spans:
            run = run_starts.get(id(span))
            if run is not None:
                stats.island_seq += 1
                island_bbox = _glyph_bbox(run)
                if island_bbox is None:
                    continue
                islands.append(
                    {
                        "id": f"M{stats.island_seq}",
                        "bbox": {
                            "left": round(island_bbox[0] * self._scale, 2),
                            "top": round(island_bbox[1] * self._scale, 2),
                            "width": round(max(1.0, (island_bbox[2] - island_bbox[0]) * self._scale), 2),
                            "height": round(max(1.0, (island_bbox[3] - island_bbox[1]) * self._scale), 2),
                        },
                    }
                )
                parts.append(f"⟦M{stats.island_seq}⟧")
            elif id(span) not in island_span_ids:
                parts.append(_span_text(span).strip())
        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if not text:
            return
        # Geometry from the PROSE glyphs: an island glyph (a radical, a stacked script)
        # can reach far above the text band, and a cell bbox inflated by it shifts the
        # render anchor a band up — two lines then print on top of each other (measured).
        # The islands' own ink is erased via their recorded bboxes, not via the cell box.
        bbox = (
            _glyph_bbox(prose_spans)
            or _glyph_bbox(spans)
            or [float(v) for v in (line.get("bbox") or (0, 0, 0, 0))]
        )
        left, top = bbox[0] * self._scale, bbox[1] * self._scale
        width = max(1.0, (bbox[2] - bbox[0]) * self._scale)
        height = max(1.0, (bbox[3] - bbox[1]) * self._scale)
        # Dominant span by GLYPH AREA (chars x size^2), not by char count: a KPI
        # line "$865 million" is a large amount plus a small unit-word, and the
        # small word has more characters — the size that visually carries the
        # line is the big one.
        dominant = max(prose_spans or spans, key=lambda span: len(_span_text(span).strip()) * float(span.get("size") or 0.0) ** 2)
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
        if islands:
            cell["islands"] = islands
        stats.cells.append(cell)


# Sentinel: the line is formula-dominated and must keep its pixels whole.
_FORMULA_LINE = object()


def _island_runs(
    spans: list[dict[str, Any]],
    kinds: list[tuple[dict[str, Any], str | None]],
    *,
    dominant_family: str | None = None,
) -> Any:
    """Group the line's math into islands, or classify the line as a formula.

    An island run is a maximal consecutive span sequence that contains at least one
    true-math span, extended over adjacent spans that belong to the formula rather
    than the prose: LETTERLESS spans (a formula's digits and operators are typeset
    in the text face — the CMR "= 8" beside a CMMI "h", the measured span pattern)
    and clearly SMALLER spans (a roman sub/superscript like the "model" in
    "d_model"). Returns a list of runs (possibly empty: a pure prose line), or
    ``_FORMULA_LINE`` when the line is formula-dominated: math glyph share above
    ``_ISLAND_MAX_MATH_SHARE``, too little prose (fewer than
    ``_ISLAND_MIN_PROSE_WORDS`` words), or prose set entirely OUTSIDE the page's
    dominant text family — a display equation writes its operator names in the math
    ecosystem's roman while real prose is set in the body face. Display equations
    and near-formula lines keep their pixels whole."""
    is_math = [kind == "math" for _span, kind in kinds]
    if not any(is_math):
        return []
    line_size = max((float(s.get("size") or 0.0) for s in spans), default=0.0)
    absorbable = [
        not any(ch.isalpha() for ch in _span_text(span))
        or (line_size > 0 and float(span.get("size") or 0.0) < _ISLAND_SCRIPT_SIZE_RATIO * line_size)
        for span in spans
    ]
    in_island = list(is_math)
    # Fixpoint expansion left+right: letterless neighbours join the island.
    changed = True
    while changed:
        changed = False
        for i in range(len(spans)):
            if in_island[i] or not absorbable[i]:
                continue
            if (i > 0 and in_island[i - 1]) or (i + 1 < len(spans) and in_island[i + 1]):
                in_island[i] = True
                changed = True
    math_chars = sum(
        len(_span_text(span).replace(" ", "")) for span, inside in zip(spans, in_island) if inside
    )
    total_chars = sum(len(_span_text(span).replace(" ", "")) for span in spans)
    prose_words = len(
        " ".join(_span_text(span) for span, inside in zip(spans, in_island) if not inside).split()
    )
    if total_chars <= 0 or math_chars / total_chars > _ISLAND_MAX_MATH_SHARE:
        return _FORMULA_LINE
    # The min-prose-words floor guards display-equation ANNOTATIONS ("where head_i = ..."),
    # which are prose-sparse. It must not drop a short prose line carrying only a footnote-
    # sized mark: an author name "Ashish Vaswani*" is 2 prose words + a 1-char CMSY star, and
    # dropping it left the author grid without a name anchor (small hallucinated duplicates).
    # A math run of at most a marker's length is never a formula worth whole-line preservation.
    if prose_words < _ISLAND_MIN_PROSE_WORDS and math_chars > _MARKER_MAX_CHARS:
        return _FORMULA_LINE
    if dominant_family is not None and not any(
        _text_family(span) == dominant_family
        for span, inside in zip(spans, in_island)
        if not inside
    ):
        return _FORMULA_LINE
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for span, inside in zip(spans, in_island):
        if inside:
            current.append(span)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


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
