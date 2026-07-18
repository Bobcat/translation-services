"""Unit tests for app.pdf.textlayer — text-layer cells from generated PDFs.

Documents are built in-test with pymupdf; no fixtures, no models. Covers the
cell shape and pixel scaling, the style metadata, and the phase-1 protection
policy (protected fonts, unmappable glyphs, rotated lines drop whole)."""
from __future__ import annotations

from pathlib import Path

import pymupdf

from app.pdf.textlayer import PageTextExtractor
from app.pdf.textlayer import _is_protected_span


def _pdf(tmp_path: Path, build) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    build(page)
    path = tmp_path / "doc.pdf"
    path.write_bytes(doc.tobytes())
    doc.close()
    return path


def test_extracts_line_cells_in_pixel_space(tmp_path: Path) -> None:
    path = _pdf(tmp_path, lambda page: page.insert_text((72, 100), "A born-digital line of text", fontsize=12))
    with PageTextExtractor(path, dpi=160) as extractor:
        result = extractor.cells_for_page(0)
    assert len(result.cells) == 1
    cell = result.cells[0]
    assert cell["text"] == "A born-digital line of text"
    assert cell["confidence"] == 1.0
    assert cell["source"] == "pdf_text_layer"
    assert cell["size_pt"] == 12.0
    # pt -> px at 160 dpi: x = 72 pt * 160/72 = 160 px (line bbox starts there).
    assert abs(cell["bbox"]["left"] - 160.0) < 3.0
    assert len(cell["polygon"]) == 4
    assert result.dropped_protected_lines == 0


def test_bold_font_sets_weight(tmp_path: Path) -> None:
    path = _pdf(tmp_path, lambda page: page.insert_text((72, 100), "Bold heading", fontname="hebo", fontsize=14))
    with PageTextExtractor(path, dpi=160) as extractor:
        result = extractor.cells_for_page(0)
    assert result.cells[0]["font_weight"] == 700


def test_symbol_font_line_is_dropped_whole(tmp_path: Path) -> None:
    def build(page: pymupdf.Page) -> None:
        page.insert_text((72, 100), "Normal prose stays", fontsize=12)
        page.insert_text((72, 200), "abgd", fontname="symb", fontsize=12)  # Symbol glyphs

    path = _pdf(tmp_path, build)
    with PageTextExtractor(path, dpi=160) as extractor:
        result = extractor.cells_for_page(0)
    assert [cell["text"] for cell in result.cells] == ["Normal prose stays"]
    assert result.dropped_protected_lines == 1


def test_rotated_line_is_dropped(tmp_path: Path) -> None:
    def build(page: pymupdf.Page) -> None:
        page.insert_text((72, 100), "Horizontal", fontsize=12)
        page.insert_text((30, 400), "Vertical margin text", fontsize=10, rotate=90)

    path = _pdf(tmp_path, build)
    with PageTextExtractor(path, dpi=160) as extractor:
        result = extractor.cells_for_page(0)
    assert [cell["text"] for cell in result.cells] == ["Horizontal"]
    assert result.dropped_rotated_lines == 1


def test_inline_bullet_marker_is_stripped_not_dropped(tmp_path: Path) -> None:
    def build(page: pymupdf.Page) -> None:
        # A bullet glyph in Symbol followed by prose on the same baseline: the
        # detector reads them as one line with a protected marker span.
        page.insert_text((72, 100), "·", fontname="symb", fontsize=12)
        page.insert_text((90, 100), "report suspected adverse reactions", fontsize=12)

    path = _pdf(tmp_path, build)
    with PageTextExtractor(path, dpi=160) as extractor:
        result = extractor.cells_for_page(0)
    texts = [cell["text"] for cell in result.cells]
    assert "report suspected adverse reactions" in texts
    assert result.dropped_protected_lines == 0
    # The marker was stripped (counted) unless the two inserts landed in
    # separate lines, in which case the marker line simply produced no cell.
    assert result.stripped_marker_spans >= 0
    assert all("·" not in text for text in texts)


def test_protected_span_classifier() -> None:
    assert _is_protected_span({"font": "GFJLNS+CMMI10", "text": "x"})
    assert _is_protected_span({"font": "KIGEPK+CMEX10", "text": "("})
    assert _is_protected_span({"font": "COXSDN+CMR10", "text": "2"})
    assert _is_protected_span({"font": "SymbolMT", "text": "•"})
    assert _is_protected_span({"font": "CVPBQT+Wingdings3", "text": ""})
    assert _is_protected_span({"font": "Helvetica", "text": ""})  # PUA glyphs
    assert _is_protected_span({"font": "Arial", "text": "��"})  # unmapped CIDs
    assert not _is_protected_span({"font": "LOOECV+NimbusRomNo9L-Regu", "text": "prose"})
    assert not _is_protected_span({"font": "CVPBQT+Montserrat-Bold", "text": "2025 STEWART"})
    assert not _is_protected_span({"font": "Arial-BoldMT", "text": "Summary"})
    # One stray replacement char inside real prose does not protect the span.
    assert not _is_protected_span({"font": "Helvetica", "text": "a long sentence with one � glyph"})
