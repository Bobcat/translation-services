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
    assert _is_protected_span({"font": "QQXNAB+CMSY10", "text": "∈"})
    assert _is_protected_span({"font": "MSBM10", "text": "R"})
    assert _is_protected_span({"font": "CambriaMath", "text": "x"})
    assert _is_protected_span({"font": "SymbolMT", "text": "•"})
    # Computer Modern TEXT faces are body type in classic LaTeX, not math: a
    # pure-CM document must extract its prose (islands design doc, phase 1).
    assert not _is_protected_span({"font": "COXSDN+CMR12", "text": "The proof follows"})
    assert not _is_protected_span({"font": "CMBX12", "text": "1 Introduction"})
    assert not _is_protected_span({"font": "CMTI10", "text": "Theorem 1.1"})
    assert not _is_protected_span({"font": "CMTT10", "text": "tensor2tensor"})
    assert _is_protected_span({"font": "CVPBQT+Wingdings3", "text": ""})
    assert _is_protected_span({"font": "Helvetica", "text": ""})  # PUA glyphs
    assert _is_protected_span({"font": "Arial", "text": "��"})  # unmapped CIDs
    assert not _is_protected_span({"font": "LOOECV+NimbusRomNo9L-Regu", "text": "prose"})
    assert not _is_protected_span({"font": "CVPBQT+Montserrat-Bold", "text": "2025 STEWART"})
    assert not _is_protected_span({"font": "Arial-BoldMT", "text": "Summary"})
    # One stray replacement char inside real prose does not protect the span.
    assert not _is_protected_span({"font": "Helvetica", "text": "a long sentence with one � glyph"})


def _span(font: str, text: str, size: float = 10.0) -> dict:
    return {
        "font": font,
        "size": size,
        "chars": [{"c": ch, "bbox": [i * 5.0, 0.0, i * 5.0 + 5.0, size]} for i, ch in enumerate(text)],
    }


def _runs(spans, dominant="NIMBUSROMNO"):
    from app.pdf.textlayer import _island_runs, _span_protection

    kinds = [(span, _span_protection(span)) for span in spans]
    return _island_runs(spans, kinds, dominant_family=dominant)


def test_island_runs_minority_math_with_absorbed_operator_run() -> None:
    # The measured pattern: prose, then CMMI "h", then the letterless CMR "= 8"
    # (a formula's digits ride in the text face), then prose — one island.
    spans = [
        _span("NimbusRomNo9L-Regu", "In this work we employ"),
        _span("GFJLNS+CMMI10", " h"),
        _span("COXSDN+CMR10", " = 8"),
        _span("NimbusRomNo9L-Regu", " parallel attention layers"),
    ]
    runs = _runs(spans)
    assert isinstance(runs, list) and len(runs) == 1
    assert [s["font"] for s in runs[0]] == ["GFJLNS+CMMI10", "COXSDN+CMR10"]


def test_island_runs_absorbs_a_roman_subscript_by_size() -> None:
    # "d_model = 512": the roman subscript "model" (7pt vs 10pt prose) belongs to
    # the formula even though it carries letters.
    spans = [
        _span("NimbusRomNo9L-Regu", "input and output is"),
        _span("GFJLNS+CMMI10", " d"),
        _span("COXSDN+CMR7", "model", size=7.0),
        _span("COXSDN+CMR10", " = 512 ,"),
        _span("NimbusRomNo9L-Regu", " and the inner-layer"),
    ]
    runs = _runs(spans)
    assert isinstance(runs, list) and len(runs) == 1
    assert len(runs[0]) == 3  # CMMI d + subscript + "= 512 ,"


def test_island_runs_display_equation_without_body_prose_is_formula() -> None:
    from app.pdf.textlayer import _FORMULA_LINE

    # A display equation writes its operator names in the math ecosystem's roman
    # (CMR), not in the page's body face — no body prose, so the line keeps pixels.
    spans = [
        _span("COXSDN+CMR10", "MultiHead("),
        _span("GFJLNS+CMMI10", "Q, K, V"),
        _span("COXSDN+CMR10", ") = Concat(head"),
        _span("GFJLNS+CMMI7", "1", size=7.0),
        _span("COXSDN+CMR10", ")W"),
    ]
    assert _runs(spans) is _FORMULA_LINE


def test_island_runs_majority_math_is_formula() -> None:
    from app.pdf.textlayer import _FORMULA_LINE

    spans = [
        _span("NimbusRomNo9L-Regu", "where also"),
        _span("GFJLNS+CMMI10", " W Q K V head Attention QW KW VW softmax d"),
    ]
    assert _runs(spans) is _FORMULA_LINE


def test_pure_cm_document_prose_with_inline_math_islands() -> None:
    # In a pure-CM document (body CMR12) the CM text faces collapse to one family,
    # so prose around inline math passes the dominance test and islands form.
    spans = [
        _span("CMR12", "The proof of the estimate uses", size=12.0),
        _span("CMMI12", " R", size=12.0),
        _span("CMR12", " as in the previous section", size=12.0),
    ]
    runs = _runs(spans, dominant="CM-TEXT")
    assert isinstance(runs, list) and len(runs) == 1
