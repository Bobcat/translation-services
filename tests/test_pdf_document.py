"""Unit tests for app.pdf: census/validation, page rasterization, PDF assembly.

All documents are generated in-test with pymupdf/PIL — no fixtures, no GPU, no
network. The census thresholds are exercised with the three page archetypes the
page classifier must separate: vector text (born-digital), one full-page image
(scanned), and a full-page image with a text layer on top (hybrid).
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from app.pdf.assemble import PageImage
from app.pdf.assemble import assemble_pdf
from app.pdf.document import PdfValidationError
from app.pdf.document import profile_pdf
from app.pdf.raster import PageRasterizer


def _png_bytes(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    out = BytesIO()
    Image.new("RGB", (width, height), color).save(out, format="PNG")
    return out.getvalue()


def _three_class_pdf() -> bytes:
    """Page 1: vector text. Page 2: one full-page image. Page 3: image + text."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "A born-digital paragraph of vector text.")
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=_png_bytes(64, 91, (200, 200, 200)))
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=_png_bytes(64, 83, (220, 220, 220)))
    page.insert_text((72, 72), "An OCR-like text layer over the scan.")
    data = doc.tobytes()
    doc.close()
    return data


def test_profile_pdf_census_classes() -> None:
    profile = profile_pdf(_three_class_pdf(), page_cap=25)
    assert profile.page_count == 3
    assert [page.page_class for page in profile.pages] == ["born-digital", "scanned", "hybrid"]
    assert profile.pages[0].text_chars > 0
    assert profile.pages[1].text_chars == 0
    assert profile.pages[1].image_coverage >= 0.9
    assert (profile.pages[0].width_pt, profile.pages[0].height_pt) == (595.0, 842.0)
    assert (profile.pages[2].width_pt, profile.pages[2].height_pt) == (612.0, 792.0)


def test_profile_pdf_page_cap() -> None:
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=100, height=100)
    data = doc.tobytes()
    doc.close()
    with pytest.raises(PdfValidationError) as excinfo:
        profile_pdf(data, page_cap=2)
    assert excinfo.value.code == "REQUEST_PDF_TOO_MANY_PAGES"


def test_profile_pdf_rejects_encrypted() -> None:
    doc = pymupdf.open()
    doc.new_page(width=100, height=100)
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    with pytest.raises(PdfValidationError) as excinfo:
        profile_pdf(data, page_cap=25)
    assert excinfo.value.code == "REQUEST_PDF_ENCRYPTED"


def test_profile_pdf_rejects_garbage() -> None:
    with pytest.raises(PdfValidationError) as excinfo:
        profile_pdf(b"this is not a pdf", page_cap=25)
    assert excinfo.value.code == "REQUEST_INVALID_INPUT"


def test_rasterizer_renders_at_dpi(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(_three_class_pdf())
    with PageRasterizer(pdf_path, dpi=160) as rasterizer:
        png = rasterizer.render_png(0)
    with Image.open(BytesIO(png)) as image:
        # 595 pt at 160 dpi -> 595 / 72 * 160 ≈ 1322 px (pymupdf rounds the matrix).
        assert abs(image.width - round(595 / 72 * 160)) <= 2
        assert abs(image.height - round(842 / 72 * 160)) <= 2


def test_assemble_pdf_preserves_page_sizes(tmp_path: Path) -> None:
    first = tmp_path / "p1.png"
    second = tmp_path / "p2.png"
    first.write_bytes(_png_bytes(120, 170, (10, 120, 240)))
    second.write_bytes(_png_bytes(240, 155, (240, 120, 10)))
    data = assemble_pdf(
        [
            PageImage(png_path=first, width_pt=595.32, height_pt=841.92),
            PageImage(png_path=second, width_pt=1224.0, height_pt=792.0),
        ]
    )
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        assert doc.page_count == 2
        assert (round(doc[0].rect.width, 2), round(doc[0].rect.height, 2)) == (595.32, 841.92)
        assert (round(doc[1].rect.width, 2), round(doc[1].rect.height, 2)) == (1224.0, 792.0)
        # Each page carries exactly the one full-bleed raster.
        assert len(doc[0].get_images(full=True)) == 1
        assert len(doc[1].get_images(full=True)) == 1
    finally:
        doc.close()


def test_assemble_pdf_requires_pages() -> None:
    with pytest.raises(ValueError):
        assemble_pdf([])
