"""Unit tests for app.pdf: census/validation, page rasterization, PDF assembly.

All documents are generated in-test with pymupdf/PIL — no fixtures, no GPU, no
network. The census thresholds are exercised with the three page archetypes the
page classifier must separate: vector text (born-digital), one full-page image
(scanned), and a full-page image with a text layer on top (hybrid).
"""
from __future__ import annotations

from io import BytesIO
import json
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


def test_rerender_pdf_replays_cached_pages_and_carries_untranslated_ones(tmp_path) -> None:
    # A document re-render must touch nothing upstream of render: each page re-renders from its
    # own cached grouping/translation, and a page the source run left untranslated (no cache —
    # an image-only page) carries its previous render through instead of failing the document.
    from PIL import Image, ImageDraw

    from app.core.config import AppSettings
    from app.tasks.rerender_pdf import run_rerender_pdf_pipeline

    source_pages = tmp_path / "source" / "pages"
    for page_no in (1, 2):
        page_dir = source_pages / f"page-{page_no:03d}"
        page_dir.mkdir(parents=True)
        img = Image.new("RGB", (300, 120), (255, 255, 255))
        ImageDraw.Draw(img).rectangle((40, 40, 200, 62), fill=(0, 0, 0))
        img.save(page_dir / "input.png")
    cached = source_pages / "page-001"
    (cached / "grouping.json").write_text(json.dumps({"units": [{
        "id": 1, "order": 1, "source_text": "HELLO",
        "bbox": {"left": 40, "top": 40, "width": 160, "height": 22},
        "members": [{"cell_id": 1, "text": "HELLO", "translate": True, "order": 1,
                     "bbox": {"left": 40, "top": 40, "width": 160, "height": 22}}],
    }]}), encoding="utf-8")
    (cached / "translation.json").write_text(
        json.dumps([{"unit_id": 1, "translated_text": "Hallo"}]), encoding="utf-8"
    )
    # The untranslated page's previous render is a distinguishable colour, so "carried through
    # verbatim" is checkable rather than merely plausible.
    passthrough = Image.new("RGB", (300, 120), (10, 200, 40))
    passthrough.save(source_pages / "page-002" / "rendered.png")

    document = {
        "page_count": 2,
        "analysis_dpi": 200,
        "pages": [
            {"page": 1, "width_pt": 595.0, "height_pt": 842.0, "artifacts": {"rendered": "stale"}},
            {"page": 2, "width_pt": 595.0, "height_pt": 842.0},
        ],
    }
    out_pages = tmp_path / "out" / "pages"
    result = run_rerender_pdf_pipeline(
        settings=AppSettings(),
        source_pages_root=source_pages,
        source_document=document,
        request={"task": "rerender_pdf", "width_fit_mode": "footprint"},
        pages_root=out_pages,
        checkpoint=lambda: None,
    )

    assert result.rendered_pdf.startswith(b"%PDF")
    assert result.document["page_count"] == 2
    assert result.metadata["translation_source"] == "cached_rerender"
    # Page 1 was re-rendered (its cached translation replaced the source ink); page 2 is the
    # previous render byte for byte.
    rerendered = Image.open(out_pages / "page-001" / "rendered.png").convert("RGB")
    assert rerendered.size == (300, 120)
    carried = Image.open(out_pages / "page-002" / "rendered.png").convert("RGB")
    assert carried.getpixel((150, 60)) == (10, 200, 40)
    # The re-rendered page stays a valid source for a further re-render.
    assert (out_pages / "page-001" / "grouping.json").exists()
    assert (out_pages / "page-001" / "translation.json").exists()
