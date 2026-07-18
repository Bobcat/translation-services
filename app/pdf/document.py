"""PDF intake: validate a document and profile its pages.

The profile is the per-page census the response and the quality benchmark carry.
In phase 0 every page takes the raster route regardless of class; the class
(born-digital | scanned | hybrid) is informational until text-layer routing
lands (docs/pdf-translation-design.md §6, §8).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf


# A page whose largest image covers at least this fraction of the page area is
# scan-shaped; combined with the extractable-char count it yields the class.
_FULL_PAGE_IMAGE_COVERAGE = 0.9


class PdfValidationError(ValueError):
    """A document rejected at intake; ``code`` maps onto the API error dialect."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class PageProfile:
    index: int
    width_pt: float
    height_pt: float
    rotation: int
    text_chars: int
    image_coverage: float
    page_class: str  # "born-digital" | "scanned" | "hybrid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "width_pt": self.width_pt,
            "height_pt": self.height_pt,
            "rotation": self.rotation,
            "text_chars": self.text_chars,
            "image_coverage": self.image_coverage,
            "page_class": self.page_class,
        }


@dataclass(frozen=True)
class DocumentProfile:
    page_count: int
    pages: list[PageProfile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_count": self.page_count,
            "pages": [page.to_dict() for page in self.pages],
        }


def profile_pdf(source: bytes | Path, *, page_cap: int) -> DocumentProfile:
    """Open ``source`` (raw upload bytes or a stored file) and profile every page.

    Raises ``PdfValidationError`` for anything the pipeline cannot take: not a
    parsable PDF, an encrypted document, or more pages than ``page_cap``.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            doc = pymupdf.open(stream=bytes(source), filetype="pdf")
        else:
            doc = pymupdf.open(str(source))
    except Exception as exc:
        raise PdfValidationError("REQUEST_INVALID_INPUT", "uploaded document_file could not be parsed as PDF") from exc

    try:
        if doc.needs_pass or doc.is_encrypted:
            raise PdfValidationError("REQUEST_PDF_ENCRYPTED", "encrypted PDF documents are not supported")
        if not doc.is_pdf:
            raise PdfValidationError("REQUEST_INVALID_INPUT", "uploaded document_file is not a PDF")
        page_count = int(doc.page_count)
        if page_count < 1:
            raise PdfValidationError("REQUEST_EMPTY_INPUT", "uploaded document_file has no pages")
        if page_count > int(page_cap):
            raise PdfValidationError(
                "REQUEST_PDF_TOO_MANY_PAGES",
                f"document has {page_count} pages; the limit per request is {int(page_cap)}",
            )
        pages = [_profile_page(doc, index) for index in range(page_count)]
        return DocumentProfile(page_count=page_count, pages=pages)
    finally:
        doc.close()


def _profile_page(doc: pymupdf.Document, index: int) -> PageProfile:
    page = doc[index]
    rect = page.rect
    text_chars = len((page.get_text("text") or "").strip())
    coverage = _largest_image_coverage(page)
    if coverage >= _FULL_PAGE_IMAGE_COVERAGE:
        page_class = "hybrid" if text_chars > 0 else "scanned"
    else:
        page_class = "born-digital"
    return PageProfile(
        index=index,
        width_pt=round(float(rect.width), 2),
        height_pt=round(float(rect.height), 2),
        rotation=int(page.rotation),
        text_chars=text_chars,
        image_coverage=round(coverage, 4),
        page_class=page_class,
    )


def _largest_image_coverage(page: pymupdf.Page) -> float:
    page_area = float(page.rect.width * page.rect.height)
    if page_area <= 0:
        return 0.0
    best = 0.0
    try:
        for image in page.get_images(full=True):
            for rect in page.get_image_rects(image[0]):
                area = float(rect.width * rect.height)
                if area > best:
                    best = area
    except Exception:
        # Image bookkeeping in broken PDFs must not fail the census; a page whose
        # images cannot be resolved profiles as text-only.
        return 0.0
    return min(1.0, best / page_area)
