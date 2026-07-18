"""Assemble rendered page images into an output PDF on the original page sizes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class PageImage:
    png_path: Path
    width_pt: float
    height_pt: float


def assemble_pdf(pages: list[PageImage]) -> bytes:
    """One raster page image per output page, drawn full-bleed on a page of the
    source's dimensions (pt) — mixed page sizes within one document survive."""
    if not pages:
        raise ValueError("assemble_pdf requires at least one page")
    doc = pymupdf.open()
    try:
        for page_image in pages:
            page = doc.new_page(width=float(page_image.width_pt), height=float(page_image.height_pt))
            page.insert_image(page.rect, filename=str(page_image.png_path))
        return doc.tobytes(garbage=4, deflate=True)
    finally:
        doc.close()
