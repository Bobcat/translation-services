"""Render single PDF pages to PNG at a fixed dpi (the analysis raster)."""
from __future__ import annotations

from pathlib import Path

import pymupdf


class PageRasterizer:
    """Holds one document open and renders pages on demand, so a multi-page loop
    does not re-parse the file per page. Use as a context manager."""

    def __init__(self, path: Path, *, dpi: int) -> None:
        self._path = Path(path)
        self._dpi = int(dpi)
        self._doc: pymupdf.Document | None = None

    def __enter__(self) -> "PageRasterizer":
        self._doc = pymupdf.open(str(self._path))
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def render_png(self, page_index: int) -> bytes:
        if self._doc is None:
            raise RuntimeError("PageRasterizer used outside its context")
        pixmap = self._doc[int(page_index)].get_pixmap(dpi=self._dpi)
        return pixmap.tobytes("png")
