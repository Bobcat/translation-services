"""Build a snapshot: the align structure plus the rendered image read back by OCR.

Re-OCR is bit-stable on identical pixels, so the read-back is a faithful, portable record of what
the render produced — text and position — without depending on exact pixel output across font /
library versions.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from app.core.config import OcrSettings
from app.ocr import run_raw_ocr
from app.regression.fixture import Snapshot
from app.regression.fixture import expected_unit_of


def reocr_rows(ocr_settings: OcrSettings, png_bytes: bytes, language: str) -> list[dict[str, Any]]:
    """Read a rendered PNG back with OCR -> ``[{text, left, top, width, height}]`` in reading order."""
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        handle.write(png_bytes)
        handle.flush()
        segments = run_raw_ocr(ocr_settings, Path(handle.name), language=language)
    rows = [
        {
            "text": str(segment.text or ""),
            "left": int(segment.bbox["left"]),
            "top": int(segment.bbox["top"]),
            "width": int(segment.bbox["width"]),
            "height": int(segment.bbox["height"]),
        }
        for segment in segments
        if str(segment.text or "").strip()
    ]
    rows.sort(key=lambda r: (r["top"] // 10, r["left"]))
    return rows


def build_snapshot(
    ocr_settings: OcrSettings,
    *,
    units: list[dict[str, Any]],
    ignored_cells: list[int],
    rendered_png: bytes,
    target_lang: str,
) -> Snapshot:
    """``units`` are the align-output unit dicts (from a live response or a replay)."""
    return Snapshot(
        expected_units=[expected_unit_of(unit) for unit in units],
        ignored_cells=sorted(int(c) for c in ignored_cells),
        reocr=reocr_rows(ocr_settings, rendered_png, target_lang),
    )
