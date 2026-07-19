"""Measurement layer: (source.pdf, translated.pdf) -> frozen measurement dict.

Deterministic on identical input files (measured: PP-DocLayout and OCR are
bit-stable on identical pixels), but environment-bound over time (model
upgrades, dpi). Everything scoring needs is captured here so evolving scoring
code can re-score history without re-running the models; the schema and model
versions ride along for exactly that reason.

Page renders are intentionally not part of the measurement: they are
reproducible from the PDFs at the recorded dpi.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pymupdf

from app.core.config import AppSettings
from app.layout import detect_layout_regions
from app.ocr import run_raw_ocr

SCHEMA_VERSION = 1
LAYOUT_MODEL = "PP-DocLayout_plus-L"
# Measurement-layer detection threshold (the model default is 0.5). On a perturbed render the
# detector splits one region's confidence over competing classes — a header detected at 0.914 on
# the clean source came back as doc_title 0.42 / paragraph_title 0.38 / header 0.35 / text 0.35
# on an externally re-rendered copy, every candidate under 0.5, so the region counted as lost
# and skewed the layout axis. 0.4 admits the winning candidate; scoring's same-family dedupe
# absorbs the extra duplicates. Measurement-only: the pipeline keeps the model default, because
# the threshold kwarg is not additive (see detect_layout_regions).
MEASURE_LAYOUT_THRESHOLD = 0.4


def measure_pair(
    *,
    settings: AppSettings,
    source_pdf: Path,
    translated_pdf: Path,
    ocr_language: str = "en",
) -> dict[str, Any]:
    dpi = int(settings.pdf.analysis_dpi)
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_dpi": dpi,
        "models": {
            "layout": LAYOUT_MODEL,
            # The detection threshold rides along: scores are only comparable between
            # measurements taken under the same threshold (a lower one admits more regions).
            "layout_threshold": MEASURE_LAYOUT_THRESHOLD,
            "ocr_backend": settings.ocr.backend,
            "ocr_version": settings.ocr.ocr_version,
            "ocr_language": ocr_language,
        },
        "source": _measure_document(settings, source_pdf, dpi=dpi, ocr_language=ocr_language),
        "translated": _measure_document(settings, translated_pdf, dpi=dpi, ocr_language=ocr_language),
    }


def _measure_document(
    settings: AppSettings, pdf_path: Path, *, dpi: int, ocr_language: str
) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    doc = pymupdf.open(str(pdf_path))
    try:
        with TemporaryDirectory(prefix="benchmark-measure-") as tmp:
            for index in range(doc.page_count):
                page = doc[index]
                pixmap = page.get_pixmap(dpi=dpi)
                png_path = Path(tmp) / f"page-{index + 1:03d}.png"
                png_path.write_bytes(pixmap.tobytes("png"))
                regions = detect_layout_regions(png_path, threshold=MEASURE_LAYOUT_THRESHOLD)
                segments = run_raw_ocr(settings.ocr, png_path, language=ocr_language)
                pages.append(
                    {
                        "index": index,
                        "width_pt": round(float(page.rect.width), 2),
                        "height_pt": round(float(page.rect.height), 2),
                        "width_px": int(pixmap.width),
                        "height_px": int(pixmap.height),
                        "regions": regions,
                        "segments": [
                            {
                                "text": str(segment.text or ""),
                                "bbox": dict(segment.bbox),
                                "confidence": round(float(segment.confidence), 4),
                            }
                            for segment in segments
                            if str(segment.text or "").strip()
                        ],
                    }
                )
    finally:
        doc.close()
    return {"page_count": len(pages), "pages": pages}
