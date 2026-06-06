"""Task pipeline: image in -> translated image out.

Current stages (OCR-inspection phase):

    ingest (canonical input, done in app.main)
    -> OCR            app.ocr.run_raw_ocr     (scene route, or document route)
    -> overlay debug  app.ocr.overlay         (boxes drawn on the input)

Planned stages (rebuild): orientation rescue -> coverage gate -> grouping
-> routing (app.translation.routing) -> translation -> re-placement (render).
Grouping, translation and rendering are intentionally disabled for now.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
from typing import Any

from app.core.config import AppSettings
from app.ocr.document_unwarp import render_paddle_document_unwarped_debug
from app.ocr import resolve_ocr_language
from app.ocr import run_raw_ocr
from app.ocr.overlay import render_original_ocr_overlay_debug
from app.ocr.segment import OcrSegment


@dataclass(frozen=True)
class TranslateImageResult:
    image: bytes
    mime_type: str
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    metrics: dict[str, float | int | None]
    debug_image: bytes | None = None
    debug_mime_type: str = "image/png"
    rectified_debug_image: bytes | None = None
    rectified_debug_mime_type: str = "image/png"
    projected_overlay_debug_image: bytes | None = None
    projected_overlay_debug_mime_type: str = "image/png"
    document_unwarped_debug_image: bytes | None = None
    document_unwarped_debug_mime_type: str = "image/png"
    ocr: dict[str, Any] | None = None
    document: dict[str, Any] | None = None


def run_translate_image_pipeline(
    *,
    settings: AppSettings,
    input_path: Path,
    input_mime_type: str,
    request: dict[str, Any],
) -> TranslateImageResult:
    del input_mime_type
    started_at = time.perf_counter()
    source_lang = str(request.get("source_lang_code") or "").strip()
    target_lang = str(request.get("target_lang_code") or "").strip()
    if not source_lang:
        raise RuntimeError("source_lang_code is required for OCR inspection")

    requested_ocr_route = str(request.get("ocr_route") or "scene").strip().lower()
    if requested_ocr_route not in {"scene", "document"}:
        raise RuntimeError(f"unsupported ocr_route: {requested_ocr_route or 'unknown'}")

    ocr_language = resolve_ocr_language(settings.ocr, source_lang)
    ocr_unwarp = bool(request.get("ocr_unwarp"))
    if requested_ocr_route == "document":
        return _run_document_route(
            settings=settings,
            input_path=input_path,
            ocr_language=ocr_language,
            ocr_unwarp=ocr_unwarp,
            source_lang=source_lang,
            target_lang=target_lang,
            started_at=started_at,
        )

    ocr_started_at = time.perf_counter()
    ocr_segments = run_raw_ocr(
        settings.ocr,
        input_path,
        language=ocr_language,
        use_doc_unwarping=ocr_unwarp,
    )
    ocr_wall_ms = _elapsed_ms(ocr_started_at)

    projected_overlay_debug = render_original_ocr_overlay_debug(
        input_path=input_path,
        ocr_segments=ocr_segments,
    )
    cells = _ocr_cells(ocr_segments)
    image = projected_overlay_debug.image or _input_png_bytes(input_path)
    metadata = {
        "ocr_backend": settings.ocr.backend,
        "ocr_language": ocr_language,
        "ocr_route": requested_ocr_route,
        "ocr_unwarp": ocr_unwarp,
        "effective_ocr_route": "scene",
        "translation_alignment": "ocr_inspect_only",
        "translation_ocr_space": "original",
        "translation_ocr_segment_count": len(ocr_segments),
        "full_translated_text": "",
        "source_lang_code": source_lang,
        "target_lang_code": target_lang,
        "note": "OCR inspection mode returns raw OCR cells only; grouping, translation, and translated rendering are intentionally disabled.",
    }
    metadata.update(projected_overlay_debug.metadata)
    metadata.update(
        {
            "rectified_ocr_applied": False,
            "rectified_ocr_used_for_translation": False,
            "output_rendering_space": "original_input",
        }
    )

    return TranslateImageResult(
        image=image,
        mime_type="image/png",
        projected_overlay_debug_image=projected_overlay_debug.image,
        projected_overlay_debug_mime_type=projected_overlay_debug.mime_type,
        segments=cells,
        metadata=metadata,
        metrics={
            "ocr_wall_ms": ocr_wall_ms,
            "llm_pool_wall_ms": 0.0,
            "translate_image_total_wall_ms": _elapsed_ms(started_at),
            "ocr_segment_count": len(ocr_segments),
            "translation_ocr_segment_count": len(ocr_segments),
            "translation_unit_count": 0,
            "llm_pool_request_count": 0,
        },
        ocr={
            "route": "scene",
            "cells": cells,
            "layout_regions": [],
        },
    )


def _run_document_route(
    *,
    settings: AppSettings,
    input_path: Path,
    ocr_language: str,
    ocr_unwarp: bool,
    source_lang: str,
    target_lang: str,
    started_at: float,
) -> TranslateImageResult:
    document_started_at = time.perf_counter()
    document_unwarped_debug = render_paddle_document_unwarped_debug(
        settings=settings.ocr,
        input_path=input_path,
        language=ocr_language,
        use_doc_unwarping=ocr_unwarp,
    )
    document_wall_ms = _elapsed_ms(document_started_at)
    output_image = document_unwarped_debug.image or _input_png_bytes(input_path)

    metadata = {
        "ocr_backend": settings.ocr.backend,
        "ocr_language": ocr_language,
        "ocr_route": "document",
        "ocr_unwarp": ocr_unwarp,
        "effective_ocr_route": "document",
        "source_lang_code": source_lang,
        "target_lang_code": target_lang,
        "translation_alignment": "document_inspect",
        "translation_ocr_space": "document_unwarped",
        "translation_ocr_segment_count": 0,
        "full_translated_text": "",
        "output_rendering_space": "document_unwarped_debug",
        "note": "Document route currently returns PPStructure document OCR/layout inspection only; translation and rendering are intentionally disabled.",
    }
    metadata.update(document_unwarped_debug.metadata)

    return TranslateImageResult(
        image=output_image,
        mime_type="image/png",
        segments=[],
        metadata=metadata,
        metrics={
            "document_route_wall_ms": document_wall_ms,
            "translate_image_total_wall_ms": _elapsed_ms(started_at),
            "document_cell_count": len(document_unwarped_debug.cells),
            "document_layout_region_count": len(document_unwarped_debug.layout_regions),
            "llm_pool_request_count": 0,
        },
        document_unwarped_debug_image=document_unwarped_debug.image,
        document_unwarped_debug_mime_type=document_unwarped_debug.mime_type,
        ocr={
            "route": "document",
            "cells": document_unwarped_debug.cells,
            "layout_regions": document_unwarped_debug.layout_regions,
        },
        document={
            "cells": document_unwarped_debug.cells,
            "layout_regions": document_unwarped_debug.layout_regions,
        },
    )


def _ocr_cells(ocr_segments: list[OcrSegment]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for segment in ocr_segments:
        text = str(segment.text or "").strip()
        if not text:
            continue
        payload: dict[str, Any] = {
            "id": len(cells) + 1,
            "text": text,
            "bbox": dict(segment.bbox),
            "confidence": round(float(segment.confidence), 4),
        }
        if segment.polygon is not None:
            payload["polygon"] = [dict(point) for point in segment.polygon]
        cells.append(payload)
    return cells


def _input_png_bytes(input_path: Path) -> bytes:
    from PIL import Image

    out = BytesIO()
    Image.open(input_path).convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - started_at) * 1000.0)
