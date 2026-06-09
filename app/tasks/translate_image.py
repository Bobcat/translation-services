"""Task pipeline: image in -> translated units out (re-placement not done yet).

Current stages:

    ingest (canonical input, done in app.main)
    -> OCR             app.ocr.run_raw_ocr        (scene route -> cells)
    -> grouping        app.grouping               (VLM hint -> translation units)
    -> routing+xlate   app.translation.translate  (per unit -> translated_text)
    -> overlay debug   app.grouping.overlay       (units drawn on the input)

Not done yet: orientation rescue, coverage gate, and re-placement (rendering the
translated text back into the image). The output image is still the debug overlay.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
from typing import Any

from app.core.config import AppSettings
from app.grouping import group_cells_into_units
from app.grouping.overlay import render_grouping_overlay_debug
from app.ocr import resolve_ocr_language
from app.ocr import run_raw_ocr
from app.ocr.overlay import render_original_ocr_overlay_debug
from app.ocr.segment import OcrSegment
from app.replacement import render_translated_image
from app.translation.translate import translate_units


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
    grouping_overlay_debug_image: bytes | None = None
    grouping_overlay_debug_mime_type: str = "image/png"
    rendered_image: bytes | None = None
    rendered_mime_type: str = "image/png"
    ocr: dict[str, Any] | None = None


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

    ocr_language = resolve_ocr_language(settings.ocr, source_lang)

    ocr_started_at = time.perf_counter()
    ocr_segments = run_raw_ocr(
        settings.ocr,
        input_path,
        language=ocr_language,
    )
    ocr_wall_ms = _elapsed_ms(ocr_started_at)

    projected_overlay_debug = render_original_ocr_overlay_debug(
        input_path=input_path,
        ocr_segments=ocr_segments,
    )
    cells = _ocr_cells(ocr_segments)

    grouping_model = str(request.get("grouping_model") or "").strip() or settings.llm_pool.grouping_model
    grouping = group_cells_into_units(
        settings=settings,
        input_path=input_path,
        cells=cells,
        model=grouping_model,
    )
    grouping_overlay_debug = render_grouping_overlay_debug(
        input_path=input_path,
        result=grouping,
        cells=cells,
    )

    image_category = str(grouping.metadata.get("category") or "")
    translator_model = str(request.get("translator_model") or "").strip() or settings.llm_pool.translator_model
    translator_mode = str(request.get("translator_mode") or "").strip() or settings.llm_pool.translator_mode
    translation_started = time.perf_counter()
    translations = translate_units(
        settings=settings,
        units=grouping.units,
        source_lang_code=source_lang,
        target_lang_code=target_lang,
        translator_model=translator_model,
        translator_mode=translator_mode,
        category=image_category,
        hint_raw=grouping.hint_raw,
        hint_units=grouping.hint_units,
    )
    translation_wall_ms = _elapsed_ms(translation_started)
    translation_by_id = {item.unit_id: item for item in translations}
    translation_units: list[dict[str, Any]] = []
    for unit in grouping.units:
        unit_dict = unit.to_dict()
        translated = translation_by_id.get(unit.id)
        if translated is not None:
            unit_dict["translated_text"] = translated.translated_text
            unit_dict["translator_model"] = translated.translator_model
            unit_dict["translation_route"] = translated.translation_route
        translation_units.append(unit_dict)
    translated_unit_count = sum(1 for item in translations if item.translated_text)
    # Actual llm-pool calls: one shared batch call (if any unit was batched) + one per
    # batch-fallback + one per plain per-unit (translategemma) translation.
    routes = [item.translation_route for item in translations]
    translation_call_count = (
        (1 if any(route.endswith("_batch") for route in routes) else 0)
        + sum(1 for route in routes if route.endswith("_batch_fallback"))
        + sum(
            1
            for route in routes
            if route and route != "skipped_empty"
            and not route.endswith("_batch")
            and not route.endswith("_batch_fallback")
        )
    )
    full_translated_text = "\n".join(item.translated_text for item in translations if item.translated_text)

    replacement_started = time.perf_counter()
    rendered_image = render_translated_image(input_path, translation_units)
    replacement_wall_ms = _elapsed_ms(replacement_started)

    image = projected_overlay_debug.image or _input_png_bytes(input_path)
    metadata = {
        "ocr_backend": settings.ocr.backend,
        "ocr_language": ocr_language,
        "translation_alignment": "units_translated_rendered",
        "translation_ocr_space": "original",
        "translation_ocr_segment_count": len(ocr_segments),
        "full_translated_text": full_translated_text,
        "source_lang_code": source_lang,
        "target_lang_code": target_lang,
        "translator_model": translator_model,
        "translator_mode": translator_mode,
        "note": "Returns OCR cells, VLM grouping, per-unit translations, and a Tier-1 re-placement rendering (model-free simple replace).",
    }
    metadata.update(projected_overlay_debug.metadata)
    metadata["grouping_model"] = grouping.model
    metadata["image_category"] = image_category
    metadata.update(grouping_overlay_debug.metadata)
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
        grouping_overlay_debug_image=grouping_overlay_debug.image,
        grouping_overlay_debug_mime_type=grouping_overlay_debug.mime_type,
        rendered_image=rendered_image,
        rendered_mime_type="image/png",
        segments=cells,
        metadata=metadata,
        metrics={
            "ocr_wall_ms": ocr_wall_ms,
            "grouping_wall_ms": float(grouping.metrics.get("grouping_wall_ms", 0.0)),
            "translation_wall_ms": translation_wall_ms,
            "replacement_wall_ms": replacement_wall_ms,
            "llm_pool_wall_ms": 0.0,
            "translate_image_total_wall_ms": _elapsed_ms(started_at),
            "ocr_segment_count": len(ocr_segments),
            "translation_ocr_segment_count": len(ocr_segments),
            "translation_unit_count": len(grouping.units),
            "translated_unit_count": translated_unit_count,
            "ignored_cell_count": len(grouping.ignored_cell_ids),
            "llm_pool_request_count": (1 if cells else 0) + translation_call_count,
        },
        ocr={
            "cells": cells,
            "translation_units": translation_units,
            "ignored_cell_ids": list(grouping.ignored_cell_ids),
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
