"""Task pipeline: image in -> translated units out (re-placement not done yet).

Current stages:

    ingest (canonical input, done in app.main)
    -> VLM hint        app.grouping.vlm           (image-only; also routes the OCR model)
    -> OCR             app.ocr.run_raw_ocr        (hint-routed model -> cells)
    -> grouping        app.grouping               (hint x cells -> translation units)
    -> routing+xlate   app.translation.translate  (per unit -> translated_text)
    -> overlay debug   app.grouping.overlay       (units drawn on the input)

Not done yet: orientation rescue, coverage gate, and re-placement (rendering the
translated text back into the image). The output image is still the debug overlay.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import time
from typing import Any
from typing import Callable

from app.core.config import AppSettings
from app.grouping import group_cells_into_units
from app.grouping import request_grouping_hint
from app.grouping.field_geometry import geometry_adjusted_hints
from app.grouping.overlay import render_grouping_overlay_debug
from app.grouping.units import TranslationUnit
from app.ocr import resolve_ocr_language
from app.ocr import run_raw_ocr
from app.ocr.overlay import render_original_ocr_overlay_debug
from app.ocr.segment import OcrSegment
from app.replacement import render_translated_image
from app.translation.prompts import resolve_structured_prompt
from app.translation.prompts import store_for
from app.translation.prompts.templates import IMAGE_DEFAULT_ID
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
    debug: dict[str, Any] | None = None  # per-request stage records persisted for review


def run_translate_image_pipeline(
    *,
    settings: AppSettings,
    input_path: Path,
    input_mime_type: str,
    request: dict[str, Any],
    checkpoint: Callable[[], None] = lambda: None,
) -> TranslateImageResult:
    """``checkpoint`` is called between stages; the runtime passes one that raises when the
    request was cancelled, so the run stops before the next expensive stage."""
    del input_mime_type
    started_at = time.perf_counter()
    source_lang = str(request.get("source_lang_code") or "").strip()
    target_lang = str(request.get("target_lang_code") or "").strip()
    if not source_lang:
        raise RuntimeError("source_lang_code is required for translation routing")

    # The VLM hint is image-only, so it can run before OCR — and its text doubles
    # as the script signal that picks the OCR model (Han/Kana -> the multilingual
    # server pair, anything else -> the en recognizer).
    grouping_model = str(request.get("grouping_model") or "").strip() or settings.llm_pool.grouping_model
    llm_calls: list[dict[str, Any]] = []  # payload + response of every VLM/LLM call, in order
    grouping_started_at = time.perf_counter()
    hint = request_grouping_hint(
        settings=settings,
        input_path=input_path,
        model=grouping_model,
        call_log=llm_calls,
    )
    grouping_wall_ms = _elapsed_ms(grouping_started_at)

    checkpoint()
    ocr_language = resolve_ocr_language(settings.ocr, hint.units)

    ocr_started_at = time.perf_counter()
    ocr_segments = run_raw_ocr(
        settings.ocr,
        input_path,
        language=ocr_language,
    )
    ocr_wall_ms = _elapsed_ms(ocr_started_at)

    cells = _ocr_cells(ocr_segments)

    align_started_at = time.perf_counter()
    grouping = group_cells_into_units(
        cells=cells,
        hint=hint,
        model=grouping_model,
    )
    align_wall_ms = _elapsed_ms(align_started_at)

    # Geometry spots rule-3/4 column `|`s the VLM missed (a separated label/amount). Fed to
    # translation by default (``use_geometry_columns``, below); also surfaced for inspection.
    hint_units_adjusted, field_geometry_changes = geometry_adjusted_hints(
        grouping.units, grouping.hint_units
    )

    # Debug overlays (OCR + grouping drawn on the image) are opt-in. Each renders and PNG-encodes
    # the full-size image (~1s+ each) — pure overhead for a non-debug caller such as the asr camera
    # app, which only fetches the rendered translation. The workbench asks for them via the request
    # flag ``debug_overlays``; everyone else gets the fast path.
    make_overlays = bool(request.get("debug_overlays"))
    projected_overlay_debug = None
    grouping_overlay_debug = None
    ocr_overlay_wall_ms = 0.0
    grouping_overlay_wall_ms = 0.0
    if make_overlays:
        overlay_started_at = time.perf_counter()
        projected_overlay_debug = render_original_ocr_overlay_debug(
            input_path=input_path,
            ocr_segments=ocr_segments,
        )
        ocr_overlay_wall_ms = _elapsed_ms(overlay_started_at)
        overlay_started_at = time.perf_counter()
        grouping_overlay_debug = render_grouping_overlay_debug(
            input_path=input_path,
            result=grouping,
            cells=cells,
        )
        grouping_overlay_wall_ms = _elapsed_ms(overlay_started_at)

    image_category = str(grouping.metadata.get("category") or "")
    translator_model = str(request.get("translator_model") or "").strip() or settings.llm_pool.translator_model
    translator_mode = str(request.get("translator_mode") or "").strip() or settings.llm_pool.translator_mode
    preserve_heuristic_text = _bool_request_flag(request, "preserve_heuristic_text", default=True)
    preserve_unchanged_text = _bool_request_flag(request, "preserve_unchanged_text", default=False)
    use_geometry_columns = _bool_request_flag(request, "use_geometry_columns", default=True)
    render_size_mode = str(request.get("render_size_mode") or "median").strip() or "median"
    erase_fill_mode = str(request.get("erase_fill_mode") or "flat").strip() or "flat"
    # Opt-in: feed the geometry-adjusted hints (a `|` injected where a column gap shows the VLM
    # missed a rule-3/4 boundary) instead of the raw VLM hints. Same translation path, different input.
    hint_units_for_translation = hint_units_adjusted if use_geometry_columns else grouping.hint_units
    units_for_translation = _units_for_preserve_heuristic_text(
        grouping.units,
        preserve_heuristic_text=preserve_heuristic_text,
    )
    checkpoint()
    translation_started = time.perf_counter()
    prompt = resolve_structured_prompt(
        store_for(settings.service.prompts_root),
        raw_prompt=request.get("translation_prompt"),
        prompt_id=request.get("translation_prompt_id"),
        default_id=IMAGE_DEFAULT_ID,
    )
    translations = translate_units(
        settings=settings,
        units=units_for_translation,
        source_lang_code=source_lang,
        target_lang_code=target_lang,
        translator_model=translator_model,
        translator_mode=translator_mode,
        category=image_category,
        hint_units=hint_units_for_translation,
        hint_block_ids=grouping.hint_block_ids,
        prompt=prompt,
        call_log=llm_calls,
        preserve_heuristic_text=preserve_heuristic_text,
        preserve_unchanged_text=preserve_unchanged_text,
    )
    translation_wall_ms = _elapsed_ms(translation_started)
    translation_by_id = {item.unit_id: item for item in translations}
    translation_units: list[dict[str, Any]] = []
    for unit in units_for_translation:
        unit_dict = unit.to_dict()
        translated = translation_by_id.get(unit.id)
        if translated is not None:
            unit_dict["translated_text"] = translated.translated_text
            unit_dict["translator_model"] = translated.translator_model
            unit_dict["translation_route"] = translated.translation_route
            unit_dict["field_translations"] = translated.field_translations
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
            if route and not route.startswith("skipped_")
            and not route.endswith("_batch")
            and not route.endswith("_batch_fallback")
        )
    )
    full_translated_text = "\n".join(item.translated_text for item in translations if item.translated_text)
    sent_input, sent_instructions = _translation_call_io(llm_calls)

    checkpoint()
    replacement_started = time.perf_counter()
    rendered_image = render_translated_image(
        input_path, translation_units, render_size_mode=render_size_mode, erase_fill_mode=erase_fill_mode
    )
    replacement_wall_ms = _elapsed_ms(replacement_started)

    debug = {
        "request": {
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "grouping_model": grouping_model,
            "translator_model": translator_model,
            "translator_mode": translator_mode,
            "preserve_heuristic_text": preserve_heuristic_text,
            "preserve_unchanged_text": preserve_unchanged_text,
            "use_geometry_columns": use_geometry_columns,
            "render_size_mode": render_size_mode,
            "erase_fill_mode": erase_fill_mode,
            "timings_ms": {
                "ocr": ocr_wall_ms,
                "grouping": grouping_wall_ms,
                "align": align_wall_ms,
                "translation": translation_wall_ms,
                "replacement": replacement_wall_ms,
                "ocr_overlay": ocr_overlay_wall_ms,
                "grouping_overlay": grouping_overlay_wall_ms,
            },
        },
        "grouping": {
            "category": image_category,
            "raw": grouping.hint_raw,
            "hint_units": list(grouping.hint_units),
            "hint_levels": list(grouping.hint_levels),
            "hint_block_ids": list(grouping.hint_block_ids),
            "hint_alignments": list(grouping.hint_alignments),
            "hint_units_adjusted": hint_units_adjusted,  # geometry-injected column `|`s (fed only when use_geometry_columns)
            "field_geometry_changes": field_geometry_changes,
            "units": [unit.to_dict() for unit in grouping.units],
        },
        "translation": [
            {
                "unit_id": item.unit_id,
                "source_text": item.source_text,
                "translated_text": item.translated_text,
                "route": item.translation_route,
                "field_translations": item.field_translations,
            }
            for item in translations
        ],
        "llm_calls": llm_calls,
    }

    image = (projected_overlay_debug.image if projected_overlay_debug else None) or _input_png_bytes(input_path)
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
        "preserve_heuristic_text": preserve_heuristic_text,
        "preserve_unchanged_text": preserve_unchanged_text,
        "use_geometry_columns": use_geometry_columns,
        "render_size_mode": render_size_mode,
        "erase_fill_mode": erase_fill_mode,
        "translation_source": "llm_pool",
        "translation_input": sent_input,
        "translation_instructions": sent_instructions,
        "note": "Returns OCR cells, VLM grouping, per-unit translations, and a Tier-1 re-placement rendering (model-free simple replace).",
    }
    if projected_overlay_debug is not None:
        metadata.update(projected_overlay_debug.metadata)
    metadata["grouping_model"] = grouping.model
    metadata["image_category"] = image_category
    if grouping_overlay_debug is not None:
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
        projected_overlay_debug_image=projected_overlay_debug.image if projected_overlay_debug else None,
        projected_overlay_debug_mime_type=projected_overlay_debug.mime_type if projected_overlay_debug else "image/png",
        grouping_overlay_debug_image=grouping_overlay_debug.image if grouping_overlay_debug else None,
        grouping_overlay_debug_mime_type=grouping_overlay_debug.mime_type if grouping_overlay_debug else "image/png",
        rendered_image=rendered_image,
        rendered_mime_type="image/png",
        segments=cells,
        metadata=metadata,
        metrics={
            "ocr_wall_ms": ocr_wall_ms,
            "grouping_wall_ms": grouping_wall_ms,
            "align_wall_ms": align_wall_ms,
            "translation_wall_ms": translation_wall_ms,
            "replacement_wall_ms": replacement_wall_ms,
            "ocr_overlay_wall_ms": ocr_overlay_wall_ms,
            "grouping_overlay_wall_ms": grouping_overlay_wall_ms,
            "llm_pool_wall_ms": 0.0,
            "translate_image_total_wall_ms": _elapsed_ms(started_at),
            "ocr_segment_count": len(ocr_segments),
            "translation_ocr_segment_count": len(ocr_segments),
            "translation_unit_count": len(grouping.units),
            "translated_unit_count": translated_unit_count,
            "ignored_cell_count": len(grouping.ignored_cell_ids),
            "llm_pool_request_count": 1 + translation_call_count,
        },
        ocr={
            "cells": cells,
            "translation_units": translation_units,
            "ignored_cell_ids": list(grouping.ignored_cell_ids),
            "field_geometry_changes": field_geometry_changes,  # rows where geometry injected a column `|`
        },
        debug=debug,
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


def _translation_call_io(llm_calls: list[dict[str, Any]]) -> tuple[str, str]:
    """The input + instructions of the primary translation call (the structured/batch
    call), so the workbench can show — and copy — exactly what was sent to the LLM.
    Empty when no translation call was made."""
    for call in llm_calls:
        if str(call.get("role") or "").startswith("translation"):
            payload = call.get("payload") or {}
            return str(payload.get("input") or ""), str(payload.get("instructions") or "")
    return "", ""


def _input_png_bytes(input_path: Path) -> bytes:
    from PIL import Image

    out = BytesIO()
    Image.open(input_path).convert("RGB").save(out, format="PNG", compress_level=1)
    return out.getvalue()


def _bool_request_flag(request: dict[str, Any], key: str, *, default: bool) -> bool:
    value = request.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _units_for_preserve_heuristic_text(
    units: list[TranslationUnit],
    *,
    preserve_heuristic_text: bool,
) -> list[TranslationUnit]:
    if preserve_heuristic_text:
        return units
    out: list[TranslationUnit] = []
    for unit in units:
        members = [replace(member, translate=True) for member in unit.members]
        source_text = " ".join(member.text for member in members if member.text).strip()
        out.append(replace(unit, members=members, source_text=source_text))
    return out


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - started_at) * 1000.0)
