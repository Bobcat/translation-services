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
from io import BytesIO
from pathlib import Path
import time
from typing import Any

from app.core.config import AppSettings
from app.grouping import group_cells_into_units
from app.grouping import request_grouping_hint
from app.grouping.overlay import render_grouping_overlay_debug
from app.grouping.vlm import parse_grouping_output
from app.ocr import resolve_ocr_language
from app.ocr import run_raw_ocr
from app.ocr.overlay import render_original_ocr_overlay_debug
from app.ocr.segment import OcrSegment
from app.replacement import render_translated_image
from app.translation.gold import GoldFixtureError
from app.translation.gold import image_identity
from app.translation.gold import load_fixture_for_image
from app.translation.gold import resolve_gold_units
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
) -> TranslateImageResult:
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
    # Opt-in reference run: simulate BOTH llm-pool responses from the fixture — the VLM hint
    # here (so the units are pinned/deterministic) and the translation below. A normal run
    # calls the VLM as usual. No fixture match -> fail (the caller asked for a reference run).
    fixture_name = str(request.get("translation_fixture") or "").strip()
    gold_fixture = load_fixture_for_image(input_path) if fixture_name else None
    if fixture_name and gold_fixture is None:
        raise GoldFixtureError(
            "translation_fixture run requested but no gold fixture matches this image "
            f"(sha256={image_identity(input_path)})"
        )
    grouping_started_at = time.perf_counter()
    if gold_fixture is not None:
        hint = parse_grouping_output(gold_fixture.vlm_output)
    else:
        hint = request_grouping_hint(
            settings=settings,
            input_path=input_path,
            model=grouping_model,
            call_log=llm_calls,
        )
    grouping_wall_ms = _elapsed_ms(grouping_started_at)

    ocr_language = resolve_ocr_language(settings.ocr, hint.units)

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

    grouping = group_cells_into_units(
        cells=cells,
        hint=hint,
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
    # Reference run: each unit takes its translation from the fixture's reference blocks by
    # hint_index (the canned hint and the reference are 1:1), so no llm-pool call. Unmatched
    # units (aligned to no hint line) stay untranslated.
    gold_unmatched: list[int] = []
    translation_started = time.perf_counter()
    if gold_fixture is not None:
        translations, gold_unmatched = resolve_gold_units(grouping.units, gold_fixture.reference_blocks)
    else:
        translations = translate_units(
            settings=settings,
            units=grouping.units,
            source_lang_code=source_lang,
            target_lang_code=target_lang,
            translator_model=translator_model,
            translator_mode=translator_mode,
            category=image_category,
            hint_units=grouping.hint_units,
            hint_block_ids=grouping.hint_block_ids,
            call_log=llm_calls,
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
            unit_dict["field_translations"] = translated.field_translations
        translation_units.append(unit_dict)
    translated_unit_count = sum(1 for item in translations if item.translated_text)
    # Actual llm-pool calls: one shared batch call (if any unit was batched) + one per
    # batch-fallback + one per plain per-unit (translategemma) translation. A gold-fixture
    # run makes no translation calls.
    routes = [item.translation_route for item in translations]
    translation_call_count = 0 if fixture_name else (
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

    debug = {
        "request": {
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "grouping_model": grouping_model,
            "translator_model": translator_model,
            "translator_mode": translator_mode,
            "timings_ms": {
                "ocr": ocr_wall_ms,
                "grouping": grouping_wall_ms,
                "translation": translation_wall_ms,
                "replacement": replacement_wall_ms,
            },
        },
        "grouping": {
            "category": image_category,
            "raw": grouping.hint_raw,
            "hint_units": list(grouping.hint_units),
            "hint_levels": list(grouping.hint_levels),
            "hint_block_ids": list(grouping.hint_block_ids),
            "hint_alignments": list(grouping.hint_alignments),
            "units": [unit.to_dict() for unit in grouping.units],
        },
        "translation": [
            {
                "unit_id": item.unit_id,
                "source_text": item.source_text,
                "translated_text": item.translated_text,
                "route": item.translation_route,
            }
            for item in translations
        ],
        "llm_calls": llm_calls,
    }

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
        "translation_source": "gold_fixture" if fixture_name else "llm_pool",
        "note": "Returns OCR cells, VLM grouping, per-unit translations, and a Tier-1 re-placement rendering (model-free simple replace).",
    }
    if gold_fixture is not None:
        metadata["gold_fixture"] = gold_fixture.name
        metadata["gold_unmatched_unit_ids"] = gold_unmatched
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
            "grouping_wall_ms": grouping_wall_ms,
            "translation_wall_ms": translation_wall_ms,
            "replacement_wall_ms": replacement_wall_ms,
            "llm_pool_wall_ms": 0.0,
            "translate_image_total_wall_ms": _elapsed_ms(started_at),
            "ocr_segment_count": len(ocr_segments),
            "translation_ocr_segment_count": len(ocr_segments),
            "translation_unit_count": len(grouping.units),
            "translated_unit_count": translated_unit_count,
            "ignored_cell_count": len(grouping.ignored_cell_ids),
            "llm_pool_request_count": 0 if fixture_name else 1 + translation_call_count,
        },
        ocr={
            "cells": cells,
            "translation_units": translation_units,
            "ignored_cell_ids": list(grouping.ignored_cell_ids),
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


def _input_png_bytes(input_path: Path) -> bytes:
    from PIL import Image

    out = BytesIO()
    Image.open(input_path).convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - started_at) * 1000.0)
