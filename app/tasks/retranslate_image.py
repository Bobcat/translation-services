"""Re-entry pipeline: re-translate a prior run's cached units with a new prompt.

The normal :mod:`app.tasks.translate_image` flow runs the heavy stages — VLM hint,
OCR, grouping — to turn an image into translation units. Those stages are
deterministic enough to reuse, so this task skips them: it reloads the cached
grouping (``units`` + ``hint_units`` + ``hint_block_ids`` + ``category``) of a
prior completed run, sends the SAME units through translation with an alternative
``translation_prompt``, re-aligns the response and re-renders. No VLM/OCR/grouping
call is made.

This exists to experiment with prompts so the translation fits the space the
original text occupies, without paying for the VLM hint every iteration.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import time
from typing import Any
from typing import Callable

from app.core.config import AppSettings
from app.grouping.units import TranslationUnit
from app.replacement import render_translated_image
from app.tasks.translate_image import TranslateImageResult
from app.tasks.translate_image import _bool_request_flag
from app.tasks.translate_image import _translation_call_io
from app.tasks.translate_image import _units_for_preserve_heuristic_text
from app.translation.prompts import resolve_structured_prompt
from app.translation.prompts import store_for
from app.translation.prompts.templates import IMAGE_DEFAULT_ID
from app.translation.translate import translate_units


def run_retranslate_image_pipeline(
    *,
    settings: AppSettings,
    input_path: Path,
    source_grouping: dict[str, Any],
    request: dict[str, Any],
    checkpoint: Callable[[], None] = lambda: None,
) -> TranslateImageResult:
    started_at = time.perf_counter()
    source_lang = str(request.get("source_lang_code") or "").strip()
    target_lang = str(request.get("target_lang_code") or "").strip()
    if not source_lang:
        raise RuntimeError("source_lang_code is required for translation routing")

    use_geometry_columns = _bool_request_flag(request, "use_geometry_columns", default=True)
    preserve_image_regions = _bool_request_flag(request, "preserve_image_regions", default=True)
    # When the align inputs were frozen (OCR cells + hint + layout regions), RE-ALIGN here so a
    # grouping flag like preserve_image_regions takes effect without re-running VLM/OCR. Re-align
    # is deterministic on the same inputs, so a default (preserve on) re-translate rebuilds the
    # exact same units the source run had — while preserve OFF turns image/chart text (a figure
    # that is really a table) into units, and toggling back ON drops it again. Older runs without
    # cached cells fall back to the cached units (the previous behaviour).
    cached_cells = source_grouping.get("cells")
    if cached_cells:
        from app.grouping import group_cells_into_units
        from app.grouping.hint_parser import parse_grouping_output

        units = list(
            group_cells_into_units(
                cells=cached_cells,
                hint=parse_grouping_output(str(source_grouping.get("raw") or "")),
                model=str(source_grouping.get("model") or ""),
                layout_regions=source_grouping.get("layout_regions") or None,
                preserve_image_regions=preserve_image_regions,
            ).units
        )
    else:
        units = [TranslationUnit.from_dict(item) for item in source_grouping.get("units") or []]
    if not units:
        raise RuntimeError("source run has no cached translation units to re-translate")
    raw_hint_units = [str(line) for line in source_grouping.get("hint_units") or []]
    adjusted_hint_units = [str(line) for line in source_grouping.get("hint_units_adjusted") or []]
    # Feed translation the same hint variant the source run fed (translate_image uses the
    # geometry-adjusted lines when use_geometry_columns is on — the default): re-translating
    # with the RAW lines would silently change the translation input, so a prompt A/B would
    # compare more than the prompt. An older grouping.json without the adjusted key (or a
    # non-parallel one) falls back to the raw lines.
    hint_units = (
        adjusted_hint_units
        if use_geometry_columns and adjusted_hint_units and len(adjusted_hint_units) == len(raw_hint_units)
        else raw_hint_units
    )
    hint_block_ids = [int(block_id) for block_id in source_grouping.get("hint_block_ids") or []]
    image_category = str(source_grouping.get("category") or "")

    translator_model = str(request.get("translator_model") or "").strip() or settings.llm_pool.translator_model
    translator_mode = str(request.get("translator_mode") or "").strip() or settings.llm_pool.translator_mode
    preserve_heuristic_text = _bool_request_flag(request, "preserve_heuristic_text", default=True)
    preserve_unchanged_text = _bool_request_flag(request, "preserve_unchanged_text", default=False)
    render_size_mode = str(request.get("render_size_mode") or "median").strip() or "median"
    erase_fill_mode = str(request.get("erase_fill_mode") or "inpaint").strip() or "inpaint"
    width_fit_mode = str(request.get("width_fit_mode") or "footprint").strip() or "footprint"
    size_metric_mode = str(request.get("size_metric_mode") or "extent").strip() or "extent"
    size_cohort_mode = str(request.get("size_cohort_mode") or "off").strip() or "off"
    units_for_translation = _units_for_preserve_heuristic_text(
        units,
        preserve_heuristic_text=preserve_heuristic_text,
    )
    prompt = resolve_structured_prompt(
        store_for(settings.service.prompts_root),
        raw_prompt=request.get("translation_prompt"),
        prompt_id=request.get("translation_prompt_id"),
        default_id=IMAGE_DEFAULT_ID,
    )

    llm_calls: list[dict[str, Any]] = []
    checkpoint()
    translation_started = time.perf_counter()
    translations = translate_units(
        settings=settings,
        units=units_for_translation,
        source_lang_code=source_lang,
        target_lang_code=target_lang,
        translator_model=translator_model,
        translator_mode=translator_mode,
        category=image_category,
        hint_units=hint_units,
        hint_block_ids=hint_block_ids,
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
    full_translated_text = "\n".join(item.translated_text for item in translations if item.translated_text)
    sent_input, sent_instructions = _translation_call_io(llm_calls)

    checkpoint()
    replacement_started = time.perf_counter()
    rendered_image = render_translated_image(
        input_path,
        translation_units,
        render_size_mode=render_size_mode,
        erase_fill_mode=erase_fill_mode,
        width_fit_mode=width_fit_mode,
        size_metric_mode=size_metric_mode,
        size_cohort_mode=size_cohort_mode,
        preserve_unchanged_text=preserve_unchanged_text,
        image_category=image_category,
        layout_regions=source_grouping.get("layout_regions") or [],
        target_lang=target_lang,
    )
    replacement_wall_ms = _elapsed_ms(replacement_started)

    # Carry the cached grouping forward so this run is itself a valid re-translate source.
    debug = {
        "request": {
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "source_request_id": str(request.get("source_request_id") or ""),
            "translation_prompt_id": prompt.id,
            "translation_prompt": prompt.system,
            "translator_model": translator_model,
            "translator_mode": translator_mode,
            "preserve_heuristic_text": preserve_heuristic_text,
            "preserve_unchanged_text": preserve_unchanged_text,
            "timings_ms": {
                "translation": translation_wall_ms,
                "replacement": replacement_wall_ms,
            },
        },
        "grouping": {
            "category": image_category,
            "raw": str(source_grouping.get("raw") or ""),
            "hint_units": list(hint_units),
            "hint_block_ids": list(hint_block_ids),
            "units": [unit.to_dict() for unit in units],
            # Carry the align inputs forward so a further re-entry can re-align from this run too.
            "cells": cached_cells or source_grouping.get("cells"),
            "layout_regions": source_grouping.get("layout_regions"),
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

    metadata = {
        "translation_alignment": "units_translated_rendered",
        "full_translated_text": full_translated_text,
        "source_lang_code": source_lang,
        "target_lang_code": target_lang,
        "translator_model": translator_model,
        "translator_mode": translator_mode,
        "preserve_heuristic_text": preserve_heuristic_text,
        "preserve_unchanged_text": preserve_unchanged_text,
        "use_geometry_columns": use_geometry_columns,
        "preserve_image_regions": preserve_image_regions,
        "render_size_mode": render_size_mode,
        "erase_fill_mode": erase_fill_mode,
        "width_fit_mode": width_fit_mode,
        "size_metric_mode": size_metric_mode,
        "size_cohort_mode": size_cohort_mode,
        "translation_source": "llm_pool_retranslate",
        "source_request_id": str(request.get("source_request_id") or ""),
        "image_category": image_category,
        "translation_input": sent_input,
        "translation_instructions": sent_instructions,
        "output_rendering_space": "original_input",
        "note": "Re-translate of a prior run's cached units with an alternative prompt (no VLM/OCR/grouping).",
    }

    # ``segments`` keeps the SAME cell shape translate_image emits ({id, text, bbox}): a consumer
    # written against one task must not break on the other. The cells come from the cached units'
    # members (OCR confidence was not cached, so that key is absent).
    segment_cells = [
        {"id": member.cell_id, "text": member.text, "bbox": dict(member.bbox or {})}
        for unit in units
        for member in unit.members
        if member.bbox
    ]

    return TranslateImageResult(
        image=_input_png_bytes(input_path),
        mime_type="image/png",
        rendered_image=rendered_image,
        rendered_mime_type="image/png",
        segments=segment_cells,
        metadata=metadata,
        metrics={
            "translation_wall_ms": translation_wall_ms,
            "replacement_wall_ms": replacement_wall_ms,
            "translate_image_total_wall_ms": _elapsed_ms(started_at),
            "translation_unit_count": len(units),
            "translated_unit_count": translated_unit_count,
        },
        ocr={"translation_units": translation_units},
        debug=debug,
    )


def _input_png_bytes(input_path: Path) -> bytes:
    from PIL import Image

    out = BytesIO()
    Image.open(input_path).convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - started_at) * 1000.0)
