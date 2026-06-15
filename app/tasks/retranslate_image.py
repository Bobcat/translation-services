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

from app.core.config import AppSettings
from app.grouping.units import TranslationUnit
from app.replacement import render_translated_image
from app.tasks.translate_image import TranslateImageResult
from app.tasks.translate_image import _translation_call_io
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
) -> TranslateImageResult:
    started_at = time.perf_counter()
    source_lang = str(request.get("source_lang_code") or "").strip()
    target_lang = str(request.get("target_lang_code") or "").strip()
    if not source_lang:
        raise RuntimeError("source_lang_code is required for translation routing")

    units = [TranslationUnit.from_dict(item) for item in source_grouping.get("units") or []]
    if not units:
        raise RuntimeError("source run has no cached translation units to re-translate")
    hint_units = [str(line) for line in source_grouping.get("hint_units") or []]
    hint_block_ids = [int(block_id) for block_id in source_grouping.get("hint_block_ids") or []]
    image_category = str(source_grouping.get("category") or "")

    translator_model = str(request.get("translator_model") or "").strip() or settings.llm_pool.translator_model
    translator_mode = str(request.get("translator_mode") or "").strip() or settings.llm_pool.translator_mode
    prompt = resolve_structured_prompt(
        store_for(settings.service.prompts_root),
        raw_prompt=request.get("translation_prompt"),
        prompt_id=request.get("translation_prompt_id"),
        default_id=IMAGE_DEFAULT_ID,
    )

    llm_calls: list[dict[str, Any]] = []
    translation_started = time.perf_counter()
    translations = translate_units(
        settings=settings,
        units=units,
        source_lang_code=source_lang,
        target_lang_code=target_lang,
        translator_model=translator_model,
        translator_mode=translator_mode,
        category=image_category,
        hint_units=hint_units,
        hint_block_ids=hint_block_ids,
        prompt=prompt,
        call_log=llm_calls,
    )
    translation_wall_ms = _elapsed_ms(translation_started)

    translation_by_id = {item.unit_id: item for item in translations}
    translation_units: list[dict[str, Any]] = []
    for unit in units:
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

    replacement_started = time.perf_counter()
    rendered_image = render_translated_image(input_path, translation_units)
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
            "timings_ms": {
                "translation": translation_wall_ms,
                "replacement": replacement_wall_ms,
            },
        },
        "grouping": {
            "category": image_category,
            "hint_units": list(hint_units),
            "hint_block_ids": list(hint_block_ids),
            "units": [unit.to_dict() for unit in units],
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

    metadata = {
        "translation_alignment": "units_translated_rendered",
        "full_translated_text": full_translated_text,
        "source_lang_code": source_lang,
        "target_lang_code": target_lang,
        "translator_model": translator_model,
        "translator_mode": translator_mode,
        "translation_source": "llm_pool_retranslate",
        "source_request_id": str(request.get("source_request_id") or ""),
        "image_category": image_category,
        "translation_input": sent_input,
        "translation_instructions": sent_instructions,
        "output_rendering_space": "original_input",
        "note": "Re-translate of a prior run's cached units with an alternative prompt (no VLM/OCR/grouping).",
    }

    return TranslateImageResult(
        image=_input_png_bytes(input_path),
        mime_type="image/png",
        rendered_image=rendered_image,
        rendered_mime_type="image/png",
        segments=[unit.to_dict() for unit in units],
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
