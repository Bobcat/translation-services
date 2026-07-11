"""Re-entry pipeline: re-render a prior run's cached translations with new render flags.

The lightest re-entry of the three tasks: :mod:`app.tasks.retranslate_image` skips
VLM/OCR/grouping but still calls the translator; this task also skips translation.
It reloads the source run's cached grouping (``units``) and translation entries,
re-joins them by ``unit_id``, and renders with the requested ``render_size_mode``/
``erase_fill_mode``. No LLM call of any kind — so an A/B of a render flag compares
exactly the render, with zero translation run-variance.
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
from app.tasks.translate_image import _units_for_preserve_heuristic_text


def run_rerender_image_pipeline(
    *,
    settings: AppSettings,
    input_path: Path,
    source_grouping: dict[str, Any],
    source_translation: list[dict[str, Any]],
    request: dict[str, Any],
    checkpoint: Callable[[], None] = lambda: None,
) -> TranslateImageResult:
    del settings
    started_at = time.perf_counter()
    units = [TranslationUnit.from_dict(item) for item in source_grouping.get("units") or []]
    if not units:
        raise RuntimeError("source run has no cached translation units to re-render")

    # The cached ``units`` are pre-filter (translate_image persists the raw align output), so the
    # source run's preserve_heuristic_text flag must be re-applied to reproduce the exact unit set
    # that was translated and rendered. The flag is carried over from the source request by
    # submit_rerender — the body cannot override it (that would need a re-translate).
    preserve_heuristic_text = _bool_request_flag(request, "preserve_heuristic_text", default=True)
    render_size_mode = str(request.get("render_size_mode") or "median").strip() or "median"
    erase_fill_mode = str(request.get("erase_fill_mode") or "inpaint").strip() or "inpaint"
    width_fit_mode = str(request.get("width_fit_mode") or "footprint").strip() or "footprint"
    size_metric_mode = str(request.get("size_metric_mode") or "extent").strip() or "extent"
    size_cohort_mode = str(request.get("size_cohort_mode") or "vlm").strip() or "vlm"
    rendered_units = _units_for_preserve_heuristic_text(
        units, preserve_heuristic_text=preserve_heuristic_text
    )

    translation_by_id = {str(entry.get("unit_id")): entry for entry in source_translation}
    translation_units: list[dict[str, Any]] = []
    for unit in rendered_units:
        unit_dict = unit.to_dict()
        entry = translation_by_id.get(str(unit.id))
        if entry is not None:
            unit_dict["translated_text"] = str(entry.get("translated_text") or "")
            unit_dict["translation_route"] = entry.get("route")
            pairs = entry.get("field_translations")
            unit_dict["field_translations"] = [tuple(pair) for pair in pairs] if pairs else None
        translation_units.append(unit_dict)
    translated_unit_count = sum(1 for u in translation_units if u.get("translated_text"))
    full_translated_text = "\n".join(
        str(u["translated_text"]) for u in translation_units if u.get("translated_text")
    )

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
    )
    replacement_wall_ms = _elapsed_ms(replacement_started)

    # Carry the cached grouping AND translation forward verbatim, so this run is itself a valid
    # source for a further re-render or re-translate (the chain never dead-ends on a render flip).
    debug = {
        "request": {
            "source_lang_code": str(request.get("source_lang_code") or ""),
            "target_lang_code": str(request.get("target_lang_code") or ""),
            "source_request_id": str(request.get("source_request_id") or ""),
            "render_size_mode": render_size_mode,
            "erase_fill_mode": erase_fill_mode,
            "width_fit_mode": width_fit_mode,
            "size_metric_mode": size_metric_mode,
            "size_cohort_mode": size_cohort_mode,
            "preserve_heuristic_text": preserve_heuristic_text,
            "timings_ms": {"replacement": replacement_wall_ms},
        },
        "grouping": source_grouping,
        "translation": source_translation,
    }

    metadata = {
        "translation_alignment": "units_translated_rendered",
        "full_translated_text": full_translated_text,
        "source_lang_code": str(request.get("source_lang_code") or ""),
        "target_lang_code": str(request.get("target_lang_code") or ""),
        "translator_model": str(request.get("translator_model") or ""),
        "translator_mode": str(request.get("translator_mode") or ""),
        "preserve_heuristic_text": preserve_heuristic_text,
        "preserve_unchanged_text": _bool_request_flag(request, "preserve_unchanged_text", default=False),
        "use_geometry_columns": _bool_request_flag(request, "use_geometry_columns", default=True),
        "render_size_mode": render_size_mode,
        "erase_fill_mode": erase_fill_mode,
        "width_fit_mode": width_fit_mode,
        "size_metric_mode": size_metric_mode,
        "size_cohort_mode": size_cohort_mode,
        "translation_source": "cached_rerender",
        "source_request_id": str(request.get("source_request_id") or ""),
        "image_category": str(source_grouping.get("category") or ""),
        "output_rendering_space": "original_input",
        "note": "Re-render of a prior run's cached translations with new render flags (no LLM call).",
    }

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
