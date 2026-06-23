"""Replay one fixture through the deterministic chain: parse hint -> align -> render.

Reuses the live stage functions so behaviour is identical to ``run_translate_image_pipeline``
minus the frozen (VLM / OCR / translator) calls. Returns the align output to diff against the
snapshot and the rendered PNG bytes to re-OCR.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.grouping import group_cells_into_units
from app.grouping.hint_parser import parse_grouping_output
from app.regression.fixture import Fixture
from app.regression.fixture import anchor_key
from app.regression.fixture import expected_unit_of
from app.replacement import render_translated_image
from app.tasks.translate_image import _units_for_preserve_heuristic_text


def replay_fixture(
    input_path: Path,
    fixture: Fixture,
) -> tuple[list[dict[str, Any]], list[int], bytes, dict[str, float]]:
    """``(actual_units, actual_ignored, rendered_png, timings)``. ``actual_units`` is the
    order-sensitive align output (one ``expected_unit_of`` entry per unit); ``rendered_png`` is the
    re-placed image; ``timings`` holds the per-stage wall-clock (ms): ``group_ms`` (parse hint +
    grouping/align) and ``render_ms``."""
    group_started = time.perf_counter()
    hint = parse_grouping_output(fixture.raw_hint)
    grouping = group_cells_into_units(cells=fixture.cells, hint=hint, model=fixture.grouping_model)

    actual_units = [expected_unit_of(unit.to_dict()) for unit in grouping.units]
    actual_ignored = sorted(int(c) for c in grouping.ignored_cell_ids)
    group_ms = (time.perf_counter() - group_started) * 1000.0

    # Only preserve_heuristic_text changes the set fed to render; re-apply it before attaching
    # the frozen translations (keyed by the unit's anchor cell).
    units = _units_for_preserve_heuristic_text(
        grouping.units, preserve_heuristic_text=fixture.preserve_heuristic_text
    )
    translation_units: list[dict[str, Any]] = []
    for unit in units:
        unit_dict = unit.to_dict()
        entry = fixture.translations.get(anchor_key(unit_dict))
        if entry is not None:
            unit_dict["translated_text"] = str(entry.get("translated_text") or "")
            pairs = entry.get("field_translations")
            unit_dict["field_translations"] = [tuple(pair) for pair in pairs] if pairs else None
        translation_units.append(unit_dict)

    render_started = time.perf_counter()
    rendered_png = render_translated_image(input_path, translation_units)
    render_ms = (time.perf_counter() - render_started) * 1000.0
    return actual_units, actual_ignored, rendered_png, {"group_ms": group_ms, "render_ms": render_ms}
