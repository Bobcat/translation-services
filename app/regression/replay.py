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
    grouping = group_cells_into_units(
        cells=fixture.cells,
        hint=hint,
        model=fixture.grouping_model,
        layout_regions=fixture.layout_regions or None,
    )

    # Re-apply preserve_heuristic_text BEFORE building the compared unit list: capture froze the
    # snapshot from the post-filter units (response.ocr.translation_units), so the align diff must
    # compare post-filter too — a fixture captured with the flag off would otherwise be
    # born-failing on member_translate. With the flag on (the default) the filter is the
    # identity, so this is exactly the old comparison for every existing fixture.
    units = _units_for_preserve_heuristic_text(
        grouping.units, preserve_heuristic_text=fixture.preserve_heuristic_text
    )
    actual_units = [expected_unit_of(unit.to_dict()) for unit in units]
    actual_ignored = sorted(int(c) for c in grouping.ignored_cell_ids)
    group_ms = (time.perf_counter() - group_started) * 1000.0
    # Attach the frozen translations without keying on an align OUTPUT. A hint-matched unit takes the
    # translation of the hint line it matched (``hint_translations`` by ``hint_index`` — independent of
    # how cells grouped). A leftover unit (matched no hint line) takes its translation by cell
    # MEMBERSHIP: cells are frozen inputs, so a cell joining/leaving (or the anchor moving on a tilted
    # line) does not detach it. A leftover key cell now ignored, or two keys colliding on one unit (a
    # merge), is a real align change the align diff already flags.
    translation_units = [unit.to_dict() for unit in units]
    for unit_dict in translation_units:
        hint_index = unit_dict.get("hint_index")
        if hint_index is not None:
            entry = fixture.hint_translations.get(str(hint_index))
        else:
            entry = next(
                (e for member in (unit_dict.get("members") or [])
                 if (e := fixture.leftover_translations.get(str(member.get("cell_id")))) is not None),
                None,
            )
        if entry is None:
            continue
        unit_dict["translated_text"] = str(entry.get("translated_text") or "")
        pairs = entry.get("field_translations")
        unit_dict["field_translations"] = [tuple(pair) for pair in pairs] if pairs else None

    render_started = time.perf_counter()
    rendered_png = render_translated_image(
        input_path,
        translation_units,
        render_size_mode=fixture.render_size_mode,
        erase_fill_mode=fixture.erase_fill_mode,
        width_fit_mode=fixture.width_fit_mode,
        size_metric_mode=fixture.size_metric_mode,
        size_cohort_mode=fixture.size_cohort_mode,
    )
    render_ms = (time.perf_counter() - render_started) * 1000.0
    return actual_units, actual_ignored, rendered_png, {"group_ms": group_ms, "render_ms": render_ms}
