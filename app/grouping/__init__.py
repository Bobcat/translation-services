"""Stage #5: cluster OCR cells into translation units.

OCR cells are authoritative (text + bbox). A VLM provides a free-text grouping
*hint* (``app.grouping.vlm``); ``app.grouping.align`` maps that hint back onto the
cells and builds the units, guaranteeing full coverage. ``group_cells_into_units``
is the stage entry.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any

from app.core.config import AppSettings
from app.grouping.align import build_units_from_hint
from app.grouping.units import GroupingResult
from app.grouping.units import TranslationUnit
from app.grouping.units import UnitMember
from app.grouping.vlm import GroupingHintError
from app.grouping.vlm import request_grouping_hint


__all__ = [
    "group_cells_into_units",
    "build_units_from_hint",
    "request_grouping_hint",
    "GroupingHintError",
    "GroupingResult",
    "TranslationUnit",
    "UnitMember",
]


def group_cells_into_units(
    *,
    settings: AppSettings,
    input_path: Path,
    cells: list[dict[str, Any]],
    model: str,
    call_log: list[dict[str, Any]] | None = None,
) -> GroupingResult:
    resolved_model = str(model or "").strip()
    if not resolved_model:
        raise GroupingHintError(
            "grouping_model is required (set llm_pool.grouping_model or pass "
            "grouping_model in the request)"
        )
    if not cells:
        return GroupingResult(
            units=[],
            ignored_cell_ids=[],
            model=resolved_model,
            metrics={"translation_unit_count": 0, "ignored_cell_count": 0},
        )

    started = time.perf_counter()
    hint = request_grouping_hint(
        settings=settings,
        input_path=input_path,
        model=resolved_model,
        call_log=call_log,
    )
    grouping_wall_ms = max(0.0, (time.perf_counter() - started) * 1000.0)

    result = build_units_from_hint(
        cells=cells,
        hint_units=hint.units,
        model=resolved_model,
        hint_levels=hint.levels,
        hint_block_ids=hint.block_ids,
    )
    return replace(
        result,
        hint_raw=hint.raw,
        hint_units=list(hint.units),
        hint_levels=list(hint.levels),
        hint_block_ids=list(hint.block_ids),
        metrics={**result.metrics, "grouping_wall_ms": grouping_wall_ms},
        metadata={
            **result.metadata,
            "grouping_hint_block_count": len(hint.units),
            "category": hint.category,
        },
    )
