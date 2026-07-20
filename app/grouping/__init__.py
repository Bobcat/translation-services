"""Stage #5: cluster OCR cells into translation units.

OCR cells are authoritative (text + bbox). A VLM provides a free-text grouping
*hint* (``app.grouping.vlm``); the caller requests that hint up front — it is
image-only, and the pipeline also routes the OCR model on its script content —
and passes it in here. ``app.grouping.align`` maps the hint back onto the cells
and builds the units, guaranteeing full coverage. ``group_cells_into_units`` is
the stage entry.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.grouping.align import build_units_from_hint
from app.grouping.align.overclaim import trim_overclaimed_hint_lines
from app.grouping.units import GroupingResult
from app.grouping.units import TranslationUnit
from app.grouping.units import UnitMember
from app.grouping.hint_parser import GroupingHint
from app.grouping.vlm import GroupingHintError
from app.grouping.vlm import request_grouping_hint


__all__ = [
    "group_cells_into_units",
    "build_units_from_hint",
    "request_grouping_hint",
    "GroupingHint",
    "GroupingHintError",
    "GroupingResult",
    "TranslationUnit",
    "UnitMember",
]


def group_cells_into_units(
    *,
    cells: list[dict[str, Any]],
    hint: GroupingHint,
    model: str,
    layout_regions: list[dict[str, Any]] | None = None,
    preserve_image_regions: bool = True,
) -> GroupingResult:
    resolved_model = str(model or "").strip()
    if not cells:
        return GroupingResult(
            units=[],
            ignored_cell_ids=[],
            model=resolved_model,
            metrics={"translation_unit_count": 0, "ignored_cell_count": 0},
        )

    result = build_units_from_hint(
        cells=cells,
        hint_units=hint.units,
        model=resolved_model,
        hint_levels=hint.levels,
        hint_block_ids=hint.block_ids,
        hint_alignments=hint.alignments,
        hint_families=hint.font_families,
        hint_weights=hint.font_weights,
        hint_sizes=hint.font_sizes,
        hint_bullets=hint.bullets,
        hint_bullet_markers=hint.bullet_markers,
        layout_regions=layout_regions,
        preserve_image_regions=preserve_image_regions,
    )
    # The hint lines the TRANSLATOR sees, with any suffix a line swallowed from the next one
    # removed (align/overclaim.py). Runs here, on the built units, because the cells are the
    # evidence for what a line may carry; the list keeps its length so every parallel hint
    # list stays indexable by hint_index.
    return replace(
        result,
        hint_raw=hint.raw,
        hint_units=trim_overclaimed_hint_lines(result.units, list(hint.units)),
        hint_levels=list(hint.levels),
        hint_block_ids=list(hint.block_ids),
        hint_alignments=list(hint.alignments),
        metadata={
            **result.metadata,
            "grouping_hint_block_count": len(hint.units),
            "category": hint.category,
        },
    )
