"""Datamodel for stage #5 (grouping): cells -> translation units.

A **cell** is one OCR box. A **translation unit** is a group of cells that form
one coherent translatable piece of text: translate its ``source_text`` as one and
re-place it into the union bbox, scaling the translation to fit.

Re-placement granularity is always the cell: every member keeps its own bbox.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class UnitMember:
    cell_id: int
    text: str
    translate: bool
    bbox: dict[str, int]
    order: int
    polygon: list[dict[str, int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        member: dict[str, Any] = {
            "cell_id": self.cell_id,
            "text": self.text,
            "translate": self.translate,
            "bbox": dict(self.bbox),
            "order": self.order,
        }
        if self.polygon is not None:
            member["polygon"] = [dict(point) for point in self.polygon]
        return member


@dataclass(frozen=True)
class TranslationUnit:
    id: int
    order: int
    members: list[UnitMember]
    bbox: dict[str, int]
    source_text: str
    # Index of the VLM hint line this unit matched (into GroupingResult.hint_units), so
    # the structured translation can map each translated line back onto its unit. None
    # for leftover cells that matched no hint.
    hint_index: int | None = None
    # Visual hierarchy of the matched hint line ("title" | "header" | "body" | "footer"),
    # the id of its element block, and its horizontal alignment ("center", None = left) —
    # the renderer coordinates font sizes per block/level and anchors lines by alignment.
    # None for leftovers and unlabeled lines.
    level: str | None = None
    block_id: int | None = None
    alignment: str | None = None
    # The VLM's per-element typography for this hint line: a named font family and a
    # weight (100-900). The renderer maps the family to an installed face and picks a bold
    # cut when the weight is high. None for leftovers and unlabeled lines.
    font_family: str | None = None
    font_weight: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "order": self.order,
            "bbox": dict(self.bbox),
            "source_text": self.source_text,
            "level": self.level,
            "block_id": self.block_id,
            "alignment": self.alignment,
            "font_family": self.font_family,
            "font_weight": self.font_weight,
            "members": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True)
class GroupingResult:
    units: list[TranslationUnit]
    ignored_cell_ids: list[int]
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | int] = field(default_factory=dict)
    # The raw VLM grouping output and its per-line hint units — carried through so the
    # structured translation can re-translate the whole block structure in one call.
    # ``hint_levels`` / ``hint_block_ids`` are parallel to ``hint_units``.
    hint_raw: str = ""
    hint_units: list[str] = field(default_factory=list)
    hint_levels: list[str | None] = field(default_factory=list)
    hint_block_ids: list[int] = field(default_factory=list)
    hint_alignments: list[str | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "units": [unit.to_dict() for unit in self.units],
            "ignored_cell_ids": list(self.ignored_cell_ids),
            "metadata": dict(self.metadata),
            "metrics": dict(self.metrics),
        }


def union_bbox(bboxes: list[dict[str, int]]) -> dict[str, int]:
    left = min(int(bbox["left"]) for bbox in bboxes)
    top = min(int(bbox["top"]) for bbox in bboxes)
    right = max(int(bbox["left"]) + int(bbox["width"]) for bbox in bboxes)
    bottom = max(int(bbox["top"]) + int(bbox["height"]) for bbox in bboxes)
    return {
        "left": left,
        "top": top,
        "width": max(0, right - left),
        "height": max(0, bottom - top),
    }
