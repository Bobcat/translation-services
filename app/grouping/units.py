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
    # Exact em size of this member's text in image pixels, when the cell source
    # knows it (a text layer's declared size); None = derive from ink geometry.
    size_px: float | None = None
    # Inline pixel islands (⟦Mn⟧ tokens in the text): [{"id": "Mn", "bbox": px}].
    # Opaque data from the cell layer; the render transplants the source pixels.
    islands: list[dict[str, Any]] | None = None

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
        if self.size_px is not None:
            member["size_px"] = self.size_px
        if self.islands is not None:
            member["islands"] = [dict(island) for island in self.islands]
        return member

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnitMember":
        polygon = data.get("polygon")
        size_px = data.get("size_px")
        return cls(
            cell_id=int(data["cell_id"]),
            text=str(data.get("text") or ""),
            translate=bool(data.get("translate", True)),
            bbox={key: int(value) for key, value in dict(data["bbox"]).items()},
            order=int(data.get("order") or 0),
            polygon=[{key: int(value) for key, value in dict(point).items()} for point in polygon]
            if polygon is not None
            else None,
            size_px=float(size_px) if size_px is not None else None,
            islands=[dict(island) for island in data["islands"]] if data.get("islands") else None,
        )


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
    # The VLM's font-size label ("<n>pt", None when unlabeled). A poor ABSOLUTE (pt->pixel scale
    # drifts per image) but a reliable EQUALITY signal — same pt = same intended size — used only
    # by the size-cohort render mode to regularise the noisy per-line OCR sizing.
    font_size: int | None = None
    # True when the hint line was a bullet-list item, plus the glyph/marker the VLM saw ("•", "-",
    # "1.", "(a)", ...). The renderer redraws "<marker> <text>" (or, in legacy mode, insets the text
    # past the original glyph left in the image so the translation does not overwrite it).
    bullet: bool = False
    bullet_marker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "order": self.order,
            "bbox": dict(self.bbox),
            "source_text": self.source_text,
            "hint_index": self.hint_index,
            "level": self.level,
            "block_id": self.block_id,
            "alignment": self.alignment,
            "font_family": self.font_family,
            "font_weight": self.font_weight,
            "font_size": self.font_size,
            "bullet": self.bullet,
            "bullet_marker": self.bullet_marker,
            "members": [member.to_dict() for member in self.members],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranslationUnit":
        hint_index = data.get("hint_index")
        block_id = data.get("block_id")
        font_weight = data.get("font_weight")
        font_size = data.get("font_size")
        return cls(
            id=int(data["id"]),
            order=int(data.get("order") or 0),
            members=[UnitMember.from_dict(member) for member in data.get("members") or []],
            bbox={key: int(value) for key, value in dict(data["bbox"]).items()},
            source_text=str(data.get("source_text") or ""),
            hint_index=int(hint_index) if hint_index is not None else None,
            level=data.get("level"),
            block_id=int(block_id) if block_id is not None else None,
            alignment=data.get("alignment"),
            font_family=data.get("font_family"),
            font_weight=int(font_weight) if font_weight is not None else None,
            font_size=int(font_size) if font_size is not None else None,
            bullet=bool(data.get("bullet", False)),
            bullet_marker=data.get("bullet_marker"),
        )


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


def _cell_box(cell: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = cell.get("bbox") or {}
    left = float(bbox.get("left") or 0.0)
    top = float(bbox.get("top") or 0.0)
    return left, top, left + float(bbox.get("width") or 0.0), top + float(bbox.get("height") or 0.0)


def _near(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Two boxes are adjacent enough to be one wrapped element: x-ranges overlap and they are
    vertically close (stacked onto the next row), or y-ranges overlap and they are horizontally
    close (split across one line). A far-off box (an embedded image, a far column) is neither."""
    height = max(a[3] - a[1], b[3] - b[1], 1.0)
    x_overlap = min(a[2], b[2]) > max(a[0], b[0])
    y_overlap = min(a[3], b[3]) > max(a[1], b[1])
    y_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    x_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    stacked = x_overlap and y_gap <= 1.5 * height       # wrapped onto the next line
    same_line = y_overlap and x_gap <= 2.0 * height     # split across one line ("... nooit.")
    return stacked or same_line
