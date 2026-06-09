"""Align the VLM grouping hint onto the authoritative OCR cells and build units.

OCR cells are the source of truth (text + bbox). The VLM hint is a list of
text-block strings in reading order. This module assigns each cell to the hint
block it best matches (normalised token overlap), groups consecutive cells with
the same assignment into a unit, and turns every unmatched cell into its own
``field`` unit. Because we build the units, coverage is guaranteed: every cell
ends up in exactly one unit, so a weak/incomplete hint lowers quality but never
fails the job.

``translate`` (a whole-cell price/URL/number is not translatable) and ``kind``
(a multi-cell block flows; a single cell is a field) are decided here by small
rules, not by the model.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.grouping.units import GroupingResult
from app.grouping.units import TranslationUnit
from app.grouping.units import UnitMember
from app.grouping.units import union_bbox


_MATCH_THRESHOLD = 0.4
_URL_SUFFIX = re.compile(r"\.(com|nl|org|net|io|de|fr|co|eu)\b")


def build_units_from_hint(
    *,
    cells: list[dict[str, Any]],
    hint_units: list[str],
    model: str,
) -> GroupingResult:
    hint_token_sets = [set(_tokens(text)) for text in hint_units]
    labels = [_best_hint(cell, hint_token_sets) for cell in cells]

    groups = _group_consecutive(labels)
    units = [
        _build_unit(cells=cells, indices=indices, unit_id=order)
        for order, (_, indices) in enumerate(groups, start=1)
    ]
    leftover = sum(1 for label, _ in groups if label is None)

    return GroupingResult(
        units=units,
        ignored_cell_ids=[],
        model=model,
        metrics={
            "translation_unit_count": len(units),
            "ignored_cell_count": 0,
            "hint_block_count": len(hint_units),
            "leftover_unit_count": leftover,
            "translatable_member_count": sum(
                1 for unit in units for member in unit.members if member.translate
            ),
        },
    )


def _best_hint(cell: dict[str, Any], hint_token_sets: list[set[str]]) -> int | None:
    cell_tokens = _tokens(str(cell.get("text") or ""))
    if not cell_tokens:
        return None
    best_index: int | None = None
    best_score = 0.0
    for index, hint_set in enumerate(hint_token_sets):
        if not hint_set:
            continue
        score = sum(1 for token in cell_tokens if token in hint_set) / len(cell_tokens)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score >= _MATCH_THRESHOLD else None


def _group_consecutive(labels: list[int | None]) -> list[tuple[int | None, list[int]]]:
    groups: list[tuple[int | None, list[int]]] = []
    for cell_index, label in enumerate(labels):
        if label is not None and groups and groups[-1][0] == label:
            groups[-1][1].append(cell_index)
        else:
            groups.append((label, [cell_index]))
    return groups


def _build_unit(*, cells: list[dict[str, Any]], indices: list[int], unit_id: int) -> TranslationUnit:
    members: list[UnitMember] = []
    for order, cell_index in enumerate(indices, start=1):
        cell = cells[cell_index]
        text = str(cell.get("text") or "")
        polygon = cell.get("polygon")
        members.append(
            UnitMember(
                cell_id=int(cell["id"]),
                text=text,
                translate=not _is_nontranslatable(text),
                bbox=dict(cell.get("bbox") or {}),
                order=order,
                polygon=[dict(point) for point in polygon] if polygon else None,
            )
        )
    bbox = union_bbox([member.bbox for member in members if member.bbox]) if members else {}
    source_text = " ".join(member.text for member in members if member.translate and member.text)
    kind = "flow" if len(members) > 1 else "field"
    return TranslationUnit(
        id=unit_id,
        order=unit_id,
        kind=kind,  # type: ignore[arg-type]
        members=members,
        bbox=bbox,
        source_text=source_text,
    )


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", stripped)


def _is_nontranslatable(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if "://" in lowered or lowered.startswith("www.") or _URL_SUFFIX.search(lowered):
        return True
    return not any(char.isalpha() for char in stripped)
