"""Align the VLM grouping hint onto the OCR cells and build units.

The strong VLM groups reliably, so its hint LEADS the structure: each hint unit (a
sentence or a table field, in reading order) becomes one translation unit. OCR cells
supply the text + position; this module assigns each cell to the hint unit it belongs to
and groups consecutive same-assignment cells into a unit.

Assignment is by DISTINCTIVE tokens (each weighted by how rare it is across the hint
units): common words ("de"/"the", in many units) can't mis-assign a cell, and a cell with
no distinctive token (a stopword fragment, OCR noise) inherits the current run instead of
splintering into its own block. Grouping stays CONSECUTIVE so repeated text (a price or
product that recurs on a receipt) forms a separate unit per row, not one merged blob.
Coverage is guaranteed: every cell ends up in exactly one unit.

``translate`` (a whole-cell price/URL/number is not translatable) and ``kind``
(a multi-cell block flows; a single cell is a field) are decided here by small
rules, not by the model.
"""
from __future__ import annotations

import re
import unicodedata
from statistics import median
from typing import Any

from app.grouping.units import GroupingResult
from app.grouping.units import TranslationUnit
from app.grouping.units import UnitMember
from app.grouping.units import union_bbox


_URL_SUFFIX = re.compile(r"\.(com|nl|org|net|io|de|fr|co|eu)\b")


def build_units_from_hint(
    *,
    cells: list[dict[str, Any]],
    hint_units: list[str],
    model: str,
) -> GroupingResult:
    hint_token_sets = [set(_tokens(text)) for text in hint_units]
    df = _token_df(hint_token_sets)

    labels: list[int | None] = []
    prev: int | None = None
    for cell in cells:
        label = _best_hint(cell, hint_token_sets, df)
        if label is None:
            label = prev  # ambiguous / unmatched cell continues the run it sits in
        else:
            prev = label
        labels.append(label)

    groups = _group_by_label(cells, labels)
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


def _token_df(hint_token_sets: list[set[str]]) -> dict[str, int]:
    """Document frequency: how many hint units contain each token."""
    df: dict[str, int] = {}
    for token_set in hint_token_sets:
        for token in token_set:
            df[token] = df.get(token, 0) + 1
    return df


def _best_hint(cell: dict[str, Any], hint_token_sets: list[set[str]], df: dict[str, int]) -> int | None:
    """The hint unit a cell UNIQUELY belongs to, or ``None`` when ambiguous. Tokens are
    weighted 1/df, so a rare word points strongly to its one unit; a common word ("de",
    in many units) scores equally for all of them → a tie → ``None``, so the caller keeps
    the cell in the run where it sits geometrically instead of snapping it to the first
    unit that happens to contain that word."""
    cell_tokens = set(_tokens(str(cell.get("text") or "")))
    if not cell_tokens:
        return None
    scores = [
        sum(1.0 / df[token] for token in cell_tokens if token in hint_set)
        for hint_set in hint_token_sets
    ]
    best = max(scores, default=0.0)
    if best <= 0.0:
        return None
    winners = [index for index, score in enumerate(scores) if best - score < 1e-9]
    return winners[0] if len(winners) == 1 else None


def _group_by_label(
    cells: list[dict[str, Any]], labels: list[int | None]
) -> list[tuple[int | None, list[int]]]:
    """Group cells by hint-unit label into spatially-contiguous units. Same-label cells
    that sit close together (in BOTH axes) form one unit; a cell isolated from the rest —
    a stray edge fragment, or a price/product that recurs on another row — splits into its
    own unit, so it can't drag the unit's anchor across the page or merge two rows. A
    scrambled OCR order therefore can't fragment a unit, yet genuine repeats stay separate.
    Unmatched cells (label ``None``) stay singletons. Cells inside a unit are read
    top-then-left (local to the unit, so a page tilt doesn't scramble them); units come out
    top-to-bottom.
    """
    def top(index: int) -> int:
        return int((cells[index].get("bbox") or {}).get("top") or 0)

    def left(index: int) -> int:
        return int((cells[index].get("bbox") or {}).get("left") or 0)

    heights = [int((cell.get("bbox") or {}).get("height") or 0) for cell in cells]
    positive = [height for height in heights if height > 0]
    gap = 2.5 * median(positive) if positive else 1.0

    by_label: dict[int, list[int]] = {}
    groups: list[tuple[int | None, list[int]]] = []
    for index, label in enumerate(labels):
        if label is None:
            groups.append((None, [index]))
        else:
            by_label.setdefault(label, []).append(index)

    for label, indices in by_label.items():
        for cluster in _spatial_clusters(cells, indices, gap):
            cluster.sort(key=lambda index: (top(index), left(index)))
            groups.append((label, cluster))

    groups.sort(key=lambda group: min(top(index) for index in group[1]))
    return groups


def _spatial_clusters(cells: list[dict[str, Any]], indices: list[int], gap: float) -> list[list[int]]:
    """Single-linkage clusters of cells whose bboxes lie within ``gap`` in BOTH axes, so a
    spatially isolated cell (a far edge fragment, a different column) lands on its own."""
    remaining = list(indices)
    clusters: list[list[int]] = []
    while remaining:
        cluster = [remaining.pop(0)]
        added = True
        while added:
            added = False
            for index in list(remaining):
                if any(_near(cells[index], cells[other], gap) for other in cluster):
                    cluster.append(index)
                    remaining.remove(index)
                    added = True
        clusters.append(cluster)
    return clusters


def _near(a: dict[str, Any], b: dict[str, Any], gap: float) -> bool:
    ba, bb = a.get("bbox") or {}, b.get("bbox") or {}
    ax0, ay0 = int(ba.get("left") or 0), int(ba.get("top") or 0)
    ax1, ay1 = ax0 + int(ba.get("width") or 0), ay0 + int(ba.get("height") or 0)
    bx0, by0 = int(bb.get("left") or 0), int(bb.get("top") or 0)
    bx1, by1 = bx0 + int(bb.get("width") or 0), by0 + int(bb.get("height") or 0)
    horizontal_gap = max(0, max(ax0, bx0) - min(ax1, bx1))
    vertical_gap = max(0, max(ay0, by0) - min(ay1, by1))
    return horizontal_gap <= gap and vertical_gap <= gap


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
