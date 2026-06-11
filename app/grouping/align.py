"""Align the VLM grouping hint onto the authoritative OCR cells and build units.

OCR cells are the source of truth for positions; the VLM hint is the source of
truth for text/structure. This module assigns each cell to the hint line it
best matches (normalised token overlap, tolerant to OCR garble), breaking ties
by an anchor-interpolated reading position (``_anchored_positions``), groups
consecutive cells with the same assignment into a unit, and turns every
unmatched cell into its own unit. Because we build the units, coverage is guaranteed: every cell
ends up in exactly one unit, so a weak/incomplete hint lowers quality but never
fails the job.

``translate`` (a whole-cell price/URL/number is not translatable) is decided here
by small rules, not by the model.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from app.grouping.units import GroupingResult
from app.grouping.units import TranslationUnit
from app.grouping.units import UnitMember
from app.grouping.units import union_bbox


_MATCH_THRESHOLD = 0.4
_FUZZY_MIN_LEN = 4
_FUZZY_RATIO = 0.8
_URL_SUFFIX = re.compile(r"\.(com|nl|org|net|io|de|fr|co|eu)\b")


def build_units_from_hint(
    *,
    cells: list[dict[str, Any]],
    hint_units: list[str],
    model: str,
    hint_levels: list[str | None] | None = None,
    hint_block_ids: list[int] | None = None,
) -> GroupingResult:
    hint_token_sets = [set(_tokens(text)) for text in hint_units]
    matches = [_match_scores(cell, hint_token_sets) for cell in cells]
    positions = _anchored_positions(cells, matches, len(hint_units))
    labels = [
        _pick_hint(match, preferred_index=pos)
        for match, pos in zip(matches, positions)
    ]

    groups = _group_consecutive(labels)
    units = [
        _build_unit(
            cells=cells,
            indices=indices,
            unit_id=order,
            hint_index=label,
            level=_hint_meta(label, hint_levels),
            block_id=_hint_meta(label, hint_block_ids),
        )
        for order, (label, indices) in enumerate(groups, start=1)
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


@dataclass(frozen=True)
class _Match:
    candidates: list[int]  # hint indices sharing the best token-overlap score
    score: float           # best matched-token mass / cell token count


def _match_scores(cell: dict[str, Any], hint_token_sets: list[set[str]]) -> _Match:
    cell_tokens = _tokens(str(cell.get("text") or ""))
    if not cell_tokens:
        return _Match(candidates=[], score=0.0)
    matched_by_index: list[tuple[int, float]] = []
    best_matched = 0.0
    for index, hint_set in enumerate(hint_token_sets):
        if not hint_set:
            continue
        matched = sum(_token_score(token, hint_set) for token in cell_tokens)
        if matched:
            matched_by_index.append((index, matched))
            if matched > best_matched:
                best_matched = matched
    if not best_matched:
        return _Match(candidates=[], score=0.0)
    return _Match(
        candidates=[index for index, matched in matched_by_index if matched == best_matched],
        score=best_matched / len(cell_tokens),
    )


def _pick_hint(match: _Match, *, preferred_index: float = 0.0) -> int | None:
    if not match.candidates or match.score < _MATCH_THRESHOLD:
        return None
    # Several hint lines can match a short cell equally (two dishes ending "en frites").
    # Break the tie by reading position: bind the cell to the hint nearest its place on the
    # page, not just the first in the list.
    return min(match.candidates, key=lambda index: abs(index - preferred_index))


def _token_score(token: str, hint_set: set[str]) -> float:
    """1.0 exact, else fuzzy (slightly lower, so exact wins a tie) for OCR garble: the cell
    must still bind to its clean VLM line when OCR splits a word ("Kaar thouder" vs
    "Kaarthouder") or drops/adds a character ("AHNEDAARDBEI" vs "AHNEDAARBEI") — otherwise
    the cell becomes a leftover, the per-unit fallback translates the garbled text in
    isolation, and the good structured translation of the VLM line is orphaned. Fuzzy =
    substring or high character similarity, both only for tokens long enough that they
    cannot collide by chance; below exact so "Kaart" still binds its own line, not
    "Kaarthouder"."""
    if token in hint_set:
        return 1.0
    if len(token) < _FUZZY_MIN_LEN:
        return 0.0
    for hint_token in hint_set:
        if len(hint_token) < _FUZZY_MIN_LEN:
            continue
        if token in hint_token or hint_token in token:
            return 0.9
        shorter, longer = sorted((len(token), len(hint_token)))
        if 2 * shorter / (shorter + longer) < _FUZZY_RATIO:  # ratio can't reach the bar
            continue
        if difflib.SequenceMatcher(None, token, hint_token).ratio() >= _FUZZY_RATIO:
            return 0.9
    return 0.0


def _anchored_positions(
    cells: list[dict[str, Any]], matches: list[_Match], n_hints: int
) -> list[float]:
    """Each cell's expected hint index, used only to break ties in ``_pick_hint``.

    Anchor-and-chain, as in OCR<->text alignment literature (RETAS, Yalniz & Manmatha
    2011: unique words anchor the alignment) and seed-chain aligners: every cell whose
    best hint is UNIQUE and confident pins (cell y -> hint index); the longest
    non-decreasing chain of those seeds keeps the map monotone (a rogue match drops
    out); every cell's position is interpolated between its surrounding anchors. The
    global linear estimate is the fallback — it mis-points on pages whose line density
    varies (a dense menu above a sparse footer)."""
    tops = [float((cell.get("bbox") or {}).get("top", 0.0)) for cell in cells]
    if not tops or n_hints <= 1:
        return [0.0] * len(cells)
    seeds = [
        (top, match.candidates[0])
        for top, match in zip(tops, matches)
        if len(match.candidates) == 1 and match.score >= _MATCH_THRESHOLD
    ]
    anchors = _chain(seeds)
    if len(anchors) < 2:
        return _linear_positions(tops, n_hints)
    return [_interpolate(top, anchors, n_hints) for top in tops]


def _chain(seeds: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """Longest non-decreasing subsequence of the seeds' hint indices (seeds are in
    reading order), deduplicated to strictly increasing y for interpolation."""
    if not seeds:
        return []
    best_len = [1] * len(seeds)
    prev = [-1] * len(seeds)
    for i in range(len(seeds)):
        for j in range(i):
            if seeds[j][1] <= seeds[i][1] and best_len[j] + 1 > best_len[i]:
                best_len[i] = best_len[j] + 1
                prev[i] = j
    index = max(range(len(seeds)), key=lambda i: best_len[i])
    chain: list[tuple[float, int]] = []
    while index != -1:
        chain.append(seeds[index])
        index = prev[index]
    chain.reverse()
    anchors: list[tuple[float, int]] = []
    for top, hint_index in chain:
        if not anchors or top > anchors[-1][0]:
            anchors.append((top, hint_index))
    return anchors


def _interpolate(top: float, anchors: list[tuple[float, int]], n_hints: int) -> float:
    """Piecewise-linear hint index at ``top``; outside the anchor range the nearest
    segment extrapolates. Clamped to the valid index range."""
    if top <= anchors[0][0]:
        (y0, i0), (y1, i1) = anchors[0], anchors[1]
    elif top >= anchors[-1][0]:
        (y0, i0), (y1, i1) = anchors[-2], anchors[-1]
    else:
        for (y0, i0), (y1, i1) in zip(anchors, anchors[1:]):
            if y0 <= top <= y1:
                break
    position = float(i0) if y1 == y0 else i0 + (top - y0) / (y1 - y0) * (i1 - i0)
    return min(max(position, 0.0), float(n_hints - 1))


def _linear_positions(tops: list[float], n_hints: int) -> list[float]:
    y_min, y_max = min(tops), max(tops)
    span = (y_max - y_min) or 1.0
    return [((top - y_min) / span) * (n_hints - 1) for top in tops]


def _group_consecutive(labels: list[int | None]) -> list[tuple[int | None, list[int]]]:
    groups: list[tuple[int | None, list[int]]] = []
    for cell_index, label in enumerate(labels):
        if label is not None and groups and groups[-1][0] == label:
            groups[-1][1].append(cell_index)
        else:
            groups.append((label, [cell_index]))
    return groups


def _hint_meta(label: int | None, values: list | None):
    if label is None or values is None or not (0 <= label < len(values)):
        return None
    return values[label]


def _build_unit(
    *,
    cells: list[dict[str, Any]],
    indices: list[int],
    unit_id: int,
    hint_index: int | None = None,
    level: str | None = None,
    block_id: int | None = None,
) -> TranslationUnit:
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
    return TranslationUnit(
        id=unit_id,
        order=unit_id,
        members=members,
        bbox=bbox,
        source_text=source_text,
        hint_index=hint_index,
        level=level,
        block_id=block_id,
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
