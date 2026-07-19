"""The position estimate: anchor confident matches into a monotone chain, then interpolate
every unmatched cell's expected hint position between its anchors (linear top-to-bottom when
no anchors survive).
"""
from __future__ import annotations

from typing import Any

from app.grouping.align.tuning import _LINE_GAP_RATIO
from app.grouping.align.tuning import _LINE_VOVERLAP_RATIO
from app.grouping.align.tuning import _MATCH_THRESHOLD

from app.grouping.align.matching import _Match


def _line_anchor(index: int, cells: list[dict[str, Any]], confident: list[int | None]) -> int | None:
    """The hint line an ambiguous cell takes from its printed-line neighbours: the nearest CONFIDENT
    cell touching it on the left and on the right of the same line. A neighbour qualifies when it
    overlaps the cell vertically (same line, tilt-tolerant) and sits within a word-gap of it — a
    column gap is too wide, so a receipt's far-apart label/amount never link and 2-D rows are left
    to the position logic. Returns the shared label when the present sides agree, else ``None`` (the
    caller falls back to sticky/position). This encodes reading-flow contiguity: a word between two
    cells of one line is part of that line, not of a vertically-nearer neighbour line."""
    box = cells[index].get("bbox") or {}
    left = float(box.get("left", 0.0))
    right = left + float(box.get("width", 0.0))
    top = float(box.get("top", 0.0))
    bottom = top + float(box.get("height", 0.0))
    height = float(box.get("height", 0.0)) or 1.0
    gap_cap = _LINE_GAP_RATIO * height
    slack = 0.3 * height  # tolerate a touch of horizontal overlap at the boundary
    left_label, left_gap = None, gap_cap + 1.0
    right_label, right_gap = None, gap_cap + 1.0
    for other_index, label in enumerate(confident):
        if other_index == index or label is None:
            continue
        other = cells[other_index].get("bbox") or {}
        other_left = float(other.get("left", 0.0))
        other_right = other_left + float(other.get("width", 0.0))
        other_top = float(other.get("top", 0.0))
        other_height = float(other.get("height", 0.0)) or 1.0
        other_bottom = other_top + other_height
        if (min(bottom, other_bottom) - max(top, other_top)) <= _LINE_VOVERLAP_RATIO * min(height, other_height):
            continue  # not on this cell's line
        if other_right <= left + slack:  # neighbour to the LEFT
            gap = left - other_right
            if gap <= gap_cap and gap < left_gap:
                left_label, left_gap = label, gap
        elif other_left >= right - slack:  # neighbour to the RIGHT
            gap = other_left - right
            if gap <= gap_cap and gap < right_gap:
                right_label, right_gap = label, gap
    sides = [label for label in (left_label, right_label) if label is not None]
    if not sides:
        return None
    return sides[0] if all(label == sides[0] for label in sides) else None


def _anchored_positions(
    cells: list[dict[str, Any]],
    matches: list[_Match],
    n_hints: int,
    cell_columns: list[int | None] | None = None,
) -> tuple[list[float], bool]:
    """Each cell's expected hint index, used only to break ties in ``_pick_hint``,
    plus whether the estimate is anchor-based (and so trustworthy enough for the
    position guard).

    Anchor-and-chain, as in OCR<->text alignment literature (RETAS, Yalniz & Manmatha
    2011: unique words anchor the alignment) and seed-chain aligners: every cell whose
    best hint is UNIQUE and confident pins (cell y -> hint index); the longest
    non-decreasing chain of those seeds keeps the map monotone (a rogue match drops
    out); every cell's position is interpolated between its surrounding anchors. The
    global linear estimate is the fallback — it mis-points on pages whose line density
    varies (a dense menu above a sparse footer).

    ``cell_columns`` (layout evidence, multi-column pages only): "lower on the page ->
    further down the hint list" only holds per COLUMN — the hint lists one flow after
    the other while their cells interleave in y, so the global chain drags every cell
    near another column toward that column's indices and the position guard then vetoes
    its true line. A cell with a column id interpolates on its column's own seed chain;
    cells without one (outside text regions, in a spanner, or in a column too seed-poor
    to chain) keep the global chain."""
    tops = [float((cell.get("bbox") or {}).get("top", 0.0)) for cell in cells]
    if not tops or n_hints <= 1:
        return [0.0] * len(cells), False
    seeds = [
        (index, top, match.candidates[0])
        for index, (top, match) in enumerate(zip(tops, matches))
        if len(match.candidates) == 1 and match.score >= _MATCH_THRESHOLD
    ]
    anchors = _chain([(top, hint) for _, top, hint in seeds])
    if len(anchors) < 2:
        return _linear_positions(tops, n_hints), False
    column_chains: dict[int, list[tuple[float, int]]] = {}
    if cell_columns is not None:
        for column_id in {c for c in cell_columns if c is not None}:
            chain = _chain(sorted(
                (top, hint) for index, top, hint in seeds if cell_columns[index] == column_id
            ))
            if len(chain) >= 2:
                column_chains[column_id] = chain
    positions = []
    for index, top in enumerate(tops):
        column_id = cell_columns[index] if cell_columns is not None else None
        chain = column_chains.get(column_id) if column_id is not None else None
        positions.append(_interpolate(top, chain or anchors, n_hints))
    return positions, True


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
