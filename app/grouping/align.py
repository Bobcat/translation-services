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

from dataclasses import dataclass
from typing import Any

from app.grouping.heuristics import _is_continuation
from app.grouping.heuristics import _is_icon_fragment
from app.grouping.heuristics import _is_nontranslatable
from app.grouping.heuristics import _near
from app.grouping.tokens import _token_score
from app.grouping.tokens import _tokens
from app.grouping.units import GroupingResult
from app.grouping.units import TranslationUnit
from app.grouping.units import UnitMember
from app.grouping.units import union_bbox


_MATCH_THRESHOLD = 0.4
_POSITION_GUARD = 3.0


def build_units_from_hint(
    *,
    cells: list[dict[str, Any]],
    hint_units: list[str],
    model: str,
    hint_levels: list[str | None] | None = None,
    hint_block_ids: list[int] | None = None,
    hint_alignments: list[str | None] | None = None,
    hint_families: list[str | None] | None = None,
    hint_weights: list[int | None] | None = None,
    hint_bullets: list[bool] | None = None,
) -> GroupingResult:
    hint_token_sets = [set(_tokens(text)) for text in hint_units]
    token_to_hints = _build_hint_index(hint_token_sets)
    matches = [
        _match_scores(cell, hint_token_sets, _candidate_hints(cell, token_to_hints))
        for cell in cells
    ]
    positions, positions_anchored = _anchored_positions(cells, matches, len(hint_units))
    labels: list[int | None] = []
    previous_label: int | None = None
    previous_cell: dict[str, Any] | None = None
    for cell, match, position in zip(cells, matches, positions):
        sticky = previous_label if _is_continuation(previous_cell, cell) else None
        label = _pick_hint(
            match,
            preferred_index=position,
            sticky=sticky,
            position_reliable=positions_anchored,
        )
        labels.append(label)
        if label is not None:
            previous_label = label
            previous_cell = cell

    groups = _group_consecutive(labels)
    groups, ignored_indices = _consolidate_hint_claims(groups, cells, hint_units)
    groups, icon_indices = _drop_icon_fragments(groups, cells, hint_units)
    ignored_indices = list(ignored_indices) + icon_indices
    units = [
        _build_unit(
            cells=cells,
            indices=indices,
            unit_id=order,
            hint_index=label,
            level=_hint_meta(label, hint_levels),
            block_id=_hint_meta(label, hint_block_ids),
            alignment=_hint_meta(label, hint_alignments),
            font_family=_hint_meta(label, hint_families),
            font_weight=_hint_meta(label, hint_weights),
            bullet=bool(_hint_meta(label, hint_bullets)),
        )
        for order, (label, indices) in enumerate(groups, start=1)
    ]
    leftover = sum(1 for label, _ in groups if label is None)
    ignored_cell_ids = [int(cells[i]["id"]) for i in ignored_indices if cells[i].get("id") is not None]

    return GroupingResult(
        units=units,
        ignored_cell_ids=ignored_cell_ids,
        model=model,
        metrics={
            "translation_unit_count": len(units),
            "ignored_cell_count": len(ignored_cell_ids),
            "hint_block_count": len(hint_units),
            "leftover_unit_count": leftover,
            "translatable_member_count": sum(
                1 for unit in units for member in unit.members if member.translate
            ),
        },
    )


def _consolidate_hint_claims(
    groups: list[tuple[int | None, list[int]]],
    cells: list[dict[str, Any]],
    hint_units: list[str],
) -> tuple[list[tuple[int | None, list[int]]], list[int]]:
    """A hint line should render once per FIELD, not once per cell-group that claims it.

    Several disjoint cell-groups can land on the same hint line: a wrapped element OCR'd as
    separate rows, the two halves of a ``|`` field row (a receipt's label-left / value-right),
    or strays whose tokens happen to overlap — an embedded image's "PENGUIN" hitting a
    "...vintage PENGUIN paperbacks" line, or a common word ("betaling") shared by two lines.
    ``_resolve_claim_clusters`` decides each claim: a distinct ``|`` field stays its own unit
    (renders at its own position), a wrapped continuation merges into the line it extends (so
    the line renders once), and a redundant stray/mismatch is dropped to ``ignored`` (stays
    original pixels). Leftover groups (matched no line) pass through."""
    by_label: dict[int, list[list[int]]] = {}
    out: list[tuple[int | None, list[int]]] = []
    for label, indices in groups:
        if label is None:
            out.append((label, indices))
        else:
            by_label.setdefault(label, []).append(indices)

    ignored: list[int] = []
    for label, claim_lists in by_label.items():
        if len(claim_lists) == 1:
            out.append((label, claim_lists[0]))
            continue
        kept_clusters, dropped = _resolve_claim_clusters(label, claim_lists, cells, hint_units)
        for members in kept_clusters:
            out.append((
                label,
                sorted(members, key=lambda i: (cells[i]["bbox"]["top"], cells[i]["bbox"]["left"])),
            ))
        ignored.extend(dropped)

    out.sort(key=lambda g: min((cells[i]["bbox"]["top"] for i in g[1]), default=0))
    return out, ignored


def _resolve_claim_clusters(
    label: int,
    claim_lists: list[list[int]],
    cells: list[dict[str, Any]],
    hint_units: list[str],
) -> tuple[list[list[int]], list[int]]:
    """Decide, per group claiming one hint line, whether it is a distinct ``|`` field (keep as
    its own unit), a wrapped continuation of an already-kept group (merge into it so the line
    renders once), or redundant — a stray/mismatch that covers no new field and adds no new
    token (drop to ``ignored``, leaving original pixels). Claims are processed by descending
    coverage of the line so the fullest claim anchors the rest; only a same-field continuation
    (new tokens, no new field) is allowed to merge, and only into a spatially adjacent group —
    so a redundant stray under a kept group (a receipt's "BETALING" below "MAESTRO") is dropped
    instead of inflating that unit's footprint. Returns (kept_member_lists, dropped_indices)."""
    fields = _hint_fields(hint_units[label])
    line_tokens: set[str] = set().union(*fields) if fields else set()

    def tokens_of(members: list[int]) -> set[str]:
        return {t for i in members for t in _tokens(str(cells[i].get("text") or ""))} & line_tokens

    def fields_of(tokens: set[str]) -> set[int]:
        return {fi for fi, fset in enumerate(fields) if fset and (tokens & fset)}

    order = sorted(range(len(claim_lists)), key=lambda k: len(tokens_of(claim_lists[k])), reverse=True)
    kept: list[list[int]] = []
    dropped: list[int] = []
    covered_tokens: set[str] = set()
    covered_fields: set[int] = set()
    for k in order:
        members = claim_lists[k]
        tokens = tokens_of(members)
        if not kept or (fields_of(tokens) - covered_fields):
            kept.append(list(members))                       # primary, or a distinct ``|`` field
        else:
            target = _merge_target(members, kept, cells) if (tokens - covered_tokens) else None
            if target is None:
                dropped.extend(members)                      # redundant stray / mismatch
                continue
            target.extend(members)                           # wrapped continuation of its line
        covered_tokens |= tokens
        covered_fields |= fields_of(tokens)
    return kept, dropped


def _drop_icon_fragments(
    groups: list[tuple[int | None, list[int]]],
    cells: list[dict[str, Any]],
    hint_units: list[str],
) -> tuple[list[tuple[int | None, list[int]]], list[int]]:
    """Drop a member that duplicates a word already in its unit AND is set apart from the rest of
    the line — an icon/badge's tiny embedded label OCR read as text (a "postnl" logo next to
    "Bezorging door PostNL") or a logo elsewhere in the image bound to a line that names it (the
    "NIKE" on a shoe pulled into a "Nike Sweet Classic …" body line). Such a fragment otherwise
    drags the unit's box onto the icon. Dropped cells go to ``ignored`` (their original pixels
    stay). Never empties a group. The spatial-outlier test is suppressed on ``|`` field rows, whose
    fields (a receipt's far-left quantity vs far-right price) are *meant* to sit apart."""
    out: list[tuple[int | None, list[int]]] = []
    dropped: list[int] = []
    for label, indices in groups:
        if label is None or len(indices) < 2:
            out.append((label, indices))
            continue
        single_field = "|" not in str(hint_units[label] or "")
        kept = [i for i in indices if not _is_icon_fragment(i, indices, cells, allow_detached=single_field)]
        out.append((label, kept if kept else indices))
        if kept:
            dropped.extend(i for i in indices if i not in kept)
    return out, dropped


def _hint_fields(hint_line: str) -> list[set[str]]:
    """Token sets of a hint line's ``|`` fields (one entry per field; the whole line as a
    single field when it carries no ``|``). Used to tell a complementary label/value split
    from a redundant stray claim."""
    parts = str(hint_line or "").split("|")
    return [set(_tokens(part)) for part in parts]


def _merge_target(members: list[int], kept: list[list[int]], cells: list[dict[str, Any]]) -> list[int] | None:
    """The kept group this claim should merge into — the first one it is spatially adjacent to
    (stacked or split across a line). ``None`` when it stands off on its own (a stray)."""
    box = _claim_box(members, cells)
    for cluster in kept:
        if _near(box, _claim_box(cluster, cells)):
            return cluster
    return None


def _claim_box(members: list[int], cells: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    ls = [float(cells[i]["bbox"]["left"]) for i in members]
    ts = [float(cells[i]["bbox"]["top"]) for i in members]
    rs = [float(cells[i]["bbox"]["left"]) + float(cells[i]["bbox"].get("width", 0)) for i in members]
    bs = [float(cells[i]["bbox"]["top"]) + float(cells[i]["bbox"].get("height", 0)) for i in members]
    return min(ls), min(ts), max(rs), max(bs)


@dataclass(frozen=True)
class _Match:
    candidates: list[int]  # hint indices sharing the best token-overlap score
    score: float           # best matched-token mass / cell token count
    full: tuple[int, ...] = ()  # of those, the lines the cell fully accounts for (every token)
    full_alpha: tuple[int, ...] = ()  # full matches for cells carrying alphabetic text


def _build_hint_index(hint_token_sets: list[set[str]]) -> dict[str, set[int]]:
    """Inverted index: each hint token -> the line indices that contain it. Lets a cell be scored
    against only the lines its tokens land in, instead of every line (O(cells x hints) -> ~O(cells)
    for ordinary text where a token is shared by few lines)."""
    index: dict[str, set[int]] = {}
    for line, hint_set in enumerate(hint_token_sets):
        for token in hint_set:
            index.setdefault(token, set()).add(line)
    return index


def _candidate_hints(cell: dict[str, Any], token_to_hints: dict[str, set[int]]) -> set[int] | None:
    """The hint lines a cell shares an EXACT token with — the only lines worth scoring for a
    cleanly-read cell. ``None`` when it shares none: its tokens are OCR garble (or non-translatable)
    that can still bind by the fuzzy substring/ratio match in _token_score, which the exact index
    cannot see, so the caller falls back to scanning every line. Garble is the minority, so the
    fallback is rare and the match stays near-linear. Verified to reproduce the full-scan result
    across the testset (incl. garbled receipts) and the fixtures; see _match_scores."""
    candidates: set[int] = set()
    for token in _tokens(str(cell.get("text") or "")):
        candidates.update(token_to_hints.get(token, ()))
    return candidates or None


def _match_scores(
    cell: dict[str, Any],
    hint_token_sets: list[set[str]],
    candidate_indices: set[int] | None = None,
) -> _Match:
    cell_tokens = _tokens(str(cell.get("text") or ""))
    if not cell_tokens:
        return _Match(candidates=[], score=0.0)
    # ``candidate_indices`` (from _build_hint_index) are the only lines that can score > 0; the rest
    # would add a 0 and be dropped anyway. Iterate them in index order so the result — ties and the
    # ``full`` set — is identical to scanning every line. None = scan all (the unindexed path).
    indices = range(len(hint_token_sets)) if candidate_indices is None else sorted(candidate_indices)
    matched_by_index: list[tuple[int, float]] = []
    best_matched = 0.0
    for index in indices:
        hint_set = hint_token_sets[index]
        if not hint_set:
            continue
        matched = sum(_token_score(token, hint_set) for token in cell_tokens)
        if matched:
            matched_by_index.append((index, matched))
            if matched > best_matched:
                best_matched = matched
    if not best_matched:
        return _Match(candidates=[], score=0.0)
    candidates = [index for index, matched in matched_by_index if matched == best_matched]
    full = tuple(index for index in candidates if best_matched + 1e-9 >= len(hint_token_sets[index]))
    has_alpha = any(any(ch.isalpha() for ch in token) for token in cell_tokens)
    return _Match(
        candidates=candidates,
        score=best_matched / len(cell_tokens),
        full=full,
        full_alpha=full if has_alpha else (),
    )


def _pick_hint(
    match: _Match,
    *,
    preferred_index: float = 0.0,
    sticky: int | None = None,
    position_reliable: bool = False,
) -> int | None:
    if not match.candidates or match.score < _MATCH_THRESHOLD:
        return None
    candidates = list(match.candidates)
    if position_reliable:
        guarded = [
            index for index in candidates
            if abs(index - preferred_index) <= _POSITION_GUARD
        ]
        if len(match.full_alpha) == 1 and guarded:
            guarded.append(match.full_alpha[0])
        candidates = list(dict.fromkeys(guarded))
        if not candidates:
            return None
    # Several hint lines can match a short cell equally (two dishes ending "en frites").
    # A continuation cell stays with its element (axis-aligned tops are tilt-distorted,
    # so position alone is a coin flip near an element boundary); otherwise bind the
    # cell to the hint nearest its place on the page.
    if sticky is not None and sticky in candidates:
        return sticky
    # Among equally-scored candidates, prefer a hint line the cell FULLY accounts for (it carries
    # every token of that line): a short brand/eyebrow that is a prefix of the title, a label above
    # its value. Without this, such a cell is a fragment of the longer neighbour too and gets pulled
    # into it on position alone. Position still breaks any remaining tie.
    full = [index for index in match.full if index in candidates]
    pool = full or candidates
    best = min(pool, key=lambda index: abs(index - preferred_index))
    return best


def _anchored_positions(
    cells: list[dict[str, Any]], matches: list[_Match], n_hints: int
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
    varies (a dense menu above a sparse footer)."""
    tops = [float((cell.get("bbox") or {}).get("top", 0.0)) for cell in cells]
    if not tops or n_hints <= 1:
        return [0.0] * len(cells), False
    seeds = [
        (top, match.candidates[0])
        for top, match in zip(tops, matches)
        if len(match.candidates) == 1 and match.score >= _MATCH_THRESHOLD
    ]
    anchors = _chain(seeds)
    if len(anchors) < 2:
        return _linear_positions(tops, n_hints), False
    return [_interpolate(top, anchors, n_hints) for top in tops], True


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
    """Runs of one label form a unit. A leftover (None) is its own unit but does NOT
    break the surrounding run: an interleaved noise cell must not split an element in
    two — the structured translation would land on both halves and render twice."""
    groups: list[tuple[int | None, list[int]]] = []
    current: tuple[int | None, list[int]] | None = None
    for cell_index, label in enumerate(labels):
        if label is None:
            groups.append((None, [cell_index]))
            continue
        if current is not None and current[0] == label:
            current[1].append(cell_index)
        else:
            current = (label, [cell_index])
            groups.append(current)
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
    alignment: str | None = None,
    font_family: str | None = None,
    font_weight: int | None = None,
    bullet: bool = False,
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
        alignment=alignment,
        font_family=font_family,
        font_weight=font_weight,
        bullet=bullet,
    )
