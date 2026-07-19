"""Matching cells to hint lines by token evidence: the inverted hint index, per-cell match
scores, confident-label selection, and the layout-column filter that keeps a weak match from
claiming a hint line in another column.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.grouping.align.tuning import _COLUMN_FILTER_MAX_SCORE
from app.grouping.align.tuning import _HINT_COLUMN_MAJORITY
from app.grouping.align.tuning import _MATCH_THRESHOLD
from app.grouping.align.tuning import _POSITION_GUARD

from app.grouping.tokens import _token_score
from app.grouping.tokens import _tokens


@dataclass
class _Match:
    candidates: list[int]  # hint indices sharing the best token-overlap score
    score: float           # best matched-token mass / cell token count
    full: tuple[int, ...] = ()  # of those, the lines the cell fully accounts for (every token)
    full_alpha: tuple[int, ...] = ()  # full matches for cells carrying alphabetic text
    # Of the candidates, the lines whose token SEQUENCE contains the cell's tokens as a contiguous
    # run (>= 2 tokens; a single token is trivially contiguous everywhere and discriminates
    # nothing). A wrapped fragment is a contiguous run of its own line; the same tokens scattered
    # over another line are coincidence — used as the tie-break of last resort in _pick_hint.
    phrase: tuple[int, ...] = ()


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
    cannot see, so the caller falls back to scanning every line. ``_match_scores`` additionally
    full-scans when the indexed result lands below the bind threshold (a partially-garbled cell),
    so pruning can never turn a bindable cell into a leftover. Garble is the minority, so both
    fallbacks are rare and the match stays near-linear."""
    candidates: set[int] = set()
    for token in _tokens(str(cell.get("text") or "")):
        candidates.update(token_to_hints.get(token, ()))
    return candidates or None


def _match_scores(
    cell: dict[str, Any],
    hint_token_sets: list[set[str]],
    candidate_indices: set[int] | None = None,
    fuzzy_sets: list[set[str]] | None = None,
    token_seqs: list[list[str]] | None = None,
) -> _Match:
    cell_tokens = _tokens(str(cell.get("text") or ""))
    if not cell_tokens:
        return _Match(candidates=[], score=0.0)
    fuzzy = hint_token_sets if fuzzy_sets is None else fuzzy_sets
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
        matched = sum(_token_score(token, hint_set, fuzzy[index]) for token in cell_tokens)
        if matched:
            matched_by_index.append((index, matched))
            if matched > best_matched:
                best_matched = matched
    if not best_matched:
        return _Match(candidates=[], score=0.0)
    # Below the bind threshold the indexed scan may not be the last word: it only saw lines the
    # cell shares an EXACT token with, but a cell with one clean token and the rest OCR garble can
    # belong to a line it matches only fuzzily ("Kaarthuder betallng pas" carries "pas" of one line
    # exactly, yet is a garbled read of "Kaarthouder betaling"). Rescan every line before letting
    # the cell fall through as a leftover. Cells the index already binds keep the indexed result
    # untouched, and an all-garble cell arrives here with ``candidate_indices`` None already — so
    # this fires only for the mixed case, which stays rare enough to keep the match near-linear.
    if candidate_indices is not None and best_matched / len(cell_tokens) < _MATCH_THRESHOLD:
        return _match_scores(cell, hint_token_sets, None, fuzzy_sets, token_seqs)
    candidates = [index for index, matched in matched_by_index if matched == best_matched]
    # ``full`` means "the cell carries every token of the line", so it is recounted on DISTINCT
    # cell tokens: the raw mass above counts duplicates separately, and per-character CJK tokens
    # repeat constantly (小心小心 has mass 4), letting a repeated fragment reach a line's set size
    # while covering only half of it. For a duplicate-free cell the distinct sum equals the mass,
    # so this is exactly the old test.
    distinct_tokens = list(dict.fromkeys(cell_tokens))
    full = tuple(
        index
        for index in candidates
        if sum(_token_score(token, hint_token_sets[index], fuzzy[index]) for token in distinct_tokens)
        + 1e-9
        >= len(hint_token_sets[index])
    )
    has_alpha = any(any(ch.isalpha() for ch in token) for token in cell_tokens)
    # Contiguity is judged on EXACT tokens only (no fuzzy): a garbled cell falls back to the
    # existing paths rather than phrase-matching a line it merely resembles.
    phrase = tuple(
        index for index in candidates
        if token_seqs is not None and len(cell_tokens) >= 2
        and _contains_run(token_seqs[index], cell_tokens)
    )
    return _Match(
        candidates=candidates,
        score=best_matched / len(cell_tokens),
        full=full,
        full_alpha=full if has_alpha else (),
        phrase=phrase,
    )


def _contains_run(sequence: list[str], run: list[str]) -> bool:
    """Whether ``run`` occurs as a contiguous slice of ``sequence``."""
    if not run or len(run) > len(sequence):
        return False
    first = run[0]
    return any(
        sequence[start] == first and sequence[start:start + len(run)] == run
        for start in range(len(sequence) - len(run) + 1)
    )


def _pick_hint(
    match: _Match,
    *,
    preferred_index: float = 0.0,
    sticky: int | None = None,
    position_reliable: bool = False,
    line_anchor: int | None = None,
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
    # Reading-flow contiguity wins first: a cell flanked on its printed line by confident cells of
    # one hint line belongs to that line, however its axis-aligned top (tilt-distorted) interpolates.
    # This is what keeps a shared word ("dieren", sitting between "Voer en aai de" and "nooit." which
    # both read line 7) from flipping to a neighbouring line on a 0.04-index position tie.
    if line_anchor is not None and line_anchor in candidates:
        return line_anchor
    # A cell that FULLY accounts for a candidate line (carries every one of its tokens) is that
    # line's own: a list item identical to its siblings but for one word ("...first item" vs
    # "...second item"), a short brand/eyebrow that is a prefix of the title, a label above its
    # value. Bind it to the nearest such line by position BEFORE sticking to the previous line —
    # sticky is for a wrapped continuation, which is a fragment and fully matches nothing, so this
    # never steals a true continuation. Without it two identical sibling items collapse onto one
    # hint (the second item is lost and the first reflows over both lines).
    full = [index for index in match.full if index in candidates]
    if full:
        return min(full, key=lambda index: abs(index - preferred_index))
    # Several hint lines can match a short cell equally (two dishes ending "en frites"). A
    # continuation fragment stays with its element (axis-aligned tops are tilt-distorted, so
    # position alone is a coin flip near an element boundary); otherwise bind to the nearest hint.
    if sticky is not None and sticky in candidates:
        return sticky
    # Phrase contiguity beats the position coin-flip: on an interleaved page (a letterhead's
    # split logo left, a contact block right) the position interpolation puts a wrapped fragment
    # nearer the OTHER column's line, while the fragment reads verbatim in its own line. Only
    # narrows — several phrase lines (two dishes ending "en frites") still go to position.
    phrase = [index for index in match.phrase if index in candidates]
    if phrase:
        candidates = phrase
    return min(candidates, key=lambda index: abs(index - preferred_index))


def _confident_label(match: _Match) -> int | None:
    """The hint line a cell unambiguously belongs to: a single best-scoring candidate above the
    match threshold. ``None`` when the cell is ambiguous (its token sits in several lines) or weak —
    such a cell does not anchor a neighbour, it gets resolved BY its confident neighbours."""
    if len(match.candidates) == 1 and match.score >= _MATCH_THRESHOLD:
        return match.candidates[0]
    return None


def _hint_block_columns(
    matches: list["_Match"], cell_columns: list[int | None]
) -> dict[int, int]:
    """Per hint block, the column its matching cells agree on (>= _HINT_COLUMN_MAJORITY of the
    score mass), else absent. Every cell whose candidates include the block votes its column,
    weighted by match score — NOT only single-candidate cells: two similar blocks (the §2 bullet
    and its neighbour) make each other's cells two-candidate, so a confident-only tally would
    leave both blocks column-less and unconstrained. The real block cells cluster in one column
    and carry the score mass; a lone cross-column orphan is outvoted."""
    from collections import Counter

    tally: dict[int, Counter] = {}
    for index, match in enumerate(matches):
        column = cell_columns[index]
        if column is None or match.score < _MATCH_THRESHOLD:
            continue
        for candidate in match.candidates:
            tally.setdefault(candidate, Counter())[column] += match.score
    columns: dict[int, int] = {}
    for label, counter in tally.items():
        column, mass = counter.most_common(1)[0]
        if mass >= _HINT_COLUMN_MAJORITY * sum(counter.values()):
            columns[label] = column
    return columns


def _column_filtered_match(match: "_Match", column: int | None, hint_columns: dict[int, int]) -> "_Match":
    """``match`` with WEAK candidates in a different column than ``column`` dropped (see
    _COLUMN_FILTER_MAX_SCORE). A strong match — the cell carries the block's text, a header or
    byline the layout repeats in the other column — is kept across columns; only a partial match
    (a cell sharing one stray token with a far-column block, landing there through the flat-index
    position guard) is the failure mode. Blocks of unknown column, and cells without a column,
    impose no constraint."""
    if column is None or match.score >= _COLUMN_FILTER_MAX_SCORE:
        return match

    def keep(index: int) -> bool:
        block_column = hint_columns.get(index)
        return block_column is None or block_column == column or index in match.full

    candidates = [index for index in match.candidates if keep(index)]
    if len(candidates) == len(match.candidates):
        return match
    return _Match(
        candidates=candidates,
        score=match.score if candidates else 0.0,
        full=tuple(index for index in match.full if keep(index)),
        full_alpha=tuple(index for index in match.full_alpha if keep(index)),
        phrase=tuple(index for index in match.phrase if keep(index)),
    )
