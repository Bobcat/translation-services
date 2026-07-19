"""The align orchestration: consolidate hint claims, match cells to hint lines, estimate
positions, and assemble the translation units with full cell coverage.
"""
from __future__ import annotations

from typing import Any

from app.grouping.align.tuning import _MATCH_THRESHOLD

from app import layout as layout_evidence
from app.grouping.align.claims import _absorb_symbol_leftovers
from app.grouping.align.claims import _consolidate_hint_claims
from app.grouping.align.claims import _drop_icon_fragments
from app.grouping.align.claims import _is_continuation
from app.grouping.align.claims import _merge_leftover_tails
from app.grouping.align.matching import _Match
from app.grouping.align.matching import _build_hint_index
from app.grouping.align.matching import _candidate_hints
from app.grouping.align.matching import _column_filtered_match
from app.grouping.align.matching import _confident_label
from app.grouping.align.matching import _hint_block_columns
from app.grouping.align.matching import _match_scores
from app.grouping.align.matching import _pick_hint
from app.grouping.align.positions import _anchored_positions
from app.grouping.align.positions import _line_anchor
from app.grouping.preserve import _is_nontranslatable
from app.grouping.preserve import _is_symbolic_label
from app.grouping.tokens import _fuzzy_tokens
from app.grouping.tokens import _token_score
from app.grouping.tokens import _tokens
from app.grouping.units import GroupingResult
from app.grouping.units import TranslationUnit
from app.grouping.units import UnitMember
from app.grouping.units import union_bbox


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
    hint_sizes: list[int | None] | None = None,
    hint_bullets: list[bool] | None = None,
    hint_bullet_markers: list[str | None] | None = None,
    layout_regions: list[dict[str, Any]] | None = None,
    preserve_image_regions: bool = True,
) -> GroupingResult:
    hint_token_sets = [set(_tokens(text)) for text in hint_units]
    # The ordered token sequences feed the phrase tie-break in _pick_hint: a cell whose tokens
    # appear as a CONTIGUOUS run of one candidate line is that line's own fragment (a wrapped
    # logo/title line), where the same tokens merely scattered over another line are coincidence.
    hint_token_seqs = [_tokens(text) for text in hint_units]
    # The fuzzy fallback scans the UNFOLDED hint tokens: ligature folding is for the exact
    # match only (see tokens._FOLD), so a ligature misread cannot ratio-match onto a line.
    hint_fuzzy_sets = [set(_fuzzy_tokens(text)) for text in hint_units]
    token_to_hints = _build_hint_index(hint_token_sets)
    # Layout evidence (app.layout), used only when its own document_gate opens: cells
    # inside image/chart regions keep their original pixels (empty match -> no label -> routed to
    # ignored below), and multi-column pages get per-column position chains. Gate closed or no
    # regions -> both stay inert and this is bit-for-bit the pre-layout pipeline.
    preserved: set[int] = set()
    cell_columns: list[int | None] | None = None
    layout_gate_open = bool(layout_regions) and layout_evidence.document_gate(layout_regions, cells)
    if layout_gate_open:
        # preserve_image_regions=False: translate and render text inside image/chart regions
        # too (a figure that is really a table screenshot, whose cells otherwise stay original
        # pixels). Column clustering is unaffected — only the preserve routing is skipped.
        if preserve_image_regions:
            preserved = layout_evidence.preserved_cell_indices(layout_regions, cells)
        cell_columns = layout_evidence.cell_columns(layout_regions, cells)
    matches = [
        _Match(candidates=[], score=0.0)
        if index in preserved
        else _match_scores(
            cell,
            hint_token_sets,
            _candidate_hints(cell, token_to_hints),
            fuzzy_sets=hint_fuzzy_sets,
            token_seqs=hint_token_seqs,
        )
        for index, cell in enumerate(cells)
    ]
    positions, positions_anchored = _anchored_positions(
        cells, matches, len(hint_units), cell_columns=cell_columns
    )
    # A cell whose best token-match is a single hint line is CONFIDENT; the rest are ambiguous (a
    # word shared by several lines). An ambiguous cell takes the line of its confident line-neighbours
    # — reading-flow contiguity — instead of a hair's-breadth position tie-break that flips it to the
    # wrong neighbouring line (see _line_anchor).
    confident = [_confident_label(match) for match in matches]
    # Column-consistency filter (multi-column pages): a hint block belongs to ONE physical column
    # (its confident cells cluster there); a cell may only take a candidate whose block is in the
    # cell's own column. Without it a cell weakly sharing a token ("in") with a block in the OTHER
    # column can land within the flat-index position guard (adjacent reading-order hints sit in
    # different columns) and get confidently mislabeled there — the VLM dropping a paragraph then
    # dumps its orphaned cell across the page. Blocks whose column is unclear impose no constraint,
    # so single-column pages (cell_columns None) and mixed blocks are untouched.
    hint_columns = _hint_block_columns(matches, cell_columns) if cell_columns is not None else {}
    if hint_columns:
        matches = [
            _column_filtered_match(match, cell_columns[index], hint_columns)
            for index, match in enumerate(matches)
        ]
        confident = [_confident_label(match) for match in matches]
    line_anchors = [
        _line_anchor(index, cells, confident) if len(match.candidates) > 1 else None
        for index, match in enumerate(matches)
    ]
    labels: list[int | None] = []
    previous_label: int | None = None
    previous_cell: dict[str, Any] | None = None
    for index, (cell, match, position) in enumerate(zip(cells, matches, positions)):
        sticky = previous_label if _is_continuation(previous_cell, cell) else None
        label = _pick_hint(
            match,
            preferred_index=position,
            sticky=sticky,
            position_reliable=positions_anchored,
            line_anchor=line_anchors[index],
        )
        labels.append(label)
        if label is not None:
            previous_label = label
            previous_cell = cell

    # Leftover rescue for multi-column pages. The anchored positions interpolate over the FLAT
    # reading order, so on a two-column page the other column's cells drag a paragraph's trailing
    # line toward their own hint indices — the position guard then rejects every text candidate
    # (score-1.0 included) before sticky can bind it, the line orphans, and its words render TWICE
    # (the hint-fed translation of its paragraph already carries them). Rescue is purely additive
    # and local: only a cell that stayed unlabeled, only onto the label of the cell geometrically
    # directly above it in its own column (same margin — _is_continuation), and only when
    #   (a) the cell's own tokens clear the bind threshold ON that neighbour's line — the token
    #       gate that keeps a stray like "BETALING" under "MAESTRO" out. Scored against the line
    #       directly (not via ``match.candidates``): an OCR-garbled token ("Dental 1") pulls the
    #       candidate list to whatever far line happens to carry the garble, while the neighbour's
    #       line is the one the continuation geometry vouches for; and
    #   (b) the cell CONTRIBUTES a token the line's current claimants do not cover yet. A wrapped
    #       continuation is the line's uncovered tail; a REPEATED row ("Project" printed twice,
    #       stacked) covers nothing new — gluing it on would get it dropped as a redundant claim
    #       and leave the second print untranslated, so it stays its own leftover unit (which
    #       still translates and renders at its own spot).
    # The claim consolidation below then merges the cell into that claim, where the redundancy
    # checks still apply.
    for index, (cell, match) in enumerate(zip(cells, matches)):
        if labels[index] is not None or match.score < _MATCH_THRESHOLD:
            continue
        cell_tokens = _tokens(str(cell.get("text") or ""))
        for j in range(index - 1, -1, -1):
            if labels[j] is None:
                continue
            if _is_continuation(cells[j], cell):
                line = labels[j]
                on_line = sum(
                    _token_score(token, hint_token_sets[line], hint_fuzzy_sets[line])
                    for token in cell_tokens
                )
                covered = {
                    token
                    for k, label_k in enumerate(labels)
                    if label_k == line
                    for token in _tokens(str(cells[k].get("text") or ""))
                }
                contributes = any(token not in covered for token in cell_tokens)
                if cell_tokens and contributes and on_line / len(cell_tokens) >= _MATCH_THRESHOLD:
                    labels[index] = line
                break  # the geometric upstairs neighbour decides; farther cells are not "above"

    groups = _group_consecutive(labels)
    groups, ignored_indices = _consolidate_hint_claims(groups, cells, hint_units)
    groups = _merge_leftover_tails(groups, cells, hint_units, hint_token_sets, hint_fuzzy_sets)
    groups, icon_indices = _drop_icon_fragments(groups, cells, hint_units)
    ignored_indices = list(ignored_indices) + icon_indices
    groups = _absorb_symbol_leftovers(groups, cells, hint_units, preserved)
    if preserved:
        # Preserve routing: a preserved cell has an empty match, so it always arrives here as
        # its own leftover group — move it to ignored (original pixels: not erased, not
        # translated, not rendered) instead of letting it become a per-cell leftover unit.
        kept_groups: list[tuple[int | None, list[int]]] = []
        for label, indices in groups:
            if label is None and all(index in preserved for index in indices):
                ignored_indices.extend(indices)
            else:
                kept_groups.append((label, indices))
        groups = kept_groups
    units = [
        _build_unit(
            cells=cells,
            indices=indices,
            unit_id=order,
            hint_index=label,
            hint_line=hint_units[label] if label is not None else None,
            level=_hint_meta(label, hint_levels),
            block_id=_hint_meta(label, hint_block_ids),
            alignment=_hint_meta(label, hint_alignments),
            font_family=_hint_meta(label, hint_families),
            font_weight=_hint_meta(label, hint_weights),
            font_size=_hint_meta(label, hint_sizes),
            bullet=bool(_hint_meta(label, hint_bullets)),
            bullet_marker=_hint_meta(label, hint_bullet_markers),
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
            "layout_gate_open": layout_gate_open,
            "layout_preserved_cell_count": len(preserved),
            "layout_column_count": len({c for c in cell_columns if c is not None})
            if cell_columns
            else 0,
        },
    )


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
    hint_line: str | None = None,
    level: str | None = None,
    block_id: int | None = None,
    alignment: str | None = None,
    font_family: str | None = None,
    font_weight: int | None = None,
    font_size: int | None = None,
    bullet: bool = False,
    bullet_marker: str | None = None,
) -> TranslationUnit:
    texts = [str(cells[cell_index].get("text") or "") for cell_index in indices]
    translate_flags = _member_translate_flags(texts, hint_line)
    # A unit consisting ENTIRELY of symbolic label tokens, several of them on
    # one geometric line ('A " A- A-': a row of rating codes), is preserved
    # whole — the translator can only hallucinate on it. Deliberately >= 2
    # members: single symbolic cells (an icon read as 'Q', a receipt code) keep
    # today's behaviour — the translation layer's noise skip already covers the
    # short ones, and the batch-length guard backstops the rest.
    if len(texts) >= 2 and all(_is_symbolic_label(text) for text in texts):
        translate_flags = [False] * len(translate_flags)
    members: list[UnitMember] = []
    for order, (cell_index, text, translate) in enumerate(
        zip(indices, texts, translate_flags), start=1
    ):
        cell = cells[cell_index]
        polygon = cell.get("polygon")
        size_px = cell.get("size_px")
        islands = cell.get("islands")
        members.append(
            UnitMember(
                cell_id=int(cell["id"]),
                text=text,
                translate=translate,
                bbox=dict(cell.get("bbox") or {}),
                order=order,
                polygon=[dict(point) for point in polygon] if polygon else None,
                size_px=float(size_px) if size_px is not None else None,
                islands=[dict(island) for island in islands] if islands else None,
            )
        )
    # Ground-truth list marker from the cell layer (a stripped inline "•" the
    # extractor recorded): stronger than the hint's bullet label, which wobbles
    # run to run. Only the first member carries the item's marker.
    if not bullet and indices:
        cell_marker = str(cells[indices[0]].get("marker") or "").strip()
        if cell_marker:
            bullet = True
            bullet_marker = bullet_marker or cell_marker

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
        font_size=font_size,
        bullet=bullet,
        bullet_marker=bullet_marker,
    )


def _member_translate_flags(texts: list[str], hint_line: str | None) -> list[bool]:
    """Base rule: a member translates unless its own text is nontranslatable (a bare number,
    URL, price). Field inheritance on top: OCR often splits a phrase's number into its own
    cell ("8 jun" -> cells "8" + "jun") while the hint keeps one field whose translation
    carries the number ("8 jun" -> "6月8日") — preserving the bare digit would render the
    date twice next to a half-erased original. So a nontranslatable member joins the
    translation when the hint FIELD covering it is a translatable phrase. The covering field
    is the unique field containing the member's tokens; when several fields share the token
    ("9 jun" and "9°/16°" both carry "9"), the unique field of an ADJACENT member decides,
    provided it covers this member too. A field that is itself nontranslatable ("11°/21°",
    a "1,69 B" price) keeps its numerics preserved, and no resolution means no flip."""
    flags = [not _is_nontranslatable(text) for text in texts]
    if not hint_line:
        return flags
    # A token-less symbol member ('&') the hint line carries verbatim: the line's translation
    # renders the symbol, so preserving the member's own pixels would print it twice (the
    # absorbed-leftover case — see _absorb_symbol_leftovers). '|' is the field separator in hint
    # text, never proof of a printed glyph.
    for index, text in enumerate(texts):
        stripped = str(text).strip()
        if stripped and stripped != "|" and not _tokens(stripped) and stripped in str(hint_line):
            flags[index] = True
    fields = [
        (set(_tokens(part)), not _is_nontranslatable(part))
        for part in str(hint_line).split("|")
    ]
    token_sets = [set(_tokens(text)) for text in texts]

    def covering(tokens: set[str]) -> list[int]:
        return [i for i, (field_tokens, _) in enumerate(fields) if tokens and tokens <= field_tokens]

    for index, (flag, tokens) in enumerate(zip(flags, token_sets)):
        if flag or not tokens:
            continue
        candidates = covering(tokens)
        if len(candidates) > 1:
            resolved = None
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(texts):
                    neighbour_fields = covering(token_sets[neighbour])
                    if len(neighbour_fields) == 1 and neighbour_fields[0] in candidates:
                        resolved = neighbour_fields[0]
                        break
            candidates = [resolved] if resolved is not None else []
        if len(candidates) == 1 and fields[candidates[0]][1]:
            flags[index] = True
    return flags
