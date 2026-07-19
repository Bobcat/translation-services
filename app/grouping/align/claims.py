"""Consolidating the hint's claims on cells: per hint line, keep the best-matching claim,
merge wrapped continuation lines, absorb redundant symbol/icon fragments, and drop strays and
second prints. Redundancy is judged BEFORE spatial merging, so an adjacent stray cannot
inflate a claim and shove a field off its line.
"""
from __future__ import annotations

from typing import Any

from app.grouping.align.tuning import _LINE_GAP_RATIO
from app.grouping.align.tuning import _LINE_VOVERLAP_RATIO
from app.grouping.align.tuning import _MATCH_THRESHOLD

from app.grouping.preserve import _is_icon_fragment
from app.grouping.tokens import _token_pair_matches
from app.grouping.tokens import _token_score
from app.grouping.tokens import _tokens
from app.grouping.units import _near


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
        kept_clusters, demoted, dropped = _resolve_claim_clusters(label, claim_lists, cells, hint_units)
        for members in kept_clusters:
            out.append((
                label,
                sorted(members, key=lambda i: (cells[i]["bbox"]["top"], cells[i]["bbox"]["left"])),
            ))
        for members in demoted:
            # A genuine second print of duplicated text: its own LEFTOVER unit (translated and
            # rendered at its own spot) instead of ignored original pixels.
            out.append((
                None,
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
        # The LINE tokens this claim covers. Exact first; a cell token that only fuzzy-matches
        # (the claim bound its line through OCR garble) covers the line token it garbles —
        # dedup must judge a claim by the same rules that bound it, or an all-garble wrapped
        # continuation ("kortlng") looks token-free, can never register a field or new content,
        # and is dropped as a stray: exactly the leftover-doubling this dedup exists to prevent.
        # For clean cells the fuzzy branch never fires, so this is the old exact intersection.
        covered: set[str] = set()
        for i in members:
            for t in _tokens(str(cells[i].get("text") or "")):
                if t in line_tokens:
                    covered.add(t)
                else:
                    covered.update(h for h in line_tokens if _token_pair_matches(t, h))
        return covered

    def fields_of(tokens: set[str]) -> set[int]:
        return {fi for fi, fset in enumerate(fields) if fset and (tokens & fset)}

    order = sorted(range(len(claim_lists)), key=lambda k: len(tokens_of(claim_lists[k])), reverse=True)
    kept: list[list[int]] = []
    demoted: list[list[int]] = []
    dropped: list[int] = []
    covered_tokens: set[str] = set()
    covered_fields: set[int] = set()
    to_merge: list[int] = []
    for k in order:
        members = claim_lists[k]
        tokens = tokens_of(members)
        if not kept or (fields_of(tokens) - covered_fields):
            kept.append(list(members))                       # primary, or a distinct ``|`` field
            covered_tokens |= tokens
            covered_fields |= fields_of(tokens)
        elif tokens - covered_tokens:
            to_merge.append(k)                               # adds new content -> merge into its line
        elif _is_second_print(members, kept, cells, line_tokens):
            # Redundant but EXACT-clean and spatially apart from every kept claim: a genuine
            # second print whose text duplicates line tokens — a label repeated per column
            # ("Part D" twice), a wrapped tail a short header line full-match-stole ("Original
            # Medicare."), a word re-occurring inside its own line ("… Insurance)."). Ignoring
            # it leaves an untranslated original in the render; instead it becomes a LEFTOVER
            # (own unit, translated and rendered at its own spot — the repeated-prints
            # doctrine). A garbled double-read stays dropped: it binds only via fuzzy, and the
            # kept claim erases the very print it garbles.
            demoted.append(list(members))
        else:
            dropped.extend(members)                          # redundant stray / mismatch

    # Merge each new-token claim into an adjacent kept group, iterating to a fixpoint: a merge grows
    # the kept box, which can then reach a claim that was out of range before. So a tilted line chains
    # in (the right-end words attach via the middle word) regardless of the token-count order — without
    # this, "0800-1995" is compared against the far-left "Of bel de" before "stoplijn" bridges them, and
    # a real continuation is wrongly dropped. A claim adjacent to nothing even after the boxes have grown
    # is a genuinely detached stray (an embedded image's text) and still drops.
    changed = True
    while changed and to_merge:
        changed = False
        still: list[int] = []
        for k in to_merge:
            members = claim_lists[k]
            if not (tokens_of(members) - covered_tokens):
                # An earlier merge already covered its tokens — same verdict as the redundant
                # branch above: a clean, spatially detached claim is a genuine second print
                # of duplicated words (a display title repeating body prose) and demotes to a
                # leftover; a garbled or overlapping stray still drops.
                if _is_second_print(members, kept, cells, line_tokens):
                    demoted.append(list(members))
                else:
                    dropped.extend(members)
                changed = True
                continue
            target = _merge_target(members, kept, cells)
            if target is None:
                still.append(k)
                continue
            target.extend(members)                           # wrapped/tilted continuation of its line
            covered_tokens |= tokens_of(members)
            changed = True
        to_merge = still
    for k in to_merge:
        members = claim_lists[k]
        # Never reached a kept group. When the merges have since covered its tokens it is
        # redundant after all — the second-print test then separates a real detached print
        # (the "Shared Care Record" display title whose words also occur in body prose;
        # English tail words a sibling line full-covers) from a garbled stray. A claim whose
        # tokens are STILL new is genuinely detached content and keeps dropping.
        if not (tokens_of(members) - covered_tokens) and _is_second_print(
            members, kept, cells, line_tokens
        ):
            demoted.append(list(members))
        else:
            dropped.extend(members)
    return kept, demoted, dropped


def _merge_leftover_tails(
    groups: list[tuple[int | None, list[int]]],
    cells: list[dict[str, Any]],
    hint_units: list[str],
    hint_token_sets: list[set[str]],
    hint_fuzzy_sets: list[set[str]],
) -> list[tuple[int | None, list[int]]]:
    """The post-consolidation twin of the leftover rescue: a single-cell LEFTOVER that sits
    geometrically directly below a labeled group's member (same column — ``_is_continuation``),
    clears the bind threshold on that group's line AND contributes a token its members do not
    cover yet, is that line's wrapped tail — merge it in so the line's translation reflows over
    both printed lines. The pre-grouping rescue cannot reach these: a tail whose text duplicates
    a short line elsewhere gets LABELED there (full match), consolidation demotes it back to a
    leftover, and only now is its true home decidable. Same gates as the rescue, so a repeated
    print (contributes nothing uncovered) still stays its own unit."""
    labeled = [(label, indices) for label, indices in groups if label is not None]
    out: list[tuple[int | None, list[int]]] = []
    for label, indices in groups:
        if label is not None or len(indices) != 1:
            out.append((label, indices))
            continue
        index = indices[0]
        cell_tokens = _tokens(str(cells[index].get("text") or ""))
        merged = False
        for target_label, target_indices in labeled:
            if not cell_tokens or not any(
                _is_continuation(cells[j], cells[index]) for j in target_indices
            ):
                continue
            on_line = sum(
                _token_score(token, hint_token_sets[target_label], hint_fuzzy_sets[target_label])
                for token in cell_tokens
            )
            if on_line / len(cell_tokens) < _MATCH_THRESHOLD:
                continue
            covered = {
                token for j in target_indices for token in _tokens(str(cells[j].get("text") or ""))
            }
            if not any(token not in covered for token in cell_tokens):
                continue  # a repeated print: contributes nothing -> stays its own unit
            target_indices.append(index)
            target_indices.sort(key=lambda i: (cells[i]["bbox"]["top"], cells[i]["bbox"]["left"]))
            merged = True
            break
        if not merged:
            out.append((label, indices))
    return out


def _is_second_print(
    members: list[int],
    kept: list[list[int]],
    cells: list[dict[str, Any]],
    line_tokens: set[str],
) -> bool:
    """Whether a redundant claim is a genuine OTHER print of duplicated text (see the demote
    branch): every cell token sits EXACTLY in the line (a fuzzy-bound claim is a garbled
    double-read of a print the kept claim already covers) and its box overlaps no kept claim
    (an overlapping clean double-read is the same print, whose pixels the kept claim erases)."""
    for i in members:
        for token in _tokens(str(cells[i].get("text") or "")):
            if token not in line_tokens:
                return False
    box = _members_bbox(members, cells)
    return not any(_bbox_overlap(box, _members_bbox(group, cells)) for group in kept)


def _members_bbox(members: list[int], cells: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    boxes = [cells[i].get("bbox") or {} for i in members]
    return (
        min(float(b.get("left", 0.0)) for b in boxes),
        min(float(b.get("top", 0.0)) for b in boxes),
        max(float(b.get("left", 0.0)) + float(b.get("width", 0.0)) for b in boxes),
        max(float(b.get("top", 0.0)) + float(b.get("height", 0.0)) for b in boxes),
    )


def _bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Whether the intersection covers a meaningful share (30%) of the smaller box."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return smaller > 0 and ix * iy > 0.3 * smaller


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


def _absorb_symbol_leftovers(
    groups: list[tuple[int | None, list[int]]],
    cells: list[dict[str, Any]],
    hint_units: list[str],
    preserved: set[int],
) -> list[tuple[int | None, list[int]]]:
    """Fold a token-less single-cell leftover (a pure symbol — '&', '+', '±') back into the
    element whose printed line it sits on. OCR often splits a styled glyph off its line into its
    own cell; token matching cannot bind it (no tokens), so it orphans as a leftover, keeps its
    original pixels, and the element's hint translation renders the symbol AGAIN next to them (a
    byline's '&' doubled). Absorbed — so its pixels are erased — only when all three hold: the
    cell sits within a word gap of one of the group's members on the same line, the group's hint
    line carries the symbol (the translation really covers it), and the cell is not
    layout-preserved. '|' never absorbs: in hint text it is the field separator, not a printed
    glyph. A symbol the hint does NOT carry (a decorative dingbat) stays its own leftover with
    its original pixels, exactly as before."""
    out: list[tuple[int | None, list[int]]] = []
    for label, indices in groups:
        if label is not None or len(indices) != 1 or indices[0] in preserved:
            out.append((label, indices))
            continue
        index = indices[0]
        text = str(cells[index].get("text") or "").strip()
        if not text or text == "|" or _tokens(text):
            out.append((label, indices))
            continue
        target: list[int] | None = None
        for other_label, other_indices in groups:
            if other_label is None or text not in str(hint_units[other_label] or ""):
                continue
            if any(_word_gap_neighbour(cells[index], cells[j]) for j in other_indices):
                target = other_indices
                break
        if target is None:
            out.append((label, indices))
            continue
        target.append(index)
        target.sort()
    return out


def _word_gap_neighbour(cell: dict[str, Any], other: dict[str, Any]) -> bool:
    """Whether ``other`` sits on ``cell``'s printed line within a word gap — the same
    vertical-overlap and gap rules as ``_line_anchor`` (tilt-tolerant, a column gap is too wide)."""
    box = cell.get("bbox") or {}
    other_box = other.get("bbox") or {}
    left, top = float(box.get("left", 0.0)), float(box.get("top", 0.0))
    height = float(box.get("height", 0.0)) or 1.0
    right, bottom = left + float(box.get("width", 0.0)), top + height
    other_left, other_top = float(other_box.get("left", 0.0)), float(other_box.get("top", 0.0))
    other_height = float(other_box.get("height", 0.0)) or 1.0
    other_right = other_left + float(other_box.get("width", 0.0))
    other_bottom = other_top + other_height
    if (min(bottom, other_bottom) - max(top, other_top)) <= _LINE_VOVERLAP_RATIO * min(height, other_height):
        return False
    gap = max(other_left - right, left - other_right)  # negative when boxes overlap
    return gap <= _LINE_GAP_RATIO * height


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


def _is_continuation(previous_cell: dict[str, Any] | None, cell: dict[str, Any]) -> bool:
    """A wrapped continuation line starts at ~the same left margin as, and directly
    below, its element's previous line. A price in the right column or the next row of
    a receipt does not qualify, so genuinely repeated rows ("BONUS ...") still bind by
    position."""
    if previous_cell is None:
        return False
    prev, this = previous_cell.get("bbox") or {}, cell.get("bbox") or {}
    height = float(this.get("height") or 0.0) or float(prev.get("height") or 0.0)
    if height <= 0:
        return False
    drop = float(this.get("top") or 0.0) - float(prev.get("top") or 0.0)
    indent = abs(float(this.get("left") or 0.0) - float(prev.get("left") or 0.0))
    return 0 < drop <= 2.5 * height and indent <= 2.0 * height
