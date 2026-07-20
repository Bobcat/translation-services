"""Split a receipt/table row into per-column cells, matching members to '|' fields."""
from __future__ import annotations

from typing import Any
import re
import difflib
import unicodedata
from app.grouping.preserve import _is_nontranslatable


# Minimum source-text similarity to bind a table-row field to a cell. Below it the row is
# not split into cells (the renderer reflows it instead) — a wrong field/cell match would
# place text in the wrong column, worse than a reflow.
_FIELD_MATCH_MIN = 0.5
# Evidence bar for DROPPING an unplaceable field because it is printed in another unit: near
# containment, not mere similarity — a short field key ("github") reaches 0.5 on one accidental
# 3-char run in unrelated prose, and a dropped field is text deleted from the render.
_FIELD_ELSEWHERE_MIN = 0.8

def _split_table_row(
    unit: dict[str, Any],
    other_texts: tuple[str, ...] | list[str] = (),
) -> list[dict[str, Any]] | None:
    """A table row (the VLM hint carried '|' fields) becomes one pseudo-unit per rendered
    COLUMN, so the renderer places each at its own x instead of reflowing the joined line over
    the row's union — which would collapse the column gaps and shift the spend text left, right
    behind the company name. This also matters when only one field remains after unchanged fields
    were filtered: render that field in its own cell and leave the preserved neighbour untouched.

    Members are grouped by the field whose SOURCE text they best match (not by order — the VLM
    does not always list fields left-to-right). A field may own SEVERAL members: a column OCR
    split into wrapped lines (a long spend description) then renders its one translation reflowed
    over those lines. Conversely several fields may share one column: a 'PRIJS | BEDRAG' hint that
    OCR read as a single box renders both translations in field order. A translatable member that
    matches NO field stays out of every cell — its original pixels are left standing (its field
    was preserve-dropped from the pairs, e.g. an unchanged company name). Returns None when the
    unit is not such a row, or no requested field can be placed."""
    pairs = unit.get("field_translations")
    if not pairs:
        return None
    # Bind each member (OCR cell) to the field whose SOURCE text it best matches and group members
    # by field — a field may own several members when OCR split a column into wrapped lines. Keep a
    # translatable member always (one we cannot place makes the split unreliable -> reflow). A
    # NON-translatable member joins when it is explicitly present as a field pair (mode "translate
    # everything", e.g. a quantity/price column), or when the field's translation reproduces it (a
    # pure-number line like '2025' the translation re-emits). Other non-translatable members (an
    # icon, a self-standing price not requested for render) are left untouched.
    columns: dict[int, list[dict[str, Any]]] = {}
    for member in _split_straddling_members(unit.get("members") or [], pairs):
        if not member.get("bbox"):
            continue
        best_field, best_score = None, 0.0
        best_rank: tuple[float, int, int] = (0.0, 0, -10**9)
        member_text = str(member.get("text") or "")
        for index, (source, _translated) in enumerate(pairs):
            rank = _field_match_rank(source, member_text)
            if rank > best_rank:
                best_field, best_score, best_rank = index, rank[0], rank
        placed = best_field is not None and best_score >= _FIELD_MATCH_MIN
        if member.get("translate"):
            if not placed:
                continue
        elif not (
            placed
            and (
                _is_nontranslatable(str(pairs[best_field][0]))
                or _reproduced_in(member, pairs[best_field][1])
            )
        ):
            continue
        columns.setdefault(best_field, []).append(member)
    if not columns:
        return None
    _rehome_column_strays(columns, pairs)
    _redistribute_duplicate_value_fields(columns, pairs)
    # A field with no member of its own shares a column (a 'PRIJS | BEDRAG' hint OCR read as one
    # box): attach its translation to the column whose members it best matches, kept in field
    # order so the cell renders the fields as written.
    column_texts: dict[int, list[tuple[int, str]]] = {field: [(field, pairs[field][1])] for field in columns}
    for index, (source, translated) in enumerate(pairs):
        if index in columns:
            continue
        best_field, best_score = None, 0.0
        for field, cell_members in columns.items():
            score = max(_field_overlap(source, str(m.get("text") or "")) for m in cell_members)
            if score > best_score:
                best_field, best_score = field, score
        if best_field is None or best_score < _FIELD_MATCH_MIN:
            # The field's source matches no placed column. Drop it from this unit ONLY on
            # positive evidence that its text is printed in ANOTHER unit (``other_texts``) and
            # not in this one — the VLM unmerges a repeated table column (an icon-margin caption
            # emitted as the first field of EVERY row while its cells exist once, in the unit
            # beside the icon): rendering it here would prepend the caption's translation to
            # every row. Without that evidence the cautious old reflow stands: a merged-box
            # field whose member text is OCR-garbled past the threshold (an order code) is
            # printed HERE, and dropping it would erase text from the image.
            printed_here = any(
                _field_overlap(source, str(member.get("text") or "")) >= _FIELD_MATCH_MIN
                for member in unit.get("members") or []
            )
            printed_elsewhere = any(
                _field_overlap(source, text) >= _FIELD_ELSEWHERE_MIN for text in other_texts
            )
            if printed_elsewhere and not printed_here:
                continue
            return None
        column_texts[best_field].append((index, translated))
    cells: list[dict[str, Any]] = []
    for field, cell_members in columns.items():
        cell = dict(unit)
        cell["table_cell"] = True  # planner: this unit is one table COLUMN (shares its width)
        cell["translated_text"] = " ".join(text for _, text in sorted(column_texts[field]))
        cell["members"] = cell_members
        # The cell's own translatable member texts, NOT the parent row's (dict(unit) copied
        # that): the renderer's identity-preserve compares a cell's translation against its
        # source, and against the whole row every split field would read as changed.
        cell["source_text"] = " ".join(
            str(m.get("text") or "") for m in cell_members if m.get("translate") and m.get("text")
        )
        cell["field_translations"] = None  # already split — don't re-enter
        cells.append(cell)
    return _merge_close_table_cells(cells)

# A straddling member: OCR jammed the line-1 text of TWO adjacent columns into one box
# ("video classification" + "Performance improvements over" with a garble at the seam). Both
# fields then compete for the box, one cell ends up spanning both columns and the cell-merge
# glues the row shut. Detected on strong evidence only: each side must match its field in one
# contiguous run of at least this many key characters (a short shared label pair like
# 'PRIJS BEDRAG' stays on the designed shared-box path), the two runs must be disjoint in the
# member, and together cover most of it. The box is then cut between the runs, proportionally
# in key space (letters+digits ~ ink).
_STRADDLE_MIN_RUN = 10
_STRADDLE_MIN_COVER = 0.6
_STRADDLE_MIN_PART_PX = 8


def _split_straddling_members(
    members: list[dict[str, Any]], pairs: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """``members`` with every straddling box (see _STRADDLE_*) cut into its two field parts;
    all other members pass through unchanged. The parts carry a shared ``split_part`` marker so
    the cell-merge can refuse to re-glue the cut."""
    out: list[dict[str, Any]] = []
    for member in members:
        parts = _straddle_parts(member, pairs)
        out.extend(parts if parts else [member])
    return out


def _straddle_parts(
    member: dict[str, Any], pairs: list[tuple[str, str]]
) -> list[dict[str, Any]] | None:
    if not member.get("translate") or not member.get("bbox"):
        return None
    raw = str(member.get("text") or "")
    key, raw_positions = _key_positions(raw)
    if len(key) < 2 * _STRADDLE_MIN_RUN:
        return None
    runs: list[tuple[int, int, int]] = []  # (start, end) in member key, field index
    for index, (source, _translated) in enumerate(pairs):
        field_key = _field_key(source)
        if len(field_key) < _STRADDLE_MIN_RUN:
            continue
        block = max(
            difflib.SequenceMatcher(None, key, field_key).get_matching_blocks(),
            key=lambda b: b.size,
        )
        if block.size >= _STRADDLE_MIN_RUN:
            runs.append((block.a, block.a + block.size, index))
    runs.sort(key=lambda run: run[1] - run[0], reverse=True)
    for i in range(len(runs)):
        for j in range(len(runs)):
            if i == j or runs[i][2] == runs[j][2]:
                continue
            left, right = (runs[i], runs[j]) if runs[i][1] <= runs[j][0] else (runs[j], runs[i])
            if left[1] > right[0]:
                continue  # overlapping runs: duplicated value, not a straddle
            if (left[1] - left[0]) + (right[1] - right[0]) < _STRADDLE_MIN_COVER * len(key):
                continue
            return _cut_member(member, raw, key, raw_positions, left, right)
    return None


def _cut_member(
    member: dict[str, Any],
    raw: str,
    key: str,
    raw_positions: list[int],
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> list[dict[str, Any]]:
    """``member`` cut between its two field runs: text split at the seam (the garble between
    the runs is dropped), bbox split proportionally in key space."""
    bbox = member["bbox"]
    fraction = ((left[1] + right[0]) / 2.0) / len(key)
    cut = int(round(float(bbox["width"]) * fraction))
    if cut < _STRADDLE_MIN_PART_PX or float(bbox["width"]) - cut < _STRADDLE_MIN_PART_PX:
        return None
    marker = member.get("cell_id") if member.get("cell_id") is not None else id(member)
    parts: list[dict[str, Any]] = []
    texts = (raw[: raw_positions[left[1] - 1] + 1].strip(), raw[raw_positions[right[0]]:].strip())
    boxes = (
        {**bbox, "width": cut},
        {**bbox, "left": bbox["left"] + cut, "width": bbox["width"] - cut},
    )
    for text, box in zip(texts, boxes):
        part = dict(member)
        part.pop("polygon", None)  # the cut is bbox-space; a stale quad would span the seam
        part["text"] = text
        part["bbox"] = box
        part["split_part"] = marker
        parts.append(part)
    return parts


def _key_positions(text: str) -> tuple[str, list[int]]:
    """``(_field_key(text), raw index of every key character)`` — the same normalization,
    tracked per character so a key-space cut maps back to a raw-text position."""
    key_chars: list[str] = []
    positions: list[int] = []
    for index, ch in enumerate(str(text or "")):
        folded = unicodedata.normalize("NFKD", ch.lower())
        for piece in folded:
            if unicodedata.combining(piece) or re.match(r"[\W_]", piece):
                continue
            key_chars.append(piece)
            positions.append(index)
    return "".join(key_chars), positions


# Members of ONE table cell stack vertically: wrapped lines of a column overlap in x by most
# of the narrower box. Below this fraction two members are in different physical columns.
_COLUMN_STACK_OVERLAP = 0.3


def _rehome_column_strays(
    columns: dict[int, list[dict[str, Any]]], pairs: list[tuple[str, str]]
) -> None:
    """Geometry arbitrates a duplicated-value tie the text match cannot see.

    A value printed in TWO physical columns (the same phrase in two table cells) matches one
    field best for BOTH copies — exact beats containment — so the far copy is pulled across
    the page into that field's cell, which then spans two columns; the field whose source
    carries the shared phrase plus more ("<phrase> via crowdsourcing") is left with only its
    remainder fragment and squeezes its whole translation onto that sliver. Members of one
    cell stack in x (wrapped lines overlap), so a member x-disjoint from every other member
    of its field is a stray; when another field's members DO x-overlap it and that field's
    source text-matches it too, it moves there. No-op for x-coherent cells — single-member
    fields and genuinely wrapped columns are untouched."""
    for field, members in columns.items():
        if len(members) < 2:
            continue
        for member in list(members):
            others = [m for m in members if m is not member]
            if not others:
                continue  # a lone member is trivially coherent (an earlier stray moved out)
            if any(_x_overlap_fraction(member, other) >= _COLUMN_STACK_OVERLAP for other in others):
                continue  # stacks with its own cell
            member_text = str(member.get("text") or "")
            for other_field, other_members in columns.items():
                if other_field == field:
                    continue
                if _field_overlap(str(pairs[other_field][0]), member_text) < _FIELD_MATCH_MIN:
                    continue
                if any(
                    _x_overlap_fraction(member, om) >= _COLUMN_STACK_OVERLAP
                    for om in other_members
                ):
                    members.remove(member)
                    other_members.append(member)
                    break


def _redistribute_duplicate_value_fields(
    columns: dict[int, list[dict[str, Any]]], pairs: list[tuple[str, str]]
) -> None:
    """A DUPLICATED column-header value (a multi-level header: "EN-DE"/"EN-FR" appear once
    under each group header) is the case _rehome_column_strays cannot fix: both cells match
    the FIRST same-value field, so both bind there and the second field stays empty — the two
    physical columns collapse into one ("EN-DE EN-DE") and the far cells erase with nothing
    redrawn. Here the target field is empty, so the stray has nowhere to move under the x-
    overlap rule. Pool the members of all same-value fields, cluster them into physical columns
    by x, and lay one cluster per field occurrence (x-order to field-index order) so each column
    renders at its own x. Only acts when the physical column count matches the field count — an
    exact multi-level header — leaving every other table untouched (the values are identical, so
    which field-index a column takes does not change its rendered text)."""
    fields_by_value: dict[str, list[int]] = {}
    for index, (source, _translated) in enumerate(pairs):
        fields_by_value.setdefault(_field_key(source), []).append(index)
    for indices in fields_by_value.values():
        if len(indices) < 2:
            continue
        pooled = [member for index in indices for member in columns.get(index, [])]
        clusters = _cluster_members_by_x(pooled)
        if len(clusters) != len(indices):
            continue  # not an exact one-column-per-field split: leave the text match's result
        clusters.sort(key=lambda cluster: min(float((m.get("bbox") or {}).get("left") or 0.0) for m in cluster))
        for index, cluster in zip(indices, clusters):
            columns[index] = cluster


def _cluster_members_by_x(members: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group members into physical columns: a member joins a cluster it x-overlaps (wrapped
    lines of one column stack in x); otherwise it starts a new column."""
    clusters: list[list[dict[str, Any]]] = []
    for member in sorted(members, key=lambda m: float((m.get("bbox") or {}).get("left") or 0.0)):
        for cluster in clusters:
            if any(_x_overlap_fraction(member, other) >= _COLUMN_STACK_OVERLAP for other in cluster):
                cluster.append(member)
                break
        else:
            clusters.append([member])
    return clusters


def _x_overlap_fraction(a: dict[str, Any], b: dict[str, Any]) -> float:
    """x-overlap of two members' boxes as a fraction of the narrower box."""
    box_a, box_b = a.get("bbox") or {}, b.get("bbox") or {}
    a0 = float(box_a.get("left") or 0.0)
    a1 = a0 + float(box_a.get("width") or 0.0)
    b0 = float(box_b.get("left") or 0.0)
    b1 = b0 + float(box_b.get("width") or 0.0)
    narrower = min(a1 - a0, b1 - b0)
    if narrower <= 0:
        return 0.0
    return (min(a1, b1) - max(a0, b0)) / narrower


def _merge_close_table_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent table fields that are really one visual column.

    Weather forecast rows often split a date and weekday as ``13 jun | Za`` even though the
    columns touch. Rendering them separately makes the translated date overrun into the weekday
    ("Jun 13Sat"). Distant fields such as menu prices stay separate.
    """
    ordered = sorted(cells, key=lambda cell: (_cell_axis_box(cell)[0], _cell_axis_box(cell)[1]))
    merged: list[dict[str, Any]] = []
    for cell in ordered:
        # Never re-glue a deliberate straddle cut: the two parts of one OCR box are adjacent by
        # construction, so the gap gauge would always merge them back into a two-column cell.
        if (
            not merged
            or _cells_share_split_box(merged[-1], cell)
            or not _should_merge_table_cells(merged[-1], cell)
        ):
            merged.append(cell)
            continue
        previous = dict(merged[-1])
        previous["translated_text"] = " ".join(
            part for part in (previous.get("translated_text"), cell.get("translated_text")) if part
        )
        previous["source_text"] = " ".join(
            part for part in (previous.get("source_text"), cell.get("source_text")) if part
        )
        previous["members"] = list(previous.get("members") or []) + list(cell.get("members") or [])
        merged[-1] = previous
    return merged

def _cells_share_split_box(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether the two cells hold parts of the same straddle-cut OCR box (see split_part)."""
    parts_a = {
        m.get("split_part") for m in (a.get("members") or []) if m.get("split_part") is not None
    }
    return any(m.get("split_part") in parts_a for m in (b.get("members") or []))


def _should_merge_table_cells(left_cell: dict[str, Any], right_cell: dict[str, Any]) -> bool:
    left = _cell_axis_box(left_cell)
    right = _cell_axis_box(right_cell)
    height = max(left[3] - left[1], right[3] - right[1], 1.0)
    y_overlap = min(left[3], right[3]) - max(left[1], right[1])
    if y_overlap < 0.5 * height:
        return False
    gap = right[0] - left[2]
    # Gauge the gap against a LINE height (median member box), not the cells' union height: a
    # column whose text WRAPS (an icon-margin caption of two lines) doubles its union height and
    # would double the allowed gap — merging the caption into the content column it sits beside.
    # Single-line fields (a date/weekday pair) keep exactly the old gauge.
    line_height = max(_median_member_height(left_cell), _median_member_height(right_cell), 1.0)
    return gap <= 0.75 * line_height


def _median_member_height(cell: dict[str, Any]) -> float:
    heights = sorted(
        float((member.get("bbox") or {}).get("height") or 0.0)
        for member in (cell.get("members") or [])
        if member.get("bbox")
    )
    return heights[len(heights) // 2] if heights else 0.0

def _cell_axis_box(cell: dict[str, Any]) -> tuple[float, float, float, float]:
    boxes = [(member.get("bbox") or {}) for member in (cell.get("members") or []) if member.get("bbox")]
    if not boxes:
        return 0.0, 0.0, 0.0, 0.0
    left = min(float(box.get("left") or 0.0) for box in boxes)
    top = min(float(box.get("top") or 0.0) for box in boxes)
    right = max(float(box.get("left") or 0.0) + float(box.get("width") or 0.0) for box in boxes)
    bottom = max(float(box.get("top") or 0.0) + float(box.get("height") or 0.0) for box in boxes)
    return left, top, right, bottom

def _field_key(text: str) -> str:
    """Comparable letters+digits of a field text, script-agnostic: lowercased, diacritics folded
    (café == cafe), every non-word character dropped. An ASCII-only key ([a-z0-9]) would be EMPTY
    for Cyrillic/Greek/CJK source text — every rank ties at zero, no member ever places, and the
    column split silently never fires for non-Latin rows."""
    folded = unicodedata.normalize("NFKD", str(text or "").lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[\W_]", "", folded)

def _field_overlap(a: str, b: str) -> float:
    """Overlap of a row member against a field source: matched run length over the SHORTER of the
    two (letters+digits only, tolerant to OCR garble like AHNEDAARBEI vs AHNEDAARDBEI). 1.0 when one
    contains the other, so a column OCR split into wrapped fragments ('advertising &') still binds
    to its full field source — which a symmetric ratio scores too low to clear _FIELD_MATCH_MIN.

    Only CONTIGUOUS runs of 3+ characters count (capped at the shorter key's length, so a 1-2 char
    member can still match by containment). A short member against a LONG field otherwise reaches
    the placement threshold on scattered 1-2 char noise — "Amazon" collects 4 stray characters in
    "$23,5 miljard aan advertising ... 2025" (4/6 = 0.67) and a company-name cell whose own field
    was preserve-dropped from the pairs then lands in the SPEND cell: the row erase swallows the
    company name and the spend text renders at the company's column. Genuine fragments match in
    one full-length run, garble in a few long runs; both are untouched by the filter."""
    na = _field_key(a)
    nb = _field_key(b)
    if not na or not nb:
        return 0.0
    min_block = min(3, len(na), len(nb))
    matched = sum(
        block.size
        for block in difflib.SequenceMatcher(None, na, nb).get_matching_blocks()
        if block.size >= min_block
    )
    return matched / min(len(na), len(nb))

def _field_match_rank(source: str, member_text: str) -> tuple[float, int, int]:
    source_key = _field_key(source)
    member_key = _field_key(member_text)
    if not source_key or not member_key:
        return 0.0, 0, -10**9
    score = _field_overlap(source, member_text)
    exact = int(source_key == member_key)
    length_closeness = -abs(len(source_key) - len(member_key))
    return score, exact, length_closeness

def _reproduced_in(member: dict[str, Any], translated: str) -> bool:
    """Whether a NON-translatable member's text is re-emitted by the unit's translation, which
    contains more than just that member. OCR sometimes splits an inline non-translatable token
    (a "1, 2, 3, 4?", a code, a URL) into its own member; the structured translation still
    translates the whole hint line, so it reproduces that token inline. Keeping the original on top
    then doubles it AND drops it from the erase/plane set (the original peeks through). We erase
    such a member like a translatable one — but only when the translation carries OTHER tokens too,
    so a standalone token translating to itself (a lone price) is left untouched.

    OCR also merges a short neighbouring word into the box (``op www.ikstopnu.nl`` — the "op" of
    "Kijk op" pulled into the URL cell), and the translation rephrases that word ("Visit") rather
    than echoing it. A token of 1-2 chars may therefore be missing; the DISTINCTIVE (>2-char)
    tokens carry the identity, so reproduction is judged on those — every long token must be
    reproduced, and the match must rest on a long token (or on the whole short member, the
    "1,2,3,4" case), never on a stray short token alone."""
    member_tokens = re.findall(r"\w+", str(member.get("text") or "").lower())
    if not member_tokens:
        return False
    translation_tokens = set(re.findall(r"\w+", str(translated or "").lower()))
    missing = [token for token in member_tokens if token not in translation_tokens]
    if any(len(token) > 2 for token in missing):  # a distinctive token is absent -> not reproduced
        return False
    long_reproduced = any(len(token) > 2 for token in member_tokens if token in translation_tokens)
    if missing and not long_reproduced:  # only short tokens matched -> too weak to call reproduced
        return False
    return bool(translation_tokens - set(member_tokens))


