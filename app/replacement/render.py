"""Stage #8 re-placement — background-matched, polygon-aware (Tier-1, model-free).

Units that share a VLM block (a wrapped dish, a body paragraph) render as one
**group**: their translations are joined and balanced over the original number of
lines, at ONE font size taken from the original's true line height — the **source
size**, so a heading stays heading-sized and body stays body-sized (the source size
carries the visual hierarchy). Width is matched separately by **horizontal condensation**:
at the source height a translated line is usually wider than its original, so the rendered
text is squeezed in x to fit the original line's width (floored, never stretched) — keeping
height while matching width, the way the reference render does. Each rendered line anchors
on its original line's plane (so the line pitch follows the original). Per plane: cover
the original with the locally-sampled **background colour** (so it reads as erased
on a flat surface — menu paper, sign panel, receipt), then draw the line and **warp
it onto the plane's polygon** so it follows the page tilt (rotation/perspective),
for a clean camera-translation look.

Two facts make this work without a model:
- the OCR polygon gives the **true line height** (tilt-invariant), so text is sized
  consistently instead of by the inflated axis-aligned bbox — see `geometry`;
- the polygon also gives the **angle**, so a flat RGBA text tile can be warped to the
  oriented region with OpenCV.

`translate: false` members and ignored cells are never touched. Textured/photographic
backgrounds still scar (a flat fill can't blend) — that is the LaMa (Tier 2) case.
See docs/re-placement.md.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw

from app.grouping.heuristics import _is_nontranslatable
from app.replacement import geometry as geo
from app.replacement.color import contrasting_fg
from app.replacement.color import sample_oriented_colors
from app.replacement.fit import break_pieces
from app.replacement.fit import fold_lone_fullwidth_punctuation
from app.replacement.fit import is_cjk_text
from app.replacement.fit import load_font


# Font size from the true (de-skewed) line height. The polygon height spans the full
# glyph extent, a touch taller than the visual cap; scale down slightly to match.
_SIZE_RATIO = 0.9
# CJK glyphs fill the em (ink ~= the full line box), where Latin ink is upper-biased and
# leaves ~30% leading below it. At _SIZE_RATIO the CJK ink overruns the source line pitch and
# consecutive lines touch/overlap, so CJK lines map height->size with a smaller ratio, taking
# roughly the same visual footprint a Latin line would. Hierarchy (relative sizes) is kept.
_CJK_SIZE_RATIO = 0.72

# Floor on horizontal condensation. The font is sized from the source HEIGHT (so the
# header/body hierarchy is preserved); a translated line at that height is usually wider
# than its original (most sans are wider than the sign's font), so the rendered text is
# squeezed horizontally to fit the original line's width — keeping height, matching width,
# the way the reference render does. Never squeeze past this floor: below it the glyphs
# read as unnaturally narrow, so the pt size is reduced instead (see _WIDTH_SLACK).
_CONDENSE_FLOOR = 0.75

# A rendered line may exceed its original plane width by this factor before we spend pt.
# Order of accommodation for a too-long translation: condense horizontally to the floor,
# then allow up to this much overrun, and only if it STILL doesn't fit reduce the source
# pt size (re-wrapping) — so the source size (and the header/body hierarchy) is preserved
# unless the line genuinely cannot fit the box within the slack.
_WIDTH_SLACK = 1.04
# Floor on the pt-shrink search (matches the plane target floor in _plan_group).
_MIN_RENDER_SIZE = 8
# A line plane narrower than this fraction of the unit's other lines, AND carrying only words
# already present on those lines, is an OCR stray (a neighbouring element's word pulled in) — not
# a real wrapped line. Kept, it forms a sliver plane that starves the whole unit's width fit.
_STRAY_LINE_WIDTH_RATIO = 0.5

# Below this group angle (degrees) the text is treated as horizontal and placed axis-aligned,
# so OCR detection noise on a flat image isn't warped into a visible slant. A genuine page
# tilt is well above it (a photographed menukaart sits at ~6°), so real perspective is kept.
_ANGLE_DEADZONE_DEG = 3.0

# Minimum source-text similarity to bind a table-row field to a cell. Below it the row is
# not split into cells (the renderer reflows it instead) — a wrong field/cell match would
# place text in the wrong column, worse than a reflow.
_FIELD_MATCH_MIN = 0.5

# Per-channel tolerance for snapping a group's per-plane background samples to one
# colour: within it the planes are one surface sampled with texture noise; beyond it
# they are genuinely different (a gradient, two panels) and stay per-plane.
_BG_SNAP_DELTA = 24

# Erase margin ABOVE/BELOW the original text. The OCR polygon's ``ymin``/``ymax`` already bound
# the glyphs (descenders included), so vertically the erase needs only a thin anti-alias margin
# — not the full ``pad``. The full pad would reach into whatever sits just above or below the
# line (a coloured header band a few px away) and erase it; a tight vertical margin keeps the
# fill on the text. The sides keep ``pad`` (and grow with the tile) for horizontal blending.
_ERASE_MARGIN = 2.0


@dataclass(frozen=True)
class _Job:
    # One tight quad per original member (word), each at its OWN tilt — not a single
    # line-spanning rectangle. A photographed line fans (perspective steepens along it),
    # so one straight rectangle pinned to the highest word floats above the lower ones;
    # with a flat fill that overshoot paints background colour past the text's band.
    erase_quads: list[list[tuple[int, int]]]
    bg_color: tuple[int, int, int]
    # None for an erase-only plane (the translation needed fewer lines than the original).
    tile: Image.Image | None
    dst_quad: list[tuple[float, float]] | None


def render_translated_image(
    input_path: Path, translation_units: list[dict[str, Any]], *, render_size_mode: str = "min"
) -> bytes:
    opened = Image.open(input_path)
    # Carry the source's ICC colour profile onto the output. ``convert("RGB")`` keeps the raw pixel
    # values but drops the profile, so without re-embedding it a colour-managed display (a phone)
    # shows the replacement as plain sRGB — the whole image reads duller/darker than the original.
    icc_profile = opened.info.get("icc_profile")
    base = opened.convert("RGB")

    jobs: list[_Job] = []
    groups = _groups(translation_units)
    snap_horizontal = _image_is_flat(translation_units)
    for group in groups:
        jobs.extend(_plan_group(base, group, snap_horizontal=snap_horizontal, render_size_mode=render_size_mode))

    # Pass 1: cover every original (along the slant) so no source text peeks through.
    erase = ImageDraw.Draw(base)
    for job in jobs:
        for quad in job.erase_quads:
            erase.polygon(quad, fill=job.bg_color)

    # Pass 2: warp each text tile onto its oriented region.
    canvas = np.asarray(base).copy()
    for job in jobs:
        if job.tile is not None:
            _composite(canvas, job)

    out = BytesIO()
    save_kwargs: dict[str, Any] = {"compress_level": 1}
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    Image.fromarray(canvas).save(out, format="PNG", **save_kwargs)
    return out.getvalue()


def _groups(units: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Consecutive units of one VLM block at one level reflow together — a wrapped
    dish, a body paragraph. The level guard keeps a heading from merging into its
    body text. Leftovers (no block — an OCR noise cell interleaved in reading order)
    stay alone but do NOT break the surrounding block's run, or one stray cell would
    split a dish back into per-line fitting."""
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    previous: tuple[Any, Any] | None = None
    for unit in units:
        key = (unit.get("block_id"), unit.get("level"))
        if key[0] is None:
            groups.append([unit])
            continue
        if current is not None and key == previous:
            current.append(unit)
        else:
            current = [unit]
            groups.append(current)
        previous = key
    return groups


def _image_is_flat(units: list[dict[str, Any]]) -> bool:
    """True when the image as a whole reads as fronto-parallel — its lines are near-horizontal
    with no real page tilt, so per-line angles are OCR detection noise to be snapped away. A
    photographed sign at an angle has a perspective gradient with a sizeable median angle and is
    NOT flat, so its (real) angles are kept. Median over all member quads is robust to the odd
    rotated stray (a lone tall glyph) that a mean would be skewed by."""
    angles: list[float] = []
    for unit in units:
        for member in unit.get("members") or []:
            if not member.get("bbox"):
                continue
            quad = geo.quad_of(member)
            if quad is not None:
                angles.append(abs(geo.angle_deg(quad)))
    return bool(angles) and median(angles) < _ANGLE_DEADZONE_DEG


def _split_table_row(unit: dict[str, Any]) -> list[dict[str, Any]] | None:
    """A table row (the VLM hint carried '|' fields) becomes one pseudo-unit per rendered
    COLUMN, so the renderer places each at its own x instead of reflowing the joined line over
    the row's union — which would collapse the column gaps and shift the spend text left, right
    behind the company name. This also matters when only one field remains after unchanged fields
    were filtered: render that field in its own cell and leave the preserved neighbour untouched.

    Members are grouped by the field whose SOURCE text they best match (not by order — the VLM
    does not always list fields left-to-right). A field may own SEVERAL members: a column OCR
    split into wrapped lines (a long spend description) then renders its one translation reflowed
    over those lines. Conversely several fields may share one column: a 'PRIJS | BEDRAG' hint that
    OCR read as a single box renders both translations in field order. Returns None when the unit
    is not such a row, or no requested field can be placed."""
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
    for member in unit.get("members") or []:
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
            return None
        column_texts[best_field].append((index, translated))
    cells: list[dict[str, Any]] = []
    for field, cell_members in columns.items():
        cell = dict(unit)
        cell["translated_text"] = " ".join(text for _, text in sorted(column_texts[field]))
        cell["members"] = cell_members
        cell["field_translations"] = None  # already split — don't re-enter
        cells.append(cell)
    return _merge_close_table_cells(cells)


def _merge_close_table_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent table fields that are really one visual column.

    Weather forecast rows often split a date and weekday as ``13 jun | Za`` even though the
    columns touch. Rendering them separately makes the translated date overrun into the weekday
    ("Jun 13Sat"). Distant fields such as menu prices stay separate.
    """
    ordered = sorted(cells, key=lambda cell: (_cell_axis_box(cell)[0], _cell_axis_box(cell)[1]))
    merged: list[dict[str, Any]] = []
    for cell in ordered:
        if not merged or not _should_merge_table_cells(merged[-1], cell):
            merged.append(cell)
            continue
        previous = dict(merged[-1])
        previous["translated_text"] = " ".join(
            part for part in (previous.get("translated_text"), cell.get("translated_text")) if part
        )
        previous["members"] = list(previous.get("members") or []) + list(cell.get("members") or [])
        merged[-1] = previous
    return merged


def _should_merge_table_cells(left_cell: dict[str, Any], right_cell: dict[str, Any]) -> bool:
    left = _cell_axis_box(left_cell)
    right = _cell_axis_box(right_cell)
    height = max(left[3] - left[1], right[3] - right[1], 1.0)
    y_overlap = min(left[3], right[3]) - max(left[1], right[1])
    if y_overlap < 0.5 * height:
        return False
    gap = right[0] - left[2]
    return gap <= 0.75 * height


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
    to its full field source — which a symmetric ratio scores too low to clear _FIELD_MATCH_MIN."""
    na = _field_key(a)
    nb = _field_key(b)
    if not na or not nb:
        return 0.0
    matched = sum(block.size for block in difflib.SequenceMatcher(None, na, nb).get_matching_blocks())
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


# An ALPHANUMERIC enumerate marker at the START of a cell: "1."/"2)"/"(a)"/"A."/"ii.". OCR reads the
# digit/letter reliably, so we redraw it as text on the cell. A GLYPH bullet ("•"/"*"/"-"/"◊") is
# deliberately NOT matched here: glyphs route to the ink-scan path that keeps the original glyph in
# place, which renders the SAME glyph uniformly whether or not OCR happened to read it on a given line
# (mixed OCR recognition across a bullet list otherwise splits identical bullets over two paths). The
# trailing ``(?=\s)`` keeps a price ("1.69") or a word from matching.
_ENUMERATE_MARKER = re.compile(
    r"^\s*(\([A-Za-z0-9]{1,3}\)|[A-Za-z0-9]{1,3}[.)])(?=\s)"
)


def _cell_marker(unit: dict[str, Any]) -> str | None:
    """The alphanumeric enumerate marker at the start of the cell, else ``None`` (no marker, or a glyph
    bullet that the ink-scan path handles). The VLM's captured marker counts only when it both leads the
    source AND is itself an enumerate form — otherwise we fall back to the pattern OCR put there."""
    source = str(unit.get("source_text") or "")
    bullet_marker = str(unit.get("bullet_marker") or "")
    if bullet_marker and source.lstrip().startswith(bullet_marker) and _ENUMERATE_MARKER.match(f"{bullet_marker} "):
        return bullet_marker
    match = _ENUMERATE_MARKER.match(source)
    return match.group(1) if match else None


def _prepend_marker(units: list[dict[str, Any]], marker: str) -> list[dict[str, Any]]:
    """A shallow copy of ``units`` with ``marker`` prepended to the first translatable line when the
    translation dropped it (idempotent), so the redrawn line keeps its "1."/"(a)" at the cell's place."""
    out = list(units)
    for index, unit in enumerate(out):
        text = str(unit.get("translated_text") or "").strip()
        if len(text) <= 1:
            continue
        if not text.lstrip().startswith(marker):
            copy = dict(unit)
            copy["translated_text"] = f"{marker} {text}"
            out[index] = copy
        break
    return out


# A glyph marker ("•"/"*"/"-"/"◊"...) that may lead the translated text. The ink-scan path keeps the
# ORIGINAL glyph in the image, so a glyph still in the text would render twice — strip one leading glyph
# (plus its space) before the inset. Alphanumeric markers take the redraw path (_prepend_marker) instead.
_LEADING_GLYPH = re.compile(r"^\s*[•·∙●○◦‣⁃*–—-]\s+")


def _strip_leading_glyph(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A shallow copy of ``units`` with a single leading glyph marker removed from the first translatable
    line, so the ink-scan path (which keeps the original glyph in place) does not render it twice."""
    out = list(units)
    for index, unit in enumerate(out):
        text = str(unit.get("translated_text") or "").strip()
        if len(text) <= 1:
            continue
        stripped = _LEADING_GLYPH.sub("", text, count=1)
        if stripped != text:
            copy = dict(unit)
            copy["translated_text"] = stripped
            out[index] = copy
        break
    return out


def _plan_group(
    base: Image.Image,
    units: list[dict[str, Any]],
    *,
    snap_horizontal: bool = False,
    render_size_mode: str = "min",
) -> list[_Job]:
    if len(units) == 1:
        cells = _split_table_row(units[0])
        if cells is not None:
            return [
                job
                for cell in cells
                for job in _plan_group(
                    base, [cell], snap_horizontal=snap_horizontal, render_size_mode=render_size_mode
                )
            ]

    # Marker routing by TYPE, not by whether OCR read it. An alphanumeric enumerate marker ("1."/"(a)")
    # is redrawn as text on the cell (the per-cell erase wipes the original, we redraw -> aligned); re-add
    # it if the translation dropped it. A glyph bullet ("•"/"*"/"◊") routes to the ink-scan path below
    # (``loose_glyph``), which keeps the original glyph in place (erase clipped off it) and insets the
    # text — so we strip a leftover glyph from the text first to avoid drawing it twice.
    cell_marker = next((marker for unit in units if (marker := _cell_marker(unit))), None)
    loose_glyph = cell_marker is None and any(unit.get("bullet") for unit in units)
    if cell_marker:
        units = _prepend_marker(units, cell_marker)
    elif loose_glyph:
        units = _strip_leading_glyph(units)

    texts: list[str] = []
    group_quads: list = []
    quad_tokens: list[set[str]] = []  # source words of each quad's member, parallel to group_quads
    for unit in units:
        translated = fold_lone_fullwidth_punctuation(str(unit.get("translated_text") or "").strip())
        # Empty or an OCR-noise single char -> leave the original alone. A single CJK character
        # is a full word ("PUSH" -> "推"), not noise, and must render.
        if not translated or (len(translated) == 1 and not is_cjk_text(translated)):
            continue
        members = [
            m for m in (unit.get("members") or [])
            if m.get("bbox") and (m.get("translate") or _reproduced_in(m, translated))
        ]
        placed = [(m, quad) for m in members if (quad := geo.quad_of(m)) is not None]
        if not placed:
            continue
        texts.append(translated)
        for member, quad in placed:
            group_quads.append(quad)
            quad_tokens.append(set(re.findall(r"[^\W\d_]+", str(member.get("text") or "").lower())))
    if not texts:
        return []

    # Planes come from geometry, not from the unit shape: cluster the group's member
    # quads into physical text lines. An element-level hint yields ONE unit spanning
    # several printed lines; a per-line hint yields one unit per line — both cluster
    # to the same planes.
    # On a flat (digital / fronto-parallel) image the lines are truly horizontal; OCR still
    # detects each quad a degree or so off, and warping the tile to that noise turns straight
    # text visibly slanted. When the WHOLE image reads as flat (``snap_horizontal``), snap a
    # near-horizontal group angle to 0. A genuinely tilted sign is NOT flat (its angles form a
    # perspective gradient — afstand-houden runs ~1° at the top to ~8° at the bottom), so its
    # small top-line angles are kept; snapping only those would break the gradient.
    angle = median(geo.angle_deg(quad) for quad in group_quads)
    clusters = _line_clusters(group_quads, angle)
    # The rendered text is warped to this angle, so it must match the band the words actually sit
    # on. Per-quad edge angles are noisy (a short word's OCR quad reads several degrees off), and
    # their median comes out biased shallow — the rendered line then drifts off a tilted band. The
    # baseline FIT through the word centres recovers the true line direction; the parallel lines of
    # a block share it, so it keeps them parallel. Falls back to the quad-median when too few words
    # carry a baseline (a one-word line, a vertical stack of single words).
    angle = _baseline_angle(clusters, angle)
    if snap_horizontal and abs(angle) < _ANGLE_DEADZONE_DEG:
        angle = 0.0
    size_ratio = _CJK_SIZE_RATIO if any(is_cjk_text(text) for text in texts) else _SIZE_RATIO
    planes: list[dict[str, Any]] = []
    for quads in clusters:
        true_height = median(geo.line_height(quad) for quad in quads)
        x_axis, y_axis, xmin, xmax, ymin, ymax = geo.oriented_frame(quads, angle)
        planes.append({
            "quads": quads,
            "tokens": _plane_source_tokens(quads, group_quads, quad_tokens),
            "target": max(8, int(true_height * size_ratio)),
            "pad": max(2.0, true_height / 6.0),
            "frame": (x_axis, y_axis, xmin, xmax, ymin, ymax),
            "width": xmax - xmin,
        })
    planes = _drop_redundant_stray_planes(planes)

    # Loose-glyph bullet (OCR never read it, e.g. "•"): keep the original glyph by starting the
    # erase/anchor at the text on the first plane — the line that carries the bullet — and centre the
    # re-rendered text on the bullet so they line up. (A marker OCR DID read is redrawn as text above.)
    if planes and loose_glyph:
        x_axis, y_axis, xmin, xmax, ymin, ymax = planes[0]["frame"]
        geometry = _bullet_geometry(base, planes[0]["frame"], angle)
        if geometry is not None and xmin < geometry[0] < xmax:
            text_start, bullet_y = geometry
            planes[0]["frame"] = (x_axis, y_axis, text_start, xmax, ymin, ymax)
            planes[0]["width"] = xmax - text_start
            planes[0]["bullet_y"] = bullet_y

    # The whole group renders at ONE size = the original's source size (true line height),
    # NOT a size chosen to fit the width. So a heading keeps heading size and body keeps
    # body size — the source size carries the hierarchy. The joined translation is balanced
    # over the original line count.
    # The units of a group share one VLM element, so one font family/weight. Take the first
    # that carries a hint (leftovers have none -> fall back to the default font).
    family = next((u.get("font_family") for u in units if u.get("font_family")), None)
    weight = next((u.get("font_weight") for u in units if u.get("font_weight")), None)
    joined = " ".join(texts)
    plane_widths = [plane["width"] for plane in planes]
    # Render at the source size, but spend pt only as a last resort: if even at the condense
    # floor a line would still exceed its plane by more than _WIDTH_SLACK, step the size down
    # (which re-wraps) until the floor suffices or the size floor is hit. If the minimum size
    # still cannot fit, leave the original pixels; this catches chatty model replies on tiny
    # OCR-noise cells instead of erasing far beyond the source footprint.
    size = _group_size(planes, render_size_mode)
    font, lines = _fit_group(joined, size=size, plane_widths=plane_widths, family=family, weight=weight)
    while size > _MIN_RENDER_SIZE and _raw_condense(font, lines, planes) < _CONDENSE_FLOOR:
        size -= 1
        font, lines = _fit_group(joined, size=size, plane_widths=plane_widths, family=family, weight=weight)
    if _raw_condense(font, lines, planes) < _CONDENSE_FLOOR:
        return []
    ascent, descent = font.getmetrics()
    centered = any(str(unit.get("alignment") or "") == "center" for unit in units)

    # Width is matched by horizontal condensation, not by shrinking the font: at the source
    # size the translated line is usually wider than its original, so squeeze it in x to fit
    # the original line's width (floored at _CONDENSE_FLOOR). One factor for the whole group
    # keeps a multi-line block visually coherent; never stretch (cap at 1.0), so a shorter
    # line just stays narrower.
    condense = _condense_scale(font, lines, planes)

    # One element usually sits on one surface: when the per-plane background samples
    # are near-equal (texture noise), snap them to their median so the erase planes
    # don't show slightly different shades per line.
    colors = [sample_oriented_colors(base, _plane_corners(plane)) for plane in planes]
    if len(colors) > 1:
        median_bg = tuple(int(median(bg[channel] for bg, _ in colors)) for channel in range(3))
        if all(max(abs(bg[c] - median_bg[c]) for c in range(3)) <= _BG_SNAP_DELTA for bg, _ in colors):
            colors = [(median_bg, contrasting_fg(median_bg))] * len(colors)

    jobs: list[_Job] = []
    for index, plane in enumerate(planes):
        x_axis, y_axis, xmin, xmax, ymin, ymax = plane["frame"]
        pad = plane["pad"]
        bg, fg = colors[index]
        # Origin = the original line's top-left in the rotated frame — line pitch and
        # perspective follow the original print, whatever the new break positions are.
        # A centered element anchors each line on its plane's CENTRE instead (the VLM
        # alignment hint); a wrong hint only moves text within the plane, nothing else.
        oy = ymin - pad
        tile: Image.Image | None = None
        dst_quad: list[tuple[float, float]] | None = None
        line = lines[index] if index < len(lines) else None  # extra planes: erase only
        if line:
            # Draw the line at its natural width, then squeeze in x by ``condense`` — this
            # keeps the source height (hierarchy) while the line fits the original width.
            text_h = max(1, int(ascent + descent))
            text_w_nat = max(1, int(font.getlength(line)))
            text_img = Image.new("RGBA", (text_w_nat, text_h), (0, 0, 0, 0))
            ImageDraw.Draw(text_img).text((0, 0), line, font=font, fill=fg + (255,))
            text_w = max(1, int(round(text_w_nat * condense)))
            if text_w != text_w_nat:
                text_img = text_img.resize((text_w, text_h), Image.LANCZOS)
            bullet_y = plane.get("bullet_y")
            if bullet_y is not None:  # centre the text's ink on the preserved bullet glyph
                rows = np.where((np.asarray(text_img)[:, :, 3] > 0).any(axis=1))[0]
                if len(rows):
                    oy = bullet_y - pad - (int(rows[0]) + int(rows[-1])) / 2.0
            tile_w = max(1, text_w + 2 * int(pad))
            tile_h = max(1, text_h + 2 * int(pad))
            ox = (xmin + xmax) / 2 - tile_w / 2 if centered else xmin - pad
            tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
            tile.paste(text_img, (int(pad), int(pad)))
            dst_quad = [
                geo.to_image(ox, oy, x_axis, y_axis),
                geo.to_image(ox + tile_w, oy, x_axis, y_axis),
                geo.to_image(ox + tile_w, oy + tile_h, x_axis, y_axis),
                geo.to_image(ox, oy + tile_h, x_axis, y_axis),
            ]
        # Erase each original word with its OWN tight quad (at the word's own tilt), grown by
        # ``pad`` on the sides and only an AA margin top/bottom. One line-spanning rectangle would
        # float above the lower words of a fanning line and, with a flat fill, paint background
        # colour past the text's band; per-word quads hug the ink and stay inside it.
        erase_quads = [
            _member_erase_quad(quad, pad, min(pad, _ERASE_MARGIN)) for quad in plane["quads"]
        ]
        # A detected bullet glyph sits LEFT of the inset text start (``xmin`` here). OCR often pulls
        # it into the first cell's box, so the per-word erase — grown by ``pad`` — would wipe it. Clip
        # the erase to start AT the text so the glyph survives. (The old single-rectangle erase started
        # at the inset frame and skipped it for free; per-word quads need this clip back.)
        if plane.get("bullet_y") is not None:
            clip_x = int(round(xmin))
            erase_quads = [[(max(x, clip_x), y) for x, y in quad] for quad in erase_quads]
        jobs.append(_Job(erase_quads=erase_quads, bg_color=bg, tile=tile, dst_quad=dst_quad))
    return jobs


def _bullet_geometry(base: Image.Image, frame: tuple, angle: float) -> tuple[float, float] | None:
    """For a bullet line, return (text_start_x, bullet_y_center) — where the text starts (past
    the leading bullet glyph and its gap) and the bullet glyph's vertical centre. None when no
    clear glyph+gap is found (or the line is tilted, where the axis-aligned scan is unreliable).
    Scans the line's vertical band from a margin LEFT of the plane edge, because the OCR cell
    box's left wanders relative to the fixed bullet (sometimes landing right of it). The original
    bullet stays in the image; the caller starts the erase/anchor at the text and centres the
    re-rendered text on the bullet. Triggered only when the VLM flagged the unit as a bullet
    item, so a stray short first word can't be mistaken for a bullet."""
    if abs(angle) > _ANGLE_DEADZONE_DEG:
        return None
    _, _, xmin, xmax, ymin, ymax = frame
    line_h = max(1, int(round(ymax - ymin)))
    x0 = max(0, int(round(xmin - 1.5 * line_h)))           # the bullet may sit left of the box
    x1 = int(round(xmin + 0.6 * (xmax - xmin)))
    y0, y1 = int(round(ymin)), int(round(ymax))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    arr = np.asarray(base.crop((x0, y0, x1, y1)).convert("L")).astype(int)
    bg = int(np.median(arr))
    mask = np.abs(arr - bg) > 60
    ink = mask.any(axis=0)                                 # columns holding a high-contrast pixel
    runs = _ink_runs(ink)
    # Find the bullet: the first SMALL (dot-sized) run that is followed by a clear gap and then
    # the text. Skipping wider runs avoids mistaking adjacent layout ink (a coloured panel/book
    # edge next to the column) for the bullet; the VLM flag guarantees a real bullet is present.
    min_width = max(2.0, 0.06 * line_h)  # a 1px anti-alias speck is not a bullet
    for i in range(len(runs) - 1):
        width = runs[i][1] - runs[i][0] + 1
        gap = runs[i + 1][0] - runs[i][1] - 1
        if min_width <= width <= 0.4 * line_h and gap >= 0.12 * line_h:
            rows = np.where(mask[:, runs[i][0]:runs[i][1] + 1].any(axis=1))[0]
            bullet_y = y0 + (rows.min() + rows.max()) / 2.0 if len(rows) else (y0 + y1) / 2.0
            return float(x0 + runs[i + 1][0]), float(bullet_y)
    return None


def _ink_runs(mask) -> list[tuple[int, int]]:
    """Contiguous (start, end) column ranges where ``mask`` is True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x, value in enumerate(mask):
        if value and start is None:
            start = x
        elif not value and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs

def _group_size(planes: list[dict[str, Any]], mode: str) -> int:
    """The group's ONE render size, chosen from its per-line targets. ``min`` (the default)
    never draws taller than the smallest measured line — but one under-measured line (ink
    without ascenders reads ~70% of cap height) drags the whole block down. ``median`` is the
    better estimator of the element's single true size, at the cost that a genuinely smaller
    line the VLM mixed into the element renders over its own band. Selectable per request
    (``render_size_mode``) to compare; an unknown value falls back to ``min``. A future smarter
    selection policy slots in here as another mode."""
    targets = [plane["target"] for plane in planes]
    if mode == "median":
        return int(median(targets))
    return min(targets)


def _baseline_angle(clusters: list[list], fallback: float) -> float:
    """The block's text-line direction, fit through the word CENTRES rather than read off the
    OCR quad edges (which jitter several degrees per word and bias the median shallow). Each line's
    words are de-meaned vertically so the parallel lines of a block all contribute to ONE shared
    slope — keeping the lines parallel while using every word for a robust fit. Falls back to
    ``fallback`` when too few words span an x-range to define a slope (a one-word line, or a
    vertical stack of single words at the same x)."""
    xs: list[float] = []
    ys: list[float] = []
    for cluster in clusters:
        centres = [(sum(p[0] for p in q) / 4.0, sum(p[1] for p in q) / 4.0) for q in cluster]
        if len(centres) < 2:
            continue
        # De-mean BOTH axes per cluster: with only y de-meaned, clusters of different x-extents
        # (a long line above a short last line) share one forced intercept, which drags the
        # fitted slope toward 0 — the very shallow bias this fit exists to remove. Centred per
        # cluster, parallel lines fit their true slope exactly.
        mean_x = sum(c[0] for c in centres) / len(centres)
        mean_y = sum(c[1] for c in centres) / len(centres)
        for cx, cy in centres:
            xs.append(cx - mean_x)
            ys.append(cy - mean_y)
    if len(xs) < 2 or (max(xs) - min(xs)) < 1.0:
        return fallback
    slope = float(np.polyfit(xs, ys, 1)[0])
    return math.degrees(math.atan(slope))


def _line_clusters(quads: list, angle: float) -> list[list]:
    """Cluster member quads into physical text lines (top to bottom in the oriented
    frame): a quad whose vertical centre falls inside the running cluster's extent is
    on the same line; line pitch puts the next line's centre below it."""
    measured = []
    for quad in quads:
        _, _, _, _, oymin, oymax = geo.oriented_frame([quad], angle)
        measured.append((oymin, oymax, quad))
    measured.sort(key=lambda item: (item[0] + item[1]) / 2)
    clusters: list[list] = []
    extent_max = float("-inf")
    for oymin, oymax, quad in measured:
        center = (oymin + oymax) / 2
        if clusters and center <= extent_max:
            clusters[-1].append(quad)
            extent_max = max(extent_max, oymax)
        else:
            clusters.append([quad])
            extent_max = oymax
    return clusters


def _member_erase_quad(quad: list, dx: float, dy: float) -> list[tuple[int, int]]:
    """A member's tight erase quad: its oriented bounding box in the member's OWN frame, grown
    ``dx`` horizontally (to swallow the glyph anti-alias halo) and only ``dy`` vertically (an AA
    margin, so the fill stays off a coloured band a few px above/below). Per-word — at the word's
    own tilt — so a fanning line's words each get hugged instead of one rectangle overshooting."""
    angle = geo.angle_deg(quad)
    x_axis, y_axis, xmin, xmax, ymin, ymax = geo.oriented_frame([quad], angle)
    xmin, xmax, ymin, ymax = xmin - dx, xmax + dx, ymin - dy, ymax + dy
    return [
        _ipoint(geo.to_image(xmin, ymin, x_axis, y_axis)),
        _ipoint(geo.to_image(xmax, ymin, x_axis, y_axis)),
        _ipoint(geo.to_image(xmax, ymax, x_axis, y_axis)),
        _ipoint(geo.to_image(xmin, ymax, x_axis, y_axis)),
    ]


def _plane_corners(plane: dict[str, Any]) -> list[tuple[float, float]]:
    """The plane's oriented bounding box as four image-space corners [TL, TR, BR, BL].
    Sampling background from this (deskewed) region instead of the axis-aligned bbox keeps
    the border ring inside a tilted coloured band — on a slanted line the axis box's corners
    reach into the surroundings (a sign's panel behind a diagonal bar) and muddy the sample."""
    x_axis, y_axis, xmin, xmax, ymin, ymax = plane["frame"]
    return [
        geo.to_image(xmin, ymin, x_axis, y_axis),
        geo.to_image(xmax, ymin, x_axis, y_axis),
        geo.to_image(xmax, ymax, x_axis, y_axis),
        geo.to_image(xmin, ymax, x_axis, y_axis),
    ]


def _plane_source_tokens(quads: list, group_quads: list, quad_tokens: list[set[str]]) -> set[str]:
    """The source words carried by a line cluster's member quads (matched by identity)."""
    tokens: set[str] = set()
    for quad in quads:
        for other, member_tokens in zip(group_quads, quad_tokens):
            if other is quad:
                tokens |= member_tokens
                break
    return tokens


def _drop_redundant_stray_planes(planes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the TOP line plane when it is an OCR stray rather than a real first line: every source
    word it carries already appears on a line below it AND it is far narrower than those. OCR
    sometimes pulls a neighbouring element's word — a heading word shared with the body below it —
    onto its own tiny line above the real text; kept, that sliver plane starves the unit's width fit
    so the translation no longer meets the condense floor and the whole unit renders nothing,
    leaving the original showing. Only the topmost line is tested: text never wraps ABOVE its first
    line, so a fully-redundant narrow line there is a stray from the element above — whereas the same
    shape at the BOTTOM is a legitimate last wrapped line (a sentence ending on a repeated word)."""
    if len(planes) < 2:
        return planes
    top, below = planes[0], planes[1:]
    below_tokens = set().union(*(plane["tokens"] for plane in below))
    redundant = bool(top["tokens"]) and top["tokens"] <= below_tokens
    narrow = top["width"] < _STRAY_LINE_WIDTH_RATIO * median(plane["width"] for plane in below)
    return below if (redundant and narrow) else planes


def _fit_group(
    text: str,
    *,
    size: int,
    plane_widths: list[float],
    family: str | None = None,
    weight: int | None = None,
) -> tuple[Any, list[str]]:
    """Render at the source ``size`` (true line height) in the unit's VLM font ``family`` /
    ``weight``, wrapped so each line fits the width of the plane it lands on (``plane_widths``).
    The font is NOT reduced to fit width — the source size, and thus the header/body hierarchy,
    is preserved; width is matched by horizontal condensation in the caller."""
    font = load_font(max(6, min(int(size), 160)), text, family=family, weight=weight)
    return font, _wrap_to_planes(font, text, plane_widths)


def _raw_condense(font: Any, lines: list[str], planes: list[dict[str, Any]]) -> float:
    """Unclamped horizontal scale needed to bring every line within its plane width + slack.

    Per line: ``plane width * _WIDTH_SLACK / natural rendered width``; the group takes the
    tightest (smallest) line factor. ``>= 1.0`` means the lines already fit within the slack;
    below ``_CONDENSE_FLOOR`` means even maximum condensation leaves a line more than the slack
    too wide (the caller then reduces the pt size)."""
    factors: list[float] = []
    for index, line in enumerate(lines):
        if index >= len(planes) or not line:
            continue
        natural = font.getlength(line)
        if natural > 0:
            factors.append(planes[index]["width"] * _WIDTH_SLACK / natural)
    return min(factors) if factors else 1.0


def _condense_scale(font: Any, lines: list[str], planes: list[dict[str, Any]]) -> float:
    """Horizontal scale that squeezes the group's lines into their original widths (plus the
    width slack), clamped to [``_CONDENSE_FLOOR``, 1.0] — never stretch a short line, never
    squeeze past the floor (the pt size is reduced upstream instead)."""
    return max(_CONDENSE_FLOOR, min(1.0, _raw_condense(font, lines, planes)))


def _wrap_to_planes(font: Any, text: str, plane_widths: list[float]) -> list[str]:
    """Wrap so each rendered line fits the width of the PLANE it lands on, in order, and the words
    are BALANCED across those lines — not greedily dumped.

    Two failures this avoids:
    - Wrapping every line to the widest plane overflows a narrow top plane (a short heading line
      above a wide one), and the caller's width-fit then shrinks the whole block toward one tiny
      line. So each line is bounded by its OWN plane width.
    - A plain greedy fill breaks an early line at its natural plane width, leaving it half-empty
      while the remainder (often a long token like a URL) piles onto the last line. So instead the
      block's natural line COUNT is taken first (greedy at the plane widths, capped at the plane
      count), then the words are spread over exactly that many lines by the smallest uniform scale
      on the plane widths that still fits — a minimax fill that keeps every line about equally full.

    A compact translation that needs fewer lines than there are planes uses fewer (the rest stay
    erase-only). On equal-width planes a balanced fill matches the original column layout."""
    content = str(text or "").strip()
    if len(plane_widths) <= 1 or not content:
        return [content]
    # Atomic wrap units, not ``.split()``: Han/Kana/CJK-symbol scripts have no spaces, so a whole
    # CJK line is one "word" and never wraps — it stays on one line and the caller condenses it to a
    # sliver. ``break_pieces`` breaks CJK per character (with kinsoku) and keeps Latin/Hangul/digits
    # as whitespace words, so each piece carries the ``glue`` to re-insert when it is not a line start.
    pieces = break_pieces(content)
    line_count = len(_greedy_wrap(font, pieces, plane_widths))  # fewest lines at natural plane width
    caps = plane_widths[:line_count]
    # Smallest scale on the plane widths that still packs the pieces into ``line_count`` lines: this
    # is the most-relaxed (least condensed) balanced fill. Binary search — monotone in the scale.
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _fits_in_lines(font, pieces, caps, mid):
            hi = mid
        else:
            lo = mid
    return _greedy_wrap(font, pieces, [cap * hi for cap in caps])


def _greedy_wrap(font: Any, pieces: list[tuple[str, str]], line_caps: list[float]) -> list[str]:
    """Greedy fill: line ``i`` takes pieces while they fit ``line_caps[i]``; the LAST cap carries
    every remaining piece (so the result never exceeds ``len(line_caps)`` lines). A piece that alone
    exceeds its cap still starts the line (never an empty line) — the caller condenses/shrinks it.
    Each piece carries the ``glue`` (a space for Latin words, empty for CJK chars) re-inserted only
    when it is not at a line start."""
    lines: list[str] = []
    current = ""
    index = 0
    last = len(line_caps) - 1
    for piece, glue in pieces:
        sep = glue if current else ""
        if index >= last:
            current = current + sep + piece
            continue
        trial = current + sep + piece
        if current and font.getlength(trial) > line_caps[index]:
            lines.append(current)
            current = piece
            index += 1
        else:
            current = trial
    lines.append(current)
    return lines


def _fits_in_lines(font: Any, pieces: list[tuple[str, str]], caps: list[float], scale: float) -> bool:
    """Whether ``pieces`` pack into ``len(caps)`` lines with each line within ``caps[i] * scale``
    (a piece wider than its cap alone still starts a line). Used to find the smallest balancing
    scale by binary search."""
    index = 0
    current = ""
    for piece, glue in pieces:
        sep = glue if current else ""
        trial = current + sep + piece
        if current and font.getlength(trial) > caps[index] * scale:
            index += 1
            if index >= len(caps):
                return False
            current = piece
        else:
            current = trial
    return True


def _composite(canvas: np.ndarray, job: _Job) -> None:
    tile = np.asarray(job.tile)
    th, tw = tile.shape[:2]
    dst = np.array(job.dst_quad, dtype=np.float32)
    height, width = canvas.shape[:2]
    x0 = max(0, int(math.floor(dst[:, 0].min())))
    y0 = max(0, int(math.floor(dst[:, 1].min())))
    x1 = min(width, int(math.ceil(dst[:, 0].max())))
    y1 = min(height, int(math.ceil(dst[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return

    src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst - np.array([x0, y0], dtype=np.float32))
    warped = cv2.warpPerspective(
        tile, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)
    )
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (roi * (1.0 - alpha) + warped[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)


def _ipoint(point: tuple[float, float]) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))
