"""The group -> placed-jobs planner: cluster lines into planes, choose size/angle, fit and
condense the text, and build each plane's erase quads and warped tile."""
from __future__ import annotations

from typing import Any
import re
import numpy as np
from statistics import median
from PIL import Image
from PIL import ImageDraw
from app.replacement import geometry as geo
from app.replacement.geometry import _plane_corners
from app.replacement.geometry import _ANGLE_DEADZONE_DEG
from app.replacement.pixels import _INK_DELTA
from app.replacement.jobs import _Job
from app.replacement.layout.compositing import _ipoint
from app.replacement.text.size import _face_ink_per_pt
from app.replacement.text.size import _group_size
from app.replacement.text.size import _quad_band_height
from app.replacement.text.size import _quad_ink_span
from app.replacement.text.angle import _line_clusters
from app.replacement.text.angle import _baseline_angle
from app.replacement.text.wrap import _fit_group
from app.replacement.text.wrap import _raw_condense
from app.replacement.text.wrap import _condense_scale
from app.replacement.text.wrap import _WIDTH_SLACK
from app.replacement.text.wrap import _CONDENSE_FLOOR
from app.replacement.text.fit import draw_text
from app.replacement.text.fit import fold_lone_fullwidth_punctuation
from app.replacement.text.fit import is_cjk_text
from app.replacement.ground.color import _CHROMA_SNAP_SPREAD
from app.replacement.ground.color import sample_oriented_colors
from app.replacement.layout.tables import _split_table_row
from app.replacement.layout.tables import _reproduced_in
from app.replacement.layout.markers import _cell_marker
from app.replacement.layout.markers import _prepend_marker
from app.replacement.layout.markers import _restore_printed_lead_marker
from app.replacement.layout.markers import _strip_leading_glyph
from app.replacement.layout.markers import _strip_unprinted_lead
from app.replacement.layout.markers import _bullet_geometry
from app.replacement.layout.sweep import _sweep_stray_ink
from app.replacement.layout.sweep import _column_mask
from app.replacement.layout.sweep import _ERASE_MARGIN
from app.replacement.layout.sweep import _hint_covers_undetected_text


# Font size from the true (de-skewed) line height. The polygon height spans the full
# glyph extent, a touch taller than the visual cap; scale down slightly to match.
_SIZE_RATIO = 0.9
# CJK glyphs fill the em (ink ~= the full line box), where Latin ink is upper-biased and
# leaves ~30% leading below it. At _SIZE_RATIO the CJK ink overruns the source line pitch and
# consecutive lines touch/overlap, so CJK lines map height->size with a smaller ratio, taking
# roughly the same visual footprint a Latin line would. Hierarchy (relative sizes) is kept.
_CJK_SIZE_RATIO = 0.72
# Floor on the pt-shrink search (matches the plane target floor in _plan_group).
_MIN_RENDER_SIZE = 8
# Relative floor on the same search, for groups whose every plane DECLARES its size
# (text-layer cells): a fit that had to fall below this fraction of the declared size is
# not a rendering but a duplicate caption under leftover ink (a text-layer paragraph whose
# protected lines dropped renders its full translation into the few surviving line planes).
# Below the floor the group keeps the source pixels, the same give-up as the condense-floor
# path. Ink-derived targets are exempt: on the OCR path deep shrink is sometimes the
# accepted rendering (small rotated label work) — extending the floor there is a separate
# decision against those baselines.
_SQUEEZE_PRESERVE_FLOOR = 0.55
# A line plane narrower than this fraction of the unit's other lines, AND carrying only words
# already present on those lines, is an OCR stray (a neighbouring element's word pulled in) — not
# a real wrapped line. Kept, it forms a sliver plane that starves the whole unit's width fit.
_STRAY_LINE_WIDTH_RATIO = 0.5
# Centered lines of one element sit on ONE axis in print, but each rendered line anchors
# on its own plane's centre, and those wobble a few px with OCR quad noise — the block
# comes out ragged. Snap the centres to their median when the spread is noise-sized.
# Measured across the testset's centered multi-line groups the spread is bimodal: quad
# noise stays under ~0.04x the line height, genuinely offset designs start at ~0.18x —
# the gate sits in the gap. (The ratio is against the plane target = 0.9x line height.)
_CENTER_SNAP_MAX_RATIO = 0.12
# "extend" width fit: a line may widen into clean background to its right by at most its
# own original width (so a short item on an empty page cannot balloon into a banner). The
# pixel gates, not this cap, are the safety: extension stops at the first ink, protected
# cell or surface change. Near the image border a typographic margin of one line height
# (~1 em, scales with the text) stays clear — on a clean page nothing else stops the
# growth and text running to the very edge of the document reads as a layout error.
_EXTEND_MAX_RATIO = 1.0
# Per-channel tolerance for snapping a group's per-plane background samples to one
# colour: within it the planes are one surface sampled with texture noise; beyond it
# they are genuinely different (a gradient, two panels) and stay per-plane.
_BG_SNAP_DELTA = 24
# Justified source text (a LaTeX paper, a report column) is detected from the SOURCE line
# geometry: with this many planes or more, left edges of all planes AND right edges of all
# but the last stay within half a line height of their median (measured on docpack papers:
# 2-22px deviation on ~1450px lines; ragged body text scatters by whole words). The render
# then re-justifies: each line's leftover is spread over its word gaps.
_JUSTIFY_MIN_LINES = 4
_JUSTIFY_EDGE_RATIO = 0.5   # max edge deviation as a fraction of the median line height
# A justified paragraph may INDENT its first line (LaTeX abstract: +43px on 27px lines) —
# a rightward shift up to this many line heights is exempt from the left-edge test; the
# indented line's RIGHT edge still has to be flush like every other.
_JUSTIFY_INDENT_MAX = 2.5
# A small minority of body lines may miss the strict tolerance by up to 2x (a hyphenated
# line end whose "-" the OCR box clips measured -22px); more than ~a fifth of them, or any
# line beyond 2x, is genuinely ragged text.
_JUSTIFY_SOFT_FRACTION = 5  # divisor: allowed soft outliers = max(1, len(planes) // 5)
# Block consistency: justify only when the block's unattainable lines stay a small
# minority — count-based (one line is ALWAYS tolerated, then ~a fifth, sharing
# _JUSTIFY_SOFT_FRACTION): a German translation (long compounds, large leftovers) has MOST
# lines over the stretch cap and falls back ragged as a WHOLE — stretching a feasible
# minority yields gaps without an achieved flush margin, worse than consistent ragged.
# A too-LONG line inside a justified block (the wrap may pack up to the 4% width slack)
# would poke OUT of the flush margin; it is squeezed onto the margin instead (and counts as
# feasible — it ends flush), down to this extra per-line condensation floor. Beyond the
# floor the overhang stands (pathological).
_JUSTIFY_SQUEEZE_FLOOR = 0.94


def _justify_overhang_width(natural_w: float, condense: float, plane_width: float) -> int | None:
    """Rendered width for a justified block's too-long fallback line: capped at the plane
    width when the extra squeeze stays above ``_JUSTIFY_SQUEEZE_FLOOR``, else ``None`` (the
    line keeps its natural condensed width and overhangs — better than illegibly narrow)."""
    condensed = natural_w * condense
    if condensed <= plane_width:
        return None
    if plane_width / condensed < _JUSTIFY_SQUEEZE_FLOOR:
        return None
    return max(1, int(round(plane_width)))
# Per-gap stretch/shrink bounds, as fractions of the natural space width. Outside them the
# line falls back to ragged left — a near-empty last-but-one line must not become a river.
_JUSTIFY_MAX_EXTRA = 1.5
_JUSTIFY_MAX_SHRINK = 0.25
# Rendered lines anchor on their plane's measured TOP; OCR quads wobble a few px, and one
# sparse tall glyph (an emoji, a math bracket) inflates a single line's top by a third of a
# line height — on a uniformly-leaded source paragraph the render then shows visibly uneven
# leading, down to near-collisions. When the source lines demonstrably sit on one uniform
# grid, each plane's top snaps onto that grid. The grid is a TRIMMED least-squares fit of
# index -> top: fit, and while the worst top misses the grid by more than noise, retire it
# (within a small budget, shared with _JUSTIFY_SOFT_FRACTION, granted from 6 lines up) and
# refit. A retired top must then be EXPLAINED by its own line's glyph extent: a sparse tall
# glyph pulls a top UP by that line's height surplus over the group's median — residual
# ~= -(height - median height) — and only that. Real structure has no such alibi and the
# group is left untouched WHOLE (no half-snapped blocks): a paragraph gap displaces a line
# withOUT extra height (a leaflet's two stacked paragraphs in one group — numerically the
# closest false-friend: residual -8.4 vs a 3px height surplus), an extra OCR plane shifts
# every later index and never converges, a ToC's designed gaps scatter throughout. Noise
# tolerance measured across the document-testset body paragraphs: quad wobble <= ~0.13x
# line height.
_PITCH_SNAP_MIN_LINES = 3
_PITCH_SNAP_NOISE_RATIO = 0.2  # top-residual tolerance, x line height


def _snap_line_pitch(planes: list[dict[str, Any]]) -> None:
    """Snap plane tops onto the group's uniform line grid (see the _PITCH_SNAP_* rationale).
    Mutates only each frame's ymin — the erase quads keep hugging the true ink; only the
    rendered line's anchor moves. No-op whenever the grid evidence fails."""
    if len(planes) < _PITCH_SNAP_MIN_LINES:
        return
    tops = [plane["frame"][4] for plane in planes]
    if any(below <= above for above, below in zip(tops, tops[1:])):
        return  # not a top-to-bottom stack of lines (side-by-side clusters)
    line_h = median(plane["true_height"] for plane in planes)
    tolerance = _PITCH_SNAP_NOISE_RATIO * line_h
    budget = len(planes) // _JUSTIFY_SOFT_FRACTION if len(planes) >= 6 else 0
    active = dict(enumerate(tops))
    trimmed: list[int] = []
    for _ in range(budget + 1):
        if len(active) < _PITCH_SNAP_MIN_LINES:
            return
        pitch, anchor = _ls_line(active)
        if pitch < 0.5 * line_h:
            return  # tighter than stacked text lines can sit
        worst_index = max(active, key=lambda i: abs(active[i] - (anchor + pitch * i)))
        if abs(active[worst_index] - (anchor + pitch * worst_index)) > tolerance:
            del active[worst_index]  # retire the worst top and refit
            trimmed.append(worst_index)
            continue
        for index in trimmed:  # every retired top needs the glyph-extent alibi
            residual = tops[index] - (anchor + pitch * index)
            surplus = planes[index]["true_height"] - line_h
            if residual > tolerance or abs(residual + surplus) > tolerance:
                return
        for index, plane in enumerate(planes):
            x_axis, y_axis, xmin, xmax, _ymin, ymax = plane["frame"]
            plane["frame"] = (x_axis, y_axis, xmin, xmax, anchor + pitch * index, ymax)
        return


def _ls_line(points: dict[int, float]) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)`` through ``{index: value}``."""
    count = len(points)
    mean_i = sum(points) / count
    mean_v = sum(points.values()) / count
    denominator = sum((i - mean_i) ** 2 for i in points)
    slope = sum((i - mean_i) * (v - mean_v) for i, v in points.items()) / denominator
    return slope, mean_v - slope * mean_i


def _planes_justified(planes: list[dict[str, Any]]) -> bool:
    """Whether the group's SOURCE lines were typeset justified (both edges flush, last line
    exempt on the right, first line may be indented) — the evidence that re-justifying the
    translation is faithful. Flush BOTH edges is geometrically incompatible with a centered
    ragged block, so this evidence may overrule a VLM "center" label at the call site."""
    if len(planes) < _JUSTIFY_MIN_LINES:
        return False
    frames = [plane["frame"] for plane in planes]
    line_h = median(frame[5] - frame[4] for frame in frames)
    tolerance = _JUSTIFY_EDGE_RATIO * max(1.0, line_h)
    lefts = [frame[2] for frame in frames]
    rights = [frame[3] for frame in frames[:-1]]
    left_devs = [left - median(lefts) for left in lefts]
    right_devs = [right - median(rights) for right in rights]
    # First-line paragraph indent: a bounded RIGHTWARD shift of line 0 only.
    if tolerance < left_devs[0] <= _JUSTIFY_INDENT_MAX * line_h:
        left_devs = left_devs[1:]
    # Fewer lines = less evidence = a stricter bar: a short ragged paragraph whose 3 line
    # ends happen to fall within ~20px (a letter's closing paragraph) must not qualify, so
    # soft outliers are only granted from 6 lines up.
    soft_budget = max(1, len(planes) // _JUSTIFY_SOFT_FRACTION) if len(planes) >= 6 else 0
    soft = 0
    for dev in (*left_devs, *right_devs):
        if abs(dev) <= tolerance:
            continue
        if abs(dev) > 2.0 * tolerance:
            return False
        soft += 1
    return soft <= soft_budget


def _justify_feasible(font: Any, line: str, target_w: float) -> bool:
    """Whether ``line`` can end flush at ``target_w``: within the per-gap stretch/shrink
    bounds, or too long but within the overhang-squeeze floor (it then renders condensed
    exactly onto the margin — flush all the same)."""
    words = [w for w in line.split(" ") if w]
    if len(words) < 2:
        return False
    natural = font.getlength(line)
    space_w = max(1.0, font.getlength(" "))
    per_gap = (target_w - natural) / (len(words) - 1)
    if per_gap > _JUSTIFY_MAX_EXTRA * space_w:
        return False
    if per_gap >= -_JUSTIFY_MAX_SHRINK * space_w:
        return True
    return natural > 0 and target_w / natural >= _JUSTIFY_SQUEEZE_FLOOR


def _justified_text_image(
    font: Any, line: str, target_w: float, fg: tuple, text_h: int
) -> Image.Image | None:
    """``line`` drawn word-by-word with its leftover spread over the word gaps so the ink
    spans exactly ``target_w`` — or ``None`` when justification is not applicable (a single
    word, a spaceless CJK line) or would exceed the per-gap stretch/shrink bounds (the
    caller then draws the plain ragged line)."""
    words = [w for w in line.split(" ") if w]
    if len(words) < 2:
        return None
    natural = font.getlength(line)
    space_w = max(1.0, font.getlength(" "))
    per_gap = (target_w - natural) / (len(words) - 1)
    if per_gap > _JUSTIFY_MAX_EXTRA * space_w or per_gap < -_JUSTIFY_MAX_SHRINK * space_w:
        return None
    image = Image.new("RGBA", (max(1, int(round(target_w))), text_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x = 0.0
    for word in words[:-1]:
        draw_text(draw, (x, 0), word, font, fg + (255,))
        x += font.getlength(word) + space_w + per_gap
    # anchor the last word's right edge on the target so rounding drift cannot fray the margin
    draw_text(draw, (target_w - font.getlength(words[-1]), 0), words[-1], font, fg + (255,))
    return image

def _translation_preserves_source(translated: str, source: str) -> bool:
    """Whether the "translation" is the source text unchanged — names, emails, URLs, an
    untranslated affiliation line. Words (letter/digit runs) must match exactly and in
    order; symbol and punctuation differences are ignored: OCR reads an affiliation's
    marker symbols as noise or not at all, while the VLM transcribes varying lookalikes
    (⋆/♠/✦ swap run to run) — symbol inequality is no evidence that the WORDS changed.
    A line without any letter (a bare price/number, whose decimal separator may localize)
    requires exact equality instead of word equality."""
    src = " ".join(str(source or "").split())
    tr = " ".join(str(translated or "").split())
    if not src:
        return False
    if tr == src:
        return True
    src_words = re.findall(r"[^\W_]+", src)
    tr_words = re.findall(r"[^\W_]+", tr)
    if not src_words or src_words != tr_words:
        return False
    return any(ch.isalpha() for word in src_words for ch in word)


def _plan_group(
    base: Image.Image,
    units: list[dict[str, Any]],
    *,
    snap_horizontal: bool = False,
    render_size_mode: str = "median",
    width_fit_mode: str = "footprint",
    band_ratio: float | None = None,
    ink_fill: bool = False,
    angle_field: tuple[float, float] | None = None,
    size_cohorts: dict[int, float] | None = None,
    base_arr: np.ndarray | None = None,
    protected_boxes: list[dict[str, Any]] | None = None,
    sweep_ok: bool | None = None,
    document_member_texts: list[tuple[Any, str]] | None = None,
    preserve_unchanged_text: bool = False,
) -> list[_Job]:
    # The read-only pixel view is materialised ONCE per render (in render_translated_image)
    # and threaded in: np.asarray(base) copies the whole frame, and _plan_group runs per
    # group — re-converting a large image dozens of times dominated the render.
    if base_arr is None:
        base_arr = np.asarray(base)
    # Decide BEFORE a table split whether this group may sweep stray ink: the split nulls the
    # field pairs on its cells, and the pairs' source texts are the hint-side evidence.
    if sweep_ok is None:
        sweep_ok = _hint_covers_undetected_text(units)
    if len(units) == 1:
        other_texts = [
            text for unit_id, text in (document_member_texts or [])
            if unit_id != units[0].get("id")
        ]
        cells = _split_table_row(units[0], other_texts)
        if cells is not None:
            return [
                job
                for cell in cells
                for job in _plan_group(
                    base,
                    [cell],
                    snap_horizontal=snap_horizontal,
                    render_size_mode=render_size_mode,
                    width_fit_mode=width_fit_mode,
                    band_ratio=band_ratio,
                    ink_fill=ink_fill,
                    angle_field=angle_field,
                    size_cohorts=size_cohorts,
                    base_arr=base_arr,
                    protected_boxes=protected_boxes,
                    sweep_ok=sweep_ok,
                    document_member_texts=document_member_texts,
                    preserve_unchanged_text=preserve_unchanged_text,
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
    own_boxes: list[dict[str, Any]] = []  # members this group ERASES — sweep-eligible ground
    exact_quad_sizes: dict[int, float] = {}  # id(quad) -> the member's declared em size in px
    island_erase_quads: dict[int, list] = {}  # id(quad) -> the member's island ink boxes
    for unit in units:
        translated = fold_lone_fullwidth_punctuation(str(unit.get("translated_text") or "").strip())
        translated = _strip_unprinted_lead(translated, unit)
        # Empty or an OCR-noise single char -> leave the original alone. A single CJK character
        # is a full word ("PUSH" -> "推"), not noise, and must render.
        if not translated or (len(translated) == 1 and not is_cjk_text(translated)):
            continue
        # Under the preserve_unchanged_text flag (the translation layer's whitespace/case
        # check, extended here with word-level symbol tolerance): untranslated content
        # (names, emails, an affiliation line whose marker symbols OCR and VLM read
        # differently) keeps its original print — skip the unit whole (no erase, no draw).
        # Re-typesetting identical text can only degrade it: approximate font and weight,
        # lost superscripts/small caps, erase residue from marks outside the OCR quads.
        if preserve_unchanged_text and _translation_preserves_source(
            translated, str(unit.get("source_text") or "")
        ):
            continue
        # Footnote-style lead marker the parse/translation lost ("*Equal contributions." ->
        # "Gelijke bijdragen.") — re-add the PRINTED marker; the bullet paths above handle
        # their own glyphs and stay out of this.
        if not cell_marker and not loose_glyph:
            translated = _restore_printed_lead_marker(translated, unit)
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
            own_boxes.append(dict(member["bbox"]))
            if member.get("size_px") is not None:
                exact_quad_sizes[id(quad)] = float(member["size_px"])
            for island in member.get("islands") or []:
                b = island.get("bbox") or {}
                left, top = float(b.get("left") or 0.0), float(b.get("top") or 0.0)
                w, h = float(b.get("width") or 0.0), float(b.get("height") or 0.0)
                island_erase_quads.setdefault(id(quad), []).append(
                    [(int(left), int(top)), (int(left + w), int(top)),
                     (int(left + w), int(top + h)), (int(left), int(top + h))]
                )
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
    # perspective gradient — a tall photographed sign runs ~1° at the top to ~8° at the
    # bottom), so its
    # small top-line angles are kept; snapping only those would break the gradient.
    angle = median(geo.angle_deg(quad) for quad in group_quads)
    if angle_field is not None:
        # A tilted image carries ONE smooth perspective gradient (see the _FIELD_*
        # constants): every group reads its angle from the document field at its own y,
        # instead of trusting its own noisy fit. One angle per group — the intra-group
        # fan (field slope x group height) stays under ~1deg on the measured signs.
        slope, intercept = angle_field
        y_centre = median(sum(p[1] for p in quad) / 4.0 for quad in group_quads)
        angle = slope * y_centre + intercept
        clusters = _line_clusters(group_quads, angle)
    else:
        clusters = _line_clusters(group_quads, angle)
        # The rendered text is warped to this angle, so it must match the band the words actually
        # sit on. Per-quad edge angles are noisy (a short word's OCR quad reads several degrees
        # off), and their median comes out biased shallow — the rendered line then drifts off a
        # tilted band. The baseline FIT through the word centres recovers the true line direction;
        # the parallel lines of a block share it, so it keeps them parallel. Falls back to the
        # quad-median when too few words carry a baseline (a one-word line, a vertical stack of
        # single words).
        angle = _baseline_angle(clusters, angle)
        if snap_horizontal and abs(angle) < _ANGLE_DEADZONE_DEG:
            angle = 0.0
    size_ratio = _CJK_SIZE_RATIO if any(is_cjk_text(text) for text in texts) else _SIZE_RATIO
    planes: list[dict[str, Any]] = []
    for quads in clusters:
        true_height = median(geo.line_height(quad) for quad in quads)
        x_axis, y_axis, xmin, xmax, ymin, ymax = geo.oriented_frame(quads, angle)
        plane = {
            "quads": quads,
            "tokens": _plane_source_tokens(quads, group_quads, quad_tokens),
            "target": max(8, int(true_height * size_ratio)),
            "true_height": true_height,
            "pad": max(2.0, true_height / 6.0),
            "frame": (x_axis, y_axis, xmin, xmax, ymin, ymax),
            "width": xmax - xmin,
        }
        # A line whose every member declares its em size (a text layer) renders at
        # exactly that size — the source's own typesetting, no ink estimation. The
        # ink-derived refinements below (band, cohort, fill) skip these planes:
        # each exists to correct a measurement this line does not need.
        declared = [exact_quad_sizes.get(id(quad)) for quad in quads]
        if declared and all(size is not None for size in declared):
            plane["target"] = max(8, int(round(median(declared))))
            plane["exact_size"] = True
        planes.append(plane)
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

    centered = any(str(unit.get("alignment") or "") == "center" for unit in units)
    # Justified source overrules a "center" label: flush both edges cannot be a centered
    # ragged block — the VLM calls an indented justified abstract "center". The group then
    # anchors left like its source (its last line measured left-anchored, not centered).
    justified = angle == 0.0 and _planes_justified(planes)
    if justified:
        centered = False
    if centered and len(planes) > 1:
        centers = [(plane["frame"][2] + plane["frame"][3]) / 2 for plane in planes]
        axis = median(centers)
        if max(abs(c - axis) for c in centers) <= _CENTER_SNAP_MAX_RATIO * median(
            plane["target"] for plane in planes
        ):
            for plane in planes:
                plane["center_x"] = axis

    # One element usually sits on one surface: when the per-plane background samples
    # are near-equal (texture noise), snap them to their median so the erase planes
    # don't show slightly different shades per line.
    colors = [sample_oriented_colors(base, _plane_corners(plane)) for plane in planes]
    if len(colors) > 1:
        median_bg = tuple(int(median(bg[channel] for bg, _ in colors)) for channel in range(3))
        if all(max(abs(bg[c] - median_bg[c]) for c in range(3)) <= _BG_SNAP_DELTA for bg, _ in colors):
            # Snap the fg too when the per-plane ink samples agree within tolerance (measurement
            # noise on one ink), to their median — the fg is ink-evidence (not derived from bg),
            # so recomputing it from the snapped bg would throw that evidence away. A genuinely
            # different-coloured line (an accent line inside the element) keeps its own ink.
            fgs = [fg for _, fg in colors]
            median_fg = tuple(int(median(fg[channel] for fg in fgs)) for channel in range(3))
            if all(max(abs(fg[c] - median_fg[c]) for c in range(3)) <= _BG_SNAP_DELTA for fg in fgs):
                colors = [(median_bg, median_fg)] * len(colors)
            else:
                colors = [(median_bg, fg) for fg in fgs]

    # Inline-accent colour cannot be re-placed: the fg is sampled per SOURCE line, but the
    # translation re-wraps over the planes, so a chromatic span's line index no longer marks
    # the same words — a blue citation run starts a line early or bleeds into the following
    # prose. In body prose a mix of chromatic and achromatic line inks therefore demotes to
    # the achromatic ink (the running text's): colour on the wrong words is worse than a
    # dropped accent. No-ops: an element whose EVERY line is the accent colour (nothing
    # mixed), and title/header/footer levels — a two-tone heading is per-line by design.
    # NAMED LIMIT: faithful inline colour needs span-level colours through translation
    # (the parked accent-spans follow-up); until then prose accents are dropped, not moved.
    if len(colors) > 1 and all(u.get("level") == "body" for u in units):
        fgs = [fg for _, fg in colors]
        achromatic = [fg for fg in fgs if max(fg) - min(fg) <= _CHROMA_SNAP_SPREAD]
        if achromatic and any(max(fg) - min(fg) > _CHROMA_SNAP_SPREAD for fg in fgs):
            base_fg = tuple(int(median(fg[channel] for fg in achromatic)) for channel in range(3))
            colors = [(bg, base_fg) for bg, _ in colors]

    # "band" size metric: clamp each plane's size target to its ink band scaled by the
    # document's own extent/band norm — a line whose polygon is stretched by sparse tall
    # glyphs sinks to the size its band says, everything else stays on the extent path
    # (one-sided: min(), never enlarges; weak ink evidence keeps the extent).
    if band_ratio is not None and angle == 0.0:
        band_px = base_arr
        for plane, (bg, _fg) in zip(planes, colors):
            if plane.get("exact_size"):
                continue
            bands = [b for q in plane["quads"] if (b := _quad_band_height(band_px, q, bg)) is not None]
            if bands:
                clamped = min(plane["true_height"], median(bands) * band_ratio)
                plane["target"] = max(8, int(clamped * size_ratio))

    # The units of a group share one VLM element, so one font family/weight. Take the first
    # that carries a hint (leftovers have none -> fall back to the default font).
    family = next((u.get("font_family") for u in units if u.get("font_family")), None)
    weight = next((u.get("font_weight") for u in units if u.get("font_weight")), None)

    # "cohort" size metric: an element the VLM gave a font-size (pt) that other elements share
    # renders at the cohort's shared OCR-median size, so a list the VLM judged one size renders
    # uniform instead of each item at its own noisy per-line measurement. A short-line item then
    # sizes UP to the cohort and re-wraps over its available planes (keeping the size instead of
    # collapsing to one small line). Only when the cohort passed the agreement gate. The clamp is
    # deliberately two-sided: an element measuring ABOVE the cohort is usually diacritic/glyph
    # inflation (an accented name's quad reads ~25% taller than its unaccented siblings), so a
    # measured-high exemption breaks list uniformity — tried and reverted; a genuinely larger
    # element mislabeled with the cohort's pt (an affiliation line labeled body-pt) stays the
    # named limit, mitigated by the unchanged-text preserve on document categories.
    if size_cohorts and angle == 0.0:
        pt = next((u.get("font_size") for u in units if u.get("font_size") is not None), None)
        cohort_height = size_cohorts.get(int(pt)) if pt is not None else None
        if cohort_height is not None:
            for plane in planes:
                if plane.get("exact_size"):
                    continue
                plane["target"] = max(8, int(cohort_height * size_ratio))

    # "fill" size metric: size each line so its rendered ink is as tall as the SOURCE line's
    # ink, matching the print's glyph fill instead of the 0.9-of-polygon undershoot (see
    # size._face_ink_per_pt). Self-calibrating per element: measure both ink spans in pixels and
    # divide. Runs LAST — it overrides the extent/band/cohort targets, since it is the direct
    # pixel match those approximate. Flat, non-CJK groups only (the scan is axis-aligned; CJK
    # em-fill is handled by the size ratio); weak ink evidence on a plane keeps its prior target.
    if ink_fill and angle == 0.0 and size_ratio == _SIZE_RATIO:
        ink_per_pt = _face_ink_per_pt(family, weight)
        if ink_per_pt > 0:
            for plane, (bg, _fg) in zip(planes, colors):
                if plane.get("exact_size"):
                    continue
                spans = [s for q in plane["quads"] if (s := _quad_ink_span(base_arr, q, bg)) is not None]
                if spans:
                    plane["target"] = max(8, int(round(median(spans) / ink_per_pt)))

    # "extend" width fit: widen each plane's usable width into VERIFIED clean background to
    # its right before fitting, so a longer translation of a short line (a list item) keeps
    # its size instead of condensing/shrinking. Strictly additive: every guard that fails
    # leaves the plane at its footprint width, i.e. exactly the "footprint" behaviour. Only
    # for axis-aligned groups (the scan is axis-aligned — same honest limit as the ink
    # sweep) and never for centered ones (growing right would break the centring).
    if width_fit_mode == "extend" and angle == 0.0 and not centered:
        for plane, (bg, _fg) in zip(planes, colors):
            plane["width"] += _clean_right_extension(base_arr, plane, bg, protected_boxes or [])
    elif angle == 0.0 and not centered:
        # Footprint mode: the 4% width slack (wrap._WIDTH_SLACK) must not carry a line INTO an
        # adjacent panel — clamp it at a WALL: a colour step inside the slack window that stays
        # non-background through the window's end (a newsletter's sidebar panel). Ink that clears
        # again within the window (a neighbour column's glyph, a thin frame line) does not clamp,
        # so signage/receipt behaviour is untouched. Tilted and centered groups keep the plain 4%
        # (the scan is axis-aligned and right-only — same honest limit as the extend fit).
        for plane, (bg, _fg) in zip(planes, colors):
            plane["slack_px"] = _wall_bounded_slack(
                base_arr, plane, bg, plane["width"] * (_WIDTH_SLACK - 1.0))

    # The whole group renders at ONE size = the original's source size (true line height),
    # NOT a size chosen to fit the width. So a heading keeps heading size and body keeps
    # body size — the source size carries the hierarchy. The joined translation is balanced
    # over the original line count.
    joined = " ".join(texts)
    plane_widths = [plane["width"] for plane in planes]
    # Render at the source size, but spend pt only as a last resort: if even at the condense
    # floor a line would still exceed its plane by more than _WIDTH_SLACK, step the size down
    # (which re-wraps) until the floor suffices or the size floor is hit. If the minimum size
    # still cannot fit, leave the original pixels; this catches chatty model replies on tiny
    # OCR-noise cells instead of erasing far beyond the source footprint.
    group_islands = _group_islands(units, base)
    size = _group_size(planes, render_size_mode)
    font, lines = _fit_group(joined, size=size, plane_widths=plane_widths, family=family, weight=weight,
                             islands=group_islands)
    # A table COLUMN cell that cannot fit its per-line footprints at the source size gets ONE
    # retry with every line at the cell's union width before pt is spent: the union is the
    # cell's own evidence (its widest line spans the column), and a short trailing source line
    # (a lone centered word under a full first line) must not drag the whole cell through the
    # pt-shrink loop because a long token cannot fit that sliver. Cells that already fit keep
    # their footprints — this path only opens where the render would otherwise crush. The
    # measured slack is zeroed: it was scanned at the old right edges.
    if (
        len(planes) > 1
        and _raw_condense(font, lines, planes) < _CONDENSE_FLOOR
        and any(unit.get("table_cell") for unit in units)
    ):
        xmin_union = min(plane["frame"][2] for plane in planes)
        xmax_union = max(plane["frame"][3] for plane in planes)
        for plane in planes:
            x_axis, y_axis, _xmin, _xmax, ymin, ymax = plane["frame"]
            plane["frame"] = (x_axis, y_axis, xmin_union, xmax_union, ymin, ymax)
            plane["width"] = xmax_union - xmin_union
            if "slack_px" in plane:
                plane["slack_px"] = 0.0
        plane_widths = [plane["width"] for plane in planes]
        font, lines = _fit_group(joined, size=size, plane_widths=plane_widths, family=family, weight=weight,
                                 islands=group_islands)
    source_size = size
    while size > _MIN_RENDER_SIZE and _raw_condense(font, lines, planes) < _CONDENSE_FLOOR:
        size -= 1
        font, lines = _fit_group(joined, size=size, plane_widths=plane_widths, family=family, weight=weight,
                                 islands=group_islands)
    if _raw_condense(font, lines, planes) < _CONDENSE_FLOOR:
        return []
    if all(plane.get("exact_size") for plane in planes) and size < max(
        _MIN_RENDER_SIZE, int(round(_SQUEEZE_PRESERVE_FLOOR * source_size))
    ):
        return []
    ascent, descent = font.getmetrics()

    # Width is matched by horizontal condensation, not by shrinking the font: at the source
    # size the translated line is usually wider than its original, so squeeze it in x to fit
    # the original line's width (floored at _CONDENSE_FLOOR). One factor for the whole group
    # keeps a multi-line block visually coherent; never stretch (cap at 1.0), so a shorter
    # line just stays narrower.
    condense = _condense_scale(font, lines, planes)

    # Flat, angle-snapped groups get the pixel-evidence cleanups: the descender bottom
    # extension per line, and — gated separately on hint coverage — the stray-ink slot sweep
    # after the jobs are built. Tilted groups skip both: the axis-aligned measurement is
    # unreliable there (honest limit).
    base_np = base_arr if angle == 0.0 else None

    # Justified rendering: every line except the last spreads its leftover over the word
    # gaps (the ``justified`` flag was decided above, before the centered machinery).
    last_text_index = max((i for i, l in enumerate(lines) if l), default=-1)
    if justified and last_text_index > 0:
        # Block consistency: stretching a feasible minority while the rest falls back gives
        # gaps WITHOUT an achieved flush margin — then the whole block renders ragged.
        body = [
            (lines[i], planes[i]) for i in range(min(last_text_index, len(planes))) if lines[i]
        ]
        feasible = sum(
            _justify_feasible(font, line, float(plane["width"]) / max(condense, 0.01))
            for line, plane in body
        )
        # Count-based, not a hard fraction: ONE unattainable line is always tolerated (a
        # 4-line block with one bad line sat at 75% and flipped the whole block ragged on
        # live-run translation wobble), beyond that ~a fifth of the block.
        if body and (len(body) - feasible) > max(1, len(body) // _JUSTIFY_SOFT_FRACTION):
            justified = False

    # After every consumer of the measured frames (colour sampling, slack/extend scans,
    # justify evidence): regularise the RENDER anchors onto the source's uniform line grid.
    _snap_line_pitch(planes)

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
            text_img = None
            if justified and index < last_text_index:
                # Target the plane width BEFORE condensation so the group-wide squeeze lands
                # the right edge exactly on the plane's flush margin.
                text_img = _justified_text_image(
                    font, line, float(plane["width"]) / max(condense, 0.01), fg, text_h
                )
            if text_img is None:
                text_w_nat = max(1, int(font.getlength(line)))
                text_img = Image.new("RGBA", (text_w_nat, text_h), (0, 0, 0, 0))
                draw_text(ImageDraw.Draw(text_img), (0, 0), line, font, fg + (255,))
            text_w_nat = text_img.width
            text_w = max(1, int(round(text_w_nat * condense)))
            if justified and index < last_text_index:
                # A too-long fallback line would poke OUT of the flush margin: squeeze it
                # onto the margin (a few percent extra condensation on this line only).
                overhang = _justify_overhang_width(text_w_nat, condense, float(plane["width"]))
                if overhang is not None:
                    text_w = overhang
            if text_w != text_w_nat:
                text_img = text_img.resize((text_w, text_h), Image.LANCZOS)
            bullet_y = plane.get("bullet_y")
            if bullet_y is not None:  # centre the text's ink on the preserved bullet glyph
                rows = np.where((np.asarray(text_img)[:, :, 3] > 0).any(axis=1))[0]
                if len(rows):
                    oy = bullet_y - pad - (int(rows[0]) + int(rows[-1])) / 2.0
            tile_w = max(1, text_w + 2 * int(pad))
            tile_h = max(1, text_h + 2 * int(pad))
            ox = plane.get("center_x", (xmin + xmax) / 2) - tile_w / 2 if centered else xmin - pad
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
        # Island ink (the formula pixels the transplant redraws) lives partly OUTSIDE the
        # prose-based member quads — a radical reaches above the band. Erase it via the
        # recorded island boxes, grown by the same AA margin.
        margin = int(min(pad, _ERASE_MARGIN))
        for quad in plane["quads"]:
            for box in island_erase_quads.get(id(quad), []):
                erase_quads.append([
                    (box[0][0] - margin, box[0][1] - margin),
                    (box[1][0] + margin, box[1][1] - margin),
                    (box[2][0] + margin, box[2][1] + margin),
                    (box[3][0] - margin, box[3][1] + margin),
                ])
        # A detected bullet glyph sits LEFT of the inset text start (``xmin`` here). OCR often pulls
        # it into the first cell's box, so the per-word erase — grown by ``pad`` — would wipe it. Clip
        # the erase to start AT the text so the glyph survives. (The old single-rectangle erase started
        # at the inset frame and skipped it for free; per-word quads need this clip back.)
        if plane.get("bullet_y") is not None:
            clip_x = int(round(xmin))
            erase_quads = [[(max(x, clip_x), y) for x, y in quad] for quad in erase_quads]
        jobs.append(_Job(erase_quads=erase_quads, bg_color=bg, tile=tile, dst_quad=dst_quad))
    if base_np is not None and sweep_ok:
        _sweep_stray_ink(base_np, planes, jobs, protected_boxes or [], own_boxes)
    return jobs

def _group_islands(units: list[dict[str, Any]], base: Image.Image) -> dict[str, Any]:
    """The group's inline pixel islands (islands design doc, phase 4): per ⟦Mn⟧ id an ink
    MASK cropped from the source raster (before any erase), its source width/height, its
    vertical offset below the line's ink top, and the line's declared em size — everything
    ``IslandFont`` needs to measure and transplant the island inside the re-typeset text.
    The mask (inverted luminance with a paper-noise floor) draws in the line's fill colour,
    so the transplant recolours and condenses exactly like the glyphs around it."""
    islands: dict[str, Any] = {}
    for unit in units:
        for member in unit.get("members") or []:
            declared = member.get("size_px")
            member_top = float((member.get("bbox") or {}).get("top") or 0.0)
            for island in member.get("islands") or []:
                bbox = island.get("bbox") or {}
                left, top = float(bbox.get("left") or 0.0), float(bbox.get("top") or 0.0)
                w, h = float(bbox.get("width") or 0.0), float(bbox.get("height") or 0.0)
                x0 = max(0, int(left)); y0 = max(0, int(top))
                x1 = min(base.width, int(round(left + w))); y1 = min(base.height, int(round(top + h)))
                if x1 - x0 < 1 or y1 - y0 < 1:
                    continue
                crop = base.crop((x0, y0, x1, y1)).convert("L")
                mask = crop.point(lambda v: max(0, min(255, int((215 - v) * 255 / 175))))
                islands[str(island.get("id") or "")] = {
                    "mask": mask,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "dy": top - member_top,
                    "declared": float(declared) if declared else float(y1 - y0),
                }
    return islands


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

def _wall_bounded_slack(
    base_px: np.ndarray,
    plane: dict[str, Any],
    bg: tuple[int, int, int],
    window_px: float,
) -> float:
    """The width-slack window, zeroed when it contains a WALL: a column from which the pixels
    stay non-background through the window's end — a colour step that never returns (an
    adjacent panel edge, or a block the window ends inside). Against a panel the original's own
    margin is the layout's intent, so no slack is spent at all: the line stays within its plane
    width instead of creeping toward the panel. Ink that clears again within the window (a
    neighbour glyph, a thin rule) yields the full window: overrunning ACROSS such marks is the
    long-standing 4% behaviour, and clamping on it would reflow every receipt row that sits
    left of a price column."""
    _x_axis, _y_axis, xmin, xmax, ymin, ymax = plane["frame"]
    height, width = base_px.shape[:2]
    y0, y1 = max(0, int(ymin)), min(height, int(ymax) + 1)
    x0 = int(round(xmax))
    x1 = min(width, x0 + int(round(window_px)) + 1)
    if x0 >= x1 or y0 >= y1:
        return window_px
    strip = base_px[y0:y1, x0:x1].astype(np.int16)
    non_bg = (np.abs(strip - np.asarray(bg, dtype=np.int16)).max(axis=2) >= _INK_DELTA).all(axis=0)
    return 0.0 if len(non_bg) and non_bg[-1] else window_px


def _clean_right_extension(
    base_px: np.ndarray,
    plane: dict[str, Any],
    bg: tuple[int, int, int],
    protected_boxes: list[dict[str, Any]],
) -> float:
    """Extra usable width right of the plane: the run of columns verified to be clean,
    continuous background — no ink against the plane's sampled colour (a colour step, a
    paper/panel edge and any glyph all read as ink), and no protected cell (another unit's
    text, whether or not it renders). Text drawn over the extension is composited onto the
    untouched original pixels; nothing is erased there, so a guard miss can at worst
    overprint — never wipe."""
    _x_axis, _y_axis, xmin, xmax, ymin, ymax = plane["frame"]
    height, width = base_px.shape[:2]
    y0, y1 = max(0, int(ymin)), min(height, int(ymax) + 1)
    x0 = int(round(xmax + plane["pad"]))
    border_margin = int(round(ymax - ymin))  # ~1 em at this line's size
    x1 = min(width - border_margin, x0 + int(round((xmax - xmin) * _EXTEND_MAX_RATIO)))
    if x0 >= x1 or y0 >= y1:
        return 0.0
    strip = base_px[y0:y1, x0:x1].astype(np.int16)
    unclean = (np.abs(strip - np.asarray(bg, dtype=np.int16)).max(axis=2) >= _INK_DELTA).any(axis=0)
    unclean |= _column_mask(protected_boxes, y0, y1, x0, x1)
    blocked = np.nonzero(unclean)[0]
    clean = int(blocked[0]) if len(blocked) else (x1 - x0)
    return max(0.0, clean - plane["pad"])
