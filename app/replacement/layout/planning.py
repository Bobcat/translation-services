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
from app.replacement.text.size import _group_size
from app.replacement.text.size import _quad_band_height
from app.replacement.text.angle import _line_clusters
from app.replacement.text.angle import _baseline_angle
from app.replacement.text.wrap import _fit_group
from app.replacement.text.wrap import _raw_condense
from app.replacement.text.wrap import _condense_scale
from app.replacement.text.wrap import _CONDENSE_FLOOR
from app.replacement.text.fit import fold_lone_fullwidth_punctuation
from app.replacement.text.fit import is_cjk_text
from app.replacement.ground.color import sample_oriented_colors
from app.replacement.layout.tables import _split_table_row
from app.replacement.layout.tables import _reproduced_in
from app.replacement.layout.markers import _cell_marker
from app.replacement.layout.markers import _prepend_marker
from app.replacement.layout.markers import _strip_leading_glyph
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

def _plan_group(
    base: Image.Image,
    units: list[dict[str, Any]],
    *,
    snap_horizontal: bool = False,
    render_size_mode: str = "median",
    width_fit_mode: str = "footprint",
    band_ratio: float | None = None,
    angle_field: tuple[float, float] | None = None,
    base_arr: np.ndarray | None = None,
    protected_boxes: list[dict[str, Any]] | None = None,
    sweep_ok: bool | None = None,
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
        cells = _split_table_row(units[0])
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
                    angle_field=angle_field,
                    base_arr=base_arr,
                    protected_boxes=protected_boxes,
                    sweep_ok=sweep_ok,
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
            own_boxes.append(dict(member["bbox"]))
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
        planes.append({
            "quads": quads,
            "tokens": _plane_source_tokens(quads, group_quads, quad_tokens),
            "target": max(8, int(true_height * size_ratio)),
            "true_height": true_height,
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

    centered = any(str(unit.get("alignment") or "") == "center" for unit in units)
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
            # Snap the fg by majority too: the per-plane fg is ink-evidence (not derived from
            # bg), so recomputing it from the snapped bg would throw that evidence away.
            fgs = [fg for _, fg in colors]
            colors = [(median_bg, max(set(fgs), key=fgs.count))] * len(colors)

    # "band" size metric: clamp each plane's size target to its ink band scaled by the
    # document's own extent/band norm — a line whose polygon is stretched by sparse tall
    # glyphs sinks to the size its band says, everything else stays on the extent path
    # (one-sided: min(), never enlarges; weak ink evidence keeps the extent).
    if band_ratio is not None and angle == 0.0:
        band_px = base_arr
        for plane, (bg, _fg) in zip(planes, colors):
            bands = [b for q in plane["quads"] if (b := _quad_band_height(band_px, q, bg)) is not None]
            if bands:
                clamped = min(plane["true_height"], median(bands) * band_ratio)
                plane["target"] = max(8, int(clamped * size_ratio))

    # "extend" width fit: widen each plane's usable width into VERIFIED clean background to
    # its right before fitting, so a longer translation of a short line (a list item) keeps
    # its size instead of condensing/shrinking. Strictly additive: every guard that fails
    # leaves the plane at its footprint width, i.e. exactly the "footprint" behaviour. Only
    # for axis-aligned groups (the scan is axis-aligned — same honest limit as the ink
    # sweep) and never for centered ones (growing right would break the centring).
    if width_fit_mode == "extend" and angle == 0.0 and not centered:
        for plane, (bg, _fg) in zip(planes, colors):
            plane["width"] += _clean_right_extension(base_arr, plane, bg, protected_boxes or [])

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
