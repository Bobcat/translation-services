"""Erase / fill layer for re-placement: routing flat-vs-model, and residue cleanup.

The flat fill paints each erase quad with its sampled background colour; on textured or
shaded ground that scars, so ``erase_fill_mode="inpaint"`` routes those jobs to the LaMa
fill (:mod:`app.replacement.ground.inpaint`) instead. ``_needs_model_fill`` is the per-job router
that decides flat-vs-model (see the ``_GROUND_*`` constants); ``_erase_mask`` builds the
model's mask; ``_residue_regions`` / ``_swallow_erase_residue`` recover leftover ink of the
erased text that the tight quads cut through.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.replacement.pixels import _INK_DELTA

if TYPE_CHECKING:
    from app.replacement.jobs import _Job

# Residue swallow: how far past the erase quads leftover ink of the erased text is searched.
# Descenders and anti-alias fringes reach a few px; anything farther is not ours to erase.
_RESIDUE_MARGIN = 12

# Table rules: a thin, wide horizontal dark line is structure, not text. A cell's erase quad
# grown by its AA pad reaches into the row gap where a rule sits and paints background over it,
# breaking the rule into fragments (measured: 5 of 7 interior rules of a booktabs table erased).
# Detect rules in the source and restore them where an erase quad covered them.
_RULE_DARK = 128          # a rule pixel is at least this dark (0-255)
_RULE_MIN_WIDTH = 150     # a continuous horizontal dark run this wide is a rule, not a glyph stroke
_RULE_MAX_HEIGHT = 4      # rules are hairlines; taller solid bands are filled cells, not rules
# A table rule is ISOLATED: it sits in a row gap with background a few px above AND below. This
# is what separates it from the horizontal lines a naive detector also catches — a photo edge
# (dark content around it) or an underline welded to its text (ink directly above). Without this
# those get restored too and re-glue words the erase had cleanly separated (measured across the
# scanned brochure and the weather fixtures).
_RULE_ISOLATION_LIGHT = 200   # background is at least this light (0-255)
_RULE_ISOLATION_GAP = 2       # rows above and below that must be background for a rule to qualify
# A table rule is ACHROMATIC (black/grey ink). A saturated horizontal line is a designed accent,
# not table structure: restoring one changed nothing visible (17 px on the measured quickguide)
# but destabilised the re-OCR of the styled graphic beside it. Leave colour to the erase.
_RULE_MAX_CHROMA = 40         # max channel spread (max-min) for a rule pixel
# Growth of the Tier-2 inpaint mask past the erase quads. The quads' vertical margin is a
# tight anti-alias allowance tuned for the flat paint; the model additionally needs the
# fringe pixels MASKED (not just covered) or it treats them as context to continue.
_INPAINT_MASK_DILATE_PX = 3

# Ground router for erase_fill_mode="inpaint": the model only fills jobs whose ground
# actually varies; on designed flat/solid ground the flat paint is right by construction,
# while a model reconstruction of a near-total hole is unstable run-to-run (washed-out
# streaks through a solid banner). Per job the ring around the quads is split into side
# bands (above/below/left/right), each band into segments along the line. Three triggers,
# all measured across the testset + the reported live misroutes (docs/testset-observations.md):
#
# 1. WITHIN-BAND spread of the segment medians (> _GROUND_MAX_SIDE_SPREAD) — shading/texture
#    drifting ALONG the line. Judged on the job's OWN surface only (pixels within
#    _GROUND_OWN_SURFACE_DELTA of the fill's bg colour): a DESIGNED boundary near the line
#    (a graphic, a panel edge, the sign's rim) is another surface the fill never touches,
#    and counting it sent solid-panel lines to the model — whose near-total-hole fill then
#    smeared the very graphics it was routed for. And judged on the NEAR ring only
#    (_GROUND_SPREAD_RING_PX): the flat fill has to blend at its seam; weathering at the
#    far ring does not scar it.
# 2. CROSS-BAND spread of all own-surface medians (> max(_GROUND_CROSS_FLOOR,
#    _GROUND_CROSS_REL x luminance)) — a smooth illumination gradient ACROSS the line
#    (top vs bottom), invisible to per-band tests. Weber: a plate's visibility scales with
#    delta/luminance, so dark ground trips at a smaller absolute delta. The threshold is
#    deliberately high (0.5 x luma): mild vertical shading (a curved receipt, a lit sign)
#    stays flat; only a gradient the eye reads as a plate goes to the model.
# 3. TEXTURE (unchanged): grain around a steady median, full ring, unfiltered.
_GROUND_RING_INNER_PX = 7
_GROUND_RING_OUTER_PX = 23
_GROUND_SPREAD_RING_PX = 14
_GROUND_OWN_SURFACE_DELTA = 60
_GROUND_MAX_SIDE_SPREAD = 20
_GROUND_CROSS_REL = 0.5
_GROUND_CROSS_FLOOR = 16
_GROUND_SEGMENTS_ALONG = 6
_GROUND_SEGMENTS_ACROSS = 3
_GROUND_MIN_BAND_PX = 60
_GROUND_MIN_SEGMENT_PX = 20
# Fine TEXTURE at a constant median: photographic ground (concrete, fabric) has a steady
# median but real grain, and a flat fill paints one colour — a smooth patch the eye reads
# instantly as an erase. Measured: designed-flat / screenshot / sign ground sits <=2,
# photographic texture >=5; a weathered sign panel lands right at the boundary (3.6) and is
# the named borderline. Computed on the full ring, unfiltered (a boundary crossing a segment
# inflates it toward the model — the safe direction for texture).
_GROUND_MAX_TEXTURE_MAD = 3.5


def _swallow_erase_residue(
    canvas: np.ndarray, original: np.ndarray, jobs: list[_Job]
) -> None:
    """Extend the flat erase to residue of the erased text itself (see
    :func:`_residue_regions`): each residue component is painted with its job's
    background colour, like the quad fill it belongs to."""
    for y0, y1, x0, x1, pixels, job in _residue_regions(original, jobs):
        region = canvas[y0:y1, x0:x1]
        region[pixels] = np.asarray(job.bg_color, dtype=np.uint8)


def _band_specs(quad_mask: np.ndarray, ring_y: np.ndarray, ring_x: np.ndarray):
    """The four side bands (above/below/left/right) of a ring as (mask, run-coordinate,
    segment count) triples — segments run ALONG the line for the horizontal bands and
    across it for the short side bands."""
    ys, xs = np.nonzero(quad_mask)
    y_lo, y_hi, x_lo, x_hi = ys.min(), ys.max(), xs.min(), xs.max()
    return (
        (ring_y < y_lo, ring_x, _GROUND_SEGMENTS_ALONG),
        (ring_y > y_hi, ring_x, _GROUND_SEGMENTS_ALONG),
        (ring_x < x_lo, ring_y, _GROUND_SEGMENTS_ACROSS),
        (ring_x > x_hi, ring_y, _GROUND_SEGMENTS_ACROSS),
    )


def _needs_model_fill(original: np.ndarray, job: _Job, occupied: np.ndarray) -> bool:
    """Ground router (see the _GROUND_* constants): True when the job's OWN ground varies —
    shading along the line, a gradient across it, or grain — so a flat plate would scar;
    False on designed flat/solid ground, where the flat paint is right by construction and
    a model reconstruction of a near-total hole is unstable."""
    height, width = original.shape[:2]
    # Work in a local window: the ground ring reaches only _GROUND_RING_OUTER_PX past the
    # quads, so a full-image mask + dilate per job is pure waste (on a large receipt with
    # many jobs the two full-frame dilates dominate the whole render). Crop to the quads'
    # bbox grown by the ring radius; every array below is that window, in local coords.
    all_pts = np.concatenate([np.asarray(q, dtype=np.int32) for q in job.erase_quads])
    margin = _GROUND_RING_OUTER_PX + 1
    x0 = max(0, int(all_pts[:, 0].min()) - margin)
    y0 = max(0, int(all_pts[:, 1].min()) - margin)
    x1 = min(width, int(all_pts[:, 0].max()) + margin + 1)
    y1 = min(height, int(all_pts[:, 1].max()) + margin + 1)
    crop = original[y0:y1, x0:x1].astype(np.int16)
    free = occupied[y0:y1, x0:x1] == 0
    quad_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for quad in job.erase_quads:
        cv2.fillPoly(quad_mask, [np.asarray(quad, dtype=np.int32) - (x0, y0)], 255)
    inner = cv2.dilate(quad_mask, _ellipse(_GROUND_RING_INNER_PX))
    bg = np.asarray(job.bg_color, dtype=np.int16)

    # Texture: full ring, unfiltered per-segment grain (see the constants block).
    outer = cv2.dilate(quad_mask, _ellipse(_GROUND_RING_OUTER_PX))
    ring_y, ring_x = np.nonzero((outer > 0) & (inner == 0) & free)
    if ring_y.size == 0:
        return False  # boxed in by other text: no ground to judge, keep the flat paint
    textures: list[float] = []
    for in_band, coord, segments in _band_specs(quad_mask, ring_y, ring_x):
        if in_band.sum() < _GROUND_MIN_BAND_PX:
            continue
        band_y, band_x, band_c = ring_y[in_band], ring_x[in_band], coord[in_band]
        lo, hi = band_c.min(), band_c.max() + 1
        for step in range(segments):
            seg = (band_c >= lo + (hi - lo) * step // segments) & (
                band_c < lo + (hi - lo) * (step + 1) // segments
            )
            if seg.sum() < _GROUND_MIN_SEGMENT_PX:
                continue
            pixels = crop[band_y[seg], band_x[seg]]
            median = np.median(pixels, axis=0)
            # Ground grain: how far the ring pixels sit from their OWN segment median (robust,
            # so an edge pixel does not inflate it). A gradient shifts the medians; texture
            # scatters pixels around a steady one — the case the spread tests cannot see.
            textures.append(float(np.median(np.abs(pixels - median).max(axis=1))))

    # Spreads: near ring (the flat fill's blend seam), own-surface pixels only.
    outer_near = cv2.dilate(quad_mask, _ellipse(_GROUND_SPREAD_RING_PX))
    ring_y, ring_x = np.nonzero((outer_near > 0) & (inner == 0) & free)
    own_medians: list[np.ndarray] = []
    if ring_y.size:
        for in_band, coord, segments in _band_specs(quad_mask, ring_y, ring_x):
            if in_band.sum() < _GROUND_MIN_BAND_PX:
                continue
            band_y, band_x, band_c = ring_y[in_band], ring_x[in_band], coord[in_band]
            lo, hi = band_c.min(), band_c.max() + 1
            band_medians = []
            for step in range(segments):
                seg = (band_c >= lo + (hi - lo) * step // segments) & (
                    band_c < lo + (hi - lo) * (step + 1) // segments
                )
                if seg.sum() < _GROUND_MIN_SEGMENT_PX:
                    continue
                pixels = crop[band_y[seg], band_x[seg]]
                own = pixels[np.abs(pixels - bg).max(axis=1) <= _GROUND_OWN_SURFACE_DELTA]
                if len(own) >= _GROUND_MIN_SEGMENT_PX:
                    band_medians.append(np.median(own, axis=0))
                # else: the segment is (mostly) another surface — the fill never touches it
            if len(band_medians) >= 2:
                stack = np.array(band_medians)
                if np.abs(stack[:, None] - stack[None, :]).max() > _GROUND_MAX_SIDE_SPREAD:
                    return True  # shading/texture drifting along the line
            own_medians.extend(band_medians)
    if len(own_medians) >= 2:
        stack = np.array(own_medians)
        cross = float(np.abs(stack[:, None] - stack[None, :]).max())
        centre = np.median(stack, axis=0)
        luma = 0.2126 * centre[0] + 0.7152 * centre[1] + 0.0722 * centre[2]
        if cross > max(_GROUND_CROSS_FLOOR, _GROUND_CROSS_REL * luma):
            return True  # an illumination gradient across the line: a plate would show
    return bool(textures) and float(np.median(textures)) > _GROUND_MAX_TEXTURE_MAD


def _ellipse(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px, radius_px))


def _restore_table_rules(canvas: np.ndarray, original: np.ndarray, jobs: list["_Job"]) -> None:
    """Restore table rules the erase pass broke. A rule is a thin (<= _RULE_MAX_HEIGHT px), wide
    (>= _RULE_MIN_WIDTH px) horizontal dark line — table structure that sits in the row gaps.
    A cell's erase quad, grown by its AA pad, reaches into the gap and paints background over the
    rule; this copies the source rule pixels back onto ``canvas`` wherever an erase quad covered
    them. Restore-only-where-erased means untouched rules stay as-is and text ink is never
    affected; called BEFORE the text composite, so text that genuinely overlaps a rule still wins.
    Generalizes to any ruled table or line-art (a floor plan's walls) — anything the erase clipped."""
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) if original.ndim == 3 else original
    dark = gray < _RULE_DARK
    if original.ndim == 3:  # black/grey ink only — a coloured accent line is not table structure
        spread = original.max(axis=2).astype(np.int16) - original.min(axis=2).astype(np.int16)
        dark &= spread <= _RULE_MAX_CHROMA
    dark = dark.astype(np.uint8)
    # Keep only pixels on a long horizontal run (a rule); a glyph stroke is far shorter.
    horizontal = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (_RULE_MIN_WIDTH, 1))
    )
    # Drop tall solid bands (a filled dark cell has long horizontal runs too, but is not a rule).
    thick = cv2.morphologyEx(
        horizontal, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, _RULE_MAX_HEIGHT + 2))
    )
    rules = cv2.subtract(horizontal, thick)
    if not rules.any():
        return
    # Isolation gate: keep only rule pixels with background a few rows above AND below, so a
    # photo edge or a text underline (dark neighbours) is not mistaken for a table rule.
    light = gray >= _RULE_ISOLATION_LIGHT
    gap = _RULE_ISOLATION_GAP
    above = np.zeros_like(light)
    above[gap:] = light[:-gap]  # background 'gap' rows above each pixel
    below = np.zeros_like(light)
    below[:-gap] = light[gap:]  # background 'gap' rows below
    rules = rules & (above & below).astype(np.uint8)
    if not rules.any():
        return
    erased = np.zeros(gray.shape, dtype=np.uint8)
    for job in jobs:
        for quad in job.erase_quads:
            cv2.fillPoly(erased, [np.asarray(quad, dtype=np.int32)], 255)
    restore = (rules > 0) & (erased > 0)
    canvas[restore] = original[restore]


def _erase_mask(original: np.ndarray, jobs: list[_Job]) -> np.ndarray:
    """Erase mask for the Tier-2 (LaMa) fill: every erase quad, dilated a few px so the
    anti-alias fringe past the tight quad edges is masked too (a half-covered glyph edge
    left in the context reads as text the model would continue into the hole), plus the
    same residue components the flat path swallows — reconstructed instead of painted."""
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    for job in jobs:
        for quad in job.erase_quads:
            cv2.fillPoly(mask, [np.asarray(quad, dtype=np.int32)], 255)
    size = _INPAINT_MASK_DILATE_PX * 2 + 1
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    for y0, y1, x0, x1, pixels, _job in _residue_regions(original, jobs):
        mask[y0:y1, x0:x1][pixels] = 255
    return mask


def _residue_regions(original: np.ndarray, jobs: list[_Job]):
    """Residue of the erased text itself: an ink component that lies mostly INSIDE the
    erase quads is a leftover part of the text being removed — a descender the tight quad
    bottom cut through, an anti-alias fringe past the quad edge. Yields
    ``(y0, y1, x0, x1, pixel_mask, job)`` per job window with residue. Detection runs on
    the ORIGINAL pixels: on an erased canvas a residue crumb no longer connects to its
    glyph's interior ink and the majority-inside test would see it as fully outside.
    Self-limiting on unreliable ground: busy texture merges the glyphs into one large
    mostly-outside component and nothing happens, and a component reaching the search
    window's border has an unknown true extent (a table rule grazing the line) and is
    skipped. Named limit: a surviving neighbour glyph already overlapped >50% by an erase
    quad is taken whole instead of left half-cut."""
    occupied = np.zeros(original.shape[:2], dtype=np.uint8)
    for job in jobs:
        for quad in job.erase_quads:
            cv2.fillPoly(occupied, [np.asarray(quad, dtype=np.int32)], 255)
    for job in jobs:
        bg = np.asarray(job.bg_color, dtype=np.int16)
        points = np.vstack([np.asarray(q, dtype=np.int32) for q in job.erase_quads])
        x0, y0 = np.maximum(points.min(axis=0) - _RESIDUE_MARGIN, 0)
        x1, y1 = points.max(axis=0) + _RESIDUE_MARGIN + 1
        window = original[y0:y1, x0:x1].astype(np.int16)
        ink = (np.abs(window - bg).max(axis=2) >= _INK_DELTA).astype(np.uint8)
        if not ink.any():
            continue
        count, labels = cv2.connectedComponents(ink, connectivity=8)
        inside = np.bincount(labels[occupied[y0:y1, x0:x1] > 0].ravel(), minlength=count)
        total = np.bincount(labels.ravel(), minlength=count)
        border = np.zeros(labels.shape, dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        ids = np.arange(count)
        swallow = (
            (ids > 0)
            & (inside * 2 >= total)
            # Fully-inside components are already under the quad paint; only the partial
            # ones need work.
            & (inside < total)
            & ~np.isin(ids, np.unique(labels[border]))
        )
        if swallow.any():
            yield y0, y1, x0, x1, np.isin(labels, ids[swallow]), job
