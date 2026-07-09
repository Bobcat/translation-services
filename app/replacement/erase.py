"""Erase / fill layer for re-placement: routing flat-vs-model, and residue cleanup.

The flat fill paints each erase quad with its sampled background colour; on textured or
shaded ground that scars, so ``erase_fill_mode="inpaint"`` routes those jobs to the LaMa
fill (:mod:`app.replacement.inpaint`) instead. ``_needs_model_fill`` is the per-job router
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
# Growth of the Tier-2 inpaint mask past the erase quads. The quads' vertical margin is a
# tight anti-alias allowance tuned for the flat paint; the model additionally needs the
# fringe pixels MASKED (not just covered) or it treats them as context to continue.
_INPAINT_MASK_DILATE_PX = 3

# Ground router for erase_fill_mode="inpaint": the model only fills jobs whose ground
# actually varies; on designed flat/solid ground the flat paint is right by construction,
# while a model reconstruction of a near-total hole is unstable run-to-run (washed-out
# streaks through a solid banner). Per job the ring around the quads is split into side
# bands (above/below/left/right), each band into segments along the line; the ground is
# "flat-safe" when within every band the segment medians agree — a designed panel or band
# is constant ALONG the line even when the sides differ from each other (red band above,
# white field below), where texture/shading (crumpled paper, photo ground) drifts along
# it. Thresholds measured across four testset archetypes; designed boundaries running
# THROUGH a line's length read as varying and go to the model — the safe direction.
_GROUND_RING_INNER_PX = 7
_GROUND_RING_OUTER_PX = 23
_GROUND_MAX_SIDE_SPREAD = 20
_GROUND_SEGMENTS_ALONG = 6
_GROUND_SEGMENTS_ACROSS = 3
_GROUND_MIN_BAND_PX = 60
_GROUND_MIN_SEGMENT_PX = 20
# Second route-to-model trigger: fine TEXTURE at a constant median. The side-spread test above
# only catches ground whose median SHIFTS along the line (a gradient or shading); photographic
# ground (concrete, fabric) has a steady median but real grain, and a flat fill paints one colour
# — a smooth patch the eye reads instantly as an erase. Measured across the testset (see
# docs/testset-observations.md): designed-flat / screenshot / sign ground sits <=2, photographic
# texture >=5, so this gate separates them cleanly with margin to spare and does not reopen the
# designed-flat wash-out (those stay well under it).
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


def _needs_model_fill(original: np.ndarray, job: _Job, occupied: np.ndarray) -> bool:
    """Ground router (see the _GROUND_* constants): True when the ground around the job
    varies along the line — texture or shading the flat paint would scar — False when
    every side band is constant along it (designed flat/solid ground, where the flat
    paint is right by construction and a model reconstruction is unstable)."""
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
    crop = original[y0:y1, x0:x1]
    quad_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for quad in job.erase_quads:
        cv2.fillPoly(quad_mask, [np.asarray(quad, dtype=np.int32) - (x0, y0)], 255)
    inner = cv2.dilate(quad_mask, _ellipse(_GROUND_RING_INNER_PX))
    outer = cv2.dilate(quad_mask, _ellipse(_GROUND_RING_OUTER_PX))
    ring_y, ring_x = np.nonzero((outer > 0) & (inner == 0) & (occupied[y0:y1, x0:x1] == 0))
    if ring_y.size == 0:
        return False  # boxed in by other text: no ground to judge, keep the flat paint
    ys, xs = np.nonzero(quad_mask)
    y_lo, y_hi, x_lo, x_hi = ys.min(), ys.max(), xs.min(), xs.max()
    textures: list[float] = []
    for in_band, coord, segments in (
        (ring_y < y_lo, ring_x, _GROUND_SEGMENTS_ALONG),
        (ring_y > y_hi, ring_x, _GROUND_SEGMENTS_ALONG),
        (ring_x < x_lo, ring_y, _GROUND_SEGMENTS_ACROSS),
        (ring_x > x_hi, ring_y, _GROUND_SEGMENTS_ACROSS),
    ):
        if in_band.sum() < _GROUND_MIN_BAND_PX:
            continue
        band_y, band_x, band_c = ring_y[in_band], ring_x[in_band], coord[in_band]
        lo, hi = band_c.min(), band_c.max() + 1
        medians = []
        for step in range(segments):
            seg = (band_c >= lo + (hi - lo) * step // segments) & (
                band_c < lo + (hi - lo) * (step + 1) // segments
            )
            if seg.sum() < _GROUND_MIN_SEGMENT_PX:
                continue
            pixels = crop[band_y[seg], band_x[seg]].astype(np.int16)
            median = np.median(pixels, axis=0)
            medians.append(median)
            # Ground grain: how far the ring pixels sit from their OWN segment median (robust,
            # so an edge pixel does not inflate it). A gradient shifts the medians; texture
            # scatters pixels around a steady one — the case the side-spread test cannot see.
            textures.append(float(np.median(np.abs(pixels - median).max(axis=1))))
        if len(medians) < 2:
            continue
        stack = np.array(medians)
        if np.abs(stack[:, None] - stack[None, :]).max() > _GROUND_MAX_SIDE_SPREAD:
            return True
    return bool(textures) and float(np.median(textures)) > _GROUND_MAX_TEXTURE_MAD


def _ellipse(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px, radius_px))


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
