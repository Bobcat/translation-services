"""Sample background/foreground colour from a cell region (Tier-1, simple).

Text concentrates in the centre of a region, so the per-channel **median of a thin
border ring** (the box edges) is a cleaner background estimate than the median of
the whole box, which mixes in the text strokes and comes out muddy. The foreground
is the region's own measured glyph ink: achromatic ink still snaps to pure black or
white (the common case, kept crisp against JPEG noise — and a fixed background-
luminance threshold misfires on both sides: light text on a mid-light panel came out
black, shadowed black receipt text came out white), while genuinely chromatic ink
renders in its sampled colour.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.replacement.pixels import _INK_DELTA


Color = tuple[int, int, int]
Point = tuple[float, float]

# Below this many ink pixels the region carries no readable evidence (an erase-only sliver)
# and the foreground falls back to contrasting with the background.
_MIN_INK_PIXELS = 24
# Ink whose channel spread stays at/below this is achromatic (black/grey/white) and renders
# as a neutral grey at its measured level; only genuinely coloured ink keeps its full sample.
# 40, not lower: photo lighting and JPEG warmth tint neutral ink (warm-white cigarette-pack
# text, thermal-grey receipt strokes measured at spread ~32) without making it a colour.
_CHROMA_SNAP_SPREAD = 40
# Achromatic ink within this luminance distance of a pole still snaps fully to pure black /
# white: crisp on clean documents (laser black measures ~L20) and free of near-pole churn.
_POLE_SNAP_LUMA = 32
# Core ink must reach this fraction of the observed extreme deviation (toward the polarity
# side). Relative to the EXTREME, not a population quantile: a glossy-box gradient can
# outnumber the glyph pixels 5:1 and drag any percentile onto the box — the extreme can't be
# outvoted.
_CORE_EXTREME_FRACTION = 0.7
# Below this light/dark luminance separation the contrast is chroma-only (orange-on-teal);
# the polarity axis degenerates and core selection falls back to raw channel deviation.
_MIN_POLARITY_EXT = 32
# A connected ink component whose interior radius exceeds this fraction of the region height
# (floored at 3px) is a BLOB — an avatar, icon or glossy-box patch, not a text stroke — and
# does not vote: a dark avatar sharing the reply-bar bbox otherwise hijacks the polarity
# extreme and poles light-grey text to black; the same shape test drops a bright box-gradient
# patch. Bold caps stay comfortably under it (stroke radius <= ~0.09x height).
_MAX_STROKE_RADIUS_FRACTION = 0.12
_MIN_STROKE_RADIUS_PX = 3.0
# Bimodal ink (a black "+" glyph beside grey placeholder text — two REAL inks in one cell):
# when >=60% of the ink mass sits in one tight luminance band and the polarity extreme lies
# far outside it, the extreme is the minority ink, not a wash tail — the majority band votes
# and the extreme is recomputed inside it. A washed unimodal tail (tiny AA'd text) stays on
# the plain extreme path: there the extreme sits close to the band.
_MODE_BAND = 24
_MODE_MASS_MIN = 0.6
_MODE_SEPARATION = 48


def sample_oriented_colors(image: Image.Image, corners: list[Point]) -> tuple[Color, Color]:
    """Background/foreground from a region given by its four oriented corners
    ``[TL, TR, BR, BL]`` in image space. The region is deskewed to an axis-aligned crop
    before the border ring is sampled, so on a tilted coloured band (a diagonal sign bar)
    the ring stays INSIDE the band — an axis-aligned box would let its corners spill into
    whatever surrounds the slant and the median would come out as that, not the band. At
    angle ~0 the oriented box equals the axis box, so flat images sample as before."""
    top_left, top_right, _bottom_right, bottom_left = corners
    width = int(round(_dist(top_left, top_right)))
    height = int(round(_dist(top_left, bottom_left)))
    if width <= 0 or height <= 0:
        return (255, 255, 255), (0, 0, 0)
    src = np.asarray(corners, dtype=np.float32)
    # Warp only the region's own bounding box, not the whole image: this runs once per
    # plane, and converting + array-ifying + warping a full ~12 Mpx frame each time to
    # extract a small crop dominated the render. Cropping to the corners' bbox and shifting
    # the source points by the same offset leaves the warp output identical (every sampled
    # source pixel lies inside the quad, hence inside the bbox).
    img_w, img_h = image.size
    x0 = max(0, int(np.floor(src[:, 0].min())))
    y0 = max(0, int(np.floor(src[:, 1].min())))
    x1 = min(img_w, int(np.ceil(src[:, 0].max())))
    y1 = min(img_h, int(np.ceil(src[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return (255, 255, 255), (0, 0, 0)
    region = np.asarray(image.crop((x0, y0, x1, y1)).convert("RGB"))
    dst = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src - np.asarray([x0, y0], dtype=np.float32), dst)
    crop = cv2.warpPerspective(region, matrix, (width, height))
    if crop.size == 0:
        return (255, 255, 255), (0, 0, 0)
    bg = tuple(int(channel) for channel in np.median(_border_pixels(crop), axis=0))
    return bg, _ink_fg(crop, bg)


def sample_region_colors(image: Image.Image, bbox: dict[str, Any]) -> tuple[Color, Color]:
    left = max(0, int(bbox.get("left") or 0))
    top = max(0, int(bbox.get("top") or 0))
    width = int(bbox.get("width") or 0)
    height = int(bbox.get("height") or 0)
    if width <= 0 or height <= 0:
        return (255, 255, 255), (0, 0, 0)

    crop = np.asarray(image.crop((left, top, left + width, top + height)).convert("RGB"))
    if crop.size == 0:
        return (255, 255, 255), (0, 0, 0)

    bg = tuple(int(channel) for channel in np.median(_border_pixels(crop), axis=0))
    return bg, _ink_fg(crop, bg)


_LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722])


def _ink_fg(crop: np.ndarray, bg: Color) -> Color:
    """The region's own glyph ink colour, in three steps.

    1. **Core selection along the polarity axis, relative to the observed extreme.** The ink
       side (lighter or darker than bg) is whichever luminance extreme reaches further; core
       ink must get within ``_CORE_EXTREME_FRACTION`` of that extreme. Anti-aliased blends and
       — the shiny-box case — a bg gradient inside the box can outnumber the glyph pixels,
       so population quantiles land on the pollutant; the extreme cannot be outvoted. With no
       luminance separation (chroma-only contrast) the strongest channel deviations vote.
    2. **Achromatic ink renders at its measured neutral LEVEL** (soft dark grey for shadowed
       receipt print, dim white for a photographed pack) instead of a hard pole; within
       ``_POLE_SNAP_LUMA`` of a pole it still snaps fully — clean documents stay crisp
       pure black/white. NAMED LIMIT: the residual warm/cool tint of a "neutral" verdict is
       dropped (rendered grey), so near-gate ink loses its warmth (binary gate kept by
       design — the continuous blend is the parked follow-up).
    3. **Chromatic ink keeps its measured colour.** Too little ink -> contrast with bg."""
    deviation = np.abs(crop.astype(np.int16) - np.asarray(bg, dtype=np.int16)).max(axis=2)
    ink_mask = deviation >= _INK_DELTA
    if int(ink_mask.sum()) < _MIN_INK_PIXELS:
        return contrasting_fg(bg)
    ink_mask = _stroke_shaped(ink_mask, crop.shape[0])
    ink_all = crop[ink_mask].astype(np.float64)
    lum = ink_all @ _LUMA_WEIGHTS
    bg_lum = _luminance(bg)
    light_ext = float(np.percentile(lum, 98)) - bg_lum
    dark_ext = bg_lum - float(np.percentile(lum, 2))
    # Majority-mode vote for bimodal ink (see _MODE_* above): keep the dominant band, then
    # recompute the extremes inside it so core selection and level follow the majority ink.
    med_lum = float(np.median(lum))
    extreme_lum = bg_lum + light_ext if light_ext >= dark_ext else bg_lum - dark_ext
    mode = np.abs(lum - med_lum) <= _MODE_BAND
    if float(mode.mean()) >= _MODE_MASS_MIN and abs(med_lum - extreme_lum) > _MODE_SEPARATION:
        ink_all = ink_all[mode]
        lum = lum[mode]
        light_ext = float(np.percentile(lum, 98)) - bg_lum
        dark_ext = bg_lum - float(np.percentile(lum, 2))
    if max(light_ext, dark_ext) >= _MIN_POLARITY_EXT:
        if light_ext >= dark_ext:
            core = ink_all[lum >= bg_lum + _CORE_EXTREME_FRACTION * light_ext]
        else:
            core = ink_all[lum <= bg_lum - _CORE_EXTREME_FRACTION * dark_ext]
    else:
        dev_ink = deviation[ink_mask]
        core = ink_all[dev_ink >= max(_INK_DELTA, int(np.percentile(dev_ink, 75)))]
    if len(core) < _MIN_INK_PIXELS:
        core = ink_all
    ink_median = np.median(core, axis=0)
    if float(ink_median.max() - ink_median.min()) <= _CHROMA_SNAP_SPREAD:
        # The LEVEL comes from the polarity extreme (p98/p2 luminance), not the core median:
        # on small glyphs no pixel is unblended and the median floats mid-grey where the
        # source reads black — the extreme is the least-blended evidence. Uniform ink is
        # unaffected (extreme == median there).
        if light_ext >= dark_ext:
            level = int(round(min(255.0, bg_lum + max(light_ext, 0.0))))
        else:
            level = int(round(max(0.0, bg_lum - max(dark_ext, 0.0))))
        if level <= _POLE_SNAP_LUMA:
            return (0, 0, 0)
        if level >= 255 - _POLE_SNAP_LUMA:
            return (255, 255, 255)
        return (level, level, level)
    return tuple(int(channel) for channel in ink_median)


def _stroke_shaped(mask: np.ndarray, region_height: int) -> np.ndarray:
    """``mask`` restricted to text-stroke-shaped components. Per connected component the max
    interior radius (distance transform) is compared against the stroke limit; blobs are
    dropped WHOLE — per-pixel filtering would leave their rims as a false ink halo. Falls
    back to the unfiltered mask when too little ink survives (an icon-only cell measures the
    icon, as before)."""
    mask_u8 = mask.astype(np.uint8)
    count, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    if count <= 1:
        return mask
    radius = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    comp_max = np.zeros(count)
    np.maximum.at(comp_max, labels.ravel(), radius.ravel())
    keep = comp_max <= max(_MIN_STROKE_RADIUS_PX, _MAX_STROKE_RADIUS_FRACTION * region_height)
    keep[0] = False  # background label
    filtered = keep[labels]
    return filtered if int(filtered.sum()) >= _MIN_INK_PIXELS else mask


def _border_pixels(crop: np.ndarray) -> np.ndarray:
    """Pixels from a thin frame around the region — background-dominated, text-light."""
    h, w = crop.shape[:2]
    band = max(1, min(h, w) // 6)
    edges = [
        crop[:band], crop[h - band:],
        crop[:, :band], crop[:, w - band:],
    ]
    return np.concatenate([edge.reshape(-1, 3) for edge in edges], axis=0)


def contrasting_fg(bg: Color) -> Color:
    return (0, 0, 0) if _luminance(bg) >= 140 else (255, 255, 255)


def _luminance(color: Any) -> float:
    return float(0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2])


def _dist(a: Point, b: Point) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))
