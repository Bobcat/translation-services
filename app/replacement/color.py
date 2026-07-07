"""Sample background/foreground colour from a cell region (Tier-1, simple).

Text concentrates in the centre of a region, so the per-channel **median of a thin
border ring** (the box edges) is a cleaner background estimate than the median of
the whole box, which mixes in the text strokes and comes out muddy. The foreground
is black or white, decided by the region's own glyph ink (pixels deviating from the
background): a fixed background-luminance threshold misfires on both sides — light
text on a mid-light panel came out black, shadowed black receipt text came out
white. True text-COLOUR sampling is a later refinement (see docs/re-placement.md).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


Color = tuple[int, int, int]
Point = tuple[float, float]

# A pixel deviating this much from the background (any channel) is glyph ink — the same
# delta the render's stray-ink machinery uses (render._INK_DELTA).
_INK_DELTA = 48
# Below this many ink pixels the region carries no readable evidence (an erase-only sliver)
# and the foreground falls back to contrasting with the background.
_MIN_INK_PIXELS = 24


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
    dst = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(np.asarray(image.convert("RGB")), matrix, (width, height))
    if crop.size == 0:
        return (255, 255, 255), (0, 0, 0)
    bg = tuple(int(channel) for channel in np.median(_border_pixels(crop), axis=0))
    return bg, _ink_polarity_fg(crop, bg)


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
    return bg, _ink_polarity_fg(crop, bg)


def _ink_polarity_fg(crop: np.ndarray, bg: Color) -> Color:
    """Black or white, whichever matches the region's own glyph ink: the median colour of
    the pixels deviating ``_INK_DELTA`` from the background, compared with the background by
    luminance. Binary — not the sampled colour itself — so rendered text stays crisp; with
    too little ink to judge, fall back to contrasting with the background."""
    deviation = np.abs(crop.astype(np.int16) - np.asarray(bg, dtype=np.int16)).max(axis=2)
    ink = crop[deviation >= _INK_DELTA]
    if len(ink) < _MIN_INK_PIXELS:
        return contrasting_fg(bg)
    ink_median = np.median(ink, axis=0)
    return (0, 0, 0) if _luminance(ink_median) < _luminance(bg) else (255, 255, 255)


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
