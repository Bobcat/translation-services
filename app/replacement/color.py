"""Sample background/foreground colour from a cell region (Tier-1, simple).

Text concentrates in the centre of a region, so the per-channel **median of a thin
border ring** (the box edges) is a cleaner background estimate than the median of
the whole box, which mixes in the text strokes and comes out muddy. The foreground
is taken as black or white, whichever contrasts with the background luminance. Real
text-colour sampling is a later refinement (see docs/re-placement.md).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


Color = tuple[int, int, int]
Point = tuple[float, float]


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
    return bg, contrasting_fg(bg)


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
    return bg, contrasting_fg(bg)


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
    luminance = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    return (0, 0, 0) if luminance >= 140 else (255, 255, 255)


def _dist(a: Point, b: Point) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))
