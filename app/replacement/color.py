"""Sample background/foreground colour from a cell region (Tier-1, simple).

Text is sparse within a cell box, so the per-channel **median** of the region is a
robust estimate of the background. The foreground is taken as black or white,
whichever contrasts with the background luminance. Real text-colour sampling is a
later refinement (see docs/re-placement.md).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


Color = tuple[int, int, int]


def sample_region_colors(image: Image.Image, bbox: dict[str, Any]) -> tuple[Color, Color]:
    left = max(0, int(bbox.get("left") or 0))
    top = max(0, int(bbox.get("top") or 0))
    width = int(bbox.get("width") or 0)
    height = int(bbox.get("height") or 0)
    if width <= 0 or height <= 0:
        return (255, 255, 255), (0, 0, 0)

    crop = image.crop((left, top, left + width, top + height)).convert("RGB")
    pixels = np.asarray(crop).reshape(-1, 3)
    if pixels.size == 0:
        return (255, 255, 255), (0, 0, 0)

    bg = tuple(int(channel) for channel in np.median(pixels, axis=0))
    return bg, _contrasting_fg(bg)


def _contrasting_fg(bg: Color) -> Color:
    luminance = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    return (0, 0, 0) if luminance >= 140 else (255, 255, 255)
