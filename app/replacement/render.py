"""Stage #8 re-placement — background-matched, polygon-aware (Tier-1, model-free).

Per unit: cover the original with the locally-sampled **background colour** (so it
reads as erased on a flat surface — menu paper, sign panel, receipt), then draw the
translation and **warp it onto the unit's polygon** so it follows the page tilt
(rotation/perspective), for a clean camera-translation look.

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

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw

from app.replacement import geometry as geo
from app.replacement.color import sample_region_colors
from app.replacement.fit import fit_text


# Font size from the true (de-skewed) line height. The polygon height spans the full
# glyph extent, a touch taller than the visual cap; scale down slightly to match.
_SIZE_RATIO = 0.9


@dataclass(frozen=True)
class _Job:
    erase_quad: list[tuple[int, int]]
    bg_color: tuple[int, int, int]
    tile: Image.Image
    dst_quad: list[tuple[float, float]]


def render_translated_image(input_path: Path, translation_units: list[dict[str, Any]]) -> bytes:
    base = Image.open(input_path).convert("RGB")

    jobs: list[_Job] = []
    for unit in translation_units:
        job = _plan_unit(base, unit)
        if job is not None:
            jobs.append(job)

    # Pass 1: cover every original (along the slant) so no source text peeks through.
    erase = ImageDraw.Draw(base)
    for job in jobs:
        erase.polygon(job.erase_quad, fill=job.bg_color)

    # Pass 2: warp each text tile onto its oriented region.
    canvas = np.asarray(base).copy()
    for job in jobs:
        _composite(canvas, job)

    out = BytesIO()
    Image.fromarray(canvas).save(out, format="PNG")
    return out.getvalue()


def _plan_unit(base: Image.Image, unit: dict[str, Any]) -> _Job | None:
    translated = str(unit.get("translated_text") or "").strip()
    if len(translated) <= 1:  # empty / OCR-noise single char -> leave the original alone
        return None
    members = [m for m in (unit.get("members") or []) if m.get("translate") and m.get("bbox")]
    quads = [quad for quad in (geo.quad_of(m) for m in members) if quad is not None]
    if not quads:
        return None

    angle = median(geo.angle_deg(quad) for quad in quads)
    true_height = median(geo.line_height(quad) for quad in quads)
    target_size = max(8, int(true_height * _SIZE_RATIO))
    pad = max(2.0, true_height / 6.0)

    x_axis, y_axis, xmin, xmax, ymin, ymax = geo.oriented_frame(quads, angle)
    region_w = xmax - xmin
    region_h = ymax - ymin
    bg, fg = sample_region_colors(base, geo.axis_bbox(quads))

    # The translation never takes more room than the original text: it starts at the
    # original size (true-height target) and steps down until it fits the unit's own
    # de-skewed footprint. A shorter translation keeps the original size — no growing.
    max_height = int(region_h + 2 * pad)
    fitted = fit_text(translated, max(1, int(region_w)), max_height, wrap=False, max_size=target_size)

    text_w = max((int(fitted.font.getlength(line)) for line in fitted.lines), default=0)
    # fit_text returns its smallest-font attempt even when that still does not fit (a
    # long chat-reply "translation" on a pictogram-sized cell). The footprint rule wins:
    # leave the original pixels rather than erase far beyond them.
    if text_w > max(1, int(region_w)):
        return None
    tile_w = max(1, text_w + 2 * int(pad))
    tile_h = max(1, fitted.line_height * len(fitted.lines) + 2 * int(pad))
    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    y = int(pad)
    for line in fitted.lines:
        tile_draw.text((int(pad), y), line, font=fitted.font, fill=fg + (255,))
        y += fitted.line_height

    # Origin = unit top-left in the rotated frame, lifted by the padding margin.
    ox, oy = xmin - pad, ymin - pad
    dst_quad = [
        geo.to_image(ox, oy, x_axis, y_axis),
        geo.to_image(ox + tile_w, oy, x_axis, y_axis),
        geo.to_image(ox + tile_w, oy + tile_h, x_axis, y_axis),
        geo.to_image(ox, oy + tile_h, x_axis, y_axis),
    ]
    # Erase region: the original extent, grown to cover the (possibly larger) text tile.
    ex1 = max(xmax + pad, ox + tile_w)
    ey1 = max(ymax + pad, oy + tile_h)
    erase_quad = [
        _ipoint(geo.to_image(ox, oy, x_axis, y_axis)),
        _ipoint(geo.to_image(ex1, oy, x_axis, y_axis)),
        _ipoint(geo.to_image(ex1, ey1, x_axis, y_axis)),
        _ipoint(geo.to_image(ox, ey1, x_axis, y_axis)),
    ]
    return _Job(erase_quad=erase_quad, bg_color=bg, tile=tile, dst_quad=dst_quad)


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
