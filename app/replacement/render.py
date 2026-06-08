"""Stage #8 re-placement — background-matched, polygon-aware (Tier-1, model-free).

Per unit: cover the original with the locally-sampled **background colour** (so it
reads as erased on a flat surface — menu paper, sign panel, receipt), then draw the
translation and **warp it onto the unit's polygon** so it follows the page tilt
(rotation/perspective), matching DeepL's camera-translation look.

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
    line_boxes = _line_boxes(members)
    n_lines = max(1, len(line_boxes))
    align = _alignment(line_boxes)

    # Cadence: render the translation in at most the ORIGINAL number of lines, at the
    # largest size that still fits the original block width — so the block keeps the
    # original shape/position and the font only shrinks to absorb a longer translation
    # (no extra lines, no overflow into the next item).
    if str(unit.get("kind") or "field") == "flow":
        wrap = True
        max_width = region_w - 2 * pad
        max_height = float(base.height)  # the line cap below is the real constraint
        max_lines: int | None = n_lines
    else:
        wrap = False
        # Single line: let it grow horizontally instead of shrinking to the (often
        # narrow) original word width — otherwise "frites" -> "French fries" gets
        # squeezed tiny next to its full-size siblings.
        max_width = 1_000_000.0
        max_height = float(target_size * 2 + 2 * pad)
        max_lines = 1
    fitted = fit_text(
        translated, max(1, int(max_width)), int(max_height), wrap=wrap, max_size=target_size, max_lines=max_lines
    )

    pad_i = int(pad)
    text_w = max((int(fitted.font.getlength(line)) for line in fitted.lines), default=0)
    tile_w = max(1, text_w + 2 * pad_i)
    tile_h = max(1, fitted.line_height * len(fitted.lines) + 2 * pad_i)
    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    y = pad_i
    for line in fitted.lines:
        line_w = int(fitted.font.getlength(line))
        x = pad_i + max(0, (text_w - line_w) // 2) if align == "center" else pad_i
        tile_draw.text((x, y), line, font=fitted.font, fill=fg + (255,))
        y += fitted.line_height

    # Anchor the tile in the rotated frame to match the original block's alignment
    # (centred headlines stay centred instead of snapping left).
    oy = ymin - pad
    ox = 0.5 * (xmin + xmax) - tile_w / 2.0 if align == "center" else xmin - pad
    dst_quad = [
        geo.to_image(ox, oy, x_axis, y_axis),
        geo.to_image(ox + tile_w, oy, x_axis, y_axis),
        geo.to_image(ox + tile_w, oy + tile_h, x_axis, y_axis),
        geo.to_image(ox, oy + tile_h, x_axis, y_axis),
    ]
    # Erase covers the original extent AND the (possibly shifted/larger) text tile.
    ex0 = min(xmin - pad, ox)
    ex1 = max(xmax + pad, ox + tile_w)
    ey1 = max(ymax + pad, oy + tile_h)
    erase_quad = [
        _ipoint(geo.to_image(ex0, oy, x_axis, y_axis)),
        _ipoint(geo.to_image(ex1, oy, x_axis, y_axis)),
        _ipoint(geo.to_image(ex1, ey1, x_axis, y_axis)),
        _ipoint(geo.to_image(ex0, ey1, x_axis, y_axis)),
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


def _line_boxes(members: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Cluster member bboxes into text lines (by vertical overlap); return each line's
    ``(left, right)``, top-sorted. OCR may give word- or line-level cells, so this
    recovers the real line structure used for both line-count and alignment.
    """
    boxes = [m["bbox"] for m in members if m.get("bbox")]
    lines: list[list[int]] = []  # [top, bottom, min_left, max_right]
    for box in sorted(boxes, key=lambda b: int(b.get("top") or 0)):
        top = int(box.get("top") or 0)
        bottom = top + int(box.get("height") or 0)
        left = int(box.get("left") or 0)
        right = left + int(box.get("width") or 0)
        for line in lines:
            overlap = min(bottom, line[1]) - max(top, line[0])
            if overlap > 0.5 * max(1, min(bottom - top, line[1] - line[0])):
                line[0], line[1] = min(line[0], top), max(line[1], bottom)
                line[2], line[3] = min(line[2], left), max(line[3], right)
                break
        else:
            lines.append([top, bottom, left, right])
    return [(line[2], line[3]) for line in lines]


def _alignment(line_boxes: list[tuple[int, int]]) -> str:
    """"center" if the line centres vary less than the line left edges, else "left".

    Right-aligned is rare for the text we re-place (prices are non-translatable) and its
    coincidental right-edge alignment misfired, so it is intentionally not detected.
    """
    if len(line_boxes) < 2:
        return "left"
    lefts = [left for left, _ in line_boxes]
    centers = [(left + right) / 2.0 for left, right in line_boxes]
    return "center" if (max(centers) - min(centers)) < (max(lefts) - min(lefts)) else "left"
