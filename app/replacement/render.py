"""Stage #8 re-placement — background-matched, polygon-aware (Tier-1, model-free).

Units that share a VLM block (a wrapped dish, a body paragraph) render as one
**group**: their translations are joined and re-broken freely over at most the
original number of lines, at ONE font size decided up front from the group's total
ink width — slack on a short line is pooled into the next instead of shrinking that
line alone. Each rendered line anchors on its original line's plane (so the line
pitch follows the original), capped at the group's widest plane. Per plane: cover
the original with the locally-sampled **background colour** (so it reads as erased
on a flat surface — menu paper, sign panel, receipt), then draw the line and **warp
it onto the plane's polygon** so it follows the page tilt (rotation/perspective),
for a clean camera-translation look.

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
from app.replacement.color import contrasting_fg
from app.replacement.color import sample_region_colors
from app.replacement.fit import load_font
from app.replacement.fit import wrap_lines


# Font size from the true (de-skewed) line height. The polygon height spans the full
# glyph extent, a touch taller than the visual cap; scale down slightly to match.
_SIZE_RATIO = 0.9

# Per-channel tolerance for snapping a group's per-plane background samples to one
# colour: within it the planes are one surface sampled with texture noise; beyond it
# they are genuinely different (a gradient, two panels) and stay per-plane.
_BG_SNAP_DELTA = 24


@dataclass(frozen=True)
class _Job:
    erase_quad: list[tuple[int, int]]
    bg_color: tuple[int, int, int]
    # None for an erase-only plane (the translation needed fewer lines than the original).
    tile: Image.Image | None
    dst_quad: list[tuple[float, float]] | None


def render_translated_image(input_path: Path, translation_units: list[dict[str, Any]]) -> bytes:
    base = Image.open(input_path).convert("RGB")

    jobs: list[_Job] = []
    for group in _groups(translation_units):
        jobs.extend(_plan_group(base, group))

    # Pass 1: cover every original (along the slant) so no source text peeks through.
    erase = ImageDraw.Draw(base)
    for job in jobs:
        erase.polygon(job.erase_quad, fill=job.bg_color)

    # Pass 2: warp each text tile onto its oriented region.
    canvas = np.asarray(base).copy()
    for job in jobs:
        if job.tile is not None:
            _composite(canvas, job)

    out = BytesIO()
    Image.fromarray(canvas).save(out, format="PNG")
    return out.getvalue()


def _groups(units: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Consecutive units of one VLM block at one level reflow together — a wrapped
    dish, a body paragraph. The level guard keeps a heading from merging into its
    body text. Leftovers (no block — an OCR noise cell interleaved in reading order)
    stay alone but do NOT break the surrounding block's run, or one stray cell would
    split a dish back into per-line fitting."""
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    previous: tuple[Any, Any] | None = None
    for unit in units:
        key = (unit.get("block_id"), unit.get("level"))
        if key[0] is None:
            groups.append([unit])
            continue
        if current is not None and key == previous:
            current.append(unit)
        else:
            current = [unit]
            groups.append(current)
        previous = key
    return groups


def _plan_group(base: Image.Image, units: list[dict[str, Any]]) -> list[_Job]:
    texts: list[str] = []
    group_quads: list = []
    for unit in units:
        translated = str(unit.get("translated_text") or "").strip()
        if len(translated) <= 1:  # empty / OCR-noise single char -> leave the original alone
            continue
        members = [m for m in (unit.get("members") or []) if m.get("translate") and m.get("bbox")]
        quads = [quad for quad in (geo.quad_of(m) for m in members) if quad is not None]
        if not quads:
            continue
        texts.append(translated)
        group_quads.extend(quads)
    if not texts:
        return []

    # Planes come from geometry, not from the unit shape: cluster the group's member
    # quads into physical text lines. An element-level hint yields ONE unit spanning
    # several printed lines; a per-line hint yields one unit per line — both cluster
    # to the same planes.
    angle = median(geo.angle_deg(quad) for quad in group_quads)
    planes: list[dict[str, Any]] = []
    for quads in _line_clusters(group_quads, angle):
        true_height = median(geo.line_height(quad) for quad in quads)
        x_axis, y_axis, xmin, xmax, ymin, ymax = geo.oriented_frame(quads, angle)
        planes.append({
            "quads": quads,
            "target": max(8, int(true_height * _SIZE_RATIO)),
            "pad": max(2.0, true_height / 6.0),
            "frame": (x_axis, y_axis, xmin, xmax, ymin, ymax),
            "width": xmax - xmin,
        })

    # The whole group renders at ONE size — never above the original — decided up front
    # from the group's total ink width; the joined translation is re-broken freely over
    # at most the original line count, every line capped at the group's WIDEST plane.
    # Slack on a short original line is pooled into the next line instead of that line
    # shrinking alone ("jus," may move up behind "red wine").
    joined = " ".join(texts)
    fitted = _fit_group(
        joined,
        start=min(plane["target"] for plane in planes),
        max_line_w=max(plane["width"] for plane in planes),
        budget=sum(plane["width"] for plane in planes),
        n_lines=len(planes),
    )
    if fitted is None:  # not even the smallest font packs -> leave the original alone
        return []
    font, lines = fitted
    ascent, descent = font.getmetrics()
    centered = any(str(unit.get("alignment") or "") == "center" for unit in units)

    # One element usually sits on one surface: when the per-plane background samples
    # are near-equal (texture noise), snap them to their median so the erase planes
    # don't show slightly different shades per line.
    colors = [sample_region_colors(base, geo.axis_bbox(plane["quads"])) for plane in planes]
    if len(colors) > 1:
        median_bg = tuple(int(median(bg[channel] for bg, _ in colors)) for channel in range(3))
        if all(max(abs(bg[c] - median_bg[c]) for c in range(3)) <= _BG_SNAP_DELTA for bg, _ in colors):
            colors = [(median_bg, contrasting_fg(median_bg))] * len(colors)

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
        ex0, ex1, ey1 = xmin - pad, xmax + pad, ymax + pad
        tile: Image.Image | None = None
        dst_quad: list[tuple[float, float]] | None = None
        line = lines[index] if index < len(lines) else None  # extra planes: erase only
        if line:
            text_w = int(font.getlength(line))
            tile_w = max(1, text_w + 2 * int(pad))
            tile_h = max(1, int(ascent + descent) + 2 * int(pad))
            ox = (xmin + xmax) / 2 - tile_w / 2 if centered else xmin - pad
            tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((int(pad), int(pad)), line, font=font, fill=fg + (255,))
            dst_quad = [
                geo.to_image(ox, oy, x_axis, y_axis),
                geo.to_image(ox + tile_w, oy, x_axis, y_axis),
                geo.to_image(ox + tile_w, oy + tile_h, x_axis, y_axis),
                geo.to_image(ox, oy + tile_h, x_axis, y_axis),
            ]
            # Erase the original extent, grown to cover the (possibly wider) line.
            ex0 = min(ex0, ox)
            ex1 = max(ex1, ox + tile_w)
            ey1 = max(ey1, oy + tile_h)
        erase_quad = [
            _ipoint(geo.to_image(ex0, oy, x_axis, y_axis)),
            _ipoint(geo.to_image(ex1, oy, x_axis, y_axis)),
            _ipoint(geo.to_image(ex1, ey1, x_axis, y_axis)),
            _ipoint(geo.to_image(ex0, ey1, x_axis, y_axis)),
        ]
        jobs.append(_Job(erase_quad=erase_quad, bg_color=bg, tile=tile, dst_quad=dst_quad))
    return jobs


def _line_clusters(quads: list, angle: float) -> list[list]:
    """Cluster member quads into physical text lines (top to bottom in the oriented
    frame): a quad whose vertical centre falls inside the running cluster's extent is
    on the same line; line pitch puts the next line's centre below it."""
    measured = []
    for quad in quads:
        _, _, _, _, oymin, oymax = geo.oriented_frame([quad], angle)
        measured.append((oymin, oymax, quad))
    measured.sort(key=lambda item: (item[0] + item[1]) / 2)
    clusters: list[list] = []
    extent_max = float("-inf")
    for oymin, oymax, quad in measured:
        center = (oymin + oymax) / 2
        if clusters and center <= extent_max:
            clusters[-1].append(quad)
            extent_max = max(extent_max, oymax)
        else:
            clusters.append([quad])
            extent_max = oymax
    return clusters


def _fit_group(
    text: str, *, start: int, max_line_w: float, budget: float, n_lines: int
) -> tuple[Any, list[str]] | None:
    """Largest size (from ``start`` down) whose text fits the group's total ink width
    AND re-breaks into at most the original line count under the per-line cap. None
    when even the smallest font cannot pack it (the caller leaves the original)."""
    for size in range(max(6, min(start, 160)), 5, -1):
        font = load_font(size, text)
        if font.getlength(text) > budget:
            continue
        lines = wrap_lines(font, text, int(max_line_w))
        if len(lines) <= n_lines and all(font.getlength(line) <= max_line_w for line in lines):
            return font, lines
    return None


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
