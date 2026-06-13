"""Stage #8 re-placement — background-matched, polygon-aware (Tier-1, model-free).

Units that share a VLM block (a wrapped dish, a body paragraph) render as one
**group**: their translations are joined and balanced over the original number of
lines, at ONE font size taken from the original's true line height — the **source
size**, so a heading stays heading-sized and body stays body-sized (the source size
carries the visual hierarchy). Width is matched separately by **horizontal condensation**:
at the source height a translated line is usually wider than its original, so the rendered
text is squeezed in x to fit the original line's width (floored, never stretched) — keeping
height while matching width, the way the reference render does. Each rendered line anchors
on its original line's plane (so the line pitch follows the original). Per plane: cover
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

# Floor on horizontal condensation. The font is sized from the source HEIGHT (so the
# header/body hierarchy is preserved); a translated line at that height is usually wider
# than its original (most sans are wider than the sign's font), so the rendered text is
# squeezed horizontally to fit the original line's width — keeping height, matching width,
# the way the reference render does. Never squeeze past this floor: below it the glyphs
# read as unnaturally narrow, so the line is allowed to overrun the original width instead.
_CONDENSE_FLOOR = 0.75

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

    # The whole group renders at ONE size = the original's source size (true line height),
    # NOT a size chosen to fit the width. So a heading keeps heading size and body keeps
    # body size — the source size carries the hierarchy. The joined translation is balanced
    # over the original line count.
    # The units of a group share one VLM element, so one font family/weight. Take the first
    # that carries a hint (leftovers have none -> fall back to the default font).
    family = next((u.get("font_family") for u in units if u.get("font_family")), None)
    weight = next((u.get("font_weight") for u in units if u.get("font_weight")), None)
    joined = " ".join(texts)
    font, lines = _fit_group(
        joined,
        size=min(plane["target"] for plane in planes),
        n_lines=len(planes),
        max_line_w=max(plane["width"] for plane in planes),
        family=family,
        weight=weight,
    )
    ascent, descent = font.getmetrics()
    centered = any(str(unit.get("alignment") or "") == "center" for unit in units)

    # Width is matched by horizontal condensation, not by shrinking the font: at the source
    # size the translated line is usually wider than its original, so squeeze it in x to fit
    # the original line's width (floored at _CONDENSE_FLOOR). One factor for the whole group
    # keeps a multi-line block visually coherent; never stretch (cap at 1.0), so a shorter
    # line just stays narrower.
    condense = _condense_scale(font, lines, planes)

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
            # Draw the line at its natural width, then squeeze in x by ``condense`` — this
            # keeps the source height (hierarchy) while the line fits the original width.
            text_h = max(1, int(ascent + descent))
            text_w_nat = max(1, int(font.getlength(line)))
            text_img = Image.new("RGBA", (text_w_nat, text_h), (0, 0, 0, 0))
            ImageDraw.Draw(text_img).text((0, 0), line, font=font, fill=fg + (255,))
            text_w = max(1, int(round(text_w_nat * condense)))
            if text_w != text_w_nat:
                text_img = text_img.resize((text_w, text_h), Image.LANCZOS)
            tile_w = max(1, text_w + 2 * int(pad))
            tile_h = max(1, text_h + 2 * int(pad))
            ox = (xmin + xmax) / 2 - tile_w / 2 if centered else xmin - pad
            tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
            tile.paste(text_img, (int(pad), int(pad)))
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
    text: str,
    *,
    size: int,
    n_lines: int,
    max_line_w: float,
    family: str | None = None,
    weight: int | None = None,
) -> tuple[Any, list[str]]:
    """Render at the source ``size`` (true line height) in the unit's VLM font ``family`` /
    ``weight``, wrapped into as few lines as fit the column width (``max_line_w``), capped at
    ``n_lines``. The font is NOT reduced to fit width — the source size, and thus the
    header/body hierarchy, is preserved; width is matched by horizontal condensation in the
    caller."""
    font = load_font(max(6, min(int(size), 160)), text, family=family, weight=weight)
    return font, _wrap_balanced(font, text, n_lines, max_line_w)


def _condense_scale(font: Any, lines: list[str], planes: list[dict[str, Any]]) -> float:
    """Horizontal scale that squeezes the group's lines into their original widths.

    Per line: original plane width / the line's natural rendered width. The group takes the
    tightest line's factor (so a multi-line block condenses uniformly), clamped to
    [``_CONDENSE_FLOOR``, 1.0] — never stretch a short line, never squeeze past the floor."""
    factors: list[float] = []
    for index, line in enumerate(lines):
        if index >= len(planes) or not line:
            continue
        natural = font.getlength(line)
        if natural > 0:
            factors.append(planes[index]["width"] / natural)
    if not factors:
        return 1.0
    return max(_CONDENSE_FLOOR, min(1.0, min(factors)))


def _wrap_balanced(font: Any, text: str, n_lines: int, max_width: float) -> list[str]:
    """Wrap ``text`` into as FEW lines as fit the column width, capped at ``n_lines``.

    First a plain greedy wrap at the original column width (``max_width``): a more compact
    translation that fits in fewer lines than the original uses fewer — no empty spreading
    over the original line count. Only when the text still needs more than ``n_lines`` lines
    at that width do we pack it into exactly ``n_lines`` by the smallest balancing width (the
    caller then condenses horizontally to fit)."""
    content = str(text or "").strip()
    if n_lines <= 1 or not content:
        return [content]
    natural = wrap_lines(font, content, int(max_width))
    if len(natural) <= n_lines:
        return natural
    lo, hi = 1, int(font.getlength(content)) + 1
    best = wrap_lines(font, content, hi)
    while lo <= hi:
        mid = (lo + hi) // 2
        lines = wrap_lines(font, content, mid)
        if len(lines) <= n_lines:
            best = lines
            hi = mid - 1
        else:
            lo = mid + 1
    return best


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
