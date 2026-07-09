"""Polygon geometry for re-placement: text angle and true (de-skewed) line height.

OCR gives each cell a rotated quad. Its axis-aligned bbox height is inflated by the
tilt (``h·cosθ + w·sinθ``), so sizing text by that height makes long/slanted lines
huge and short ones tiny even when the original glyphs are the same size. Working in
the text's own rotated frame recovers the **true line height** (consistent across
lines) and the **angle** (so the text can be warped to follow the slant).
"""
from __future__ import annotations

import math
from typing import Any

Point = tuple[float, float]


def quad_of(member: dict[str, Any]) -> list[Point] | None:
    """Ordered [TL, TR, BR, BL] from a member's polygon, falling back to its bbox."""
    quad = _ordered(member.get("polygon"))
    if quad is not None:
        return quad
    return _ordered(_bbox_points(member.get("bbox") or {}))


def angle_deg(quad: list[Point]) -> float:
    """Text-line angle in degrees from the top and bottom edges (mean), normalised."""
    top = math.degrees(math.atan2(quad[1][1] - quad[0][1], quad[1][0] - quad[0][0]))
    bottom = math.degrees(math.atan2(quad[2][1] - quad[3][1], quad[2][0] - quad[3][0]))
    return _normalize(0.5 * (_normalize(top) + _normalize(bottom)))


def line_height(quad: list[Point]) -> float:
    """True line height: mean of the two side edges (TL-BL, TR-BR), tilt-invariant."""
    return 0.5 * (_dist(quad[0], quad[3]) + _dist(quad[1], quad[2]))


def oriented_frame(quads: list[list[Point]], angle: float) -> tuple[Point, Point, float, float, float, float]:
    """Project all quad points onto axes rotated by ``angle``.

    Returns ``(x_axis, y_axis, xmin, xmax, ymin, ymax)`` — the unit's oriented
    bounding box in the rotated frame. Map a rotated-frame point ``(px, py)`` back
    to image space with :func:`to_image` (the axes are orthonormal).
    """
    rad = math.radians(angle)
    x_axis = (math.cos(rad), math.sin(rad))
    y_axis = (-math.sin(rad), math.cos(rad))
    pts = [p for quad in quads for p in quad]
    xs = [p[0] * x_axis[0] + p[1] * x_axis[1] for p in pts]
    ys = [p[0] * y_axis[0] + p[1] * y_axis[1] for p in pts]
    return x_axis, y_axis, min(xs), max(xs), min(ys), max(ys)


def to_image(px: float, py: float, x_axis: Point, y_axis: Point) -> Point:
    return (px * x_axis[0] + py * y_axis[0], px * x_axis[1] + py * y_axis[1])


def _ordered(points: Any) -> list[Point] | None:
    if not points or len(points) < 4:
        return None
    pts = [(float(p["x"]), float(p["y"])) for p in points[:4]]
    top_left = min(pts, key=lambda p: p[0] + p[1])
    bottom_right = max(pts, key=lambda p: p[0] + p[1])
    top_right = max(pts, key=lambda p: p[0] - p[1])
    bottom_left = min(pts, key=lambda p: p[0] - p[1])
    quad = [top_left, top_right, bottom_right, bottom_left]
    if len({quad[i] for i in range(4)}) < 4:
        return None
    return quad


def _bbox_points(bbox: dict[str, Any]) -> list[dict[str, int]] | None:
    width = int(bbox.get("width") or 0)
    height = int(bbox.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    left = int(bbox.get("left") or 0)
    top = int(bbox.get("top") or 0)
    right, bottom = left + width, top + height
    return [{"x": left, "y": top}, {"x": right, "y": top}, {"x": right, "y": bottom}, {"x": left, "y": bottom}]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _normalize(angle: float) -> float:
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


# Below this group angle (degrees) the text is treated as horizontal and placed axis-aligned,
# so OCR detection noise on a flat image isn't warped into a visible slant. A genuine page
# tilt is well above it (a photographed menu card sits at ~6°), so real perspective is kept.
_ANGLE_DEADZONE_DEG = 3.0

def _plane_corners(plane: dict[str, Any]) -> list[tuple[float, float]]:
    """The plane's oriented bounding box as four image-space corners [TL, TR, BR, BL].
    Sampling background from this (deskewed) region instead of the axis-aligned bbox keeps
    the border ring inside a tilted coloured band — on a slanted line the axis box's corners
    reach into the surroundings (a sign's panel behind a diagonal bar) and muddy the sample."""
    x_axis, y_axis, xmin, xmax, ymin, ymax = plane["frame"]
    return [
        to_image(xmin, ymin, x_axis, y_axis),
        to_image(xmax, ymin, x_axis, y_axis),
        to_image(xmax, ymax, x_axis, y_axis),
        to_image(xmin, ymax, x_axis, y_axis),
    ]
