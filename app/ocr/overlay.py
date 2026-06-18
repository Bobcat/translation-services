from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from app.ocr.segment import OcrSegment


@dataclass(frozen=True)
class ProjectedOverlayDebug:
    image: bytes | None
    metadata: dict[str, Any]
    mime_type: str = "image/png"


def render_original_ocr_overlay_debug(
    *,
    input_path: Path,
    ocr_segments: list[OcrSegment],
) -> ProjectedOverlayDebug:
    polygons = [
        _segment_points(segment)
        for segment in ocr_segments
        if str(segment.text or "").strip()
    ]
    polygons = [polygon for polygon in polygons if polygon]
    if not polygons:
        return _overlay_result(
            image=None,
            reason="no_original_ocr_segments",
            segment_count=0,
            out_of_bounds_count=0,
        )

    from PIL import Image
    from PIL import ImageDraw

    with Image.open(input_path) as original:
        image = original.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    out_of_bounds_count = 0
    for polygon in polygons:
        if not _polygon_intersects_image(polygon, image.size):
            out_of_bounds_count += 1
        draw.polygon(polygon, fill=(255, 255, 255, 150), outline=(255, 255, 255, 230))

    out = BytesIO()
    Image.alpha_composite(image, overlay).convert("RGB").save(out, format="PNG", compress_level=1)
    return _overlay_result(
        image=out.getvalue(),
        reason="original_ocr_overlay_debug_applied",
        segment_count=len(polygons),
        out_of_bounds_count=out_of_bounds_count,
    )


def _overlay_result(
    *,
    image: bytes | None,
    reason: str,
    segment_count: int,
    out_of_bounds_count: int,
) -> ProjectedOverlayDebug:
    return ProjectedOverlayDebug(
        image=image,
        metadata={
            "projected_overlay_debug_applied": image is not None,
            "projected_overlay_debug_reason": reason,
            "projected_overlay_debug_source": "original_ocr",
            "projected_overlay_debug_segment_count": int(segment_count),
            "projected_overlay_debug_out_of_bounds_count": int(out_of_bounds_count),
        },
    )


def _segment_points(segment: OcrSegment) -> list[tuple[float, float]]:
    polygon = _ordered_quad(segment.polygon or [])
    if polygon is not None:
        return polygon
    bbox = segment.bbox
    left = float(bbox.get("left") or 0.0)
    top = float(bbox.get("top") or 0.0)
    right = left + max(1.0, float(bbox.get("width") or 0.0))
    bottom = top + max(1.0, float(bbox.get("height") or 0.0))
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _ordered_quad(points: list[dict[str, int]] | list[dict[str, float]]) -> list[tuple[float, float]] | None:
    normalized = [
        (float(point.get("x", 0.0)), float(point.get("y", 0.0)))
        for point in points
        if isinstance(point, dict)
    ]
    if len(normalized) < 4:
        return None
    normalized = normalized[:4]
    top_left = min(normalized, key=lambda point: point[0] + point[1])
    bottom_right = max(normalized, key=lambda point: point[0] + point[1])
    top_right = max(normalized, key=lambda point: point[0] - point[1])
    bottom_left = min(normalized, key=lambda point: point[0] - point[1])
    ordered = [top_left, top_right, bottom_right, bottom_left]
    if len(set(ordered)) < 4:
        return None
    return ordered


def _polygon_intersects_image(polygon: list[tuple[float, float]], image_size: tuple[int, int]) -> bool:
    width, height = image_size
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return max(xs) >= 0 and min(xs) <= width and max(ys) >= 0 and min(ys) <= height
