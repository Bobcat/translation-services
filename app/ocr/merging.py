from __future__ import annotations

import math
from statistics import median

from app.ocr.segment import bbox_polygon
from app.ocr.segment import bbox_left
from app.ocr.segment import bbox_top
from app.ocr.segment import OcrSegment
from app.ocr.segment import union_bbox


def merge_same_line_segments(segments: list[OcrSegment]) -> list[OcrSegment]:
    if not segments:
        return []

    rows: list[list[OcrSegment]] = []
    for segment in sorted(segments, key=lambda item: (bbox_top(item), bbox_left(item))):
        target_row = next((row for row in rows if _belongs_to_visual_line(row, segment)), None)
        if target_row is None:
            rows.append([segment])
        else:
            target_row.append(segment)

    merged = [_merge_visual_line(row) for row in rows]
    return sorted(merged, key=lambda item: (bbox_top(item), bbox_left(item)))


def _belongs_to_visual_line(row: list[OcrSegment], segment: OcrSegment) -> bool:
    row_bbox = union_bbox(row)
    row_top = int(row_bbox["top"])
    row_bottom = row_top + int(row_bbox["height"])
    segment_top = bbox_top(segment)
    segment_bottom = segment_top + int(segment.bbox.get("height") or 0)
    row_height = max(1, row_bottom - row_top)
    segment_height = max(1, segment_bottom - segment_top)
    vertical_overlap = max(0, min(row_bottom, segment_bottom) - max(row_top, segment_top))
    overlap_ratio = vertical_overlap / max(1, min(row_height, segment_height))
    row_center = row_top + (row_height / 2.0)
    segment_center = segment_top + (segment_height / 2.0)
    center_delta = abs(row_center - segment_center)
    same_y = overlap_ratio >= 0.45 or center_delta <= max(8.0, max(row_height, segment_height) * 0.35)
    if not same_y:
        return False

    row_right = max(bbox_left(item) + int(item.bbox.get("width") or 0) for item in row)
    gap = bbox_left(segment) - row_right
    if gap <= 0:
        return True
    return gap <= max(20.0, max(row_height, segment_height) * 1.8)


def _merge_visual_line(row: list[OcrSegment]) -> OcrSegment:
    ordered = sorted(row, key=bbox_left)
    text = " ".join(str(segment.text or "").strip() for segment in ordered if str(segment.text or "").strip())
    confidence = sum(float(segment.confidence) for segment in ordered) / max(1, len(ordered))
    bbox = union_bbox(ordered)
    polygon = ordered[0].polygon if len(ordered) == 1 else _merged_polygon(ordered, bbox)
    return OcrSegment(text=text, bbox=bbox, confidence=confidence, polygon=polygon)


def _merged_polygon(row: list[OcrSegment], bbox: dict[str, int]) -> list[dict[str, int]]:
    polygons = [segment.polygon for segment in row if segment.polygon]
    if not polygons:
        return bbox_polygon(bbox)
    points: list[tuple[float, float]] = []
    angles: list[float] = []
    for polygon in polygons:
        ordered = _ordered_quad(polygon)
        if ordered is None:
            continue
        points.extend(ordered)
        angles.append(_angle(ordered[0], ordered[1]))
        angles.append(_angle(ordered[3], ordered[2]))
    if len(points) < 4 or not angles:
        return bbox_polygon(bbox)

    angle = float(median(sorted(_normalize_angle(value) for value in angles)))
    radians = math.radians(angle)
    x_axis = (math.cos(radians), math.sin(radians))
    y_axis = (-math.sin(radians), math.cos(radians))
    x_values = [_project(point, x_axis) for point in points]
    y_values = [_project(point, y_axis) for point in points]
    corners = [
        _from_projection(min(x_values), min(y_values), x_axis, y_axis),
        _from_projection(max(x_values), min(y_values), x_axis, y_axis),
        _from_projection(max(x_values), max(y_values), x_axis, y_axis),
        _from_projection(min(x_values), max(y_values), x_axis, y_axis),
    ]
    return [{"x": max(0, int(round(point[0]))), "y": max(0, int(round(point[1])))} for point in corners]


def _ordered_quad(points: list[dict[str, int]]) -> list[tuple[float, float]] | None:
    if len(points) < 4:
        return None
    normalized = [(float(point["x"]), float(point["y"])) for point in points[:4]]
    top_left = min(normalized, key=lambda point: point[0] + point[1])
    bottom_right = max(normalized, key=lambda point: point[0] + point[1])
    top_right = max(normalized, key=lambda point: point[0] - point[1])
    bottom_left = min(normalized, key=lambda point: point[0] - point[1])
    ordered = [top_left, top_right, bottom_right, bottom_left]
    if len(set(ordered)) < 4:
        return None
    return ordered


def _angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    return _normalize_angle(math.degrees(math.atan2(second[1] - first[1], second[0] - first[0])))


def _normalize_angle(angle: float) -> float:
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def _project(point: tuple[float, float], axis: tuple[float, float]) -> float:
    return (point[0] * axis[0]) + (point[1] * axis[1])


def _from_projection(
    x_value: float,
    y_value: float,
    x_axis: tuple[float, float],
    y_axis: tuple[float, float],
) -> tuple[float, float]:
    return (
        (x_axis[0] * x_value) + (y_axis[0] * y_value),
        (x_axis[1] * x_value) + (y_axis[1] * y_value),
    )
