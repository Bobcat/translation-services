from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrSegment:
    text: str
    bbox: dict[str, int]
    confidence: float
    polygon: list[dict[str, int]] | None = None


def bbox_left(segment: OcrSegment) -> int:
    return int(segment.bbox.get("left") or 0)


def bbox_top(segment: OcrSegment) -> int:
    return int(segment.bbox.get("top") or 0)


def union_bbox(segments: list[OcrSegment]) -> dict[str, int]:
    left = min(int(segment.bbox["left"]) for segment in segments)
    top = min(int(segment.bbox["top"]) for segment in segments)
    right = max(int(segment.bbox["left"]) + int(segment.bbox["width"]) for segment in segments)
    bottom = max(int(segment.bbox["top"]) + int(segment.bbox["height"]) for segment in segments)
    return {
        "left": left,
        "top": top,
        "width": max(0, right - left),
        "height": max(0, bottom - top),
    }


def bbox_polygon(bbox: dict[str, int]) -> list[dict[str, int]]:
    left = int(bbox.get("left") or 0)
    top = int(bbox.get("top") or 0)
    right = left + int(bbox.get("width") or 0)
    bottom = top + int(bbox.get("height") or 0)
    return [
        {"x": left, "y": top},
        {"x": right, "y": top},
        {"x": right, "y": bottom},
        {"x": left, "y": bottom},
    ]
