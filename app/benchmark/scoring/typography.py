"""The typography axis: stray target ink outside every region, and font-size ratios
drifting between matched text regions (readers notice broken ratios, not absolute sizes).
"""
from __future__ import annotations

import statistics
from typing import Any

from app.benchmark.scoring.matching import _family
from app.benchmark.scoring.matching import _segment_center


def _typography(
    src: dict[str, Any], tgt: dict[str, Any], matching: dict[str, Any]
) -> tuple[float, float, float]:
    src_regions = matching["source_regions"]
    tgt_regions = matching["target_regions"]
    tgt_segments = list(tgt.get("segments") or [])

    # Stray text: share of target OCR ink whose center falls outside every region.
    total_area = 0.0
    stray_area = 0.0
    for segment in tgt_segments:
        bbox = dict(segment.get("bbox") or {})
        area = float(bbox.get("width") or 0.0) * float(bbox.get("height") or 0.0)
        total_area += area
        center_x, center_y = _segment_center(segment)
        inside = any(
            region["box"][0] <= center_x <= region["box"][2]
            and region["box"][1] <= center_y <= region["box"][3]
            for region in tgt_regions
        )
        if not inside:
            stray_area += area
    stray_share = (stray_area / total_area) if total_area > 0 else 0.0

    # Size-ratio drift: per matched text-region pair, the median segment height
    # ratio target/source; drift = relative spread of those ratios. A uniform
    # scale (all regions x0.9) is fine; one region collapsing while its
    # siblings hold is what this catches.
    src_segments = list(src.get("segments") or [])
    ratios: list[float] = []
    for src_index, tgt_index, _iou_value in matching["matches"]:
        if _family(src_regions[src_index]["label"]) != "text":
            continue
        src_height = _median_segment_height(src_regions[src_index], src_segments)
        tgt_height = _median_segment_height(tgt_regions[tgt_index], tgt_segments)
        if src_height > 0 and tgt_height > 0:
            ratios.append(tgt_height / src_height)
    if len(ratios) >= 2:
        low, high = min(ratios), max(ratios)
        ratio_drift = (high / low) - 1.0 if low > 0 else 0.0
    else:
        ratio_drift = 0.0

    typography = 100.0 * max(0.0, 1.0 - min(1.0, 0.5 * stray_share + 0.5 * min(1.0, ratio_drift)))
    return typography, stray_share, ratio_drift


def _median_segment_height(region: dict[str, Any], segments: list[dict[str, Any]]) -> float:
    box = region["box"]
    heights = []
    for segment in segments:
        center_x, center_y = _segment_center(segment)
        if box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]:
            heights.append(float(dict(segment.get("bbox") or {}).get("height") or 0.0))
    return statistics.median(heights) if heights else 0.0
