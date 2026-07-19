"""Region matching between the two sides of a pair: the geometry behind the layout axis,
the per-page overlays and the region-gated indicators.

Dedupe -> greedy same-family 1:1 on IoU -> coverage classification of the unmatched rest
(splits/merges/nested detections become "covered": granularity, not geometry change). Also
home to the small box/segment geometry helpers the sibling modules share.
"""
from __future__ import annotations

import math
from typing import Any

from app.layout import PRESERVE_LABELS
from app.layout import STRUCTURE_LABELS
from app.layout import TEXT_LABELS


# Regions below this detector confidence are noise for measurement purposes. 0.4 matches the
# measurement-layer detection threshold (app.benchmark.measurement.MEASURE_LAYOUT_THRESHOLD):
# on a perturbed render the model splits one region's confidence over competing classes,
# dropping every candidate under the old 0.5 floor — see the detector appendix in
# docs/pdf-benchmark-regression-design.md.
_REGION_MIN_SCORE = 0.4


# The structure FLAGS keep the old floor: they are hard did-a-picture/table-disappear signals,
# and the 0.4-0.5 band is exactly where the detector is unstable (a crest inside a matched
# header detected as a small image on one side only would flip a flag spuriously).
_FLAG_MIN_SCORE = 0.5


# Minimum IoU for two same-family regions to count as the same region.
_MATCH_MIN_IOU = 0.1


# Same-side same-family regions above this IoU are one detection reported twice.
_DUPLICATE_IOU = 0.85


# An unmatched region covered at least this much by the other side's same-family
# regions is detector granularity (split/merge/nested), not a layout change.
_COVERED_FRACTION = 0.8


def _match_page(src_page: dict[str, Any], tgt_page: dict[str, Any]) -> dict[str, Any]:
    """The one matching used by the layout axis, the overlays and retention:
    dedupe -> greedy 1:1 -> coverage classification of the unmatched rest."""
    src_regions = _prepare_regions(src_page)
    tgt_regions = _prepare_regions(tgt_page)
    matches, unmatched_src, unmatched_tgt = _match_regions(src_regions, tgt_regions)
    covered_src = sorted(
        i for i in unmatched_src if _covered_fraction(src_regions[i], tgt_regions) >= _COVERED_FRACTION
    )
    covered_tgt = sorted(
        j for j in unmatched_tgt if _covered_fraction(tgt_regions[j], src_regions) >= _COVERED_FRACTION
    )
    return {
        "source_regions": src_regions,
        "target_regions": tgt_regions,
        "matches": matches,
        "covered_source": covered_src,
        "lost": [i for i in unmatched_src if i not in covered_src],
        "covered_translated": covered_tgt,
        "invented": [j for j in unmatched_tgt if j not in covered_tgt],
    }


def page_region_statuses(src_page: dict[str, Any], tgt_page: dict[str, Any]) -> dict[str, Any]:
    """Per-region match status for one page pair, for the overlay renderer:
    ``{"source": [{label, box, status, iou}], "translated": […]}`` with status
    matched | covered | lost (source side) | invented (translated side)."""
    matching = _match_page(src_page, tgt_page)
    iou_by_src = {i: iou for i, _j, iou in matching["matches"]}
    iou_by_tgt = {j: iou for _i, j, iou in matching["matches"]}

    def status_for(index: int, covered: list[int], unmatched_status: str) -> str:
        if index in covered:
            return "covered"
        return unmatched_status

    return {
        "source": [
            {
                "label": region["label"],
                "box": region["box"],
                "status": "matched" if index in iou_by_src
                else status_for(index, matching["covered_source"], "lost"),
                "iou": round(iou_by_src[index], 3) if index in iou_by_src else None,
            }
            for index, region in enumerate(matching["source_regions"])
        ],
        "translated": [
            {
                "label": region["label"],
                "box": region["box"],
                "status": "matched" if index in iou_by_tgt
                else status_for(index, matching["covered_translated"], "invented"),
                "iou": round(iou_by_tgt[index], 3) if index in iou_by_tgt else None,
            }
            for index, region in enumerate(matching["target_regions"])
        ],
    }


def _prepare_regions(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Usable regions with near-identical same-family duplicates dropped
    (the detector sometimes reports one region twice; keep the confident one)."""
    regions = sorted(_usable_regions(page), key=lambda region: -region["score"])
    kept: list[dict[str, Any]] = []
    for region in regions:
        duplicate = any(
            _family(existing["label"]) == _family(region["label"])
            and _iou(existing["box"], region["box"]) >= _DUPLICATE_IOU
            for existing in kept
        )
        if not duplicate:
            kept.append(region)
    return kept


def _usable_regions(page: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for region in page.get("regions") or []:
        score = float(region.get("score") or 0.0)
        if score < _REGION_MIN_SCORE:
            continue
        coordinate = list(region.get("coordinate") or [])
        if len(coordinate) < 4:
            continue
        out.append(
            {
                "label": str(region.get("label") or ""),
                "score": score,
                "box": [float(v) for v in coordinate[:4]],
            }
        )
    return out


def _family(label: str) -> str:
    if label in PRESERVE_LABELS:
        return "image"
    if label in STRUCTURE_LABELS:
        return "table"
    if label in TEXT_LABELS:
        return "text"
    return "other"


def _match_regions(
    src_regions: list[dict[str, Any]], tgt_regions: list[dict[str, Any]]
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy same-family matching on descending IoU. Returns (matches as
    (src_idx, tgt_idx, iou), unmatched src indices, unmatched tgt indices)."""
    candidates: list[tuple[float, int, int]] = []
    for i, src in enumerate(src_regions):
        for j, tgt in enumerate(tgt_regions):
            if _family(src["label"]) != _family(tgt["label"]):
                continue
            iou = _iou(src["box"], tgt["box"])
            if iou >= _MATCH_MIN_IOU:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    matches: list[tuple[int, int, float]] = []
    used_src: set[int] = set()
    used_tgt: set[int] = set()
    for iou, i, j in candidates:
        if i in used_src or j in used_tgt:
            continue
        used_src.add(i)
        used_tgt.add(j)
        matches.append((i, j, iou))
    unmatched_src = [i for i in range(len(src_regions)) if i not in used_src]
    unmatched_tgt = [j for j in range(len(tgt_regions)) if j not in used_tgt]
    return matches, unmatched_src, unmatched_tgt


def _covered_fraction(region: dict[str, Any], others: list[dict[str, Any]]) -> float:
    family = _family(region["label"])
    boxes = [other["box"] for other in others if _family(other["label"]) == family]
    area = _area(region["box"])
    if area <= 0:
        return 0.0
    return _union_intersection_area(region["box"], boxes) / area


def _union_intersection_area(box: list[float], boxes: list[list[float]]) -> float:
    """Area of ``box`` covered by the union of ``boxes`` (coordinate compression;
    the handful of regions per page keeps the grid tiny)."""
    clipped = []
    for other in boxes:
        left, top = max(box[0], other[0]), max(box[1], other[1])
        right, bottom = min(box[2], other[2]), min(box[3], other[3])
        if right > left and bottom > top:
            clipped.append((left, top, right, bottom))
    if not clipped:
        return 0.0
    xs = sorted({c[0] for c in clipped} | {c[2] for c in clipped})
    ys = sorted({c[1] for c in clipped} | {c[3] for c in clipped})
    total = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            center_x = (xs[i] + xs[i + 1]) / 2.0
            center_y = (ys[j] + ys[j + 1]) / 2.0
            if any(c[0] <= center_x <= c[2] and c[1] <= center_y <= c[3] for c in clipped):
                total += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return total


def _weight(region: dict[str, Any]) -> float:
    """sqrt(area): big structures matter more than captions, but not linearly —
    a full-page image must not drown out every text line on the page."""
    return math.sqrt(max(0.0, _area(region["box"])))


def _iou(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _segment_center(segment: dict[str, Any]) -> tuple[float, float]:
    bbox = dict(segment.get("bbox") or {})
    left = float(bbox.get("left") or 0.0)
    top = float(bbox.get("top") or 0.0)
    return left + float(bbox.get("width") or 0.0) / 2.0, top + float(bbox.get("height") or 0.0) / 2.0
