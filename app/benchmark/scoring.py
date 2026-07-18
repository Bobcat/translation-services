"""Scoring layer: measurement dict -> per-axis scores + flags. Pure code, no models.

``score_measurement`` is the only entry point; everything it derives is a pure
function of the measurement, so any scoring change can re-score stored history
(``scripts/benchmark.py rescore``). Every raw quantity an axis is computed from
is also carried in the output, so a later scoring version can redefine an axis
without re-measuring.

Axes are named after OBSERVATIONS, never intent. Every eligible source segment
lands in exactly one of three observable states (shares sum to 100%):
- changed     present in the output, different from the source (~translated)
- unchanged   present, verbatim the source. Deliberate keep or missed
              translation — indistinguishable from the pair alone, so this is
              an INDICATOR, not a scored axis. The deliberate-keep set is a
              property of the document (every correct system keeps the same
              proper nouns), so the cross-system unchanged-delta on the same
              document is the actionable signal. Whether a keep was right
              belongs to an LLM/human judge, outside this scoring.
- missing     gone from the output. Wrong under every interpretation.

Axes (0-100; corners anchored by the identity baseline, see the design doc):
- layout      class-family-aware region matching between source and
              translated pages; sqrt-area-weighted IoU with truly lost and
              invented regions counting 0. Aggregated over the document by
              region weight, not per page — a two-region cover page no longer
              outweighs a fifty-region spread.
- retention   100 - missing share. Identity anchors at 100.
- typography  penalizes stray text (target OCR ink outside every detected
              region) and non-uniform font-size ratios across matched text
              regions (readers notice broken ratios, not absolute sizes).

Matching (v3) separates real geometry change from detector granularity:
- near-identical same-family detections on one side are deduped (the detector
  sometimes reports a region twice);
- after 1:1 greedy matching, an unmatched region whose area is largely covered
  by same-family regions on the OTHER side is "covered" — a split/merge or
  nested-detection artifact, excluded from the layout score entirely (neither
  penalized nor credited). Only regions with no counterpart coverage count as
  lost (source side) or invented (translated side).
A detector miss on one side (content visibly present but no region reported)
still counts as lost/invented — that is measurement noise the weighting can
only dampen, named in the design doc.

Flags (hard, not scores): page count, image/chart and table region counts.
"""
from __future__ import annotations

import math
import re
import statistics
from typing import Any

from app.grouping.layout import PRESERVE_LABELS
from app.grouping.layout import STRUCTURE_LABELS
from app.grouping.layout import TEXT_LABELS

SCORING_VERSION = 4

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
# Segments shorter than this (in letters) are too ambiguous for the text-fate split.
_ELIGIBLE_MIN_LETTERS = 4

_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
_URL_RE = re.compile(r"(https?://|www\.|@)", re.IGNORECASE)


def score_measurement(measurement: dict[str, Any]) -> dict[str, Any]:
    source_pages = list((measurement.get("source") or {}).get("pages") or [])
    target_pages = list((measurement.get("translated") or {}).get("pages") or [])
    paired = list(zip(source_pages, target_pages))

    per_page: list[dict[str, Any]] = []
    layout_numerator = 0.0
    layout_denominator = 0.0
    for src, tgt in paired:
        page = _score_page(src, tgt)
        layout_numerator += page.pop("_layout_numerator")
        layout_denominator += page.pop("_layout_denominator")
        per_page.append(page)

    axes = {
        "layout": 100.0 if layout_denominator <= 0 else 100.0 * layout_numerator / layout_denominator,
        "retention": _mean([page["retention"] for page in per_page]),
        "typography": _mean([page["typography"] for page in per_page]),
    }
    # Document-level text-fate totals (segment-weighted, unlike the page-mean axes).
    eligible = sum(page["raw"]["eligible_source_segments"] for page in per_page)
    unchanged = sum(page["raw"]["unchanged_segments"] for page in per_page)
    missing = sum(page["raw"]["missing_segments"] for page in per_page)
    indicators = {
        "eligible_segments": eligible,
        "changed_segments": max(0, eligible - unchanged - missing),
        "unchanged_segments": unchanged,
        "missing_segments": missing,
        "unchanged_share": round(100.0 * unchanged / eligible, 2) if eligible else 0.0,
        "missing_share": round(100.0 * missing / eligible, 2) if eligible else 0.0,
    }
    flags = {
        "page_count_equal": len(source_pages) == len(target_pages),
        "page_count_source": len(source_pages),
        "page_count_translated": len(target_pages),
        "image_regions_source": _region_count(source_pages, PRESERVE_LABELS),
        "image_regions_translated": _region_count(target_pages, PRESERVE_LABELS),
        "table_regions_source": _region_count(source_pages, STRUCTURE_LABELS),
        "table_regions_translated": _region_count(target_pages, STRUCTURE_LABELS),
    }
    flags["image_regions_equal"] = flags["image_regions_source"] == flags["image_regions_translated"]
    flags["table_regions_equal"] = flags["table_regions_source"] == flags["table_regions_translated"]
    return {
        "scoring_version": SCORING_VERSION,
        "axes": {key: round(value, 2) for key, value in axes.items()},
        "indicators": indicators,
        "flags": flags,
        "per_page": per_page,
    }


def _score_page(src: dict[str, Any], tgt: dict[str, Any]) -> dict[str, Any]:
    matching = _match_page(src, tgt)
    src_regions = matching["source_regions"]
    tgt_regions = matching["target_regions"]
    matches = matching["matches"]

    # Layout: sqrt-area-weighted. Matched source regions contribute IoU x weight,
    # truly lost source regions and invented target regions contribute 0 at their
    # weight; covered regions are excluded (granularity, not geometry change).
    numerator = sum(iou * _weight(src_regions[i]) for i, _j, iou in matches)
    denominator = (
        sum(_weight(src_regions[i]) for i, _j, _iou in matches)
        + sum(_weight(src_regions[i]) for i in matching["lost"])
        + sum(_weight(tgt_regions[j]) for j in matching["invented"])
    )
    layout = 100.0 if denominator <= 0 else 100.0 * numerator / denominator

    retention, unchanged_count, missing_count, eligible_count = _text_fate(src, tgt, matching)
    typography, stray_share, ratio_drift = _typography(src, tgt, matching)

    return {
        "page": int(src.get("index", 0)) + 1,
        "layout": round(layout, 2),
        "retention": round(retention, 2),
        "typography": round(typography, 2),
        "raw": {
            "regions_source": len(src_regions),
            "regions_translated": len(tgt_regions),
            "regions_matched": len(matches),
            "regions_lost": len(matching["lost"]),
            "regions_invented": len(matching["invented"]),
            "regions_covered_source": len(matching["covered_source"]),
            "regions_covered_translated": len(matching["covered_translated"]),
            "mean_matched_iou": round(_mean([iou for _i, _j, iou in matches]), 4),
            "eligible_source_segments": eligible_count,
            "changed_segments": max(0, eligible_count - unchanged_count - missing_count),
            "unchanged_segments": unchanged_count,
            "missing_segments": missing_count,
            "stray_text_share": round(stray_share, 4),
            "size_ratio_drift": round(ratio_drift, 4),
        },
        "_layout_numerator": numerator,
        "_layout_denominator": denominator,
    }


# --- matching ----------------------------------------------------------------


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


# --- text fate (changed | unchanged | missing) ------------------------------


def _text_fate(
    src: dict[str, Any],
    tgt: dict[str, Any],
    matching: dict[str, Any],
) -> tuple[float, int, int, int]:
    """Returns (retention, unchanged_count, missing_count, eligible_count)."""
    src_regions = matching["source_regions"]
    tgt_regions = matching["target_regions"]
    src_segments = list(src.get("segments") or [])
    tgt_segments = list(tgt.get("segments") or [])

    eligible = [seg for seg in src_segments if _eligible_segment(str(seg.get("text") or ""))]
    eligible_keys = {_normalize(str(seg.get("text") or "")) for seg in eligible}

    # Unchanged: target segments whose normalized text verbatim-matches an
    # eligible source segment. Each source key counts at most once. Deliberate
    # keep or missed translation — undecidable here, reported as an indicator.
    unchanged_keys = {
        _normalize(str(seg.get("text") or ""))
        for seg in tgt_segments
        if _eligible_segment(str(seg.get("text") or ""))
    } & eligible_keys
    unchanged_count = len(unchanged_keys)

    # Missing: eligible source segments inside a text region that is truly lost,
    # or whose matched counterpart carries no text at all. Covered regions are
    # granularity artifacts: their area exists on the other side, so their
    # segments are not counted missing here (the unchanged/changed split still
    # sees their text via OCR).
    matched_tgt_by_src = {i: j for i, j, _iou_value in matching["matches"]}
    lost_indices = set(matching["lost"])
    missing_count = 0
    for seg in eligible:
        region_index = _containing_region_index(seg, src_regions, family="text")
        if region_index is None:
            continue
        if region_index in lost_indices:
            missing_count += 1
            continue
        target_index = matched_tgt_by_src.get(region_index)
        if target_index is not None and not _region_has_text(tgt_regions[target_index], tgt_segments):
            missing_count += 1

    eligible_count = len(eligible_keys)
    if eligible_count == 0:
        return 100.0, unchanged_count, missing_count, 0
    retention = 100.0 * max(0.0, 1.0 - missing_count / eligible_count)
    return retention, unchanged_count, missing_count, eligible_count


def _eligible_segment(text: str) -> bool:
    """Alphabetic enough to be translatable prose; skips the deliberately
    preserved classes (numbers, prices, URLs, short codes)."""
    stripped = text.strip()
    if not stripped or _URL_RE.search(stripped):
        return False
    words = _WORD_RE.findall(stripped)
    return sum(len(word) for word in words) >= _ELIGIBLE_MIN_LETTERS


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold()))


def _containing_region_index(
    segment: dict[str, Any], regions: list[dict[str, Any]], *, family: str
) -> int | None:
    center_x, center_y = _segment_center(segment)
    best: tuple[float, int] | None = None
    for index, region in enumerate(regions):
        if _family(region["label"]) != family:
            continue
        box = region["box"]
        if box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]:
            area = _area(box)
            if best is None or area < best[0]:  # smallest containing region wins
                best = (area, index)
    return best[1] if best else None


def _region_has_text(region: dict[str, Any], segments: list[dict[str, Any]]) -> bool:
    box = region["box"]
    for segment in segments:
        center_x, center_y = _segment_center(segment)
        if box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]:
            return True
    return False


def _segment_center(segment: dict[str, Any]) -> tuple[float, float]:
    bbox = dict(segment.get("bbox") or {})
    left = float(bbox.get("left") or 0.0)
    top = float(bbox.get("top") or 0.0)
    return left + float(bbox.get("width") or 0.0) / 2.0, top + float(bbox.get("height") or 0.0) / 2.0


# --- typography ------------------------------------------------------------


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


def _region_count(pages: list[dict[str, Any]], labels: set[str]) -> int:
    return sum(
        1
        for page in pages
        for region in _usable_regions(page)
        if region["label"] in labels and region["score"] >= _FLAG_MIN_SCORE
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 100.0
