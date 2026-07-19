"""The text-fate split behind the unchanged/missing indicators: every eligible source
segment is changed (translated), unchanged (verbatim) or missing — observations, never
intent. Region-coupled by construction (missing rides on the matching), which is why the
retention axis it once fed was retired; see docs/benchmark-method.md.
"""
from __future__ import annotations

import re
from typing import Any

from app.benchmark.scoring.matching import _area
from app.benchmark.scoring.matching import _family
from app.benchmark.scoring.matching import _segment_center


# Segments shorter than this (in letters) are too ambiguous for the text-fate split.
_ELIGIBLE_MIN_LETTERS = 4


_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


_URL_RE = re.compile(r"(https?://|www\.|@)", re.IGNORECASE)


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
