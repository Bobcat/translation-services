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
- anchors     share of the source's translation-INVARIANT content still
              present in the translation: digit anchors (NFKC-normalized,
              grouping/decimal separators stripped, digit runs of >=2),
              matched as a DOCUMENT-wide multiset so reflow across page
              boundaries does not count as loss. Detector-free and
              translation-style-free — the direct text-survival signal.
- typography  penalizes stray text (target OCR ink outside every detected
              region) and non-uniform font-size ratios across matched text
              regions (readers notice broken ratios, not absolute sizes).

Retention as an axis (100 - missing share) was RETIRED in v5: it rode on the
same region matching as layout, so a re-typesetting system's reflow counted as
text loss while the text was demonstrably present (see the detector appendix in
the design doc). It survives as the ``region_retention`` indicator; the
``volume_ratio`` indicator (script-aware text units, translated/source) is the
coarse loss backstop for prose without anchors — compare it across systems on
the same document, where language-pair inflation is constant.

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

import statistics
from typing import Any

from app.benchmark.scoring.anchors import _match_anchors
from app.benchmark.scoring.anchors import _volume_units
from app.benchmark.scoring.anchors import anchor_details
from app.benchmark.scoring.matching import _FLAG_MIN_SCORE
from app.benchmark.scoring.matching import _match_page
from app.benchmark.scoring.matching import _usable_regions
from app.benchmark.scoring.matching import _weight
from app.benchmark.scoring.matching import page_region_statuses
from app.benchmark.scoring.textfate import _text_fate
from app.benchmark.scoring.typography import _typography
from app.layout import PRESERVE_LABELS
from app.layout import STRUCTURE_LABELS

__all__ = ["SCORING_VERSION", "score_measurement", "anchor_details", "page_region_statuses"]

SCORING_VERSION = 5


def score_measurement(measurement: dict[str, Any]) -> dict[str, Any]:
    source_pages = list((measurement.get("source") or {}).get("pages") or [])
    target_pages = list((measurement.get("translated") or {}).get("pages") or [])
    paired = list(zip(source_pages, target_pages))

    per_page: list[dict[str, Any]] = []
    layout_numerator = 0.0
    layout_denominator = 0.0
    matched_weight_total = 0.0
    for src, tgt in paired:
        page = _score_page(src, tgt)
        layout_numerator += page.pop("_layout_numerator")
        layout_denominator += page.pop("_layout_denominator")
        matched_weight_total += page.pop("_matched_weight")
        per_page.append(page)

    # Anchors and volume are document-wide over ALL pages (not the zipped prefix): a
    # re-typesetting system may move content across page boundaries, and a dropped/extra
    # page's text belongs in these totals (the page-count flag reports the mismatch itself).
    anchor_match = _match_anchors(source_pages, target_pages)
    anchors_total = anchor_match["total"]
    anchors_survived = anchor_match["survived"]
    volume_source = _volume_units(source_pages)
    volume_translated = _volume_units(target_pages)

    axes = {
        "layout": 100.0 if layout_denominator <= 0 else 100.0 * layout_numerator / layout_denominator,
        "anchors": 100.0 if anchors_total == 0 else 100.0 * anchors_survived / anchors_total,
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
        # The retired retention axis, kept as an indicator (region-matching-coupled).
        "region_retention": round(_mean([page["retention"] for page in per_page]), 2),
        "anchors_source": anchors_total,
        "anchors_survived": anchors_survived,
        "volume_units_source": volume_source,
        "volume_units_translated": volume_translated,
        "volume_ratio": round(volume_translated / volume_source, 3) if volume_source else None,
        # Share of region weight that stayed unmatched (lost + invented) in the layout
        # matching — the view's confidence signal: when this dominates, L/T (and reg-R)
        # move on matching noise rather than real geometry change. Additive since v5.
        "layout_noise_share": round(
            1.0 - matched_weight_total / layout_denominator, 3
        ) if layout_denominator > 0 else 0.0,
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
        "_matched_weight": sum(_weight(src_regions[i]) for i, _j, _iou in matches),
    }


def _region_count(pages: list[dict[str, Any]], labels: set[str]) -> int:
    return sum(
        1
        for page in pages
        for region in _usable_regions(page)
        if region["label"] in labels and region["score"] >= _FLAG_MIN_SCORE
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 100.0
