"""Text-line tilt: flatness, per-line-cluster baseline fit, and the document angle field."""
from __future__ import annotations

from typing import Any
import math
import numpy as np
from statistics import median
from app.replacement import geometry as geo
from app.replacement.geometry import _ANGLE_DEADZONE_DEG
from app.replacement.layout.groups import _groups


# Angle FIELD (tilted images): a photographed tilted sign is a perspective gradient —
# every text line's angle follows one smooth, near-linear function of image y (measured
# across the tilted testset: linear fits with <=0.5deg MAD over the whole document),
# while each group's own baseline fit wobbles +-0.5-1deg around it and a one-word group
# can sit several degrees off. Reading every group's angle from one document-level fit
# renders the sign as a single printed surface. Evidence gates (all must hold, else the
# per-group path): a single-cell line's edge angle only counts when its quad is clearly
# wider than tall (a square-ish quad's edge direction is meaningless — the measured
# outliers were all near-square logos/prices); enough samples over enough y-range to
# define a slope; residuals no wider than a real gradient's.
_FIELD_MIN_SAMPLES = 5
_FIELD_MIN_EDGE_ASPECT = 2.0  # single-cell sample: quad width >= this x line height
_FIELD_MIN_SPAN_RATIO = 10.0  # sample y-span >= this x the median sampled line height
_FIELD_MAX_RESIDUAL_MAD_DEG = 1.0

def _image_is_flat(units: list[dict[str, Any]]) -> bool:
    """True when the image as a whole reads as fronto-parallel — its lines are near-horizontal
    with no real page tilt, so per-line angles are OCR detection noise to be snapped away. A
    photographed sign at an angle has a perspective gradient with a sizeable median angle and is
    NOT flat, so its (real) angles are kept. Median over all member quads is robust to the odd
    rotated stray (a lone tall glyph) that a mean would be skewed by."""
    angles: list[float] = []
    for unit in units:
        for member in unit.get("members") or []:
            if not member.get("bbox"):
                continue
            quad = geo.quad_of(member)
            if quad is not None:
                angles.append(abs(geo.angle_deg(quad)))
    return bool(angles) and median(angles) < _ANGLE_DEADZONE_DEG

def _document_angle_field(units: list[dict[str, Any]]) -> tuple[float, float] | None:
    """The document's text angle as a function of y — ``(slope_deg_per_px, intercept_deg)``
    — or None when the image is flat or the evidence fails the gates (see the _FIELD_*
    constants). Samples: each multi-word line contributes its baseline-fit angle; a
    single-cell line contributes its quad edge angle only when clearly wider than tall.
    The fit is Theil-Sen (median of pairwise slopes), so a stray sample cannot tip it."""
    if _image_is_flat(units):
        return None
    samples: list[tuple[float, float]] = []  # (line centre y, angle deg)
    heights: list[float] = []
    for group in _groups(units):
        quads = [
            quad
            for unit in group
            for member in (unit.get("members") or [])
            if member.get("bbox") and (quad := geo.quad_of(member)) is not None
        ]
        if not quads:
            continue
        seed = median(geo.angle_deg(quad) for quad in quads)
        for cluster in _line_clusters(quads, seed):
            centres = [(sum(p[0] for p in q) / 4.0, sum(p[1] for p in q) / 4.0) for q in cluster]
            y_centre = sum(c[1] for c in centres) / len(centres)
            xs = [c[0] for c in centres]
            if len(centres) >= 2 and (max(xs) - min(xs)) >= 1.0:
                samples.append((y_centre, _baseline_angle([cluster], seed)))
                heights.append(median(geo.line_height(q) for q in cluster))
            elif len(cluster) == 1:
                quad = cluster[0]
                width = 0.5 * (
                    math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])
                    + math.hypot(quad[2][0] - quad[3][0], quad[2][1] - quad[3][1])
                )
                height = geo.line_height(quad)
                if width >= _FIELD_MIN_EDGE_ASPECT * height:
                    samples.append((y_centre, geo.angle_deg(quad)))
                    heights.append(height)
    if len(samples) < _FIELD_MIN_SAMPLES:
        return None
    ys = [y for y, _ in samples]
    y_span = max(ys) - min(ys)
    if y_span < _FIELD_MIN_SPAN_RATIO * median(heights):
        return None
    # Pairs closer in y than 5% of the span carry no slope information, only noise.
    min_dy = 0.05 * y_span
    slopes = [
        (a2 - a1) / (y2 - y1)
        for i, (y1, a1) in enumerate(samples)
        for y2, a2 in samples[i + 1 :]
        if abs(y2 - y1) >= min_dy
    ]
    if not slopes:
        return None
    slope = median(slopes)
    intercept = median(a - slope * y for y, a in samples)
    if median(abs(a - (slope * y + intercept)) for y, a in samples) > _FIELD_MAX_RESIDUAL_MAD_DEG:
        return None
    return float(slope), float(intercept)

def _baseline_angle(clusters: list[list], fallback: float) -> float:
    """The block's text-line direction, fit through the word CENTRES rather than read off the
    OCR quad edges (which jitter several degrees per word and bias the median shallow). Each line's
    words are de-meaned vertically so the parallel lines of a block all contribute to ONE shared
    slope — keeping the lines parallel while using every word for a robust fit. Falls back to
    ``fallback`` when too few words span an x-range to define a slope (a one-word line, or a
    vertical stack of single words at the same x)."""
    xs: list[float] = []
    ys: list[float] = []
    for cluster in clusters:
        centres = [(sum(p[0] for p in q) / 4.0, sum(p[1] for p in q) / 4.0) for q in cluster]
        if len(centres) < 2:
            continue
        # De-mean BOTH axes per cluster: with only y de-meaned, clusters of different x-extents
        # (a long line above a short last line) share one forced intercept, which drags the
        # fitted slope toward 0 — the very shallow bias this fit exists to remove. Centred per
        # cluster, parallel lines fit their true slope exactly.
        mean_x = sum(c[0] for c in centres) / len(centres)
        mean_y = sum(c[1] for c in centres) / len(centres)
        for cx, cy in centres:
            xs.append(cx - mean_x)
            ys.append(cy - mean_y)
    if len(xs) < 2 or (max(xs) - min(xs)) < 1.0:
        return fallback
    slope = float(np.polyfit(xs, ys, 1)[0])
    return math.degrees(math.atan(slope))

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
