"""Source size of a line: the band metric (how tall a line renders) and the group size."""
from __future__ import annotations

from functools import lru_cache
from typing import Any
from collections import defaultdict
import numpy as np
from statistics import median, pstdev
from PIL import Image
from PIL import ImageDraw
from app.replacement import geometry as geo
from app.replacement.geometry import _ANGLE_DEADZONE_DEG
from app.replacement.geometry import _plane_corners
from app.replacement.pixels import _INK_DELTA
from app.replacement.text.angle import _image_is_flat
from app.replacement.text.fit import load_font
from app.replacement.ground.color import sample_oriented_colors


# "fill" size metric (size_metric_mode): render each line so ITS OWN ink is as tall as the
# SOURCE line's ink, instead of the polygon height x a fixed ratio. The default 0.9 ratio
# undershoots — the polygon spans ascender-to-descender but the mapped face's ink at that pt
# fills less of it, so body text renders ~10-20% smaller than the print and reads airy. Fill
# measures both sides in pixels (self-calibrating, per element, no global scale) and matches
# them: target pt = source ink span / the face's ink-per-pt. Flat groups only (the source
# scan is axis-aligned) and never CJK (the em-fill ratio already handles those).
_FILL_PROBE = "Aghpxldijk"   # ascender + descender + x-height, so the ink span is the full face extent
_FILL_REF_SIZE = 100


@lru_cache(maxsize=32)
def _face_ink_per_pt(family: str | None, weight: int | None) -> float:
    """Rendered ink height per pt for the mapped face: draw a full-extent probe at a reference
    size and measure its ink pixel span. Cached per (family, weight). ~0.9 for the serif face."""
    font = load_font(_FILL_REF_SIZE, _FILL_PROBE, family=family, weight=weight)
    ascent, descent = font.getmetrics()
    image = Image.new("L", (max(4, int(font.getlength(_FILL_PROBE)) + 8), ascent + descent + 8), 0)
    ImageDraw.Draw(image).text((2, 2), _FILL_PROBE, font=font, fill=255)
    rows = np.nonzero((np.asarray(image) > 60).any(axis=1))[0]
    span = float(rows.max() - rows.min() + 1) if len(rows) else float(_FILL_REF_SIZE)
    return span / _FILL_REF_SIZE if span > 0 else 0.9


def _quad_ink_span(base_px: np.ndarray, quad, bg: tuple[int, int, int]) -> float | None:
    """Vertical extent of the quad's ink: the span between the first and last row carrying any
    ink against ``bg`` (measured the same way as the face probe — full ascender-to-descender).
    None when the box is too small or carries no ink."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0, x1 = max(0, int(min(xs))), min(base_px.shape[1], int(max(xs)) + 1)
    y0, y1 = max(0, int(min(ys))), min(base_px.shape[0], int(max(ys)) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    window = base_px[y0:y1, x0:x1].astype(np.int16)
    inked = (np.abs(window - np.asarray(bg, dtype=np.int16)).max(axis=2) >= _INK_DELTA).any(axis=1)
    rows = np.nonzero(inked)[0]
    return float(rows.max() - rows.min() + 1) if len(rows) else None


# "band" size metric (size_metric_mode): the OCR polygon's full ink extent is the default
# size source, but sparse tall glyphs — parentheses (as a marker OR mid-text), brackets, a
# stray swash — stretch it far past the text band while the band itself stays put
# (measured on a numbered list: polygon 69-76px, text band 36px, vs 47-50/35 on plain
# siblings). No LOCAL gate separates that inflation from measurement noise reliably
# (three measurement rounds: absolute band ratios, doc-normalised ratios and fringe
# ratios all overlap with normal lines), so the correction is per-DOCUMENT and one-sided:
# the document's median polygon/band ratio anchors what "normal" looks like on this very
# image, and a line's height is clamped to band * that ratio — outliers sink to the
# document norm, everything else is untouched. Weak ink evidence falls back to the extent.
_BAND_STRONG_ROW_FRACTION = 0.15  # rows holding >= this fraction of the peak ink count
_BAND_MIN_ROWS = 3
_BAND_MIN_PEAK = 4
_BAND_MIN_SAMPLES = 4  # document ratio needs at least this many measured quads

# "cohort" size metric (size_cohort_mode="vlm"): the VLM gives sibling elements ONE font-size
# label (pt), so its per-element pt is a reliable EQUALITY signal even though its absolute
# pt->pixel scale drifts per image. OCR true-height per element is the absolute scale but is
# noisy line to line (ink extent varies with glyph content: a lowercase word reads shorter than
# one with ascenders). So: elements the VLM labelled with one pt form a size cohort; when their
# OCR heights AGREE (low spread — the VLM's equal claim holds), snap the whole cohort to its OCR
# median, making the list render at one size. A cohort whose OCR heights DISAGREE (the VLM's
# claim is wrong, or a genuine outlier) is left on per-element OCR. Measured across the list
# fixtures: same-pt cohorts sit at 3-6% CV, genuinely-different sizes land in different pt
# cohorts (adv-budgets 14/16/24pt) — so the gate separates cleanly.
_COHORT_MIN_MEMBERS = 3   # need a few same-pt elements before trusting the cohort
_COHORT_MAX_CV = 0.15     # OCR spread within a cohort must stay under this to snap

def _quad_band_height(base_px: np.ndarray, quad, bg: tuple[int, int, int]) -> float | None:
    """Height of the quad's STRONG ink band: the row span where the ink column count
    reaches at least _BAND_STRONG_ROW_FRACTION of the peak row. Sparse fringe rows (a
    parenthesis tip, an anti-alias fade) fall below it; the band tracks the glyph core.
    None when the ink evidence is too weak to trust."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0, x1 = max(0, int(min(xs))), min(base_px.shape[1], int(max(xs)) + 1)
    y0, y1 = max(0, int(min(ys))), min(base_px.shape[0], int(max(ys)) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    window = base_px[y0:y1, x0:x1].astype(np.int16)
    counts = (np.abs(window - np.asarray(bg, dtype=np.int16)).max(axis=2) >= _INK_DELTA).sum(axis=1)
    if counts.max() < _BAND_MIN_PEAK:
        return None
    strong = np.nonzero(counts >= max(2, _BAND_STRONG_ROW_FRACTION * counts.max()))[0]
    if len(strong) < _BAND_MIN_ROWS:
        return None
    return float(strong.max() - strong.min() + 1)

def _document_band_ratio(base: Image.Image, units: list[dict[str, Any]]) -> float | None:
    """Median polygon-height / band-height over the document's member quads — what the
    extent-to-band relation looks like on this image's NORMAL lines (OCR box generosity
    varies per archetype, so the anchor must come from the image itself). None without
    enough evidence, or on tilted images where the axis-aligned scan is unreliable."""
    if not _image_is_flat(units):
        return None
    base_px = np.asarray(base)
    ratios: list[float] = []
    for unit in units:
        for member in (unit.get("members") or []):
            if not member.get("bbox"):
                continue
            quad = geo.quad_of(member)
            if quad is None or abs(geo.angle_deg(quad)) >= _ANGLE_DEADZONE_DEG:
                continue
            height = geo.line_height(quad)
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            frame = ((1.0, 0.0), (0.0, 1.0), min(xs), max(xs), min(ys), max(ys))
            bg, _fg = sample_oriented_colors(base, _plane_corners({"frame": frame, "pad": max(2.0, height / 6.0)}))
            band = _quad_band_height(base_px, quad, bg)
            if band:
                ratios.append(height / band)
    if len(ratios) < _BAND_MIN_SAMPLES:
        return None
    return float(median(ratios))

def _document_size_cohorts(units: list[dict[str, Any]]) -> dict[int, float]:
    """Map each VLM font-size (pt) to its cohort's median OCR true-height, for cohorts the VLM
    judged one size AND OCR agrees on (>= _COHORT_MIN_MEMBERS elements, spread under
    _COHORT_MAX_CV). A group whose pt is in the map renders at the cohort's shared size instead
    of its own noisy per-line measurement; a pt not in the map (too few elements, or OCR
    disagrees) keeps per-element OCR sizing. Only flat images (tilted line-height reads are
    unreliable)."""
    if not _image_is_flat(units):
        return {}
    by_pt: dict[int, list[float]] = defaultdict(list)
    for unit in units:
        pt = unit.get("font_size")
        if pt is None:
            continue
        heights = [
            geo.line_height(quad)
            for member in (unit.get("members") or [])
            if member.get("bbox") and (quad := geo.quad_of(member)) is not None
        ]
        if heights:
            by_pt[int(pt)].append(median(heights))
    cohorts: dict[int, float] = {}
    for pt, heights in by_pt.items():
        if len(heights) < _COHORT_MIN_MEMBERS:
            continue
        centre = median(heights)
        if centre > 0 and pstdev(heights) / centre <= _COHORT_MAX_CV:
            cohorts[pt] = centre
    return cohorts


def _group_size(planes: list[dict[str, Any]], mode: str) -> int:
    """The group's ONE render size, chosen from its per-line targets. ``min`` (the default)
    never draws taller than the smallest measured line — but one under-measured line (ink
    without ascenders reads ~70% of cap height) drags the whole block down. ``median`` is the
    better estimator of the element's single true size, at the cost that a genuinely smaller
    line the VLM mixed into the element renders over its own band. Selectable per request
    (``render_size_mode``) to compare; an unknown value falls back to ``min``. A future smarter
    selection policy slots in here as another mode."""
    targets = [plane["target"] for plane in planes]
    if mode == "median":
        return int(median(targets))
    return min(targets)
