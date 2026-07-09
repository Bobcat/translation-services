"""Source size of a line: the band metric (how tall a line renders) and the group size."""
from __future__ import annotations

from typing import Any
import numpy as np
from statistics import median
from PIL import Image
from app.replacement import geometry as geo
from app.replacement.geometry import _ANGLE_DEADZONE_DEG
from app.replacement.geometry import _plane_corners
from app.replacement.pixels import _INK_DELTA
from app.replacement.text.angle import _image_is_flat
from app.replacement.ground.color import sample_oriented_colors


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
