"""Text bands of a page: the strips its lines are set in, and each strip's own right margin.

A page's lines do not run to the paper edge — they stop at a margin, and a multi-column page
stops each column at the gutter as well. The pixel scan behind the ``extend`` width fit sees
neither: it stops at ink, at a protected cell and at ~1 em from the IMAGE edge — the right
reference for a photo or a sign, where the image edge is the only frame there is, and the
wrong one for a document. Unbounded it grows a line across the gutter into the next column
(measured: one line spanning the full width of a two-column page) or ~260px past a paper's
right margin, to within 39px of the sheet.

Two sources of evidence, in order:

  * **layout regions** — the page detector already runs per page and its regions are cached,
    so a body column arrives LABELLED. This is the primary source: measured over 25 document
    pages, it leaves 13% of ceilings reaching past other text where the projection below
    leaves 40%, and it is never looser on any page.
  * **an x-axis projection of the boxes** — where no region covers a line (47% of document
    boxes, and most of a poster's), an empty corridor with occupied ground on BOTH sides is
    read as a gutter and the strip between two gutters takes its rightmost box as its margin.
    Honest but blind to labels: ONE page-wide element (a figure caption, a centred page
    number) fills the corridor and the whole page reads as a single column — which is exactly
    how the ceiling came to sit at the far column's edge and stop nothing.

In both cases the margin is the rightmost BOX, never a region's own edge: a region drawn too
wide then cannot inflate the margin, and one drawn too narrow only clamps tighter — the
failure that costs nothing.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.layout import COLUMN_LABELS


# A region below this confidence is not evidence. Matches the cell->region assignment gate in
# app.layout.evidence: the same regions, read for the same kind of claim.
_MIN_REGION_SCORE = 0.6


# An empty x corridor this wide (in image pixels), with occupied ground on BOTH sides,
# is a gutter. Below it the whitespace is word spacing or a table's column gap, not a
# page division. Measured on two-column papers: the gutter runs ~37px at 200 dpi while
# inter-word gaps stay well under 20px.
_GUTTER_MIN_WIDTH = 25


def text_bands(
    boxes: list[dict[str, Any]], regions: list[dict[str, Any]] | None = None
) -> list[dict[str, float]]:
    """The page's text bands: ``{"left", "right", "top", "bottom", "margin"}`` in image pixels.

    Region bands come FIRST, so a line inside a labelled body column takes that column's
    margin; the projection bands that follow cover everything the regions leave unclaimed
    (they span the full page height, so a lookup always lands somewhere they apply)."""
    body = _body_boxes(boxes, regions or [])
    return _region_bands(boxes, regions or []) + _projection_bands(boxes, body)


def _body_boxes(
    boxes: list[dict[str, Any]], regions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The boxes a body-column region claims. Feeding the projection only these is what lets
    it find the gutters a page-wide element would otherwise fill: the caption, the banner
    heading and the centred page number are not body text, so they no longer vote. Empty when
    there are no regions, and the projection then sees every box (its own historic behaviour)."""
    claimed: list[dict[str, Any]] = []
    for region in regions:
        if str(region.get("label") or "") not in COLUMN_LABELS:
            continue
        if float(region.get("score") or 0.0) < _MIN_REGION_SCORE:
            continue
        coordinate = list(region.get("coordinate") or [])
        if len(coordinate) != 4:
            continue
        x0, y0, x1, y1 = (float(value) for value in coordinate)
        claimed.extend(b for b in boxes if b and b.get("width") and _centre_in(b, x0, y0, x1, y1))
    return claimed


def _region_bands(
    boxes: list[dict[str, Any]], regions: list[dict[str, Any]]
) -> list[dict[str, float]]:
    """One band per body-column region that actually holds text, its margin the rightmost box
    inside it. A region holding no box is not evidence about any line and is skipped — which
    is what keeps a stray detection from bounding anything."""
    bands: list[dict[str, float]] = []
    for region in regions:
        if str(region.get("label") or "") not in COLUMN_LABELS:
            continue
        if float(region.get("score") or 0.0) < _MIN_REGION_SCORE:
            continue
        coordinate = list(region.get("coordinate") or [])
        if len(coordinate) != 4:
            continue
        x0, y0, x1, y1 = (float(value) for value in coordinate)
        inside = [b for b in boxes if b and b.get("width") and _centre_in(b, x0, y0, x1, y1)]
        if not inside:
            continue
        bands.append({
            "left": x0,
            "right": x1,
            "top": y0,
            "bottom": y1,
            "margin": max(float(b["left"]) + float(b["width"]) for b in inside),
        })
    return bands


def _centre_in(box: dict[str, Any], x0: float, y0: float, x1: float, y1: float) -> bool:
    cx = float(box["left"]) + float(box.get("width") or 0.0) / 2.0
    cy = float(box["top"]) + float(box.get("height") or 0.0) / 2.0
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _projection_bands(
    boxes: list[dict[str, Any]], body: list[dict[str, Any]] | None = None
) -> list[dict[str, float]]:
    """Gutter-split bands from an x-axis projection of the boxes.

    ``body`` (the boxes a column region claims) narrows the projection to running text where
    that evidence exists, so a page-wide caption or heading no longer fills a gutter and the
    columns separate for the lines the regions did NOT claim — the ones that fall through to
    here. Bands still span the whole page width, so every lookup lands."""
    span_source = body if body and len(body) >= 2 else boxes
    spans = [
        (float(b["left"]), float(b["left"]) + float(b["width"]))
        for b in span_source
        if b and b.get("width")
    ]
    if len(spans) < 2:
        return []
    width = int(max(right for _left, right in spans)) + 1
    occupied = np.zeros(width + 1, dtype=bool)
    for left, right in spans:
        occupied[max(0, int(left)):max(0, int(right))] = True

    # Cut points: the start of every gutter-wide empty corridor that has occupied ground
    # on both sides. A trailing empty margin is NOT a cut — a design image's free space
    # right of its longest line is exactly what the extend fit is for.
    cuts: list[int] = []
    run_start: int | None = None
    for x in range(width + 1):
        if not occupied[x]:
            if run_start is None:
                run_start = x
        elif run_start is not None:
            if x - run_start >= _GUTTER_MIN_WIDTH and occupied[:run_start].any():
                cuts.append(x)
            run_start = None

    edges = [0.0, *(float(c) for c in cuts), float(width)]
    bands: list[dict[str, float]] = []
    for start, end in zip(edges, edges[1:]):
        members = [right for left, right in spans if start <= left < end]
        if not members:
            continue
        bands.append({
            "left": start, "right": end,
            "top": float("-inf"), "bottom": float("inf"),
            "margin": max(members),
        })
    return bands


def band_margin_at(bands: list[dict[str, float]], x: float, y: float) -> float | None:
    """The right margin for the point ``(x, y)``, or ``None`` when no band covers it (then
    nothing is bounded — the caller keeps its own limits).

    Bands overlap: a region band sits on top of the projection band it came from, and two
    detected regions can overlap each other. The TIGHTEST covering margin wins, so an overlap
    can only ever bound a line more, never less — at worst it withdraws the extension and the
    line keeps its own footprint, which is the behaviour without this fit at all."""
    margins = [
        band["margin"]
        for band in bands
        if band["left"] <= x < band["right"] and band["top"] <= y <= band["bottom"]
    ]
    return min(margins) if margins else None
