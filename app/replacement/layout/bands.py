"""Text bands of a page: the vertical strips its lines are set in, and each strip's own
right margin.

A page's lines do not run to the paper edge — they stop at a margin, and a multi-column
page stops each column at the gutter as well. Both are readable from the source cells
alone: project every member box onto the x axis, and an EMPTY corridor with occupied
ground on either side is a gutter; the rightmost box of the strip between two gutters is
that strip's margin.

The pixel scan behind the ``extend`` width fit cannot see either. It stops at ink, at a
protected cell and at ~1 em from the IMAGE edge — the right reference for a photo or a
sign, where the image edge is the only frame there is, and the wrong one for a document,
which carries its own margin. Unbounded it grows a line across the gutter into the next
column (measured: one line spanning the full width of a two-column page) or ~260px past
a paper's right margin, to within 39px of the sheet.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# An empty x corridor this wide (in image pixels), with occupied ground on BOTH sides,
# is a gutter. Below it the whitespace is word spacing or a table's column gap, not a
# page division. Measured on two-column papers: the gutter runs ~37px at 200 dpi while
# inter-word gaps stay well under 20px.
_GUTTER_MIN_WIDTH = 25


def text_bands(boxes: list[dict[str, Any]]) -> list[dict[str, float]]:
    """The page's text bands, left to right: ``{"left", "right", "margin"}`` in image
    pixels. ``margin`` is the rightmost edge any source box in the band reaches — the
    band's own right margin. Fewer than two boxes yields no bands (nothing to bound)."""
    spans = [
        (float(b["left"]), float(b["left"]) + float(b["width"]))
        for b in boxes
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
        bands.append({"left": start, "right": end, "margin": max(members)})
    return bands


def band_margin_at(bands: list[dict[str, float]], x: float) -> float | None:
    """The right margin of the band a line starting at ``x`` belongs to, or ``None`` when
    no band covers it (then nothing is bounded — the caller keeps its own limits)."""
    for band in bands:
        if band["left"] <= x < band["right"]:
            return band["margin"]
    return None
