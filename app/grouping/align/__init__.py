"""Align the VLM grouping hint onto the authoritative OCR cells and build units.

OCR cells are the source of truth for positions; the VLM hint is the source of
truth for text/structure. This module assigns each cell to the hint line it
best matches (normalised token overlap, tolerant to OCR garble), breaking ties
by an anchor-interpolated reading position (``_anchored_positions``), groups
consecutive cells with the same assignment into a unit, and turns every
unmatched cell into its own unit. Because we build the units, coverage is guaranteed: every cell
ends up in exactly one unit, so a weak/incomplete hint lowers quality but never
fails the job.

``translate`` (a whole-cell price/URL/number is not translatable) is decided here
by small rules, not by the model.
"""
from app.grouping.align.build import build_units_from_hint

__all__ = ["build_units_from_hint"]
