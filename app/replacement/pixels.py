"""Shared pixel-analysis constant for re-placement."""
# A pixel deviating this much from the sampled background (any channel) counts as glyph ink.
# Shared so the erase layer (residue), the render measurements (size band / extend / stray
# sweep) and colour sampling all judge "ink" by the same threshold.
_INK_DELTA = 48
