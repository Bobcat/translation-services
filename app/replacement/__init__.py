"""Stage #8: re-placement — render translated text back into the image.

Tier-1 "simple replace" (model-free): per unit, sample the background colour, erase
the translatable members' boxes, and draw the fitted translation. See
docs/re-placement.md for the directions and the upgrade path (LaMa inpainting,
overlay/callout, real colour sampling, rotation/perspective).
"""
from __future__ import annotations

from app.replacement.render import render_translated_image


__all__ = ["render_translated_image"]
