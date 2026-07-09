"""The placed render job — the unit of work the planner produces and the erase / composite
passes consume."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class _Job:
    # One tight quad per original member (word), each at its OWN tilt — not a single
    # line-spanning rectangle. A photographed line fans (perspective steepens along it),
    # so one straight rectangle pinned to the highest word floats above the lower ones;
    # with a flat fill that overshoot paints background colour past the text's band.
    erase_quads: list[list[tuple[int, int]]]
    bg_color: tuple[int, int, int]
    # None for an erase-only plane (the translation needed fewer lines than the original).
    tile: Image.Image | None
    dst_quad: list[tuple[float, float]] | None
