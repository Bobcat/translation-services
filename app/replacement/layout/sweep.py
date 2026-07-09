"""Stray-ink cleanup on flat, angle-snapped images: sweep leftover source ink from a line's slot."""
from __future__ import annotations

from typing import Any
import re
import numpy as np
from app.replacement.jobs import _Job
from app.replacement.pixels import _INK_DELTA


# Stray-ink cleanup (flat, angle-snapped images only): a pixel counts as ink when any channel
# deviates ``_INK_DELTA`` (app.replacement.pixels) from the plane's sampled background. Both
# mechanisms are self-limiting on unreliable ground: the band sweep runs only when the
# un-claimed part of the band is overwhelmingly background (texture fails the fraction guard
# and nothing happens), and quad growth only sticks when it reaches a CLEAN row within its
# cap (texture never does).
_SWEEP_MAX_INK_FRACTION = 0.35

# Erase margin ABOVE/BELOW the original text. The OCR polygon's ``ymin``/``ymax`` already bound
# the glyphs (descenders included), so vertically the erase needs only a thin anti-alias margin
# — not the full ``pad``. The full pad would reach into whatever sits just above or below the
# line (a coloured header band a few px away) and erase it; a tight vertical margin keeps the
# fill on the text. The sides keep ``pad`` (and grow with the tile) for horizontal blending.
_ERASE_MARGIN = 2.0

def _sweep_stray_ink(
    base_np: np.ndarray,
    planes: list[dict[str, Any]],
    jobs: list[_Job],
    protected_boxes: list[dict[str, Any]],
    own_boxes: list[dict[str, Any]],
) -> None:
    """Erase the LINE SLOT of the group's erase-only planes — lines whose content the reflow
    moved into the lines above (the translation needed fewer lines than the original, and the
    hint-coverage gate already established that this line's text lives on in the translation).
    A superseded line's slot may hold ink the member erase cannot reach: words OCR never
    detected (they double under the translation) and glyph edges the OCR box undershot (digit
    bottoms survive as dash-like remnants). The slot runs to halfway the neighbouring planes —
    ink past that midline belongs to the neighbour. Planes that still receive a tile are left
    alone: unclaimed ink next to still-standing content can be REAL content the pipeline
    missed (a receipt's time or card digits), and hiding a miss is worse than showing it.

    Members of OTHER units (preserved fields, skipped units, interleaved leftovers) are
    protected ground: when one touches the slot, only the un-protected ink runs are erased
    instead of the full slot. The group's own erased members are not protected — their
    leftovers are exactly what the slot erase is for. A slot whose unclaimed part is not
    overwhelmingly background is skipped entirely (texture: the measurement means nothing)."""
    height, width = base_np.shape[:2]
    group_x0 = int(max(0, min(plane["frame"][2] for plane in planes)))
    group_x1 = int(min(width, max(plane["frame"][3] for plane in planes)))
    own_keys = {
        (int(b.get("left", 0)), int(b.get("top", 0)), int(b.get("width", 0)), int(b.get("height", 0)))
        for b in own_boxes
    }
    other_boxes = [
        b for b in protected_boxes
        if (int(b.get("left", 0)), int(b.get("top", 0)), int(b.get("width", 0)), int(b.get("height", 0)))
        not in own_keys
    ]
    for index, plane in enumerate(planes):
        if jobs[index].tile is not None:
            continue  # a drawn line: unclaimed neighbours may be missed content, keep them
        frame = plane["frame"]
        ymin, ymax = float(frame[4]), float(frame[5])
        line_h = max(1.0, ymax - ymin)
        # The line's slot: to halfway the neighbouring plane, or — at the block's edge — far
        # enough to cover under/overshooting glyph edges (descender depth scales with size).
        edge_ext = max(_ERASE_MARGIN, 0.35 * line_h)
        top_ext = (ymin - float(planes[index - 1]["frame"][5])) / 2 if index else edge_ext
        bottom_ext = (float(planes[index + 1]["frame"][4]) - ymax) / 2 if index + 1 < len(planes) else edge_ext
        band0 = max(0, int(ymin - max(_ERASE_MARGIN, min(top_ext, line_h))))
        band1 = min(height, int(ymax + max(_ERASE_MARGIN, min(bottom_ext, line_h))))
        x0 = group_x0 if plane.get("bullet_y") is None else max(group_x0, int(frame[2]))
        x1 = group_x1
        if band1 <= band0 or x1 <= x0:
            continue
        band = base_np[band0:band1, x0:x1].astype(int)
        bg = np.array(jobs[index].bg_color)
        inked = (np.abs(band - bg).max(axis=2) > _INK_DELTA).any(axis=0)
        own_cols = _column_mask(own_boxes, band0, band1, x0, x1)
        outside_own = ~own_cols
        if outside_own.any() and inked[outside_own].mean() > _SWEEP_MAX_INK_FRACTION:
            continue  # unclaimed part is mostly ink: not a flat background, do not touch it
        protected_cols = _column_mask(other_boxes, band0, band1, x0, x1)
        if not protected_cols.any():
            jobs[index].erase_quads.append([(x0, band0), (x1, band0), (x1, band1), (x0, band1)])
            continue
        # A protected member shares the slot: erase only the un-protected ink runs.
        candidate = inked & ~protected_cols
        run_start = None
        gap = 0
        runs: list[tuple[int, int]] = []
        for col in range(len(candidate) + 1):
            on = col < len(candidate) and candidate[col]
            if on:
                if run_start is None:
                    run_start = col
                gap = 0
            elif run_start is not None:
                gap += 1
                if gap > 3 or col == len(candidate):
                    runs.append((run_start, col - gap + 1))
                    run_start = None
                    gap = 0
        for r0, r1 in runs:
            if r1 - r0 < 2:
                continue  # single-pixel specks: dust/JPEG noise, not glyphs
            e_x0 = max(0, x0 + r0 - 2)
            e_x1 = min(width, x0 + r1 + 2)
            jobs[index].erase_quads.append(
                [(e_x0, band0), (e_x1, band0), (e_x1, band1), (e_x0, band1)]
            )

def _column_mask(
    boxes: list[dict[str, Any]], band0: int, band1: int, x0: int, x1: int
) -> np.ndarray:
    """Columns of the ``x0..x1`` window covered by any box that vertically overlaps the band."""
    mask = np.zeros(max(0, x1 - x0), dtype=bool)
    for box in boxes:
        b_top = int(box.get("top", 0))
        if b_top + int(box.get("height", 0)) < band0 or b_top > band1:
            continue
        b_x0 = max(0, int(box.get("left", 0)) - int(_ERASE_MARGIN) - x0)
        b_x1 = min(x1 - x0, int(box.get("left", 0)) + int(box.get("width", 0)) + int(_ERASE_MARGIN) - x0)
        if b_x1 > b_x0:
            mask[b_x0:b_x1] = True
    return mask

def _ink_runs(mask) -> list[tuple[int, int]]:
    """Contiguous (start, end) column ranges where ``mask`` is True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x, value in enumerate(mask):
        if value and start is None:
            start = x
        elif not value and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs

def _hint_covers_undetected_text(units: list[dict[str, Any]]) -> bool:
    """Whether the group's HINT-side source (the field pairs' source texts) carries tokens its
    detected members do not account for — the signal that this line holds text OCR never boxed
    but the translation DOES cover (the VLM read it). Only then may unclaimed ink be treated as
    superseded source text; where hint and OCR both missed it (a receipt's time or card digits),
    the translation does not cover it and the pixels must stay: a visible miss beats a hidden
    one."""
    source = " ".join(
        str(source_text)
        for unit in units
        for source_text, _translated in (unit.get("field_translations") or [])
    )
    if not source:
        return False
    member_tokens = {
        token
        for unit in units
        for member in (unit.get("members") or [])
        for token in re.findall(r"\w+", str(member.get("text") or "").lower())
    }
    return any(token not in member_tokens for token in re.findall(r"\w+", source.lower()))
