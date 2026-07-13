"""Stage #8 re-placement — background-matched, polygon-aware (Tier-1, model-free).

Units that share a VLM block (a wrapped dish, a body paragraph) render as one
**group**: their translations are joined and balanced over the original number of
lines, at ONE font size taken from the original's true line height — the **source
size**, so a heading stays heading-sized and body stays body-sized (the source size
carries the visual hierarchy). Width is matched separately by **horizontal condensation**:
at the source height a translated line is usually wider than its original, so the rendered
text is squeezed in x to fit the original line's width (floored, never stretched) — keeping
height while matching width, the way the reference render does. Each rendered line anchors
on its original line's plane (so the line pitch follows the original). Per plane: cover
the original with the locally-sampled **background colour** (so it reads as erased
on a flat surface — menu paper, sign panel, receipt), then draw the line and **warp
it onto the plane's polygon** so it follows the page tilt (rotation/perspective),
for a clean camera-translation look.

Two facts make this work without a model:
- the OCR polygon gives the **true line height** (tilt-invariant), so text is sized
  consistently instead of by the inflated axis-aligned bbox — see `geometry`;
- the polygon also gives the **angle**, so a flat RGBA text tile can be warped to the
  oriented region with OpenCV.

`translate: false` members and ignored cells are never touched. Textured/photographic
backgrounds scar under the flat fill (it can't blend); ``erase_fill_mode="inpaint"``
switches pass 1 to the LaMa reconstruction (Tier 2, :mod:`app.replacement.ground.inpaint`).

Composition root: this module owns only ``render_translated_image`` — the pipeline (open
the image, plan each group into placed jobs, erase pass 1, warp the tiles on in pass 2,
encode). Each concern lives in a named module a reader reaches from a render symptom
(the full map is docs/replacement-architecture.md):

  layout/  planning.py (group -> jobs) - groups.py - tables.py (columns) -
           markers.py (bullets) - sweep.py (stray ink) - compositing.py (tile warp)
  text/    angle.py (line tilt) - size.py (source size) - wrap.py (break/condense) - fit.py
  ground/  color.py (bg sample) - erase.py (flat-vs-model + residue) - inpaint.py (LaMa)
  shared   geometry.py - jobs.py (_Job) - pixels.py (_INK_DELTA)

See docs/re-placement.md for the approach, docs/replacement-architecture.md for the tree.
"""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw

from app.replacement.ground.inpaint import inpaint_mask
from app.replacement.jobs import _Job
from app.replacement.layout.groups import _groups
from app.replacement.layout.planning import _plan_group
from app.replacement.text.angle import _document_angle_field
from app.replacement.text.angle import _image_is_flat
from app.replacement.text.size import _document_band_ratio
from app.replacement.text.size import _document_size_cohorts
from app.replacement.layout.compositing import _composite
from app.replacement.ground.erase import _GROUND_RING_INNER_PX
from app.replacement.ground.erase import _ellipse
from app.replacement.ground.erase import _erase_mask
from app.replacement.ground.erase import _needs_model_fill
from app.replacement.ground.erase import _swallow_erase_residue


def render_translated_image(
    input_path: Path,
    translation_units: list[dict[str, Any]],
    *,
    render_size_mode: str = "median",
    erase_fill_mode: str = "flat",
    width_fit_mode: str = "footprint",
    size_metric_mode: str = "extent",
    size_cohort_mode: str = "off",
) -> bytes:
    opened = Image.open(input_path)
    # Carry the source's ICC colour profile onto the output. ``convert("RGB")`` keeps the raw pixel
    # values but drops the profile, so without re-embedding it a colour-managed display (a phone)
    # shows the replacement as plain sRGB — the whole image reads duller/darker than the original.
    icc_profile = opened.info.get("icc_profile")
    base = opened.convert("RGB")

    jobs: list[_Job] = []
    groups = _groups(translation_units)
    base_arr = np.asarray(base)  # read-only pixel view, shared across all groups
    snap_horizontal = _image_is_flat(translation_units)
    # Every member box of EVERY unit — rendered, preserved or skipped — is protected ground for
    # the stray-ink sweep: only ink no unit accounts for may be treated as leftover source text.
    protected_boxes = [
        dict(member["bbox"])
        for unit in translation_units
        for member in (unit.get("members") or [])
        if member.get("bbox")
    ]
    # Every member text with its unit id: the table split needs to know whether an unplaceable
    # '|' field is printed in ANOTHER unit (an unmerged repeated column) before dropping it.
    document_member_texts = [
        (unit.get("id"), str(member.get("text") or ""))
        for unit in translation_units
        for member in (unit.get("members") or [])
    ]
    band_ratio = _document_band_ratio(base, translation_units) if size_metric_mode == "band" else None
    angle_field = _document_angle_field(translation_units)
    size_cohorts = _document_size_cohorts(translation_units) if size_cohort_mode == "vlm" else None
    for group in groups:
        jobs.extend(
            _plan_group(
                base,
                group,
                snap_horizontal=snap_horizontal,
                render_size_mode=render_size_mode,
                width_fit_mode=width_fit_mode,
                band_ratio=band_ratio,
                angle_field=angle_field,
                size_cohorts=size_cohorts,
                base_arr=base_arr,
                protected_boxes=protected_boxes,
                document_member_texts=document_member_texts,
            )
        )

    # Pass 1: cover every original (along the slant) so no source text peeks through.
    # "flat" paints each quad with its sampled background colour (Tier-1, model-free);
    # "inpaint" reconstructs the background under the same quads with LaMa (Tier-2,
    # see app/replacement/inpaint.py). Two model-free fills sat behind "inpaint" before
    # and were removed (2026-07-06): cv2 Telea diffusion (transports boundary pixels
    # inward — glyph residue, JPEG chroma halos and overlapping icons smear across the
    # fill) and a per-job least-squares colour-plane fit (on designed flat bands the
    # per-line fits land on slightly different shades — patchwork). LaMa reconstructs
    # rather than transports, which is exactly that failure-mode split.
    original = np.asarray(base).copy()
    if erase_fill_mode == "inpaint":
        # Hybrid: flat paint stays the fill for jobs on designed flat/solid ground (it is
        # right there by construction); the model only reconstructs the jobs whose ground
        # varies — texture or shading a flat rectangle would scar.
        occupied = np.zeros(original.shape[:2], dtype=np.uint8)
        for job in jobs:
            for quad in job.erase_quads:
                cv2.fillPoly(occupied, [np.asarray(quad, dtype=np.int32)], 255)
        occupied = cv2.dilate(occupied, _ellipse(_GROUND_RING_INNER_PX))
        routed = [(job, _needs_model_fill(original, job, occupied)) for job in jobs]
        flat_jobs = [job for job, to_model in routed if not to_model]
        model_jobs = [job for job, to_model in routed if to_model]
        canvas = original.copy()
        for job in flat_jobs:
            for quad in job.erase_quads:
                cv2.fillPoly(canvas, [np.asarray(quad, dtype=np.int32)], job.bg_color)
        _swallow_erase_residue(canvas, original, flat_jobs)
        if model_jobs:
            canvas = inpaint_mask(canvas, _erase_mask(original, model_jobs))
    else:
        erase = ImageDraw.Draw(base)
        for job in jobs:
            for quad in job.erase_quads:
                erase.polygon(quad, fill=job.bg_color)
        canvas = np.asarray(base).copy()
        _swallow_erase_residue(canvas, original, jobs)

    # Pass 2: warp each text tile onto its oriented region.
    for job in jobs:
        if job.tile is not None:
            _composite(canvas, job)

    out = BytesIO()
    save_kwargs: dict[str, Any] = {"compress_level": 1}
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    Image.fromarray(canvas).save(out, format="PNG", **save_kwargs)
    return out.getvalue()


