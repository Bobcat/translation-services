"""Tier-2 erase fill: LaMa background reconstruction (model-based, GPU-only).

The Tier-1 flat fill covers each erased line with one sampled colour; on textured or
photographic ground that reads as a scar. This module fills the erase mask with the
big-lama TorchScript model (Fourier-convolution CNN, one deterministic forward pass)
instead: it *reconstructs* the background from the surrounding context rather than
transporting boundary pixels inward — the failure mode that killed the Telea fill
(residue, JPEG chroma halos and overlapping icons smearing across the fill).

VRAM is bounded deterministically regardless of the input photo's resolution: the
model only ever sees a crop around the mask, capped at ``inpaint.pixel_budget_px``
(downscaled when larger; only masked pixels are pasted back, so unmasked pixels never
lose resolution). Activations scale linearly with input pixels (~750 MiB/Mpx on top of
~200 MiB weights), and a lock serializes forward passes, so the process peak is one
budget-sized crop — ~1.6 GiB at the default 1.5 Mpx.
"""
from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import load_settings


# Context ring around the mask bbox fed to the model: LaMa fills from surrounding
# texture, so the crop must include real background beyond the erased lines.
_CONTEXT_FRACTION = 0.15
_CONTEXT_MIN_PX = 32
# The model's down/up path needs input sides divisible by 8 (edge-padded, then cropped).
_PAD_MODULO = 8
# Feather on the paste border. When the crop was downscaled to the pixel budget the fill
# comes back through a resize; a hard edge against untouched full-res pixels would read
# as a seam. The erase mask is dilated past the ink, so the blend ring touches
# background, never glyphs.
_FEATHER_PX = 2
# Tiny crops are upscaled toward the model's training scale (256px crops) before the
# forward: at 1:1 a thumbnail-sized sign photo reads its glyph-scale features as texture
# to continue and the fill comes back as junk. The factor is capped at 2x — beyond it
# the hole grows past what the context ring supports and the fill washes out to grey
# (measured at 3x/4x).
_MIN_WORK_SHORT_SIDE = 320
_MAX_UPSCALE = 2.0
# A hole that touches the image border has no context on that side and the fill comes
# back unanchored (a dark smudge in the corner). Mirror-padding that side gives the
# model real background to anchor on — but only when the mirrored strip is actually
# safe to mirror. Two guards, both load-bearing:
# - the strip must CONTAIN background: mirroring a near-fully-masked strip just
#   enlarges the hole;
# - the background must be near-UNIFORM: mirroring a distinctive feature (a sign's
#   mounting hole) plants a copy next to the real one, and the Fourier-convolution
#   model reads the pair as a period to repeat across the whole hole. Featureful
#   borders keep the model's own boundary handling.
_REFLECT_MIN_ANCHOR = 0.25
_REFLECT_MAX_ANCHOR_STD = 20.0

_LOCK = threading.Lock()
_MODEL: Any | None = None
_BUDGET_PX: int = 0


def budget_scale(width: int, height: int, budget_px: int) -> float:
    """Downscale factor that brings width*height inside the pixel budget (1.0 = keep)."""
    area = width * height
    if area <= budget_px:
        return 1.0
    return math.sqrt(budget_px / area)


def work_scale(width: int, height: int, budget_px: int) -> float:
    """Resize factor for the crop fed to the model: down to the pixel budget, or up
    toward the training scale for tiny crops (capped, and never past the budget)."""
    scale = budget_scale(width, height, budget_px)
    if scale < 1.0:
        return scale
    short = min(width, height)
    if short >= _MIN_WORK_SHORT_SIDE:
        return 1.0
    return min(_MAX_UPSCALE, _MIN_WORK_SHORT_SIDE / short, math.sqrt(budget_px / (width * height)))


def context_window(
    mask: np.ndarray, *, margin_fraction: float = _CONTEXT_FRACTION, margin_min: int = _CONTEXT_MIN_PX
) -> tuple[int, int, int, int] | None:
    """Crop window (x0, y0, x1, y1) around the mask's set pixels, grown by a context
    margin and clamped to the image. None when the mask is empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    margin = max(margin_min, int(margin_fraction * max(x1 - x0, y1 - y0)))
    height, width = mask.shape[:2]
    return (max(0, x0 - margin), max(0, y0 - margin), min(width, x1 + margin), min(height, y1 + margin))


def border_pads(
    crop: np.ndarray,
    crop_mask: np.ndarray,
    window: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    pad: int = _CONTEXT_MIN_PX,
) -> tuple[int, int, int, int]:
    """Reflect padding (top, bottom, left, right) for a crop window: a side is padded
    when the crop sits on the image border there, the mask reaches that border, and the
    strip to be mirrored is safe to mirror — anchored (>= _REFLECT_MIN_ANCHOR unmasked)
    and near-uniform background (unmasked max channel std <= _REFLECT_MAX_ANCHOR_STD)."""
    x0, y0, x1, y1 = window
    height, width = image_size

    def side(at_border: bool, edge: np.ndarray, strip: np.ndarray, strip_img: np.ndarray) -> int:
        if not (at_border and edge.any()):
            return 0
        unmasked = strip == 0
        if unmasked.mean() < _REFLECT_MIN_ANCHOR:
            return 0
        if strip_img[unmasked].astype(np.float32).std(axis=0).max() > _REFLECT_MAX_ANCHOR_STD:
            return 0
        return pad

    return (
        side(y0 == 0, crop_mask[:2], crop_mask[:pad], crop[:pad]),
        side(y1 == height, crop_mask[-2:], crop_mask[-pad:], crop[-pad:]),
        side(x0 == 0, crop_mask[:, :2], crop_mask[:, :pad], crop[:, :pad]),
        side(x1 == width, crop_mask[:, -2:], crop_mask[:, -pad:], crop[:, -pad:]),
    )


def inpaint_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill ``mask>0`` pixels of an RGB uint8 image with LaMa-reconstructed background.

    Returns a copy; pixels outside the mask's feather ring stay byte-identical. Raises
    when no CUDA GPU or no checkpoint is available — the caller opted into the
    model-based fill explicitly, a silent flat fallback would misreport what rendered.
    """
    window = context_window(mask)
    if window is None:
        return image.copy()
    x0, y0, x1, y1 = window
    crop = image[y0:y1, x0:x1]
    crop_mask = ((mask[y0:y1, x0:x1] > 0) * np.uint8(255))

    # Mirror-pad the sides where the hole touches the image border (the mask is mirrored
    # too, so reflected glyph ink lands inside the hole, not in the context).
    pad_top, pad_bottom, pad_left, pad_right = border_pads(crop, crop_mask, window, image.shape[:2])
    frame, frame_mask = crop, crop_mask
    if pad_top or pad_bottom or pad_left or pad_right:
        frame = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        frame_mask = cv2.copyMakeBorder(
            crop_mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )

    model, budget_px = _runtime()
    frame_h, frame_w = frame.shape[:2]
    scale = work_scale(frame_w, frame_h, budget_px)
    work, work_mask = frame, frame_mask
    if scale != 1.0:
        size = (max(_PAD_MODULO, round(frame_w * scale)), max(_PAD_MODULO, round(frame_h * scale)))
        work = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        # Coverage-preserving mask resize: NEAREST drops 1-2px residue crumbs on the way
        # down, the model then sees that ink as context and hands it straight back.
        work_mask = (cv2.resize(frame_mask, size, interpolation=cv2.INTER_AREA) > 0) * np.uint8(255)

    import torch

    pad_h = -work.shape[0] % _PAD_MODULO
    pad_w = -work.shape[1] % _PAD_MODULO
    padded = np.pad(work, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    padded_mask = np.pad(work_mask, ((0, pad_h), (0, pad_w)), mode="edge")
    with _LOCK, torch.inference_mode():
        img_t = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).cuda()
        mask_t = (torch.from_numpy(padded_mask).unsqueeze(0).unsqueeze(0).cuda() > 0).float()
        out = model(img_t, mask_t)[0].permute(1, 2, 0).cpu().numpy()
    filled = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)[: work.shape[0], : work.shape[1]]
    if scale != 1.0:
        interp = cv2.INTER_LINEAR if scale < 1.0 else cv2.INTER_AREA
        filled = cv2.resize(filled, (frame_w, frame_h), interpolation=interp)
    filled = filled[pad_top : frame_h - pad_bottom, pad_left : frame_w - pad_right]

    alpha = crop_mask.astype(np.float32) / 255.0
    if _FEATHER_PX:
        k = _FEATHER_PX * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        # Feather OUTWARD only: inside the mask the fill must fully replace the original
        # (a blur-weakened interior lets the erased ink bleed back through on small
        # residue crumbs, which never reach alpha 1 at their centre).
        alpha = np.maximum(alpha, (crop_mask > 0).astype(np.float32))
    blended = crop.astype(np.float32) * (1.0 - alpha[..., None]) + filled.astype(np.float32) * alpha[..., None]
    result = image.copy()
    result[y0:y1, x0:x1] = np.round(blended).astype(np.uint8)
    return result


def _runtime() -> tuple[Any, int]:
    """Lazy singleton: (jit model on cuda, pixel budget). Loaded on the first inpaint
    render so the service starts (and every non-inpaint request runs) without torch."""
    global _MODEL, _BUDGET_PX
    with _LOCK:
        if _MODEL is None:
            # Must be set before torch's CUDA allocator initializes: without expandable
            # segments the FFT workspace churn fragments the cache and the reserved VRAM
            # overshoots the budget by ~50% (measured 2.8 vs 1.3 GiB at 1.5 Mpx).
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("erase_fill_mode='inpaint' needs a CUDA GPU (LaMa is GPU-only here)")
            settings = load_settings().inpaint
            path = Path(settings.model_path).expanduser()
            if not path.exists():
                raise RuntimeError(f"LaMa checkpoint not found at {path} (config: inpaint.model_path)")
            model = torch.jit.load(str(path), map_location="cuda")
            model.eval()
            _MODEL, _BUDGET_PX = model, settings.pixel_budget_px
    return _MODEL, _BUDGET_PX
