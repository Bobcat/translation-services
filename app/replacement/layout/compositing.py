"""Warp a rendered text tile onto its plane and paste it into the canvas."""
from __future__ import annotations

import math
import numpy as np
import cv2
from app.replacement.jobs import _Job


def _composite(canvas: np.ndarray, job: _Job) -> None:
    tile = np.asarray(job.tile)
    th, tw = tile.shape[:2]
    dst = np.array(job.dst_quad, dtype=np.float32)
    height, width = canvas.shape[:2]
    x0 = max(0, int(math.floor(dst[:, 0].min())))
    y0 = max(0, int(math.floor(dst[:, 1].min())))
    x1 = min(width, int(math.ceil(dst[:, 0].max())))
    y1 = min(height, int(math.ceil(dst[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return

    src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst - np.array([x0, y0], dtype=np.float32))
    warped = cv2.warpPerspective(
        tile, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)
    )
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (roi * (1.0 - alpha) + warped[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

def _ipoint(point: tuple[float, float]) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))
