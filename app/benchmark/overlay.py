"""Overlay renders for the benchmark detail view: the run's page with its
layout regions drawn in match-status colours (matched green, lost red,
invented orange), so a surprising layout score is explainable at a glance.

Renders on demand from the run's stored PDF at the measurement's dpi and the
frozen regions in measurement.json, then caches under ``<run>/overlays/`` —
the measurement never changes for a stored run, so the cache never expires.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image
from PIL import ImageDraw

from app.benchmark.scoring import SCORING_VERSION
from app.benchmark.scoring import page_region_statuses
from app.benchmark.store import BenchmarkRun

_STATUS_COLORS = {
    "matched": (22, 163, 74),    # green
    "covered": (37, 99, 235),    # blue: detector granularity (split/merge/nested), not scored
    "lost": (220, 38, 38),       # red
    "invented": (234, 88, 12),   # orange
}
_FILL_ALPHA = 36
_OUTLINE_WIDTH = 4


def overlay_png(run: BenchmarkRun, *, side: str, page_index: int) -> bytes:
    """``side`` is "source" or "translated"; raises ValueError on bad input."""
    if side not in ("source", "translated"):
        raise ValueError("side must be source or translated")
    # Statuses depend on the scoring version, so the cache is keyed by it: a
    # rescore must never serve yesterday's overlay next to today's numbers.
    cache_path = run.path / "overlays" / f"v{SCORING_VERSION}-{side}-page-{page_index + 1:03d}.png"
    if cache_path.exists():
        return cache_path.read_bytes()

    measurement = run.load_measurement()
    src_pages = list((measurement.get("source") or {}).get("pages") or [])
    tgt_pages = list((measurement.get("translated") or {}).get("pages") or [])
    if page_index < 0 or page_index >= min(len(src_pages), len(tgt_pages)):
        raise ValueError("page index out of range")
    statuses = page_region_statuses(src_pages[page_index], tgt_pages[page_index])[side]

    pdf_path = run.path / ("source.pdf" if side == "source" else "translated.pdf")
    dpi = int(measurement.get("analysis_dpi") or 160)
    doc = pymupdf.open(str(pdf_path))
    try:
        pixmap = doc[page_index].get_pixmap(dpi=dpi)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples).convert("RGBA")
    finally:
        doc.close()

    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for region in statuses:
        color = _STATUS_COLORS[region["status"]]
        box = [round(v) for v in region["box"]]
        draw.rectangle(box, fill=(*color, _FILL_ALPHA), outline=(*color, 255), width=_OUTLINE_WIDTH)
        text = region["label"] + (f" {region['iou']:.2f}" if region.get("iou") is not None else "")
        draw.text((box[0] + 6, max(2, box[1] - 14)), text, fill=(*color, 255))
    composed = Image.alpha_composite(image, layer).convert("RGB")

    out = BytesIO()
    composed.save(out, format="PNG", compress_level=6)
    payload = out.getvalue()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    return payload
