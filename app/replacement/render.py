"""Tier-1 re-placement: render translated units back into the image (simple replace).

For each unit with a translation: take the region of its translatable members,
sample the background colour, fill (erase) that region, and draw the fitted
translation in a contrasting colour. ``field`` units are single-line; ``flow`` units
are word-wrapped over the union region. Non-translatable members (`translate: false`)
and ignored cells are never touched. Upright only — rotation/perspective is a later
iteration (see docs/re-placement.md).
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw

from app.replacement.color import sample_region_colors
from app.replacement.fit import FittedText
from app.replacement.fit import fit_text


def render_translated_image(input_path: Path, translation_units: list[dict[str, Any]]) -> bytes:
    base = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(base)

    for unit in translation_units:
        translated = str(unit.get("translated_text") or "").strip()
        if not translated:
            continue
        members = unit.get("members") or []
        translatable = [m for m in members if m.get("translate") and m.get("bbox")]
        region = _union_bbox([m["bbox"] for m in translatable])
        if region is None:
            continue

        bg, fg = sample_region_colors(base, region)
        region_box = _xy(region)
        if region_box is not None:
            draw.rectangle(region_box, fill=bg)

        wrap = str(unit.get("kind") or "field") == "flow"
        fitted = fit_text(translated, region["width"], region["height"], wrap=wrap)
        _draw_lines(draw, fitted, region, fg)

    out = BytesIO()
    base.save(out, format="PNG")
    return out.getvalue()


def _draw_lines(draw: ImageDraw.ImageDraw, fitted: FittedText, region: dict[str, int], fg: tuple[int, int, int]) -> None:
    total_height = fitted.line_height * len(fitted.lines)
    y = region["top"] + max(0, (region["height"] - total_height) // 2)
    for line in fitted.lines:
        line_width = int(fitted.font.getlength(line))
        x = region["left"] + max(0, (region["width"] - line_width) // 2)
        draw.text((x, y), line, font=fitted.font, fill=fg)
        y += fitted.line_height


def _union_bbox(bboxes: list[dict[str, Any]]) -> dict[str, int] | None:
    valid = [bbox for bbox in bboxes if bbox]
    if not valid:
        return None
    left = min(int(bbox.get("left") or 0) for bbox in valid)
    top = min(int(bbox.get("top") or 0) for bbox in valid)
    right = max(int(bbox.get("left") or 0) + int(bbox.get("width") or 0) for bbox in valid)
    bottom = max(int(bbox.get("top") or 0) + int(bbox.get("height") or 0) for bbox in valid)
    return {"left": left, "top": top, "width": max(1, right - left), "height": max(1, bottom - top)}


def _xy(bbox: dict[str, Any]) -> tuple[int, int, int, int] | None:
    left = int(bbox.get("left") or 0)
    top = int(bbox.get("top") or 0)
    right = left + int(bbox.get("width") or 0)
    bottom = top + int(bbox.get("height") or 0)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)
