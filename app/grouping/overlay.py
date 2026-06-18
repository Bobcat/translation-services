"""Debug overlay for stage #5 grouping.

Draws, on top of the input image, what the VLM decided:

- each translation unit in its own colour (member boxes filled, union box outlined,
  a ``order`` label);
- ``translate: false`` members outlined in red (kept as-is, not re-rendered);
- ignored cells in grey with a cross.

Visual inspection tool — this is how we judge grouping quality over the testset
without scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from app.grouping.units import GroupingResult


@dataclass(frozen=True)
class GroupingOverlayDebug:
    image: bytes | None
    metadata: dict[str, Any]
    mime_type: str = "image/png"


_UNIT_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (23, 190, 207),
    (188, 189, 34),
]
_RED = (214, 39, 40)
_GREY = (127, 127, 127)


def render_grouping_overlay_debug(
    *,
    input_path: Path,
    result: GroupingResult,
    cells: list[dict[str, Any]],
) -> GroupingOverlayDebug:
    if not result.units and not result.ignored_cell_ids:
        return GroupingOverlayDebug(
            image=None,
            metadata={
                "grouping_overlay_debug_applied": False,
                "grouping_overlay_debug_reason": "no_units",
            },
        )

    from PIL import Image
    from PIL import ImageDraw

    with Image.open(input_path) as original:
        base = original.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    short_side = min(base.size)
    line_width = max(2, round(short_side / 400))
    font = _load_font(max(14, round(short_side / 45)))

    cell_bbox = {int(cell["id"]): dict(cell.get("bbox") or {}) for cell in cells if "id" in cell}

    for index, unit in enumerate(result.units):
        color = _UNIT_COLORS[index % len(_UNIT_COLORS)]
        for member in unit.members:
            box = _xy(member.bbox)
            if box is None:
                continue
            draw.rectangle(box, fill=color + (50,))
            outline = _RED if not member.translate else color
            draw.rectangle(box, outline=outline + (255,), width=line_width)
        union = _xy(unit.bbox)
        if union is not None:
            draw.rectangle(union, outline=color + (255,), width=line_width + 1)
            _label(draw, union, f"{unit.order}", color, font)

    for cell_id in result.ignored_cell_ids:
        box = _xy(cell_bbox.get(int(cell_id), {}))
        if box is None:
            continue
        draw.rectangle(box, fill=_GREY + (40,), outline=_GREY + (220,), width=line_width)
        draw.line([box[0], box[1]], fill=_GREY + (220,), width=line_width)
        draw.line([(box[0][0], box[1][1]), (box[1][0], box[0][1])], fill=_GREY + (220,), width=line_width)

    out = BytesIO()
    Image.alpha_composite(base, overlay).convert("RGB").save(out, format="PNG", compress_level=1)
    return GroupingOverlayDebug(
        image=out.getvalue(),
        metadata={
            "grouping_overlay_debug_applied": True,
            "grouping_overlay_debug_unit_count": len(result.units),
            "grouping_overlay_debug_ignored_count": len(result.ignored_cell_ids),
        },
    )


def _xy(bbox: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if not bbox:
        return None
    left = int(bbox.get("left") or 0)
    top = int(bbox.get("top") or 0)
    right = left + int(bbox.get("width") or 0)
    bottom = top + int(bbox.get("height") or 0)
    if right <= left or bottom <= top:
        return None
    return (left, top), (right, bottom)


def _label(draw: Any, union: tuple[tuple[int, int], tuple[int, int]], text: str, color: tuple[int, int, int], font: Any) -> None:
    (left, top), _ = union
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = len(text) * 7, 12
    pad = 2
    draw.rectangle(
        [(left, top), (left + text_w + 2 * pad, top + text_h + 2 * pad)],
        fill=color + (235,),
    )
    draw.text((left + pad, top + pad), text, fill=(255, 255, 255, 255), font=font)


def _load_font(size: int) -> Any:
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()
