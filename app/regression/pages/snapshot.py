"""Build a snapshot: the align structure plus the rendered image read back by OCR.

Re-OCR is bit-stable on identical pixels, so the read-back is a faithful, portable record of what
the render produced — text and position — without depending on exact pixel output across font /
library versions.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from app.core.config import OcrSettings
from app.ocr import run_raw_ocr
from app.regression.pages.fixture import Snapshot
from app.regression.pages.fixture import expected_unit_of

# Translation target codes the renderer produces, mapped to the PaddleOCR model codes (it uses
# 'ch'/'japan'/'korean', not 'zh'/'ja'/'ko' — the raw codes raise "no models available"). Latin
# codes (en, nl, fr, de, …) pass through. The re-OCR only needs a valid model read consistently;
# both capture and replay map the same way, so the comparison stays apples-to-apples.
_OCR_LANG_BY_TARGET = {"zh": "ch", "zh-cn": "ch", "zh-tw": "chinese_cht", "ja": "japan", "ko": "korean"}


def ocr_language_for_target(target_lang: str) -> str:
    """The OCR model code that reads text rendered for ``target_lang``. Public: the pdf harness
    uses the same mapping for benchmark-on-replay measurements."""
    code = str(target_lang or "").strip().lower()
    return _OCR_LANG_BY_TARGET.get(code, code or "en")


def reocr_rows(ocr_settings: OcrSettings, png_bytes: bytes, language: str) -> list[dict[str, Any]]:
    """Read a rendered PNG back with OCR -> ``[{text, left, top, width, height}]`` in reading order."""
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        handle.write(png_bytes)
        handle.flush()
        path = Path(handle.name)
        try:
            segments = run_raw_ocr(ocr_settings, path, language=ocr_language_for_target(language))
        except RuntimeError:
            # Unknown/unsupported code: fall back to the Latin model so a capture never 500s.
            # Capture and replay fall back identically, so the comparison stays consistent.
            segments = run_raw_ocr(ocr_settings, path, language="en")
    rows = [
        {
            "text": str(segment.text or ""),
            "left": int(segment.bbox["left"]),
            "top": int(segment.bbox["top"]),
            "width": int(segment.bbox["width"]),
            "height": int(segment.bbox["height"]),
        }
        for segment in segments
        if str(segment.text or "").strip()
    ]
    rows.sort(key=lambda r: (r["top"] // 10, r["left"]))
    return rows


def build_snapshot(
    ocr_settings: OcrSettings,
    *,
    units: list[dict[str, Any]],
    ignored_cells: list[int],
    rendered_png: bytes,
    target_lang: str,
) -> Snapshot:
    """``units`` are the align-output unit dicts (from a live response or a replay)."""
    return Snapshot(
        expected_units=[expected_unit_of(unit) for unit in units],
        ignored_cells=sorted(int(c) for c in ignored_cells),
        reocr=reocr_rows(ocr_settings, rendered_png, target_lang),
    )


# Box colours for the reviewer's marked-up snapshot: where a segment disappeared or moved
# (red), and where an extra segment sits in the actual (orange).
_DIFF_BOX_COLORS = {"missing": (220, 30, 30), "moved": (220, 30, 30), "extra": (235, 140, 20)}


_DIFF_BOX_PAD = 6


_DIFF_BOX_WIDTH = 3


def write_snapshot_diff(variant_path, boxes: list[dict[str, Any]]) -> None:
    """``snapshot_diff.png``: the snapshot with a box around every mismatched re-OCR segment,
    so a reviewer flipping snapshot/actual sees WHERE to look instead of searching. Only the
    snapshot copy is marked — the actual stays clean for judging (and re-baselining). Align-only
    failures yield no boxes; the unmarked copy is still written so the viewer never 404s.
    Public: the pdf document harness writes the same artifact per page dir."""
    snapshot_png = variant_path / "snapshot.png"
    if not snapshot_png.exists():
        return
    from PIL import Image
    from PIL import ImageDraw

    image = Image.open(snapshot_png).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box["left"]) - _DIFF_BOX_PAD
        top = int(box["top"]) - _DIFF_BOX_PAD
        right = int(box["left"]) + int(box["width"]) + _DIFF_BOX_PAD
        bottom = int(box["top"]) + int(box["height"]) + _DIFF_BOX_PAD
        draw.rectangle((left, top, right, bottom),
                       outline=_DIFF_BOX_COLORS.get(box.get("kind"), (220, 30, 30)),
                       width=_DIFF_BOX_WIDTH)
    image.save(variant_path / "snapshot_diff.png")
