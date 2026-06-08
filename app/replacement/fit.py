"""Fit text into a box: the largest font size that fits, with optional wrapping.

Used by the renderer to size the translation to its target region. Single-line for
``field`` units, word-wrapped for ``flow`` units. Measurement uses the font metrics
directly (no draw context needed).
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import ImageFont


_MIN_SIZE = 6
_FONT_NAMES = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")


@dataclass(frozen=True)
class FittedText:
    font: ImageFont.FreeTypeFont
    lines: list[str]
    line_height: int


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for name in _FONT_NAMES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def fit_text(text: str, max_width: int, max_height: int, *, wrap: bool) -> FittedText:
    """Largest font (down to a floor) whose rendered lines fit max_width x max_height."""
    content = str(text or "").strip()
    start = max(_MIN_SIZE, min(int(max_height), 160))
    for size in range(start, _MIN_SIZE - 1, -1):
        font = load_font(size)
        lines = _wrap(font, content, max_width) if wrap else [content]
        line_height = _line_height(font)
        widest = max((_text_width(font, line) for line in lines), default=0)
        if widest <= max_width and line_height * len(lines) <= max_height:
            return FittedText(font=font, lines=lines, line_height=line_height)

    font = load_font(_MIN_SIZE)
    lines = _wrap(font, content, max_width) if wrap else [content]
    return FittedText(font=font, lines=lines, line_height=_line_height(font))


def _wrap(font: ImageFont.FreeTypeFont, text: str, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or _text_width(font, candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _text_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    try:
        return font.getlength(text)
    except Exception:
        left, _, right, _ = font.getbbox(text)
        return right - left


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    try:
        ascent, descent = font.getmetrics()
        return int(ascent + descent)
    except Exception:
        return int(getattr(font, "size", 12) * 1.2)
