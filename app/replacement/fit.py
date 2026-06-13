"""Fit text into a box: the largest font size that fits, with optional wrapping.

Used by the renderer to size the translation to its target region. Measurement uses
the font metrics directly (no draw context needed).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont


_MIN_SIZE = 6
# Ultimate fallback (no family hint, or a mapped font is missing). Regular first, bold
# second. DejaVu has no CJK glyphs, so CJK text is drawn in the CJK font instead.
_FONT_NAMES = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")

# The VLM names a font per element (e.g. "Helvetica", "Times New Roman", "Courier New").
# We don't have those exact (proprietary) faces, so map by CATEGORY to a metric-compatible
# installed face: sans -> Arimo (Arial/Helvetica metrics), serif -> Tinos (Times), mono ->
# Cousine (Courier). The VLM wavers on the exact name (Georgia vs Times) but stays within a
# category, so category mapping is the robust target.
# NOTE: these live in the per-user fonts dir for now (no system install / repo vendoring
# decided yet); a missing file degrades gracefully to the DejaVu fallback above.
_GF_DIR = Path("~/.local/share/fonts/gf").expanduser()
_BOLD_THRESHOLD = 600  # VLM font-weight (100-900) at/above which we pick the bold cut
# category -> (regular file, bold file). bold None = the regular file is a variable font;
# push its weight axis instead of loading a separate cut.
_FAMILY_FONTS = {
    "sans": ("Arimo[wght].ttf", None),
    "serif": ("Tinos-Regular.ttf", "Tinos-Bold.ttf"),
    "mono": ("Cousine-Regular.ttf", "Cousine-Bold.ttf"),
}
_SERIF_HINTS = ("times", "georgia", "serif", "garamond", "minion", "cambria", "palatino", "baskerville", "didot", "book antiqua")
_MONO_HINTS = ("courier", "mono", "consol", "menlo", "monaco", "typewriter")


def _has_cjk(text: str) -> bool:
    # Han + Kana + CJK symbols/fullwidth — the ranges DejaVu renders as tofu.
    return any(
        "　" <= ch <= "〿"
        or "぀" <= ch <= "ヿ"
        or "㐀" <= ch <= "鿿"
        or "豈" <= ch <= "﫿"
        or "＀" <= ch <= "￯"
        for ch in str(text or "")
    )


@lru_cache(maxsize=1)
def _cjk_font_path() -> str | None:
    """Path to a CJK-capable font, reusing the one PaddleX ships (downloaded on first
    use, then cached). None if unavailable — CJK then falls back to tofu, as before."""
    try:
        from paddlex.utils.fonts import PINGFANG_FONT

        return str(PINGFANG_FONT.path)
    except Exception:
        return None


@dataclass(frozen=True)
class FittedText:
    font: ImageFont.FreeTypeFont
    lines: list[str]
    line_height: int


def load_font(
    size: int, text: str = "", *, family: str | None = None, weight: int | None = None
) -> ImageFont.FreeTypeFont:
    """Font for a rendered line. CJK text always uses the CJK font (it overrides the family,
    which has no CJK glyphs). Otherwise the VLM ``family``/``weight`` map to an installed
    face (bold cut at high weight); with no family hint, or a missing mapped file, we fall
    back to DejaVu — preserving the previous behaviour for unlabeled lines."""
    size = max(_MIN_SIZE, int(size))
    if _has_cjk(text):
        cjk = _cjk_font_path()
        return _first_loadable((cjk, *_FONT_NAMES) if cjk else _FONT_NAMES, size)
    mapped = _load_mapped_font(family, weight, size)
    if mapped is not None:
        return mapped
    return _first_loadable(_FONT_NAMES, size)


def _first_loadable(names: tuple[str | None, ...], size: int) -> ImageFont.FreeTypeFont:
    for name in names:
        if not name:
            continue
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _font_category(family: str) -> str:
    lowered = family.lower()
    if any(hint in lowered for hint in _MONO_HINTS):
        return "mono"
    if any(hint in lowered for hint in _SERIF_HINTS):
        return "serif"
    return "sans"


def _load_mapped_font(family: str | None, weight: int | None, size: int) -> ImageFont.FreeTypeFont | None:
    """Mapped face for a VLM family/weight, or None to let the caller fall back to DejaVu."""
    if not str(family or "").strip():
        return None
    regular, bold = _FAMILY_FONTS[_font_category(family)]
    want_bold = weight is not None and int(weight) >= _BOLD_THRESHOLD
    try:
        if want_bold and bold is not None:
            return ImageFont.truetype(str(_GF_DIR / bold), size)
        font = ImageFont.truetype(str(_GF_DIR / regular), size)
        if want_bold:  # variable family with no separate bold cut -> push the weight axis
            try:
                font.set_variation_by_axes([int(weight)])
            except Exception:
                pass
        return font
    except Exception:
        return None


def fit_text(
    text: str,
    max_width: int,
    max_height: int,
    *,
    wrap: bool,
    max_size: int | None = None,
    max_lines: int | None = None,
) -> FittedText:
    """Largest font (down to a floor) whose rendered lines fit the constraints.

    ``max_size`` caps the starting size — pass the original text height so the
    translation stays at roughly the original scale and only shrinks to fit.
    ``max_lines`` caps the line count — pass the original block's line count to keep
    the translation in the same number of lines (it shrinks the font to do so), so the
    re-placed text keeps the original cadence instead of growing extra lines.
    """
    content = str(text or "").strip()
    ceiling = int(max_size) if max_size else int(max_height)
    start = max(_MIN_SIZE, min(ceiling, 160))
    for size in range(start, _MIN_SIZE - 1, -1):
        font = load_font(size, content)
        lines = wrap_lines(font, content, max_width) if wrap else [content]
        line_height = _line_height(font)
        widest = max((_text_width(font, line) for line in lines), default=0)
        fits_box = widest <= max_width and line_height * len(lines) <= max_height
        fits_lines = max_lines is None or len(lines) <= max_lines
        if fits_box and fits_lines:
            return FittedText(font=font, lines=lines, line_height=line_height)

    font = load_font(_MIN_SIZE, content)
    lines = wrap_lines(font, content, max_width) if wrap else [content]
    return FittedText(font=font, lines=lines, line_height=_line_height(font))


def wrap_lines(font: ImageFont.FreeTypeFont, text: str, max_width: int) -> list[str]:
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
