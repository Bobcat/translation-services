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
# Korean (Hangul) is in neither PingFang nor DejaVu, so it renders as tofu there. Route it to
# Noto Sans KR in the same per-user fonts dir. It is a variable font whose default instance is
# Thin, so we pin a regular weight when loading. Missing file degrades to the CJK/DejaVu chain.
_KOREAN_FONT = _GF_DIR / "NotoSansKR[wght].ttf"
_REGULAR_WEIGHT = 400
# Scripts the Latin family faces (and mostly DejaVu) render as tofu — route each to its Noto Sans
# face in the same per-user fonts dir when the text uses it. Variable fonts: their Regular instance
# is pinned at load (some, e.g. Hebrew, default to Thin). A missing file degrades to the family /
# DejaVu chain. (CJK + Hangul are handled separately above.) Each entry: (file, char ranges).
_SCRIPT_FONTS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("NotoSansArabic[wght].ttf", (("؀", "ۿ"), ("ݐ", "ݿ"), ("ࢠ", "ࣿ"))),   # Arabic (+ Urdu, Persian)
    ("NotoSansDevanagari[wght].ttf", (("ऀ", "ॿ"),)),                       # Hindi
    ("NotoSansBengali[wght].ttf", (("ঀ", "৿"),)),
    ("NotoSansThai[wght].ttf", (("฀", "๿"),)),
    ("NotoSansHebrew[wght].ttf", (("֐", "׿"),)),
    ("NotoSansTamil[wght].ttf", (("஀", "௿"),)),
)
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


def _has_hangul(text: str) -> bool:
    # Hangul syllables + Jamo + compatibility Jamo — Korean script, tofu in DejaVu and PingFang.
    return any(
        "가" <= ch <= "힣"          # U+AC00..U+D7A3 syllables
        or "ᄀ" <= ch <= "ᇿ"        # U+1100..U+11FF Jamo
        or "ㄱ" <= ch <= "ㆎ"        # U+3130..U+318E compatibility Jamo
        for ch in str(text or "")
    )


def is_cjk_text(text: str) -> bool:
    """True when the text contains CJK script (Han/Kana/CJK symbols or Hangul). The renderer
    uses this to size CJK lines tighter — their glyphs fill the em, unlike upper-biased Latin."""
    return _has_cjk(text) or _has_hangul(text)


@lru_cache(maxsize=1)
def _cjk_font_path() -> str | None:
    """Path to a CJK-capable font, reusing the one PaddleX ships (downloaded on first
    use, then cached). None if unavailable — CJK then falls back to tofu, as before."""
    try:
        from paddlex.utils.fonts import PINGFANG_FONT

        return str(PINGFANG_FONT.path)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _korean_font_path() -> str | None:
    """Path to a Hangul-capable font (Noto Sans KR in the per-user fonts dir), or None when it
    has not been provisioned — Korean then falls back to the CJK font / tofu."""
    return str(_KOREAN_FONT) if _KOREAN_FONT.exists() else None


def _load_korean_font(size: int) -> ImageFont.FreeTypeFont | None:
    """Noto Sans KR pinned to a regular weight (a variable font; its default instance is Thin)."""
    path = _korean_font_path()
    if not path:
        return None
    try:
        font = ImageFont.truetype(path, size)
        try:
            font.set_variation_by_axes([_REGULAR_WEIGHT])
        except Exception:
            pass
        return font
    except Exception:
        return None


def _dominant_script(text: str) -> str | None:
    """The Noto font file for the complex script (Arabic / Devanagari / Bengali / Thai / Hebrew /
    Tamil) covering the MOST of ``text``, or None. Majority, not first match: the Devanagari danda
    "।" (U+0964/0965) ends Bengali, Tamil, … sentences too, so a first-match scan would send a whole
    Bengali line to the Devanagari face (tofu) on that one shared char — counting keeps the line on
    its own script, the danda being one char against its many letters."""
    best_file: str | None = None
    best_count = 0
    for font_file, ranges in _SCRIPT_FONTS:
        count = sum(lo <= ch <= hi for ch in text for lo, hi in ranges)
        if count > best_count:
            best_file, best_count = font_file, count
    return best_file


def _load_script_font(text: str, size: int) -> ImageFont.FreeTypeFont | None:
    """The complex-script face for ``text`` (see _dominant_script), pinned to its Regular instance,
    else None so the caller falls back to the family / DejaVu chain. Missing file -> None."""
    font_file = _dominant_script(text)
    if font_file is None:
        return None
    path = _GF_DIR / font_file
    if not path.exists():
        return None
    try:
        font = ImageFont.truetype(str(path), size)
        try:
            font.set_variation_by_name("Regular")
        except Exception:
            pass
        return font
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
    """Font for a rendered line, by script. Korean (Hangul) uses Noto Sans KR; other CJK uses the
    CJK font; a complex script (Arabic/Devanagari/Bengali/Thai/Hebrew/Tamil) uses its Noto Sans
    face — all override the family, which has no such glyphs. Otherwise the VLM ``family``/
    ``weight`` map to an installed face (bold cut at high weight); with no family hint, or a missing
    mapped file, we fall back to DejaVu — preserving the previous behaviour for unlabeled lines."""
    size = max(_MIN_SIZE, int(size))
    if _has_hangul(text):
        korean = _load_korean_font(size)
        if korean is not None:
            return korean
        # Noto Sans KR not provisioned: fall through to the CJK font (tofu for Hangul, graceful).
    if _has_cjk(text) or _has_hangul(text):
        cjk = _cjk_font_path()
        return _first_loadable((cjk, *_FONT_NAMES) if cjk else _FONT_NAMES, size)
    script = _load_script_font(text, size)
    if script is not None:
        return script
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


# Kinsoku: closing punctuation must not START a wrapped line (glue it to the char before),
# opening punctuation must not END one (glue it to the char after).
_CJK_CLOSING = set("。、，．！？：；）」』】｝〕》〉")
_CJK_OPENING = set("（「『【〔《〈｛")


def break_pieces(text: str) -> list[tuple[str, str]]:
    """Atomic wrap units as (piece, glue) — ``glue`` is the separator to re-insert before the
    piece when it is not at a line start. Han/Kana/CJK-symbol chars each break individually
    (the script has no spaces); closing punctuation is kept on the preceding char and opening
    punctuation on the following one (kinsoku), so a line never starts with 。）」 nor ends with
    （「. Everything else (Latin, Hangul, digits) stays a whitespace-delimited word, so those
    scripts wrap exactly as before."""
    pieces: list[tuple[str, str]] = []
    word = ""
    word_glue = ""
    pending_space = False
    pending_open = ""      # opening punctuation held to prepend to the next piece
    pending_open_glue = ""

    def emit(piece: str, glue: str) -> None:
        nonlocal pending_open, pending_open_glue
        if pending_open:
            piece, glue = pending_open + piece, pending_open_glue
            pending_open, pending_open_glue = "", ""
        pieces.append((piece, glue))

    for ch in text:
        if ch.isspace():
            if word:
                emit(word, word_glue)
                word = ""
            pending_space = True
            continue
        glue = " " if pending_space else ""
        pending_space = False
        if _has_cjk(ch):
            if word:
                emit(word, word_glue)
                word = ""
            if ch in _CJK_OPENING:
                if not pending_open:
                    pending_open_glue = glue
                pending_open += ch
            elif ch in _CJK_CLOSING and pending_open:
                pending_open += ch  # e.g. "（）" — keep together for the next piece
            elif ch in _CJK_CLOSING and pieces:
                prev, prev_glue = pieces[-1]
                pieces[-1] = (prev + ch, prev_glue)
            else:
                emit(ch, glue)
        elif word:
            word += ch
        else:
            word, word_glue = ch, glue
    if word:
        emit(word, word_glue)
    if pending_open:  # text ended with opening punctuation (degenerate) — emit as its own piece
        pieces.append((pending_open, pending_open_glue))
    return pieces


def wrap_lines(font: ImageFont.FreeTypeFont, text: str, max_width: int) -> list[str]:
    pieces = break_pieces(text)
    if not pieces:
        return [""]
    lines: list[str] = []
    current = ""
    for piece, glue in pieces:
        candidate = current + (glue if current else "") + piece
        if not current or _text_width(font, candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = piece
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
