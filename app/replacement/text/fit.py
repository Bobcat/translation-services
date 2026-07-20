"""Fit text into a box: the largest font size that fits, with optional wrapping.

Used by the renderer to size the translation to its target region. Measurement uses
the font metrics directly (no draw context needed).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageFont


_MIN_SIZE = 6
# Ultimate fallback (no family hint, or a mapped font is missing). Regular first, bold
# second. DejaVu has no CJK glyphs, so CJK text is drawn in the CJK font instead.
_FONT_NAMES = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
# Italic counterpart of the DejaVu fallback, for text-layer-flagged italic on a unit
# WITHOUT a family hint (a leftover cell: a figure label like an emphasised method name).
_FONT_NAMES_ITALIC = ("DejaVuSans-Oblique.ttf", *_FONT_NAMES)

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
# Italic cuts of the same faces, same (regular, bold) shape. Selected when the text
# layer flags the group's text italic (OCR carries no style axis, so scanned images
# never take this table); a missing file degrades to the roman cut, not to DejaVu.
_FAMILY_ITALIC_FONTS = {
    "sans": ("Arimo-Italic[wght].ttf", None),
    "serif": ("Tinos-Italic.ttf", "Tinos-BoldItalic.ttf"),
    "mono": ("Cousine-Italic.ttf", "Cousine-BoldItalic.ttf"),
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
_SERIF_HINTS = (
    "times", "georgia", "serif", "garamond", "minion", "cambria", "palatino",
    "baskerville", "didot", "book antiqua",
    # The academic (LaTeX) serifs: the VLM names "Computer Modern" on every classic paper,
    # and the text layer's own faces are Latin/Nimbus clones of it. Unmapped these fell to
    # the SANS fallback, flipping a serif paper's whole character (islands design doc:
    # transplanted serif formulas sat inside sans prose).
    "computer modern", "latin modern", "lmroman", "cmr", "nimbus rom", "nimbusrom",
)
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


def _has_cjk_letters(text: str) -> bool:
    # Actual Han/Kana glyphs — NOT the symbol/fullwidth ranges _has_cjk also spans.
    return any(
        "぀" <= ch <= "ヿ"
        or "㐀" <= ch <= "鿿"
        or "豈" <= ch <= "﫿"
        for ch in str(text or "")
    )


def fold_lone_fullwidth_punctuation(text: str) -> str:
    """Fold fullwidth/CJK punctuation to its ASCII form in a line that carries NO CJK letters.

    A model translating out of Japanese/Chinese often keeps one fullwidth mark ("DANGER！");
    that single character makes ``is_cjk_text`` true, which shrinks the whole group to the CJK
    size ratio and reroutes the line to the CJK face — a ~20% size drop and a family change off
    one punctuation glyph. Real CJK text (any Han/Kana/Hangul letter present) is left untouched:
    its fullwidth punctuation is correct. Only marks whose NFKC image is ASCII fold ("！"→"!",
    ideographic space→space); an unfoldable "。" stays and the line then still routes to the CJK
    face that has its glyph — tofu is never a possible outcome of this fold."""
    value = str(text or "")
    if not value or _has_cjk_letters(value) or _has_hangul(value):
        return value
    out: list[str] = []
    changed = False
    for ch in value:
        if "　" <= ch <= "〿" or "＀" <= ch <= "￯":
            folded = unicodedata.normalize("NFKC", ch)
            if folded.isascii():
                out.append(folded)
                changed = True
                continue
        out.append(ch)
    return "".join(out) if changed else value


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


@lru_cache(maxsize=8)
def _coverage_probe(path: str) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(path, 32)
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _covers(path: str, ch: str) -> bool:
    """Whether the face at ``path`` has a real glyph for ``ch``, probed by comparing its
    rendered bitmap against the face's .notdef box (U+0378 is permanently unassigned, so it
    always renders .notdef) — no cmap parser or font-lib dependency. A face that maps a
    missing char to a BLANK glyph (instead of the box) reads as covered and renders blank;
    that face would never have shown a tofu box either, so nothing is lost."""
    probe = _coverage_probe(path)
    if probe is None:
        return True  # unprobeable: assume covered, keeping the previous behaviour
    try:
        notdef = Image.Image()._new(probe.getmask("͸"))
        mask = Image.Image()._new(probe.getmask(ch))
    except Exception:
        return True
    return mask.size != notdef.size or mask.tobytes() != notdef.tobytes()


class FallbackFont:
    """A primary face plus a fallback face for the characters the primary has no glyph for.

    The mapped family faces (Tinos/Arimo/Cousine) cover Latin but not the misc-symbol
    blocks — the ⋆ ⋄ ✦ ♡ affiliation markers of an academic paper render as tofu boxes.
    DejaVu covers them, so those characters draw in DejaVu while the words around them keep
    the family face. Quacks like FreeTypeFont for the renderer's call sites (``getlength`` /
    ``getmetrics`` / ``.size``); ``draw_text`` consumes ``runs`` to draw each segment in its
    own face. Constructed by ``load_font`` only when the text actually carries uncovered
    characters, so covered-only text keeps the exact previous single-face path."""

    def __init__(self, primary: ImageFont.FreeTypeFont, fallback: ImageFont.FreeTypeFont):
        self.primary = primary
        self.fallback = fallback

    @property
    def size(self) -> int:
        return self.primary.size

    def getmetrics(self) -> tuple[int, int]:
        return self.primary.getmetrics()

    def runs(self, text: str) -> list[tuple[ImageFont.FreeTypeFont, str]]:
        """Maximal consecutive segments, each with the face that has its glyphs."""
        out: list[list] = []
        for ch in str(text):
            face = self.fallback if self._needs_fallback(ch) else self.primary
            if out and out[-1][0] is face:
                out[-1][1] += ch
            else:
                out.append([face, ch])
        return [(face, segment) for face, segment in out]

    def _needs_fallback(self, ch: str) -> bool:
        if ord(ch) < 128:  # ASCII is covered by every mapped face
            return False
        primary_path = getattr(self.primary, "path", None)
        fallback_path = getattr(self.fallback, "path", None)
        if not primary_path or not fallback_path or _covers(primary_path, ch):
            return False
        return _covers(fallback_path, ch)

    def getlength(self, text: str) -> float:
        return sum(face.getlength(segment) for face, segment in self.runs(text))


_ISLAND_TOKEN_RE = re.compile(r"⟦M\d+⟧")  # ⟦Mn⟧ inline pixel island


class IslandFont:
    """A face plus fixed-width inline pixel islands (islands design doc, phase 4).

    A ⟦Mn⟧ token measures at its island's source width scaled to the face size, and
    draws as the island's ink MASK in the line's fill colour (``draw.bitmap``), so a
    transplanted formula condenses, justifies and recolours exactly like the glyphs
    around it. Quacks like FreeTypeFont for the render call sites, following
    ``FallbackFont``; text segments recurse through ``draw_text``, so an island line
    keeps the symbol-fallback behaviour of its face."""

    def __init__(self, inner: Any, islands: dict[str, dict[str, Any]]):
        self.inner = inner
        # id -> {"mask": L image at source px, "w"/"h": source px, "dy": px below the
        # line's ink top, "declared": the line's em size in source px}
        self.islands = islands

    @property
    def size(self) -> int:
        return self.inner.size

    def getmetrics(self) -> tuple[int, int]:
        return self.inner.getmetrics()

    def scale_for(self, island: dict[str, Any]) -> float:
        declared = float(island.get("declared") or 0.0)
        return float(self.inner.size) / declared if declared > 0 else 1.0

    def segments(self, text: str) -> list[tuple[str, str]]:
        """``[("text"|"island", value)]`` in order; island values are the ⟦Mn⟧ tokens."""
        out: list[tuple[str, str]] = []
        pos = 0
        for match in _ISLAND_TOKEN_RE.finditer(text):
            if match.start() > pos:
                out.append(("text", text[pos:match.start()]))
            out.append(("island", match.group(0)))
            pos = match.end()
        if pos < len(text):
            out.append(("text", text[pos:]))
        return out

    def getlength(self, text: str) -> float:
        total = 0.0
        for kind, value in self.segments(str(text)):
            if kind == "island":
                island = self.islands.get(value[1:-1])
                total += (
                    float(island["w"]) * self.scale_for(island)
                    if island is not None
                    else self.inner.getlength(value)
                )
            else:
                total += self.inner.getlength(value)
        return total


def draw_text(draw: Any, xy: tuple[float, float], text: str, font: Any, fill: Any) -> None:
    """``draw.text`` that understands ``FallbackFont``: each run draws in its own face,
    x-advanced by the run's width and baseline-aligned to the primary (the faces' ascents
    differ a few px at the same pt size). ``IslandFont`` segments draw text through the
    inner face and islands as ink masks. A plain font takes the plain path."""
    if isinstance(font, IslandFont):
        x, y = xy
        ascent, descent = font.getmetrics()
        text_h = max(1, int(ascent + descent))
        for kind, value in font.segments(str(text)):
            if kind == "island":
                island = font.islands.get(value[1:-1])
                if island is None:
                    continue
                scale = font.scale_for(island)
                advance = float(island["w"]) * scale
                w = max(1, int(round(float(island["w"]) * scale)))
                h = max(1, int(round(float(island["h"]) * scale)))
                dy = int(round(float(island.get("dy") or 0.0) * scale))
                if h > text_h:  # a radical/stacked script taller than the line: squeeze into the box
                    w = max(1, int(round(w * text_h / h)))
                    h = text_h
                dy = max(0, min(dy, text_h - h))
                mask = island["mask"].resize((w, h), Image.LANCZOS)
                draw.bitmap((int(round(x)), int(round(y + dy))), mask, fill=fill)
                x += advance
            else:
                draw_text(draw, (x, y), value, font.inner, fill)
                x += font.inner.getlength(value)
        return
    if not isinstance(font, FallbackFont):
        draw.text(xy, text, font=font, fill=fill)
        return
    x, y = xy
    primary_ascent = font.primary.getmetrics()[0]
    for face, segment in font.runs(text):
        draw.text((x, y + primary_ascent - face.getmetrics()[0]), segment, font=face, fill=fill)
        x += face.getlength(segment)


def _with_symbol_fallback(font: ImageFont.FreeTypeFont, text: str, size: int) -> Any:
    """Wrap ``font`` in a ``FallbackFont`` when ``text`` carries characters the face has no
    glyph for and the DejaVu fallback does; otherwise return ``font`` unchanged (the no-op
    path for every covered-only line)."""
    path = getattr(font, "path", None)
    if not path:
        return font
    missing = {ch for ch in str(text) if ord(ch) >= 128 and not _covers(path, ch)}
    if not missing:
        return font
    fallback = _first_loadable(_FONT_NAMES, size)
    fallback_path = getattr(fallback, "path", None)
    if not fallback_path or fallback_path == path:
        return font
    if not any(_covers(fallback_path, ch) for ch in missing):
        return font
    return FallbackFont(font, fallback)


@dataclass(frozen=True)
class FittedText:
    font: ImageFont.FreeTypeFont
    lines: list[str]
    line_height: int


def load_font(
    size: int, text: str = "", *, family: str | None = None, weight: int | None = None,
    italic: bool = False,
) -> ImageFont.FreeTypeFont:
    """Font for a rendered line, by script. Korean (Hangul) uses Noto Sans KR; other CJK uses the
    CJK font; a complex script (Arabic/Devanagari/Bengali/Thai/Hebrew/Tamil) uses its Noto Sans
    face — all override the family, which has no such glyphs. Otherwise the VLM ``family``/
    ``weight`` map to an installed face (bold cut at high weight); with no family hint, or a missing
    mapped file, we fall back to DejaVu — preserving the previous behaviour for unlabeled lines.
    A mapped face missing glyphs the text needs (misc symbols) gets a per-character DejaVu
    fallback (``FallbackFont``) instead of tofu boxes."""
    size = max(_MIN_SIZE, int(size))
    if _has_hangul(text):
        korean = _load_korean_font(size)
        if korean is not None:
            return korean
        # Noto Sans KR not provisioned: fall through to the CJK font (tofu for Hangul, graceful).
    if _has_cjk(text) or _has_hangul(text):
        cjk = _cjk_font_path()
        font = _first_loadable((cjk, *_FONT_NAMES) if cjk else _FONT_NAMES, size)
        # A CJK-mixed line's Latin run can carry a glyph the CJK face lacks (PingFang has no
        # U+0141 "Ł" — the Łukasz author refs dropped their initial). Same per-character DejaVu
        # fallback the mapped path already applies, so the missing glyph draws instead of vanishing.
        return _with_symbol_fallback(font, text, size)
    script = _load_script_font(text, size)
    if script is not None:
        return script
    mapped = _load_mapped_font(family, weight, size, italic=italic)
    if mapped is not None:
        return _with_symbol_fallback(mapped, text, size)
    return _first_loadable(_FONT_NAMES_ITALIC if italic else _FONT_NAMES, size)


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


def _load_mapped_font(
    family: str | None, weight: int | None, size: int, *, italic: bool = False
) -> ImageFont.FreeTypeFont | None:
    """Mapped face for a VLM family/weight, or None to let the caller fall back to DejaVu."""
    if not str(family or "").strip():
        return None
    category = _font_category(family)
    tables = (_FAMILY_ITALIC_FONTS, _FAMILY_FONTS) if italic else (_FAMILY_FONTS,)
    want_bold = weight is not None and int(weight) >= _BOLD_THRESHOLD
    for table in tables:
        regular, bold = table[category]
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
            continue
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


# A separator that must keep its WIDTH through the wrap instead of collapsing to a word space:
# exactly one em in every mapped face, blank (no glyph outline), and the width print uses between
# a section number and its heading. ``break_pieces`` is where a run of whitespace becomes glue, so
# it is the one place that has to know the difference.
EM_SPACE = " "


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
    pending_wide = False    # the pending separator was an EM space: keep its width, don't collapse it
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
            pending_wide = pending_wide or ch == EM_SPACE
            continue
        glue = ((EM_SPACE if pending_wide else " ") if pending_space else "")
        pending_space = False
        pending_wide = False
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
