"""Break and condense a group's translated text onto its planes."""
from __future__ import annotations

from typing import Any
from app.replacement.text.fit import IslandFont
from app.replacement.text.fit import load_font
from app.replacement.text.fit import break_pieces


# Floor on horizontal condensation. The font is sized from the source HEIGHT (so the
# header/body hierarchy is preserved); a translated line at that height is usually wider
# than its original (most sans are wider than the sign's font), so the rendered text is
# squeezed horizontally to fit the original line's width — keeping height, matching width,
# the way the reference render does. Never squeeze past this floor: below it the glyphs
# read as unnaturally narrow, so the pt size is reduced instead (see _WIDTH_SLACK).
_CONDENSE_FLOOR = 0.75

# A rendered line may exceed its original plane width by this factor before we spend pt.
# Order of accommodation for a too-long translation: condense horizontally to the floor,
# then allow up to this much overrun, and only if it STILL doesn't fit reduce the source
# pt size (re-wrapping) — so the source size (and the header/body hierarchy) is preserved
# unless the line genuinely cannot fit the box within the slack.
_WIDTH_SLACK = 1.04
# ...but the slack is only spendable over VERIFIED clean background: on axis-aligned flat
# groups the planner measures the pixels right of each plane (same scan as the
# "extend_to_margin" width fit) and stores the verified run as ``plane["slack_px"]`` — a colour step (a newsletter's
# sidebar panel), another unit's cell or any ink caps it. Without that evidence (tilt, centered
# groups, planes built in tests) the plain 4% stands: 4% of a page-wide line is tens of pixels,
# which crossed visibly into an adjacent layout panel the original kept a margin to.


def _allowed_width(plane: dict[str, Any], plane_width: float) -> float:
    """The width a rendered line may occupy on this plane before pt is spent: the plane width
    plus the 4% slack, capped at the plane's verified-clean right run when the planner
    measured one."""
    slack = plane_width * (_WIDTH_SLACK - 1.0)
    verified = plane.get("slack_px")
    if verified is not None:
        slack = min(slack, float(verified))
    return plane_width + slack

def _fit_group(
    text: str,
    *,
    size: int,
    plane_widths: list[float],
    family: str | None = None,
    weight: int | None = None,
    italic: bool = False,
    islands: dict[str, Any] | None = None,
    justified: bool = False,
    hyphenator: Any = None,
) -> tuple[Any, list[str]]:
    """Render at the source ``size`` (true line height) in the unit's VLM font ``family`` /
    ``weight``, wrapped so each line fits the width of the plane it lands on (``plane_widths``).
    The font is NOT reduced to fit width — the source size, and thus the header/body hierarchy,
    is preserved; width is matched by horizontal condensation in the caller. ``islands`` wraps
    the font so ⟦Mn⟧ pixel-island tokens measure (and later draw) at their source width.
    ``hyphenator`` (justified only) breaks a long compound at a line boundary — see
    ``_greedy_wrap``."""
    font = load_font(max(6, min(int(size), 160)), text, family=family, weight=weight, italic=italic)
    if islands:
        font = IslandFont(font, islands)
    return font, _wrap_to_planes(font, text, plane_widths, justified=justified, hyphenator=hyphenator)

def _raw_condense(font: Any, lines: list[str], planes: list[dict[str, Any]]) -> float:
    """Unclamped horizontal scale needed to bring every line within its plane width + slack.

    Per line: ``_allowed_width(plane) / natural rendered width``; the group takes the
    tightest (smallest) line factor. ``>= 1.0`` means the lines already fit within the slack;
    below ``_CONDENSE_FLOOR`` means even maximum condensation leaves a line more than the slack
    too wide (the caller then reduces the pt size)."""
    factors: list[float] = []
    for index, line in enumerate(lines):
        if index >= len(planes) or not line:
            continue
        natural = font.getlength(line)
        if natural > 0:
            factors.append(_allowed_width(planes[index], float(planes[index]["width"])) / natural)
    return min(factors) if factors else 1.0

def _condense_scale(font: Any, lines: list[str], planes: list[dict[str, Any]]) -> float:
    """Horizontal scale that squeezes the group's lines into their original widths (plus the
    width slack), clamped to [``_CONDENSE_FLOOR``, 1.0] — never stretch a short line, never
    squeeze past the floor (the pt size is reduced upstream instead)."""
    return max(_CONDENSE_FLOOR, min(1.0, _raw_condense(font, lines, planes)))

def _wrap_to_planes(
    font: Any, text: str, plane_widths: list[float], *, justified: bool = False, hyphenator: Any = None
) -> list[str]:
    """Wrap so each rendered line fits the width of the PLANE it lands on, in order, and the words
    are BALANCED across those lines — not greedily dumped.

    Two failures this avoids:
    - Wrapping every line to the widest plane overflows a narrow top plane (a short heading line
      above a wide one), and the caller's width-fit then shrinks the whole block toward one tiny
      line. So each line is bounded by its OWN plane width.
    - A plain greedy fill breaks an early line at its natural plane width, leaving it half-empty
      while the remainder (often a long token like a URL) piles onto the last line. So instead the
      block's natural line COUNT is taken first (greedy at the plane widths, capped at the plane
      count), then the words are spread over exactly that many lines by the smallest uniform scale
      on the plane widths that still fits — a minimax fill that keeps every line about equally full.

    A compact translation that needs fewer lines than there are planes uses fewer (the rest stay
    erase-only). On equal-width planes a balanced fill matches the original column layout."""
    content = str(text or "").strip()
    if len(plane_widths) <= 1 or not content:
        return [content]
    # Atomic wrap units, not ``.split()``: Han/Kana/CJK-symbol scripts have no spaces, so a whole
    # CJK line is one "word" and never wraps — it stays on one line and the caller condenses it to a
    # sliver. ``break_pieces`` breaks CJK per character (with kinsoku) and keeps Latin/Hangul/digits
    # as whitespace words, so each piece carries the ``glue`` to re-insert when it is not a line start.
    pieces = break_pieces(content)
    if justified:
        # Justified setting packs each line as FULL as its plane allows and leaves the
        # remainder to the last line — which justify renders ragged anyway (how a set
        # paragraph ends). The balanced fill below spreads the leftover over EVERY line
        # instead; under a narrow serif face that pushed many lines past the per-gap
        # stretch cap and flipped whole justified blocks to ragged (measured: 7 of 23
        # lines infeasible, tail to 3.5x the space width).
        return _greedy_wrap(font, pieces, plane_widths, hyphenator=hyphenator)
    line_count = len(_greedy_wrap(font, pieces, plane_widths))  # fewest lines at natural plane width
    caps = plane_widths[:line_count]
    # Smallest scale on the plane widths that still packs the pieces into ``line_count`` lines: this
    # is the most-relaxed (least condensed) balanced fill. Binary search — monotone in the scale.
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _fits_in_lines(font, pieces, caps, mid):
            hi = mid
        else:
            lo = mid
    return _greedy_wrap(font, pieces, [cap * hi for cap in caps])

# The shortest word worth breaking. pyphen keeps 2 letters either side of a break, so a 5-letter
# word can only break 2|3; below that a hyphen buys almost no width and reads as noise.
_HYPHEN_MIN_WORD = 6


def _hyphen_head(font: Any, word: str, room: float, hyphenator: Any) -> tuple[str, str] | None:
    """``(head, tail)`` for the LARGEST hyphenated head of ``word`` whose ink (with the hyphen)
    fits ``room`` — or ``None`` when the word does not break (too short, no letter core) or no
    break fits. Largest-that-fits so the head fills as much of the line's remaining room as it
    can without overrunning; the tail flows on as the next line's start.

    Breaks the word's ALPHABETIC CORE, ignoring surrounding punctuation: a caption's
    "literatuuronderzoekstabellen:" (trailing colon) or a citation's "(Kwiatkowski" (leading
    paren) must still break — the punctuation just travels with the head or the tail. The core
    must be all letters, so a token with an internal hyphen ("vision-artikelen"), digits
    ("2.228") or a lone marker ("(2)") is left whole."""
    start, end = 0, len(word)
    while start < end and not word[start].isalpha():
        start += 1
    while end > start and not word[end - 1].isalpha():
        end -= 1
    core = word[start:end]
    if len(core) < _HYPHEN_MIN_WORD or not core.isalpha():
        return None
    chosen = None
    for position in hyphenator.positions(core):
        cut = start + position
        if font.getlength(f"{word[:cut]}-") <= room:
            chosen = cut
    if chosen is None:
        return None
    return word[:chosen] + "-", word[chosen:]


def _greedy_wrap(
    font: Any, pieces: list[tuple[str, str]], line_caps: list[float], hyphenator: Any = None
) -> list[str]:
    """Greedy fill: line ``i`` takes pieces while they fit ``line_caps[i]``; the LAST cap carries
    every remaining piece (so the result never exceeds ``len(line_caps)`` lines). A piece that alone
    exceeds its cap still starts the line (never an empty line) — the caller condenses/shrinks it.
    Each piece carries the ``glue`` (a space for Latin words, empty for CJK chars) re-inserted only
    when it is not at a line start.

    Given a ``hyphenator`` (justified blocks only), a word that will not fit the current line is
    BROKEN there instead of wrapping whole: its head (up to the last break that fits) stays on the
    line and its tail becomes the next line's first piece, which then keeps filling. Doing this
    inside the single forward fill is what makes it re-flow — the words after the break move up to
    fill the tail's line, so a hard boundary (a long Dutch/German compound) no longer leaves a gap
    the justify cannot close. Only breaks a plain word (``_hyphen_head``); punctuation and the last
    line are left whole."""
    lines: list[str] = []
    current = ""
    index = 0
    last = len(line_caps) - 1
    queue = list(pieces)
    position = 0
    while position < len(queue):
        piece, glue = queue[position]
        sep = glue if current else ""
        if index >= last:  # the last line carries the remainder, ragged by design — never break
            current = current + sep + piece
            position += 1
            continue
        if current and font.getlength(current + sep + piece) > line_caps[index]:
            split = (
                _hyphen_head(font, piece, line_caps[index] - font.getlength(current + sep), hyphenator)
                if hyphenator is not None
                else None
            )
            if split is not None:
                head, tail = split
                lines.append(current + sep + head)
                current = ""
                index += 1
                queue[position] = (tail, glue)  # re-processed as the next line's start
                continue
            lines.append(current)
            current = piece
            index += 1
        else:
            current = current + sep + piece
        position += 1
    lines.append(current)
    return lines

def _fits_in_lines(font: Any, pieces: list[tuple[str, str]], caps: list[float], scale: float) -> bool:
    """Whether ``pieces`` pack into ``len(caps)`` lines with each line within ``caps[i] * scale``
    (a piece wider than its cap alone still starts a line). Used to find the smallest balancing
    scale by binary search."""
    index = 0
    current = ""
    for piece, glue in pieces:
        sep = glue if current else ""
        trial = current + sep + piece
        if current and font.getlength(trial) > caps[index] * scale:
            index += 1
            if index >= len(caps):
                return False
            current = piece
        else:
            current = trial
    return True
