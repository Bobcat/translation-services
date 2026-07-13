"""Break and condense a group's translated text onto its planes."""
from __future__ import annotations

from typing import Any
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
# groups the planner measures the pixels right of each plane (same scan as the "extend" width
# fit) and stores the verified run as ``plane["slack_px"]`` — a colour step (a newsletter's
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
) -> tuple[Any, list[str]]:
    """Render at the source ``size`` (true line height) in the unit's VLM font ``family`` /
    ``weight``, wrapped so each line fits the width of the plane it lands on (``plane_widths``).
    The font is NOT reduced to fit width — the source size, and thus the header/body hierarchy,
    is preserved; width is matched by horizontal condensation in the caller."""
    font = load_font(max(6, min(int(size), 160)), text, family=family, weight=weight)
    return font, _wrap_to_planes(font, text, plane_widths)

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

def _wrap_to_planes(font: Any, text: str, plane_widths: list[float]) -> list[str]:
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

def _greedy_wrap(font: Any, pieces: list[tuple[str, str]], line_caps: list[float]) -> list[str]:
    """Greedy fill: line ``i`` takes pieces while they fit ``line_caps[i]``; the LAST cap carries
    every remaining piece (so the result never exceeds ``len(line_caps)`` lines). A piece that alone
    exceeds its cap still starts the line (never an empty line) — the caller condenses/shrinks it.
    Each piece carries the ``glue`` (a space for Latin words, empty for CJK chars) re-inserted only
    when it is not at a line start."""
    lines: list[str] = []
    current = ""
    index = 0
    last = len(line_caps) - 1
    for piece, glue in pieces:
        sep = glue if current else ""
        if index >= last:
            current = current + sep + piece
            continue
        trial = current + sep + piece
        if current and font.getlength(trial) > line_caps[index]:
            lines.append(current)
            current = piece
            index += 1
        else:
            current = trial
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
