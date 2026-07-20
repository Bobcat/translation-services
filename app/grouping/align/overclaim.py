"""A hint line that swallowed the text of the NEXT one.

The grouping VLM reads a page as running prose, so where a sentence breaks across a column
(or a page) it sometimes writes that sentence out in full on the line where it STARTS, and
then writes the continuation again as its own line. Both lines match cells — the first in the
column where the sentence begins, the second in the column where it continues — so both are
translated and both are rendered: the tail prints twice, once crammed into the single source
line at the foot of the first column and once, correctly, at the head of the next.

The cells are the arbiter. A line may only carry text its OWN cells print; a suffix that is
exactly another hint line's text, and that this line's cells do not print, belongs to that
other line. Trimming it here (before translation) also spares the translator the duplicate.

Deliberately narrow: an equal-length repeat is NOT an over-claim — a page that prints the same
warning twice is two honest lines, and both must translate (repeat-sign behaviour).
"""
from __future__ import annotations

from app.grouping.tokens import _tokens
from app.grouping.units import TranslationUnit


# A shorter tail than this is not evidence: "To address these challenges:" style fragments and
# repeated headers recur across a page for honest reasons, and cutting on them would silently
# drop real text. The measured case runs to 20 words.
_MIN_TAIL_WORDS = 6


def _normalised(text: str) -> str:
    return " ".join(_tokens(str(text or "")))


def trim_overclaimed_hint_lines(
    units: list[TranslationUnit], hint_units: list[str]
) -> list[str]:
    """``hint_units`` with each line's over-claimed suffix removed (the list keeps its length
    and order — every parallel hint list stays indexable by the same position)."""
    texts = list(hint_units)
    own_text: dict[int, str] = {}
    for unit in units:
        if unit.hint_index is None:
            continue
        printed = " ".join(member.text for member in unit.members if member.text)
        own_text[unit.hint_index] = f"{own_text.get(unit.hint_index, '')} {printed}".strip()

    for index, text in enumerate(texts):
        words = str(text or "").split()
        if len(words) <= _MIN_TAIL_WORDS:
            continue
        own = _normalised(own_text.get(index, ""))
        for other, other_text in enumerate(texts):
            if other == index or other not in own_text:
                continue
            other_words = str(other_text or "").split()
            if not (_MIN_TAIL_WORDS <= len(other_words) < len(words)):
                continue
            tail = _normalised(" ".join(words[-len(other_words):]))
            if not tail or tail != _normalised(other_text):
                continue
            if tail in own:
                continue  # this line's own cells DO print it: an honest second print
            texts[index] = " ".join(words[: -len(other_words)]).strip()
            break
    return texts
