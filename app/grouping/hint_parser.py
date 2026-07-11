"""Parse the grouping VLM's text response into a structured ``GroupingHint``.

The grouping VLM is asked for one strict line per element —
``<t|h|b|m>|<font>|<size>pt|<weight>|<l|c|r>: text``. Even with greedy decode and an unchanged
prompt + image (see ``request_grouping_hint``) the response drifts run to run — serving-level
non-determinism, not sampling — so this layer parses defensively: across otherwise identical runs
the label may be wrapped in ``**...**`` / ``*...*`` or bare, the importance code may be
dropped (``|Roboto|16pt|400|l: …``), alignment may be a letter or a spelled word, the ``:`` may be
missing, or a label may sit on its own line above the text. Every regex here absorbs one such
variation so the label never leaks into the translated text.

``app.grouping.vlm`` owns the request (prompt + image + call) and delegates here for the response;
the translation stage has its own output-tolerance in ``app.translation.translate``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class GroupingHint:
    category: str
    units: list[str]
    raw: str = ""
    # Parallel to ``units``: the visual hierarchy level of each line
    # ("title" | "header" | "body" | "footer", None when unlabeled), the index of the
    # element block it belongs to, its horizontal alignment ("center", None = left), and
    # the VLM's per-element typography: font_family (a named font, None when unlabeled),
    # font_weight (100-900, None when unlabeled) and font_size (the label's "<n>pt", None when
    # unlabeled). The size is an unreliable ABSOLUTE (its pt->pixel scale drifts per image) but a
    # reliable EQUALITY signal — the model gives sibling elements one pt — used only by the
    # size-cohort render mode; the default sizing comes from OCR true-height.
    levels: list[str | None] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    alignments: list[str | None] = field(default_factory=list)
    font_families: list[str | None] = field(default_factory=list)
    font_weights: list[int | None] = field(default_factory=list)
    font_sizes: list[int | None] = field(default_factory=list)
    # Parallel to ``units``: True when the line was a bullet-list item (the VLM prefixed it with the
    # "@blt|@<bullet>|" sentinel, stripped here), and the glyph/marker the VLM saw (substituted into
    # "@<bullet>": "•", "-", "1.", "(a)", ...). The renderer redraws the marker before the text (or,
    # in legacy mode, insets past the original glyph left in the image).
    bullets: list[bool] = field(default_factory=list)
    bullet_markers: list[str | None] = field(default_factory=list)


# The prompt asks for a wrapped label, but serving non-determinism (and other model families) makes
# the model vary the wrapper and drop the importance code from run to run. Match four shapes so a
# dropped wrapper/code never leaks the label into the text: a "**label**", a "*label:*" (single
# wrapper char, colon required inside — so a stray "*emphasis*" in the text is not mistaken for a
# label; the opening char may be "*" OR "|", since some models — e.g. Qwen — open with "|"), a bare
# "t|...:" importance code + fields (the code may be a single letter t/h/b/m OR spelled out —
# "title|"/"header|"/"body|"/"metadata|" — since the model sometimes writes the word the prompt
# names instead of the letter), or a "fmt" typography label whose importance code was dropped ("|Roboto|16pt|400|
# l:") — anchored on a "|<digits>pt|" field so an ordinary "word | value" row is never one. The bare
# and fmt runs tolerate a dangling opening "*": on a bullet line g4 opens the wrapper but writes the
# "|@blt" element where the closing "*" (and often the ":") belong — "*b|Arial|11pt|400|l|@blt|…" or
# "*b|…l:|@blt|…" — so si never closes and, without eating that leading "*", the whole label+sentinel
# leaked into the text (16/20 runs on a bullet flyer). The bare/fmt runs still stop at ":" and "@",
# so the "|@blt" item stays in the text for _BULLET_SENTINEL to strip. A
# a stray "'" from a quoted template is tolerated both before the label and right after the colon
# ("'t|..c:' text"); a model that wraps the WHOLE "'label:text'" line in one quote pair is unwrapped
# earlier, in parse_grouping_output (both quote habits occur run to run). The bare/fmt field runs
# stop at "@" so a colon-less bullet line ("...|l|@bullet|item") keeps its item in the text
# (handled by _BULLET_SENTINEL) instead of swallowing it into the label.
_LABELED_LINE = re.compile(
    r"^\s*'?\s*(?:"
    r"\*\*(?P<st>(?:(?!\*\*).)+?)\*\*"
    r"|[*|](?P<si>[^*\n]+?):\s*\*"
    r"|\*?\s*(?P<bare>(?:title|header|body|metadata|footer|[thbm])\s*\|[^:\[\]\n@]*)"
    r"|\*?\s*(?P<fmt>\|\s*[^:|\[\]\n]*\|\s*\d{1,3}\s*pt\s*\|[^:\[\]\n@]*)"
    r")\s*'?\s*:?\s*'?\s*\**\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

# Alignment values. The prompt asks for a single-letter l/c/r field, but tolerate the spelled
# words and both British/American "centre"/"center"/"centred"/"centered" spellings. Only
# "center" is acted on downstream; left/right both anchor at the line's own edge. The words and
# single letters are also skipped by _parse_label_fonts so they aren't taken for a font family.
_ALIGN_CENTER = re.compile(r"\bcent(?:ered|er|red|re)\b")
_ALIGN_WORD = re.compile(r"cent(?:ered|er|red|re)|left|right|[lcr]")

# Leading markdown decoration ("- item", "* item", "# heading") — only stripped when whitespace
# (or line end) follows, so a numeric sign ("-2,00") survives.
_MARKDOWN_LEAD = re.compile(r"^[-*#]+(?=\s|$)")

# A bullet-list item: the prompt has the VLM emit the element as "|@blt|@<bullet>|<item>" — a stable
# "@blt" sentinel, then the actual glyph it SAW substituted into "@<bullet>". Asking for substitution
# (rather than a fixed "@bullet") stops the model rewriting the sentinel itself, e.g. "@bullet"->"@-".
# We strip the sentinel AND the optional substituted glyph field and flag the unit; the glyph stays in
# the image (the renderer insets the text past it). Tolerant of the older "|@bullet|"/"|@<glyph>|"
# forms and of a missing glyph field; the "|" framing + line-start "@" keep it off ordinary text.
_BULLET_SENTINEL = re.compile(
    r"^\s*\|?\s*@(?P<sentinel>blt|bullet|[•·∙●○◦‣⁃*–—-])(?:\s*\|\s*(?P<marker>@?[^|]*)\|)?\s*\|?\s*",
    re.IGNORECASE,
)
# The sentinel without the marker field — used to re-match when the "marker" turned out to be an
# ordinary |-field of the row's content.
_BULLET_SENTINEL_BARE = re.compile(
    r"^\s*\|?\s*@(?:blt|bullet|[•·∙●○◦‣⁃*–—-])\s*\|?\s*",
    re.IGNORECASE,
)
# A substituted bullet glyph is a couple of characters at most ("•", "-", "1.", "(iv)"). Anything
# longer without an explicit "@" is the row's first content field ("@blt|Prijs | 12,50"), which
# must stay in the text.
_BULLET_MARKER_MAX_LEN = 4
def _bullet_of(text: str) -> tuple[bool, str | None, str]:
    """``(is_bullet, marker, remaining_text)`` for a possible bullet line. The marker is the glyph the
    VLM substituted into a SEPARATE ``@<bullet>`` field (``@blt|@•|item``), stripped from the text so
    the renderer redraws it. When there is no separate field (``@blt|1. item``) the marker is already
    in the content — we return ``None`` and leave it; we never invent a default bullet (the original
    may have none, e.g. a plain numbered list)."""
    match = _BULLET_SENTINEL.match(text)
    if not match:
        return False, None, text
    captured = (match.group("marker") or "").strip()
    explicit = captured.startswith("@")
    marker_text = captured.lstrip("@").strip()
    if marker_text and not explicit and len(marker_text) > _BULLET_MARKER_MAX_LEN:
        # Not a glyph but the first content field of a colon-less row — re-match without the
        # marker group so the field survives in the text.
        bare = _BULLET_SENTINEL_BARE.match(text)
        return True, None, text[bare.end():].strip() if bare else text
    sentinel = match.group("sentinel")
    if marker_text:
        marker = marker_text
    elif sentinel and sentinel.lower() not in ("blt", "bullet"):
        marker = sentinel  # old format: the glyph was the sentinel itself
    else:
        marker = None  # only the "@blt" word -> the marker (if any) stays in the text
    return True, marker, text[match.end():].strip()

# The label's first '|'-field is a single-letter importance code (the v3 prompt: t/h/b/m),
# or the spelled-out word some models write instead.
_LEVEL_BY_CODE = {"t": "title", "h": "header", "b": "body", "m": "footer"}
_LEVEL_BY_WORD = {"title": "title", "header": "header", "body": "body", "metadata": "footer", "footer": "footer"}


def parse_grouping_output(output_text: str) -> GroupingHint:
    """Split the VLM output into the classification and the labeled hint lines.

    The leading ``Image classification: ...`` line becomes the category; every other
    non-empty line is one unit. A label prefix (see ``_LABELED_LINE``: bold-wrapped,
    single-star, or a bare/fmt pipe label) is stripped into the line's level — the model
    wavers between ``label: text``, the text inside the wrapper, and a standalone label
    line above the text; all three parse.
    Every LABELED line starts a new block (the label boundary is the element boundary);
    unlabeled lines (a wrapped continuation) join the block and inherit its level.
    Blank lines and separator lines (``###`` / ``-----``) also close the block; leading
    bullet/markdown markers are dropped. A legacy ``CATEGORY:`` first line still parses
    as the category.
    """
    category = ""
    units: list[str] = []
    levels: list[str | None] = []
    block_ids: list[int] = []
    alignments: list[str | None] = []
    font_families: list[str | None] = []
    font_weights: list[int | None] = []
    font_sizes: list[int | None] = []
    bullets: list[bool] = []
    bullet_markers: list[str | None] = []
    block = 0
    block_open = False
    block_level: str | None = None
    block_alignment: str | None = None
    block_family: str | None = None
    block_weight: int | None = None
    block_size: int | None = None

    def close_block() -> None:
        nonlocal block, block_open, block_level, block_alignment, block_family, block_weight, block_size
        if block_open:
            block += 1
            block_open = False
        block_level = None
        block_alignment = None
        block_family = None
        block_weight = None
        block_size = None

    for raw in str(output_text or "").splitlines():
        line = raw.strip()
        if not line or _is_separator(line):
            close_block()
            continue
        # Some models (e.g. Qwen) take the OUTPUT FORMAT's quoted "'label:text'" example literally and
        # wrap the WHOLE line in one quote pair, every line. Unwrap a single matched outer pair so the
        # closing quote can't leak into the text (and the leading one out of the label). Only when both
        # ends match, so a real trailing apostrophe on an unquoted line is left alone.
        if len(line) >= 2 and line[0] == line[-1] == "'":
            line = line[1:-1].strip()
        if line.lower().lstrip("*' ").startswith("image classification"):
            if not category and ":" in line:
                category = line.split(":", 1)[1].strip().strip("*'").strip()
            continue
        level: str | None = None
        match = _LABELED_LINE.match(line)
        if match:
            label = (match.group("st") or match.group("si")
                     or match.group("bare") or match.group("fmt") or "").strip()
            text = match.group("rest").strip()
            if not text and ":" in label:  # a label that carries its text after a ":" inside it
                label, text = (part.strip() for part in label.split(":", 1))
            level = _level_of(label)
            # A bare/fmt pipe label or a single-star "*...:*" label is unmistakably a label even
            # when no level is read (some models drop the importance code) — strip it so it cannot
            # leak into the text; the level then falls back to None for that element.
            pipe_label = bool(match.group("bare") or match.group("fmt") or match.group("si"))
            # But a BARE match with no text can also be a content row whose first field happens to
            # be a level word/letter (a receipt tax row "B | 1,69", a table row "Title | Mr").
            # A real standalone label shows its label-ness: several |-fields, a "<n>pt" size, or
            # the trailing ':' the prompt format ends a label with. Without any of those, keep the
            # row as text rather than deleting it as a label.
            if not text and match.group("bare") and not _bare_label_stands_alone(label, line):
                level = None
                pipe_label = False
            if level is not None or pipe_label:
                close_block()  # a label starts a new element
                line = text
                block_level = level
                block_alignment = _alignment_of(label)
                block_family, block_weight, block_size = _parse_label_fonts(label)
                if not line:  # standalone "[Level 3 / Body]" -> labels the lines below
                    continue
        if level is None:
            level = block_level  # continuation line inherits the block's level
        if not category and not units and line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()
            continue
        # Strip a leading markdown bullet/heading marker only when whitespace follows it: a bare
        # "- Alpha" is decoration, but the "-" of "-2,00 korting" is the amount's sign and must
        # reach the translation. (The .strip("*") after still unwraps "*emphasis*"/"**bold**".)
        cleaned = _MARKDOWN_LEAD.sub("", line).strip().strip("*").strip()
        bullet, marker, cleaned = _bullet_of(cleaned)
        if not cleaned or _is_separator(cleaned):
            continue
        units.append(cleaned)
        levels.append(level)
        block_ids.append(block)
        alignments.append(block_alignment)
        font_families.append(block_family)
        font_weights.append(block_weight)
        font_sizes.append(block_size)
        bullets.append(bullet)
        bullet_markers.append(marker)
        block_open = True
    return GroupingHint(
        category=category,
        units=units,
        raw=str(output_text or ""),
        levels=levels,
        block_ids=block_ids,
        alignments=alignments,
        font_families=font_families,
        font_weights=font_weights,
        font_sizes=font_sizes,
        bullets=bullets,
        bullet_markers=bullet_markers,
    )


def _bare_label_stands_alone(label: str, line: str) -> bool:
    """Whether a bare label with no text after it is a genuine standalone label (it labels the
    lines below) rather than a content row starting with a level word/letter."""
    return (
        label.count("|") >= 2
        or re.search(r"\d{1,3}\s*pt\b", label, re.IGNORECASE) is not None
        or line.rstrip("'* ").endswith(":")
    )


def _level_of(label: str) -> str | None:
    # The first '|'-field is the importance code (t/h/b/m) or its spelled-out word; take those
    # exactly. The first-LETTER fallback (a truncated/mangled first field) applies only when the
    # label carries '|'-fields at all: a bare bolded text word ("**Menu**", "**Totaal**") is image
    # text, not a label, and must not become one just because m/t happen to be level codes.
    first = label.split("|", 1)[0].strip().lower()
    if first in _LEVEL_BY_CODE:
        return _LEVEL_BY_CODE[first]
    if first in _LEVEL_BY_WORD:
        return _LEVEL_BY_WORD[first]
    if "|" in label and first[:1] in _LEVEL_BY_CODE:
        return _LEVEL_BY_CODE[first[:1]]
    return None


def _alignment_of(label: str) -> str | None:
    """The alignment field (the v3 prompt's trailing l/c/r): 'c'/'centered' -> "center";
    'l'/'r' -> None (left and right both anchor at the line's own edge). Falls back to a
    legacy "| centered" appended anywhere in the label."""
    last = label.rsplit("|", 1)[-1].strip().lower() if "|" in label else ""
    if last in {"c", "center", "centre", "centered", "centred"}:
        return "center"
    if last in {"l", "left", "r", "right"}:
        return None
    return "center" if _ALIGN_CENTER.search(label.lower()) else None


def _parse_label_fonts(label: str) -> tuple[str | None, int | None, int | None]:
    """Pull the font-family, font-weight and font-size out of a typography label.

    The v3 label reads "<importance> | <font-family> | <font-size>pt | <font-weight> |
    <alignment>"; only some fields may be present and their order can wobble. Weight is the
    100-900 integer; family is the first remaining field that is not a size ("18pt"), a weight,
    or an alignment token (l/c/r or centered/left/right). The size (an unreliable ABSOLUTE, but a
    reliable EQUALITY signal — the model gives sibling elements one pt) is kept only for the
    optional size-cohort render mode; the default sizing still comes from OCR true-height.
    Returns (None, None, None) when the label carries no font fields."""
    parts = [part.strip() for part in label.split("|")]
    family: str | None = None
    weight: int | None = None
    size: int | None = None
    for part in parts[1:]:  # parts[0] is the importance code (t/h/b/m)
        lowered = part.lower()
        if not part or _ALIGN_WORD.fullmatch(lowered):
            continue
        pt = re.fullmatch(r"(\d{1,3})\s*pt", lowered)
        if pt:  # font-size: kept for the size-cohort mode, ignored by the default sizing
            size = int(pt.group(1))
            continue
        match = re.fullmatch(r"[1-9]00", part)
        if match:
            weight = int(part)
            continue
        if family is None:
            family = part
    return family, weight, size


def _is_separator(line: str) -> bool:
    compact = line.replace(" ", "")
    return len(compact) >= 3 and set(compact) <= {"#", "-", "=", "*", ":"}
