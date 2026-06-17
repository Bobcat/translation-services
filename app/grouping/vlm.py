"""Stage #5 grouping hint: ask a VLM for a structural analysis of the text.

The VLM reads the image and returns, in reading order, an image **classification**
plus one labeled line per printed line of the document — the hierarchy label
(Title/Header/Body/Footer) comes from visual cues (size, weight, position) that OCR
line heights cannot give reliably. Blank lines separate semantic blocks; table rows
carry ``|`` between their fields. It does NOT do cell-id bookkeeping.

The returned lines are only a *hint*: ``app.grouping.align`` maps them back onto the
authoritative OCR cells (text + bbox) and builds the units. A weak or incomplete hint
therefore lowers quality but does not fail the job. The classification is near-free
extra context (e.g. feeds the translation prompt: "restaurant menu" stops gemma
rendering "hoofdgerechten" as "fundamental rights"); the per-line level + block id
feed the renderer's size coordination. We ask for it in "just a few words" on purpose:
the category is injected into the translator prompt, and the translator preserves words
from the category that also appear in the text — so a verbose, product-naming label
("Nike advertisement for Sweet Classic High **shoe**s") leaks that word and leaves it
untranslated ("DE SHOE" instead of "DE SCHOEN"). Keeping it generic avoids that.
"""
from __future__ import annotations

import base64
import copy
import re
from dataclasses import dataclass
from dataclasses import field
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from app.core.config import AppSettings


_RESPONSES_PATH = "/v1/responses"
_MAX_OUTPUT_TOKENS = 4096

# Sent as the user turn; the system role stays blank (llm-pool uses " "). Kept generic
# on purpose — no wording tied to a specific image type. Design:
#   - ELEMENT level: one labeled line per semantic element (a whole dish with its
#     wrapped lines merged, a receipt row, a paragraph). The label boundary IS the item
#     boundary — each labeled line becomes its own block, and the renderer reflows the
#     translation over the element's physical lines (recovered by geometry).
#   - the label carries the visual typography OCR cannot give, as a strict pipe-delimited
#     field list: "[<importance t/h/b/m> | <font-family> | <font-size>pt | <font-weight> |
#     <alignment l/c/r>]: text". parse_grouping_output strips it into a per-line level,
#     font_family, font_weight and alignment. Size is requested but currently unused (OCR
#     true-height drives the rendered size). font_family/weight are PER ELEMENT — a title and
#     a body line can differ (e.g. an ad with a sans headline and a serif paragraph) — so the
#     renderer picks a face per unit instead of one hard-coded font.
#   - alignment is a REQUIRED l/c/r field (the strict format emits it every line, which keeps
#     the ':' separator and the per-line label stable — an earlier exception-marked "| centered"
#     drifted, omitting the ':' and re-emitting labels mid-row that leaked into the text). Only
#     "c" is acted on; l/r both anchor at the line's own edge. A wrong hint is only cosmetic:
#     the renderer moves the in-plane anchor.
#   - "table row -> '|' between its fields" marks tabular layout; at element level it
#     also splits a menu dish from its price column. (This is the field '|' in the text,
#     distinct from the '|' separating the label's fields inside the brackets.)
# Price/number cells are flagged non-translatable in align (_is_nontranslatable); the
# '|' itself is not yet parsed into field structure.
_SYSTEM_PROMPT = " "

_USER_INSTRUCTION = (
    "# TASK\n"
    "Perform a structural analysis of the text in this image. Reconstruct the document's "
    "hierarchy by labeling each element in its natural reading order.**\n\n"
    "# INSTRUCTIONS\n"
    "1. **Analyze Visual Cues:** Use font size, font weight (boldness), spatial "
    "positioning, and grouping to determine the importance of each text element. Put a t "
    "for title, h for header, b for body or a m for metadata/footers.\n\n"
    "2. **Reading order:** Process the text from top to bottom, following the natural "
    "reading order. For every piece of text, immediately precede it with its "
    "classification.\n\n"
    "2.1 **Icons:** Ignore graphical icons and pictograms (a calendar, home, gear, "
    "magnifier, info '(i)', a brand logo, etc.). They are not text and not part of any "
    "element — do not output them or a name/placeholder for them. (List bullets are handled "
    "in 4.1.)\n\n"
    "3. **Table rows :** If an element is a **table row**, put '|' between its fields.\n\n"
    "4. **Field values:** If an element has the format <Field-label> <Field-value>, put "
    "'|' between the label and the value.\n\n"
    "4.1 **Bullet-list items:** If an element has the format <bullet> <item>, output as "
    "|@bullet|<item>.\n\n"
    "5. For **font-family**, provide your best best guess for a **specific font name** do "
    "NOT just mention serif or sans-serif.\n\n"
    "6. **Alignment:** Determine the text elements alignment on the document. Put an l for "
    "left, c for centered or an r for right inside the label. Table rows and header rows "
    "are never centered.\n\n"
    "# OUTPUT FORMAT (EXACT)\n"
    "**Image classification: <classification in a few words>**\n"
    "**<Importance t, h, b or m>|<font-family>|<font-size>pt| font-weight (100-900)>|"
    "<alignment l, c or r>**: <text element>"
)

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# Image formats the VLM backend can decode. llama-server (llama.cpp's mtmd/clip) loads images
# through stb_image, which handles JPEG/PNG but NOT WebP: a webp data-URI is silently dropped
# and the model answers as if no image was sent (it refuses with "please provide the image").
# So we transcode any non-JPEG/PNG input to PNG before sending — this keeps the grouping hint
# backend-agnostic (a vLLM/Pillow backend would accept webp, llama-server does not). Done here
# rather than in the pool only because translation-services is the single client that needs it.
_VLM_SAFE_MIME = {"image/jpeg", "image/png"}


class GroupingHintError(RuntimeError):
    """Raised when the grouping VLM call itself fails (transport / empty)."""


@dataclass(frozen=True)
class GroupingHint:
    category: str
    units: list[str]
    raw: str = ""
    # Parallel to ``units``: the visual hierarchy level of each line
    # ("title" | "header" | "body" | "footer", None when unlabeled), the index of the
    # element block it belongs to, its horizontal alignment ("center", None = left), and
    # the VLM's per-element typography: font_family (a named font, None when unlabeled)
    # and font_weight (100-900, None when unlabeled). The label's font-size is parsed off
    # but intentionally not kept — OCR true-height drives the rendered size.
    levels: list[str | None] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    alignments: list[str | None] = field(default_factory=list)
    font_families: list[str | None] = field(default_factory=list)
    font_weights: list[int | None] = field(default_factory=list)
    # Parallel to ``units``: True when the line was a bullet-list item (the VLM prefixed it
    # with the "@blt|" sentinel, stripped here). The original bullet glyph is left in the image;
    # the renderer only insets the text past it so the translation does not overwrite it.
    bullets: list[bool] = field(default_factory=list)


def request_grouping_hint(
    *,
    settings: AppSettings,
    input_path: Path,
    model: str,
    call_log: list[dict[str, Any]] | None = None,
) -> GroupingHint:
    if not str(model or "").strip():
        raise GroupingHintError(
            "grouping_model is required (set llm_pool.grouping_model or pass "
            "grouping_model in the request)"
        )
    content = [
        {"type": "text", "text": _USER_INSTRUCTION},
        {"type": "image_url", "image_url": {"url": _data_uri(input_path)}},
    ]
    payload = {
        "model": model,
        "input": content,
        "instructions": _SYSTEM_PROMPT or " ",
        "allow_remote": False,
        "stream": False,
        # Greedy + fully explicit (don't lean on per-model pool defaults): grouping
        # must be deterministic (same image -> same units).
        "decoding": {
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
    }
    output_text = _call_llm_pool(
        base_url=settings.llm_pool.base_url,
        payload=payload,
        timeout=settings.llm_pool.request_timeout_s,
        call_log=call_log,
    )
    return parse_grouping_output(output_text)


# The label's wrapper is whatever the model emits — it wavers between "[...]", "**...**" and no
# wrapper at all from run to run. Match all three so a dropped wrapper never leaks the label into
# the text: an optionally-bold "[label]", a "**label**", or a bare "t|...:" (the importance code +
# fields, used to recognise an unwrapped label without catching ordinary "word: ..." text).
_LABELED_LINE = re.compile(
    r"^\s*(?:"
    r"\**\s*\[(?P<br>[^\[\]]+)\]\**"
    r"|\*\*(?P<st>(?:(?!\*\*).)+?)\*\*"
    r"|\*(?P<si>[^*\n]+?)\*"
    r"|(?P<bare>[thbm]\s*\|[^:\[\]\n]*)"
    r")\s*:?\s*\**\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

# Alignment values. The prompt asks for a single-letter l/c/r field, but tolerate the spelled
# words and both British/American "centre"/"center"/"centred"/"centered" spellings. Only
# "center" is acted on downstream; left/right both anchor at the line's own edge. The words and
# single letters are also skipped by _parse_label_fonts so they aren't taken for a font family.
_ALIGN_CENTER = re.compile(r"\bcent(?:ered|er|red|re)\b")
_ALIGN_WORD = re.compile(r"cent(?:ered|er|red|re)|left|right|[lcr]")

# A bullet-list item: the prompt has the VLM emit the text element as "|@bullet|<item>". We strip
# the sentinel (tolerating leading/trailing pipes and the older "@blt" marker) and flag the unit;
# the original bullet glyph stays in the image (the renderer insets the text past it).
_BULLET_SENTINEL = re.compile(r"^\s*\|?\s*@(?:bullet|blt)\b\s*\|?\s*", re.IGNORECASE)

# The label's first '|'-field is a single-letter importance code (the v3 prompt: t/h/b/m).
_LEVEL_BY_CODE = {"t": "title", "h": "header", "b": "body", "m": "footer"}

# Fallback substring -> level for a spelled-out role or the legacy "[Level n / Role]" wording.
_LEVEL_BY_LABEL = (
    ("level 1", "title"),
    ("level 2", "header"),
    ("level 3", "body"),
    ("title", "title"),
    ("header", "header"),
    ("body", "body"),
    ("footer", "footer"),
    ("metadata", "footer"),
)


def parse_grouping_output(output_text: str) -> GroupingHint:
    """Split the VLM output into the classification and the labeled hint lines.

    The leading ``[Image Classification: ...]`` line becomes the category; every other
    non-empty line is one unit. A ``[Level n / ...]`` / ``[Metadata/Footer]`` prefix is
    stripped into the line's level — the model wavers between ``[Label] text``,
    ``[Label: text]`` and a standalone ``[Label]`` line above the text; all three parse.
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
    bullets: list[bool] = []
    block = 0
    block_open = False
    block_level: str | None = None
    block_alignment: str | None = None
    block_family: str | None = None
    block_weight: int | None = None

    def close_block() -> None:
        nonlocal block, block_open, block_level, block_alignment, block_family, block_weight
        if block_open:
            block += 1
            block_open = False
        block_level = None
        block_alignment = None
        block_family = None
        block_weight = None

    for raw in str(output_text or "").splitlines():
        line = raw.strip()
        if not line or _is_separator(line):
            close_block()
            continue
        level: str | None = None
        match = _LABELED_LINE.match(line)
        if match:
            label = (match.group("br") or match.group("st") or match.group("si") or match.group("bare") or "").strip()
            text = match.group("rest").strip()
            if not text and ":" in label:  # "[Metadata/Footer: 23:53]" variant
                label, text = (part.strip() for part in label.split(":", 1))
            if label.lower().startswith("image classification"):
                if not category:
                    category = text
                continue
            level = _level_of(label)
            if level is not None:
                close_block()  # a label starts a new element
                line = text
                block_level = level
                block_alignment = _alignment_of(label)
                block_family, block_weight = _parse_label_fonts(label)
                if not line:  # standalone "[Level 3 / Body]" -> labels the lines below
                    continue
        if level is None:
            level = block_level  # continuation line inherits the block's level
        if not category and not units and line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()
            continue
        cleaned = line.lstrip("-*#").strip().strip("*").strip()
        bullet = bool(_BULLET_SENTINEL.match(cleaned))
        if bullet:  # strip the "@blt|" sentinel; the glyph itself stays in the image
            cleaned = _BULLET_SENTINEL.sub("", cleaned).strip()
        if not cleaned or _is_separator(cleaned):
            continue
        units.append(cleaned)
        levels.append(level)
        block_ids.append(block)
        alignments.append(block_alignment)
        font_families.append(block_family)
        font_weights.append(block_weight)
        bullets.append(bullet)
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
        bullets=bullets,
    )


def _level_of(label: str) -> str | None:
    # v3: the first '|'-field is the importance code (t/h/b/m); take it (exact, then by first
    # letter so "title"/"header"/"body"/"metadata" also map). Else fall back to substring words.
    first = label.split("|", 1)[0].strip().lower()
    if first in _LEVEL_BY_CODE:
        return _LEVEL_BY_CODE[first]
    if first[:1] in _LEVEL_BY_CODE:
        return _LEVEL_BY_CODE[first[:1]]
    lowered = label.lower()
    for key, level in _LEVEL_BY_LABEL:
        if key in lowered:
            return level
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


def _parse_label_fonts(label: str) -> tuple[str | None, int | None]:
    """Pull the font-family and font-weight out of a typography label.

    The v3 label reads "<importance> | <font-family> | <font-size>pt | <font-weight> |
    <alignment>"; only some fields may be present and their order can wobble. Weight is the
    100-900 integer; family is the first remaining field that is not a size ("18pt"), a weight,
    or an alignment token (l/c/r or centered/left/right). Size is deliberately discarded.
    Returns (None, None) when the label carries no font fields."""
    parts = [part.strip() for part in label.split("|")]
    family: str | None = None
    weight: int | None = None
    for part in parts[1:]:  # parts[0] is the importance code (t/h/b/m)
        lowered = part.lower()
        if not part or _ALIGN_WORD.fullmatch(lowered):
            continue
        if re.fullmatch(r"\d{1,3}\s*pt", lowered):  # font-size: parsed off, not kept
            continue
        match = re.fullmatch(r"[1-9]00", part)
        if match:
            weight = int(part)
            continue
        if family is None:
            family = part
    return family, weight


def _is_separator(line: str) -> bool:
    compact = line.replace(" ", "")
    return len(compact) >= 3 and set(compact) <= {"#", "-", "=", "*", ":"}


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """A copy of the request payload with the base64 image data-URI replaced by a short note
    (the image is saved separately as a PNG), so the saved payload stays readable."""
    redacted = copy.deepcopy(payload)
    for item in redacted.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "image_url":
            url = str((item.get("image_url") or {}).get("url") or "")
            if url.startswith("data:"):
                item["image_url"]["url"] = f"<image data-uri, {len(url)} chars - redacted>"
    return redacted


def _call_llm_pool(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
    call_log: list[dict[str, Any]] | None = None,
    role: str = "grouping_vlm",
) -> str:
    url = f"{base_url}{_RESPONSES_PATH}"
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        raise GroupingHintError(
            f"llm-pool /v1/responses HTTP {exc.response.status_code}: {body or exc}"
        ) from exc
    except httpx.HTTPError as exc:
        raise GroupingHintError(f"llm-pool /v1/responses unavailable: {exc}") from exc

    if not isinstance(data, dict):
        raise GroupingHintError("llm-pool /v1/responses returned a non-object response")
    if call_log is not None:
        call_log.append({"role": role, "payload": _redact_payload(payload), "response": data})
    output_text = str(data.get("output_text") or "").strip()
    if not output_text:
        raise GroupingHintError("llm-pool /v1/responses returned empty output_text")
    return output_text


def _data_uri(input_path: Path) -> str:
    suffix = input_path.suffix.lower()
    mime = _MIME_BY_SUFFIX.get(suffix)
    if mime is None:
        raise GroupingHintError(f"unsupported image type for grouping: {suffix or 'unknown'}")
    raw = input_path.read_bytes()
    if mime not in _VLM_SAFE_MIME:  # e.g. webp -> PNG, so stb_image can decode it
        raw, mime = _to_png(raw), "image/png"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _to_png(raw: bytes) -> bytes:
    """Decode an image the VLM backend can't read (webp) and re-encode it as PNG."""
    try:
        with Image.open(BytesIO(raw)) as img:
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format="PNG")
    except Exception as exc:
        raise GroupingHintError(f"failed to transcode image to PNG for grouping: {exc}") from exc
    return buffer.getvalue()
