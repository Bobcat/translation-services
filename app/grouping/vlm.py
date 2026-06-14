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
from pathlib import Path
from typing import Any

import httpx

from app.core.config import AppSettings


_RESPONSES_PATH = "/v1/responses"
_MAX_OUTPUT_TOKENS = 4096

# Sent as the user turn; the system role stays blank (llm-pool uses " "). Kept generic
# on purpose — no wording tied to a specific image type. Design:
#   - ELEMENT level: one labeled line per semantic element (a whole dish with its
#     wrapped lines merged, a receipt row, a paragraph). The label boundary IS the item
#     boundary — each labeled line becomes its own block, and the renderer reflows the
#     translation over the element's physical lines (recovered by geometry).
#   - the label carries the visual typography OCR cannot give:
#     "[Level n / Role | <font-family> | <font-size>pt | <font-weight> | centered]".
#     parse_grouping_output strips it into a per-line level, font_family and font_weight.
#     Size is requested but currently unused (OCR true-height drives the rendered size).
#     font_family/weight are PER ELEMENT — a title and a body line can differ (e.g. an ad
#     with a sans headline and a serif paragraph) — so the renderer picks a face per unit
#     instead of one hard-coded font.
#   - alignment is EXCEPTION-marked ("| centered" inside the label, left = default):
#     near-free in tokens, ambiguity falls back to left. The shared-left-margin clause is
#     load-bearing; the tilt clause was dropped (verified unnecessary and it cost tokens).
#     A wrong hint is only cosmetic: the renderer moves the anchor within the same plane.
#   - "table row -> '|' between its fields" marks tabular layout; at element level it
#     also splits a menu dish from its price column. (This is the field '|' in the text,
#     distinct from the '|' separating the label's font fields inside the brackets.)
# Price/number cells are flagged non-translatable in align (_is_nontranslatable); the
# '|' itself is not yet parsed into field structure.
_SYSTEM_PROMPT = " "

_USER_INSTRUCTION = (
    "**Perform a structural analysis of the text in this image. Reconstruct the "
    "document's hierarchy by labeling each element in its natural reading order.**\n\n"
    "**Instructions:**\n\n"
    "1. **Analyze Visual Cues:** Use font size, font weight (boldness), spatial "
    "positioning, and grouping to determine the importance of each text element.\n\n"
    "2. **Linear Labeling (Do Not Group):** Do **not** group all elements of the same "
    "level together. Instead, process the text from top to bottom, following the "
    "natural reading order. For every piece of text, immediately precede it with its "
    "classification:\n"
    "    * **[Level 1 / Title | <font-family> | <font-size>pt | <font-weight (100-900)>]:** "
    "The most prominent text (largest/boldest).\n"
    "    * **[Level 2 / Header | <font-family> | <font-size>pt | <font-weight (100-900)>]:** "
    "Sub-headings that introduce new sections.\n"
    "    * **[Level 3 / Body | <font-family> | <font-size>pt | <font-weight (100-900)>]:** "
    "Descriptive or supporting text.\n"
    "    * **[Metadata/Footer]:** Small text at the edges, such as contact info, "
    "dates, or fine print.\n\n"
    "3. **Tables:** If an element is a table row, put '|' between its fields.\n\n"
    "3.1. **Field values:** If an element has the format <Field-label> <Field-value>, "
    "put '|' between the label and the value.\n\n"
    "4. For font-family, provide your best best guess for a **specific font name** do "
    "NOT just mention serif or sans-serif.\n\n"
    "5. **Centered:** Append ' | centered' inside the classification only when an "
    "element is horizontally centered. Elements that share a common left margin (a "
    "list, a column) are left-aligned, never centered.\n\n"
    "6. **Output Format:** Present the final structure using **Markdown formatting**. "
    "The output must begin with a single line in the format **[Image Classification: "
    "<just a few words>]**, followed by a continuous stream of labeled text that "
    "follows the visual flow of the document. Output only the text."
)

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


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


# The model emits the label in Markdown (we ask for it), so tolerate bold/emphasis and a
# colon around the bracket: "**[Level 1 / Title]** text", "[Level 3 / Body]: text", etc.
_LABELED_LINE = re.compile(r"^\**\s*\[(?P<label>[^\[\]]+)\]\**\s*:?\s*\**\s*(?P<rest>.*)$")

# Substring -> level, checked in order. "level N" first: the model sometimes drops the
# word after the slash; the word checks catch the "[Metadata/Footer]" style labels.
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
            label = match.group("label").strip()
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
                block_alignment = "center" if re.search(r"\bcent(?:e?red)\b", label.lower()) else None
                block_family, block_weight = _parse_label_fonts(label)
                if not line:  # standalone "[Level 3 / Body]" -> labels the lines below
                    continue
        if level is None:
            level = block_level  # continuation line inherits the block's level
        if not category and not units and line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()
            continue
        cleaned = line.lstrip("-*#").strip().strip("*").strip()
        if not cleaned or _is_separator(cleaned):
            continue
        units.append(cleaned)
        levels.append(level)
        block_ids.append(block)
        alignments.append(block_alignment)
        font_families.append(block_family)
        font_weights.append(block_weight)
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
    )


def _level_of(label: str) -> str | None:
    lowered = label.lower()
    for key, level in _LEVEL_BY_LABEL:
        if key in lowered:
            return level
    return None


def _parse_label_fonts(label: str) -> tuple[str | None, int | None]:
    """Pull the font-family and font-weight out of a typography label.

    The label reads "Level n / Role | <font-family> | <font-size>pt | <font-weight> |
    centered"; only some fields may be present and their order can wobble. Weight is the
    100-900 integer; family is the first remaining field that is not a size ("18pt"), a
    weight, or the centred marker. Size is deliberately discarded. Returns (None, None)
    when the label carries no font fields (e.g. a bare "[Metadata/Footer]")."""
    parts = [part.strip() for part in label.split("|")]
    family: str | None = None
    weight: int | None = None
    for part in parts[1:]:  # parts[0] is the "Level n / Role" field
        lowered = part.lower()
        if not part or re.fullmatch(r"cent(?:e?red)", lowered):
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
    encoded = base64.b64encode(input_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
