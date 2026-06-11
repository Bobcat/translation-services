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
feed the renderer's size coordination.
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
# on purpose — no wording tied to a specific image type. Design (the user's original
# structural-analysis prompt + only the Tables rule; verified on all data/test-images/):
#   - ELEMENT level: one labeled line per semantic element (a whole dish with its
#     wrapped lines merged, a receipt row, a paragraph). The label boundary IS the item
#     boundary — each labeled line becomes its own block, and the renderer reflows the
#     translation over the element's physical lines (recovered by geometry).
#   - the hierarchy labels ([Level 1 / Title] ... [Metadata/Footer]) carry the visual
#     size/weight hierarchy OCR cannot give; parse_grouping_output strips them into a
#     per-line level.
#   - "table row -> '|' between its fields" marks tabular layout; at element level it
#     also splits a menu dish from its price column.
# Price/number cells are flagged non-translatable in align (_is_nontranslatable); the
# '|' itself is not yet parsed into field structure.
_SYSTEM_PROMPT = " "

_USER_INSTRUCTION = (
    "**Perform a structural analysis of the text in this image. Reconstruct the "
    "document's hierarchy by labeling each element in its natural reading order.**\n\n"
    "**Instructions:**\n"
    "1. **Analyze Visual Cues:** Use font size, font weight (boldness), spatial "
    "positioning, and grouping to determine the importance of each text element.\n"
    "2. **Linear Labeling (Do Not Group):** Do **not** group all elements of the same "
    "level together. Instead, process the text from top to bottom, following the "
    "natural reading order. For every piece of text, immediately precede it with its "
    "classification:\n"
    "    * **[Level 1 / Title]:** The most prominent text (largest/boldest).\n"
    "    * **[Level 2 / Header]:** Sub-headings that introduce new sections.\n"
    "    * **[Level 3 / Body]:** Descriptive or supporting text.\n"
    "    * **[Metadata/Footer]:** Small text at the edges, such as contact info, "
    "dates, or fine print.\n"
    "3. **Tables:** If an element is a table row, put '|' between its fields.\n"
    "4. **Output Format:** Present the final structure using **Markdown formatting**. "
    "The output must begin with a single line in the format **[Image Classification: "
    "<short description>]**, followed by a continuous stream of labeled text that "
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
    # ("title" | "header" | "body" | "footer", None when unlabeled) and the index of
    # the blank-line-separated block it belongs to.
    levels: list[str | None] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)


def request_grouping_hint(
    *,
    settings: AppSettings,
    input_path: Path,
    model: str,
    call_log: list[dict[str, Any]] | None = None,
) -> GroupingHint:
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


_LABELED_LINE = re.compile(r"^\[(?P<label>[^\[\]]+)\]\s*:?\s*(?P<rest>.*)$")

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
    block = 0
    block_open = False
    block_level: str | None = None

    def close_block() -> None:
        nonlocal block, block_open, block_level
        if block_open:
            block += 1
            block_open = False
        block_level = None

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
                if not line:  # standalone "[Level 3 / Body]" -> labels the lines below
                    continue
        if level is None:
            level = block_level  # continuation line inherits the block's level
        if not category and not units and line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()
            continue
        cleaned = line.lstrip("-*•#").strip().strip("*").strip()
        if not cleaned or _is_separator(cleaned):
            continue
        units.append(cleaned)
        levels.append(level)
        block_ids.append(block)
        block_open = True
    return GroupingHint(
        category=category,
        units=units,
        raw=str(output_text or ""),
        levels=levels,
        block_ids=block_ids,
    )


def _level_of(label: str) -> str | None:
    lowered = label.lower()
    for key, level in _LEVEL_BY_LABEL:
        if key in lowered:
            return level
    return None


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
