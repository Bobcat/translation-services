"""Stage #5 grouping hint: ask a VLM which text belongs together (+ a category).

The VLM (small Gemma) reads the image and returns, in reading order, an image
**category** plus the groups of text as plain text — the format it is actually good
at (it groups a menu correctly this way, even reading prices our OCR missed). It does
NOT do cell-id bookkeeping; that brittle burden is gone.

The returned units are only a *hint*: ``app.grouping.align`` maps them back onto the
authoritative OCR cells (text + bbox) and builds the units. A weak or incomplete hint
therefore lowers quality but does not fail the job. The category is near-free extra
context (e.g. feeds the translation prompt: "restaurant menu" stops gemma rendering
"hoofdgerechten" as "fundamental rights").

We deliberately do NOT ask the VLM for font sizes: the same small model cannot do
good grouping *and* sizes in one call (asking for sizes pushes it to per-line units,
and the size field comes back as the price). Sizing is done from the OCR polygon.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.core.config import AppSettings


_RESPONSES_PATH = "/v1/responses"
_MAX_OUTPUT_TOKENS = 4096

# Sent as the user turn; the system role stays blank (llm-pool uses " "). Kept generic
# on purpose — no wording tied to a specific image type. The strong 26B VLM groups
# reliably, so we let its STRUCTURE lead and read it back faithfully:
#   - "Preserve newlines" keeps each original line, so a heading and its explanation
#     stay on separate lines; parse_grouping_output then splits a block into one unit per
#     SENTENCE (heading and explanation become separate units, each re-placed at its own
#     position/size) instead of one mashed paragraph.
#   - the table-row "|" marks field boundaries (e.g. a receipt's qty/desc/price columns),
#     which line up with the separate OCR boxes; parse splits a "|" line into one unit
#     per field so each maps to its own box.
#   - the blank line before "# OUTPUT FORMAT" keeps grouping stable on dense layouts.
# parse_grouping_output reads the category line, splits blocks on "###", then into units.
_SYSTEM_PROMPT = " "

_USER_INSTRUCTION = (
    "# TASKS\n"
    "1) Categorize the image in a few words.\n"
    "2) Group ALL text and numbers into semantically related units. Preserve newlines.\n"
    "3) If a line appears to be a table row put '|' between the fields.\n\n"
    "# OUTPUT FORMAT\n"
    "CATEGORY: <category>\n###\n<unit 1>\n###\n<unit 2>\n###\n..."
)

# A hint line ending in one of these closes a unit; a line ending otherwise (e.g. a
# wrapped line ending in a comma) joins the next — so a sentence wrapped across two lines
# stays one unit while a heading and its explanation, stacked on consecutive lines, split.
_SENTENCE_END = (".", "!", "?")

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


def request_grouping_hint(
    *,
    settings: AppSettings,
    input_path: Path,
    model: str,
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
    )
    return parse_grouping_output(output_text)


def parse_grouping_output(output_text: str) -> GroupingHint:
    """Split the VLM output into the image category and the hint units.

    The leading ``CATEGORY:`` line becomes the category. The rest is split into blocks on
    separator lines (``###`` / ``-----``); each block is then split into units by
    :func:`_segment_block` (one unit per sentence, one per ``|`` field) so a heading and
    its explanation — or a table row's columns — become separate units even when the model
    keeps them in one block. Leading bullet/markdown markers are dropped.
    """
    category = ""
    blocks: list[list[str]] = [[]]
    for raw in str(output_text or "").splitlines():
        line = raw.strip()
        if not category and line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()
            continue
        if _is_separator(line):
            if blocks[-1]:
                blocks.append([])
            continue
        cleaned = line.lstrip("-*•#").strip().strip("*").strip()
        if cleaned:
            blocks[-1].append(cleaned)
    units: list[str] = []
    for block in blocks:
        units.extend(_segment_block(block))
    return GroupingHint(category=category, units=units)


def _segment_block(lines: list[str]) -> list[str]:
    """Split a block into hint units: each ``|``-separated field is its own unit (table
    columns map 1:1 to the separate OCR boxes), and prose lines break at sentence ends
    (``_SENTENCE_END``) while a wrapped line ending mid-sentence joins the next. So a
    heading and its explanation split, a table row splits into its columns, but a single
    sentence wrapped across lines stays one unit. The block's trailing lines close a unit.
    """
    units: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            units.append(" ".join(current))
            current.clear()

    for line in lines:
        if "|" in line:
            flush()
            units.extend(field.strip() for field in line.split("|") if field.strip())
            continue
        current.append(line)
        if line.endswith(_SENTENCE_END):
            flush()
    flush()
    return units


def _is_separator(line: str) -> bool:
    compact = line.replace(" ", "")
    return len(compact) >= 3 and set(compact) <= {"#", "-", "=", "*", ":"}


def _call_llm_pool(*, base_url: str, payload: dict[str, Any], timeout: float) -> str:
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
