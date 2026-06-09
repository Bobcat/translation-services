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
# on purpose — no wording tied to a specific image type. Design:
#   - "Preserve newlines" keeps each ORIGINAL line as its own line, so
#     parse_grouping_output yields one hint unit per line. align then maps each OCR cell
#     onto its line ~1:1 (it does NOT merge a multi-line dish into one block), so every
#     line re-places onto its own cell at its own size — faithful to the original layout,
#     no re-wrapping or cramming.
#   - "table row -> '|' between fields" marks tabular layout (receipt columns, a menu
#     price) so a row's fields stay distinct.
#   - the blank line before "# OUTPUT FORMAT" keeps output stable on dense layouts.
# parse_grouping_output reads the category line, drops "###"/separators, and yields one
# unit per remaining line. Price/number cells are flagged non-translatable in align
# (_is_nontranslatable); the '|' itself is not yet parsed into field structure.
_SYSTEM_PROMPT = " "

_USER_INSTRUCTION = (
    "# TASKS\n"
    "1) Categorize the image in a few words.\n"
    "2) Group ALL text and numbers into semantically related units. Preserve newlines.\n"
    "3) If a line appears to be a table row put '|' between the fields.\n\n"
    "# OUTPUT FORMAT\n"
    "CATEGORY: <category>\n###\n<unit 1>\n###\n..."
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

    The leading ``CATEGORY:`` line becomes the category; every other non-empty line is
    one unit (the model puts each unit on its own line). Separator lines (``###`` /
    ``-----``) and leading bullet/markdown markers are dropped.
    """
    category = ""
    units: list[str] = []
    for raw in str(output_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if not category and line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()
            continue
        if _is_separator(line):
            continue
        cleaned = line.lstrip("-*•#").strip().strip("*").strip()
        if cleaned:
            units.append(cleaned)
    return GroupingHint(category=category, units=units)


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
