"""Stage #5 grouping hint: ask a VLM which text belongs together.

The VLM (qwen-vl) reads the image and returns the groups of text as plain text,
one group per line, in reading order — the format it is actually good at (a small
VLM groups a menu correctly this way, even reading prices our OCR missed). It does
NOT do cell-id bookkeeping; that brittle burden is gone.

The returned lines are only a *hint*: ``app.grouping.align`` maps them back onto
the authoritative OCR cells (text + bbox) and builds the units. A weak or
incomplete hint therefore lowers quality but does not fail the job.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

from app.core.config import AppSettings


_RESPONSES_PATH = "/v1/responses"
_MAX_OUTPUT_TOKENS = 4096

# A short instruction works best for the small VLM. It groups text into units; for
# tabular regions it separates the fields with "|". parse_hint_lines then drops the
# "----------" unit separators and splits "|" rows into per-field hint units.
# Sent as the user turn; the system role stays blank (llm-pool uses " ").
_SYSTEM_PROMPT = " "

_USER_INSTRUCTION = (
    "Group text and numbers into semantically related units. If a unit has a tabular "
    "structure, put a '|' between the fields. Output only the units separated by "
    "'\\n----------\\n'"
)

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class GroupingHintError(RuntimeError):
    """Raised when the grouping VLM call itself fails (transport / empty)."""


def request_grouping_hint(
    *,
    settings: AppSettings,
    input_path: Path,
    model: str,
) -> list[str]:
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
    return parse_hint_lines(output_text)


def parse_hint_lines(output_text: str) -> list[str]:
    """Split the VLM transcription into hint units.

    A plain line is one unit. A markdown table row (``| a | b | c |``) is split on
    ``|`` into one unit per non-empty column, so tabular fields stay separate; the
    dashes-only separator row is dropped.
    """
    units: list[str] = []
    for raw in str(output_text or "").splitlines():
        line = raw.strip().lstrip("-*•").strip().strip("*").strip()
        if not line:
            continue
        if "|" in line:
            cells = [cell.strip().strip("*").strip() for cell in line.split("|")]
            cells = [cell for cell in cells if cell]
            if not cells or _is_separator_row(cells):
                continue
            units.extend(cells)
        else:
            units.append(line)
    return units


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell) <= {"-", ":"} for cell in cells)


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
